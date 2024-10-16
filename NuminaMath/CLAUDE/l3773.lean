import Mathlib

namespace NUMINAMATH_CALUDE_departure_sequences_count_l3773_377312

/-- The number of trains --/
def num_trains : ℕ := 6

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of trains per group --/
def trains_per_group : ℕ := 3

/-- The number of fixed trains (A and B) in the first group --/
def fixed_trains : ℕ := 2

/-- Theorem: The number of different departure sequences for the trains --/
theorem departure_sequences_count : 
  (num_trains - fixed_trains - trains_per_group) * 
  (Nat.factorial trains_per_group) * 
  (Nat.factorial trains_per_group) = 144 := by
  sorry

end NUMINAMATH_CALUDE_departure_sequences_count_l3773_377312


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_is_eleven_fourths_l3773_377333

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal_of_repeating_decimal : ℚ := 11 / 4

/-- Theorem: The reciprocal of the common fraction form of 0.363636... is 11/4 -/
theorem reciprocal_of_repeating_decimal_is_eleven_fourths :
  (1 : ℚ) / repeating_decimal = reciprocal_of_repeating_decimal := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_is_eleven_fourths_l3773_377333


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3773_377382

/-- Given a geometric sequence {a_n}, if a_4 and a_8 are the roots of x^2 - 8x + 9 = 0, then a_6 = 3 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 4 + a 8 = 8) →                    -- sum of roots
  (a 4 * a 8 = 9) →                    -- product of roots
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3773_377382


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l3773_377323

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 flat discount) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (20% off the amount over $150) -/
def savingsC (price : ℝ) : ℝ := 0.20 * (price - 150)

/-- The theorem to be proved -/
theorem coupon_savings_difference : 
  ∃ (min_price max_price : ℝ),
    (∀ price, price > 150 → 
      (savingsA price > savingsB ∧ savingsA price > savingsC price) ↔ 
      (min_price ≤ price ∧ price ≤ max_price)) ∧
    max_price - min_price = 400 :=
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l3773_377323


namespace NUMINAMATH_CALUDE_investment_problem_l3773_377366

/-- The investment problem -/
theorem investment_problem (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 16000)
  (hc : c_investment = 20000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 86400) :
  ∃ a_investment : ℕ, 
    a_investment * total_profit = 
      (total_profit - b_investment * total_profit / (a_investment + b_investment + c_investment) - 
       c_investment * total_profit / (a_investment + b_investment + c_investment)) :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3773_377366


namespace NUMINAMATH_CALUDE_function_properties_l3773_377346

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Theorem to prove
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 40) = f x) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3773_377346


namespace NUMINAMATH_CALUDE_percentage_difference_l3773_377321

theorem percentage_difference (x : ℝ) : 
  (x / 100) * 170 - 0.35 * 300 = 31 → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3773_377321


namespace NUMINAMATH_CALUDE_sin_squared_alpha_plus_5pi_12_l3773_377380

theorem sin_squared_alpha_plus_5pi_12 (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - 2 * α) = 3 / 5) : 
  Real.sin (α + 5 * Real.pi / 12) ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_alpha_plus_5pi_12_l3773_377380


namespace NUMINAMATH_CALUDE_special_triangle_sides_l3773_377377

/-- A triangle with an inscribed circle that passes through trisection points of a median -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle passes through trisection points of a median -/
  trisects_median : Bool
  /-- The sides of the triangle -/
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ

/-- The theorem about the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) 
  (h_radius : t.r = 3 * Real.sqrt 2)
  (h_trisects : t.trisects_median = true) :
  t.side_a = 5 * Real.sqrt 7 ∧ 
  t.side_b = 13 * Real.sqrt 7 ∧ 
  t.side_c = 10 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l3773_377377


namespace NUMINAMATH_CALUDE_square_arrangement_exists_l3773_377316

/-- A square in the plane --/
structure Square where
  sideLength : ℕ
  position : ℝ × ℝ

/-- An arrangement of squares --/
def Arrangement (n : ℕ) := Fin n → Square

/-- Two squares touch if they share a vertex --/
def touches (s1 s2 : Square) : Prop := sorry

/-- An arrangement is valid if no two squares overlap --/
def validArrangement (arr : Arrangement n) : Prop := sorry

/-- An arrangement satisfies the touching condition if each square touches exactly two others --/
def satisfiesTouchingCondition (arr : Arrangement n) : Prop := sorry

/-- Main theorem: For n ≥ 5, there exists a valid arrangement where each square touches exactly two others --/
theorem square_arrangement_exists (n : ℕ) (h : n ≥ 5) :
  ∃ (arr : Arrangement n), validArrangement arr ∧ satisfiesTouchingCondition arr := by
  sorry

end NUMINAMATH_CALUDE_square_arrangement_exists_l3773_377316


namespace NUMINAMATH_CALUDE_fiona_reaches_food_l3773_377339

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the predator pads
def predator_pads : Set ℕ := {4, 7}

-- Define the food pad
def food_pad : ℕ := 12

-- Define Fiona's starting pad
def start_pad : ℕ := 0

-- Define the probability of hopping to the next pad
def hop_prob : ℚ := 1/2

-- Define the probability of jumping two pads
def jump_prob : ℚ := 1/2

-- Define a function to represent the probability of reaching a pad safely
def safe_prob : ℕ → ℚ := sorry

-- State the theorem
theorem fiona_reaches_food : safe_prob food_pad = 1/32 := by sorry

end NUMINAMATH_CALUDE_fiona_reaches_food_l3773_377339


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3773_377374

theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3773_377374


namespace NUMINAMATH_CALUDE_susie_pizza_sales_l3773_377365

/-- Represents the pizza sales scenario --/
structure PizzaSales where
  slice_price : ℕ
  whole_price : ℕ
  slices_sold : ℕ
  total_earnings : ℕ

/-- Calculates the number of whole pizzas sold --/
def whole_pizzas_sold (s : PizzaSales) : ℕ :=
  (s.total_earnings - s.slice_price * s.slices_sold) / s.whole_price

/-- Theorem stating that under the given conditions, 3 whole pizzas were sold --/
theorem susie_pizza_sales :
  let s : PizzaSales := {
    slice_price := 3,
    whole_price := 15,
    slices_sold := 24,
    total_earnings := 117
  }
  whole_pizzas_sold s = 3 := by sorry

end NUMINAMATH_CALUDE_susie_pizza_sales_l3773_377365


namespace NUMINAMATH_CALUDE_correct_num_arrangements_l3773_377355

/-- The number of arrangements of 3 girls and 6 boys in a row, 
    with boys at both ends and no two girls adjacent. -/
def num_arrangements : ℕ := 43200

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The number of boys -/
def num_boys : ℕ := 6

/-- Theorem stating that the number of arrangements satisfying the given conditions is 43200 -/
theorem correct_num_arrangements : 
  (num_girls = 3 ∧ num_boys = 6) → num_arrangements = 43200 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_arrangements_l3773_377355


namespace NUMINAMATH_CALUDE_relay_race_selection_methods_l3773_377353

/-- The number of students good at sprinting -/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race -/
def selected_students : ℕ := 4

/-- The number of possible positions for A and B (they must be consecutive with A before B) -/
def positions_for_AB : ℕ := 3

/-- The number of remaining students to be selected -/
def remaining_students : ℕ := total_students - 2

/-- The number of positions to be filled by the remaining students -/
def positions_to_fill : ℕ := selected_students - 2

theorem relay_race_selection_methods :
  (positions_for_AB * (remaining_students.factorial / (remaining_students - positions_to_fill).factorial)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_selection_methods_l3773_377353


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3773_377300

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  C = π / 3 →
  b = 8 →
  (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3 →
  -- Definitions
  a > 0 →
  b > 0 →
  c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Theorem statements
  c = 7 ∧ Real.cos (B - C) = 13/14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3773_377300


namespace NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l3773_377335

theorem four_digit_cubes_divisible_by_16 : 
  (∃! (list : List ℕ), 
    (∀ n ∈ list, 1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0) ∧ 
    list.length = 3) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l3773_377335


namespace NUMINAMATH_CALUDE_binomial_distribution_n_l3773_377306

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ :=
  X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ :=
  X.n * X.p * (1 - X.p)

theorem binomial_distribution_n (X : BinomialDistribution) 
  (h_exp : expectation X = 15)
  (h_var : variance X = 11.25) :
  X.n = 60 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_n_l3773_377306


namespace NUMINAMATH_CALUDE_lizard_eye_difference_l3773_377360

theorem lizard_eye_difference : ∀ (E W S : ℕ),
  E = 3 →
  W = 3 * E →
  S = 7 * W →
  S + W - E = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_eye_difference_l3773_377360


namespace NUMINAMATH_CALUDE_production_cost_reduction_l3773_377372

/-- Represents the equation for production cost reduction over two years -/
theorem production_cost_reduction (initial_cost target_cost : ℝ) (x : ℝ) :
  initial_cost = 200000 →
  target_cost = 150000 →
  initial_cost * (1 - x)^2 = target_cost :=
by
  sorry

end NUMINAMATH_CALUDE_production_cost_reduction_l3773_377372


namespace NUMINAMATH_CALUDE_beadshop_profit_l3773_377317

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end NUMINAMATH_CALUDE_beadshop_profit_l3773_377317


namespace NUMINAMATH_CALUDE_power_sum_equality_l3773_377326

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3773_377326


namespace NUMINAMATH_CALUDE_helmet_pricing_and_purchase_l3773_377343

/-- Represents the types of helmets -/
inductive HelmetType
  | A
  | B

/-- Represents the wholesale and retail prices of helmets -/
structure HelmetPrices where
  wholesale : HelmetType → ℕ
  retail : HelmetType → ℕ

/-- Represents the sales data for helmets -/
structure SalesData where
  revenue : HelmetType → ℕ
  volume_ratio : ℕ  -- B's volume relative to A's

/-- Represents the purchase plan for helmets -/
structure PurchasePlan where
  total_helmets : ℕ
  budget : ℕ

/-- Main theorem statement -/
theorem helmet_pricing_and_purchase
  (prices : HelmetPrices)
  (sales : SalesData)
  (plan : PurchasePlan)
  (h1 : prices.wholesale HelmetType.A = 30)
  (h2 : prices.wholesale HelmetType.B = 20)
  (h3 : prices.retail HelmetType.A = prices.retail HelmetType.B + 15)
  (h4 : sales.revenue HelmetType.A = 450)
  (h5 : sales.revenue HelmetType.B = 600)
  (h6 : sales.volume_ratio = 2)
  (h7 : plan.total_helmets = 100)
  (h8 : plan.budget = 2350) :
  (prices.retail HelmetType.A = 45 ∧
   prices.retail HelmetType.B = 30) ∧
  (∀ m : ℕ, m * prices.wholesale HelmetType.A +
   (plan.total_helmets - m) * prices.wholesale HelmetType.B ≤ plan.budget →
   m ≤ 35) :=
sorry

end NUMINAMATH_CALUDE_helmet_pricing_and_purchase_l3773_377343


namespace NUMINAMATH_CALUDE_distance_after_12_hours_l3773_377370

/-- The distance between two people walking in opposite directions -/
def distance_between (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

/-- Theorem: Two people walking in opposite directions for 12 hours
    at speeds of 7 km/hr and 3 km/hr will be 120 km apart -/
theorem distance_after_12_hours :
  distance_between 7 3 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_12_hours_l3773_377370


namespace NUMINAMATH_CALUDE_number_problem_l3773_377332

theorem number_problem (x : ℝ) : 0.60 * x - 40 = 50 → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3773_377332


namespace NUMINAMATH_CALUDE_smallest_block_with_143_hidden_cubes_l3773_377398

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of hidden cubes in a block. -/
def hiddenCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Calculates the total number of cubes in a block. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem stating the smallest number of cubes in a block with 143 hidden cubes. -/
theorem smallest_block_with_143_hidden_cubes :
  ∃ (d : BlockDimensions),
    hiddenCubes d = 143 ∧
    totalCubes d = 336 ∧
    (∀ (d' : BlockDimensions), hiddenCubes d' = 143 → totalCubes d' ≥ 336) := by
  sorry

#check smallest_block_with_143_hidden_cubes

end NUMINAMATH_CALUDE_smallest_block_with_143_hidden_cubes_l3773_377398


namespace NUMINAMATH_CALUDE_arrangement_remainder_l3773_377319

/-- The number of green marbles -/
def green_marbles : ℕ := 5

/-- The maximum number of red marbles that satisfies the condition -/
def max_red_marbles : ℕ := 16

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def num_arrangements : ℕ := Nat.choose (green_marbles + max_red_marbles) green_marbles

/-- The theorem to be proved -/
theorem arrangement_remainder : num_arrangements % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l3773_377319


namespace NUMINAMATH_CALUDE_distribution_problem_l3773_377328

/-- The number of ways to distribute n indistinguishable objects among k distinct groups,
    with each group receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribution_problem :
  distribute 12 6 = 462 := by sorry

end NUMINAMATH_CALUDE_distribution_problem_l3773_377328


namespace NUMINAMATH_CALUDE_sequence_inequality_l3773_377310

theorem sequence_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∀ n : ℕ, (a^n / (n : ℝ)^b) < (a^(n+1) / ((n+1) : ℝ)^b) :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3773_377310


namespace NUMINAMATH_CALUDE_simplify_expression_l3773_377302

theorem simplify_expression : 
  (Real.sqrt 450 / Real.sqrt 250) + (Real.sqrt 294 / Real.sqrt 147) = (3 * Real.sqrt 10 + 5 * Real.sqrt 2) / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3773_377302


namespace NUMINAMATH_CALUDE_fraction_inequality_l3773_377358

theorem fraction_inequality (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3773_377358


namespace NUMINAMATH_CALUDE_append_two_to_three_digit_number_l3773_377351

/-- Given a three-digit number with digits h, t, and u, appending 2 results in 1000h + 100t + 10u + 2 -/
theorem append_two_to_three_digit_number (h t u : ℕ) :
  let original := 100 * h + 10 * t + u
  let appended := original * 10 + 2
  appended = 1000 * h + 100 * t + 10 * u + 2 := by
sorry

end NUMINAMATH_CALUDE_append_two_to_three_digit_number_l3773_377351


namespace NUMINAMATH_CALUDE_outflow_symmetry_outflow_ratio_replacement_time_ratio_l3773_377388

/-- Represents the structure of a sewage purification tower -/
structure SewageTower where
  layers : Nat
  outlets : Nat
  flow_distribution : List (List Rat)

/-- Calculates the outflow for a given outlet -/
def outflow (tower : SewageTower) (outlet : Nat) : Rat :=
  sorry

/-- Theorem stating that outflows of outlet 2 and 4 are equal -/
theorem outflow_symmetry (tower : SewageTower) :
  tower.outlets = 5 → outflow tower 2 = outflow tower 4 :=
  sorry

/-- Theorem stating the ratio of outflows for outlets 1, 2, and 3 -/
theorem outflow_ratio (tower : SewageTower) :
  tower.outlets = 5 →
  ∃ (k : Rat), outflow tower 1 = k ∧ outflow tower 2 = 4*k ∧ outflow tower 3 = 6*k :=
  sorry

/-- Calculates the wear rate for a given triangle in the tower -/
def wear_rate (tower : SewageTower) (triangle : Nat) : Rat :=
  sorry

/-- Theorem stating the replacement time ratio for slowest and fastest wearing triangles -/
theorem replacement_time_ratio (tower : SewageTower) :
  ∃ (slow fast : Nat),
    wear_rate tower slow = (1/8 : Rat) * wear_rate tower fast ∧
    ∀ t, wear_rate tower t ≥ wear_rate tower slow ∧
         wear_rate tower t ≤ wear_rate tower fast :=
  sorry

end NUMINAMATH_CALUDE_outflow_symmetry_outflow_ratio_replacement_time_ratio_l3773_377388


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3773_377356

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 3 * r) = 300)
  (h2 : P * (1 + 8 * r) = 400) :
  r = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3773_377356


namespace NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l3773_377371

/-- Represents a student's test score -/
def TestScore := ℕ

/-- Represents a group of students who took the test -/
def StudentGroup := List TestScore

/-- Represents the different score ranges -/
inductive ScoreRange
  | AboveOrEqual120
  | Between90And120
  | Between75And90
  | Between60And75
  | Below60

/-- Function to calculate the proportion of students in each score range -/
def calculateProportions (students : StudentGroup) : ScoreRange → ℚ :=
  sorry

/-- Function to perform frequency distribution -/
def frequencyDistribution (students : StudentGroup) : ScoreRange → ℕ :=
  sorry

/-- Theorem stating that frequency distribution is necessary and sufficient
    to determine the proportions of students in different score ranges -/
theorem frequency_distribution_necessary_sufficient
  (students : StudentGroup)
  (h : students.length = 800) :
  (∀ range, calculateProportions students range =
    (frequencyDistribution students range : ℚ) / 800) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_necessary_sufficient_l3773_377371


namespace NUMINAMATH_CALUDE_circle_line_distance_l3773_377392

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 4)^2 = 4}
  let line : Set (ℝ × ℝ) := {p | a * p.1 + p.2 - 1 = 0}
  let center : ℝ × ℝ := (1, 4)
  (∀ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt ≥ 1) ∧
  (∃ p ∈ line, ((p.1 - center.1)^2 + (p.2 - center.2)^2).sqrt = 1) →
  a = -4/3 := by
sorry


end NUMINAMATH_CALUDE_circle_line_distance_l3773_377392


namespace NUMINAMATH_CALUDE_student_lecture_assignment_l3773_377395

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 5 students to 3 lectures is 243 -/
theorem student_lecture_assignment :
  assignment_count 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_student_lecture_assignment_l3773_377395


namespace NUMINAMATH_CALUDE_intersection_equals_B_B_proper_superset_A_l3773_377397

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

-- Theorem for part 1
theorem intersection_equals_B (m : ℝ) : (A ∩ B m) = B m ↔ m ≤ 1 := by sorry

-- Theorem for part 2
theorem B_proper_superset_A (m : ℝ) : A ⊂ B m ↔ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_B_proper_superset_A_l3773_377397


namespace NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l3773_377309

/-- The set S of points (x, y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 ≤ 5) ∨
               (5 = y - 6 ∧ x + 3 ≤ 5) ∨
               (x + 3 = y - 6 ∧ 5 ≤ x + 3)}

/-- The three rays that make up set S -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 11}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 11 ∧ p.1 ≤ 2}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 ≥ 2}

/-- The common point of the three rays -/
def commonPoint : ℝ × ℝ := (2, 11)

theorem S_is_three_rays_with_common_point :
  S = ray1 ∪ ray2 ∪ ray3 ∧
  ray1 ∩ ray2 ∩ ray3 = {commonPoint} :=
by sorry

end NUMINAMATH_CALUDE_S_is_three_rays_with_common_point_l3773_377309


namespace NUMINAMATH_CALUDE_two_year_growth_at_fifty_percent_l3773_377369

/-- The growth factor for a principal amount over a given number of years,
    given an annual interest rate and annual compounding. -/
def growthFactor (rate : ℝ) (years : ℕ) : ℝ :=
  (1 + rate) ^ years

/-- Theorem stating that with a 50% annual interest rate and 2 years of growth,
    the principal will grow by a factor of 2.25 -/
theorem two_year_growth_at_fifty_percent :
  growthFactor 0.5 2 = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_two_year_growth_at_fifty_percent_l3773_377369


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3773_377303

theorem sixth_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l3773_377303


namespace NUMINAMATH_CALUDE_fence_posts_count_l3773_377354

/-- Represents the fence setup for a rectangular field -/
structure FenceSetup where
  wallLength : ℕ
  rectLength : ℕ
  rectWidth : ℕ
  postSpacing : ℕ
  gateWidth : ℕ

/-- Calculates the number of posts required for the fence setup -/
def calculatePosts (setup : FenceSetup) : ℕ :=
  sorry

/-- Theorem stating that the specific fence setup requires 19 posts -/
theorem fence_posts_count :
  let setup : FenceSetup := {
    wallLength := 120,
    rectLength := 80,
    rectWidth := 50,
    postSpacing := 10,
    gateWidth := 20
  }
  calculatePosts setup = 19 := by
  sorry

end NUMINAMATH_CALUDE_fence_posts_count_l3773_377354


namespace NUMINAMATH_CALUDE_chess_game_probability_l3773_377352

theorem chess_game_probability (draw_prob win_prob lose_prob : ℚ) : 
  draw_prob = 1/2 →
  win_prob = 1/3 →
  draw_prob + win_prob + lose_prob = 1 →
  lose_prob = 1/6 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3773_377352


namespace NUMINAMATH_CALUDE_popsicles_left_l3773_377378

def initial_grape : ℕ := 2
def initial_cherry : ℕ := 13
def initial_banana : ℕ := 2
def initial_mango : ℕ := 8
def initial_strawberry : ℕ := 4
def initial_orange : ℕ := 6

def cherry_eaten : ℕ := 3
def grape_eaten : ℕ := 1

def total_initial : ℕ := initial_grape + initial_cherry + initial_banana + initial_mango + initial_strawberry + initial_orange

def total_eaten : ℕ := cherry_eaten + grape_eaten

theorem popsicles_left : total_initial - total_eaten = 31 := by
  sorry

end NUMINAMATH_CALUDE_popsicles_left_l3773_377378


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3773_377331

theorem triangle_equilateral (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (condition : a^2 + b^2 + 2*c^2 - 2*a*c - 2*b*c = 0) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3773_377331


namespace NUMINAMATH_CALUDE_rain_probability_l3773_377324

theorem rain_probability (p : ℝ) (n : ℕ) (hp : p = 3 / 4) (hn : n = 4) :
  1 - (1 - p)^n = 255 / 256 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l3773_377324


namespace NUMINAMATH_CALUDE_fraction_problem_l3773_377375

theorem fraction_problem (f : ℚ) : f * 50 - 4 = 6 → f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3773_377375


namespace NUMINAMATH_CALUDE_equality_from_divisibility_l3773_377322

theorem equality_from_divisibility (a b : ℕ+) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_from_divisibility_l3773_377322


namespace NUMINAMATH_CALUDE_hyperbola_m_equation_l3773_377308

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- The equation of a hyperbola in the form y²/a - x²/b = c -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a - x^2 / h.b = h.c

/-- Two hyperbolas have common asymptotes if they have the same a/b ratio -/
def common_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_m_equation 
  (n : Hyperbola)
  (hn_eq : hyperbola_equation n = fun x y ↦ y^2 / 4 - x^2 / 2 = 1)
  (m : Hyperbola)
  (hm_asymp : common_asymptotes m n)
  (hm_point : hyperbola_equation m (-2) 4) :
  hyperbola_equation m = fun x y ↦ y^2 / 8 - x^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_equation_l3773_377308


namespace NUMINAMATH_CALUDE_square_divisible_by_six_between_30_and_150_l3773_377348

theorem square_divisible_by_six_between_30_and_150 (x : ℕ) :
  (∃ n : ℕ, x = n^2) →  -- x is a square number
  x % 6 = 0 →           -- x is divisible by 6
  30 < x →              -- x is greater than 30
  x < 150 →             -- x is less than 150
  x = 36 ∨ x = 144 :=   -- x is either 36 or 144
by sorry

end NUMINAMATH_CALUDE_square_divisible_by_six_between_30_and_150_l3773_377348


namespace NUMINAMATH_CALUDE_smallest_domain_size_l3773_377344

-- Define the function f
def f : ℕ → ℕ
| 7 => 22
| n => if n % 2 = 1 then 3 * n + 1 else n / 2

-- Define the sequence of f applications starting from 7
def fSequence : ℕ → ℕ
| 0 => 7
| n + 1 => f (fSequence n)

-- Define the set of unique elements in the sequence
def uniqueElements (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).image fSequence

-- Theorem statement
theorem smallest_domain_size :
  ∃ n : ℕ, (uniqueElements n).card = 13 ∧
  ∀ m : ℕ, m < n → (uniqueElements m).card < 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_domain_size_l3773_377344


namespace NUMINAMATH_CALUDE_circle_symmetry_l3773_377349

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem circle_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ circle2 y x ∧ line_of_symmetry x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3773_377349


namespace NUMINAMATH_CALUDE_complex_power_sum_l3773_377314

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (Real.pi/4)) :
  z^8 + 1/z^8 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3773_377314


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l3773_377364

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l3773_377364


namespace NUMINAMATH_CALUDE_degree_of_specific_polynomial_l3773_377318

/-- The degree of a polynomial of the form (aᵏ * bⁿ) where a and b are polynomials -/
def degree_product_power (deg_a deg_b k n : ℕ) : ℕ := k * deg_a + n * deg_b

/-- The degree of the polynomial (x³ + x + 1)⁵ * (x⁴ + x² + 1)² -/
def degree_specific_polynomial : ℕ :=
  degree_product_power 3 4 5 2

theorem degree_of_specific_polynomial :
  degree_specific_polynomial = 23 := by sorry

end NUMINAMATH_CALUDE_degree_of_specific_polynomial_l3773_377318


namespace NUMINAMATH_CALUDE_min_sum_of_prime_factors_l3773_377329

theorem min_sum_of_prime_factors (x : ℕ) : 
  let sequence_sum := 25 * (x + 12)
  ∃ (p₁ p₂ p₃ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ sequence_sum = p₁ * p₂ * p₃ →
  ∀ (q₁ q₂ q₃ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ sequence_sum = q₁ * q₂ * q₃ →
  q₁ + q₂ + q₃ ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_prime_factors_l3773_377329


namespace NUMINAMATH_CALUDE_intersection_chord_length_l3773_377357

/-- The line L: 3x - y - 6 = 0 -/
def line_L (x y : ℝ) : Prop := 3 * x - y - 6 = 0

/-- The circle C: x^2 + y^2 - 2x - 4y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- The length of the chord AB formed by the intersection of line L and circle C -/
noncomputable def chord_length : ℝ := Real.sqrt 10

theorem intersection_chord_length :
  chord_length = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l3773_377357


namespace NUMINAMATH_CALUDE_modulo_nine_equivalence_l3773_377336

theorem modulo_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2022 ≡ n [ZMOD 9] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_equivalence_l3773_377336


namespace NUMINAMATH_CALUDE_stones_sent_away_l3773_377307

theorem stones_sent_away (original_stones kept_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : kept_stones = 15) : 
  original_stones - kept_stones = 63 := by
  sorry

end NUMINAMATH_CALUDE_stones_sent_away_l3773_377307


namespace NUMINAMATH_CALUDE_michelle_taxi_ride_cost_l3773_377304

/-- Calculate the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxiRideCost (initialFee : ℝ) (distance : ℝ) (chargePerMile : ℝ) : ℝ :=
  initialFee + distance * chargePerMile

/-- Theorem stating that Michelle's taxi ride cost $12 -/
theorem michelle_taxi_ride_cost :
  taxiRideCost 2 4 2.5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_michelle_taxi_ride_cost_l3773_377304


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l3773_377341

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3*x^3 - 2*x^2 - 2500*x + 434

theorem polynomial_value_at_two :
  f 2 = -3390 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l3773_377341


namespace NUMINAMATH_CALUDE_total_pizza_weight_l3773_377381

/-- Represents the weight of a pizza with toppings -/
structure Pizza where
  base : Nat
  toppings : List Nat

/-- Calculates the total weight of a pizza -/
def totalWeight (p : Pizza) : Nat :=
  p.base + p.toppings.sum

/-- Rachel's pizza -/
def rachelPizza : Pizza :=
  { base := 400
  , toppings := [100, 50, 60] }

/-- Bella's pizza -/
def bellaPizza : Pizza :=
  { base := 350
  , toppings := [75, 55, 35] }

/-- Theorem: The total weight of Rachel's and Bella's pizzas is 1125 grams -/
theorem total_pizza_weight :
  totalWeight rachelPizza + totalWeight bellaPizza = 1125 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_weight_l3773_377381


namespace NUMINAMATH_CALUDE_andy_wrong_answers_l3773_377305

/-- Represents the number of wrong answers for each person -/
structure WrongAnswers where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The test conditions -/
def testConditions (w : WrongAnswers) : Prop :=
  w.andy + w.beth = w.charlie + w.daniel ∧
  w.andy + w.daniel = w.beth + w.charlie + 6 ∧
  w.charlie = 7

theorem andy_wrong_answers (w : WrongAnswers) :
  testConditions w → w.andy = 20 := by
  sorry

#check andy_wrong_answers

end NUMINAMATH_CALUDE_andy_wrong_answers_l3773_377305


namespace NUMINAMATH_CALUDE_order_of_magnitude_l3773_377340

theorem order_of_magnitude (x : ℝ) (hx : 0.95 < x ∧ x < 1.05) :
  x < x^(x^(x^x)) ∧ x^(x^(x^x)) < x^x := by sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l3773_377340


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l3773_377327

theorem simultaneous_inequalities (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l3773_377327


namespace NUMINAMATH_CALUDE_expression_equals_36_l3773_377315

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l3773_377315


namespace NUMINAMATH_CALUDE_percentage_equality_l3773_377390

theorem percentage_equality : (10 : ℚ) / 100 * 200 = (20 : ℚ) / 100 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3773_377390


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3773_377320

theorem quadratic_equation_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  (α ^ 2 - 5 * α + 6 = 0) ∧ (β ^ 2 - 5 * β + 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3773_377320


namespace NUMINAMATH_CALUDE_third_bakery_needs_twelve_sacks_l3773_377391

/-- The number of weeks Antoine supplies strawberries -/
def weeks : ℕ := 4

/-- The total number of sacks Antoine supplies in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := total_sacks / weeks - (first_bakery_sacks + second_bakery_sacks)

theorem third_bakery_needs_twelve_sacks : third_bakery_sacks = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_bakery_needs_twelve_sacks_l3773_377391


namespace NUMINAMATH_CALUDE_no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l3773_377334

-- Define a type for figures on a plane
structure PlaneFigure where
  -- Add necessary fields here
  isBounded : Bool
  hasParallelAxes : Bool

-- Define a predicate for having two parallel, non-coincident symmetry axes
def hasParallelSymmetryAxes (f : PlaneFigure) : Prop :=
  f.hasParallelAxes

theorem no_bounded_figure_with_parallel_axes :
  ¬ ∃ (f : PlaneFigure), f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

theorem exists_unbounded_figure_with_parallel_axes :
  ∃ (f : PlaneFigure), ¬f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

end NUMINAMATH_CALUDE_no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l3773_377334


namespace NUMINAMATH_CALUDE_apple_bags_sum_l3773_377373

theorem apple_bags_sum : 
  let golden_delicious : ℚ := 17/100
  let macintosh : ℚ := 17/100
  let cortland : ℚ := 33/100
  golden_delicious + macintosh + cortland = 67/100 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_sum_l3773_377373


namespace NUMINAMATH_CALUDE_banana_pear_ratio_l3773_377359

theorem banana_pear_ratio : 
  ∀ (dishes bananas pears : ℕ),
  dishes = 160 →
  pears = 50 →
  dishes = bananas + 10 →
  ∃ k : ℕ, bananas = k * pears →
  bananas / pears = 3 := by
sorry

end NUMINAMATH_CALUDE_banana_pear_ratio_l3773_377359


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3773_377396

/-- Given an ellipse with equation x²/4 + y²/2 = 1, prove that the perimeter of the triangle
    formed by any point on the ellipse and its foci is 4 + 2√2. -/
theorem ellipse_triangle_perimeter (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  x^2 / 4 + y^2 / 2 = 1 →
  P = (x, y) →
  F₁.1^2 / 4 + F₁.2^2 / 2 = 1 →
  F₂.1^2 / 4 + F₂.2^2 / 2 = 1 →
  ∃ c : ℝ, c^2 = 2 ∧ 
    dist P F₁ + dist P F₂ + dist F₁ F₂ = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3773_377396


namespace NUMINAMATH_CALUDE_function_property_l3773_377347

/-- Given a function f(x) = ax^4 + bx^2 - x + 1 where a and b are real numbers,
    if f(2) = 9, then f(-2) = 13 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^4 + b * x^2 - x + 1
  f 2 = 9 → f (-2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3773_377347


namespace NUMINAMATH_CALUDE_intersection_slope_l3773_377387

/-- Given two lines p and q that intersect at (4, 11), 
    where p has equation y = 2x + 3 and q has equation y = mx + 1,
    prove that m = 2.5 -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 3 → y = m*x + 1 → x = 4 ∧ y = 11) →
  m = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l3773_377387


namespace NUMINAMATH_CALUDE_josh_marbles_l3773_377345

/-- The number of marbles Josh has after losing some, given the initial conditions. -/
theorem josh_marbles (colors : Nat) (initial_per_color : Nat) (lost_per_color : Nat) :
  colors = 5 →
  initial_per_color = 16 →
  lost_per_color = 7 →
  colors * (initial_per_color - lost_per_color) = 45 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3773_377345


namespace NUMINAMATH_CALUDE_march_birth_percentage_l3773_377383

def total_people : ℕ := 100
def march_births : ℕ := 8

theorem march_birth_percentage :
  (march_births : ℚ) / (total_people : ℚ) * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_march_birth_percentage_l3773_377383


namespace NUMINAMATH_CALUDE_total_distance_walked_l3773_377361

def distance_school_to_david : ℝ := 0.2
def distance_david_to_home : ℝ := 0.7

theorem total_distance_walked : 
  distance_school_to_david + distance_david_to_home = 0.9 := by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3773_377361


namespace NUMINAMATH_CALUDE_binomial_cube_expansion_l3773_377367

theorem binomial_cube_expansion :
  101^3 + 3*(101^2)*2 + 3*101*(2^2) + 2^3 = 103^3 := by sorry

end NUMINAMATH_CALUDE_binomial_cube_expansion_l3773_377367


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3773_377330

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, 0 < x ∧ x < π/2 → x > Real.sin x) ↔
  ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ x ≤ Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3773_377330


namespace NUMINAMATH_CALUDE_basketball_team_girls_l3773_377325

theorem basketball_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  boys + girls = total →
  boys + (girls / 3) = attended →
  boys = total - girls →
  girls = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_girls_l3773_377325


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3773_377379

theorem right_triangle_acute_angles (α β : Real) : 
  α = 30 → β = 90 → ∃ γ : Real, γ = 60 ∧ α + β + γ = 180 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3773_377379


namespace NUMINAMATH_CALUDE_distribute_five_two_correct_l3773_377362

def number_of_correct_locations : ℕ := 2
def total_objects : ℕ := 5

/-- The number of ways to distribute n distinct objects to n distinct locations
    such that exactly k objects are in their correct locations -/
def distribute (n k : ℕ) : ℕ := sorry

theorem distribute_five_two_correct :
  distribute total_objects number_of_correct_locations = 20 := by sorry

end NUMINAMATH_CALUDE_distribute_five_two_correct_l3773_377362


namespace NUMINAMATH_CALUDE_parabola_equation_l3773_377393

/-- A parabola with focus on the line 3x - 4y - 12 = 0 has standard equation x^2 = 6y and directrix y = 3 -/
theorem parabola_equation (x y : ℝ) :
  (∃ (a b : ℝ), 3*a - 4*b - 12 = 0 ∧ (x - a)^2 + (y - b)^2 = (y - 3)^2) →
  x^2 = 6*y ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3773_377393


namespace NUMINAMATH_CALUDE_total_limes_and_plums_l3773_377394

theorem total_limes_and_plums (L M P : ℕ) (hL : L = 25) (hM : M = 32) (hP : P = 12) :
  L + M + P = 69 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_and_plums_l3773_377394


namespace NUMINAMATH_CALUDE_binomial_square_special_case_l3773_377363

theorem binomial_square_special_case (a b : ℝ) : (2*a - 3*b)^2 = 4*a^2 - 12*a*b + 9*b^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_special_case_l3773_377363


namespace NUMINAMATH_CALUDE_spade_calculation_l3773_377337

/-- The ⋆ operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : 
  let z : ℝ := 2
  spade 2 (spade 3 (1 + z)) = 4 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l3773_377337


namespace NUMINAMATH_CALUDE_nap_time_is_three_hours_l3773_377311

-- Define flight duration in minutes
def flight_duration : ℕ := 11 * 60 + 20

-- Define durations of activities in minutes
def reading_time : ℕ := 2 * 60
def movie_time : ℕ := 4 * 60
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 1 * 60 + 10

-- Define total activity time
def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Define nap time in hours
def nap_time_hours : ℕ := (flight_duration - total_activity_time) / 60

-- Theorem statement
theorem nap_time_is_three_hours : nap_time_hours = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_is_three_hours_l3773_377311


namespace NUMINAMATH_CALUDE_unknown_number_value_l3773_377338

theorem unknown_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l3773_377338


namespace NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l3773_377368

theorem school_students_count : ℕ → Prop :=
  fun total_students =>
    let below_8 := (total_students : ℚ) * (1 / 5)
    let age_8 := 48
    let age_9_11 := 2 * age_8
    let above_11 := (5 / 6) * age_9_11
    total_students = below_8 + age_8 + age_9_11 + above_11 →
    total_students = 280

-- The proof goes here
theorem prove_school_students_count : ∃ n : ℕ, school_students_count n :=
sorry

end NUMINAMATH_CALUDE_school_students_count_prove_school_students_count_l3773_377368


namespace NUMINAMATH_CALUDE_stating_three_pairs_l3773_377301

/-- 
A function that returns the number of ordered pairs (m,n) of positive integers 
satisfying m ≥ n and m^2 - n^2 = 72
-/
def count_pairs : ℕ := 
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = 72) 
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- 
Theorem stating that there are exactly 3 ordered pairs (m,n) of positive integers 
satisfying m ≥ n and m^2 - n^2 = 72
-/
theorem three_pairs : count_pairs = 3 := by
  sorry

end NUMINAMATH_CALUDE_stating_three_pairs_l3773_377301


namespace NUMINAMATH_CALUDE_best_fit_model_l3773_377350

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Determines if a model has better fit than another based on R² values -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.r_squared > model2.r_squared

theorem best_fit_model (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.05)
  (h2 : model2.r_squared = 0.49)
  (h3 : model3.r_squared = 0.89)
  (h4 : model4.r_squared = 0.98) :
  better_fit model4 model1 ∧ better_fit model4 model2 ∧ better_fit model4 model3 :=
sorry

end NUMINAMATH_CALUDE_best_fit_model_l3773_377350


namespace NUMINAMATH_CALUDE_tower_heights_theorem_l3773_377376

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the possible height contributions of a brick -/
def HeightContributions : List ℕ := [4, 10, 19]

/-- The total number of bricks -/
def TotalBricks : ℕ := 94

/-- Calculates the number of distinct tower heights -/
def distinctTowerHeights (brickDims : BrickDimensions) (contributions : List ℕ) (totalBricks : ℕ) : ℕ :=
  sorry

theorem tower_heights_theorem (brickDims : BrickDimensions) 
    (h1 : brickDims.length = 4 ∧ brickDims.width = 10 ∧ brickDims.height = 19) :
    distinctTowerHeights brickDims HeightContributions TotalBricks = 465 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_theorem_l3773_377376


namespace NUMINAMATH_CALUDE_vector_sum_problem_l3773_377386

theorem vector_sum_problem :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![-2, 4]
  (v1 + 3 • v2) = ![-1, 9] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l3773_377386


namespace NUMINAMATH_CALUDE_minimum_cost_is_74_l3773_377389

-- Define the box types
inductive BoxType
| A
| B

-- Define the problem parameters
def totalVolume : ℕ := 15
def boxCapacity : BoxType → ℕ
  | BoxType.A => 2
  | BoxType.B => 3
def boxPrice : BoxType → ℕ
  | BoxType.A => 13
  | BoxType.B => 15
def discountThreshold : ℕ := 3
def discountAmount : ℕ := 10

-- Define a purchase plan
def PurchasePlan := BoxType → ℕ

-- Calculate the total volume of a purchase plan
def totalVolumeOfPlan (plan : PurchasePlan) : ℕ :=
  (plan BoxType.A) * (boxCapacity BoxType.A) + (plan BoxType.B) * (boxCapacity BoxType.B)

-- Calculate the cost of a purchase plan
def costOfPlan (plan : PurchasePlan) : ℕ :=
  let basePrice := (plan BoxType.A) * (boxPrice BoxType.A) + (plan BoxType.B) * (boxPrice BoxType.B)
  if plan BoxType.A ≥ discountThreshold then basePrice - discountAmount else basePrice

-- Define a valid purchase plan
def isValidPlan (plan : PurchasePlan) : Prop :=
  totalVolumeOfPlan plan = totalVolume

-- Theorem to prove
theorem minimum_cost_is_74 :
  ∃ (plan : PurchasePlan), isValidPlan plan ∧
    ∀ (otherPlan : PurchasePlan), isValidPlan otherPlan → costOfPlan plan ≤ costOfPlan otherPlan ∧
    costOfPlan plan = 74 :=
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_74_l3773_377389


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3773_377399

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3773_377399


namespace NUMINAMATH_CALUDE_triangle_side_equations_l3773_377342

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the altitude and median
def altitude_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0
def median_equation (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Define the equations of the sides
def side_AB_equation (x y : ℝ) : Prop := 2*x - y + 1 = 0
def side_BC_equation (x y : ℝ) : Prop := 2*x + 3*y - 7 = 0
def side_AC_equation (x y : ℝ) : Prop := y = 1

theorem triangle_side_equations 
  (tri : Triangle)
  (h1 : tri.A = (0, 1))
  (h2 : ∀ x y, altitude_equation x y → (x - tri.A.1) * (tri.B.2 - tri.A.2) = -(y - tri.A.2) * (tri.B.1 - tri.A.1))
  (h3 : ∀ x y, median_equation x y → 2 * x = tri.A.1 + tri.C.1 ∧ 2 * y = tri.A.2 + tri.C.2) :
  (∀ x y, side_AB_equation x y ↔ (y - tri.A.2) = ((tri.B.2 - tri.A.2) / (tri.B.1 - tri.A.1)) * (x - tri.A.1)) ∧
  (∀ x y, side_BC_equation x y ↔ (y - tri.B.2) = ((tri.C.2 - tri.B.2) / (tri.C.1 - tri.B.1)) * (x - tri.B.1)) ∧
  (∀ x y, side_AC_equation x y ↔ (y - tri.A.2) = ((tri.C.2 - tri.A.2) / (tri.C.1 - tri.A.1)) * (x - tri.A.1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l3773_377342


namespace NUMINAMATH_CALUDE_production_days_calculation_l3773_377385

/-- Proves that given the conditions of the production problem, n must equal 9 -/
theorem production_days_calculation (n : ℕ) 
  (h1 : (n : ℝ) * 50 / n = 50)  -- Average for n days is 50
  (h2 : ((n : ℝ) * 50 + 90) / (n + 1) = 54)  -- New average for n+1 days is 54
  : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3773_377385


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3773_377384

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 320 →
  train_speed_kmh = 45 →
  time_to_pass = 36.8 →
  ∃ (bridge_length : ℝ), bridge_length = 140 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * time_to_pass) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3773_377384


namespace NUMINAMATH_CALUDE_metal_bar_sets_l3773_377313

theorem metal_bar_sets (total_bars : ℕ) (bars_per_set : ℕ) (h1 : total_bars = 14) (h2 : bars_per_set = 7) :
  total_bars / bars_per_set = 2 := by
  sorry

end NUMINAMATH_CALUDE_metal_bar_sets_l3773_377313
