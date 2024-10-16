import Mathlib

namespace NUMINAMATH_CALUDE_calculator_squaring_l221_22161

theorem calculator_squaring (initial : ℕ) (target : ℕ) : 
  (initial = 3 ∧ target = 2000) → 
  (∃ n : ℕ, initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) → 
  (∃ n : ℕ, n = 3 ∧ initial^(2^n) > target ∧ ∀ m : ℕ, m < n → initial^(2^m) ≤ target) :=
by sorry

end NUMINAMATH_CALUDE_calculator_squaring_l221_22161


namespace NUMINAMATH_CALUDE_all_statements_imply_target_l221_22179

theorem all_statements_imply_target (p q r : Prop) :
  (p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ ¬q ∧ ¬r → ((p → q) → ¬r)) ∧
  (p ∧ q ∧ ¬r → ((p → q) → ¬r)) ∧
  (¬p ∧ q ∧ ¬r → ((p → q) → ¬r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_target_l221_22179


namespace NUMINAMATH_CALUDE_sams_mystery_books_l221_22124

theorem sams_mystery_books (total_books : ℝ) (used_adventure_books : ℝ) (new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13)
  (h3 : new_crime_books = 15) :
  total_books - (used_adventure_books + new_crime_books) = 17 :=
by sorry

end NUMINAMATH_CALUDE_sams_mystery_books_l221_22124


namespace NUMINAMATH_CALUDE_transformed_graph_point_l221_22168

theorem transformed_graph_point (f : ℝ → ℝ) (h : f 12 = 5) :
  ∃ (x y : ℝ), 1.5 * y = (f (3 * x) + 3) / 3 ∧ x = 4 ∧ y = 16 / 9 ∧ x + y = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_transformed_graph_point_l221_22168


namespace NUMINAMATH_CALUDE_article_cost_l221_22198

/-- The cost of an article given specific selling price conditions -/
theorem article_cost (selling_price_high : ℝ) (selling_price_low : ℝ) 
  (price_difference : ℝ) (gain_difference_percent : ℝ) :
  selling_price_high = 350 →
  selling_price_low = 340 →
  price_difference = selling_price_high - selling_price_low →
  gain_difference_percent = 5 →
  price_difference = (gain_difference_percent / 100) * 200 →
  200 = price_difference / (gain_difference_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l221_22198


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l221_22106

/-- Proves the cost of cheaper candy given the conditions of the mixture --/
theorem candy_mixture_cost (expensive_weight : ℝ) (expensive_cost : ℝ) 
  (cheaper_weight : ℝ) (mixture_cost : ℝ) (total_weight : ℝ) :
  expensive_weight = 20 →
  expensive_cost = 8 →
  cheaper_weight = 40 →
  mixture_cost = 6 →
  total_weight = expensive_weight + cheaper_weight →
  ∃ (cheaper_cost : ℝ),
    cheaper_cost = 5 ∧
    expensive_weight * expensive_cost + cheaper_weight * cheaper_cost = 
      total_weight * mixture_cost :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l221_22106


namespace NUMINAMATH_CALUDE_carbonic_acid_formation_l221_22126

-- Define the molecules and their quantities
structure Molecule where
  name : String
  moles : ℕ

-- Define the reaction
def reaction (reactant1 reactant2 product : Molecule) : Prop :=
  reactant1.name = "CO2" ∧ 
  reactant2.name = "H2O" ∧ 
  product.name = "H2CO3" ∧
  reactant1.moles = reactant2.moles ∧
  product.moles = min reactant1.moles reactant2.moles

-- Theorem statement
theorem carbonic_acid_formation 
  (co2 : Molecule) 
  (h2o : Molecule) 
  (h2co3 : Molecule) :
  co2.name = "CO2" →
  h2o.name = "H2O" →
  h2co3.name = "H2CO3" →
  co2.moles = 3 →
  h2o.moles = 3 →
  reaction co2 h2o h2co3 →
  h2co3.moles = 3 :=
by sorry

end NUMINAMATH_CALUDE_carbonic_acid_formation_l221_22126


namespace NUMINAMATH_CALUDE_worker_completion_time_l221_22132

/-- Given two workers A and B, where A is thrice as fast as B, 
    and together they can complete a job in 18 days,
    prove that A alone can complete the job in 24 days. -/
theorem worker_completion_time 
  (speed_A speed_B : ℝ) 
  (combined_time : ℝ) :
  speed_A = 3 * speed_B →
  1 / speed_A + 1 / speed_B = 1 / combined_time →
  combined_time = 18 →
  1 / speed_A = 1 / 24 :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l221_22132


namespace NUMINAMATH_CALUDE_jan_drove_more_than_ian_l221_22151

/-- Prove that Jan drove 174 miles more than Ian given the conditions --/
theorem jan_drove_more_than_ian (ian_time : ℝ) (ian_speed : ℝ) : 
  let han_time := ian_time + 1.5
  let han_speed := ian_speed + 6
  let jan_time := ian_time + 3
  let jan_speed := ian_speed + 8
  let ian_distance := ian_speed * ian_time
  let han_distance := han_speed * han_time
  han_distance - ian_distance = 84 →
  jan_speed * jan_time - ian_speed * ian_time = 174 :=
by sorry

end NUMINAMATH_CALUDE_jan_drove_more_than_ian_l221_22151


namespace NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l221_22148

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

theorem intersection_theorem (m : ℝ) :
  A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 := by sorry

theorem subset_theorem (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_subset_theorem_l221_22148


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_9_4_l221_22112

theorem smallest_four_digit_mod_9_4 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n % 9 = 4) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m % 9 = 4 → m ≥ n) ∧ 
  (n = 1003) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_9_4_l221_22112


namespace NUMINAMATH_CALUDE_min_trees_for_three_types_l221_22133

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset TreeType)
  (total_count : ℕ)
  (type_count : TreeType → ℕ)
  (total_is_100 : total_count = 100)
  (sum_of_types : (type_count TreeType.Birch) + (type_count TreeType.Spruce) + (type_count TreeType.Pine) + (type_count TreeType.Aspen) = total_count)
  (all_types_in_85 : ∀ (subset : Finset TreeType), subset.card = 85 → (∀ t : TreeType, t ∈ subset))

/-- The main theorem to be proved -/
theorem min_trees_for_three_types (g : Grove) :
  ∃ (n : ℕ), n = 69 ∧ 
  (∀ (subset : Finset TreeType), subset.card ≥ n → (∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset)) ∧
  (∃ (subset : Finset TreeType), subset.card = n - 1 ∧ (∀ (t1 t2 t3 : TreeType), t1 ≠ t2 → t2 ≠ t3 → t1 ≠ t3 → ¬(t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset))) :=
sorry

end NUMINAMATH_CALUDE_min_trees_for_three_types_l221_22133


namespace NUMINAMATH_CALUDE_valid_set_iff_ge_four_l221_22176

/-- A set of positive integers satisfying the given conditions -/
def ValidSet (n : ℕ) (S : Finset ℕ) : Prop :=
  (S.card = n) ∧
  (∀ x ∈ S, x > 0 ∧ x < 2^(n-1)) ∧
  (∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ ∅ → B ≠ ∅ → A ≠ B →
    (A.sum id ≠ B.sum id))

/-- The main theorem stating the existence of a valid set if and only if n ≥ 4 -/
theorem valid_set_iff_ge_four (n : ℕ) :
  (∃ S : Finset ℕ, ValidSet n S) ↔ n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_set_iff_ge_four_l221_22176


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l221_22181

/-- Theorem about a specific triangle ABC --/
theorem triangle_abc_properties :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = π / 3 →  -- 60° in radians
  b = 1 →
  c = 4 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →  -- Cosine rule
  (a = Real.sqrt 13 ∧ 
   (1 / 2) * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l221_22181


namespace NUMINAMATH_CALUDE_equal_gender_probability_l221_22117

def total_students : ℕ := 8
def men_count : ℕ := 4
def women_count : ℕ := 4
def selection_size : ℕ := 4

theorem equal_gender_probability :
  let total_ways := Nat.choose total_students selection_size
  let ways_to_choose_men := Nat.choose men_count (selection_size / 2)
  let ways_to_choose_women := Nat.choose women_count (selection_size / 2)
  (ways_to_choose_men * ways_to_choose_women : ℚ) / total_ways = 18 / 35 := by
  sorry

end NUMINAMATH_CALUDE_equal_gender_probability_l221_22117


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l221_22184

theorem sqrt_equation_solution (x : ℝ) : 
  x > 2 → (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 5 / 2) → x = 50 / 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l221_22184


namespace NUMINAMATH_CALUDE_angle_XYZ_measure_l221_22104

-- Define the regular octagon
def RegularOctagon : Type := Unit

-- Define the square inside the octagon
def Square : Type := Unit

-- Define the vertices
def X : RegularOctagon := Unit.unit
def Y : Square := Unit.unit
def Z : Square := Unit.unit

-- Define the angle measure function
def angle_measure : RegularOctagon → Square → Square → ℝ := sorry

-- State the theorem
theorem angle_XYZ_measure (o : RegularOctagon) (s : Square) :
  angle_measure X Y Z = 90 := by sorry

end NUMINAMATH_CALUDE_angle_XYZ_measure_l221_22104


namespace NUMINAMATH_CALUDE_hexagon_area_decrease_l221_22166

/-- The area decrease of a regular hexagon when its sides are shortened -/
theorem hexagon_area_decrease (initial_area : ℝ) (side_decrease : ℝ) : 
  initial_area = 150 * Real.sqrt 3 →
  side_decrease = 3 →
  let original_side := Real.sqrt (200 / 3)
  let new_side := original_side - side_decrease
  let new_area := 3 * Real.sqrt 3 / 2 * new_side ^ 2
  initial_area - new_area = 76.5 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_hexagon_area_decrease_l221_22166


namespace NUMINAMATH_CALUDE_workshop_allocation_valid_l221_22182

/-- Represents the allocation of workers in a workshop producing bolts and nuts. -/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given worker allocation satisfies the workshop conditions. -/
def is_valid_allocation (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
    (nuts_per_bolt : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.bolt_workers + allocation.nut_workers = total_workers ∧
  2 * (bolts_per_worker * allocation.bolt_workers) = nuts_per_worker * allocation.nut_workers

/-- Theorem stating that the specific allocation of 40 bolt workers and 50 nut workers
    is a valid solution to the workshop problem. -/
theorem workshop_allocation_valid : 
  is_valid_allocation 90 15 24 2 ⟨40, 50⟩ := by
  sorry


end NUMINAMATH_CALUDE_workshop_allocation_valid_l221_22182


namespace NUMINAMATH_CALUDE_harrys_journey_l221_22152

theorem harrys_journey (total_time bus_time_so_far : ℕ) 
  (h1 : total_time = 60)
  (h2 : bus_time_so_far = 15)
  (h3 : ∃ (total_bus_time walking_time : ℕ), 
    total_bus_time + walking_time = total_time ∧
    walking_time = total_bus_time / 2) :
  ∃ (remaining_bus_time : ℕ), 
    remaining_bus_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_harrys_journey_l221_22152


namespace NUMINAMATH_CALUDE_right_triangle_groups_l221_22144

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_groups :
  ¬ (is_right_triangle 1.5 2 3) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 9 12 15) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_groups_l221_22144


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l221_22163

/-- A triangle with interior angles in the ratio 1:2:3 has its largest angle equal to 90 degrees -/
theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l221_22163


namespace NUMINAMATH_CALUDE_min_value_expression_l221_22159

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 6*x + 4*y + 18 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l221_22159


namespace NUMINAMATH_CALUDE_snow_probability_both_days_l221_22196

def prob_snow_monday : ℝ := 0.4
def prob_snow_tuesday : ℝ := 0.3

theorem snow_probability_both_days :
  let prob_both_days := prob_snow_monday * prob_snow_tuesday
  prob_both_days = 0.12 := by sorry

end NUMINAMATH_CALUDE_snow_probability_both_days_l221_22196


namespace NUMINAMATH_CALUDE_glasses_per_pitcher_l221_22158

theorem glasses_per_pitcher (total_glasses : ℕ) (num_pitchers : ℕ) 
  (h1 : total_glasses = 54) 
  (h2 : num_pitchers = 9) : 
  total_glasses / num_pitchers = 6 := by
sorry

end NUMINAMATH_CALUDE_glasses_per_pitcher_l221_22158


namespace NUMINAMATH_CALUDE_sequence_is_geometric_l221_22127

theorem sequence_is_geometric (a : ℝ) (h : a ≠ 0) :
  (∃ S : ℕ → ℝ, ∀ n : ℕ, S n = a^n - 1) →
  (∃ r : ℝ, ∀ n : ℕ, ∃ u : ℕ → ℝ, u (n+1) = r * u n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_is_geometric_l221_22127


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l221_22123

-- Define the curve
def C (x : ℝ) : ℝ := x^2 + x

-- Define the derivative of the curve
def C_derivative (x : ℝ) : ℝ := 2*x + 1

-- Define the slope of the tangent line at x = 1
def tangent_slope : ℝ := C_derivative 1

-- Define the condition for perpendicularity
def perpendicular_condition (a : ℝ) : Prop :=
  tangent_slope * a = -1

-- The theorem to prove
theorem tangent_perpendicular_line : 
  ∃ (a : ℝ), perpendicular_condition a ∧ a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l221_22123


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l221_22102

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability that no two adjacent people roll the same number -/
def no_adjacent_same_prob : ℚ := 441 / 8192

theorem circular_table_dice_probability :
  let n := num_people
  let s := die_sides
  (n : ℚ) > 0 ∧ (s : ℚ) > 0 →
  no_adjacent_same_prob = 441 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l221_22102


namespace NUMINAMATH_CALUDE_max_cards_purchasable_l221_22177

def initial_money : ℚ := 965 / 100
def earned_money : ℚ := 535 / 100
def card_cost : ℚ := 95 / 100

theorem max_cards_purchasable : 
  ⌊(initial_money + earned_money) / card_cost⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_max_cards_purchasable_l221_22177


namespace NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l221_22116

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to get the units digit of a number raised to a power
def unitsDigitPower (base : ℕ) (exp : ℕ) : ℕ :=
  unitsDigit (unitsDigit base ^ exp)

theorem sum_units_digits_734_99_347_83 : 
  (unitsDigitPower 734 99 + unitsDigitPower 347 83) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l221_22116


namespace NUMINAMATH_CALUDE_visited_neither_country_l221_22169

theorem visited_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) : 
  total = 60 → iceland = 35 → norway = 23 → both = 31 → 
  total - (iceland + norway - both) = 33 := by
sorry

end NUMINAMATH_CALUDE_visited_neither_country_l221_22169


namespace NUMINAMATH_CALUDE_lily_sees_leo_l221_22193

/-- The time Lily can see Leo given their speeds and distances -/
theorem lily_sees_leo (lily_speed leo_speed initial_distance final_distance : ℝ) : 
  lily_speed = 15 → 
  leo_speed = 9 → 
  initial_distance = 0.75 → 
  final_distance = 0.75 → 
  (initial_distance + final_distance) / (lily_speed - leo_speed) * 60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lily_sees_leo_l221_22193


namespace NUMINAMATH_CALUDE_product_decrease_theorem_l221_22101

theorem product_decrease_theorem :
  ∃ (a b c d e : ℕ), 
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) = 15 * (a * b * c * d * e) := by
  sorry

end NUMINAMATH_CALUDE_product_decrease_theorem_l221_22101


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l221_22172

def ellipse_equation (x y : ℝ) : Prop := x^2 / 27 + y^2 / 36 = 1

def hyperbola_equation (a b x y : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1

theorem hyperbola_standard_equation :
  ∃ a b : ℝ,
    (∀ x y : ℝ, ellipse_equation x y → hyperbola_equation a b x y) ∧
    hyperbola_equation a b (Real.sqrt 15) 4 ∧
    a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l221_22172


namespace NUMINAMATH_CALUDE_quadratic_root_negative_reciprocal_l221_22118

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is the negative reciprocal of the other, then c = -a. -/
theorem quadratic_root_negative_reciprocal (a b c : ℝ) (α β : ℝ) : 
  a ≠ 0 →  -- Ensure the equation is quadratic
  a * α^2 + b * α + c = 0 →  -- α is a root
  a * β^2 + b * β + c = 0 →  -- β is a root
  β = -1 / α →  -- One root is the negative reciprocal of the other
  c = -a := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_negative_reciprocal_l221_22118


namespace NUMINAMATH_CALUDE_sum_factorials_25_divisible_by_26_l221_22156

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_25_divisible_by_26 :
  ∃ k : ℕ, sum_factorials 25 = 26 * k :=
sorry

end NUMINAMATH_CALUDE_sum_factorials_25_divisible_by_26_l221_22156


namespace NUMINAMATH_CALUDE_largest_factor_of_consecutive_product_l221_22154

theorem largest_factor_of_consecutive_product (n : ℕ) : 
  n % 10 = 4 → 120 ∣ n * (n + 1) * (n + 2) ∧ 
  ∀ m : ℕ, m > 120 → ∃ k : ℕ, k % 10 = 4 ∧ ¬(m ∣ k * (k + 1) * (k + 2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_factor_of_consecutive_product_l221_22154


namespace NUMINAMATH_CALUDE_average_weight_problem_l221_22173

theorem average_weight_problem (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ) 
  (group2_avg_weight : ℚ) (total_avg_weight : ℚ) :
  total_boys = group1_boys + group2_boys →
  total_boys = 24 →
  group1_boys = 16 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  (group1_boys * (50.25 : ℚ) + group2_boys * group2_avg_weight) / total_boys = total_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l221_22173


namespace NUMINAMATH_CALUDE_roof_collapse_time_l221_22137

/-- The number of days it takes for Bill's roof to collapse under the weight of leaves -/
def days_to_collapse (roof_capacity : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : ℕ :=
  (roof_capacity * leaves_per_pound) / leaves_per_day

/-- Theorem stating that it takes 5000 days for Bill's roof to collapse -/
theorem roof_collapse_time :
  days_to_collapse 500 100 1000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_roof_collapse_time_l221_22137


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l221_22146

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 16 * x^2 + m * x + 4 = 0) ↔ m = 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l221_22146


namespace NUMINAMATH_CALUDE_perpendicular_distance_approx_l221_22191

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  d : Point3D
  a : Point3D
  b : Point3D
  c : Point3D

/-- Calculates the perpendicular distance from a point to a plane defined by three points -/
def perpendicularDistance (p : Point3D) (a b c : Point3D) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem perpendicular_distance_approx (p : Parallelepiped) : 
  p.length = 5 ∧ p.width = 3 ∧ p.height = 2 ∧
  p.d = ⟨0, 0, 0⟩ ∧ p.a = ⟨5, 0, 0⟩ ∧ p.b = ⟨0, 3, 0⟩ ∧ p.c = ⟨0, 0, 2⟩ →
  abs (perpendicularDistance p.d p.a p.b p.c - 1.9) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_distance_approx_l221_22191


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l221_22115

theorem beef_weight_loss_percentage (initial_weight : Real) (processed_weight : Real) 
  (h1 : initial_weight = 861.54)
  (h2 : processed_weight = 560) :
  let weight_loss := initial_weight - processed_weight
  let percentage_loss := (weight_loss / initial_weight) * 100
  ∃ ε > 0, abs (percentage_loss - 34.99) < ε :=
by sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l221_22115


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l221_22190

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 25) = Real.sqrt 170 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l221_22190


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l221_22131

theorem smallest_right_triangle_area :
  let side1 : ℝ := 6
  let side2 : ℝ := 8
  let area1 : ℝ := (1/2) * side1 * side2
  let area2 : ℝ := (1/2) * side1 * Real.sqrt (side2^2 - side1^2)
  min area1 area2 = 6 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l221_22131


namespace NUMINAMATH_CALUDE_soda_cost_l221_22134

/-- The cost of items in cents -/
structure Cost where
  burger : ℕ
  soda : ℕ
  fry : ℕ

/-- The problem statement -/
theorem soda_cost (c : Cost) : 
  (3 * c.burger + 2 * c.soda + 2 * c.fry = 590) ∧ 
  (2 * c.burger + 3 * c.soda + c.fry = 610) → 
  c.soda = 140 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l221_22134


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l221_22188

def g (x : ℝ) : ℝ := 2 * x^6 + 3 * x^4 - x^2 + 7

theorem g_sum_symmetric (h : g 5 = 29) : g 5 + g (-5) = 58 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l221_22188


namespace NUMINAMATH_CALUDE_father_age_twice_marika_l221_22108

/-- The year when Marika's father's age will be twice Marika's age -/
def target_year : ℕ := 2036

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of reference -/
def reference_year : ℕ := 2006

/-- Father's age is five times Marika's age in 2006 -/
def father_age_2006 : ℕ := 5 * marika_age_2006

theorem father_age_twice_marika (y : ℕ) :
  y = target_year →
  father_age_2006 + (y - reference_year) = 2 * (marika_age_2006 + (y - reference_year)) :=
by sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_l221_22108


namespace NUMINAMATH_CALUDE_base_ten_is_only_solution_l221_22100

/-- Represents the number in base b as a function of n -/
def number (b n : ℕ) : ℚ :=
  (b^(3*n) - b^(2*n+1) + 7 * b^(2*n) + b^(n+1) - 7 * b^n - 1) / (3 * (b - 1))

/-- Predicate to check if a rational number is a perfect cube -/
def is_perfect_cube (q : ℚ) : Prop :=
  ∃ m : ℤ, q = (m : ℚ)^3

theorem base_ten_is_only_solution :
  ∀ b : ℕ, b ≥ 9 →
  (∀ n : ℕ, ∃ N : ℕ, ∀ m : ℕ, m ≥ N → is_perfect_cube (number b m)) →
  b = 10 :=
sorry

end NUMINAMATH_CALUDE_base_ten_is_only_solution_l221_22100


namespace NUMINAMATH_CALUDE_complex_number_problem_l221_22178

theorem complex_number_problem (a b c : ℂ) (h_a_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l221_22178


namespace NUMINAMATH_CALUDE_f_comparison_and_max_value_l221_22189

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem f_comparison_and_max_value :
  (f (π / 4) > f (π / 6)) ∧
  (∀ x : ℝ, f x ≤ -3/2) ∧
  (∃ x : ℝ, f x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_f_comparison_and_max_value_l221_22189


namespace NUMINAMATH_CALUDE_fixed_internet_charge_l221_22160

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  total_eq : totalCharge = callCharge + internetCharge

/-- Theorem stating the fixed monthly internet charge -/
theorem fixed_internet_charge 
  (jan : MonthlyBill) 
  (feb : MonthlyBill) 
  (jan_total : jan.totalCharge = 46)
  (feb_total : feb.totalCharge = 76)
  (feb_call_charge : feb.callCharge = 2 * jan.callCharge)
  : jan.internetCharge = 16 := by
  sorry

end NUMINAMATH_CALUDE_fixed_internet_charge_l221_22160


namespace NUMINAMATH_CALUDE_prob_both_type_a_prob_different_types_l221_22186

/-- Represents the total number of questions -/
def total_questions : ℕ := 6

/-- Represents the number of type A questions -/
def type_a_questions : ℕ := 4

/-- Represents the number of type B questions -/
def type_b_questions : ℕ := 2

/-- Represents the number of questions to be selected -/
def selected_questions : ℕ := 2

/-- The probability of selecting 2 questions of type A -/
theorem prob_both_type_a : 
  (Nat.choose type_a_questions selected_questions : ℚ) / 
  (Nat.choose total_questions selected_questions : ℚ) = 2/5 := by sorry

/-- The probability of selecting 2 questions of different types -/
theorem prob_different_types :
  ((type_a_questions * type_b_questions : ℚ) / 
  (Nat.choose total_questions selected_questions : ℚ)) = 8/15 := by sorry

end NUMINAMATH_CALUDE_prob_both_type_a_prob_different_types_l221_22186


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l221_22119

/-- Proves that the ratio of cheetahs to snakes is 7:10 given the zoo animal counts --/
theorem zoo_animal_ratio : 
  ∀ (snakes arctic_foxes leopards bee_eaters alligators cheetahs total : ℕ),
  snakes = 100 →
  arctic_foxes = 80 →
  leopards = 20 →
  bee_eaters = 10 * leopards →
  alligators = 2 * (arctic_foxes + leopards) →
  total = 670 →
  total = snakes + arctic_foxes + leopards + bee_eaters + alligators + cheetahs →
  (cheetahs : ℚ) / snakes = 7 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l221_22119


namespace NUMINAMATH_CALUDE_rook_domino_tiling_impossibility_l221_22107

theorem rook_domino_tiling_impossibility :
  ∀ (rook_positions : Finset (Fin 10 × Fin 10)),
    (rook_positions.card = 10) →
    (∀ (r1 r2 : Fin 10 × Fin 10), r1 ∈ rook_positions → r2 ∈ rook_positions → r1 ≠ r2 →
      (r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)) →
    ¬∃ (domino_placements : Finset (Fin 10 × Fin 10 × Bool)),
      (domino_placements.card = 45) ∧
      (∀ (d : Fin 10 × Fin 10 × Bool), d ∈ domino_placements →
        (d.1, d.2.1) ∉ rook_positions ∧
        (if d.2.2 then (d.1 + 1, d.2.1) ∉ rook_positions
         else (d.1, d.2.1 + 1) ∉ rook_positions)) ∧
      (∀ (p : Fin 10 × Fin 10), p ∉ rook_positions →
        (∃ (d : Fin 10 × Fin 10 × Bool), d ∈ domino_placements ∧
          (d.1, d.2.1) = p ∨
          (if d.2.2 then (d.1 + 1, d.2.1) = p else (d.1, d.2.1 + 1) = p))) :=
by
  sorry


end NUMINAMATH_CALUDE_rook_domino_tiling_impossibility_l221_22107


namespace NUMINAMATH_CALUDE_jiahao_estimate_l221_22197

theorem jiahao_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x + 2) - (y - 1) > x - y := by
  sorry

end NUMINAMATH_CALUDE_jiahao_estimate_l221_22197


namespace NUMINAMATH_CALUDE_factorial_200_less_than_100_pow_200_l221_22120

-- Define factorial
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Theorem statement
theorem factorial_200_less_than_100_pow_200 :
  factorial 200 < 100^200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_200_less_than_100_pow_200_l221_22120


namespace NUMINAMATH_CALUDE_mean_interior_angles_quadrilateral_l221_22187

-- Define a quadrilateral
def Quadrilateral : Type := Unit

-- Define the function that gives the sum of interior angles of a quadrilateral
def sum_interior_angles (q : Quadrilateral) : ℝ := 360

-- Define the number of interior angles in a quadrilateral
def num_interior_angles (q : Quadrilateral) : ℕ := 4

-- Theorem: The mean value of the measures of the four interior angles of any quadrilateral is 90°
theorem mean_interior_angles_quadrilateral (q : Quadrilateral) :
  (sum_interior_angles q) / (num_interior_angles q : ℝ) = 90 := by sorry

end NUMINAMATH_CALUDE_mean_interior_angles_quadrilateral_l221_22187


namespace NUMINAMATH_CALUDE_gardener_tree_rows_l221_22192

/-- Proves that the initial number of rows is 24 given the gardener's tree planting conditions -/
theorem gardener_tree_rows : ∀ r : ℕ, 
  (42 * r = 28 * (r + 12)) → r = 24 := by
  sorry

end NUMINAMATH_CALUDE_gardener_tree_rows_l221_22192


namespace NUMINAMATH_CALUDE_mnp_product_l221_22128

theorem mnp_product (a b x y : ℝ) (m n p : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) → 
  m * n * p = 2 := by sorry

end NUMINAMATH_CALUDE_mnp_product_l221_22128


namespace NUMINAMATH_CALUDE_a_is_integer_l221_22195

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => ((2 * n + 3) * a (n + 1) + 3 * (n + 1) * a n) / (n + 2)

theorem a_is_integer (n : ℕ) : ∃ k : ℤ, a n = k := by
  sorry

end NUMINAMATH_CALUDE_a_is_integer_l221_22195


namespace NUMINAMATH_CALUDE_ellipse_and_distance_l221_22194

/-- An ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The configuration of points and lines for the problem -/
structure Configuration (C : Ellipse) where
  right_focus : ℝ × ℝ
  passing_point : ℝ × ℝ
  M : ℝ
  l : ℝ → ℝ → Prop
  A : ℝ × ℝ
  B : ℝ × ℝ
  N : ℝ × ℝ
  h₁ : right_focus = (Real.sqrt 3, 0)
  h₂ : passing_point = (-1, Real.sqrt 3 / 2)
  h₃ : C.a^2 * (passing_point.1^2 / C.a^2 + passing_point.2^2 / C.b^2) = C.a^2
  h₄ : l M A.2 ∧ l M B.2
  h₅ : A.2 > 0 ∧ B.2 < 0
  h₆ : (A.1 - M)^2 + A.2^2 = 4 * ((B.1 - M)^2 + B.2^2)
  h₇ : N.1^2 + N.2^2 = 4/7
  h₈ : ∀ x y, l x y → (x - N.1)^2 + (y - N.2)^2 ≥ 4/7

/-- The main theorem to be proved -/
theorem ellipse_and_distance (C : Ellipse) (cfg : Configuration C) :
  C.a^2 = 4 ∧ C.b^2 = 1 ∧ 
  (cfg.M - cfg.N.1)^2 + cfg.N.2^2 = (4 * Real.sqrt 21 / 21)^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_distance_l221_22194


namespace NUMINAMATH_CALUDE_least_eight_binary_digits_l221_22153

/-- The number of binary digits required to represent a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  (Nat.log2 n.val) + 1

/-- Theorem: 128 is the least positive integer that requires 8 binary digits -/
theorem least_eight_binary_digits :
  (∀ m : ℕ+, m < 128 → binaryDigits m < 8) ∧ binaryDigits 128 = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_eight_binary_digits_l221_22153


namespace NUMINAMATH_CALUDE_replaced_person_weight_l221_22140

def initial_group_size : ℕ := 8
def average_weight_increase : ℚ := 5/2
def new_person_weight : ℕ := 85

theorem replaced_person_weight :
  ∃ (w : ℕ), w = initial_group_size * average_weight_increase + new_person_weight - initial_group_size * average_weight_increase :=
by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l221_22140


namespace NUMINAMATH_CALUDE_hash_two_three_l221_22105

-- Define the operation #
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- Theorem statement
theorem hash_two_three : hash 2 3 = 12 := by sorry

end NUMINAMATH_CALUDE_hash_two_three_l221_22105


namespace NUMINAMATH_CALUDE_twelve_thousand_scientific_notation_l221_22139

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Checks if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  n = sn.coefficient * (10 : ℝ) ^ sn.exponent

theorem twelve_thousand_scientific_notation :
  ∃ sn : ScientificNotation, represents sn 12000 ∧ sn.coefficient = 1.2 ∧ sn.exponent = 4 := by
  sorry

end NUMINAMATH_CALUDE_twelve_thousand_scientific_notation_l221_22139


namespace NUMINAMATH_CALUDE_eggs_in_box_l221_22164

/-- The number of eggs Harry takes from the box -/
def eggs_taken : ℕ := 5

/-- The number of eggs left in the box after Harry takes some -/
def eggs_left : ℕ := 42

/-- The initial number of eggs in the box -/
def initial_eggs : ℕ := eggs_taken + eggs_left

theorem eggs_in_box : initial_eggs = 47 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l221_22164


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l221_22129

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- Function to get the number in a given square -/
def getNumber (s : Square) : Nat :=
  s.row * boardSize + s.col + 1

/-- The four corners of the board -/
def corners : List Square := [
  { row := 0, col := 0 },             -- Top left
  { row := 0, col := boardSize - 1 }, -- Top right
  { row := boardSize - 1, col := 0 }, -- Bottom left
  { row := boardSize - 1, col := boardSize - 1 }  -- Bottom right
]

/-- The sum of numbers in the four corners -/
def cornerSum : Nat := (corners.map getNumber).sum

theorem corner_sum_is_164 : cornerSum = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_164_l221_22129


namespace NUMINAMATH_CALUDE_isabellas_paintable_area_l221_22125

/-- Calculates the total paintable area for a set of identical rooms -/
def totalPaintableArea (
  numRooms : ℕ
  ) (length width height : ℝ
  ) (unpaintableAreaPerRoom : ℝ
  ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableAreaPerRoom
  numRooms * paintableAreaPerRoom

/-- Proves that the total paintable area for Isabella's bedrooms is 1592 square feet -/
theorem isabellas_paintable_area :
  totalPaintableArea 4 15 11 9 70 = 1592 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_paintable_area_l221_22125


namespace NUMINAMATH_CALUDE_complex_equation_solution_l221_22135

theorem complex_equation_solution :
  let z : ℂ := ((1 - Complex.I)^2 + 3 * (1 + Complex.I)) / (2 - Complex.I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - Complex.I ∧ z = 1 + Complex.I ∧ a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l221_22135


namespace NUMINAMATH_CALUDE_trigonometric_identities_l221_22122

theorem trigonometric_identities :
  (Real.sin (20 * π / 180))^2 + (Real.cos (80 * π / 180))^2 + Real.sqrt 3 * Real.sin (20 * π / 180) * Real.cos (80 * π / 180) = 1/4 ∧
  (Real.sin (20 * π / 180))^2 + (Real.cos (50 * π / 180))^2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l221_22122


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l221_22110

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 33 ∧ 
  (∀ (m : ℕ), m < n → ¬(87 ∣ (13605 - m))) ∧ 
  (87 ∣ (13605 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l221_22110


namespace NUMINAMATH_CALUDE_total_students_from_stratified_sample_l221_22167

/-- Given a stratified sample from a high school population, prove the total number of students. -/
theorem total_students_from_stratified_sample
  (total_sample : ℕ)
  (first_year_sample : ℕ)
  (third_year_sample : ℕ)
  (total_second_year : ℕ)
  (h1 : total_sample = 45)
  (h2 : first_year_sample = 20)
  (h3 : third_year_sample = 10)
  (h4 : total_second_year = 300) :
  ∃ (total_students : ℕ), total_students = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_students_from_stratified_sample_l221_22167


namespace NUMINAMATH_CALUDE_tan_half_theta_l221_22157

theorem tan_half_theta (θ : Real) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  (1 + Real.cos θ ≠ 0 → Real.tan (θ / 2) = 1 / 2) ∧
  (1 + Real.cos θ = 0 → ¬∃ (x : Real), Real.tan (θ / 2) = x) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_theta_l221_22157


namespace NUMINAMATH_CALUDE_specific_arrangement_eq_3456_l221_22175

/-- The number of ways to arrange players from different teams in a row -/
def arrange_players (num_teams : ℕ) (team_sizes : List ℕ) : ℕ :=
  (Nat.factorial num_teams) * (team_sizes.map Nat.factorial).prod

/-- The specific arrangement for the given problem -/
def specific_arrangement : ℕ :=
  arrange_players 4 [3, 2, 3, 2]

/-- Theorem stating that the specific arrangement equals 3456 -/
theorem specific_arrangement_eq_3456 : specific_arrangement = 3456 := by
  sorry

end NUMINAMATH_CALUDE_specific_arrangement_eq_3456_l221_22175


namespace NUMINAMATH_CALUDE_positive_A_value_l221_22130

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h1 : hash A 7 = 72) (h2 : A > 0) : A = 11 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l221_22130


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l221_22103

/-- For an ellipse with equation x²/4 + y²/9 = 1, the focal length is 2√5 -/
theorem ellipse_focal_length : 
  ∀ (x y : ℝ), x^2/4 + y^2/9 = 1 → 
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 ∧ 
  (∃ (c : ℝ), c^2 = 5 ∧ f = 2*c) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l221_22103


namespace NUMINAMATH_CALUDE_multiplier_satisfies_equation_l221_22165

/-- The multiplier that satisfies the equation when the number is 5.0 -/
def multiplier : ℝ := 7

/-- The given number in the problem -/
def number : ℝ := 5.0

/-- Theorem stating that the multiplier satisfies the equation -/
theorem multiplier_satisfies_equation : 
  4 * number + multiplier * number = 55 := by sorry

end NUMINAMATH_CALUDE_multiplier_satisfies_equation_l221_22165


namespace NUMINAMATH_CALUDE_walking_speed_problem_l221_22149

/-- Given two people walking in the same direction for 10 hours, where one walks at 7.5 kmph
    and they end up 20 km apart, prove that the speed of the other person is 9.5 kmph. -/
theorem walking_speed_problem (v : ℝ) 
  (h1 : (v - 7.5) * 10 = 20) : v = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l221_22149


namespace NUMINAMATH_CALUDE_calculate_books_arlo_book_count_l221_22147

/-- Given a ratio of books to pens and a total number of items, calculate the number of books. -/
theorem calculate_books (book_ratio : ℕ) (pen_ratio : ℕ) (total_items : ℕ) : ℕ :=
  let total_ratio := book_ratio + pen_ratio
  let items_per_part := total_items / total_ratio
  book_ratio * items_per_part

/-- Prove that given a ratio of books to pens of 7:3 and a total of 400 stationery items, the number of books is 280. -/
theorem arlo_book_count : calculate_books 7 3 400 = 280 := by
  sorry

end NUMINAMATH_CALUDE_calculate_books_arlo_book_count_l221_22147


namespace NUMINAMATH_CALUDE_planes_parallel_condition_l221_22174

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines and planes
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Define the intersection operation for lines
variable (intersect : Line → Line → Set Point)

-- Define the specific lines and planes
variable (m n l₁ l₂ : Line) (α β : Plane) (M : Point)

-- State the theorem
theorem planes_parallel_condition 
  (h1 : subset m α)
  (h2 : subset n α)
  (h3 : subset l₁ β)
  (h4 : subset l₂ β)
  (h5 : intersect l₁ l₂ = {M})
  (h6 : parallel m l₁)
  (h7 : parallel n l₂) :
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_condition_l221_22174


namespace NUMINAMATH_CALUDE_unique_zero_iff_a_nonpositive_l221_22109

/-- A function f(x) = x^3 - 3ax has a unique zero if and only if a ≤ 0 -/
theorem unique_zero_iff_a_nonpositive (a : ℝ) :
  (∃! x, x^3 - 3*a*x = 0) ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_unique_zero_iff_a_nonpositive_l221_22109


namespace NUMINAMATH_CALUDE_max_students_distribution_l221_22143

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1230) (h_pencils : pencils = 920) : 
  (Nat.gcd pens pencils) = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l221_22143


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l221_22136

def B : Set ℕ := {n | ∃ k : ℕ, n = k + (k + 1) + (k + 2) + (k + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, (∀ b ∈ B, m ∣ b) → m ∣ d) ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l221_22136


namespace NUMINAMATH_CALUDE_lemonade_syrup_parts_l221_22113

/-- Given a solution with water and lemonade syrup, prove the original amount of syrup --/
theorem lemonade_syrup_parts (x : ℝ) : 
  x > 0 → -- Ensure x is positive
  x / (x + 8) ≠ 1/5 → -- Ensure the original solution is not already 20% syrup
  x / (x + 8 - 2.1428571428571423 + 2.1428571428571423) = 1/5 → -- After replacement, solution is 20% syrup
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_syrup_parts_l221_22113


namespace NUMINAMATH_CALUDE_additional_money_needed_l221_22121

/-- The cost of the dictionary -/
def dictionary_cost : ℚ := 5.50

/-- The cost of the dinosaur book -/
def dinosaur_book_cost : ℚ := 11.25

/-- The cost of the children's cookbook -/
def cookbook_cost : ℚ := 5.75

/-- The cost of the science experiment kit -/
def science_kit_cost : ℚ := 8.40

/-- The cost of the set of colored pencils -/
def pencils_cost : ℚ := 3.60

/-- The amount Emir has saved -/
def saved_amount : ℚ := 24.50

/-- The theorem stating how much more money Emir needs -/
theorem additional_money_needed :
  dictionary_cost + dinosaur_book_cost + cookbook_cost + science_kit_cost + pencils_cost - saved_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l221_22121


namespace NUMINAMATH_CALUDE_f_properties_l221_22180

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 + a*x|

-- Define the property of being monotonically increasing on [0,1]
def monotone_increasing_on_unit_interval (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → g x ≤ g y

-- Define M(a) as the maximum value of f(x) on [0,1]
noncomputable def M (a : ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc 0 1), f a x

-- State the theorem
theorem f_properties (a : ℝ) :
  (monotone_increasing_on_unit_interval (f a) ↔ a ≤ -2 ∨ a ≥ 0) ∧
  (∃ (a_min : ℝ), ∀ (a : ℝ), M a_min ≤ M a ∧ M a_min = 3 - 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l221_22180


namespace NUMINAMATH_CALUDE_birds_storks_difference_l221_22183

theorem birds_storks_difference (initial_storks initial_birds additional_birds : ℕ) : 
  initial_storks = 5 →
  initial_birds = 3 →
  additional_birds = 4 →
  (initial_birds + additional_birds) - initial_storks = 2 :=
by sorry

end NUMINAMATH_CALUDE_birds_storks_difference_l221_22183


namespace NUMINAMATH_CALUDE_sum_product_equal_470_l221_22162

theorem sum_product_equal_470 : 
  (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equal_470_l221_22162


namespace NUMINAMATH_CALUDE_order_xyz_l221_22142

theorem order_xyz (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (x : ℝ) (hx : x = (a+b)*(c+d))
  (y : ℝ) (hy : y = (a+c)*(b+d))
  (z : ℝ) (hz : z = (a+d)*(b+c)) :
  x < y ∧ y < z :=
by sorry

end NUMINAMATH_CALUDE_order_xyz_l221_22142


namespace NUMINAMATH_CALUDE_fraction_power_approximation_l221_22145

theorem fraction_power_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000000000000001 ∧ 
  |((1 : ℝ) / 9)^2 - 0.012345679012345678| < ε :=
sorry

end NUMINAMATH_CALUDE_fraction_power_approximation_l221_22145


namespace NUMINAMATH_CALUDE_max_house_paintable_area_l221_22155

/-- The total area of walls to be painted in Max's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Theorem stating the total area of walls to be painted in Max's house -/
theorem max_house_paintable_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end NUMINAMATH_CALUDE_max_house_paintable_area_l221_22155


namespace NUMINAMATH_CALUDE_A_intersect_C_U_B_eq_open_zero_closed_two_l221_22141

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x > 2}

-- Define the complement of B in the universal set ℝ
def C_U_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem A_intersect_C_U_B_eq_open_zero_closed_two : 
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_C_U_B_eq_open_zero_closed_two_l221_22141


namespace NUMINAMATH_CALUDE_texas_integrated_school_students_l221_22185

theorem texas_integrated_school_students (original_classes : ℕ) (students_per_class : ℕ) (new_classes : ℕ) : 
  original_classes = 15 → 
  students_per_class = 20 → 
  new_classes = 5 → 
  (original_classes + new_classes) * students_per_class = 400 := by
sorry

end NUMINAMATH_CALUDE_texas_integrated_school_students_l221_22185


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l221_22170

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 700)
  (h2 : blue_percent = 45/100)
  (h3 : red_percent = 23/100)
  (h4 : green_percent = 15/100) :
  ⌊(1 - (blue_percent + red_percent + green_percent)) * total_students⌋ = 119 := by
sorry

end NUMINAMATH_CALUDE_students_wearing_other_colors_l221_22170


namespace NUMINAMATH_CALUDE_polly_cooking_time_l221_22114

/-- The number of minutes Polly spends cooking breakfast each day -/
def breakfast_time : ℕ := 20

/-- The number of minutes Polly spends cooking lunch each day -/
def lunch_time : ℕ := 5

/-- The number of minutes Polly spends cooking dinner on 4 days of the week -/
def dinner_time_short : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on the other 3 days of the week -/
def dinner_time_long : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Polly spends less time cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of days Polly spends more time cooking dinner -/
def long_dinner_days : ℕ := days_in_week - short_dinner_days

/-- The total time Polly spends cooking in a week -/
def total_cooking_time : ℕ :=
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  dinner_time_short * short_dinner_days +
  dinner_time_long * long_dinner_days

/-- Theorem stating that Polly spends 305 minutes cooking in a week -/
theorem polly_cooking_time : total_cooking_time = 305 := by
  sorry

end NUMINAMATH_CALUDE_polly_cooking_time_l221_22114


namespace NUMINAMATH_CALUDE_expression_evaluation_l221_22199

theorem expression_evaluation (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l221_22199


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l221_22171

theorem shaded_area_ratio (square_side : ℝ) (h : square_side = 8) :
  let r := square_side / 2
  let semicircle_area := π * r^2 / 2
  let quarter_circle_area := π * r^2 / 4
  let shaded_area := 2 * semicircle_area - quarter_circle_area
  let full_circle_area := π * r^2
  shaded_area / full_circle_area = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l221_22171


namespace NUMINAMATH_CALUDE_intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l221_22150

noncomputable def f (x : ℝ) := Real.exp x - x

theorem intersection_angle_zero : 
  let slope := (deriv f) 0
  slope = 0 := by sorry

-- The angle of intersection is the arctangent of the slope
theorem curve_intersects_y_axis_at_zero_angle : 
  Real.arctan ((deriv f) 0) = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_angle_zero_curve_intersects_y_axis_at_zero_angle_l221_22150


namespace NUMINAMATH_CALUDE_oreo_cheesecake_graham_crackers_l221_22111

theorem oreo_cheesecake_graham_crackers :
  ∀ (G : ℕ) (oreos : ℕ),
  oreos = 15 →
  (∃ (cheesecakes : ℕ),
    cheesecakes * 2 = G - 4 ∧
    cheesecakes * 3 ≤ oreos ∧
    ∀ (c : ℕ), c * 2 ≤ G - 4 ∧ c * 3 ≤ oreos → c ≤ cheesecakes) →
  G = 14 := by sorry

end NUMINAMATH_CALUDE_oreo_cheesecake_graham_crackers_l221_22111


namespace NUMINAMATH_CALUDE_sequence_properties_l221_22138

/-- Arithmetic sequence with first term 3 -/
def a (n : ℕ) : ℚ := 3 * n

/-- Sum of first n terms of arithmetic sequence -/
def S (n : ℕ) : ℚ := n * (3 + a n) / 2

/-- Geometric sequence with first term 1 -/
def b (n : ℕ) : ℚ := 3^(n - 1)

/-- Sum of first n terms of the sequence {1/S_n} -/
def T (n : ℕ) : ℚ := (2 * n) / (3 * (n + 1))

theorem sequence_properties :
  (∀ n, n > 0 → b n > 0) ∧  -- All terms of b_n are positive
  b 1 = 1 ∧                 -- b_1 = 1
  b 2 + S 2 = 12 ∧          -- b_2 + S_2 = 12
  a 3 = b 3 ∧               -- a_3 = b_3
  (∀ n, a n = 3 * n) ∧      -- General term of a_n
  (∀ n, b n = 3^(n - 1)) ∧  -- General term of b_n
  (∀ n, T n = (2 * n) / (3 * (n + 1))) -- Sum of first n terms of {1/S_n}
  := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l221_22138
