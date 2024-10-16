import Mathlib

namespace NUMINAMATH_CALUDE_charles_learning_time_l4178_417808

/-- The number of days it takes to learn one vowel, given the total days and number of vowels -/
def days_per_vowel (total_days : ℕ) (num_vowels : ℕ) : ℕ :=
  total_days / num_vowels

/-- Theorem stating that it takes 7 days to learn one vowel -/
theorem charles_learning_time :
  days_per_vowel 35 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_charles_learning_time_l4178_417808


namespace NUMINAMATH_CALUDE_carpet_ratio_l4178_417844

theorem carpet_ratio (house1 house2 house3 total : ℕ) 
  (h1 : house1 = 12)
  (h2 : house2 = 20)
  (h3 : house3 = 10)
  (h_total : total = 62)
  (h_sum : house1 + house2 + house3 + (total - (house1 + house2 + house3)) = total) :
  (total - (house1 + house2 + house3)) / house3 = 2 := by
sorry

end NUMINAMATH_CALUDE_carpet_ratio_l4178_417844


namespace NUMINAMATH_CALUDE_side_significant_digits_l4178_417890

-- Define the area of the square
def area : ℝ := 0.6400

-- Define the precision of the area measurement
def area_precision : ℝ := 0.0001

-- Define the function to calculate the number of significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits :
  let side := Real.sqrt area
  count_significant_digits side = 4 := by sorry

end NUMINAMATH_CALUDE_side_significant_digits_l4178_417890


namespace NUMINAMATH_CALUDE_surface_area_combined_shape_l4178_417838

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified shape -/
def surfaceAreaModifiedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Calculates the surface area of the combined shape -/
def surfaceAreaCombinedShape (original : CubeDimensions) (removed : CubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the combined shape is 38 cm² -/
theorem surface_area_combined_shape :
  let original := CubeDimensions.mk 2 2 2
  let removed := CubeDimensions.mk 1 1 1
  surfaceAreaCombinedShape original removed = 38 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_combined_shape_l4178_417838


namespace NUMINAMATH_CALUDE_hyperbola_orthogonal_asymptotes_l4178_417828

/-- A hyperbola is defined by its coefficients a, b, c, d, e, f in the equation ax^2 + 2bxy + cy^2 + dx + ey + f = 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Asymptotes of a hyperbola are orthogonal -/
def has_orthogonal_asymptotes (h : Hyperbola) : Prop :=
  h.a + h.c = 0

/-- The theorem stating that a hyperbola has orthogonal asymptotes if and only if a + c = 0 -/
theorem hyperbola_orthogonal_asymptotes (h : Hyperbola) :
  has_orthogonal_asymptotes h ↔ h.a + h.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_orthogonal_asymptotes_l4178_417828


namespace NUMINAMATH_CALUDE_expression_evaluation_l4178_417896

theorem expression_evaluation :
  let f (x : ℝ) := (((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4*x + 4) / x))
  f 1 = -1/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4178_417896


namespace NUMINAMATH_CALUDE_least_x_value_l4178_417847

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : Prime (x / (9 * p))) (h4 : Odd (x / (9 * p))) :
  x ≥ 90 ∧ ∃ (x₀ : ℕ), x₀ = 90 ∧ 
    Prime (x₀ / (9 * p)) ∧ Odd (x₀ / (9 * p)) :=
sorry

end NUMINAMATH_CALUDE_least_x_value_l4178_417847


namespace NUMINAMATH_CALUDE_starters_with_twin_l4178_417874

def total_players : ℕ := 16
def starters : ℕ := 6
def twins : ℕ := 2

theorem starters_with_twin (total_players starters twins : ℕ) :
  total_players = 16 →
  starters = 6 →
  twins = 2 →
  (Nat.choose total_players starters) - (Nat.choose (total_players - twins) starters) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_starters_with_twin_l4178_417874


namespace NUMINAMATH_CALUDE_inequality_preservation_l4178_417864

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 2 > y - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l4178_417864


namespace NUMINAMATH_CALUDE_inequality_range_l4178_417854

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) → 
  -2 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l4178_417854


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4178_417840

theorem fraction_inequality_solution_set (x : ℝ) :
  (x - 1) / (x - 3) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l4178_417840


namespace NUMINAMATH_CALUDE_decimal_38_to_binary_l4178_417876

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_38_to_binary :
  decimalToBinary 38 = [false, true, true, false, false, true] := by
  sorry

#eval decimalToBinary 38

end NUMINAMATH_CALUDE_decimal_38_to_binary_l4178_417876


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l4178_417861

/-- Calculates the profit percentage of a dishonest dealer who uses a reduced weight. -/
theorem dishonest_dealer_profit_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_positive : claimed_weight > 0)
  (actual_weight_positive : actual_weight > 0)
  (actual_weight_less_than_claimed : actual_weight < claimed_weight) :
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  ((1000 - 780) / 780) * 100 :=
by sorry

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l4178_417861


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l4178_417834

/-- Given that M(3,7) is the midpoint of AB and A(9,3), prove that the sum of B's coordinates is 8 -/
theorem sum_of_coordinates_B (A B M : ℝ × ℝ) : 
  A = (9, 3) → M = (3, 7) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l4178_417834


namespace NUMINAMATH_CALUDE_polynomial_division_l4178_417800

theorem polynomial_division (a : ℝ) (h : a ≠ 0) :
  (9 * a^6 - 12 * a^3) / (3 * a^3) = 3 * a^3 - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l4178_417800


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4178_417835

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4178_417835


namespace NUMINAMATH_CALUDE_stacys_farm_goats_l4178_417851

/-- Calculates the number of goats on Stacy's farm given the conditions --/
theorem stacys_farm_goats (chickens : ℕ) (piglets : ℕ) (sick_animals : ℕ) :
  chickens = 26 →
  piglets = 40 →
  sick_animals = 50 →
  (chickens + piglets + (34 : ℕ)) / 2 = sick_animals →
  34 = (2 * sick_animals) - chickens - piglets :=
by
  sorry

end NUMINAMATH_CALUDE_stacys_farm_goats_l4178_417851


namespace NUMINAMATH_CALUDE_point_on_y_axis_l4178_417846

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the y-axis
def on_y_axis (p : Point2D) : Prop := p.x = 0

-- Define our point P in terms of parameter a
def P (a : ℝ) : Point2D := ⟨2*a - 6, a + 1⟩

-- Theorem statement
theorem point_on_y_axis :
  ∃ a : ℝ, on_y_axis (P a) → P a = ⟨0, 4⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l4178_417846


namespace NUMINAMATH_CALUDE_least_integer_with_remainders_l4178_417877

theorem least_integer_with_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n →
    (m % 5 ≠ 4 ∨ m % 6 ≠ 5 ∨ m % 7 ≠ 6 ∨ m % 8 ≠ 7 ∨ m % 9 ≠ 8 ∨ m % 10 ≠ 9)) ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n = 2519 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_remainders_l4178_417877


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_l4178_417891

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

theorem min_voters_for_tall_giraffe :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_l4178_417891


namespace NUMINAMATH_CALUDE_batsman_highest_score_l4178_417870

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 63)
  (j : score_difference = 150)
  (k : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = 
      ((total_innings - 2 : ℕ) : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l4178_417870


namespace NUMINAMATH_CALUDE_intersection_polyhedron_volume_l4178_417889

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The polyhedron formed by the intersection of a regular tetrahedron with its image under symmetry relative to the midpoint of its height -/
def IntersectionPolyhedron (t : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The volume of the intersection polyhedron is (a^3 * √2) / 54 -/
theorem intersection_polyhedron_volume (t : RegularTetrahedron) :
    volume (IntersectionPolyhedron t) = (t.edge_length^3 * Real.sqrt 2) / 54 :=
  sorry

end NUMINAMATH_CALUDE_intersection_polyhedron_volume_l4178_417889


namespace NUMINAMATH_CALUDE_chairs_count_l4178_417862

-- Define the variables
variable (chair_price : ℚ) (table_price : ℚ) (num_chairs : ℕ)

-- Define the conditions
def condition1 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = num_chairs * table_price - 320

def condition2 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = (num_chairs - 5) * table_price

def condition3 (chair_price table_price : ℚ) : Prop :=
  3 * table_price = 5 * chair_price + 48

-- State the theorem
theorem chairs_count 
  (h1 : condition1 chair_price table_price num_chairs)
  (h2 : condition2 chair_price table_price num_chairs)
  (h3 : condition3 chair_price table_price) :
  num_chairs = 20 := by
  sorry

end NUMINAMATH_CALUDE_chairs_count_l4178_417862


namespace NUMINAMATH_CALUDE_diophantine_approximation_2005_l4178_417827

theorem diophantine_approximation_2005 (m n : ℕ+) : 
  |n * Real.sqrt 2005 - m| > (1 : ℝ) / (90 * n) := by sorry

end NUMINAMATH_CALUDE_diophantine_approximation_2005_l4178_417827


namespace NUMINAMATH_CALUDE_percentage_of_75_to_125_l4178_417879

theorem percentage_of_75_to_125 : (75 : ℝ) / 125 * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_75_to_125_l4178_417879


namespace NUMINAMATH_CALUDE_fifth_month_sale_l4178_417848

/-- Proves that the sale in the 5th month is 6029, given the conditions of the problem -/
theorem fifth_month_sale (
  average_sale : ℕ)
  (first_month_sale : ℕ)
  (second_month_sale : ℕ)
  (third_month_sale : ℕ)
  (fourth_month_sale : ℕ)
  (sixth_month_sale : ℕ)
  (h1 : average_sale = 5600)
  (h2 : first_month_sale = 5266)
  (h3 : second_month_sale = 5768)
  (h4 : third_month_sale = 5922)
  (h5 : fourth_month_sale = 5678)
  (h6 : sixth_month_sale = 4937) :
  first_month_sale + second_month_sale + third_month_sale + fourth_month_sale + 6029 + sixth_month_sale = 6 * average_sale :=
by sorry

#eval 5266 + 5768 + 5922 + 5678 + 6029 + 4937
#eval 6 * 5600

end NUMINAMATH_CALUDE_fifth_month_sale_l4178_417848


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l4178_417843

theorem mean_proportional_problem (n : ℝ) : (156 : ℝ) ^ 2 = n * 104 → n = 234 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l4178_417843


namespace NUMINAMATH_CALUDE_capital_calculation_l4178_417826

/-- Calculates the capital of a business partner who joined later --/
def calculate_capital (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) : ℕ :=
  let x_share := x_capital * 12
  let y_share := y_capital * 12
  let z_months := 12 - z_join_month
  let total_ratio := x_share + y_share
  ((z_profit * total_ratio) / (total_profit - z_profit)) / z_months

theorem capital_calculation (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) :
  x_capital = 20000 →
  y_capital = 25000 →
  z_profit = 14000 →
  total_profit = 50000 →
  z_join_month = 5 →
  calculate_capital x_capital y_capital z_profit total_profit z_join_month = 30000 := by
  sorry

#eval calculate_capital 20000 25000 14000 50000 5

end NUMINAMATH_CALUDE_capital_calculation_l4178_417826


namespace NUMINAMATH_CALUDE_regular_price_is_15_l4178_417895

-- Define the variables
def num_shirts : ℕ := 20
def discount_rate : ℚ := 0.2
def tax_rate : ℚ := 0.1
def total_paid : ℚ := 264

-- Define the theorem
theorem regular_price_is_15 :
  ∃ (regular_price : ℚ),
    regular_price * num_shirts * (1 - discount_rate) * (1 + tax_rate) = total_paid ∧
    regular_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_price_is_15_l4178_417895


namespace NUMINAMATH_CALUDE_antBGrainCalculation_l4178_417897

/-- Represents the work rate of an ant in units per hour -/
def WorkRate := ℚ

/-- Represents the amount of grain in units -/
def GrainAmount := ℚ

/-- Calculates the amount of grain transported by ant B given the work rates and total grain amount -/
def antBGrain (rateA rateB rateC totalGrain : ℚ) : ℚ :=
  let totalRate := rateA + rateB + rateC
  let ratioB := rateB / totalRate
  ratioB * totalGrain

theorem antBGrainCalculation (rateA rateB rateC totalGrain : ℚ) :
  rateA = 1/10 ∧ rateB = 1/8 ∧ rateC = 1/6 ∧ totalGrain = 24 →
  antBGrain rateA rateB rateC totalGrain = 42 := by
  sorry

end NUMINAMATH_CALUDE_antBGrainCalculation_l4178_417897


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4178_417850

theorem quadratic_inequality_solution (x : ℝ) :
  -x^2 - 2*x + 3 < 0 ↔ x < -3 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4178_417850


namespace NUMINAMATH_CALUDE_down_payment_proof_l4178_417810

-- Define the number of people
def num_people : ℕ := 3

-- Define the individual payment amount
def individual_payment : ℚ := 1166.67

-- Function to round to nearest dollar
def round_to_dollar (x : ℚ) : ℕ := 
  (x + 0.5).floor.toNat

-- Define the total down payment
def total_down_payment : ℕ := num_people * round_to_dollar individual_payment

-- Theorem statement
theorem down_payment_proof : total_down_payment = 3501 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_proof_l4178_417810


namespace NUMINAMATH_CALUDE_apartment_room_sizes_l4178_417894

/-- The apartment shared by Jenny, Martha, and Sam has three rooms with a total area of 800 square feet. Jenny's room is 100 square feet larger than Martha's, and Sam's room is 50 square feet smaller than Martha's. This theorem proves that Jenny's and Sam's rooms combined have an area of 550 square feet. -/
theorem apartment_room_sizes (total_area : ℝ) (martha_size : ℝ) 
  (h1 : total_area = 800)
  (h2 : martha_size + (martha_size + 100) + (martha_size - 50) = total_area) :
  (martha_size + 100) + (martha_size - 50) = 550 := by
  sorry

end NUMINAMATH_CALUDE_apartment_room_sizes_l4178_417894


namespace NUMINAMATH_CALUDE_kay_weight_training_time_l4178_417804

/-- Represents the weekly exercise schedule -/
structure ExerciseSchedule where
  total_time : ℕ
  aerobics_ratio : ℕ
  weight_training_ratio : ℕ

/-- Calculates the time spent on weight training given an exercise schedule -/
def weight_training_time (schedule : ExerciseSchedule) : ℕ :=
  (schedule.total_time * schedule.weight_training_ratio) / (schedule.aerobics_ratio + schedule.weight_training_ratio)

/-- Theorem: Given Kay's exercise schedule, she spends 100 minutes on weight training -/
theorem kay_weight_training_time :
  let kay_schedule : ExerciseSchedule := {
    total_time := 250,
    aerobics_ratio := 3,
    weight_training_ratio := 2
  }
  weight_training_time kay_schedule = 100 := by
  sorry

end NUMINAMATH_CALUDE_kay_weight_training_time_l4178_417804


namespace NUMINAMATH_CALUDE_sibling_age_sum_l4178_417832

/-- Given the ages and age differences of three siblings, prove the sum of the youngest and oldest siblings' ages. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 → 
  maggie + ralph = 19 := by
  sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l4178_417832


namespace NUMINAMATH_CALUDE_remove_five_blocks_count_l4178_417809

/-- Represents the number of exposed blocks after removing n blocks -/
def E (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * n + 1

/-- Represents the number of blocks in the k-th layer from the top -/
def blocks_in_layer (k : ℕ) : ℕ := 4^(k-1)

/-- The total number of ways to remove 5 blocks from the stack -/
def remove_five_blocks : ℕ := 
  (E 0) * (E 1) * (E 2) * (E 3) * (E 4) - (E 0) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2) * (blocks_in_layer 2)

theorem remove_five_blocks_count : remove_five_blocks = 3384 := by
  sorry

end NUMINAMATH_CALUDE_remove_five_blocks_count_l4178_417809


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4178_417801

theorem absolute_value_equation_solution (x : ℝ) : 
  |3*x - 2| + |3*x + 1| = 3 ↔ x = -2/3 ∨ (-1/3 < x ∧ x ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4178_417801


namespace NUMINAMATH_CALUDE_unique_triple_l4178_417893

theorem unique_triple (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 → 
  (bc + 1) % a = 0 → 
  (ac + 1) % b = 0 → 
  (ab + 1) % c = 0 → 
  a = 2 ∧ b = 3 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_l4178_417893


namespace NUMINAMATH_CALUDE_snake_body_length_l4178_417887

/-- Given a snake with a head that is one-tenth of its total length and a total length of 10 feet,
    the length of the rest of its body minus the head is 9 feet. -/
theorem snake_body_length (total_length : ℝ) (head_ratio : ℝ) : 
  total_length = 10 → head_ratio = 1/10 → total_length * (1 - head_ratio) = 9 := by
  sorry

end NUMINAMATH_CALUDE_snake_body_length_l4178_417887


namespace NUMINAMATH_CALUDE_square_remainder_l4178_417859

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l4178_417859


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l4178_417852

theorem subtraction_of_fractions : 
  (2 + 1/4) - 2/3 = 1 + 7/12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l4178_417852


namespace NUMINAMATH_CALUDE_arielle_age_l4178_417883

theorem arielle_age (elvie_age arielle_age : ℕ) : 
  elvie_age = 10 → 
  elvie_age + arielle_age + elvie_age * arielle_age = 131 → 
  arielle_age = 11 := by
sorry

end NUMINAMATH_CALUDE_arielle_age_l4178_417883


namespace NUMINAMATH_CALUDE_dot_product_equality_l4178_417872

/-- Given points in 2D space, prove that OA · OP₃ = OP₁ · OP₂ -/
theorem dot_product_equality (α β : ℝ) :
  let O : ℝ × ℝ := (0, 0)
  let P₁ : ℝ × ℝ := (Real.cos α, Real.sin α)
  let P₂ : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let P₃ : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
  let A : ℝ × ℝ := (1, 0)
  (A.1 - O.1) * (P₃.1 - O.1) + (A.2 - O.2) * (P₃.2 - O.2) =
  (P₁.1 - O.1) * (P₂.1 - O.1) + (P₁.2 - O.2) * (P₂.2 - O.2) :=
by
  sorry

#check dot_product_equality

end NUMINAMATH_CALUDE_dot_product_equality_l4178_417872


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l4178_417881

theorem max_consecutive_integers_sum (k : ℕ) : k ≤ 81 ↔ ∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l4178_417881


namespace NUMINAMATH_CALUDE_strawberry_pies_count_l4178_417892

/-- Given a total number of pies and a ratio for different types of pies,
    calculate the number of pies of a specific type. -/
theorem strawberry_pies_count
  (total_pies : ℕ)
  (apple_ratio blueberry_ratio cherry_ratio strawberry_ratio : ℕ)
  (h_total : total_pies = 48)
  (h_ratios : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4 ∧ strawberry_ratio = 1) :
  (strawberry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio + strawberry_ratio) = 4 :=
by sorry

end NUMINAMATH_CALUDE_strawberry_pies_count_l4178_417892


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l4178_417802

theorem smallest_angle_in_triangle (x y z : ℝ) (hx : x = 60) (hy : y = 70) 
  (hsum : x + y + z = 180) : min x (min y z) = 50 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l4178_417802


namespace NUMINAMATH_CALUDE_product_in_base9_l4178_417888

/-- Converts a base-9 number to its decimal (base-10) equivalent -/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to its base-9 equivalent -/
def decimalToBase9 (n : ℕ) : ℕ := sorry

theorem product_in_base9 :
  decimalToBase9 (base9ToDecimal 327 * base9ToDecimal 6) = 2406 := by sorry

end NUMINAMATH_CALUDE_product_in_base9_l4178_417888


namespace NUMINAMATH_CALUDE_expression_value_l4178_417880

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4178_417880


namespace NUMINAMATH_CALUDE_remaining_payment_l4178_417822

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 80) (h2 : deposit_percentage = 0.1) :
  let total_cost := deposit / deposit_percentage
  total_cost - deposit = 720 := by
sorry

end NUMINAMATH_CALUDE_remaining_payment_l4178_417822


namespace NUMINAMATH_CALUDE_complex_angle_pi_third_l4178_417885

theorem complex_angle_pi_third (z : ℂ) : 
  z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_angle_pi_third_l4178_417885


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l4178_417863

/-- The number of intersection points for k lines in a plane -/
def num_intersections (k : ℕ) : ℕ := sorry

/-- The maximum number of intersection points after adding one more line to k lines -/
def max_intersections_after_adding_line (k : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of intersection points after adding one more line
    to k lines is equal to the number of intersection points for k lines plus k -/
theorem max_intersections_theorem (k : ℕ) :
  max_intersections_after_adding_line k = num_intersections k + k := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l4178_417863


namespace NUMINAMATH_CALUDE_min_value_theorem_l4178_417823

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) : 
  2/a + 1/b ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4178_417823


namespace NUMINAMATH_CALUDE_jerrys_average_score_l4178_417842

theorem jerrys_average_score (current_total : ℝ) (desired_average : ℝ) (fourth_test_score : ℝ) :
  (current_total / 3 + 2 = desired_average) →
  (current_total + fourth_test_score) / 4 = desired_average →
  fourth_test_score = 98 →
  current_total / 3 = 90 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l4178_417842


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_min_perimeter_achieved_l4178_417871

theorem min_perimeter_rectangle (length width : ℝ) : 
  length > 0 → width > 0 → length * width = 64 → 
  2 * (length + width) ≥ 32 := by
  sorry

theorem min_perimeter_achieved (length width : ℝ) :
  length > 0 → width > 0 → length * width = 64 →
  2 * (length + width) = 32 ↔ length = 8 ∧ width = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_min_perimeter_achieved_l4178_417871


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_l4178_417824

/-- 
A cubic equation x^3 + ax^2 + bx + c = 0 has three real roots forming an arithmetic progression 
if and only if the following conditions are satisfied:
1) ab/3 - 2a^3/27 - c = 0
2) a^3/3 - b ≥ 0
-/
theorem cubic_arithmetic_progression (a b c : ℝ) : 
  (∃ x y z : ℝ, x < y ∧ y < z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) ∧
    y - x = z - y) ↔ 
  (a*b/3 - 2*a^3/27 - c = 0 ∧ a^3/3 - b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_l4178_417824


namespace NUMINAMATH_CALUDE_AB_product_l4178_417811

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 2; 0, -2]
def B_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -1/2; 0, 2]

theorem AB_product :
  let B := B_inv⁻¹
  A * B = !![1, 5/4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_AB_product_l4178_417811


namespace NUMINAMATH_CALUDE_test_score_problem_l4178_417839

/-- Prove that given a test with 30 questions, where each correct answer is worth 20 points
    and each incorrect answer deducts 5 points, if all questions are answered and the total
    score is 325, then the number of correct answers is 19. -/
theorem test_score_problem (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) 
    (total_score : ℕ) (h1 : total_questions = 30) (h2 : correct_points = 20) 
    (h3 : incorrect_points = 5) (h4 : total_score = 325) : 
    ∃ (correct_answers : ℕ), 
      correct_answers * correct_points + 
      (total_questions - correct_answers) * (correct_points - incorrect_points) = 
      total_score ∧ correct_answers = 19 := by
  sorry

end NUMINAMATH_CALUDE_test_score_problem_l4178_417839


namespace NUMINAMATH_CALUDE_power_division_l4178_417814

theorem power_division (x : ℕ) : 8^15 / 64^3 = 8^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l4178_417814


namespace NUMINAMATH_CALUDE_triangle_sum_equality_l4178_417856

theorem triangle_sum_equality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = a^2)
  (eq2 : y^2 + y*z + z^2 = b^2)
  (eq3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p * (p - a) * (p - b) * (p - c)) / 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_equality_l4178_417856


namespace NUMINAMATH_CALUDE_geometric_series_problem_l4178_417815

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 15
  let b₁ : ℝ := 3
  let a₂ : ℝ := 15
  let b₂ : ℝ := 3 + n
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let S₁ : ℝ := a₁ / (1 - r₁)
  let S₂ : ℝ := a₂ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l4178_417815


namespace NUMINAMATH_CALUDE_final_mixture_volume_l4178_417830

/-- Represents an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  concentration : ℝ

/-- The problem setup -/
def mixture_problem (mixture30 mixture50 mixtureFinal : AlcoholMixture) : Prop :=
  mixture30.concentration = 0.30 ∧
  mixture50.concentration = 0.50 ∧
  mixtureFinal.concentration = 0.45 ∧
  mixture30.volume = 2.5 ∧
  mixtureFinal.volume = mixture30.volume + mixture50.volume ∧
  mixture30.volume * mixture30.concentration + mixture50.volume * mixture50.concentration =
    mixtureFinal.volume * mixtureFinal.concentration

/-- The theorem statement -/
theorem final_mixture_volume
  (mixture30 mixture50 mixtureFinal : AlcoholMixture)
  (h : mixture_problem mixture30 mixture50 mixtureFinal) :
  mixtureFinal.volume = 10 :=
sorry

end NUMINAMATH_CALUDE_final_mixture_volume_l4178_417830


namespace NUMINAMATH_CALUDE_prob_a_not_less_than_b_expected_tests_scheme_b_l4178_417803

/-- Represents the two testing schemes -/
inductive TestScheme
| A
| B

/-- Represents the possible outcomes of a test -/
inductive TestResult
| Positive
| Negative

/-- The total number of swimmers -/
def totalSwimmers : ℕ := 5

/-- The number of swimmers who have taken stimulants -/
def stimulantUsers : ℕ := 1

/-- The number of swimmers tested in the first step of Scheme B -/
def schemeBFirstTest : ℕ := 3

/-- Function to calculate the probability that Scheme A requires no fewer tests than Scheme B -/
def probANotLessThanB : ℚ :=
  18/25

/-- Function to calculate the expected number of tests in Scheme B -/
def expectedTestsSchemeB : ℚ :=
  2.4

/-- Theorem stating the probability that Scheme A requires no fewer tests than Scheme B -/
theorem prob_a_not_less_than_b :
  probANotLessThanB = 18/25 := by sorry

/-- Theorem stating the expected number of tests in Scheme B -/
theorem expected_tests_scheme_b :
  expectedTestsSchemeB = 2.4 := by sorry

end NUMINAMATH_CALUDE_prob_a_not_less_than_b_expected_tests_scheme_b_l4178_417803


namespace NUMINAMATH_CALUDE_age_difference_l4178_417825

theorem age_difference (A B C : ℕ) (h : C = A - 13) : A + B - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4178_417825


namespace NUMINAMATH_CALUDE_words_with_vowels_l4178_417812

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels :
  total_words - words_without_vowels = 6752 := by sorry

end NUMINAMATH_CALUDE_words_with_vowels_l4178_417812


namespace NUMINAMATH_CALUDE_pump_fill_time_l4178_417899

theorem pump_fill_time (P : ℝ) (h1 : P > 0) (h2 : 14 > 0) :
  1 / P - 1 / 14 = 1 / (7 / 3) → P = 2 := by
  sorry

end NUMINAMATH_CALUDE_pump_fill_time_l4178_417899


namespace NUMINAMATH_CALUDE_overlap_length_l4178_417875

theorem overlap_length (total_length edge_to_edge_distance : ℝ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : ∃ x : ℝ, total_length = edge_to_edge_distance + 6 * x) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = edge_to_edge_distance + 6 * x := by
sorry

end NUMINAMATH_CALUDE_overlap_length_l4178_417875


namespace NUMINAMATH_CALUDE_triangle_two_solutions_range_l4178_417853

theorem triangle_two_solutions_range (a b : ℝ) (B : ℝ) (h1 : b = 2) (h2 : B = 45 * π / 180) :
  (∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧ 
   a * Real.sin B < b ∧ b < a) ↔ 
  (2 < a ∧ a < 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_range_l4178_417853


namespace NUMINAMATH_CALUDE_expression_evaluation_l4178_417817

theorem expression_evaluation : 
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4178_417817


namespace NUMINAMATH_CALUDE_product_of_one_plus_greater_than_eight_l4178_417807

theorem product_of_one_plus_greater_than_eight
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_prod : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_one_plus_greater_than_eight_l4178_417807


namespace NUMINAMATH_CALUDE_circle_symmetry_l4178_417886

/-- Given a circle and a line of symmetry, prove that another circle is symmetric to the given circle about the line. -/
theorem circle_symmetry (x y : ℝ) :
  let original_circle := (x - 1)^2 + (y - 2)^2 = 1
  let symmetry_line := x - y - 2 = 0
  let symmetric_circle := (x - 4)^2 + (y + 1)^2 = 1
  (∀ (x₀ y₀ : ℝ), original_circle → 
    ∃ (x₁ y₁ : ℝ), symmetric_circle ∧ 
    ((x₀ + x₁) / 2 - (y₀ + y₁) / 2 - 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l4178_417886


namespace NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l4178_417813

theorem ivy_collectors_edition_dolls 
  (dina_dolls : ℕ)
  (ivy_dolls : ℕ)
  (h1 : dina_dolls = 60)
  (h2 : dina_dolls = 2 * ivy_dolls)
  (h3 : ivy_dolls > 0)
  : (2 : ℚ) / 3 * ivy_dolls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivy_collectors_edition_dolls_l4178_417813


namespace NUMINAMATH_CALUDE_minimum_cost_is_2200_l4178_417869

/-- Represents the transportation problem for washing machines -/
structure TransportationProblem where
  totalWashingMachines : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeACapacity : ℕ
  typeBCapacity : ℕ
  typeACost : ℕ
  typeBCost : ℕ

/-- Calculates the minimum transportation cost for the given problem -/
def minimumTransportationCost (p : TransportationProblem) : ℕ :=
  sorry

/-- The main theorem stating that the minimum transportation cost is 2200 yuan -/
theorem minimum_cost_is_2200 :
  let p : TransportationProblem := {
    totalWashingMachines := 100,
    typeATrucks := 4,
    typeBTrucks := 8,
    typeACapacity := 20,
    typeBCapacity := 10,
    typeACost := 400,
    typeBCost := 300
  }
  minimumTransportationCost p = 2200 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_is_2200_l4178_417869


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4178_417816

theorem fraction_evaluation : (20 + 15) / (30 - 25) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4178_417816


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l4178_417829

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l4178_417829


namespace NUMINAMATH_CALUDE_eve_distance_difference_l4178_417882

def running_intervals : List ℝ := [0.75, 0.85, 0.95]
def walking_intervals : List ℝ := [0.50, 0.65, 0.75, 0.80]

theorem eve_distance_difference :
  (running_intervals.sum - walking_intervals.sum) = -0.15 := by
  sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l4178_417882


namespace NUMINAMATH_CALUDE_train_distance_theorem_l4178_417841

-- Define the speeds of the trains
def speed_train1 : ℝ := 20
def speed_train2 : ℝ := 25

-- Define the difference in distance traveled
def distance_difference : ℝ := 50

-- Define the theorem
theorem train_distance_theorem :
  ∀ (t : ℝ), -- t represents the time taken for trains to meet
  t > 0 → -- time is positive
  speed_train1 * t + speed_train2 * t = -- total distance is sum of distances traveled by both trains
  speed_train1 * t + (speed_train1 * t + distance_difference) → -- one train travels 50 km more
  speed_train1 * t + (speed_train1 * t + distance_difference) = 450 -- total distance is 450 km
  := by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l4178_417841


namespace NUMINAMATH_CALUDE_linear_function_properties_l4178_417806

def LinearFunction (m c : ℝ) : ℝ → ℝ := fun x ↦ m * x + c

theorem linear_function_properties (f : ℝ → ℝ) (m c : ℝ) 
  (h1 : ∃ k : ℝ, ∀ x, f x + 2 = 3 * k * x)
  (h2 : f 1 = 4)
  (h3 : f = LinearFunction m c) :
  (f = LinearFunction 6 (-2)) ∧ 
  (∀ a b : ℝ, f (-1) = a ∧ f 2 = b → a < b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l4178_417806


namespace NUMINAMATH_CALUDE_vertical_equality_puzzle_l4178_417836

theorem vertical_equality_puzzle :
  ∃ (a b c d e f g h i j : ℕ),
    a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 5 ∧ e = 4 ∧ f = 0 ∧ g = 6 ∧ h = 7 ∧ i = 2 ∧ j = 3 ∧
    (100 * a + 10 * b + c) - (10 * d + c) = (100 * a + 10 * e + f) ∧
    g * h = 10 * e + i ∧
    (10 * j + j) + (10 * g + d) = 10 * b + c ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_equality_puzzle_l4178_417836


namespace NUMINAMATH_CALUDE_extra_large_posters_count_l4178_417849

def total_posters : ℕ := 200

def small_posters : ℕ := total_posters / 4
def medium_posters : ℕ := total_posters / 3
def large_posters : ℕ := total_posters / 5

def extra_large_posters : ℕ := total_posters - (small_posters + medium_posters + large_posters)

theorem extra_large_posters_count : extra_large_posters = 44 := by
  sorry

end NUMINAMATH_CALUDE_extra_large_posters_count_l4178_417849


namespace NUMINAMATH_CALUDE_carol_extra_chore_earnings_l4178_417805

/-- Proves that given the conditions, Carol earns $1.50 per extra chore -/
theorem carol_extra_chore_earnings
  (weekly_allowance : ℚ)
  (num_weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : weekly_allowance = 20)
  (h2 : num_weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  (total_amount - weekly_allowance * num_weeks) / (avg_extra_chores * num_weeks) = 3/2 :=
sorry

end NUMINAMATH_CALUDE_carol_extra_chore_earnings_l4178_417805


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l4178_417865

-- Define the sets
def A : Set ℝ := {y | ∃ x, y = x^2}
def B : Set ℝ := {x | x > 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l4178_417865


namespace NUMINAMATH_CALUDE_manny_marbles_l4178_417866

theorem manny_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) (kept_packs : ℕ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  kept_packs = 25 →
  neil_fraction = 1/8 →
  let total_packs := total_marbles / marbles_per_pack
  let given_packs := total_packs - kept_packs
  let neil_packs := neil_fraction * total_packs
  let manny_packs := given_packs - neil_packs
  manny_packs / total_packs = 1/4 := by sorry

end NUMINAMATH_CALUDE_manny_marbles_l4178_417866


namespace NUMINAMATH_CALUDE_total_sticks_is_129_l4178_417818

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

/-- Theorem stating that the total number of sticks needed is 129 -/
theorem total_sticks_is_129 : total_sticks = 129 := by
  sorry

#eval total_sticks

end NUMINAMATH_CALUDE_total_sticks_is_129_l4178_417818


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l4178_417884

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l4178_417884


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l4178_417831

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  h_n : n ≥ 2
  h_distinct : ∀ i j, i ≠ j → x i ≠ x j
  h_line : ∀ i, y i = (1/3) * x i - 5

/-- The sample correlation coefficient of a set of data points -/
def sampleCorrelationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the sample correlation coefficient is 1 
    for data points satisfying the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) :
  sampleCorrelationCoefficient data = 1 :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l4178_417831


namespace NUMINAMATH_CALUDE_chocolate_percentage_l4178_417898

theorem chocolate_percentage (total : ℕ) (remaining : ℕ) : 
  total = 80 →
  remaining = 28 →
  (total / 2 : ℚ) * (1 - 1/2) + (total / 2) * (1 - 80/100) = remaining →
  80/100 = 1 - (remaining / (total / 2)) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_percentage_l4178_417898


namespace NUMINAMATH_CALUDE_max_distance_sin_cosin_l4178_417873

/-- The maximum distance between sin x and sin(π/2 - x) for any real x is √2 -/
theorem max_distance_sin_cosin (x : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), |Real.sin x - Real.sin (π/2 - x)| ≤ m ∧ 
  ∃ (y : ℝ), |Real.sin y - Real.sin (π/2 - y)| = m ∧ 
  m = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_distance_sin_cosin_l4178_417873


namespace NUMINAMATH_CALUDE_bears_in_shipment_bears_shipment_proof_l4178_417860

/-- The number of bears in a toy store shipment -/
theorem bears_in_shipment (initial_stock : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  shelves * bears_per_shelf - initial_stock

/-- Proof that the number of bears in the shipment is 7 -/
theorem bears_shipment_proof :
  bears_in_shipment 5 2 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_shipment_bears_shipment_proof_l4178_417860


namespace NUMINAMATH_CALUDE_melanie_dimes_l4178_417855

theorem melanie_dimes (initial_dimes mother_dimes final_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : mother_dimes = 4)
  (h3 : final_dimes = 19) :
  final_dimes - (initial_dimes + mother_dimes) = 8 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_l4178_417855


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4178_417820

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l4178_417820


namespace NUMINAMATH_CALUDE_base9_addition_l4178_417857

/-- Convert a base-9 number to its decimal representation -/
def base9ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Convert a decimal number to its base-9 representation -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
    aux n []

theorem base9_addition : 
  decimalToBase9 (base9ToDecimal [6, 5, 2] + base9ToDecimal [2, 4, 6] + base9ToDecimal [3, 7]) = 
  [2, 8, 9] :=
sorry

end NUMINAMATH_CALUDE_base9_addition_l4178_417857


namespace NUMINAMATH_CALUDE_translate_line_example_l4178_417868

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translate_line (l : Line) (y_shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + y_shift }

/-- The theorem stating that translating y = 3x - 3 upwards by 5 units results in y = 3x + 2 -/
theorem translate_line_example :
  let original_line : Line := { slope := 3, intercept := -3 }
  let translated_line := translate_line original_line 5
  translated_line = { slope := 3, intercept := 2 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_example_l4178_417868


namespace NUMINAMATH_CALUDE_two_members_absent_l4178_417858

/-- Represents a trivia team with its properties and scoring. -/
structure TriviaTeam where
  totalMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Calculates the number of members who didn't show up for a trivia game. -/
def membersAbsent (team : TriviaTeam) : ℕ :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up. -/
theorem two_members_absent (team : TriviaTeam)
  (h1 : team.totalMembers = 5)
  (h2 : team.pointsPerMember = 6)
  (h3 : team.totalPoints = 18) :
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5, pointsPerMember := 6, totalPoints := 18 }

end NUMINAMATH_CALUDE_two_members_absent_l4178_417858


namespace NUMINAMATH_CALUDE_tv_price_proof_l4178_417845

def is_divisible_by (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem tv_price_proof (a b : ℕ) (h1 : a < 10) (h2 : b < 10) :
  let total_price := a * 10000 + 6000 + 700 + 90 + b
  is_divisible_by total_price 72 →
  (total_price / 72 : ℚ) = 511 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_proof_l4178_417845


namespace NUMINAMATH_CALUDE_square_area_increase_l4178_417837

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l4178_417837


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l4178_417833

theorem parallel_line_through_point (x y : ℝ) : 
  let P : ℝ × ℝ := (0, 2)
  let L₁ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y = 0}
  let L₂ : Set (ℝ × ℝ) := {(x, y) | 2 * x - y + 2 = 0}
  (P ∈ L₂) ∧ (∃ k : ℝ, k ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ L₁ ↔ (k * x, k * y) ∈ L₂) :=
by
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l4178_417833


namespace NUMINAMATH_CALUDE_defective_products_m1_l4178_417821

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective m3_defective : ℝ) (non_defective_total : ℝ) :
  m1_production = 25 ∧ 
  m2_production = 35 ∧ 
  m3_production = 40 ∧ 
  m2_defective = 4 ∧ 
  m3_defective = 5 ∧ 
  non_defective_total = 96.1 →
  (100 - non_defective_total - (m2_production * m2_defective / 100 + m3_production * m3_defective / 100)) / m1_production * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_defective_products_m1_l4178_417821


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l4178_417878

theorem consecutive_even_integers_sum (a : ℤ) : 
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a % 2 = 0 ∧ 
    a + c = 146) →
  a + (a + 2) + (a + 4) + (a + 6) = 296 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l4178_417878


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l4178_417867

theorem arithmetic_sequence_sum_remainder (n : ℕ) (a d : ℤ) (h : n = 2013) (h1 : a = 105) (h2 : d = 35) :
  (n * (2 * a + (n - 1) * d) / 2) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l4178_417867


namespace NUMINAMATH_CALUDE_range_of_a_l4178_417819

/-- Proposition p -/
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

/-- The set of x satisfying proposition p -/
def A : Set ℝ := {x | p x}

/-- The set of x satisfying proposition q -/
def B (a : ℝ) : Set ℝ := {x | q x a}

/-- The condition that ¬p is a necessary but not sufficient condition for ¬q -/
def condition (a : ℝ) : Prop := A ⊂ B a ∧ A ≠ B a

/-- The theorem stating the range of a -/
theorem range_of_a : ∀ a : ℝ, condition a ↔ 0 ≤ a ∧ a ≤ 1/2 ∧ a ≠ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4178_417819
