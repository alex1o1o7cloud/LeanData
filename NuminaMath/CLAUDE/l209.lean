import Mathlib

namespace NUMINAMATH_CALUDE_gcd_one_powers_of_two_l209_20963

def sequence_a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => sequence_a n + n * (sequence_a n - 1)

theorem gcd_one_powers_of_two (m : ℕ) :
  (∀ n, Nat.gcd m (sequence_a n) = 1) ↔ ∃ t : ℕ, m = 2^t :=
sorry

end NUMINAMATH_CALUDE_gcd_one_powers_of_two_l209_20963


namespace NUMINAMATH_CALUDE_initial_column_size_l209_20942

theorem initial_column_size (total_people : ℕ) (initial_columns : ℕ) (people_per_column : ℕ) : 
  total_people = initial_columns * people_per_column →
  total_people = 40 * 12 →
  initial_columns = 16 →
  people_per_column = 30 := by
sorry

end NUMINAMATH_CALUDE_initial_column_size_l209_20942


namespace NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_l209_20967

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem stating that the molecular weight of Dinitrogen trioxide is 76.02 g/mol -/
theorem dinitrogen_trioxide_weight : molecular_weight_N2O3 = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_l209_20967


namespace NUMINAMATH_CALUDE_initial_worth_is_30_l209_20932

/-- Represents the value of a single gold coin -/
def coin_value : ℕ := 6

/-- Represents the number of coins Roman sold -/
def sold_coins : ℕ := 3

/-- Represents the number of coins Roman has left after the sale -/
def remaining_coins : ℕ := 2

/-- Represents the amount of money Roman has after the sale -/
def money_after_sale : ℕ := 12

/-- Theorem stating that the initial total worth of Roman's gold coins was $30 -/
theorem initial_worth_is_30 :
  (sold_coins + remaining_coins) * coin_value = 30 :=
sorry

end NUMINAMATH_CALUDE_initial_worth_is_30_l209_20932


namespace NUMINAMATH_CALUDE_divisors_of_500_l209_20994

theorem divisors_of_500 : Finset.card (Nat.divisors 500) = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_500_l209_20994


namespace NUMINAMATH_CALUDE_pumpkin_count_l209_20983

/-- The number of pumpkins grown by Sandy -/
def sandy_pumpkins : ℕ := 51

/-- The number of pumpkins grown by Mike -/
def mike_pumpkins : ℕ := 23

/-- The total number of pumpkins grown by Sandy and Mike -/
def total_pumpkins : ℕ := sandy_pumpkins + mike_pumpkins

theorem pumpkin_count : total_pumpkins = 74 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_count_l209_20983


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l209_20989

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l209_20989


namespace NUMINAMATH_CALUDE_recreation_percentage_is_twenty_percent_l209_20912

/-- Calculates the percentage of earnings allocated for recreation and relaxation -/
def recreation_percentage (earnings_per_customer : ℚ) (fixed_expenses : ℚ) 
  (num_customers : ℕ) (savings : ℚ) : ℚ :=
  let total_earnings := earnings_per_customer * num_customers
  let total_expenses := fixed_expenses + savings
  let recreation_money := total_earnings - total_expenses
  (recreation_money / total_earnings) * 100

/-- Proves that the percentage of earnings allocated for recreation and relaxation is 20% -/
theorem recreation_percentage_is_twenty_percent :
  recreation_percentage 18 280 80 872 = 20 := by
  sorry

end NUMINAMATH_CALUDE_recreation_percentage_is_twenty_percent_l209_20912


namespace NUMINAMATH_CALUDE_polynomial_simplification_and_division_l209_20962

theorem polynomial_simplification_and_division (x : ℝ) (h : x ≠ -1) :
  ((3 * x^3 + 4 * x^2 - 5 * x + 2) - (2 * x^3 - x^2 + 6 * x - 8)) / (x + 1) =
  x^2 + 4 * x - 15 + 25 / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_and_division_l209_20962


namespace NUMINAMATH_CALUDE_total_sides_is_118_l209_20924

/-- The number of sides for each shape --/
def sides_of_shape (shape : String) : ℕ :=
  match shape with
  | "triangle" => 3
  | "square" => 4
  | "pentagon" => 5
  | "hexagon" => 6
  | "heptagon" => 7
  | "octagon" => 8
  | "nonagon" => 9
  | "hendecagon" => 11
  | "circle" => 0
  | _ => 0

/-- The count of each shape in the top layer --/
def top_layer : List (String × ℕ) :=
  [("triangle", 6), ("nonagon", 1), ("heptagon", 2)]

/-- The count of each shape in the middle layer --/
def middle_layer : List (String × ℕ) :=
  [("square", 4), ("hexagon", 2), ("hendecagon", 1)]

/-- The count of each shape in the bottom layer --/
def bottom_layer : List (String × ℕ) :=
  [("octagon", 3), ("circle", 5), ("pentagon", 1), ("nonagon", 1)]

/-- Calculate the total number of sides for a given layer --/
def total_sides_in_layer (layer : List (String × ℕ)) : ℕ :=
  layer.foldl (fun acc (shape, count) => acc + count * sides_of_shape shape) 0

/-- The main theorem stating that the total number of sides is 118 --/
theorem total_sides_is_118 :
  total_sides_in_layer top_layer +
  total_sides_in_layer middle_layer +
  total_sides_in_layer bottom_layer = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_is_118_l209_20924


namespace NUMINAMATH_CALUDE_ginger_water_bottle_capacity_l209_20976

/-- Proves that Ginger's water bottle holds 2 cups given the problem conditions -/
theorem ginger_water_bottle_capacity 
  (hours_worked : ℕ) 
  (bottles_for_plants : ℕ) 
  (total_cups_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottles_for_plants = 5)
  (h3 : total_cups_used = 26) :
  (total_cups_used : ℚ) / (hours_worked + bottles_for_plants : ℚ) = 2 := by
  sorry

#check ginger_water_bottle_capacity

end NUMINAMATH_CALUDE_ginger_water_bottle_capacity_l209_20976


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l209_20955

theorem absolute_value_inequality (a : ℝ) : |a| ≠ -|-a| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l209_20955


namespace NUMINAMATH_CALUDE_sum_difference_arithmetic_sequences_l209_20903

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (λ i => a₁ + i * d)

def sum_difference (seq1 seq2 : List ℕ) : ℕ :=
  (seq1.zip seq2).map (λ (a, b) => a - b) |>.sum

theorem sum_difference_arithmetic_sequences : 
  let seq1 := arithmetic_sequence 2101 1 123
  let seq2 := arithmetic_sequence 401 1 123
  sum_difference seq1 seq2 = 209100 := by
sorry

#eval sum_difference (arithmetic_sequence 2101 1 123) (arithmetic_sequence 401 1 123)

end NUMINAMATH_CALUDE_sum_difference_arithmetic_sequences_l209_20903


namespace NUMINAMATH_CALUDE_equation_holds_iff_sum_ten_l209_20934

theorem equation_holds_iff_sum_ten (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_sum_ten_l209_20934


namespace NUMINAMATH_CALUDE_clock_setting_time_l209_20971

/-- Represents a 24-hour clock time -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a clock time, wrapping around if necessary -/
def addMinutes (t : ClockTime) (m : ℤ) : ClockTime :=
  sorry

/-- Subtracts minutes from a clock time, wrapping around if necessary -/
def subtractMinutes (t : ClockTime) (m : ℕ) : ClockTime :=
  sorry

theorem clock_setting_time 
  (initial_time : ClockTime)
  (elapsed_hours : ℕ)
  (gain_rate : ℕ)
  (loss_rate : ℕ)
  (h : elapsed_hours = 20)
  (hgain : gain_rate = 1)
  (hloss : loss_rate = 2)
  (hfinal_diff : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 addMinutes (subtractMinutes initial_time (elapsed_hours * loss_rate)) 60)
  (hfinal_time : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 { hours := 12, minutes := 0, valid := sorry }) :
  initial_time = { hours := 15, minutes := 40, valid := sorry } :=
sorry

end NUMINAMATH_CALUDE_clock_setting_time_l209_20971


namespace NUMINAMATH_CALUDE_harry_pet_feeding_cost_l209_20986

/-- Represents the annual cost of feeding Harry's pets -/
def annual_feeding_cost (num_geckos num_iguanas num_snakes : ℕ) 
  (gecko_meals_per_month iguana_meals_per_month : ℕ) 
  (snake_meals_per_year : ℕ)
  (gecko_meal_cost iguana_meal_cost snake_meal_cost : ℕ) : ℕ :=
  (num_geckos * gecko_meals_per_month * 12 * gecko_meal_cost) +
  (num_iguanas * iguana_meals_per_month * 12 * iguana_meal_cost) +
  (num_snakes * snake_meals_per_year * snake_meal_cost)

/-- Theorem stating the annual cost of feeding Harry's pets -/
theorem harry_pet_feeding_cost :
  annual_feeding_cost 3 2 4 2 3 6 8 12 20 = 1920 := by
  sorry

#eval annual_feeding_cost 3 2 4 2 3 6 8 12 20

end NUMINAMATH_CALUDE_harry_pet_feeding_cost_l209_20986


namespace NUMINAMATH_CALUDE_min_value_expression_l209_20916

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z) ≥ (5.5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l209_20916


namespace NUMINAMATH_CALUDE_circular_arrangement_pairs_l209_20919

/-- Represents the number of adjacent pairs of children of the same gender -/
def adjacentPairs (total : Nat) (groups : Nat) : Nat :=
  total - groups

/-- The problem statement -/
theorem circular_arrangement_pairs (boys girls groups : Nat) 
  (h1 : boys = 15)
  (h2 : girls = 20)
  (h3 : adjacentPairs boys groups = (2 : Nat) / (3 : Nat) * adjacentPairs girls groups) :
  boys + girls - (adjacentPairs boys groups + adjacentPairs girls groups) = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_pairs_l209_20919


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l209_20975

theorem trigonometric_equation_solution (x : ℝ) :
  (4 * Real.sin x ^ 4 + Real.cos (4 * x) = 1 + 12 * Real.cos x ^ 4) ↔
  (∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l209_20975


namespace NUMINAMATH_CALUDE_complex_magnitude_l209_20956

theorem complex_magnitude (z : ℂ) (h : (3 + Complex.I) / z = 1 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l209_20956


namespace NUMINAMATH_CALUDE_car_price_increase_l209_20944

/-- Proves that given a discount and profit on the original price, 
    we can calculate the percentage increase on the discounted price. -/
theorem car_price_increase 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (h1 : discount_rate = 0.40) 
  (h2 : profit_rate = 0.08000000000000007) : 
  let discounted_price := original_price * (1 - discount_rate)
  let selling_price := original_price * (1 + profit_rate)
  let increase_rate := (selling_price - discounted_price) / discounted_price
  increase_rate = 0.8000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_car_price_increase_l209_20944


namespace NUMINAMATH_CALUDE_base9_multiplication_l209_20952

/-- Converts a base 9 number represented as a list of digits to its decimal equivalent -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 9 * acc + d) 0

/-- Converts a decimal number to its base 9 representation as a list of digits -/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- The main theorem statement -/
theorem base9_multiplication :
  let a := base9ToDecimal [3, 2, 7]
  let b := base9ToDecimal [1, 2]
  decimalToBase9 (a * b) = [4, 0, 4, 5] := by
  sorry


end NUMINAMATH_CALUDE_base9_multiplication_l209_20952


namespace NUMINAMATH_CALUDE_problem_solution_l209_20913

theorem problem_solution (x : ℝ) : (0.75 * x = (1/3) * x + 110) → x = 264 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l209_20913


namespace NUMINAMATH_CALUDE_megan_markers_proof_l209_20939

def final_markers (initial : ℕ) (robert_factor : ℕ) (elizabeth_taken : ℕ) : ℕ :=
  initial + robert_factor * initial - elizabeth_taken

theorem megan_markers_proof :
  final_markers 2475 3 1650 = 8250 := by
  sorry

end NUMINAMATH_CALUDE_megan_markers_proof_l209_20939


namespace NUMINAMATH_CALUDE_binomial_sum_l209_20905

theorem binomial_sum : (Nat.choose 10 3) + (Nat.choose 10 4) = 330 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l209_20905


namespace NUMINAMATH_CALUDE_smallest_geometric_distinct_digits_l209_20959

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem smallest_geometric_distinct_digits : 
  (∀ n : ℕ, is_three_digit n → 
    digits_are_distinct n → 
    is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) → 
    124 ≤ n) ∧ 
  is_three_digit 124 ∧ 
  digits_are_distinct 124 ∧ 
  is_geometric_sequence (124 / 100) ((124 / 10) % 10) (124 % 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_geometric_distinct_digits_l209_20959


namespace NUMINAMATH_CALUDE_combustion_reaction_l209_20972

/-- Represents the balanced chemical equation for the combustion of methane with chlorine and oxygen -/
structure BalancedEquation where
  ch4 : ℕ
  cl2 : ℕ
  o2 : ℕ
  co2 : ℕ
  hcl : ℕ
  h2o : ℕ
  balanced : ch4 = 1 ∧ cl2 = 4 ∧ o2 = 4 ∧ co2 = 1 ∧ hcl = 4 ∧ h2o = 2

/-- Represents the given quantities and products in the reaction -/
structure ReactionQuantities where
  ch4_given : ℕ
  cl2_given : ℕ
  co2_produced : ℕ
  hcl_produced : ℕ

/-- Theorem stating the required amount of O2 and produced amount of H2O -/
theorem combustion_reaction 
  (eq : BalancedEquation) 
  (quant : ReactionQuantities) 
  (h_ch4 : quant.ch4_given = 24) 
  (h_cl2 : quant.cl2_given = 48) 
  (h_co2 : quant.co2_produced = 24) 
  (h_hcl : quant.hcl_produced = 48) :
  ∃ (o2_required h2o_produced : ℕ), 
    o2_required = 96 ∧ 
    h2o_produced = 48 :=
  sorry

end NUMINAMATH_CALUDE_combustion_reaction_l209_20972


namespace NUMINAMATH_CALUDE_scientific_notation_of_86000_l209_20908

theorem scientific_notation_of_86000 (average_price : ℝ) : 
  average_price = 86000 → average_price = 8.6 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_86000_l209_20908


namespace NUMINAMATH_CALUDE_train_speed_l209_20917

/-- Proves that the speed of a train is 72 km/hr, given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 9) :
  (length / time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l209_20917


namespace NUMINAMATH_CALUDE_tangent_inclination_range_l209_20950

open Real

theorem tangent_inclination_range (x : ℝ) : 
  let y := sin x
  let slope := cos x
  let θ := arctan slope
  0 ≤ θ ∧ θ < π ∧ (θ ≤ π/4 ∨ 3*π/4 ≤ θ) := by sorry

end NUMINAMATH_CALUDE_tangent_inclination_range_l209_20950


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l209_20968

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 72) 
  (h_gcd : Nat.gcd x y = 8) : 
  x * y = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l209_20968


namespace NUMINAMATH_CALUDE_expression_simplification_l209_20943

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.sqrt 3 + 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (2*a - 4)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l209_20943


namespace NUMINAMATH_CALUDE_parabola_equation_and_min_ratio_l209_20925

/-- Represents a parabola with focus F and parameter p -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  h : p > 0

/-- A point on the parabola -/
def PointOnParabola (para : Parabola) (P : ℝ × ℝ) : Prop :=
  P.2^2 = 2 * para.p * P.1

theorem parabola_equation_and_min_ratio 
  (para : Parabola) 
  (P : ℝ × ℝ) 
  (h_P_on_parabola : PointOnParabola para P)
  (h_P_ordinate : P.2 = 4)
  (h_PF_distance : Real.sqrt ((P.1 - para.F.1)^2 + (P.2 - para.F.2)^2) = 4) :
  -- 1. The equation of the parabola is y^2 = 8x
  (∀ (x y : ℝ), PointOnParabola para (x, y) ↔ y^2 = 8*x) ∧
  -- 2. Minimum value of |MF| / |AB| is 1/2
  (∀ (A B : ℝ × ℝ) (h_A_on_parabola : PointOnParabola para A) 
                    (h_B_on_parabola : PointOnParabola para B)
                    (h_A_ne_B : A ≠ B)
                    (h_A_ne_P : A ≠ P)
                    (h_B_ne_P : B ≠ P),
   ∃ (M : ℝ × ℝ),
     -- Angle bisector of ∠APB is perpendicular to x-axis
     (∃ (k : ℝ), (A.2 - P.2) / (A.1 - P.1) = k ∧ (B.2 - P.2) / (B.1 - P.1) = -1/k) →
     -- M is on x-axis and perpendicular bisector of AB
     (M.2 = 0 ∧ (M.1 - (A.1 + B.1)/2) * ((B.2 - A.2)/(B.1 - A.1)) + (M.2 - (A.2 + B.2)/2) = 0) →
     -- Minimum value of |MF| / |AB| is 1/2
     (Real.sqrt ((M.1 - para.F.1)^2 + (M.2 - para.F.2)^2) / 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_min_ratio_l209_20925


namespace NUMINAMATH_CALUDE_alex_phone_bill_l209_20904

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (included_texts : ℕ) (text_cost : ℚ) 
                         (included_hours : ℕ) (minute_cost : ℚ)
                         (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let extra_texts := max (texts_sent - included_texts) 0
  let extra_minutes := max ((hours_talked - included_hours) * 60) 0
  base_cost + (extra_texts : ℚ) * text_cost + (extra_minutes : ℚ) * minute_cost

theorem alex_phone_bill :
  calculate_phone_bill 25 20 0.1 20 (15 / 100) 150 25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_alex_phone_bill_l209_20904


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l209_20965

theorem smallest_angle_in_triangle (x y z : ℝ) : 
  x + y + z = 180 →  -- Sum of angles in a triangle is 180°
  x + y = 45 →       -- Sum of two angles is 45°
  y = x - 5 →        -- One angle is 5° less than the other
  x > 0 ∧ y > 0 ∧ z > 0 →  -- All angles are positive
  min x (min y z) = 20 :=  -- The smallest angle is 20°
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l209_20965


namespace NUMINAMATH_CALUDE_trigonometric_identity_l209_20940

theorem trigonometric_identity (x y : ℝ) :
  3 * Real.cos (x + y) * Real.sin x + Real.sin (x + y) * Real.cos x =
  3 * Real.cos x * Real.cos y * Real.sin x - 3 * Real.sin x * Real.sin y * Real.sin x +
  Real.sin x * Real.cos y * Real.cos x + Real.cos x * Real.sin y * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l209_20940


namespace NUMINAMATH_CALUDE_three_dogs_walking_time_l209_20979

def base_charge : ℕ := 20
def per_minute_charge : ℕ := 1
def total_earnings : ℕ := 171
def one_dog_time : ℕ := 10
def two_dogs_time : ℕ := 7

def calculate_charge (dogs : ℕ) (minutes : ℕ) : ℕ :=
  dogs * (base_charge + per_minute_charge * minutes)

theorem three_dogs_walking_time :
  ∃ (x : ℕ), 
    calculate_charge 1 one_dog_time + 
    calculate_charge 2 two_dogs_time + 
    calculate_charge 3 x = total_earnings ∧ 
    x = 9 := by sorry

end NUMINAMATH_CALUDE_three_dogs_walking_time_l209_20979


namespace NUMINAMATH_CALUDE_place_mat_side_length_l209_20901

theorem place_mat_side_length (r : ℝ) (n : ℕ) (x : ℝ) : 
  r = 5 →
  n = 8 →
  x = 2 * r * Real.sin (π / (2 * n)) →
  x = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_place_mat_side_length_l209_20901


namespace NUMINAMATH_CALUDE_parabola_b_value_l209_20982

/-- A parabola passing through two points -/
structure Parabola where
  a : ℝ
  b : ℝ
  passes_through_1_2 : 2 = 1^2 + a * 1 + b
  passes_through_3_2 : 2 = 3^2 + a * 3 + b

/-- The value of b for the parabola passing through (1,2) and (3,2) is 5 -/
theorem parabola_b_value (p : Parabola) : p.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_b_value_l209_20982


namespace NUMINAMATH_CALUDE_total_green_peaches_is_fourteen_l209_20974

/-- The number of baskets containing peaches. -/
def num_baskets : ℕ := 7

/-- The number of green peaches in each basket. -/
def green_peaches_per_basket : ℕ := 2

/-- The total number of green peaches in all baskets. -/
def total_green_peaches : ℕ := num_baskets * green_peaches_per_basket

theorem total_green_peaches_is_fourteen : total_green_peaches = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_green_peaches_is_fourteen_l209_20974


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l209_20970

/-- Definition of a quadratic equation in x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x - 6

/-- Theorem: The given equation is a quadratic equation in x -/
theorem equation_is_quadratic : is_quadratic_in_x f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l209_20970


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l209_20931

/-- The area of a parallelogram generated by two vectors -/
def parallelogram_area (a b : ℝ × ℝ) : ℝ := sorry

theorem parallelogram_area_calculation 
  (a b : ℝ × ℝ) 
  (h1 : parallelogram_area a b = 20)
  (u : ℝ × ℝ := (1/2 : ℝ) • a + (5/2 : ℝ) • b)
  (v : ℝ × ℝ := 3 • a - 2 • b) :
  parallelogram_area u v = 130 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l209_20931


namespace NUMINAMATH_CALUDE_train_crossing_signal_pole_l209_20936

/-- Given a train and a platform with the following properties:
  * The train is 300 meters long
  * The platform is 250 meters long
  * The train crosses the platform in 33 seconds
  This theorem proves that the time taken for the train to cross a signal pole is 18 seconds. -/
theorem train_crossing_signal_pole
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : ℝ :=
let total_distance := train_length + platform_length
let train_speed := total_distance / platform_crossing_time
let signal_pole_crossing_time := train_length / train_speed
18

/-- The proof of the theorem -/
lemma train_crossing_signal_pole_proof
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 250)
  (h3 : platform_crossing_time = 33)
  : train_crossing_signal_pole train_length platform_length platform_crossing_time h1 h2 h3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_signal_pole_l209_20936


namespace NUMINAMATH_CALUDE_inequality_system_solution_l209_20937

theorem inequality_system_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | x > -2*a ∧ x > 3*a} = {x : ℝ | x > -2*a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l209_20937


namespace NUMINAMATH_CALUDE_ancient_tower_height_l209_20930

/-- Proves that the height of an ancient tower is 14.4 meters given the conditions of Xiao Liang's height and shadow length, and the tower's shadow length. -/
theorem ancient_tower_height 
  (xiao_height : ℝ) 
  (xiao_shadow : ℝ) 
  (tower_shadow : ℝ) 
  (h1 : xiao_height = 1.6)
  (h2 : xiao_shadow = 2)
  (h3 : tower_shadow = 18) :
  (xiao_height / xiao_shadow) * tower_shadow = 14.4 :=
by sorry

end NUMINAMATH_CALUDE_ancient_tower_height_l209_20930


namespace NUMINAMATH_CALUDE_lindas_furniture_spending_l209_20993

/-- Given Linda's original savings and the cost of a TV, 
    prove the fraction of her savings spent on furniture. -/
theorem lindas_furniture_spending 
  (original_savings : ℚ) 
  (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 150) : 
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_spending_l209_20993


namespace NUMINAMATH_CALUDE_inequality_proof_l209_20969

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l209_20969


namespace NUMINAMATH_CALUDE_root_implies_m_values_l209_20958

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2*m - 6 = 0) → (m = -2 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_values_l209_20958


namespace NUMINAMATH_CALUDE_power_function_through_point_l209_20953

theorem power_function_through_point (a k : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^k = ((a - 1):ℝ) * x^k) → 
  (a - 1) * (Real.sqrt 2)^k = 2 → 
  a + k = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l209_20953


namespace NUMINAMATH_CALUDE_complement_M_inter_N_eq_singleton_three_l209_20992

def M : Set ℤ := {x | x < 3}
def N : Set ℤ := {x | x < 4}
def U : Set ℤ := Set.univ

theorem complement_M_inter_N_eq_singleton_three :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_eq_singleton_three_l209_20992


namespace NUMINAMATH_CALUDE_rain_probability_l209_20951

theorem rain_probability (p_monday p_tuesday p_no_rain : ℝ) 
  (h1 : p_monday = 0.62)
  (h2 : p_tuesday = 0.54)
  (h3 : p_no_rain = 0.28)
  : p_monday + p_tuesday - (1 - p_no_rain) = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l209_20951


namespace NUMINAMATH_CALUDE_total_people_in_line_l209_20957

/-- The number of people in a ticket line -/
def ticket_line (people_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  people_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 people in the ticket line -/
theorem total_people_in_line :
  ticket_line 6 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_line_l209_20957


namespace NUMINAMATH_CALUDE_quadratic_factorization_l209_20998

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 25 * x^2 - 195 * x - 198 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -420 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l209_20998


namespace NUMINAMATH_CALUDE_linear_coefficient_is_correct_l209_20914

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

-- Define the coefficient of the linear term
def linear_coefficient : ℝ := -2

-- Theorem statement
theorem linear_coefficient_is_correct :
  ∃ (a c : ℝ), ∀ x, quadratic_equation x ↔ x^2 + linear_coefficient * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_is_correct_l209_20914


namespace NUMINAMATH_CALUDE_martha_journey_distance_l209_20988

theorem martha_journey_distance :
  -- Initial conditions
  ∀ (initial_speed : ℝ) (initial_distance : ℝ) (speed_increase : ℝ) (late_time : ℝ),
  initial_speed = 45 →
  initial_distance = 45 →
  speed_increase = 10 →
  late_time = 0.75 →
  -- The actual journey time
  ∃ (t : ℝ),
  -- The total distance
  ∃ (d : ℝ),
  -- Equation for the journey if continued at initial speed
  d = initial_speed * (t + late_time) ∧
  -- Equation for the actual journey with increased speed
  d - initial_distance = (initial_speed + speed_increase) * (t - 1) →
  -- The distance to the meeting place
  d = 230.625 :=
by sorry

end NUMINAMATH_CALUDE_martha_journey_distance_l209_20988


namespace NUMINAMATH_CALUDE_even_decreasing_function_a_equals_two_l209_20909

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if f(x) > f(y) for all 0 < x < y -/
def IsDecreasingOnPositives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem even_decreasing_function_a_equals_two (a : ℤ) :
  IsEven (fun x => x^(a^2 - 4*a)) →
  IsDecreasingOnPositives (fun x => x^(a^2 - 4*a)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_function_a_equals_two_l209_20909


namespace NUMINAMATH_CALUDE_find_a_solution_set_g_solution_set_h_l209_20947

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x^2 - 4 * x + 6

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ -3 < x ∧ x < 1

-- Theorem 1
theorem find_a (a : ℝ) (h : solution_set_condition a) : a = 3 :=
sorry

-- Define the quadratic function for part 2
def g (x : ℝ) : ℝ := 2 * x^2 - x - 3

-- Theorem 2
theorem solution_set_g :
  ∀ x, g x > 0 ↔ x < -1 ∨ x > 3/2 :=
sorry

-- Define the quadratic function for part 3
def h (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + b * x + 3

-- Theorem 3
theorem solution_set_h (b : ℝ) :
  (∀ x, h b x ≥ 0) ↔ -6 ≤ b ∧ b ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_find_a_solution_set_g_solution_set_h_l209_20947


namespace NUMINAMATH_CALUDE_junior_prom_dancer_ratio_l209_20922

theorem junior_prom_dancer_ratio :
  let total_kids : ℕ := 140
  let slow_dancers : ℕ := 25
  let non_slow_dancers : ℕ := 10
  let total_dancers : ℕ := slow_dancers + non_slow_dancers
  (total_dancers : ℚ) / total_kids = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_junior_prom_dancer_ratio_l209_20922


namespace NUMINAMATH_CALUDE_prove_necklace_sum_l209_20902

def necklace_sum (H J x S : ℕ) : Prop :=
  (H = J + 5) ∧ 
  (x = J / 2) ∧ 
  (S = 2 * H) ∧ 
  (H = 25) →
  H + J + x + S = 105

theorem prove_necklace_sum : 
  ∀ (H J x S : ℕ), necklace_sum H J x S :=
by
  sorry

end NUMINAMATH_CALUDE_prove_necklace_sum_l209_20902


namespace NUMINAMATH_CALUDE_percentage_women_non_union_l209_20966

/-- Represents the percentage of employees in a company who are men -/
def percent_men : ℝ := 0.56

/-- Represents the percentage of employees in a company who are unionized -/
def percent_unionized : ℝ := 0.60

/-- Represents the percentage of non-union employees who are women -/
def percent_women_non_union : ℝ := 0.65

/-- Theorem stating that the percentage of women among non-union employees is 65% -/
theorem percentage_women_non_union :
  percent_women_non_union = 0.65 := by sorry

end NUMINAMATH_CALUDE_percentage_women_non_union_l209_20966


namespace NUMINAMATH_CALUDE_min_m_value_l209_20933

open Real

-- Define the inequality function
def f (x m : ℝ) : Prop := x + m * log x + exp (-x) ≥ x^m

-- State the theorem
theorem min_m_value :
  (∀ x > 1, ∀ m : ℝ, f x m) →
  (∃ m₀ : ℝ, m₀ = -exp 1 ∧ (∀ m : ℝ, (∀ x > 1, f x m) → m ≥ m₀)) :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l209_20933


namespace NUMINAMATH_CALUDE_square_side_length_l209_20900

theorem square_side_length (overlap1 overlap2 overlap3 non_overlap_total : ℝ) 
  (h1 : overlap1 = 2)
  (h2 : overlap2 = 5)
  (h3 : overlap3 = 8)
  (h4 : non_overlap_total = 117)
  (h5 : overlap1 > 0 ∧ overlap2 > 0 ∧ overlap3 > 0 ∧ non_overlap_total > 0) :
  ∃ (side_length : ℝ), 
    side_length = 7 ∧ 
    3 * side_length ^ 2 = non_overlap_total + 2 * (overlap1 + overlap2 + overlap3) :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l209_20900


namespace NUMINAMATH_CALUDE_shekar_average_marks_l209_20935

theorem shekar_average_marks :
  let math_score := 76
  let science_score := 65
  let social_studies_score := 82
  let english_score := 67
  let biology_score := 95
  let total_score := math_score + science_score + social_studies_score + english_score + biology_score
  let num_subjects := 5
  (total_score / num_subjects : ℚ) = 77 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l209_20935


namespace NUMINAMATH_CALUDE_x_and_a_ranges_l209_20999

theorem x_and_a_ranges (x m a : ℝ) 
  (h1 : x^2 - 4*a*x + 3*a^2 < 0)
  (h2 : x = (1/2)^(m-1))
  (h3 : 1 < m ∧ m < 2) :
  (a = 1/4 → 1/2 < x ∧ x < 3/4) ∧
  (1/3 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_x_and_a_ranges_l209_20999


namespace NUMINAMATH_CALUDE_work_efficiency_increase_l209_20915

theorem work_efficiency_increase (original_months : ℕ) (actual_months : ℕ) (x : ℚ) : 
  original_months = 20 →
  actual_months = 18 →
  (1 : ℚ) / actual_months = (1 : ℚ) / original_months * (1 + x / 100) :=
by sorry

end NUMINAMATH_CALUDE_work_efficiency_increase_l209_20915


namespace NUMINAMATH_CALUDE_right_trapezoid_area_l209_20978

/-- The area of a right trapezoid with specific dimensions -/
theorem right_trapezoid_area (upper_base lower_base height : ℝ) 
  (h1 : upper_base = 25)
  (h2 : lower_base - 15 = height)
  (h3 : height > 0) : 
  (upper_base + lower_base) * height / 2 = 175 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_area_l209_20978


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l209_20946

/-- A point P(x, y) is in the first quadrant if and only if x > 0 and y > 0 -/
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The x-coordinate of point P is m - 1 -/
def x_coord (m : ℝ) : ℝ := m - 1

/-- The y-coordinate of point P is m + 2 -/
def y_coord (m : ℝ) : ℝ := m + 2

/-- Theorem: If point P(m-1, m+2) is in the first quadrant, then m > 1 -/
theorem point_in_first_quadrant_implies_m_gt_one (m : ℝ) :
  in_first_quadrant (x_coord m) (y_coord m) → m > 1 :=
by sorry

/-- Theorem: If m > 1, then point P(m-1, m+2) is in the first quadrant -/
theorem m_gt_one_implies_point_in_first_quadrant (m : ℝ) :
  m > 1 → in_first_quadrant (x_coord m) (y_coord m) :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_implies_m_gt_one_m_gt_one_implies_point_in_first_quadrant_l209_20946


namespace NUMINAMATH_CALUDE_cricket_score_theorem_l209_20911

/-- Represents the score of a cricket match -/
def CricketScore := ℕ

/-- Calculates the total runs from boundaries -/
def boundaryRuns (boundaries : ℕ) : ℕ := 4 * boundaries

/-- Calculates the total runs from sixes -/
def sixRuns (sixes : ℕ) : ℕ := 6 * sixes

/-- Theorem: Given the conditions of the cricket match, prove the total score is 142 runs -/
theorem cricket_score_theorem (boundaries sixes : ℕ) 
  (h1 : boundaries = 12)
  (h2 : sixes = 2)
  (h3 : (57.74647887323944 : ℚ) / 100 * 142 = 142 - boundaryRuns boundaries - sixRuns sixes) :
  142 = boundaryRuns boundaries + sixRuns sixes + 
    ((57.74647887323944 : ℚ) / 100 * 142).floor :=
by sorry

end NUMINAMATH_CALUDE_cricket_score_theorem_l209_20911


namespace NUMINAMATH_CALUDE_wall_completion_time_proof_l209_20964

/-- Represents the wall dimensions -/
structure WallDimensions where
  thickness : ℝ
  length : ℝ
  height : ℝ

/-- Represents the working conditions -/
structure WorkingConditions where
  normal_pace : ℝ
  break_duration : ℝ
  break_count : ℕ
  faster_rate : ℝ
  faster_duration : ℝ
  min_work_between_breaks : ℝ

/-- Calculates the shortest possible time to complete the wall -/
def shortest_completion_time (wall : WallDimensions) (conditions : WorkingConditions) : ℝ :=
  sorry

theorem wall_completion_time_proof (wall : WallDimensions) (conditions : WorkingConditions) :
  wall.thickness = 0.25 ∧
  wall.length = 50 ∧
  wall.height = 2 ∧
  conditions.normal_pace = 26 ∧
  conditions.break_duration = 0.5 ∧
  conditions.break_count = 6 ∧
  conditions.faster_rate = 1.25 ∧
  conditions.faster_duration = 1 ∧
  conditions.min_work_between_breaks = 0.75 →
  shortest_completion_time wall conditions = 27.25 :=
by sorry

end NUMINAMATH_CALUDE_wall_completion_time_proof_l209_20964


namespace NUMINAMATH_CALUDE_function_sum_derivative_difference_l209_20973

/-- Given a function f(x) = a*sin(3x) + b*x^3 + 4 where a and b are real numbers,
    prove that f(2014) + f(-2014) + f'(2015) - f'(-2015) = 8 -/
theorem function_sum_derivative_difference (a b : ℝ) : 
  let f (x : ℝ) := a * Real.sin (3 * x) + b * x^3 + 4
  let f' (x : ℝ) := 3 * a * Real.cos (3 * x) + 3 * b * x^2
  f 2014 + f (-2014) + f' 2015 - f' (-2015) = 8 := by
sorry

end NUMINAMATH_CALUDE_function_sum_derivative_difference_l209_20973


namespace NUMINAMATH_CALUDE_lunch_expense_calculation_l209_20991

theorem lunch_expense_calculation (initial_money : ℝ) (gasoline_expense : ℝ) (gift_expense_per_person : ℝ) (grandma_gift_per_person : ℝ) (return_trip_money : ℝ) :
  initial_money = 50 →
  gasoline_expense = 8 →
  gift_expense_per_person = 5 →
  grandma_gift_per_person = 10 →
  return_trip_money = 36.35 →
  let total_money := initial_money + 2 * grandma_gift_per_person
  let total_expense := gasoline_expense + 2 * gift_expense_per_person
  let lunch_expense := total_money - total_expense - return_trip_money
  lunch_expense = 15.65 := by
sorry

end NUMINAMATH_CALUDE_lunch_expense_calculation_l209_20991


namespace NUMINAMATH_CALUDE_work_completion_time_l209_20945

/-- Given two workers a and b, where a is thrice as fast as b, proves that if a can complete
    a work alone in 40 days, then a and b together can complete the work in 30 days. -/
theorem work_completion_time
  (rate_a rate_b : ℝ)  -- Rates at which workers a and b work
  (h1 : rate_a = 3 * rate_b)  -- a is thrice as fast as b
  (h2 : rate_a * 40 = 1)  -- a alone completes the work in 40 days
  : (rate_a + rate_b) * 30 = 1 :=  -- a and b together complete the work in 30 days
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l209_20945


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l209_20920

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 * (x - 1)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-1) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1) 2, f x = min) ∧
    max = 4 ∧ min = -2 := by
  sorry


end NUMINAMATH_CALUDE_f_max_min_on_interval_l209_20920


namespace NUMINAMATH_CALUDE_base_of_exponent_l209_20980

theorem base_of_exponent (x : ℕ) (h : x = 14) :
  (∀ y : ℕ, y > x → ¬(3^y ∣ 9^7)) ∧ (3^x ∣ 9^7) →
  ∃ b : ℕ, b^7 = 9^7 ∧ b = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_of_exponent_l209_20980


namespace NUMINAMATH_CALUDE_conjunction_implies_left_prop_l209_20907

theorem conjunction_implies_left_prop (p q : Prop) : (p ∧ q) → p := by
  sorry

end NUMINAMATH_CALUDE_conjunction_implies_left_prop_l209_20907


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l209_20961

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x - a/x)^9
def coeff_x3 (a : ℝ) : ℝ := -binomial 9 3 * a^3

-- Theorem statement
theorem expansion_coefficient_implies_a_value (a : ℝ) : 
  coeff_x3 a = -84 → a = 1 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l209_20961


namespace NUMINAMATH_CALUDE_line_plane_relationships_l209_20960

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relationships between lines and planes
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def parallel_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry
def perpendicular_line_line (l1 : Line) (l2 : Line) : Prop := sorry
def parallel_line_line (l1 : Line) (l2 : Line) : Prop := sorry

-- Define the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (hm : m ≠ n) (hα : α ≠ β) : 
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   perpendicular_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (¬ (parallel_line_plane m α ∧ 
      perpendicular_line_plane n β ∧ 
      perpendicular_plane_plane α β → 
      parallel_line_line m n)) ∧
  (perpendicular_line_plane m α ∧ 
   parallel_line_plane n β ∧ 
   parallel_plane_plane α β → 
   perpendicular_line_line m n) ∧
  (perpendicular_line_plane m α ∧ 
   perpendicular_line_plane n β ∧ 
   parallel_plane_plane α β → 
   parallel_line_line m n) := by
  sorry


end NUMINAMATH_CALUDE_line_plane_relationships_l209_20960


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l209_20927

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : total_kids / 2 = total_kids / 2) -- Half of kids go to soccer camp
  (h3 : (total_kids / 2) / 4 = (total_kids / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  : total_kids / 2 - (total_kids / 2) / 4 = 750 := by
  sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l209_20927


namespace NUMINAMATH_CALUDE_quirky_triangle_characterization_l209_20906

/-- A triangle is quirky if there exist integers r₁, r₂, r₃, not all zero, 
    such that r₁θ₁ + r₂θ₂ + r₃θ₃ = 0, where θ₁, θ₂, θ₃ are the measures of the triangle's angles. -/
def IsQuirky (θ₁ θ₂ θ₃ : ℝ) : Prop :=
  ∃ r₁ r₂ r₃ : ℤ, (r₁ ≠ 0 ∨ r₂ ≠ 0 ∨ r₃ ≠ 0) ∧ r₁ * θ₁ + r₂ * θ₂ + r₃ * θ₃ = 0

/-- The angles of a triangle with side lengths n-1, n, n+1 -/
def TriangleAngles (n : ℕ) : (ℝ × ℝ × ℝ) :=
  sorry

theorem quirky_triangle_characterization (n : ℕ) (h : n ≥ 3) :
  let (θ₁, θ₂, θ₃) := TriangleAngles n
  IsQuirky θ₁ θ₂ θ₃ ↔ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 :=
sorry

end NUMINAMATH_CALUDE_quirky_triangle_characterization_l209_20906


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_l209_20987

/-- Operation on three numbers as defined in the problem -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating the sum of results from two specific triangles -/
theorem sum_of_triangle_ops : 
  triangle_op 4 2 3 + triangle_op 3 5 1 = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_l209_20987


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l209_20926

theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 25) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 1160 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l209_20926


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l209_20997

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l209_20997


namespace NUMINAMATH_CALUDE_first_same_side_after_104_minutes_l209_20954

/-- Represents a person walking around a pentagonal square -/
structure Walker where
  start_point : Fin 5
  speed : ℝ

/-- The time when two walkers are first on the same side of a pentagonal square -/
def first_same_side_time (perimeter : ℝ) (walker_a walker_b : Walker) : ℝ :=
  sorry

/-- The main theorem -/
theorem first_same_side_after_104_minutes :
  let perimeter : ℝ := 2000
  let walker_a : Walker := { start_point := 0, speed := 50 }
  let walker_b : Walker := { start_point := 2, speed := 46 }
  first_same_side_time perimeter walker_a walker_b = 104 := by
  sorry

end NUMINAMATH_CALUDE_first_same_side_after_104_minutes_l209_20954


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l209_20981

theorem imaginary_part_of_complex_expression : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I) * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l209_20981


namespace NUMINAMATH_CALUDE_suitable_squares_are_1_4_9_49_l209_20928

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A number is suitable if it's the smallest among all natural numbers with the same digit sum -/
def is_suitable (n : ℕ) : Prop :=
  ∀ m : ℕ, digit_sum m = digit_sum n → n ≤ m

/-- The set of all suitable square numbers -/
def suitable_squares : Set ℕ :=
  {n : ℕ | is_suitable n ∧ ∃ k : ℕ, n = k^2}

/-- Theorem: The set of suitable square numbers is exactly {1, 4, 9, 49} -/
theorem suitable_squares_are_1_4_9_49 : suitable_squares = {1, 4, 9, 49} := by sorry

end NUMINAMATH_CALUDE_suitable_squares_are_1_4_9_49_l209_20928


namespace NUMINAMATH_CALUDE_sin_A_range_l209_20996

theorem sin_A_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- positive angles
  A + B + C = π ∧ -- sum of angles in a triangle
  C = π / 3 ∧ 
  a = 6 ∧ 
  1 ≤ b ∧ b ≤ 4 ∧
  a / (Real.sin A) = b / (Real.sin B) ∧ -- sine rule
  a / (Real.sin A) = c / (Real.sin C) ∧ -- sine rule
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) -- cosine rule
  →
  3 * Real.sqrt 93 / 31 ≤ Real.sin A ∧ Real.sin A ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_sin_A_range_l209_20996


namespace NUMINAMATH_CALUDE_octagon_diagonals_l209_20984

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Proof that an octagon has 20 diagonals -/
theorem octagon_diagonals : num_diagonals 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l209_20984


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l209_20929

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) (h3 : downstream_time = 3) (h4 : upstream_time = 3) :
  ∃ (v_m : ℝ), v_m = 11.5 ∧ 
    (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2 = v_m :=
by sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l209_20929


namespace NUMINAMATH_CALUDE_dart_board_probability_l209_20948

/-- The probability of a dart landing in the center square of a regular hexagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let center_square_area := s^2 / 3
  center_square_area / hexagon_area = 2 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l209_20948


namespace NUMINAMATH_CALUDE_harrietts_pennies_l209_20923

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- The problem statement -/
theorem harrietts_pennies :
  let quarters := 10
  let dimes := 3
  let nickels := 3
  let total_cents := 300  -- $3 in cents
  let other_coins_value := 
    quarters * coin_value "quarter" + 
    dimes * coin_value "dime" + 
    nickels * coin_value "nickel"
  let pennies := total_cents - other_coins_value
  pennies = 5 := by sorry

end NUMINAMATH_CALUDE_harrietts_pennies_l209_20923


namespace NUMINAMATH_CALUDE_count_valid_a_l209_20941

-- Define the system of inequalities
def system_inequalities (a : ℤ) (x : ℤ) : Prop :=
  6 * x - 5 ≥ a ∧ (x : ℚ) / 4 - (x - 1 : ℚ) / 6 < 1 / 2

-- Define the equation
def equation (a : ℤ) (y : ℚ) : Prop :=
  4 * y - 3 * (a : ℚ) = 2 * (y - 3)

-- Main theorem
theorem count_valid_a : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ a : ℤ, a ∈ s ↔ 
      (∃! (sol : Finset ℤ), sol.card = 2 ∧ 
        (∀ x : ℤ, x ∈ sol ↔ system_inequalities a x)) ∧
      (∃ y : ℚ, y > 0 ∧ equation a y))) := by sorry

end NUMINAMATH_CALUDE_count_valid_a_l209_20941


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l209_20995

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (m a : ℝ) : ℝ := a

/-- A line in slope-intercept form is defined by y = mx + b, where m is the slope and b is the y-intercept. -/
def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem y_intercept_of_line :
  y_intercept 2 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l209_20995


namespace NUMINAMATH_CALUDE_exam_scores_l209_20918

theorem exam_scores (average : ℝ) (difference : ℝ) 
  (h_average : average = 98) 
  (h_difference : difference = 2) : 
  ∃ (chinese math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧ 
    chinese = 97 ∧ 
    math = 99 := by
  sorry

end NUMINAMATH_CALUDE_exam_scores_l209_20918


namespace NUMINAMATH_CALUDE_guaranteed_payoff_probability_l209_20990

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The score in the game is the sum of two die rolls -/
def score (roll1 roll2 : Die) : Nat := roll1.val + roll2.val + 2

/-- The maximum possible score in the game -/
def max_score : Nat := 12

/-- The number of players in the game -/
def num_players : Nat := 22

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : Die) : Rat := 1 / 6

theorem guaranteed_payoff_probability :
  let guaranteed_score := max_score
  let prob_guaranteed_score := (prob_single_roll ⟨5, by norm_num⟩) * (prob_single_roll ⟨5, by norm_num⟩)
  (∀ s, s < guaranteed_score → ∃ (rolls : Fin num_players → Die × Die), 
    ∃ i, score (rolls i).1 (rolls i).2 ≥ s) ∧ 
  prob_guaranteed_score = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_payoff_probability_l209_20990


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l209_20938

theorem simplify_and_evaluate (x y : ℝ) (h : x / y = 3) :
  (1 + y^2 / (x^2 - y^2)) * ((x - y) / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l209_20938


namespace NUMINAMATH_CALUDE_equal_money_distribution_l209_20977

theorem equal_money_distribution (younger_money : ℝ) (h : younger_money > 0) :
  let elder_money := 1.25 * younger_money
  let transfer_amount := 0.1 * elder_money
  elder_money - transfer_amount = younger_money + transfer_amount := by
  sorry

end NUMINAMATH_CALUDE_equal_money_distribution_l209_20977


namespace NUMINAMATH_CALUDE_cookie_count_l209_20949

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the number of full smaller rectangles that can fit into a larger rectangle -/
def fullRectanglesFit (large : Dimensions) (small : Dimensions) : ℕ :=
  (large.length / small.length) * (large.width / small.width)

theorem cookie_count :
  let sheet := Dimensions.mk 30 24
  let cookie := Dimensions.mk 3 4
  fullRectanglesFit sheet cookie = 60 := by
  sorry

#eval fullRectanglesFit (Dimensions.mk 30 24) (Dimensions.mk 3 4)

end NUMINAMATH_CALUDE_cookie_count_l209_20949


namespace NUMINAMATH_CALUDE_proportional_sampling_l209_20921

theorem proportional_sampling :
  let total_population : ℕ := 162
  let elderly_population : ℕ := 27
  let middle_aged_population : ℕ := 54
  let young_population : ℕ := 81
  let sample_size : ℕ := 36

  let elderly_sample : ℕ := 6
  let middle_aged_sample : ℕ := 12
  let young_sample : ℕ := 18

  elderly_population + middle_aged_population + young_population = total_population →
  elderly_sample + middle_aged_sample + young_sample = sample_size →
  (elderly_sample : ℚ) / sample_size = (elderly_population : ℚ) / total_population ∧
  (middle_aged_sample : ℚ) / sample_size = (middle_aged_population : ℚ) / total_population ∧
  (young_sample : ℚ) / sample_size = (young_population : ℚ) / total_population :=
by
  sorry

end NUMINAMATH_CALUDE_proportional_sampling_l209_20921


namespace NUMINAMATH_CALUDE_farm_ratio_l209_20910

def cows : ℕ := 21
def horses : ℕ := 6

theorem farm_ratio : (cows / horses : ℚ) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l209_20910


namespace NUMINAMATH_CALUDE_prism_volume_l209_20985

/-- The volume of a right rectangular prism with face areas 30, 50, and 75 square centimeters is 335 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 335 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l209_20985
