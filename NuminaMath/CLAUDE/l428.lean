import Mathlib

namespace fishes_from_ontario_erie_l428_42814

/-- The number of fishes taken from Lake Huron and Michigan -/
def huron_michigan : ℕ := 30

/-- The number of fishes taken from Lake Superior -/
def superior : ℕ := 44

/-- The total number of fishes brought home -/
def total : ℕ := 97

/-- The number of fishes taken from Lake Ontario and Erie -/
def ontario_erie : ℕ := total - (huron_michigan + superior)

theorem fishes_from_ontario_erie : ontario_erie = 23 := by
  sorry

end fishes_from_ontario_erie_l428_42814


namespace calculation_proofs_l428_42848

theorem calculation_proofs :
  (4.4 * 25 = 110) ∧
  (13.2 * 1.1 - 8.45 = 6.07) ∧
  (76.84 * 103 - 7.684 * 30 = 7684) ∧
  ((2.8 + 3.85 / 3.5) / 3 = 1.3) := by
  sorry

end calculation_proofs_l428_42848


namespace ac_cube_l428_42817

theorem ac_cube (a b c : ℝ) (h1 : a * b = 1) (h2 : b + c = 0) : (a * c)^3 = -1 := by
  sorry

end ac_cube_l428_42817


namespace find_a_l428_42897

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- State the theorem
theorem find_a : ∃ a : ℝ, (A a ∪ B a = {0, 1, 2, 4, 16}) → a = 4 := by
  sorry

end find_a_l428_42897


namespace rectangle_area_diagonal_l428_42892

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d, 
    its area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end rectangle_area_diagonal_l428_42892


namespace train_length_calculation_l428_42819

/-- The length of each train in meters -/
def train_length : ℝ := 79.92

/-- The speed of the faster train in km/hr -/
def faster_speed : ℝ := 52

/-- The speed of the slower train in km/hr -/
def slower_speed : ℝ := 36

/-- The time it takes for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := (faster_speed - slower_speed) * 1000 / 3600
  2 * train_length = relative_speed * passing_time := by sorry

end train_length_calculation_l428_42819


namespace parallelepiped_volume_l428_42847

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (k : ℝ) : Fin 3 → ℝ := ![2, k, 3]
def vector3 (k : ℝ) : Fin 3 → ℝ := ![2, 3, k]

def matrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of (λ i j => match i, j with
    | 0, 0 => 3 | 0, 1 => 2 | 0, 2 => 2
    | 1, 0 => 4 | 1, 1 => k | 1, 2 => 3
    | 2, 0 => 5 | 2, 1 => 3 | 2, 2 => k
    | _, _ => 0)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ |Matrix.det (matrix k)| = 30 → k = 3 + Real.sqrt 10 := by
  sorry

end parallelepiped_volume_l428_42847


namespace complement_of_A_in_U_l428_42845

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 3}

theorem complement_of_A_in_U :
  Set.compl A = {2, 4} := by sorry

end complement_of_A_in_U_l428_42845


namespace half_plus_five_equals_thirteen_l428_42831

theorem half_plus_five_equals_thirteen (n : ℝ) : (1/2) * n + 5 = 13 → n = 16 := by
  sorry

end half_plus_five_equals_thirteen_l428_42831


namespace cow_herd_division_l428_42874

theorem cow_herd_division (n : ℕ) : 
  (n / 2 : ℚ) + (n / 4 : ℚ) + (n / 5 : ℚ) + 7 = n → n = 140 := by
  sorry

end cow_herd_division_l428_42874


namespace triangle_side_length_l428_42852

/-- Given a triangle ABC with angle A = π/6, side a = 1, and side b = √3, 
    the length of side c is either 2 or 1. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → a = 1 → b = Real.sqrt 3 → 
  (c = 2 ∨ c = 1) := by sorry

end triangle_side_length_l428_42852


namespace tangent_line_minimum_value_l428_42835

/-- Given a function f(x) = ax² + b/x where a > 0 and b > 0, 
    and its tangent line at x = 1 passes through (3/2, 1/2),
    prove that the minimum value of 1/a + 1/b is 9 -/
theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => a * x^2 + b / x
  let f' := fun x => 2 * a * x - b / x^2
  let tangent_slope := f' 1
  let tangent_point := (1, f 1)
  (tangent_slope * (3/2 - 1) = 1/2 - f 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 9) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 1/b' = 9) :=
by sorry

end tangent_line_minimum_value_l428_42835


namespace stratified_sampling_l428_42878

theorem stratified_sampling (total_capacity : ℕ) (sample_capacity : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ) :
  total_capacity = 56 →
  sample_capacity = 14 →
  ratio_A = 1 →
  ratio_B = 2 →
  ratio_C = 4 →
  ∃ (sample_A sample_B sample_C : ℕ),
    sample_A = 2 ∧
    sample_B = 4 ∧
    sample_C = 8 ∧
    sample_A + sample_B + sample_C = sample_capacity ∧
    sample_A * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_A ∧
    sample_B * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_B ∧
    sample_C * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_C :=
by sorry

end stratified_sampling_l428_42878


namespace points_on_line_value_at_2_l428_42893

/-- A linear function passing through given points -/
def linear_function (x : ℝ) : ℝ := x - 1

/-- The given points satisfy the linear function -/
theorem points_on_line : 
  linear_function (-1) = -2 ∧ 
  linear_function 0 = -1 ∧ 
  linear_function 1 = 0 := by sorry

/-- The y-value corresponding to x = 2 is 1 -/
theorem value_at_2 : linear_function 2 = 1 := by sorry

end points_on_line_value_at_2_l428_42893


namespace lending_years_calculation_l428_42872

/-- Proves that the number of years the first part is lent is 5 -/
theorem lending_years_calculation (total_sum : ℝ) (second_part : ℝ) 
  (first_rate : ℝ) (second_rate : ℝ) (second_years : ℕ) :
  total_sum = 2665 →
  second_part = 1332.5 →
  first_rate = 0.03 →
  second_rate = 0.05 →
  second_years = 3 →
  let first_part := total_sum - second_part
  let first_interest := first_part * first_rate
  let second_interest := second_part * second_rate * second_years
  first_interest * (5 : ℝ) = second_interest :=
by sorry

end lending_years_calculation_l428_42872


namespace ten_coin_flips_sequences_l428_42873

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end ten_coin_flips_sequences_l428_42873


namespace root_difference_quadratic_l428_42802

/-- The nonnegative difference between the roots of x^2 + 30x + 180 = -36 is 6 -/
theorem root_difference_quadratic : 
  let f : ℝ → ℝ := λ x => x^2 + 30*x + 216
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 := by
sorry

end root_difference_quadratic_l428_42802


namespace absolute_value_inequality_l428_42824

theorem absolute_value_inequality (x : ℝ) : |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 := by
  sorry

end absolute_value_inequality_l428_42824


namespace negation_equivalence_l428_42830

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 2*x + x + 1 ≤ 0) ↔ (∀ x : ℤ, 2*x + x + 1 > 0) :=
by sorry

end negation_equivalence_l428_42830


namespace min_cost_disinfectants_l428_42801

/-- Represents the price and quantity of disinfectants A and B -/
structure Disinfectants where
  price_A : ℕ
  price_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Calculates the total cost of purchasing disinfectants -/
def total_cost (d : Disinfectants) : ℕ :=
  d.price_A * d.quantity_A + d.price_B * d.quantity_B

/-- Represents the constraints on quantities of disinfectants -/
def valid_quantities (d : Disinfectants) : Prop :=
  d.quantity_A + d.quantity_B = 30 ∧
  d.quantity_A ≥ d.quantity_B + 5 ∧
  d.quantity_A ≤ 2 * d.quantity_B

theorem min_cost_disinfectants :
  ∃ (d : Disinfectants),
    d.price_A = 45 ∧
    d.price_B = 35 ∧
    9 * d.price_A + 6 * d.price_B = 615 ∧
    8 * d.price_A + 12 * d.price_B = 780 ∧
    valid_quantities d ∧
    (∀ (d' : Disinfectants), valid_quantities d' → total_cost d ≤ total_cost d') ∧
    total_cost d = 1230 :=
by
  sorry

end min_cost_disinfectants_l428_42801


namespace six_digit_integers_count_is_60_l428_42849

/-- The number of different six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 5 -/
def sixDigitIntegersCount : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of different six-digit integers 
    that can be formed using the digits 1, 1, 3, 3, 3, and 5 is equal to 60 -/
theorem six_digit_integers_count_is_60 : sixDigitIntegersCount = 60 := by
  sorry

end six_digit_integers_count_is_60_l428_42849


namespace range_of_k_for_two_roots_l428_42846

open Real

theorem range_of_k_for_two_roots (g : ℝ → ℝ) (k : ℝ) :
  (∀ x, g x = 2 * sin (2 * x - π / 6)) →
  (∀ x ∈ Set.Icc 0 (π / 2), (g x - k = 0 → ∃ y ∈ Set.Icc 0 (π / 2), x ≠ y ∧ g y - k = 0)) ↔
  k ∈ Set.Icc 1 2 ∧ k ≠ 2 :=
sorry

end range_of_k_for_two_roots_l428_42846


namespace imaginary_part_of_z_l428_42820

theorem imaginary_part_of_z (z : ℂ) : z = (1 : ℂ) / (1 + Complex.I) → z.im = -1/2 := by
  sorry

end imaginary_part_of_z_l428_42820


namespace new_trailers_correct_l428_42811

/-- Represents the trailer park scenario -/
structure TrailerPark where
  initial_count : ℕ
  initial_avg_age : ℕ
  years_passed : ℕ
  current_avg_age : ℕ

/-- Calculates the number of new trailers added -/
def new_trailers (park : TrailerPark) : ℕ :=
  13

/-- Theorem stating that the calculated number of new trailers is correct -/
theorem new_trailers_correct (park : TrailerPark) 
  (h1 : park.initial_count = 30)
  (h2 : park.initial_avg_age = 10)
  (h3 : park.years_passed = 5)
  (h4 : park.current_avg_age = 12) :
  new_trailers park = 13 := by
  sorry

#check new_trailers_correct

end new_trailers_correct_l428_42811


namespace cost_calculation_l428_42853

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 20.50

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  ∀ mango_cost rice_cost : ℝ,
  mango_rice_relation mango_cost rice_cost →
  flour_rice_relation rice_cost →
  total_cost mango_cost rice_cost = 311.60 := by
sorry

end cost_calculation_l428_42853


namespace composite_product_ratio_l428_42844

def first_twelve_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21]

def product_first_six : ℕ := (first_twelve_composites.take 6).prod

def product_next_six : ℕ := (first_twelve_composites.drop 6).prod

theorem composite_product_ratio : 
  (product_first_six : ℚ) / product_next_six = 2 / 245 := by sorry

end composite_product_ratio_l428_42844


namespace rectangle_toothpicks_l428_42815

/-- Calculate the number of toothpicks needed for a rectangular grid -/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A rectangular grid with length 20 and width 10 requires 430 toothpicks -/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end rectangle_toothpicks_l428_42815


namespace income_ratio_uma_bala_l428_42876

theorem income_ratio_uma_bala (uma_income : ℕ) (uma_expenditure bala_expenditure : ℕ) 
  (h1 : uma_income = 16000)
  (h2 : uma_expenditure = 7 * bala_expenditure / 6)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000)
  : uma_income / (uma_income - 2000) = 8 / 7 :=
by sorry

end income_ratio_uma_bala_l428_42876


namespace girls_without_pets_girls_without_pets_proof_l428_42813

theorem girls_without_pets (total_students : ℕ) (boys_fraction : ℚ) 
  (girls_with_dogs : ℚ) (girls_with_cats : ℚ) : ℕ :=
  let girls_fraction := 1 - boys_fraction
  let total_girls := (total_students : ℚ) * girls_fraction
  let girls_without_pets_fraction := 1 - girls_with_dogs - girls_with_cats
  let girls_without_pets := total_girls * girls_without_pets_fraction
  8

theorem girls_without_pets_proof :
  girls_without_pets 30 (1/3) (2/5) (1/5) = 8 := by
  sorry

end girls_without_pets_girls_without_pets_proof_l428_42813


namespace at_least_one_non_negative_l428_42826

theorem at_least_one_non_negative (x : ℝ) : 
  let m := x^2 - 1
  let n := 2*x + 2
  max m n ≥ 0 := by sorry

end at_least_one_non_negative_l428_42826


namespace last_year_winner_ounces_l428_42883

/-- The amount of ounces in each hamburger -/
def hamburger_ounces : ℕ := 4

/-- The number of hamburgers Tonya needs to eat to beat last year's winner -/
def hamburgers_to_beat : ℕ := 22

/-- Theorem: The amount of ounces eaten by last year's winner is 88 -/
theorem last_year_winner_ounces : 
  hamburger_ounces * hamburgers_to_beat - hamburger_ounces = 88 := by
  sorry

end last_year_winner_ounces_l428_42883


namespace hearty_beads_count_l428_42857

/-- The number of beads Hearty has in total -/
def total_beads (blue_packages red_packages beads_per_package : ℕ) : ℕ :=
  (blue_packages + red_packages) * beads_per_package

/-- Proof that Hearty has 320 beads in total -/
theorem hearty_beads_count :
  total_beads 3 5 40 = 320 := by
  sorry

end hearty_beads_count_l428_42857


namespace probability_at_least_one_success_l428_42865

theorem probability_at_least_one_success (p : ℝ) (n : ℕ) (h1 : p = 3/10) (h2 : n = 2) :
  1 - (1 - p)^n = 51/100 := by
  sorry

end probability_at_least_one_success_l428_42865


namespace min_t_value_l428_42804

/-- Ellipse C with eccentricity sqrt(2)/2 passing through (1, sqrt(2)/2) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (2, 0)

/-- Line containing point P -/
def line_P (x y : ℝ) : Prop := x + y = 1

/-- Vector relation between OA, OB, and OP -/
def vector_relation (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * P.1 ∧ A.2 + B.2 = t * P.2

/-- Main theorem: Minimum value of t -/
theorem min_t_value :
  ∃ (t_min : ℝ), 
    (∀ (A B P : ℝ × ℝ) (t : ℝ),
      ellipse_C A.1 A.2 → 
      ellipse_C B.1 B.2 → 
      line_P P.1 P.2 →
      vector_relation A B P t →
      t ≥ t_min) ∧
    t_min = 2 - Real.sqrt 6 :=
sorry

end min_t_value_l428_42804


namespace cut_polygon_perimeter_decrease_l428_42887

/-- Represents a polygon -/
structure Polygon where
  perimeter : ℝ
  perim_pos : perimeter > 0

/-- Represents the result of cutting a polygon with a straight line -/
structure CutPolygon where
  original : Polygon
  part1 : Polygon
  part2 : Polygon

/-- Theorem: The perimeter of each part resulting from cutting a polygon
    with a straight line is less than the perimeter of the original polygon -/
theorem cut_polygon_perimeter_decrease (cp : CutPolygon) :
  cp.part1.perimeter < cp.original.perimeter ∧
  cp.part2.perimeter < cp.original.perimeter := by
  sorry

end cut_polygon_perimeter_decrease_l428_42887


namespace correct_average_l428_42896

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_readings : List (ℚ × ℚ)) : 
  n = 20 ∧ 
  initial_avg = 15 ∧ 
  incorrect_readings = [(42, 52), (68, 78), (85, 95)] →
  (n : ℚ) * initial_avg + (incorrect_readings.map (λ p => p.2 - p.1)).sum = n * (16.5 : ℚ) := by
  sorry

end correct_average_l428_42896


namespace chessboard_diagonal_squares_l428_42803

/-- The number of squares a diagonal passes through on a chessboard -/
def diagonalSquares (width : Nat) (height : Nat) : Nat :=
  width + height + Nat.gcd width height - 2

/-- Theorem: The diagonal of a 1983 × 999 chessboard passes through 2979 squares -/
theorem chessboard_diagonal_squares :
  diagonalSquares 1983 999 = 2979 := by
  sorry

end chessboard_diagonal_squares_l428_42803


namespace min_value_sqrt_sum_l428_42866

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : 0 < a + b + c) :
  2 ≤ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
    a' * b' + b' * c' + c' * a' = a' + b' + c' ∧ 0 < a' + b' + c' ∧
    Real.sqrt (a' * b') + Real.sqrt (b' * c') + Real.sqrt (c' * a') = 2 :=
by sorry

end min_value_sqrt_sum_l428_42866


namespace simplify_like_terms_l428_42877

theorem simplify_like_terms (x : ℝ) : 5*x + 2*x + 7*x = 14*x := by
  sorry

end simplify_like_terms_l428_42877


namespace max_true_statements_l428_42800

theorem max_true_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 ≤ 4) ∧
  (∃ (s1 s2 s3 s4 s5 : Bool),
    s1 = (1 / a > 1 / b) ∧
    s2 = (a^2 > b^2) ∧
    s3 = (a > b) ∧
    s4 = (|a| > 1) ∧
    s5 = (b < 1) ∧
    s1 + s2 + s3 + s4 + s5 = 4) :=
by sorry

end max_true_statements_l428_42800


namespace gcd_bound_l428_42895

theorem gcd_bound (a b : ℕ) (h : ℕ) (h_int : (a + 1) / b + (b + 1) / a = h) :
  Nat.gcd a b ≤ Nat.sqrt (a + b) := by
  sorry

end gcd_bound_l428_42895


namespace special_calculator_problem_l428_42898

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Applies the calculator's operation to a two-digit number -/
def calculator_operation (x : ℕ) : ℕ :=
  reverse_digits (2 * x) + 2

theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 → calculator_operation x = 27 → x = 26 := by
sorry

end special_calculator_problem_l428_42898


namespace lawn_width_is_60_l428_42886

/-- Represents a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  road_width : ℝ
  road_area : ℝ

/-- Theorem: The width of the lawn is 60 meters -/
theorem lawn_width_is_60 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.road_width = 10)
  (h3 : lawn.road_area = 1300)
  : lawn.width = 60 := by
  sorry

#check lawn_width_is_60

end lawn_width_is_60_l428_42886


namespace map_distance_calculation_l428_42867

/-- Calculates the distance on a map given travel time, speed, and map scale -/
theorem map_distance_calculation (travel_time : ℝ) (average_speed : ℝ) (map_scale : ℝ) :
  travel_time = 6.5 →
  average_speed = 60 →
  map_scale = 0.01282051282051282 →
  travel_time * average_speed * map_scale = 5 := by
  sorry

end map_distance_calculation_l428_42867


namespace cos_2theta_minus_15_deg_l428_42841

theorem cos_2theta_minus_15_deg (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (θ + π / 12) = 4 / 5) : 
  Real.cos (2 * θ - π / 12) = 17 * Real.sqrt 2 / 50 := by
  sorry

end cos_2theta_minus_15_deg_l428_42841


namespace leonard_younger_than_nina_l428_42854

/-- Given the ages of Leonard, Nina, and Jerome, prove that Leonard is 4 years younger than Nina. -/
theorem leonard_younger_than_nina :
  ∀ (leonard nina jerome : ℕ),
    leonard = 6 →
    nina = jerome / 2 →
    leonard + nina + jerome = 36 →
    nina - leonard = 4 :=
by sorry

end leonard_younger_than_nina_l428_42854


namespace bicycle_race_finishers_l428_42821

theorem bicycle_race_finishers :
  let initial_racers : ℕ := 50
  let joined_racers : ℕ := 30
  let dropped_racers : ℕ := 30
  let racers_after_joining := initial_racers + joined_racers
  let racers_after_doubling := 2 * racers_after_joining
  let finishers := racers_after_doubling - dropped_racers
  finishers = 130 :=
by sorry

end bicycle_race_finishers_l428_42821


namespace cone_angle_bisecting_volume_l428_42894

/-- 
Given a cone with the following properties:
- A perpendicular is dropped from the center of the base to the slant height
- This perpendicular rotates about the cone's axis
- The surface of rotation divides the cone's volume in half

The angle between the slant height and the axis is arccos(1 / 2^(1/4))
-/
theorem cone_angle_bisecting_volume (R h : ℝ) (hR : R > 0) (hh : h > 0) : 
  let α := Real.arccos ((1 : ℝ) / 2^(1/4))
  let V := (1/3) * π * R^2 * h
  let V_rotated := (1/3) * π * (R * (Real.cos α)^2)^2 * h
  V_rotated = (1/2) * V :=
by sorry

end cone_angle_bisecting_volume_l428_42894


namespace greatest_divisor_with_remainders_l428_42862

theorem greatest_divisor_with_remainders : Nat.gcd (3589 - 23) (5273 - 41) = 2 := by
  sorry

end greatest_divisor_with_remainders_l428_42862


namespace min_packs_for_100_cans_l428_42836

/-- Represents the number of cans in each pack size -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cans_in_pack (p : PackSize) : Nat :=
  match p with
  | PackSize.small => 8
  | PackSize.medium => 14
  | PackSize.large => 28

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def total_cans (c : PackCombination) : Nat :=
  c.small * cans_in_pack PackSize.small +
  c.medium * cans_in_pack PackSize.medium +
  c.large * cans_in_pack PackSize.large

/-- Calculates the total number of packs in a combination -/
def total_packs (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- Predicate to check if a combination is valid (exactly 100 cans) -/
def is_valid_combination (c : PackCombination) : Prop :=
  total_cans c = 100

/-- Theorem: The minimum number of packs to buy exactly 100 cans is 5 -/
theorem min_packs_for_100_cans :
  ∃ (c : PackCombination),
    is_valid_combination c ∧
    total_packs c = 5 ∧
    (∀ (c' : PackCombination), is_valid_combination c' → total_packs c' ≥ 5) :=
  sorry

end min_packs_for_100_cans_l428_42836


namespace y_increases_with_x_on_positive_slope_line_l428_42879

/-- Given two points on a line with a positive slope, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is less than the y-coordinate of the second point. -/
theorem y_increases_with_x_on_positive_slope_line 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 3 * x₁ + 4) 
  (h2 : y₂ = 3 * x₂ + 4) 
  (h3 : x₁ < x₂) : 
  y₁ < y₂ := by
  sorry

end y_increases_with_x_on_positive_slope_line_l428_42879


namespace sequence_equality_l428_42839

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end sequence_equality_l428_42839


namespace pyramid_height_l428_42816

theorem pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : base_perimeter = 40) (h_apex_dist : apex_to_vertex = 12) :
  let side_length := base_perimeter / 4
  let diagonal := side_length * Real.sqrt 2
  let half_diagonal := diagonal / 2
  Real.sqrt (apex_to_vertex ^ 2 - half_diagonal ^ 2) = Real.sqrt 94 := by
  sorry

end pyramid_height_l428_42816


namespace senior_employee_bonus_l428_42860

/-- Proves that the senior employee receives $3,100 given the conditions of the bonus distribution -/
theorem senior_employee_bonus (total_bonus : ℕ) (difference : ℕ) (senior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  senior_share = total_bonus - difference →
  2 * senior_share = total_bonus + difference →
  senior_share = 3100 := by
sorry

end senior_employee_bonus_l428_42860


namespace jacks_total_money_l428_42870

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end jacks_total_money_l428_42870


namespace mono_properties_l428_42805

/-- Represents a monomial with coefficient and variables --/
structure Monomial where
  coeff : ℤ
  vars : List (Char × ℕ)

/-- Calculate the degree of a monomial --/
def degree (m : Monomial) : ℕ :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

/-- The monomial -4mn^5 --/
def mono : Monomial :=
  { coeff := -4
    vars := [('m', 1), ('n', 5)] }

theorem mono_properties : (mono.coeff = -4) ∧ (degree mono = 6) := by
  sorry

end mono_properties_l428_42805


namespace percentage_sum_proof_l428_42806

theorem percentage_sum_proof : 
  ∃ (x : ℝ), x * 400 + 0.45 * 250 = 224.5 ∧ x = 0.28 := by
  sorry

end percentage_sum_proof_l428_42806


namespace square_diagonal_triangle_l428_42856

theorem square_diagonal_triangle (s : ℝ) (h : s = 12) :
  let square_side := s
  let triangle_leg := s
  let triangle_hypotenuse := s * Real.sqrt 2
  let triangle_area := (s^2) / 2
  (triangle_leg = 12 ∧ 
   triangle_hypotenuse = 12 * Real.sqrt 2 ∧ 
   triangle_area = 72) :=
by sorry

end square_diagonal_triangle_l428_42856


namespace sqrt_inequality_l428_42810

theorem sqrt_inequality (a : ℝ) (h : a > 2) : Real.sqrt (a + 2) + Real.sqrt (a - 2) < 2 * Real.sqrt a := by
  sorry

end sqrt_inequality_l428_42810


namespace first_term_value_l428_42833

/-- Given a sequence {aₙ} with sum Sₙ, prove that a₁ = 1/2 -/
theorem first_term_value (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, S n = (a 1 * (4^n - 1)) / 3) →   -- Condition 1
  a 4 = 32 →                             -- Condition 2
  a 1 = 1/2 :=                           -- Conclusion
by sorry

end first_term_value_l428_42833


namespace point_A_coordinates_l428_42888

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line x + y + 3 = 0 -/
def line (p : Point) : Prop :=
  p.x + p.y + 3 = 0

/-- Two points are symmetric about a line if their midpoint lies on the line
    and the line is perpendicular to the line segment connecting the points -/
def symmetric_about (a b : Point) : Prop :=
  let midpoint : Point := ⟨(a.x + b.x) / 2, (a.y + b.y) / 2⟩
  line midpoint ∧ (a.y - b.y) = (a.x - b.x)

/-- The main theorem -/
theorem point_A_coordinates :
  ∀ (A : Point),
    symmetric_about A ⟨1, 2⟩ →
    A.x = -5 ∧ A.y = -4 := by
  sorry

end point_A_coordinates_l428_42888


namespace ribbon_division_theorem_l428_42875

theorem ribbon_division_theorem (p q r s : ℝ) :
  p + q + r + s = 36 →
  (p + q) / 2 + (r + s) / 2 = 18 := by
  sorry

end ribbon_division_theorem_l428_42875


namespace daily_harvest_l428_42842

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

/-- Theorem stating that the total number of sacks harvested daily is 360 -/
theorem daily_harvest : total_sacks = 360 := by
  sorry

end daily_harvest_l428_42842


namespace pencil_cost_l428_42863

/-- Given a pen and a pencil where the pen costs half the price of the pencil,
    and their total cost is $12, prove that the pencil costs $8. -/
theorem pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = pencil_cost / 2 →
  pen_cost + pencil_cost = 12 →
  pencil_cost = 8 := by
sorry

end pencil_cost_l428_42863


namespace floor_times_self_110_l428_42812

theorem floor_times_self_110 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 110 ∧ x = 11 := by
  sorry

end floor_times_self_110_l428_42812


namespace probability_prime_sum_two_dice_l428_42885

def die_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (a b : ℕ) : Prop := is_prime (a + b)

def favorable_outcomes : ℕ := 23

def total_outcomes : ℕ := die_sides * die_sides

theorem probability_prime_sum_two_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 23 / 64 := by sorry

end probability_prime_sum_two_dice_l428_42885


namespace tuesday_wednesday_most_available_l428_42807

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

-- Define the students
inductive Student
| alice
| bob
| cindy
| david
| eva

-- Define the availability function
def availability (s : Student) (d : Day) : Bool :=
  match s, d with
  | Student.alice, Day.monday => false
  | Student.alice, Day.tuesday => true
  | Student.alice, Day.wednesday => false
  | Student.alice, Day.thursday => true
  | Student.alice, Day.friday => true
  | Student.alice, Day.saturday => false
  | Student.bob, Day.monday => true
  | Student.bob, Day.tuesday => false
  | Student.bob, Day.wednesday => true
  | Student.bob, Day.thursday => false
  | Student.bob, Day.friday => false
  | Student.bob, Day.saturday => true
  | Student.cindy, Day.monday => false
  | Student.cindy, Day.tuesday => false
  | Student.cindy, Day.wednesday => true
  | Student.cindy, Day.thursday => false
  | Student.cindy, Day.friday => false
  | Student.cindy, Day.saturday => true
  | Student.david, Day.monday => true
  | Student.david, Day.tuesday => true
  | Student.david, Day.wednesday => false
  | Student.david, Day.thursday => false
  | Student.david, Day.friday => true
  | Student.david, Day.saturday => false
  | Student.eva, Day.monday => false
  | Student.eva, Day.tuesday => true
  | Student.eva, Day.wednesday => true
  | Student.eva, Day.thursday => true
  | Student.eva, Day.friday => false
  | Student.eva, Day.saturday => false

-- Count available students for a given day
def availableStudents (d : Day) : Nat :=
  (Student.alice :: Student.bob :: Student.cindy :: Student.david :: Student.eva :: []).filter (fun s => availability s d) |>.length

-- Theorem stating that Tuesday and Wednesday have the most available students
theorem tuesday_wednesday_most_available :
  (availableStudents Day.tuesday = availableStudents Day.wednesday) ∧
  (∀ d : Day, availableStudents d ≤ availableStudents Day.tuesday) :=
by sorry

end tuesday_wednesday_most_available_l428_42807


namespace green_balls_count_l428_42859

/-- Represents the contents and properties of a bag of colored balls. -/
structure BagOfBalls where
  total : Nat
  white : Nat
  yellow : Nat
  red : Nat
  purple : Nat
  prob_not_red_purple : Rat

/-- Calculates the number of green balls in the bag. -/
def green_balls (bag : BagOfBalls) : Nat :=
  bag.total - bag.white - bag.yellow - bag.red - bag.purple

/-- Theorem stating the number of green balls in the specific bag described in the problem. -/
theorem green_balls_count (bag : BagOfBalls) 
  (h1 : bag.total = 60)
  (h2 : bag.white = 22)
  (h3 : bag.yellow = 5)
  (h4 : bag.red = 6)
  (h5 : bag.purple = 9)
  (h6 : bag.prob_not_red_purple = 3/4) :
  green_balls bag = 18 := by
  sorry

#eval green_balls { 
  total := 60, 
  white := 22, 
  yellow := 5, 
  red := 6, 
  purple := 9, 
  prob_not_red_purple := 3/4 
}

end green_balls_count_l428_42859


namespace power_of_power_of_five_l428_42855

theorem power_of_power_of_five : (5^4)^2 = 390625 := by sorry

end power_of_power_of_five_l428_42855


namespace heartsuit_three_four_l428_42823

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end heartsuit_three_four_l428_42823


namespace gcd_lcm_equation_solutions_l428_42850

theorem gcd_lcm_equation_solutions :
  let S : Set (ℕ × ℕ) := {(8, 513), (513, 8), (215, 2838), (2838, 215),
                          (258, 1505), (1505, 258), (235, 2961), (2961, 235)}
  ∀ α β : ℕ, (Nat.gcd α β + Nat.lcm α β = 4 * (α + β) + 2021) ↔ (α, β) ∈ S :=
by sorry

end gcd_lcm_equation_solutions_l428_42850


namespace number_problem_l428_42868

theorem number_problem (x : ℝ) : (10 * x = x + 34.65) → x = 3.85 := by
  sorry

end number_problem_l428_42868


namespace not_p_sufficient_not_necessary_for_not_q_l428_42891

-- Define the sets corresponding to ¬p and ¬q
def not_p (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x > 1

-- Define the original conditions p and q
def p (x : ℝ) : Prop := 0 < x ∧ x < 2
def q (x : ℝ) : Prop := 1 / x ≥ 1

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ 
  (∃ x, not_q x ∧ ¬(not_p x)) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l428_42891


namespace divisibility_by_three_l428_42809

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
sorry

end divisibility_by_three_l428_42809


namespace mary_baking_cake_l428_42899

theorem mary_baking_cake (total_flour total_sugar remaining_flour_diff : ℕ) 
  (h1 : total_flour = 9)
  (h2 : total_sugar = 6)
  (h3 : remaining_flour_diff = 7) :
  total_sugar - (total_flour - remaining_flour_diff) = 4 := by
  sorry

end mary_baking_cake_l428_42899


namespace sqrt_equation_solutions_l428_42861

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8} = {39/4, 7/4} := by
  sorry

end sqrt_equation_solutions_l428_42861


namespace garage_sale_books_sold_l428_42884

def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

theorem garage_sale_books_sold :
  let initial_books : ℕ := 108
  let remaining_books : ℕ := 66
  books_sold initial_books remaining_books = 42 := by
  sorry

end garage_sale_books_sold_l428_42884


namespace quadratic_inequality_solution_set_l428_42808

theorem quadratic_inequality_solution_set (x : ℝ) : 
  (8 * x^2 + 10 * x - 16 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 3/4) := by
  sorry

end quadratic_inequality_solution_set_l428_42808


namespace inequality_proof_l428_42871

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end inequality_proof_l428_42871


namespace quadratic_equation_c_value_l428_42881

theorem quadratic_equation_c_value (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = (-8 + Real.sqrt 20) / 4 ∨ x = (-8 - Real.sqrt 20) / 4) →
  c = 5.5 := by
sorry

end quadratic_equation_c_value_l428_42881


namespace expand_product_l428_42890

theorem expand_product (x : ℝ) : 4 * (x + 3) * (x + 6) = 4 * x^2 + 36 * x + 72 := by
  sorry

end expand_product_l428_42890


namespace quadratic_inequality_l428_42822

theorem quadratic_inequality (x : ℝ) : 
  (2 * x^2 - 5 * x - 12 > 0 ↔ x < -3/2 ∨ x > 4) ∧
  (2 * x^2 - 5 * x - 12 < 0 ↔ -3/2 < x ∧ x < 4) :=
by sorry

end quadratic_inequality_l428_42822


namespace sin_negative_31pi_over_6_l428_42832

theorem sin_negative_31pi_over_6 : Real.sin (-31 * π / 6) = 1 / 2 := by
  sorry

end sin_negative_31pi_over_6_l428_42832


namespace quadratic_function_theorem_l428_42858

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_theorem (a b c : ℝ) :
  (f a b c 0 = 0) →
  (∀ x, f a b c (x + 1) = f a b c x + x + 1) →
  (∀ x, f a b c x = x^2 / 2 + x / 2) :=
by sorry

end quadratic_function_theorem_l428_42858


namespace fruit_seller_inventory_l428_42843

theorem fruit_seller_inventory (apples oranges bananas pears grapes : ℕ) : 
  (apples - apples / 2 + 20 = 370) →
  (oranges - oranges * 35 / 100 = 195) →
  (bananas - bananas * 3 / 5 + 15 = 95) →
  (pears - pears * 45 / 100 = 50) →
  (grapes - grapes * 3 / 10 = 140) →
  (apples = 700 ∧ oranges = 300 ∧ bananas = 200 ∧ pears = 91 ∧ grapes = 200) :=
by sorry

end fruit_seller_inventory_l428_42843


namespace triangle_properties_l428_42827

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the following properties when certain conditions are met. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a - b = 2 →
  c = 4 →
  Real.sin A = 2 * Real.sin B →
  (a = 4 ∧ b = 2 ∧ Real.cos B = 7/8) ∧
  Real.sin (2*B - π/6) = (21 * Real.sqrt 5 - 17) / 64 := by
  sorry

end triangle_properties_l428_42827


namespace natasha_maria_earnings_l428_42828

theorem natasha_maria_earnings (t : ℚ) : 
  (t - 4) * (3 * t - 4) = (3 * t - 12) * (t + 2) → t = 20 / 11 := by
  sorry

end natasha_maria_earnings_l428_42828


namespace percent_increase_revenue_l428_42840

/-- Given two positive real numbers M and N representing revenues in millions for two consecutive years,
    this theorem states that the percent increase in revenue relative to the sum of the revenues of both years
    is equal to 100 * (M - N) / (M + N). -/
theorem percent_increase_revenue (M N : ℝ) (hM : M > 0) (hN : N > 0) :
  (M - N) / (M + N) * 100 = 100 * (M - N) / (M + N) := by sorry

end percent_increase_revenue_l428_42840


namespace complex_modulus_equality_not_implies_square_equality_l428_42864

theorem complex_modulus_equality_not_implies_square_equality :
  ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end complex_modulus_equality_not_implies_square_equality_l428_42864


namespace hyperbola_focal_length_l428_42818

/-- The focal length of a hyperbola with given properties -/
theorem hyperbola_focal_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := 2  -- eccentricity
  let d := Real.sqrt 3  -- distance from focus to asymptote
  2 * Real.sqrt (a^2 + b^2) = 4 :=
by sorry

end hyperbola_focal_length_l428_42818


namespace jakes_initial_money_l428_42834

theorem jakes_initial_money (M : ℝ) : 
  (M - 2800 - (M - 2800) / 2) * 3 / 4 = 825 → M = 5000 := by
  sorry

end jakes_initial_money_l428_42834


namespace z_less_than_y_l428_42837

/-- 
Given:
- w is 40% less than u, so w = 0.6u
- u is 40% less than y, so u = 0.6y
- z is greater than w by 50% of w, so z = 1.5w

Prove that z is 46% less than y, which means z = 0.54y
-/
theorem z_less_than_y (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by
  sorry

end z_less_than_y_l428_42837


namespace certain_number_problem_l428_42880

theorem certain_number_problem : 
  ∃ N : ℕ, (N / 5 + N + 5 = 65) ∧ (N = 50) :=
by sorry

end certain_number_problem_l428_42880


namespace streetlights_per_square_l428_42838

theorem streetlights_per_square 
  (total_streetlights : ℕ) 
  (num_squares : ℕ) 
  (unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : num_squares = 15) 
  (h3 : unused_streetlights = 20) : 
  (total_streetlights - unused_streetlights) / num_squares = 12 :=
by
  sorry

end streetlights_per_square_l428_42838


namespace equation_solution_l428_42851

theorem equation_solution :
  let f (x : ℝ) := x^2 * (x - 2) - (4 * x^2 + 4)
  ∀ x : ℝ, x ≠ 2 → (f x = 0 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by sorry

end equation_solution_l428_42851


namespace janet_dermatologist_distance_l428_42829

def dermatologist_distance (x : ℝ) := x
def gynecologist_distance : ℝ := 50
def car_efficiency : ℝ := 20
def gas_used : ℝ := 8

theorem janet_dermatologist_distance :
  ∃ x : ℝ, 
    dermatologist_distance x = 30 ∧ 
    2 * dermatologist_distance x + 2 * gynecologist_distance = car_efficiency * gas_used :=
by sorry

end janet_dermatologist_distance_l428_42829


namespace triangle_angle_sum_l428_42869

theorem triangle_angle_sum (x : ℝ) : 
  36 + 90 + x = 180 → x = 54 := by
sorry

end triangle_angle_sum_l428_42869


namespace consecutive_numbers_sum_l428_42889

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → 
  (n + 2) + (n + 3) + (n + 4) = 66 := by
sorry

end consecutive_numbers_sum_l428_42889


namespace statement_equivalence_l428_42882

theorem statement_equivalence (P Q R : Prop) :
  (P → (Q ∧ ¬R)) ↔ ((¬Q ∨ R) → ¬P) := by
  sorry

end statement_equivalence_l428_42882


namespace function_difference_inequality_l428_42825

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x)
  {a b : ℝ}
  (hab : a > b) :
  f a - f b > g a - g b :=
sorry

end function_difference_inequality_l428_42825
