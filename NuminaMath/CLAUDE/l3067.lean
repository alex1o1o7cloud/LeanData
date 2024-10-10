import Mathlib

namespace base_seven_digits_of_1200_l3067_306773

theorem base_seven_digits_of_1200 : ∃ n : ℕ, (7^(n-1) ≤ 1200 ∧ 1200 < 7^n) ∧ n = 4 := by
  sorry

end base_seven_digits_of_1200_l3067_306773


namespace solution_exists_l3067_306706

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ log10 (x^2 / y^3) = 1 ∧ log10 (x^2 * y^3) = 7

-- Theorem statement
theorem solution_exists :
  ∃ x y : ℝ, equation_system x y ∧ (x = 100 ∨ x = -100) ∧ y = 10 :=
by sorry

end solution_exists_l3067_306706


namespace count_possible_D_values_l3067_306789

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if a list of digits are all distinct -/
def all_distinct (digits : List Digit) : Prop :=
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

/-- Converts a list of digits to a natural number -/
def to_nat (digits : List Digit) : ℕ :=
  digits.foldl (λ acc d => 10 * acc + d.val) 0

/-- The main theorem -/
theorem count_possible_D_values :
  ∃ (possible_D_values : Finset Digit),
    (∀ A B C E D : Digit,
      all_distinct [A, B, C, E, D] →
      to_nat [A, B, C, E, B] + to_nat [B, C, E, D, A] = to_nat [D, B, D, D, D] →
      D ∈ possible_D_values) ∧
    possible_D_values.card = 7 := by
  sorry

end count_possible_D_values_l3067_306789


namespace bridge_length_bridge_length_proof_l3067_306770

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that the bridge length is 215 meters -/
theorem bridge_length_proof :
  bridge_length 160 45 30 = 215 := by
  sorry

end bridge_length_bridge_length_proof_l3067_306770


namespace additional_cost_proof_l3067_306751

/-- Additional cost per international letter --/
def additional_cost_per_letter : ℚ := 55 / 100

/-- Number of letters --/
def num_letters : ℕ := 4

/-- Number of domestic letters --/
def num_domestic : ℕ := 2

/-- Number of international letters --/
def num_international : ℕ := 2

/-- Domestic postage rate per letter --/
def domestic_rate : ℚ := 108 / 100

/-- Weight of first international letter (in grams) --/
def weight_letter1 : ℕ := 25

/-- Weight of second international letter (in grams) --/
def weight_letter2 : ℕ := 45

/-- Rate for Country A for letters below 50 grams (per gram) --/
def rate_A_below50 : ℚ := 5 / 100

/-- Rate for Country B for letters below 50 grams (per gram) --/
def rate_B_below50 : ℚ := 4 / 100

/-- Total postage paid --/
def total_paid : ℚ := 630 / 100

theorem additional_cost_proof :
  let domestic_cost := num_domestic * domestic_rate
  let international_cost1 := weight_letter1 * rate_A_below50
  let international_cost2 := weight_letter2 * rate_B_below50
  let total_calculated := domestic_cost + international_cost1 + international_cost2
  let additional_total := total_paid - total_calculated
  additional_total / num_international = additional_cost_per_letter := by
  sorry

end additional_cost_proof_l3067_306751


namespace a_closed_form_l3067_306778

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 5 * a (n + 1) - 6 * a n + 4^(n + 1)

theorem a_closed_form (n : ℕ) :
  a n = 2^(n + 1) - 3^(n + 1) + 2 * 4^n :=
by sorry

end a_closed_form_l3067_306778


namespace library_digital_format_l3067_306725

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for a book being available in digital format
variable (isDigital : Book → Prop)

-- Define the theorem
theorem library_digital_format (h : ¬∀ (b : Book), isDigital b) :
  (∃ (b : Book), ¬isDigital b) ∧ (¬∀ (b : Book), isDigital b) := by
  sorry

end library_digital_format_l3067_306725


namespace no_inscribed_circle_l3067_306713

/-- A pentagon is represented by a list of its side lengths -/
def Pentagon := List ℝ

/-- Check if a list represents a valid pentagon with sides 1, 2, 5, 6, 7 -/
def isValidPentagon (p : Pentagon) : Prop :=
  p.length = 5 ∧ p.toFinset = {1, 2, 5, 6, 7}

/-- Sum of three elements in a list -/
def sumThree (l : List ℝ) (i j k : ℕ) : ℝ :=
  (l.get? i).getD 0 + (l.get? j).getD 0 + (l.get? k).getD 0

/-- Check if the sum of two non-adjacent sides is greater than or equal to
    the sum of the remaining three sides -/
def hasInvalidPair (p : Pentagon) : Prop :=
  (p.get? 0).getD 0 + (p.get? 2).getD 0 ≥ sumThree p 1 3 4 ∨
  (p.get? 0).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 1 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 3).getD 0 ≥ sumThree p 0 2 4 ∨
  (p.get? 1).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 2 3 ∨
  (p.get? 2).getD 0 + (p.get? 4).getD 0 ≥ sumThree p 0 1 3

theorem no_inscribed_circle (p : Pentagon) (h : isValidPentagon p) :
  hasInvalidPair p := by
  sorry


end no_inscribed_circle_l3067_306713


namespace bill_difference_l3067_306740

/-- The number of $20 bills Mandy has -/
def mandy_twenty_bills : ℕ := 3

/-- The number of $50 bills Manny has -/
def manny_fifty_bills : ℕ := 2

/-- The value of a $20 bill -/
def twenty_bill_value : ℕ := 20

/-- The value of a $50 bill -/
def fifty_bill_value : ℕ := 50

/-- The value of a $10 bill -/
def ten_bill_value : ℕ := 10

/-- Theorem stating the difference in $10 bills between Manny and Mandy -/
theorem bill_difference :
  (manny_fifty_bills * fifty_bill_value) / ten_bill_value -
  (mandy_twenty_bills * twenty_bill_value) / ten_bill_value = 4 := by
  sorry

end bill_difference_l3067_306740


namespace no_primes_divisible_by_25_l3067_306759

theorem no_primes_divisible_by_25 : ∀ p : ℕ, Nat.Prime p → ¬(25 ∣ p) := by
  sorry

end no_primes_divisible_by_25_l3067_306759


namespace inequality_proof_l3067_306755

theorem inequality_proof (s x y z : ℝ) 
  (hs : s > 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h : s * x > z * y) : 
  ¬ (
    (x > z ∧ -x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z ∧ s > z / x) ∨
    (x > z ∧ -x > -z ∧ s < y / x) ∨
    (x > z ∧ s > z / x ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z) ∨
    (x > z ∧ s > z / x) ∨
    (x > z ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x) ∨
    (-x > -z ∧ s < y / x) ∨
    (s > z / x ∧ s < y / x) ∨
    (x > z) ∨
    (-x > -z) ∨
    (s > z / x) ∨
    (s < y / x)
  ) :=
sorry

end inequality_proof_l3067_306755


namespace wrench_can_turn_bolt_l3067_306709

/-- Represents a wrench with a regular hexagonal shape -/
structure Wrench where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a bolt with a square head -/
structure Bolt where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Condition for a wrench to turn a bolt -/
def canTurn (w : Wrench) (b : Bolt) : Prop :=
  Real.sqrt 3 / Real.sqrt 2 < b.sideLength / w.sideLength ∧ 
  b.sideLength / w.sideLength ≤ 3 - Real.sqrt 3

/-- Theorem stating the condition for a wrench to turn a bolt -/
theorem wrench_can_turn_bolt (w : Wrench) (b : Bolt) : 
  canTurn w b ↔ 
    (∃ (x : ℝ), b.sideLength = x * w.sideLength ∧ 
      Real.sqrt 3 / Real.sqrt 2 < x ∧ x ≤ 3 - Real.sqrt 3) :=
sorry

end wrench_can_turn_bolt_l3067_306709


namespace smallest_n_congruence_l3067_306799

theorem smallest_n_congruence (n : ℕ) : n = 3 ↔ (
  n > 0 ∧
  17 * n ≡ 136 [ZMOD 5] ∧
  ∀ m : ℕ, m > 0 → m < n → ¬(17 * m ≡ 136 [ZMOD 5])
) := by sorry

end smallest_n_congruence_l3067_306799


namespace water_polo_team_selection_l3067_306726

theorem water_polo_team_selection (total_members : ℕ) (starting_team_size : ℕ) (goalie_count : ℕ) :
  total_members = 18 →
  starting_team_size = 8 →
  goalie_count = 1 →
  (total_members.choose goalie_count) * ((total_members - goalie_count).choose (starting_team_size - goalie_count)) = 222768 :=
by sorry

end water_polo_team_selection_l3067_306726


namespace set_intersection_empty_l3067_306741

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- State the theorem
theorem set_intersection_empty (a : ℝ) : 
  (A a ∩ B = ∅) ↔ ((1/2 ≤ a ∧ a ≤ 2) ∨ a > 3) := by sorry

end set_intersection_empty_l3067_306741


namespace dividing_line_sum_of_squares_l3067_306768

/-- A circle in the first quadrant of the coordinate plane -/
structure Circle where
  diameter : ℝ
  center : ℝ × ℝ

/-- The region R formed by the union of ten circles -/
def region_R : Set (ℝ × ℝ) :=
  sorry

/-- The line m with slope -1 that divides region_R into two equal areas -/
structure DividingLine where
  a : ℕ
  b : ℕ
  c : ℕ
  slope_neg_one : a = b
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  divides_equally : sorry

/-- Theorem stating that for the line m, a^2 + b^2 + c^2 = 6 -/
theorem dividing_line_sum_of_squares (m : DividingLine) :
  m.a^2 + m.b^2 + m.c^2 = 6 := by
  sorry

end dividing_line_sum_of_squares_l3067_306768


namespace percentage_problem_l3067_306758

theorem percentage_problem (N P : ℝ) : 
  N = 150 → N = (P / 100) * N + 126 → P = 16 := by
  sorry

end percentage_problem_l3067_306758


namespace poultry_farm_solution_l3067_306705

/-- Represents the poultry farm problem --/
def poultry_farm_problem (initial_chickens initial_guinea_fowls : ℕ)
  (daily_loss_chickens daily_loss_turkeys daily_loss_guinea_fowls : ℕ)
  (days : ℕ) (total_birds_left : ℕ) : Prop :=
  let initial_turkeys := 200
  let total_initial_birds := initial_chickens + initial_turkeys + initial_guinea_fowls
  let total_loss := (daily_loss_chickens + daily_loss_turkeys + daily_loss_guinea_fowls) * days
  total_initial_birds - total_loss = total_birds_left

/-- Theorem stating the solution to the poultry farm problem --/
theorem poultry_farm_solution :
  poultry_farm_problem 300 80 20 8 5 7 349 := by
  sorry

#check poultry_farm_solution

end poultry_farm_solution_l3067_306705


namespace square_rhombus_diagonal_distinction_l3067_306762

/-- A quadrilateral with four equal sides -/
structure Rhombus :=
  (side_length : ℝ)
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

/-- A square is a rhombus with equal diagonals -/
structure Square extends Rhombus :=
  (diagonals_equal : diagonal1 = diagonal2)

/-- Theorem stating that squares have equal diagonals, but rhombuses don't necessarily have this property -/
theorem square_rhombus_diagonal_distinction :
  ∃ (s : Square) (r : Rhombus), s.diagonal1 = s.diagonal2 ∧ r.diagonal1 ≠ r.diagonal2 :=
sorry

end square_rhombus_diagonal_distinction_l3067_306762


namespace solve_for_y_l3067_306766

theorem solve_for_y (x y : ℤ) (h1 : x^2 + x + 6 = y - 6) (h2 : x = -5) : y = 32 := by
  sorry

end solve_for_y_l3067_306766


namespace amelia_tuesday_distance_l3067_306771

/-- The distance Amelia drove on Tuesday -/
def tuesday_distance (total_distance monday_distance remaining_distance : ℕ) : ℕ :=
  total_distance - (monday_distance + remaining_distance)

theorem amelia_tuesday_distance :
  tuesday_distance 8205 907 6716 = 582 := by
  sorry

end amelia_tuesday_distance_l3067_306771


namespace crow_probability_l3067_306746

/-- Represents the number of crows of each color on each tree -/
structure CrowDistribution where
  birch_white : ℕ
  birch_black : ℕ
  oak_white : ℕ
  oak_black : ℕ

/-- The probability that the number of white crows on the birch tree remains the same -/
def prob_same (d : CrowDistribution) : ℚ :=
  (d.birch_black * (d.oak_black + 1) + d.birch_white * (d.oak_white + 1)) / (50 * 51)

/-- The probability that the number of white crows on the birch tree changes -/
def prob_change (d : CrowDistribution) : ℚ :=
  (d.birch_black * d.oak_white + d.birch_white * d.oak_black) / (50 * 51)

theorem crow_probability (d : CrowDistribution) 
  (h1 : d.birch_white + d.birch_black = 50)
  (h2 : d.oak_white + d.oak_black = 50)
  (h3 : d.birch_white > 0)
  (h4 : d.oak_white > 0)
  (h5 : d.birch_black ≥ d.birch_white)
  (h6 : d.oak_black ≥ d.oak_white ∨ d.oak_black + 1 = d.oak_white) :
  prob_same d > prob_change d := by
  sorry

end crow_probability_l3067_306746


namespace income_calculation_l3067_306700

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 9 →  -- income and expenditure ratio is 9:8
  income = expenditure + savings → -- income equals expenditure plus savings
  savings = 4000 → -- savings are 4000
  income = 36000 := by -- prove that income is 36000
sorry

end income_calculation_l3067_306700


namespace factorization_of_2x_squared_minus_18_l3067_306743

theorem factorization_of_2x_squared_minus_18 (x : ℝ) : 2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by
  sorry

end factorization_of_2x_squared_minus_18_l3067_306743


namespace lines_coincide_by_rotation_l3067_306707

/-- Two lines that intersect can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (P : ℝ × ℝ), P.1 * Real.sin α = P.2 ∧ 
  ∃ (θ : ℝ), ∀ (x y : ℝ), 
    y = x * Real.sin α ↔ 
    (x - P.1) * Real.cos θ - (y - P.2) * Real.sin θ = 
    ((x - P.1) * Real.sin θ + (y - P.2) * Real.cos θ) * 2 + c :=
sorry

end lines_coincide_by_rotation_l3067_306707


namespace sphere_volume_ratio_l3067_306734

theorem sphere_volume_ratio (S₁ S₂ V₁ V₂ : ℝ) (h_positive : S₁ > 0 ∧ S₂ > 0) (h_surface_ratio : S₁ / S₂ = 1 / 3) : 
  V₁ / V₂ = 1 / (3 * Real.sqrt 3) := by
  sorry

end sphere_volume_ratio_l3067_306734


namespace martha_cakes_l3067_306701

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : ℝ) (cakes_per_child : ℝ) : ℝ :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3 18 = 54 := by
  sorry

end martha_cakes_l3067_306701


namespace unique_a_for_linear_equation_l3067_306736

def is_linear_equation (a : ℝ) : Prop :=
  (|a| - 1 = 1) ∧ (a - 2 ≠ 0)

theorem unique_a_for_linear_equation :
  ∃! a : ℝ, is_linear_equation a ∧ a = -2 :=
sorry

end unique_a_for_linear_equation_l3067_306736


namespace equation_one_solutions_l3067_306764

theorem equation_one_solutions :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 :=
by sorry

end equation_one_solutions_l3067_306764


namespace min_disks_required_l3067_306780

def disk_capacity : ℝ := 1.44

def file_count : ℕ := 40

def file_sizes : List ℝ := [0.95, 0.95, 0.95, 0.95, 0.95] ++ 
                           List.replicate 15 0.65 ++ 
                           List.replicate 20 0.45

def total_file_size : ℝ := file_sizes.sum

theorem min_disks_required : 
  ∀ (arrangement : List (List ℝ)),
    (arrangement.length < 17 → 
     ∃ (disk : List ℝ), disk ∈ arrangement ∧ disk.sum > disk_capacity) ∧
    (∃ (valid_arrangement : List (List ℝ)), 
      valid_arrangement.length = 17 ∧
      valid_arrangement.join.sum = total_file_size ∧
      ∀ (disk : List ℝ), disk ∈ valid_arrangement → disk.sum ≤ disk_capacity) :=
by sorry

end min_disks_required_l3067_306780


namespace dilation_problem_l3067_306745

def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

theorem dilation_problem : 
  let center := (0 : ℂ) + 5*I
  let scale := (3 : ℂ)
  let z := (3 : ℂ) + 2*I
  dilation center scale z = (9 : ℂ) - 4*I := by
  sorry

end dilation_problem_l3067_306745


namespace profit_percentage_doubling_l3067_306721

theorem profit_percentage_doubling (cost_price : ℝ) (original_profit_percentage : ℝ) 
  (h1 : original_profit_percentage = 60) :
  let original_selling_price := cost_price * (1 + original_profit_percentage / 100)
  let new_selling_price := 2 * original_selling_price
  let new_profit := new_selling_price - cost_price
  let new_profit_percentage := (new_profit / cost_price) * 100
  new_profit_percentage = 220 := by
  sorry

end profit_percentage_doubling_l3067_306721


namespace equation_solutions_l3067_306787

def equation (x y : ℝ) : Prop :=
  x^2 + x*y + y^2 + 2*x - 3*y - 3 = 0

def solution_set : Set (ℝ × ℝ) :=
  {(1, 2), (1, 0), (-5, 2), (-5, 6), (-3, 0)}

theorem equation_solutions :
  (∀ (x y : ℝ), (x, y) ∈ solution_set ↔ equation x y) ∧
  equation 1 2 :=
sorry

end equation_solutions_l3067_306787


namespace coaches_average_age_l3067_306750

theorem coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (num_girls : ℕ) 
  (num_boys : ℕ) 
  (num_coaches : ℕ) 
  (girls_average : ℕ) 
  (boys_average : ℕ) 
  (h1 : total_members = 50)
  (h2 : overall_average = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_coaches = 5)
  (h6 : girls_average = 16)
  (h7 : boys_average = 17)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * overall_average - num_girls * girls_average - num_boys * boys_average) / num_coaches = 32 := by
  sorry

end coaches_average_age_l3067_306750


namespace sum_58_29_rounded_to_nearest_ten_l3067_306765

/-- Rounds a number to the nearest multiple of 10 -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 58 and 29 rounded to the nearest ten is 90 -/
theorem sum_58_29_rounded_to_nearest_ten :
  roundToNearestTen (58 + 29) = 90 := by
  sorry

end sum_58_29_rounded_to_nearest_ten_l3067_306765


namespace floor_plus_one_l3067_306785

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the ceiling function
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

-- Statement to prove
theorem floor_plus_one (x : ℝ) : floor (x + 1) = floor x + 1 := by
  sorry

end floor_plus_one_l3067_306785


namespace pet_shop_grooming_l3067_306753

/-- The pet shop grooming problem -/
theorem pet_shop_grooming (poodle_time terrier_time total_time : ℕ) 
  (terrier_count : ℕ) (poodle_count : ℕ) : 
  poodle_time = 30 →
  terrier_time = poodle_time / 2 →
  terrier_count = 8 →
  total_time = 210 →
  poodle_count * poodle_time + terrier_count * terrier_time = total_time →
  poodle_count = 3 := by
  sorry

end pet_shop_grooming_l3067_306753


namespace arithmetic_calculation_l3067_306782

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end arithmetic_calculation_l3067_306782


namespace intersection_point_parallel_through_point_perpendicular_with_y_intercept_l3067_306749

-- Define the lines l₁ and l₂
def l₁ (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l₂ (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Scenario 1: l₁ and l₂ intersect at point P(m, 1)
theorem intersection_point (m n : ℝ) : 
  (l₁ m n m 1 ∧ l₂ m m 1) → (m = 1/3 ∧ n = -73/9) := by sorry

-- Scenario 2: l₁ is parallel to l₂ and passes through (3, -1)
theorem parallel_through_point (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y ↔ l₂ m x y) ∧ l₁ m n 3 (-1) → 
  ((m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20)) := by sorry

-- Scenario 3: l₁ is perpendicular to l₂ and y-intercept of l₁ is -1
theorem perpendicular_with_y_intercept (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y → l₂ m x y → m * m = -1) ∧ l₁ m n 0 (-1) →
  (m = 0 ∧ n = 8) := by sorry

end intersection_point_parallel_through_point_perpendicular_with_y_intercept_l3067_306749


namespace camping_trip_percentage_l3067_306718

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (students_not_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (18 * total_students) / 100)
  (h2 : students_not_more_than_100 = (75 * (students_more_than_100 + students_not_more_than_100)) / 100) :
  (students_more_than_100 + students_not_more_than_100) * 100 / total_students = 72 :=
by sorry

end camping_trip_percentage_l3067_306718


namespace division_multiplication_error_percentage_l3067_306763

theorem division_multiplication_error_percentage (x : ℝ) (h : x > 0) :
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 0.5 ∧
  (|(x / 8 - 8 * x)| / (8 * x)) * 100 = 98 + ε := by
  sorry

end division_multiplication_error_percentage_l3067_306763


namespace polynomial_must_be_constant_l3067_306742

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Sum of decimal digits of an integer's absolute value -/
def sumDecimalDigits (n : ℤ) : ℕ :=
  sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Predicate for Fibonacci numbers -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem polynomial_must_be_constant (P : IntPolynomial) :
  (∀ n : ℕ, n > 0 → ¬isFibonacci (sumDecimalDigits (P.eval n))) →
  P.degree = 0 := by
  sorry

end polynomial_must_be_constant_l3067_306742


namespace zeros_and_range_of_f_l3067_306774

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

theorem zeros_and_range_of_f (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 (-2) x = 0 ↔ x = 3 ∨ x = -1) ∧
  (∀ b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1) :=
sorry

end zeros_and_range_of_f_l3067_306774


namespace seating_theorem_standing_theorem_distribution_theorem_l3067_306747

/- Problem 1 -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

theorem seating_theorem : seating_arrangements 8 3 = 24 := by
  sorry

/- Problem 2 -/
def standing_arrangements (total_people : ℕ) (condition : Bool) : ℕ :=
  sorry

theorem standing_theorem : standing_arrangements 5 true = 60 := by
  sorry

/- Problem 3 -/
def distribute_spots (total_spots : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem distribution_theorem : distribute_spots 10 7 = 84 := by
  sorry

end seating_theorem_standing_theorem_distribution_theorem_l3067_306747


namespace solution_set_of_inequality_l3067_306767

theorem solution_set_of_inequality (x : ℝ) :
  (x - 50) * (60 - x) > 0 ↔ x ∈ Set.Ioo 50 60 := by sorry

end solution_set_of_inequality_l3067_306767


namespace complex_equation_imag_part_l3067_306711

theorem complex_equation_imag_part :
  ∀ z : ℂ, z * (1 + Complex.I) = (3 : ℂ) + 2 * Complex.I →
  Complex.im z = -1/2 := by
  sorry

end complex_equation_imag_part_l3067_306711


namespace original_price_calculation_l3067_306733

/-- Given an article sold for $35 with a 75% gain, prove that the original price was $20. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) 
  (h1 : sale_price = 35)
  (h2 : gain_percent = 75) :
  ∃ (original_price : ℝ), 
    sale_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 20 := by
  sorry

end original_price_calculation_l3067_306733


namespace daps_to_dips_l3067_306708

/-- Representation of the currency conversion problem -/
structure Currency where
  daps : ℚ
  dops : ℚ
  dips : ℚ

/-- The conversion rates between currencies -/
def conversion_rates : Currency → Prop
  | c => c.daps * 4 = c.dops * 5 ∧ c.dops * 10 = c.dips * 4

/-- Theorem stating the equivalence of 125 daps to 50 dips -/
theorem daps_to_dips (c : Currency) (h : conversion_rates c) : 
  c.daps * 50 = c.dips * 125 := by
  sorry

end daps_to_dips_l3067_306708


namespace union_A_B_when_m_2_intersection_A_B_empty_iff_l3067_306760

-- Define sets A and B
def A : Set ℝ := {x | (4 : ℝ) / (x + 1) > 1}
def B (m : ℝ) : Set ℝ := {x | (x - m - 4) * (x - m + 1) > 0}

-- Part 1
theorem union_A_B_when_m_2 : A ∪ B 2 = {x : ℝ | x < 3 ∨ x > 6} := by sorry

-- Part 2
theorem intersection_A_B_empty_iff (m : ℝ) : A ∩ B m = ∅ ↔ -1 ≤ m ∧ m ≤ 0 := by sorry

end union_A_B_when_m_2_intersection_A_B_empty_iff_l3067_306760


namespace intersection_one_element_l3067_306716

theorem intersection_one_element (a : ℝ) : 
  let A : Set ℝ := {1, a, 5}
  let B : Set ℝ := {2, a^2 + 1}
  (∃! x, x ∈ A ∩ B) → a = 0 ∨ a = -2 := by
sorry

end intersection_one_element_l3067_306716


namespace composite_sum_l3067_306779

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : ∃ k : ℕ, x^2 + x*y - y = k^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b := by
  sorry

end composite_sum_l3067_306779


namespace minimum_force_to_submerge_cube_l3067_306790

-- Define constants
def cube_volume : Real := 10e-6  -- 10 cm³ converted to m³
def cube_density : Real := 400   -- kg/m³
def water_density : Real := 1000 -- kg/m³
def gravity : Real := 10         -- m/s²

-- Define the minimum force function
def minimum_submerge_force (v : Real) (ρ_cube : Real) (ρ_water : Real) (g : Real) : Real :=
  (ρ_water - ρ_cube) * v * g

-- Theorem statement
theorem minimum_force_to_submerge_cube :
  minimum_submerge_force cube_volume cube_density water_density gravity = 0.06 := by
  sorry

end minimum_force_to_submerge_cube_l3067_306790


namespace inverse_f_at_seven_l3067_306727

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem inverse_f_at_seven (x : ℝ) : f x = 7 → x = 101 := by
  sorry

end inverse_f_at_seven_l3067_306727


namespace min_max_values_l3067_306769

theorem min_max_values (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → Real.sqrt a + Real.sqrt b ≥ Real.sqrt x + Real.sqrt y) ∧
  (∀ x y, x + y = 1 ∧ x > 0 ∧ y > 0 → 1 / (a + 2*b) + 1 / (2*a + b) ≤ 1 / (x + 2*y) + 1 / (2*x + y)) ∧
  a^2 + b^2 = 1/2 ∧
  Real.sqrt a + Real.sqrt b = Real.sqrt 2 ∧
  1 / (a + 2*b) + 1 / (2*a + b) = 4/3 :=
by sorry

end min_max_values_l3067_306769


namespace triangle_area_l3067_306788

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is (18 + 8√3) / 25 when a = √3, c = 8/5, and A = π/3 -/
theorem triangle_area (a b c A B C : ℝ) : 
  a = Real.sqrt 3 →
  c = 8 / 5 →
  A = π / 3 →
  (1 / 2) * a * c * Real.sin B = (18 + 8 * Real.sqrt 3) / 25 := by
  sorry

end triangle_area_l3067_306788


namespace appropriate_presentation_lengths_l3067_306776

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration : Set ℝ := { x | 20 ≤ x ∧ x ≤ 40 }

/-- The ideal speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the range of appropriate word counts for a presentation -/
def AppropriateWordCount : Set ℕ :=
  { w | ∃ (d : ℝ), d ∈ PresentationDuration ∧ 
    (↑w : ℝ) ≥ 20 * SpeechRate ∧ (↑w : ℝ) ≤ 40 * SpeechRate }

/-- Theorem stating that 2700, 3900, and 4500 words are appropriate presentation lengths -/
theorem appropriate_presentation_lengths :
  2700 ∈ AppropriateWordCount ∧
  3900 ∈ AppropriateWordCount ∧
  4500 ∈ AppropriateWordCount :=
by sorry

end appropriate_presentation_lengths_l3067_306776


namespace expression_value_l3067_306798

theorem expression_value : ∀ x y : ℝ, x = 2 ∧ y = 3 → x^3 + y^2 * (x^2 * y) = 116 := by
  sorry

end expression_value_l3067_306798


namespace sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l3067_306744

-- Part 1
theorem sqrt_expression_equals_sqrt_3 :
  Real.sqrt 12 - Real.sqrt 48 + 9 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

-- Part 2
theorem sqrt_difference_times_sqrt_3_equals_neg_sqrt_6 :
  (Real.sqrt 8 - Real.sqrt 18) * Real.sqrt 3 = -Real.sqrt 6 := by sorry

end sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l3067_306744


namespace mark_spending_l3067_306724

/-- Represents the grocery items Mark buys -/
inductive GroceryItem
  | Apple
  | Bread
  | Cheese
  | Cereal

/-- Represents Mark's grocery shopping trip -/
structure GroceryShopping where
  prices : GroceryItem → ℕ
  quantities : GroceryItem → ℕ
  appleBuyOneGetOneFree : Bool
  couponValue : ℕ
  couponThreshold : ℕ

def calculateTotalSpending (shopping : GroceryShopping) : ℕ :=
  sorry

theorem mark_spending (shopping : GroceryShopping) 
  (h1 : shopping.prices GroceryItem.Apple = 2)
  (h2 : shopping.prices GroceryItem.Bread = 3)
  (h3 : shopping.prices GroceryItem.Cheese = 6)
  (h4 : shopping.prices GroceryItem.Cereal = 5)
  (h5 : shopping.quantities GroceryItem.Apple = 4)
  (h6 : shopping.quantities GroceryItem.Bread = 5)
  (h7 : shopping.quantities GroceryItem.Cheese = 3)
  (h8 : shopping.quantities GroceryItem.Cereal = 4)
  (h9 : shopping.appleBuyOneGetOneFree = true)
  (h10 : shopping.couponValue = 10)
  (h11 : shopping.couponThreshold = 50)
  : calculateTotalSpending shopping = 47 := by
  sorry

end mark_spending_l3067_306724


namespace smallest_solution_abs_equation_l3067_306757

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, x * |x| = 3 * x + 4 → x ≥ 4 :=
by
  sorry

end smallest_solution_abs_equation_l3067_306757


namespace multiply_32519_9999_l3067_306791

theorem multiply_32519_9999 : 32519 * 9999 = 324857481 := by
  sorry

end multiply_32519_9999_l3067_306791


namespace arithmetic_sequence_problem_l3067_306738

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a₁ d 5 = a₁^2 →
  a₁ * arithmetic_sequence a₁ d 21 = (arithmetic_sequence a₁ d 5)^2 →
  a₁ = 4 := by
  sorry


end arithmetic_sequence_problem_l3067_306738


namespace average_age_proof_l3067_306720

def john_age (mary_age : ℕ) : ℕ := 2 * mary_age

def tonya_age : ℕ := 60

theorem average_age_proof (mary_age : ℕ) (h1 : john_age mary_age = tonya_age / 2) :
  (mary_age + john_age mary_age + tonya_age) / 3 = 35 := by
  sorry

end average_age_proof_l3067_306720


namespace geometric_sequence_fourth_term_l3067_306728

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ ≠ 0)
  (h₂ : a₂ = 3 * a₁ + 3)
  (h₃ : a₃ = 6 * a₁ + 6)
  (h₄ : a₂^2 = a₁ * a₃)  -- Condition for geometric sequence
  : ∃ (r : ℝ), r ≠ 0 ∧ a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ r * a₃ = -24 :=
sorry

end geometric_sequence_fourth_term_l3067_306728


namespace polynomial_roots_arithmetic_progression_l3067_306761

/-- If a polynomial x^4 + jx^2 + kx + 256 has four distinct real roots in arithmetic progression, then j = -80 -/
theorem polynomial_roots_arithmetic_progression (j k : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 
    (∀ (x : ℝ), x^4 + j*x^2 + k*x + 256 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    (b - a = c - b) ∧ (c - b = d - c)) →
  j = -80 := by sorry

end polynomial_roots_arithmetic_progression_l3067_306761


namespace smallest_n_with_seven_in_squares_l3067_306704

def contains_seven (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + 7 * b ∧ b ≤ 9

theorem smallest_n_with_seven_in_squares : 
  (∀ m : ℕ, m < 26 → ¬(contains_seven (m^2) ∧ contains_seven ((m+1)^2))) ∧
  (contains_seven (26^2) ∧ contains_seven (27^2)) :=
sorry

end smallest_n_with_seven_in_squares_l3067_306704


namespace triangle_angle_measure_l3067_306777

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) ∧
  C = π/6 ∧
  a = 1 ∧
  b = Real.sqrt 3 →
  B = 2*π/3 := by sorry

end triangle_angle_measure_l3067_306777


namespace total_nails_needed_l3067_306715

def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

theorem total_nails_needed : nails_per_plank * number_of_planks = 32 := by
  sorry

end total_nails_needed_l3067_306715


namespace polynomial_division_remainder_l3067_306729

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 22 * X + 58 = (X - 6) * q + 34 := by sorry

end polynomial_division_remainder_l3067_306729


namespace children_education_expense_l3067_306797

def monthly_salary (saved_amount : ℚ) (savings_rate : ℚ) : ℚ :=
  saved_amount / savings_rate

def total_expenses (rent milk groceries petrol misc education : ℚ) : ℚ :=
  rent + milk + groceries + petrol + misc + education

theorem children_education_expense 
  (rent milk groceries petrol misc : ℚ)
  (savings_rate saved_amount : ℚ)
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : petrol = 2000)
  (h5 : misc = 5200)
  (h6 : savings_rate = 1/10)
  (h7 : saved_amount = 2300)
  : ∃ (education : ℚ), 
    education = 2500 ∧ 
    total_expenses rent milk groceries petrol misc education = 
      monthly_salary saved_amount savings_rate := by
  sorry

end children_education_expense_l3067_306797


namespace arithmetic_equality_l3067_306714

theorem arithmetic_equality : 1234562 - 12 * 3 * 2 = 1234490 := by
  sorry

end arithmetic_equality_l3067_306714


namespace disney_banquet_residents_l3067_306722

theorem disney_banquet_residents (total_attendees : ℕ) (resident_price non_resident_price : ℚ) (total_revenue : ℚ) :
  total_attendees = 586 →
  resident_price = 12.95 →
  non_resident_price = 17.95 →
  total_revenue = 9423.70 →
  ∃ (residents non_residents : ℕ),
    residents + non_residents = total_attendees ∧
    residents * resident_price + non_residents * non_resident_price = total_revenue ∧
    residents = 220 :=
by sorry

end disney_banquet_residents_l3067_306722


namespace tetrahedron_volume_l3067_306772

/-- 
Given a tetrahedron with:
- a, b: lengths of two opposite edges
- d: distance between edges a and b
- φ: angle between edges a and b
- V: volume of the tetrahedron

The volume V is equal to (1/6) * a * b * d * sin(φ)
-/
theorem tetrahedron_volume 
  (a b d φ V : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hφ : 0 < φ ∧ φ < π) 
  (hV : V > 0) :
  V = (1/6) * a * b * d * Real.sin φ :=
sorry

end tetrahedron_volume_l3067_306772


namespace same_heads_probability_l3067_306792

/-- The number of possible outcomes when tossing two pennies -/
def keiko_outcomes : ℕ := 4

/-- The number of possible outcomes when tossing three pennies -/
def ephraim_outcomes : ℕ := 8

/-- The number of ways Keiko and Ephraim can get the same number of heads -/
def matching_outcomes : ℕ := 7

/-- The total number of possible outcomes when Keiko tosses two pennies and Ephraim tosses three pennies -/
def total_outcomes : ℕ := keiko_outcomes * ephraim_outcomes

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := matching_outcomes / total_outcomes

theorem same_heads_probability : probability = 7 / 32 := by
  sorry

end same_heads_probability_l3067_306792


namespace consecutive_squares_sum_equality_l3067_306717

theorem consecutive_squares_sum_equality :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = (n+3)^2 + (n+4)^2 := by
  sorry

end consecutive_squares_sum_equality_l3067_306717


namespace consecutive_numbers_divisibility_l3067_306754

theorem consecutive_numbers_divisibility (k : ℕ) :
  let r₁ := k % 2022
  let r₂ := (k + 1) % 2022
  let r₃ := (k + 2) % 2022
  Prime (r₁ + r₂ + r₃) →
  (k % 2022 = 0) ∨ ((k + 1) % 2022 = 0) ∨ ((k + 2) % 2022 = 0) :=
by sorry

end consecutive_numbers_divisibility_l3067_306754


namespace max_inscribed_triangles_count_l3067_306730

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) :=
  (h_pos : 0 < b ∧ b < a)

/-- A right-angled isosceles triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) :=
  (vertex : ℝ × ℝ)
  (h_on_ellipse : (vertex.1^2 / a^2) + (vertex.2^2 / b^2) = 1)
  (h_right_angled : True)  -- Placeholder for the right-angled condition
  (h_isosceles : True)     -- Placeholder for the isosceles condition
  (h_vertex_b : vertex.1 = 0 ∧ vertex.2 = b)

/-- The maximum number of right-angled isosceles triangles inscribed in an ellipse -/
def max_inscribed_triangles (e : Ellipse a b) : ℕ :=
  3

theorem max_inscribed_triangles_count (a b : ℝ) (e : Ellipse a b) :
  ∃ (n : ℕ), n ≤ max_inscribed_triangles e ∧
  ∀ (m : ℕ), (∃ (triangles : Fin m → InscribedTriangle e), 
    ∀ (i j : Fin m), i ≠ j → triangles i ≠ triangles j) → m ≤ n :=
sorry

end max_inscribed_triangles_count_l3067_306730


namespace fourth_year_students_without_glasses_l3067_306702

theorem fourth_year_students_without_glasses 
  (total_students : ℕ) 
  (fourth_year_students : ℕ) 
  (students_with_glasses : ℕ) 
  (students_without_glasses : ℕ) :
  total_students = 8 * fourth_year_students - 32 →
  students_with_glasses = students_without_glasses + 10 →
  total_students = 1152 →
  fourth_year_students = students_with_glasses + students_without_glasses →
  students_without_glasses = 69 :=
by sorry

end fourth_year_students_without_glasses_l3067_306702


namespace problem_statement_l3067_306786

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  2 * a + 2 * b - 3 * a * b = 9 := by
  sorry

end problem_statement_l3067_306786


namespace inscribed_rectangle_sides_l3067_306719

/-- A triangle with sides 3, 4, and 5 -/
structure Triangle345 where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5

/-- A rectangle inscribed in a Triangle345 -/
structure InscribedRectangle (t : Triangle345) where
  short_side : ℝ
  long_side : ℝ
  h_double : long_side = 2 * short_side
  h_inscribed : short_side > 0 ∧ long_side > 0 ∧ long_side ≤ t.c

theorem inscribed_rectangle_sides (t : Triangle345) (r : InscribedRectangle t) :
  r.short_side = 48 / 67 ∧ r.long_side = 96 / 67 := by
  sorry

end inscribed_rectangle_sides_l3067_306719


namespace unique_intersection_l3067_306712

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 5) = 3 * Complex.abs (z + 5)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the intersection condition
def intersectsOnce (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- Theorem statement
theorem unique_intersection :
  ∃! k, intersectsOnce k ∧ k = 12.5 := by sorry

end unique_intersection_l3067_306712


namespace max_omega_for_monotonic_sine_l3067_306737

theorem max_omega_for_monotonic_sine (A ω : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc (-3 * π / 4) (-π / 6),
    ∀ y ∈ Set.Icc (-3 * π / 4) (-π / 6),
    x < y → A * Real.sin (x + ω * π / 2) < A * Real.sin (y + ω * π / 2)) →
  ω ≤ 3 / 2 :=
by sorry

end max_omega_for_monotonic_sine_l3067_306737


namespace tan_nine_pi_fourth_l3067_306710

theorem tan_nine_pi_fourth : Real.tan (9 * π / 4) = 1 := by
  sorry

end tan_nine_pi_fourth_l3067_306710


namespace shirts_before_buying_l3067_306735

/-- Given that Sarah bought new shirts and now has a total number of shirts,
    prove that the number of shirts she had before is the difference between
    the total and the new shirts. -/
theorem shirts_before_buying (total : ℕ) (new : ℕ) (before : ℕ) 
    (h1 : total = before + new) : before = total - new := by
  sorry

end shirts_before_buying_l3067_306735


namespace book_cost_price_l3067_306739

/-- Given a book sold for Rs 90 with a profit rate of 80%, prove that the cost price is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) (h1 : selling_price = 90) (h2 : profit_rate = 80) :
  ∃ (cost_price : ℝ), cost_price = 50 ∧ profit_rate / 100 = (selling_price - cost_price) / cost_price :=
by sorry

end book_cost_price_l3067_306739


namespace player_a_advantage_l3067_306723

/-- Represents the outcome of a roll of two dice -/
structure DiceRoll :=
  (sum : Nat)
  (probability : Rat)

/-- Calculates the expected value for a player given a list of dice rolls -/
def expectedValue (rolls : List DiceRoll) : Rat :=
  rolls.foldl (fun acc roll => acc + roll.sum * roll.probability) 0

/-- Represents the game rules -/
def gameRules (roll : DiceRoll) : Rat :=
  if roll.sum % 2 = 1 then roll.sum * roll.probability
  else if roll.sum = 2 then 0
  else -roll.sum * roll.probability

/-- The list of all possible dice rolls and their probabilities -/
def allRolls : List DiceRoll := [
  ⟨2, 1/36⟩, ⟨3, 1/18⟩, ⟨4, 1/12⟩, ⟨5, 1/9⟩, ⟨6, 5/36⟩, 
  ⟨7, 1/6⟩, ⟨8, 5/36⟩, ⟨9, 1/9⟩, ⟨10, 1/12⟩, ⟨11, 1/18⟩, ⟨12, 1/36⟩
]

/-- The expected value for player A per roll -/
def expectedValueA : Rat := allRolls.foldl (fun acc roll => acc + gameRules roll) 0

theorem player_a_advantage : 
  expectedValueA > 0 ∧ 36 * expectedValueA = 2 := by sorry


end player_a_advantage_l3067_306723


namespace position_of_81st_number_l3067_306775

/-- Represents the triangular number pattern where each row has one more number than the previous row. -/
def TriangularPattern : Nat → Nat → Nat
  | row, pos => if pos ≤ row then (row * (row - 1)) / 2 + pos else 0

/-- The position of a number in the triangular pattern. -/
structure Position where
  row : Nat
  pos : Nat

/-- Finds the position of the nth number in the triangular pattern. -/
def findPosition (n : Nat) : Position :=
  let row := (Nat.sqrt (8 * n + 1) - 1) / 2 + 1
  let pos := n - (row * (row - 1)) / 2
  ⟨row, pos⟩

theorem position_of_81st_number :
  findPosition 81 = ⟨13, 3⟩ := by sorry

end position_of_81st_number_l3067_306775


namespace round_trip_average_speed_l3067_306756

/-- Calculate the average speed for a round trip given specific segments and speeds -/
theorem round_trip_average_speed 
  (total_distance : ℝ)
  (train_distance train_speed : ℝ)
  (car_to_y_distance car_to_y_speed : ℝ)
  (bus_distance bus_speed : ℝ)
  (car_return_distance car_return_speed : ℝ)
  (plane_speed : ℝ)
  (h1 : total_distance = 1500)
  (h2 : train_distance = 500)
  (h3 : train_speed = 60)
  (h4 : car_to_y_distance = 700)
  (h5 : car_to_y_speed = 50)
  (h6 : bus_distance = 300)
  (h7 : bus_speed = 40)
  (h8 : car_return_distance = 600)
  (h9 : car_return_speed = 60)
  (h10 : plane_speed = 500)
  : ∃ (average_speed : ℝ), abs (average_speed - 72.03) < 0.01 := by
  sorry

end round_trip_average_speed_l3067_306756


namespace problem_solution_l3067_306796

theorem problem_solution (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*x^2 + 1
  let g : ℝ → ℝ := λ x ↦ -x^3 + 3*x^2 + x - 7
  (f x + g x = x - 6) → (g x = -x^3 + 3*x^2 + x - 7) := by
  sorry

end problem_solution_l3067_306796


namespace equal_share_money_l3067_306781

theorem equal_share_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 3.75)
  (h2 : num_people = 3) : 
  total_amount / num_people = 1.25 := by
  sorry

end equal_share_money_l3067_306781


namespace seymour_fertilizer_calculation_l3067_306703

/-- Calculates the total fertilizer needed for Seymour's plant shop --/
theorem seymour_fertilizer_calculation : 
  let petunia_flats : ℕ := 4
  let petunias_per_flat : ℕ := 8
  let petunia_fertilizer : ℕ := 8
  let rose_flats : ℕ := 3
  let roses_per_flat : ℕ := 6
  let rose_fertilizer : ℕ := 3
  let sunflower_flats : ℕ := 5
  let sunflowers_per_flat : ℕ := 10
  let sunflower_fertilizer : ℕ := 6
  let orchid_flats : ℕ := 2
  let orchids_per_flat : ℕ := 4
  let orchid_fertilizer : ℕ := 4
  let venus_flytraps : ℕ := 2
  let venus_flytrap_fertilizer : ℕ := 2
  
  petunia_flats * petunias_per_flat * petunia_fertilizer +
  rose_flats * roses_per_flat * rose_fertilizer +
  sunflower_flats * sunflowers_per_flat * sunflower_fertilizer +
  orchid_flats * orchids_per_flat * orchid_fertilizer +
  venus_flytraps * venus_flytrap_fertilizer = 646 := by
  sorry

#check seymour_fertilizer_calculation

end seymour_fertilizer_calculation_l3067_306703


namespace salary_change_percentage_l3067_306748

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.6)
  let final_salary := decreased_salary * (1 + 0.6)
  final_salary = initial_salary * 0.64 ∧ 
  (initial_salary - final_salary) / initial_salary = 0.36 :=
by sorry

end salary_change_percentage_l3067_306748


namespace calculation_one_l3067_306731

theorem calculation_one :
  (27 : ℝ) ^ (1/3) + (1/9).sqrt / (-2/3) + |(-(1/2))| = 3 := by sorry

end calculation_one_l3067_306731


namespace half_plus_five_equals_fifteen_l3067_306784

theorem half_plus_five_equals_fifteen (n : ℝ) : (1/2) * n + 5 = 15 → n = 20 := by
  sorry

end half_plus_five_equals_fifteen_l3067_306784


namespace intersection_sum_l3067_306793

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 5*x < 0}
def N (p : ℝ) : Set ℝ := {x | p < x ∧ x < 6}

-- Define the intersection of M and N
def M_intersect_N (p q : ℝ) : Set ℝ := {x | 2 < x ∧ x < q}

-- Theorem statement
theorem intersection_sum (p q : ℝ) : 
  M ∩ N p = M_intersect_N p q → p + q = 7 := by
  sorry

end intersection_sum_l3067_306793


namespace reservoir_shortage_l3067_306752

/-- Represents a water reservoir with its capacity and current amount --/
structure Reservoir where
  capacity : ℝ
  current_amount : ℝ
  normal_level : ℝ
  h1 : current_amount = 14
  h2 : current_amount = 2 * normal_level
  h3 : current_amount = 0.7 * capacity

/-- The difference between the total capacity and the normal level is 13 million gallons --/
theorem reservoir_shortage (r : Reservoir) : r.capacity - r.normal_level = 13 := by
  sorry

#check reservoir_shortage

end reservoir_shortage_l3067_306752


namespace carnations_ordered_l3067_306794

/-- Proves that given the specified conditions, the number of carnations ordered is 375 -/
theorem carnations_ordered (tulips : ℕ) (roses : ℕ) (price_per_flower : ℕ) (total_expenses : ℕ) : 
  tulips = 250 → roses = 320 → price_per_flower = 2 → total_expenses = 1890 →
  ∃ carnations : ℕ, carnations = 375 ∧ 
    price_per_flower * (tulips + roses + carnations) = total_expenses := by
  sorry

#check carnations_ordered

end carnations_ordered_l3067_306794


namespace smallest_abcd_l3067_306795

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_range : a ∈ Finset.range 10
  b_range : b ∈ Finset.range 10
  c_range : c ∈ Finset.range 10
  d_range : d ∈ Finset.range 10
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a FourDigitNumber to its numerical value -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Represents the conditions of the problem -/
structure ProblemConditions where
  ab : Nat
  a : Nat
  b : Nat
  ab_two_digit : ab ∈ Finset.range 100
  ab_eq : ab = 10 * a + b
  a_not_eq_b : a ≠ b
  result : FourDigitNumber
  multiplication_condition : ab * a = result.toNat

/-- The main theorem stating that the smallest ABCD satisfying the conditions is 2046 -/
theorem smallest_abcd (conditions : ProblemConditions) :
  ∀ other : FourDigitNumber,
    (∃ other_conditions : ProblemConditions, other_conditions.result = other) →
    conditions.result.toNat ≤ other.toNat ∧ conditions.result.toNat = 2046 := by
  sorry


end smallest_abcd_l3067_306795


namespace expected_twos_is_one_third_l3067_306732

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 5 / 6

/-- The expected number of 2's when rolling two standard dice -/
def expected_twos : ℚ := 1 / 3

/-- Theorem: The expected number of 2's when rolling two standard dice is 1/3 -/
theorem expected_twos_is_one_third :
  expected_twos = 1 / 3 := by
  sorry

end expected_twos_is_one_third_l3067_306732


namespace angle_equivalence_l3067_306783

theorem angle_equivalence :
  ∃ (α : ℝ) (k : ℤ), -27/4 * π = α + 2*k*π ∧ 0 ≤ α ∧ α < 2*π ∧ α = 5*π/4 ∧ k = -8 :=
by sorry

end angle_equivalence_l3067_306783
