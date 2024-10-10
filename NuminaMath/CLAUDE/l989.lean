import Mathlib

namespace triangle_dot_product_l989_98979

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b - c = 1,
    and the area of the triangle is √3, then the dot product of vectors AB and AC is 13/4 -/
theorem triangle_dot_product (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b - c = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  b * c * Real.cos A = 13/4 := by
  sorry

end triangle_dot_product_l989_98979


namespace no_curious_numbers_l989_98991

def CuriousNumber (f : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, f x = f (a - x)

theorem no_curious_numbers (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f x ≠ x) :
  ¬ (∃ a ∈ ({60, 62, 823} : Set ℤ), CuriousNumber f a) :=
sorry

end no_curious_numbers_l989_98991


namespace sqrt_product_equality_l989_98908

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 48 * Real.sqrt 6 = 72 * Real.sqrt 3 := by
  sorry

end sqrt_product_equality_l989_98908


namespace sum_of_19th_powers_zero_l989_98958

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
  sorry

end sum_of_19th_powers_zero_l989_98958


namespace sqrt_sum_equality_l989_98936

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equality_l989_98936


namespace cantaloupe_price_l989_98927

/-- Represents the problem of finding the price of cantaloupes --/
def CantalouperPriceProblem (C : ℚ) : Prop :=
  let initial_cantaloupes : ℕ := 30
  let initial_honeydews : ℕ := 27
  let dropped_cantaloupes : ℕ := 2
  let rotten_honeydews : ℕ := 3
  let final_cantaloupes : ℕ := 8
  let final_honeydews : ℕ := 9
  let honeydew_price : ℚ := 3
  let total_revenue : ℚ := 85
  let sold_cantaloupes : ℕ := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews : ℕ := initial_honeydews - final_honeydews - rotten_honeydews
  C * sold_cantaloupes + honeydew_price * sold_honeydews = total_revenue

/-- Theorem stating that the price of each cantaloupe is $2 --/
theorem cantaloupe_price : ∃ C : ℚ, CantalouperPriceProblem C ∧ C = 2 := by
  sorry

end cantaloupe_price_l989_98927


namespace abs_neg_five_eq_five_l989_98931

theorem abs_neg_five_eq_five : abs (-5 : ℤ) = 5 := by
  sorry

end abs_neg_five_eq_five_l989_98931


namespace superhero_payment_l989_98966

/-- Superhero payment calculation -/
theorem superhero_payment (W : ℝ) : 
  let superman_productivity := 0.1 * W
  let flash_productivity := 2 * superman_productivity
  let combined_productivity := superman_productivity + flash_productivity
  let remaining_work := 0.9 * W
  let combined_time := remaining_work / combined_productivity
  let superman_total_time := 1 + combined_time
  let flash_total_time := combined_time
  let payment (t : ℝ) := 90 / t
  (payment superman_total_time = 22.5) ∧ (payment flash_total_time = 30) :=
by sorry

end superhero_payment_l989_98966


namespace grid_segment_sums_equal_area_l989_98983

/-- Represents a convex polygon with vertices at integer grid points --/
structure ConvexGridPolygon where
  vertices : List (Int × Int)
  is_convex : Bool
  no_sides_on_gridlines : Bool

/-- Calculates the sum of lengths of horizontal grid segments within the polygon --/
def sum_horizontal_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the sum of lengths of vertical grid segments within the polygon --/
def sum_vertical_segments (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Calculates the area of the polygon --/
def polygon_area (polygon : ConvexGridPolygon) : ℝ :=
  sorry

/-- Theorem stating that for a convex polygon with vertices at integer grid points
    and no sides along grid lines, the sum of horizontal grid segment lengths equals
    the sum of vertical grid segment lengths, and both equal the polygon's area --/
theorem grid_segment_sums_equal_area (polygon : ConvexGridPolygon) :
  sum_horizontal_segments polygon = sum_vertical_segments polygon ∧
  sum_horizontal_segments polygon = polygon_area polygon :=
  sorry

end grid_segment_sums_equal_area_l989_98983


namespace sector_area_sixty_degrees_radius_six_l989_98916

/-- The area of a circular sector with central angle π/3 and radius 6 is 6π -/
theorem sector_area_sixty_degrees_radius_six : 
  let r : ℝ := 6
  let α : ℝ := π / 3
  let sector_area := (1 / 2) * r^2 * α
  sector_area = 6 * π := by sorry

end sector_area_sixty_degrees_radius_six_l989_98916


namespace normal_dist_probability_l989_98919

variable (ξ : Real)
variable (μ δ : Real)

-- ξ follows a normal distribution with mean μ and variance δ²
def normal_dist (ξ μ δ : Real) : Prop := sorry

-- Probability function
noncomputable def P (event : Real → Prop) : Real := sorry

theorem normal_dist_probability 
  (h1 : normal_dist ξ μ δ)
  (h2 : P (λ x => x > 4) = P (λ x => x < 2))
  (h3 : P (λ x => x ≤ 0) = 0.2) :
  P (λ x => 0 < x ∧ x < 6) = 0.6 := by sorry

end normal_dist_probability_l989_98919


namespace ten_thousandths_digit_of_7_32_l989_98999

theorem ten_thousandths_digit_of_7_32 : ∃ (d : ℕ), d < 10 ∧ 
  (∃ (n : ℕ), (7 : ℚ) / 32 = (n * 10 + d : ℚ) / 100000 ∧ d = 5) := by
  sorry

end ten_thousandths_digit_of_7_32_l989_98999


namespace min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l989_98900

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 :=
by sorry

theorem min_value_reciprocal_sum_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 :=
by sorry

end min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l989_98900


namespace travelers_checks_denomination_l989_98961

theorem travelers_checks_denomination (total_checks : ℕ) (total_worth : ℝ) (spendable_checks : ℕ) (remaining_average : ℝ) :
  total_checks = 30 →
  total_worth = 1800 →
  spendable_checks = 18 →
  remaining_average = 75 →
  (total_worth - (total_checks - spendable_checks : ℝ) * remaining_average) / spendable_checks = 50 :=
by sorry

end travelers_checks_denomination_l989_98961


namespace bridget_skittles_l989_98957

/-- If Bridget has 4 Skittles, Henry has 4 Skittles, and Henry gives all of his Skittles to Bridget,
    then Bridget will have 8 Skittles in total. -/
theorem bridget_skittles (bridget_initial : ℕ) (henry_initial : ℕ)
    (h1 : bridget_initial = 4)
    (h2 : henry_initial = 4) :
    bridget_initial + henry_initial = 8 := by
  sorry

end bridget_skittles_l989_98957


namespace power_division_rule_l989_98947

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end power_division_rule_l989_98947


namespace circle_intersection_axes_l989_98985

theorem circle_intersection_axes (m : ℝ) :
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1) ∧
  (∃ x : ℝ, (x - m + 1)^2 + m^2 = 1) ∧
  (∃ y : ℝ, (1 - m)^2 + (y - m)^2 = 1) →
  0 ≤ m ∧ m ≤ 1 := by
sorry

end circle_intersection_axes_l989_98985


namespace least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l989_98988

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 5 ∧ digits.toFinset.card = 5

def divisible_by_digits_except_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5 → n % d = 0

theorem least_five_digit_divisible_by_digits :
  ∀ n : ℕ,
    is_five_digit n ∧
    all_digits_different n ∧
    divisible_by_digits_except_five n →
    12376 ≤ n :=
by sorry

theorem twelve_three_seven_six_satisfies_conditions :
  is_five_digit 12376 ∧
  all_digits_different 12376 ∧
  divisible_by_digits_except_five 12376 :=
by sorry

end least_five_digit_divisible_by_digits_twelve_three_seven_six_satisfies_conditions_l989_98988


namespace abs_neg_eight_eq_eight_l989_98914

theorem abs_neg_eight_eq_eight :
  abs (-8 : ℤ) = 8 := by
  sorry

end abs_neg_eight_eq_eight_l989_98914


namespace quadratic_roots_identity_l989_98974

theorem quadratic_roots_identity (a b c : ℝ) : 
  (∃ x y : ℝ, x = Real.sin (42 * π / 180) ∧ y = Real.sin (48 * π / 180) ∧ 
    (∀ z : ℝ, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y)) →
  b^2 = a^2 + 2*a*c :=
by sorry

end quadratic_roots_identity_l989_98974


namespace larger_number_proof_l989_98906

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 25) (h2 : Nat.lcm a b = 25 * 14 * 16) :
  max a b = 400 := by
  sorry

end larger_number_proof_l989_98906


namespace functional_equation_2013_l989_98922

/-- Given a function f: ℝ → ℝ satisfying f(x-y) = f(x) + f(y) - 2xy for all real x and y,
    prove that f(2013) = 4052169 -/
theorem functional_equation_2013 (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - y) = f x + f y - 2 * x * y) : 
    f 2013 = 4052169 := by
  sorry

end functional_equation_2013_l989_98922


namespace room_width_calculation_l989_98949

/-- Given a rectangular room with length 5 meters, prove that its width is 4.75 meters
    when the cost of paving is 900 per square meter and the total cost is 21375. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5 →
  cost_per_sqm = 900 →
  total_cost = 21375 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

#eval (21375 / 900) / 5

end room_width_calculation_l989_98949


namespace geometric_sequence_ratio_l989_98909

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : 6 * a 1 + 4 * a 2 = 2 * a 3) :
  (a 11 + a 13 + a 16 + a 20 + a 21) / (a 8 + a 10 + a 13 + a 17 + a 18) = 27 := by
sorry

end geometric_sequence_ratio_l989_98909


namespace robin_candy_count_l989_98954

/-- Given Robin's initial candy count, the number she ate, and the number her sister gave her, 
    her final candy count is equal to 37. -/
theorem robin_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) 
    (h1 : initial = 23) 
    (h2 : eaten = 7) 
    (h3 : received = 21) : 
  initial - eaten + received = 37 := by
  sorry

end robin_candy_count_l989_98954


namespace singer_songs_released_l989_98973

/-- Given a singer's work schedule and total time spent, calculate the number of songs released --/
theorem singer_songs_released 
  (hours_per_day : ℕ) 
  (days_per_song : ℕ) 
  (total_hours : ℕ) 
  (h1 : hours_per_day = 10)
  (h2 : days_per_song = 10)
  (h3 : total_hours = 300) :
  total_hours / (hours_per_day * days_per_song) = 3 := by
  sorry

end singer_songs_released_l989_98973


namespace smallest_product_l989_98980

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by
  sorry

end smallest_product_l989_98980


namespace odd_painted_faces_count_l989_98956

/-- Represents a cube with its number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Represents the block of cubes -/
def Block := List Cube

/-- Creates a 6x6x1 block of painted cubes -/
def create_block : Block :=
  sorry

/-- Counts the number of cubes with an odd number of painted faces -/
def count_odd_painted (block : Block) : Nat :=
  sorry

/-- Theorem stating that the number of cubes with an odd number of painted faces is 16 -/
theorem odd_painted_faces_count (block : Block) : 
  block = create_block → count_odd_painted block = 16 := by
  sorry

end odd_painted_faces_count_l989_98956


namespace probability_of_selecting_female_student_l989_98996

theorem probability_of_selecting_female_student :
  let total_students : ℕ := 4
  let female_students : ℕ := 3
  let male_students : ℕ := 1
  female_students + male_students = total_students →
  (female_students : ℚ) / total_students = 3 / 4 :=
by
  sorry

end probability_of_selecting_female_student_l989_98996


namespace basketball_substitutions_l989_98932

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (5 - k) * (11 + k) * substitution_ways k

def total_substitution_ways : ℕ :=
  (List.range 6).map substitution_ways |> List.sum

theorem basketball_substitutions :
  total_substitution_ways % 1000 = 736 := by
  sorry

end basketball_substitutions_l989_98932


namespace homework_problems_l989_98930

theorem homework_problems (t p : ℕ) (ht : t > 0) (hp : p > 10) : 
  (∀ (t' : ℕ), t' > 0 → p * t = (2 * p - 2) * (t' - 1) → t' = t) →
  p * t = 48 := by
sorry

end homework_problems_l989_98930


namespace heartsuit_property_false_l989_98913

-- Define the ♥ operation for real numbers
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that 3(x ♥ y) ≠ (3x) ♥ y for all real x and y
theorem heartsuit_property_false :
  ∀ x y : ℝ, 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end heartsuit_property_false_l989_98913


namespace leadership_selection_count_l989_98995

def tribe_size : ℕ := 12
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 2

def leadership_selection_ways : ℕ :=
  tribe_size *
  (Nat.choose (tribe_size - 1) num_supporting_chiefs) *
  (Nat.choose (tribe_size - 1 - num_supporting_chiefs) (num_supporting_chiefs * num_inferior_officers_per_chief) /
   Nat.factorial num_supporting_chiefs)

theorem leadership_selection_count :
  leadership_selection_ways = 248040 :=
by sorry

end leadership_selection_count_l989_98995


namespace tim_bought_three_dozens_l989_98977

/-- The number of dozens of eggs Tim bought -/
def dozens_bought (egg_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (12 * egg_price)

/-- Theorem stating that Tim bought 3 dozens of eggs -/
theorem tim_bought_three_dozens :
  dozens_bought (1/2) 18 = 3 := by
  sorry

end tim_bought_three_dozens_l989_98977


namespace product_base_conversion_l989_98981

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat := sorry

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat := sorry

theorem product_base_conversion :
  let binary := [true, true, false, true]  -- 1101 in base 2
  let ternary := [2, 0, 2]  -- 202 in base 3
  (base2To10 binary) * (base3To10 ternary) = 260 := by sorry

end product_base_conversion_l989_98981


namespace last_four_digits_5_power_2017_l989_98945

/-- The last four digits of 5^n, represented as an integer between 0 and 9999 -/
def lastFourDigits (n : ℕ) : ℕ := 5^n % 10000

theorem last_four_digits_5_power_2017 :
  lastFourDigits 5 = 3125 ∧
  lastFourDigits 6 = 5625 ∧
  lastFourDigits 7 = 8125 →
  lastFourDigits 2017 = 3125 := by
sorry

end last_four_digits_5_power_2017_l989_98945


namespace max_product_sum_l989_98928

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 4, 5} : Finset ℕ) →
  b ∈ ({1, 3, 4, 5} : Finset ℕ) →
  c ∈ ({1, 3, 4, 5} : Finset ℕ) →
  d ∈ ({1, 3, 4, 5} : Finset ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a * b + b * c + c * d + d * a) ≤ 42 :=
by sorry

end max_product_sum_l989_98928


namespace radio_loss_percentage_l989_98901

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Proves that the loss percentage for a radio with cost price 2400 and selling price 2100 is 12.5%. -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 2400
  let selling_price : ℚ := 2100
  loss_percentage cost_price selling_price = 25/2 := by
  sorry

end radio_loss_percentage_l989_98901


namespace five_attraction_permutations_l989_98912

theorem five_attraction_permutations : Nat.factorial 5 = 120 := by
  sorry

end five_attraction_permutations_l989_98912


namespace zucchini_amount_l989_98935

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def zucchini_price : ℝ := 2
def quarts_yield : ℝ := 4
def quart_price : ℝ := 10

theorem zucchini_amount (zucchini_pounds : ℝ) :
  eggplant_pounds * eggplant_price +
  zucchini_pounds * zucchini_price +
  tomato_pounds * tomato_price +
  onion_pounds * onion_price +
  basil_pounds * basil_price * 2 =
  quarts_yield * quart_price →
  zucchini_pounds = 4 := by sorry

end zucchini_amount_l989_98935


namespace quadratic_equation_properties_l989_98911

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Main theorem about the quadratic equation x^2 + mx + m - 2 = 0 -/
theorem quadratic_equation_properties (p : QuadraticParams) :
  -- If -2 is one root, the other root is 0
  (∃ (r : QuadraticRoots), r.x₁ = -2 ∧ r.x₂ = 0 ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- The equation always has two distinct real roots
  (∀ (x : ℝ), x^2 + p.m * x + p.m - 2 = 0 → 
    ∃ (r : QuadraticRoots), r.x₁ ≠ r.x₂ ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- If x₁^2 + x₂^2 + m(x₁ + x₂) = m^2 + 1, then m = -3 or m = 1
  (∀ (r : QuadraticRoots), 
    r.x₁^2 + r.x₂^2 + p.m * (r.x₁ + r.x₂) = p.m^2 + 1 →
    p.m = -3 ∨ p.m = 1) :=
by sorry

end quadratic_equation_properties_l989_98911


namespace probability_A_and_B_selected_l989_98964

theorem probability_A_and_B_selected (total_students : ℕ) (selected_students : ℕ) 
  (h1 : total_students = 5) (h2 : selected_students = 3) :
  (Nat.choose (total_students - 2) (selected_students - 2)) / (Nat.choose total_students selected_students) = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l989_98964


namespace exponent_multiplication_l989_98902

theorem exponent_multiplication (a : ℝ) : a * a^3 = a^4 := by sorry

end exponent_multiplication_l989_98902


namespace negative_integer_solutions_of_inequality_l989_98942

theorem negative_integer_solutions_of_inequality :
  {x : ℤ | x < 0 ∧ 3 * x + 1 ≥ -5} = {-2, -1} := by
  sorry

end negative_integer_solutions_of_inequality_l989_98942


namespace excursion_dates_correct_l989_98987

/-- Represents the four excursion locations --/
inductive Location
| Carpathians
| Kyiv
| Forest
| Museum

/-- Represents a calendar month --/
structure Month where
  number : Nat
  days : Nat
  first_day_sunday : Bool

/-- Represents an excursion --/
structure Excursion where
  location : Location
  month : Month
  day : Nat

/-- Checks if a given day is the first Sunday after the first Saturday --/
def is_first_sunday_after_saturday (m : Month) (d : Nat) : Prop :=
  d = 8 ∧ m.first_day_sunday

/-- The theorem to prove --/
theorem excursion_dates_correct (feb mar : Month) 
  (e1 e2 e3 e4 : Excursion) : 
  feb.number = 2 → 
  mar.number = 3 → 
  feb.days = 28 → 
  mar.days = 31 → 
  feb.first_day_sunday = true → 
  mar.first_day_sunday = true → 
  e1.location = Location.Carpathians → 
  e2.location = Location.Kyiv → 
  e3.location = Location.Forest → 
  e4.location = Location.Museum → 
  e1.month = feb ∧ e1.day = 1 ∧
  e2.month = feb ∧ is_first_sunday_after_saturday feb e2.day ∧
  e3.month = mar ∧ e3.day = 1 ∧
  e4.month = mar ∧ is_first_sunday_after_saturday mar e4.day :=
sorry

end excursion_dates_correct_l989_98987


namespace a_equals_one_sufficient_not_necessary_l989_98943

theorem a_equals_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → (a - 1) * (a - 2) = 0) ∧
  (∃ a : ℝ, a ≠ 1 ∧ (a - 1) * (a - 2) = 0) :=
by sorry

end a_equals_one_sufficient_not_necessary_l989_98943


namespace little_john_theorem_l989_98998

def little_john_problem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) : ℚ :=
  initial_amount - (given_to_each_friend * num_friends) - amount_left

theorem little_john_theorem (initial_amount : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) (amount_left : ℚ) :
  little_john_problem initial_amount given_to_each_friend num_friends amount_left =
  initial_amount - (given_to_each_friend * num_friends) - amount_left :=
by
  sorry

#eval little_john_problem 10.50 2.20 2 3.85

end little_john_theorem_l989_98998


namespace derivative_at_one_is_negative_one_l989_98969

open Real

theorem derivative_at_one_is_negative_one
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, f x = 2 * x * (deriv f 1) + log x) →
  deriv f 1 = -1 := by
sorry

end derivative_at_one_is_negative_one_l989_98969


namespace isabel_games_l989_98972

/-- The number of DS games Isabel had initially -/
def initial_games : ℕ := 90

/-- The number of DS games Isabel gave away -/
def games_given_away : ℕ := 87

/-- The number of DS games Isabel has left -/
def games_left : ℕ := 3

/-- Theorem stating that the initial number of games is equal to the sum of games given away and games left -/
theorem isabel_games : initial_games = games_given_away + games_left := by
  sorry

end isabel_games_l989_98972


namespace gift_cost_problem_l989_98915

theorem gift_cost_problem (initial_friends : ℕ) (dropped_out : ℕ) (extra_cost : ℝ) :
  initial_friends = 10 →
  dropped_out = 4 →
  extra_cost = 8 →
  ∃ (total_cost : ℝ),
    total_cost / (initial_friends - dropped_out : ℝ) = total_cost / initial_friends + extra_cost ∧
    total_cost = 120 := by
  sorry

end gift_cost_problem_l989_98915


namespace expand_product_l989_98993

theorem expand_product (x : ℝ) : (3*x + 4) * (2*x + 7) = 6*x^2 + 29*x + 28 := by
  sorry

end expand_product_l989_98993


namespace ryan_marbles_ryan_has_28_marbles_l989_98903

theorem ryan_marbles (chris_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let total_marbles := chris_marbles + remaining_marbles * 2
  total_marbles - chris_marbles

theorem ryan_has_28_marbles :
  ryan_marbles 12 20 = 28 :=
by sorry

end ryan_marbles_ryan_has_28_marbles_l989_98903


namespace six_balls_four_boxes_l989_98939

/-- The number of ways to partition n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def partition_count (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 ways to partition 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem six_balls_four_boxes : partition_count 6 4 = 2 := by
  sorry

end six_balls_four_boxes_l989_98939


namespace committee_formation_count_l989_98994

def club_size : ℕ := 30
def committee_size : ℕ := 5

def ways_to_form_committee : ℕ :=
  club_size * (Nat.choose (club_size - 1) (committee_size - 1))

theorem committee_formation_count :
  ways_to_form_committee = 712530 := by
  sorry

end committee_formation_count_l989_98994


namespace solve_for_c_l989_98953

/-- Given two functions p and q, where p(x) = 3x - 9 and q(x) = 4x - c,
    prove that c = 4 when p(q(3)) = 15 -/
theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) 
    (hp : ∀ x, p x = 3 * x - 9)
    (hq : ∀ x, q x = 4 * x - c)
    (h_eq : p (q 3) = 15) : 
  c = 4 := by
  sorry

end solve_for_c_l989_98953


namespace inequality_solution_set_l989_98992

theorem inequality_solution_set :
  ∀ x : ℝ, (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 := by
sorry

end inequality_solution_set_l989_98992


namespace notebook_profit_l989_98955

/-- Calculates the profit from selling notebooks -/
def calculate_profit (
  num_notebooks : ℕ
  ) (purchase_price : ℚ)
    (sell_price : ℚ) : ℚ :=
  num_notebooks * sell_price - num_notebooks * purchase_price

/-- Proves that the profit from selling 1200 notebooks, 
    purchased at 4 for $5 and sold at 5 for $8, is $420 -/
theorem notebook_profit : 
  calculate_profit 1200 (5/4) (8/5) = 420 := by
  sorry

end notebook_profit_l989_98955


namespace line_through_point_parallel_to_given_l989_98924

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the equation of the new line
def new_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given : 
  (∀ (x y : ℝ), new_line x y ↔ ∃ (k : ℝ), x - point.1 = k * 1 ∧ y - point.2 = k * (-1/2)) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ ∧ given_line x₂ y₂ → 
    (x₂ - x₁) * (-1/2) = (y₂ - y₁) * 1) ∧
  new_line point.1 point.2 :=
sorry

end line_through_point_parallel_to_given_l989_98924


namespace g_ge_f_implies_t_range_l989_98968

noncomputable def g (x : ℝ) : ℝ := Real.log x + 3 / (4 * x) - (1 / 4) * x - 1

def f (t x : ℝ) : ℝ := x^2 - 2 * t * x + 4

theorem g_ge_f_implies_t_range (t : ℝ) :
  (∀ x1 ∈ Set.Ioo 0 2, ∃ x2 ∈ Set.Icc 1 2, g x1 ≥ f t x2) →
  t ≥ 17/8 :=
by sorry

end g_ge_f_implies_t_range_l989_98968


namespace algebraic_identity_l989_98940

theorem algebraic_identity (a b c d : ℝ) :
  (a^2 + b^2) * (a*b + c*d) - a*b * (a^2 + b^2 - c^2 - d^2) = (a*c + b*d) * (a*d + b*c) := by
  sorry

end algebraic_identity_l989_98940


namespace rectangle_area_proof_l989_98926

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units,
    prove that y = 9. -/
theorem rectangle_area_proof (y : ℝ) : y > 0 → y * 5 = 45 → y = 9 := by
  sorry

end rectangle_area_proof_l989_98926


namespace cubic_expression_value_l989_98910

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end cubic_expression_value_l989_98910


namespace x_y_negative_l989_98989

theorem x_y_negative (x y : ℝ) (h1 : x - y > 2*x) (h2 : x + y < 0) : x < 0 ∧ y < 0 := by
  sorry

end x_y_negative_l989_98989


namespace field_length_is_16_l989_98963

/-- Proves that the length of a rectangular field is 16 meters given specific conditions --/
theorem field_length_is_16 (w : ℝ) (l : ℝ) : 
  l = 2 * w →  -- length is double the width
  16 = (1/8) * (l * w) →  -- pond area (4^2) is 1/8 of field area
  l = 16 := by
sorry


end field_length_is_16_l989_98963


namespace min_functional_digits_l989_98962

def is_representable (digits : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ digits ∨ ∃ a b, a ∈ digits ∧ b ∈ digits ∧ a + b = n

def is_valid_digit_set (digits : Finset ℕ) : Prop :=
  ∀ n, n ≥ 1 ∧ n ≤ 99999999 → is_representable digits n

theorem min_functional_digits :
  ∃ digits : Finset ℕ, digits.card = 5 ∧ is_valid_digit_set digits ∧
  ∀ smaller_digits : Finset ℕ, smaller_digits.card < 5 → ¬is_valid_digit_set smaller_digits :=
sorry

end min_functional_digits_l989_98962


namespace cost_of_potatoes_l989_98904

/-- Proves that the cost of each bag of potatoes is $6 -/
theorem cost_of_potatoes (chicken_price : ℝ) (celery_price : ℝ) (total_cost : ℝ) :
  chicken_price = 3 →
  celery_price = 2 →
  total_cost = 35 →
  (5 * chicken_price + 4 * celery_price + 2 * ((total_cost - 5 * chicken_price - 4 * celery_price) / 2)) = total_cost →
  (total_cost - 5 * chicken_price - 4 * celery_price) / 2 = 6 :=
by sorry

end cost_of_potatoes_l989_98904


namespace x_value_l989_98978

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end x_value_l989_98978


namespace second_oil_price_l989_98970

/-- Given two types of oil mixed together, calculate the price of the second oil -/
theorem second_oil_price (volume1 volume2 : ℝ) (price1 mixed_price : ℝ) :
  volume1 = 10 →
  volume2 = 5 →
  price1 = 50 →
  mixed_price = 55.33 →
  (volume1 * price1 + volume2 * (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2) / (volume1 + volume2) = mixed_price →
  (volume1 * price1 + volume2 * mixed_price - volume1 * price1) / volume2 = 65.99 := by
  sorry

#eval (10 * 50 + 5 * 55.33 * 3 - 10 * 50) / 5

end second_oil_price_l989_98970


namespace circle_equation_l989_98965

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2}

-- Define the line L: 2x - 3y - 1 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - 3 * p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 0)

theorem circle_equation :
  (∃ c : ℝ × ℝ, c ∈ L ∧ c ∈ C) ∧  -- The center of C lies on L
  A ∈ C ∧ B ∈ C →                  -- C passes through A and B
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end circle_equation_l989_98965


namespace nine_in_M_ten_not_in_M_l989_98920

/-- The set M of integers that can be expressed as the difference of two squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- 9 belongs to the set M -/
theorem nine_in_M : (9 : ℤ) ∈ M := by sorry

/-- 10 does not belong to the set M -/
theorem ten_not_in_M : (10 : ℤ) ∉ M := by sorry

end nine_in_M_ten_not_in_M_l989_98920


namespace factor_sum_l989_98967

theorem factor_sum (R S : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + R*X^2 + S) → 
  R + S = 54 := by
sorry

end factor_sum_l989_98967


namespace min_obtuse_triangles_2003gon_l989_98929

/-- A polygon inscribed in a circle -/
structure InscribedPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : InscribedPolygon) where
  triangle_count : ℕ
  triangle_count_eq : triangle_count = P.n - 2
  obtuse_count : ℕ
  acute_count : ℕ
  right_count : ℕ
  total_count : obtuse_count + acute_count + right_count = triangle_count
  max_non_obtuse : acute_count + right_count ≤ 2

/-- The theorem statement -/
theorem min_obtuse_triangles_2003gon :
  let P : InscribedPolygon := ⟨2003, by norm_num⟩
  ∀ T : Triangulation P, T.obtuse_count ≥ 1999 :=
by sorry

end min_obtuse_triangles_2003gon_l989_98929


namespace increased_value_l989_98905

theorem increased_value (x : ℝ) (p : ℝ) (h1 : x = 1200) (h2 : p = 40) :
  x * (1 + p / 100) = 1680 := by
  sorry

end increased_value_l989_98905


namespace kim_no_tests_probability_l989_98917

theorem kim_no_tests_probability 
  (p_math : ℝ) 
  (p_history : ℝ) 
  (h_math : p_math = 5/8) 
  (h_history : p_history = 1/3) 
  (h_independent : True)  -- Represents the independence of events
  : 1 - p_math - p_history + p_math * p_history = 1/4 := by
  sorry

end kim_no_tests_probability_l989_98917


namespace chord_length_in_circle_l989_98938

theorem chord_length_in_circle (r d : ℝ) (hr : r = 5) (hd : d = 3) :
  let half_chord := Real.sqrt (r^2 - d^2)
  2 * half_chord = 8 := by sorry

end chord_length_in_circle_l989_98938


namespace total_chestnuts_weight_l989_98959

/-- The weight of chestnuts Eun-soo picked in kilograms -/
def eun_soo_kg : ℝ := 2

/-- The weight of chestnuts Eun-soo picked in grams (in addition to the kilograms) -/
def eun_soo_g : ℝ := 600

/-- The weight of chestnuts Min-gi picked in grams -/
def min_gi_g : ℝ := 3700

/-- The conversion factor from kilograms to grams -/
def kg_to_g : ℝ := 1000

theorem total_chestnuts_weight : 
  eun_soo_kg * kg_to_g + eun_soo_g + min_gi_g = 6300 := by
  sorry

end total_chestnuts_weight_l989_98959


namespace mayoral_election_votes_l989_98986

theorem mayoral_election_votes 
  (votes_Z : ℕ) 
  (h1 : votes_Z = 25000)
  (votes_Y : ℕ) 
  (h2 : votes_Y = votes_Z - (2 / 5 : ℚ) * votes_Z)
  (votes_X : ℕ) 
  (h3 : votes_X = votes_Y + (1 / 2 : ℚ) * votes_Y) :
  votes_X = 22500 := by
sorry

end mayoral_election_votes_l989_98986


namespace boys_usual_time_to_school_l989_98975

/-- Proves that if a boy walks at 7/6 of his usual rate and arrives at school 4 minutes early, 
    his usual time to reach school is 28 minutes. -/
theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) 
  (h2 : usual_time > 0) 
  (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 4)) : 
  usual_time = 28 := by
sorry

end boys_usual_time_to_school_l989_98975


namespace congruence_problem_l989_98976

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (6 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (8 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 3 := by
  sorry

end congruence_problem_l989_98976


namespace profit_equals_700_at_5_profit_equals_original_at_3_total_profit_after_two_years_l989_98984

-- Define the profit function
def profit_function (x : ℝ) : ℝ := 10 * x^2 + 90 * x

-- Define the original monthly profit
def original_monthly_profit : ℝ := 1.2

-- Theorem 1
theorem profit_equals_700_at_5 :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ profit_function x = 700 ∧ x = 5 :=
sorry

-- Theorem 2
theorem profit_equals_original_at_3 :
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 12 ∧ profit_function x = 120 * x ∧ x = 3 :=
sorry

-- Theorem 3
theorem total_profit_after_two_years :
  12 * (10 * 12 + 90) + 12 * 320 = 6360 :=
sorry

end profit_equals_700_at_5_profit_equals_original_at_3_total_profit_after_two_years_l989_98984


namespace train_length_l989_98934

theorem train_length (time : Real) (speed_kmh : Real) (length : Real) : 
  time = 2.222044458665529 →
  speed_kmh = 162 →
  length = speed_kmh * (1000 / 3600) * time →
  length = 100 := by
sorry


end train_length_l989_98934


namespace common_ratio_of_geometric_series_l989_98923

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 5 / 3
  | 1 => 30 / 7
  | 2 => 180 / 49
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, ∀ n : ℕ, n > 0 → geometric_series (n + 1) = r * geometric_series n :=
sorry

end common_ratio_of_geometric_series_l989_98923


namespace q_sum_l989_98946

/-- Given a function q: ℝ → ℝ where q(1) = 3, prove that q(1) + q(2) = 8 -/
theorem q_sum (q : ℝ → ℝ) (h : q 1 = 3) : q 1 + q 2 = 8 := by
  sorry

end q_sum_l989_98946


namespace expand_and_simplify_l989_98948

theorem expand_and_simplify (n b : ℝ) : (n + 2*b)^2 - 4*b^2 = n^2 + 4*n*b := by
  sorry

end expand_and_simplify_l989_98948


namespace benny_candy_bars_l989_98982

/-- The number of candy bars Benny bought -/
def num_candy_bars : ℕ := sorry

/-- The cost of the soft drink in dollars -/
def soft_drink_cost : ℕ := 2

/-- The cost of each candy bar in dollars -/
def candy_bar_cost : ℕ := 5

/-- The total amount Benny spent in dollars -/
def total_spent : ℕ := 27

theorem benny_candy_bars :
  soft_drink_cost + num_candy_bars * candy_bar_cost = total_spent ∧
  num_candy_bars = 5 := by sorry

end benny_candy_bars_l989_98982


namespace total_canoes_april_l989_98950

def canoe_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else initial * (3^months - 1) / 2

theorem total_canoes_april : canoe_production 5 4 = 200 := by
  sorry

end total_canoes_april_l989_98950


namespace only_point_0_neg2_satisfies_l989_98951

def point_satisfies_inequalities (x y : ℝ) : Prop :=
  x + y - 1 < 0 ∧ x - y + 1 > 0

theorem only_point_0_neg2_satisfies : 
  ¬(point_satisfies_inequalities 0 2) ∧
  ¬(point_satisfies_inequalities (-2) 0) ∧
  point_satisfies_inequalities 0 (-2) ∧
  ¬(point_satisfies_inequalities 2 0) :=
sorry

end only_point_0_neg2_satisfies_l989_98951


namespace parallel_segments_y_coordinate_l989_98921

/-- Given four points A, B, X, and Y in a 2D plane, where segment AB is parallel to segment XY,
    prove that the y-coordinate of Y is -1. -/
theorem parallel_segments_y_coordinate
  (A B X Y : ℝ × ℝ)
  (hA : A = (-2, -2))
  (hB : B = (2, -6))
  (hX : X = (1, 5))
  (hY : Y = (7, Y.2))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (Y.1 - X.1) * (B.2 - A.2)) :
  Y.2 = -1 :=
sorry

end parallel_segments_y_coordinate_l989_98921


namespace max_oranges_donated_l989_98997

theorem max_oranges_donated (n : ℕ) : ∃ (q r : ℕ), n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
sorry

end max_oranges_donated_l989_98997


namespace three_digit_number_property_l989_98918

theorem three_digit_number_property (a b c : ℕ) : 
  a ≠ 0 → 
  a < 10 → b < 10 → c < 10 →
  10 * b + c = 8 * a →
  10 * a + b = 8 * c →
  (10 * a + c) / b = 17 :=
sorry

end three_digit_number_property_l989_98918


namespace investment_return_is_25_percent_l989_98941

/-- Calculates the percentage return on investment for a given dividend rate, face value, and purchase price of shares. -/
def percentageReturn (dividendRate : ℚ) (faceValue : ℚ) (purchasePrice : ℚ) : ℚ :=
  (dividendRate * faceValue / purchasePrice) * 100

/-- Theorem stating that for the given conditions, the percentage return on investment is 25%. -/
theorem investment_return_is_25_percent :
  let dividendRate : ℚ := 125 / 1000
  let faceValue : ℚ := 40
  let purchasePrice : ℚ := 20
  percentageReturn dividendRate faceValue purchasePrice = 25 := by
  sorry

end investment_return_is_25_percent_l989_98941


namespace intersection_A_B_intersection_complement_A_B_l989_98990

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for (¬ᵤA) ∩ (¬ᵤB)
theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} := by sorry

end intersection_A_B_intersection_complement_A_B_l989_98990


namespace old_socks_thrown_away_l989_98960

def initial_socks : ℕ := 11
def new_socks : ℕ := 26
def final_socks : ℕ := 33

theorem old_socks_thrown_away : 
  initial_socks + new_socks - final_socks = 4 := by
  sorry

end old_socks_thrown_away_l989_98960


namespace angle_sum_inequality_l989_98952

theorem angle_sum_inequality (α β γ x y z : ℝ) 
  (h_angles : α + β + γ = Real.pi)
  (h_sum : x + y + z = 0) :
  y * z * Real.sin α ^ 2 + z * x * Real.sin β ^ 2 + x * y * Real.sin γ ^ 2 ≤ 0 := by
  sorry

end angle_sum_inequality_l989_98952


namespace eugene_payment_l989_98925

/-- The cost of a single T-shirt in dollars -/
def tshirt_cost : ℚ := 20

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℚ := 80

/-- The cost of a single pair of shoes in dollars -/
def shoes_cost : ℚ := 150

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The number of pairs of shoes Eugene buys -/
def num_shoes : ℕ := 2

/-- The total cost before discount -/
def total_cost_before_discount : ℚ :=
  tshirt_cost * num_tshirts + pants_cost * num_pants + shoes_cost * num_shoes

/-- The amount Eugene has to pay after the discount -/
def amount_to_pay : ℚ := total_cost_before_discount * (1 - discount_rate)

theorem eugene_payment :
  amount_to_pay = 558 := by sorry

end eugene_payment_l989_98925


namespace cubic_root_ratio_l989_98944

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 3 ∨ x = 4 ∨ x = 5) :
  c / d = 47 / 60 := by
sorry

end cubic_root_ratio_l989_98944


namespace five_people_handshakes_l989_98937

/-- The number of handshakes in a group of n people where each person
    shakes hands with every other person exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 5 people, where each person shakes hands with
    every other person exactly once, the total number of handshakes is 10 -/
theorem five_people_handshakes :
  handshakes 5 = 10 := by
  sorry

#eval handshakes 5  -- To verify the result

end five_people_handshakes_l989_98937


namespace average_weight_of_class_l989_98907

/-- The average weight of a class with two sections -/
theorem average_weight_of_class 
  (studentsA : ℕ) (studentsB : ℕ) 
  (avgWeightA : ℚ) (avgWeightB : ℚ) :
  studentsA = 40 →
  studentsB = 30 →
  avgWeightA = 50 →
  avgWeightB = 60 →
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB : ℚ) = 3800 / 70 := by
  sorry

#eval (3800 : ℚ) / 70

end average_weight_of_class_l989_98907


namespace jungkook_weight_proof_l989_98971

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

/-- Jungkook's base weight in kilograms -/
def base_weight_kg : ℕ := 54

/-- Additional weight in grams -/
def additional_weight_g : ℕ := 154

/-- Jungkook's total weight in grams -/
def jungkook_weight_g : ℕ := base_weight_kg * kg_to_g + additional_weight_g

theorem jungkook_weight_proof : jungkook_weight_g = 54154 := by
  sorry

end jungkook_weight_proof_l989_98971


namespace probability_two_red_balls_l989_98933

/-- The probability of picking 2 red balls from a bag containing 3 red balls, 4 blue balls, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 4 →
  green_balls = 4 →
  (red_balls.choose 2 : ℚ) / (total_balls.choose 2) = 3 / 55 := by
  sorry

end probability_two_red_balls_l989_98933
