import Mathlib

namespace desk_chair_prices_l408_40848

/-- Proves that given a desk and chair set with a total price of 115 yuan,
    where the chair is 45 yuan cheaper than the desk,
    the price of the chair is 35 yuan and the price of the desk is 80 yuan. -/
theorem desk_chair_prices (total_price : ℕ) (price_difference : ℕ)
    (h1 : total_price = 115)
    (h2 : price_difference = 45) :
    ∃ (chair_price desk_price : ℕ),
      chair_price = 35 ∧
      desk_price = 80 ∧
      chair_price + desk_price = total_price ∧
      desk_price = chair_price + price_difference :=
by
  sorry

end desk_chair_prices_l408_40848


namespace win_sector_area_l408_40826

/-- Proves that for a circular spinner with given radius and winning probabilities,
    the combined area of winning sectors is as calculated. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 15) (h2 : p = 1/6) :
  2 * p * π * r^2 = 75 * π := by
  sorry

end win_sector_area_l408_40826


namespace complex_sum_exponential_form_l408_40818

theorem complex_sum_exponential_form :
  10 * Complex.exp (2 * π * I / 11) + 10 * Complex.exp (15 * π * I / 22) =
  10 * Real.sqrt 2 * Complex.exp (19 * π * I / 44) := by
  sorry

end complex_sum_exponential_form_l408_40818


namespace hyperbola_equation_l408_40882

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation --/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ k : ℝ, (a / b = Real.sqrt 3) ∧ 
   (∃ x y : ℝ, x^2 = 24*y ∧ (y^2 / a^2 - x^2 / b^2 = 1))) →
  (∃ x y : ℝ, y^2 / 27 - x^2 / 9 = 1) :=
sorry

end hyperbola_equation_l408_40882


namespace largest_consecutive_composites_l408_40851

theorem largest_consecutive_composites : ∃ (n : ℕ), 
  (n ≤ 36) ∧ 
  (∀ i ∈ Finset.range 7, 30 ≤ n - i ∧ n - i < 40 ∧ ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → 
    ¬(∀ i ∈ Finset.range 7, 30 ≤ m - i ∧ m - i < 40 ∧ ¬(Nat.Prime (m - i)))) :=
by sorry

end largest_consecutive_composites_l408_40851


namespace taxi_problem_l408_40801

def taxi_distances : List Int := [9, -3, -5, 4, 8, 6, 3, -6, -4, 10]
def price_per_km : ℝ := 2.4

theorem taxi_problem (distances : List Int) (price : ℝ) 
  (h_distances : distances = taxi_distances) (h_price : price = price_per_km) :
  (distances.sum = 22) ∧ 
  ((distances.map Int.natAbs).sum * price = 139.2) := by
  sorry

end taxi_problem_l408_40801


namespace sqrt_meaningful_l408_40850

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_meaningful_l408_40850


namespace square_park_circumference_l408_40889

/-- The circumference of a square park with side length 5 kilometers is 20 kilometers. -/
theorem square_park_circumference :
  ∀ (side_length circumference : ℝ),
  side_length = 5 →
  circumference = 4 * side_length →
  circumference = 20 :=
by
  sorry

end square_park_circumference_l408_40889


namespace expression_equals_two_power_thirty_l408_40871

theorem expression_equals_two_power_thirty :
  (((16^16 / 16^14)^3 * 8^6) / 2^12) = 2^30 := by
  sorry

end expression_equals_two_power_thirty_l408_40871


namespace at_least_two_equations_have_real_solutions_l408_40807

theorem at_least_two_equations_have_real_solutions (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  let eq1 := fun x => (x - a) * (x - b) = x - c
  let eq2 := fun x => (x - c) * (x - b) = x - a
  let eq3 := fun x => (x - a) * (x - c) = x - b
  let has_real_solution := fun f => ∃ x : ℝ, f x
  (has_real_solution eq1 ∧ has_real_solution eq2) ∨
  (has_real_solution eq1 ∧ has_real_solution eq3) ∨
  (has_real_solution eq2 ∧ has_real_solution eq3) :=
by sorry

end at_least_two_equations_have_real_solutions_l408_40807


namespace exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l408_40881

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def exactlyOneBlack (outcome : SampleSpace) : Prop :=
  (outcome.1 = 2 ∧ outcome.2 = 0) ∨ (outcome.1 = 2 ∧ outcome.2 = 1) ∨
  (outcome.1 = 3 ∧ outcome.2 = 0) ∨ (outcome.1 = 3 ∧ outcome.2 = 1)

def bothBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∧ outcome.2 = 3

def atLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 = 2 ∨ outcome.1 = 3 ∨ outcome.2 = 2 ∨ outcome.2 = 3

def bothRed (outcome : SampleSpace) : Prop :=
  outcome.1 = 0 ∧ outcome.2 = 1

-- Theorem statements
theorem exactlyOneBlack_bothBlack_mutuallyExclusive :
  ∀ (outcome : SampleSpace), ¬(exactlyOneBlack outcome ∧ bothBlack outcome) :=
sorry

theorem atLeastOneBlack_bothRed_complementary :
  ∀ (outcome : SampleSpace), atLeastOneBlack outcome ↔ ¬(bothRed outcome) :=
sorry

end exactlyOneBlack_bothBlack_mutuallyExclusive_atLeastOneBlack_bothRed_complementary_l408_40881


namespace nested_sqrt_fraction_l408_40840

/-- Given a real number x satisfying the equation x = 2 + √3 / x,
    prove that 1 / ((x + 2)(x - 3)) = (√3 + 5) / (-22) -/
theorem nested_sqrt_fraction (x : ℝ) (hx : x = 2 + Real.sqrt 3 / x) :
  1 / ((x + 2) * (x - 3)) = (Real.sqrt 3 + 5) / (-22) := by
  sorry

end nested_sqrt_fraction_l408_40840


namespace max_constant_quadratic_real_roots_l408_40860

theorem max_constant_quadratic_real_roots :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 6*x + c = 0) → c ≤ 9 :=
by sorry

end max_constant_quadratic_real_roots_l408_40860


namespace largest_b_value_l408_40865

theorem largest_b_value (b : ℝ) : 
  (3 * b + 4) * (b - 3) = 9 * b → 
  b ≤ (4 + 4 * Real.sqrt 5) / 6 ∧ 
  ∃ (b : ℝ), (3 * b + 4) * (b - 3) = 9 * b ∧ b = (4 + 4 * Real.sqrt 5) / 6 := by
  sorry

end largest_b_value_l408_40865


namespace percentage_loss_l408_40892

/-- Calculate the percentage of loss in a sale transaction -/
theorem percentage_loss (cost_price selling_price : ℚ) (h1 : cost_price = 1800) (h2 : selling_price = 1620) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end percentage_loss_l408_40892


namespace no_solution_iff_k_equals_seven_l408_40843

theorem no_solution_iff_k_equals_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end no_solution_iff_k_equals_seven_l408_40843


namespace cubic_sum_minus_product_l408_40894

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 50) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1125 := by
  sorry

end cubic_sum_minus_product_l408_40894


namespace polygon_with_1080_degrees_is_octagon_l408_40873

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 :=
by sorry

end polygon_with_1080_degrees_is_octagon_l408_40873


namespace xy_inequality_l408_40888

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end xy_inequality_l408_40888


namespace gcd_divisibility_l408_40884

theorem gcd_divisibility (p q r s : ℕ+) : 
  (Nat.gcd p.val q.val = 21) →
  (Nat.gcd q.val r.val = 45) →
  (Nat.gcd r.val s.val = 75) →
  (120 < Nat.gcd s.val p.val) →
  (Nat.gcd s.val p.val < 180) →
  9 ∣ p.val :=
by sorry

end gcd_divisibility_l408_40884


namespace multiply_both_sides_by_x_minus_3_l408_40879

variable (f g : ℝ → ℝ)
variable (x : ℝ)

theorem multiply_both_sides_by_x_minus_3 :
  f x = g x → (x - 3) * f x = (x - 3) * g x := by
  sorry

end multiply_both_sides_by_x_minus_3_l408_40879


namespace star_operation_result_l408_40885

-- Define the operation *
def star : Fin 4 → Fin 4 → Fin 4
| 1, 1 => 1 | 1, 2 => 2 | 1, 3 => 3 | 1, 4 => 4
| 2, 1 => 2 | 2, 2 => 4 | 2, 3 => 1 | 2, 4 => 3
| 3, 1 => 3 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 2
| 4, 1 => 4 | 4, 2 => 3 | 4, 3 => 2 | 4, 4 => 1

-- State the theorem
theorem star_operation_result : star 4 (star 3 2) = 4 := by sorry

end star_operation_result_l408_40885


namespace tangent_line_equation_l408_40862

/-- The equation of the tangent line to y = x³ + x + 1 at (1, 3) is 4x - y - 1 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + x + 1
  let point : ℝ × ℝ := (1, 3)
  let tangent_line (x y : ℝ) := 4*x - y - 1 = 0
  (∀ x, tangent_line x (f x)) ∧ tangent_line point.1 point.2 := by
  sorry


end tangent_line_equation_l408_40862


namespace min_value_x_plus_y_l408_40816

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x - x*y + 6*y = 0) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2*z - z*w + 6*w = 0 → x + y ≤ z + w ∧ x + y = 8 + 4 * Real.sqrt 3 :=
by sorry

end min_value_x_plus_y_l408_40816


namespace base6_addition_subtraction_l408_40891

/-- Converts a base 6 number to its decimal representation -/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to its base 6 representation -/
def decimalToBase6 (n : ℕ) : ℕ := sorry

theorem base6_addition_subtraction :
  decimalToBase6 ((base6ToDecimal 35 + base6ToDecimal 14) - base6ToDecimal 20) = 33 := by sorry

end base6_addition_subtraction_l408_40891


namespace prime_arithmetic_progression_difference_l408_40822

theorem prime_arithmetic_progression_difference (a : ℕ → ℕ) (d : ℕ) :
  (∀ k, k ∈ Finset.range 15 → Nat.Prime (a k)) →
  (∀ k, k ∈ Finset.range 14 → a (k + 1) = a k + d) →
  (∀ k l, k < l → k ∈ Finset.range 15 → l ∈ Finset.range 15 → a k < a l) →
  d > 30000 := by
  sorry

end prime_arithmetic_progression_difference_l408_40822


namespace probability_product_216_l408_40863

/-- A standard die has 6 faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := Finset.range 6

/-- The probability of an event occurring when rolling a standard die. -/
def DieProbability (event : Finset ℕ) : ℚ :=
  event.card / StandardDie.card

/-- The product of three numbers obtained from rolling three standard dice. -/
def ThreeDiceProduct (a b c : ℕ) : ℕ := a * b * c

/-- The event of rolling three sixes on three standard dice. -/
def ThreeSixes : Finset (ℕ × ℕ × ℕ) :=
  {(6, 6, 6)}

theorem probability_product_216 :
  DieProbability (ThreeSixes.image (fun (a, b, c) => ThreeDiceProduct a b c)) = 1 / 216 :=
sorry

end probability_product_216_l408_40863


namespace sin_45_is_proposition_l408_40825

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s ≠ "") ∧ (truth_value = true ∨ truth_value = false)

-- State the theorem
theorem sin_45_is_proposition : 
  is_proposition "sin(45°) = 1" := by
  sorry

end sin_45_is_proposition_l408_40825


namespace rectangular_prism_width_l408_40821

/-- The width of a rectangular prism with given dimensions and diagonal length -/
theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w ^ 2 = 39 ∧ d ^ 2 = l ^ 2 + w ^ 2 + h ^ 2 := by
  sorry

end rectangular_prism_width_l408_40821


namespace x_percent_of_z_l408_40812

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.50 * z) : x = 0.60 * z := by
  sorry

end x_percent_of_z_l408_40812


namespace probability_divisible_by_five_l408_40809

/- Define the spinner outcomes -/
def spinner : Finset ℕ := {1, 2, 4, 5}

/- Define a function to check if a number is divisible by 5 -/
def divisible_by_five (n : ℕ) : Bool :=
  n % 5 = 0

/- Define a function to create a three-digit number from three spins -/
def make_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

/- Main theorem -/
theorem probability_divisible_by_five :
  (Finset.filter (fun n => divisible_by_five (make_number n.1 n.2.1 n.2.2))
    (spinner.product (spinner.product spinner))).card /
  (spinner.product (spinner.product spinner)).card = 1 / 4 := by
  sorry

end probability_divisible_by_five_l408_40809


namespace min_value_h_l408_40831

theorem min_value_h (x : ℝ) (hx : x > 0) :
  x^2 + 1/x^2 + 1/(x^2 + 1/x^2) ≥ 2.5 := by
  sorry

end min_value_h_l408_40831


namespace inverse_proportion_l408_40864

/-- Given two real numbers x and y that are inversely proportional,
    prove that if x + y = 30 and x - y = 10, then y = 50 when x = 4 -/
theorem inverse_proportion (x y : ℝ) (h1 : ∃ k : ℝ, x * y = k)
    (h2 : x + y = 30) (h3 : x - y = 10) : 
    x = 4 → y = 50 := by
  sorry

end inverse_proportion_l408_40864


namespace woodchopper_theorem_l408_40813

/-- A woodchopper who gets a certain number of wood blocks per tree and chops a certain number of trees per day -/
structure Woodchopper where
  blocks_per_tree : ℕ
  trees_per_day : ℕ

/-- Calculate the total number of wood blocks obtained after a given number of days -/
def total_blocks (w : Woodchopper) (days : ℕ) : ℕ :=
  w.blocks_per_tree * w.trees_per_day * days

/-- Theorem: A woodchopper who gets 3 blocks per tree and chops 2 trees per day obtains 30 blocks after 5 days -/
theorem woodchopper_theorem :
  let ragnar : Woodchopper := { blocks_per_tree := 3, trees_per_day := 2 }
  total_blocks ragnar 5 = 30 := by sorry

end woodchopper_theorem_l408_40813


namespace angle_ABC_is_30_l408_40830

-- Define the angles
def angle_CBD : ℝ := 90
def angle_ABD : ℝ := 60

-- Theorem statement
theorem angle_ABC_is_30 :
  ∀ (angle_ABC : ℝ),
  angle_ABD + angle_ABC + angle_CBD = 180 →
  angle_ABC = 30 :=
by
  sorry

end angle_ABC_is_30_l408_40830


namespace sugar_water_concentration_l408_40855

theorem sugar_water_concentration (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end sugar_water_concentration_l408_40855


namespace triangle_arithmetic_sequence_angles_l408_40842

theorem triangle_arithmetic_sequence_angles (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- angles are positive
  A + B + C = 180 →  -- sum of angles in a triangle
  ∃ (d : ℝ), C - B = B - A →  -- arithmetic sequence condition
  B = 60 := by sorry

end triangle_arithmetic_sequence_angles_l408_40842


namespace negation_existence_positive_real_l408_40852

theorem negation_existence_positive_real (R_plus : Set ℝ) :
  (¬ ∃ x ∈ R_plus, x > x^2) ↔ (∀ x ∈ R_plus, x ≤ x^2) :=
by sorry

end negation_existence_positive_real_l408_40852


namespace license_plate_count_l408_40839

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of distinct letters that are repeated -/
def repeated_letters : ℕ := 2

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ := 7776000

theorem license_plate_count :
  (Nat.choose alphabet_size repeated_letters) *
  (alphabet_size - repeated_letters) *
  (Nat.choose letter_positions repeated_letters) *
  (Nat.choose (letter_positions - repeated_letters) repeated_letters) *
  (Nat.factorial digit_positions) = license_plate_combinations :=
by sorry

end license_plate_count_l408_40839


namespace ellipse_properties_l408_40800

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

/-- Theorem about the properties of the ellipse -/
theorem ellipse_properties :
  ∃ (a b c : ℝ),
    a = 5 ∧ b = 4 ∧ c = 3 ∧
    (∀ x y, is_ellipse x y →
      (2 * a = 10 ∧ b = 4) ∧
      (is_ellipse (-c) 0 ∧ is_ellipse c 0) ∧
      (is_ellipse (-a) 0 ∧ is_ellipse a 0 ∧ is_ellipse 0 b ∧ is_ellipse 0 (-b))) :=
by sorry

end ellipse_properties_l408_40800


namespace min_flowers_for_outstanding_pioneer_l408_40896

/-- Represents the number of small red flowers needed for one small red flag -/
def flowers_per_flag : ℕ := 5

/-- Represents the number of small red flags needed for one badge -/
def flags_per_badge : ℕ := 4

/-- Represents the number of badges needed for one small gold cup -/
def badges_per_cup : ℕ := 3

/-- Represents the number of small gold cups needed to be an outstanding Young Pioneer -/
def cups_needed : ℕ := 2

/-- Theorem stating the minimum number of small red flowers needed to be an outstanding Young Pioneer -/
theorem min_flowers_for_outstanding_pioneer : 
  cups_needed * badges_per_cup * flags_per_badge * flowers_per_flag = 120 := by
  sorry

end min_flowers_for_outstanding_pioneer_l408_40896


namespace log_43_between_consecutive_integers_l408_40866

theorem log_43_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 43 / Real.log 10 ∧ Real.log 43 / Real.log 10 < d ∧ c + d = 3 := by
  sorry

end log_43_between_consecutive_integers_l408_40866


namespace sonia_and_joss_moving_l408_40833

/-- Calculates the time spent filling the car per trip given the total moving time,
    number of trips, and driving time per trip. -/
def time_filling_car_per_trip (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) : ℕ :=
  let total_minutes := total_moving_time * 60
  let total_driving_time := driving_time_per_trip * num_trips
  let total_filling_time := total_minutes - total_driving_time
  total_filling_time / num_trips

/-- Theorem stating that given the specific conditions of the problem,
    the time spent filling the car per trip is 40 minutes. -/
theorem sonia_and_joss_moving (total_moving_time : ℕ) (num_trips : ℕ) (driving_time_per_trip : ℕ) :
  total_moving_time = 7 →
  num_trips = 6 →
  driving_time_per_trip = 30 →
  time_filling_car_per_trip total_moving_time num_trips driving_time_per_trip = 40 :=
by
  sorry

#eval time_filling_car_per_trip 7 6 30

end sonia_and_joss_moving_l408_40833


namespace saree_price_calculation_l408_40817

theorem saree_price_calculation (final_price : ℝ) 
  (h1 : final_price = 224) 
  (discount1 : ℝ) (h2 : discount1 = 0.3)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end saree_price_calculation_l408_40817


namespace unique_prime_power_of_four_minus_one_l408_40883

theorem unique_prime_power_of_four_minus_one :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (4^n - 1) :=
sorry

end unique_prime_power_of_four_minus_one_l408_40883


namespace recurring_decimal_one_zero_six_l408_40877

-- Define the recurring decimal notation
def recurring_decimal (whole : ℕ) (recurring : ℕ) : ℚ :=
  whole + (recurring : ℚ) / 99

-- State the theorem
theorem recurring_decimal_one_zero_six :
  recurring_decimal 1 6 = 35 / 33 :=
by
  -- The proof would go here
  sorry

end recurring_decimal_one_zero_six_l408_40877


namespace gcd_2814_1806_l408_40841

theorem gcd_2814_1806 : Nat.gcd 2814 1806 = 42 := by
  sorry

end gcd_2814_1806_l408_40841


namespace smallest_n_and_y_over_x_l408_40838

theorem smallest_n_and_y_over_x :
  ∃ (n : ℕ+) (x y : ℝ), 
    x > 0 ∧ y > 0 ∧
    (Complex.I : ℂ)^2 = -1 ∧
    (x + 2*y*Complex.I)^(n:ℕ) = (x - 2*y*Complex.I)^(n:ℕ) ∧
    (∀ (m : ℕ+), m < n → ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (a + 2*b*Complex.I)^(m:ℕ) = (a - 2*b*Complex.I)^(m:ℕ)) ∧
    n = 3 ∧
    y / x = Real.sqrt 3 / 2 :=
by sorry

end smallest_n_and_y_over_x_l408_40838


namespace abs_zero_iff_eq_l408_40899

theorem abs_zero_iff_eq (y : ℚ) : |5 * y - 7| = 0 ↔ y = 7 / 5 := by sorry

end abs_zero_iff_eq_l408_40899


namespace similar_polygons_perimeter_l408_40805

theorem similar_polygons_perimeter (A₁ A₂ P₁ P₂ : ℝ) : 
  A₁ / A₂ = 1 / 16 →  -- ratio of areas
  P₂ - P₁ = 9 →       -- difference in perimeters
  P₁ = 3 :=           -- perimeter of smaller polygon
by sorry

end similar_polygons_perimeter_l408_40805


namespace square_wire_length_l408_40845

theorem square_wire_length (area : ℝ) (side_length : ℝ) (wire_length : ℝ) : 
  area = 324 → 
  area = side_length ^ 2 → 
  wire_length = 4 * side_length → 
  wire_length = 72 := by
sorry

end square_wire_length_l408_40845


namespace max_k_is_19_l408_40834

/-- Represents a two-digit number -/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9

/-- Represents a three-digit number formed by inserting a digit between two others -/
def ThreeDigitNumber (a c b : Nat) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ c ≤ 9 ∧ b ≤ 9

/-- The value of a two-digit number -/
def twoDigitValue (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number -/
def threeDigitValue (a c b : Nat) : Nat :=
  100 * a + 10 * c + b

/-- The theorem stating that the maximum value of k is 19 -/
theorem max_k_is_19 :
  ∀ a b c k : Nat,
  TwoDigitNumber a b →
  ThreeDigitNumber a c b →
  threeDigitValue a c b = k * twoDigitValue a b →
  k ≤ 19 :=
sorry

end max_k_is_19_l408_40834


namespace arithmetic_mean_of_fractions_l408_40867

theorem arithmetic_mean_of_fractions : 
  let a := 9/12
  let b := 5/6
  let c := 7/8
  b = (a + c) / 2 := by sorry

end arithmetic_mean_of_fractions_l408_40867


namespace quadrant_line_relationships_l408_40898

/-- A line passing through the first, second, and fourth quadrants -/
structure QuadrantLine where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_quadrants : 
    ∃ (x₁ y₁ x₂ y₂ x₄ y₄ : ℝ),
      (x₁ > 0 ∧ y₁ > 0 ∧ a * x₁ + b * y₁ + c = 0) ∧
      (x₂ < 0 ∧ y₂ > 0 ∧ a * x₂ + b * y₂ + c = 0) ∧
      (x₄ > 0 ∧ y₄ < 0 ∧ a * x₄ + b * y₄ + c = 0)

/-- The relationships between a, b, and c for a line passing through the first, second, and fourth quadrants -/
theorem quadrant_line_relationships (l : QuadrantLine) : l.a * l.b > 0 ∧ l.b * l.c < 0 := by
  sorry

end quadrant_line_relationships_l408_40898


namespace merchant_profit_theorem_l408_40895

/-- Calculate the profit for a single item -/
def calculate_profit (purchase_price markup_percent discount_percent : ℚ) : ℚ :=
  let selling_price := purchase_price * (1 + markup_percent / 100)
  let discounted_price := selling_price * (1 - discount_percent / 100)
  discounted_price - purchase_price

/-- Calculate the total gross profit for three items -/
def total_gross_profit (
  jacket_price jeans_price shirt_price : ℚ)
  (jacket_markup jeans_markup shirt_markup : ℚ)
  (jacket_discount jeans_discount shirt_discount : ℚ) : ℚ :=
  calculate_profit jacket_price jacket_markup jacket_discount +
  calculate_profit jeans_price jeans_markup jeans_discount +
  calculate_profit shirt_price shirt_markup shirt_discount

theorem merchant_profit_theorem :
  total_gross_profit 60 45 30 25 30 15 20 10 5 = 10.43 := by
  sorry

end merchant_profit_theorem_l408_40895


namespace candidates_per_state_l408_40875

theorem candidates_per_state (total : ℕ) (selected_A selected_B : ℕ) 
  (h1 : selected_A = total * 6 / 100)
  (h2 : selected_B = total * 7 / 100)
  (h3 : selected_B = selected_A + 79) :
  total = 7900 := by
  sorry

end candidates_per_state_l408_40875


namespace uniform_count_l408_40811

theorem uniform_count (pants_cost shirt_cost tie_cost socks_cost total_spend : ℚ) 
  (h1 : pants_cost = 20)
  (h2 : shirt_cost = 2 * pants_cost)
  (h3 : tie_cost = shirt_cost / 5)
  (h4 : socks_cost = 3)
  (h5 : total_spend = 355) :
  (total_spend / (pants_cost + shirt_cost + tie_cost + socks_cost) : ℚ) = 5 := by
  sorry

end uniform_count_l408_40811


namespace max_markable_nodes_6x6_l408_40808

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- A node in the grid -/
structure Node :=
  (x : Nat)
  (y : Nat)

/-- Checks if a node is on the edge of the grid -/
def isEdgeNode (g : Grid) (n : Node) : Bool :=
  n.x = 0 || n.x = g.size || n.y = 0 || n.y = g.size

/-- Checks if a node is a corner node -/
def isCornerNode (g : Grid) (n : Node) : Bool :=
  (n.x = 0 || n.x = g.size) && (n.y = 0 || n.y = g.size)

/-- Counts the number of nodes in a grid -/
def nodeCount (g : Grid) : Nat :=
  (g.size + 1) * (g.size + 1)

/-- Theorem: The maximum number of markable nodes in a 6x6 grid is 45 -/
theorem max_markable_nodes_6x6 (g : Grid) (h : g.size = 6) :
  nodeCount g - (4 : Nat) = 45 := by
  sorry

#check max_markable_nodes_6x6

end max_markable_nodes_6x6_l408_40808


namespace confidence_interval_for_population_mean_l408_40846

-- Define the sample data
def sample_data : List (Float × Nat) := [(-2, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 1)]

-- Define the sample size
def n : Nat := 10

-- Define the confidence level
def confidence_level : Float := 0.95

-- Define the critical t-value for 9 degrees of freedom and 95% confidence
def t_critical : Float := 2.262

-- State the theorem
theorem confidence_interval_for_population_mean :
  let sample_mean := (sample_data.map (λ (x, freq) => x * freq.toFloat)).sum / n.toFloat
  let sample_variance := (sample_data.map (λ (x, freq) => freq.toFloat * (x - sample_mean)^2)).sum / (n.toFloat - 1)
  let sample_std_dev := sample_variance.sqrt
  let margin_of_error := t_critical * (sample_std_dev / (n.toFloat.sqrt))
  0.363 < sample_mean - margin_of_error ∧ sample_mean + margin_of_error < 3.837 := by
  sorry


end confidence_interval_for_population_mean_l408_40846


namespace fraction_simplification_l408_40814

theorem fraction_simplification (x : ℝ) 
  (h1 : x + 1 ≠ 0) (h2 : 2 + x ≠ 0) (h3 : 2 - x ≠ 0) (h4 : x = 0) : 
  (x^2 - 4*x + 4) / (x + 1) / ((3 / (x + 1)) - x + 1) = 1 := by
  sorry

end fraction_simplification_l408_40814


namespace temperature_conversion_fraction_l408_40823

theorem temperature_conversion_fraction : 
  ∀ (t k : ℝ) (fraction : ℝ),
    t = fraction * (k - 32) →
    (t = 20 ∧ k = 68) →
    fraction = 5 / 9 := by
  sorry

end temperature_conversion_fraction_l408_40823


namespace ship_speed_problem_l408_40828

theorem ship_speed_problem (speed_diff : ℝ) (time : ℝ) (final_distance : ℝ) :
  speed_diff = 3 →
  time = 2 →
  final_distance = 174 →
  ∃ (speed1 speed2 : ℝ),
    speed2 = speed1 + speed_diff ∧
    (speed1 * time)^2 + (speed2 * time)^2 = final_distance^2 ∧
    speed1 = 60 ∧
    speed2 = 63 := by
  sorry

end ship_speed_problem_l408_40828


namespace cactus_jump_difference_l408_40802

theorem cactus_jump_difference (num_cacti : ℕ) (total_distance : ℝ) 
  (derek_hops_per_gap : ℕ) (rory_jumps_per_gap : ℕ) 
  (h1 : num_cacti = 31) 
  (h2 : total_distance = 3720) 
  (h3 : derek_hops_per_gap = 30) 
  (h4 : rory_jumps_per_gap = 10) : 
  ∃ (diff : ℝ), abs (diff - 8.27) < 0.01 ∧ 
  diff = (total_distance / ((num_cacti - 1) * rory_jumps_per_gap)) - 
         (total_distance / ((num_cacti - 1) * derek_hops_per_gap)) :=
by sorry

end cactus_jump_difference_l408_40802


namespace parallel_vectors_subtraction_l408_40806

def a : ℝ × ℝ := (-1, -3)
def b (t : ℝ) : ℝ × ℝ := (2, t)

theorem parallel_vectors_subtraction :
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b t →
  a - b t = (-3, -9) := by
  sorry

end parallel_vectors_subtraction_l408_40806


namespace calculation_proof_l408_40820

theorem calculation_proof : (3.14 - 1) ^ 0 * (-1/4) ^ (-2) = 16 := by
  sorry

end calculation_proof_l408_40820


namespace square_lake_side_length_l408_40856

/-- Proves the length of each side of a square lake given Jake's swimming and rowing speeds and the time it takes to row around the lake. -/
theorem square_lake_side_length 
  (swimming_speed : ℝ) 
  (rowing_speed : ℝ) 
  (rowing_time : ℝ) 
  (h1 : swimming_speed = 3) 
  (h2 : rowing_speed = 2 * swimming_speed) 
  (h3 : rowing_time = 10) : 
  (rowing_speed * rowing_time) / 4 = 15 := by
  sorry

#check square_lake_side_length

end square_lake_side_length_l408_40856


namespace employee_pay_l408_40893

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 638 →
  x = 1.2 * y →
  total_pay = x + y →
  y = 290 := by
sorry

end employee_pay_l408_40893


namespace gcd_105_88_l408_40829

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end gcd_105_88_l408_40829


namespace closest_to_10_l408_40886

def numbers : List ℝ := [9.998, 10.1, 10.09, 10.001]

def distance_to_10 (x : ℝ) : ℝ := |x - 10|

theorem closest_to_10 : 
  ∀ x ∈ numbers, distance_to_10 10.001 ≤ distance_to_10 x :=
by sorry

end closest_to_10_l408_40886


namespace equation_solutions_l408_40857

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7) ∧
  (∀ x : ℝ, 2*(x+1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) := by
sorry

end equation_solutions_l408_40857


namespace fraction_difference_simplification_l408_40880

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (1011 : ℚ) / 1010 - 1010 / 1011 = (2021 : ℚ) / q ∧ Nat.gcd 2021 q.val = 1 := by
  sorry

end fraction_difference_simplification_l408_40880


namespace bear_ratio_l408_40847

theorem bear_ratio (black_bears : ℕ) (brown_bears : ℕ) (white_bears : ℕ) :
  black_bears = 60 →
  brown_bears = black_bears + 40 →
  black_bears + brown_bears + white_bears = 190 →
  (black_bears : ℚ) / white_bears = 2 / 1 :=
by sorry

end bear_ratio_l408_40847


namespace fraction_equation_solution_l408_40869

theorem fraction_equation_solution : 
  ∀ (A B : ℚ), 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 → 
    (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) → 
  A + B = 31/5 := by
sorry

end fraction_equation_solution_l408_40869


namespace infinitely_many_solutions_l408_40859

theorem infinitely_many_solutions : 
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧ 
  ∀ (x y z : ℤ), (x, y, z) ∈ S → x^2 + y^2 + z^2 = x^3 + y^3 + z^3 :=
sorry

end infinitely_many_solutions_l408_40859


namespace cylinder_base_area_at_different_heights_l408_40878

/-- Represents the properties of a cylinder with constant volume -/
structure Cylinder where
  volume : ℝ
  height : ℝ
  base_area : ℝ
  height_positive : height > 0
  volume_eq : volume = height * base_area

/-- Theorem about the base area of a cylinder with constant volume -/
theorem cylinder_base_area_at_different_heights
  (c : Cylinder)
  (h_initial : c.height = 12)
  (s_initial : c.base_area = 2)
  (h_final : ℝ)
  (h_final_positive : h_final > 0)
  (h_final_value : h_final = 4.8) :
  let s_final := c.volume / h_final
  s_final = 5 := by sorry

end cylinder_base_area_at_different_heights_l408_40878


namespace cube_root_eight_plus_negative_two_power_zero_l408_40810

theorem cube_root_eight_plus_negative_two_power_zero : 
  (8 : ℝ) ^ (1/3) + (-2 : ℝ) ^ 0 = 3 := by sorry

end cube_root_eight_plus_negative_two_power_zero_l408_40810


namespace problem_1_l408_40890

theorem problem_1 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x / y - y / x - (x^2 + y^2) / (x * y) = -2 * y / x :=
sorry

end problem_1_l408_40890


namespace other_number_is_three_l408_40897

theorem other_number_is_three (x y : ℝ) : 
  x + y = 10 → 
  2 * x = 3 * y + 5 → 
  (x = 7 ∨ y = 7) → 
  (x = 3 ∨ y = 3) :=
by sorry

end other_number_is_three_l408_40897


namespace fitness_center_membership_ratio_l408_40837

theorem fitness_center_membership_ratio :
  ∀ (f m : ℕ) (f_avg m_avg total_avg : ℚ),
    f_avg = 45 →
    m_avg = 20 →
    total_avg = 28 →
    (f_avg * f + m_avg * m) / (f + m) = total_avg →
    (f : ℚ) / m = 8 / 17 := by
  sorry

end fitness_center_membership_ratio_l408_40837


namespace mary_younger_than_albert_l408_40870

/-- Proves that Mary is 10 years younger than Albert given the conditions -/
theorem mary_younger_than_albert (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 5 →
  albert_age - mary_age = 10 := by
sorry

end mary_younger_than_albert_l408_40870


namespace min_value_reciprocal_sum_l408_40858

theorem min_value_reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : x + 4 * y = 3) :
  ∀ z w : ℝ, z * w > 0 → z + 4 * w = 3 → (1 / x + 1 / y) ≤ (1 / z + 1 / w) :=
by sorry

end min_value_reciprocal_sum_l408_40858


namespace andy_distance_to_market_l408_40861

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

theorem andy_distance_to_market :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end andy_distance_to_market_l408_40861


namespace rectangle_shorter_side_l408_40815

theorem rectangle_shorter_side 
  (perimeter : ℝ) 
  (area : ℝ) 
  (h_perimeter : perimeter = 60) 
  (h_area : area = 200) :
  ∃ (shorter_side longer_side : ℝ),
    shorter_side ≤ longer_side ∧
    2 * (shorter_side + longer_side) = perimeter ∧
    shorter_side * longer_side = area ∧
    shorter_side = 10 :=
by sorry

end rectangle_shorter_side_l408_40815


namespace sally_balloons_l408_40849

/-- The number of blue balloons each person has -/
structure Balloons where
  joan : ℕ
  sally : ℕ
  jessica : ℕ

/-- The total number of blue balloons -/
def total_balloons (b : Balloons) : ℕ := b.joan + b.sally + b.jessica

/-- Theorem stating Sally's number of balloons -/
theorem sally_balloons (b : Balloons) 
  (h1 : b.joan = 9)
  (h2 : b.jessica = 2)
  (h3 : total_balloons b = 16) :
  b.sally = 5 := by
  sorry

end sally_balloons_l408_40849


namespace reverse_geometric_difference_l408_40824

/-- A 3-digit number is reverse geometric if it has 3 distinct digits which,
    when read from right to left, form a geometric sequence. -/
def is_reverse_geometric (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    0 < r ∧
    (b : ℚ) = c * r ∧
    (a : ℚ) = b * r

def largest_reverse_geometric : ℕ := sorry

def smallest_reverse_geometric : ℕ := sorry

theorem reverse_geometric_difference :
  largest_reverse_geometric - smallest_reverse_geometric = 789 :=
sorry

end reverse_geometric_difference_l408_40824


namespace square_sum_from_linear_and_product_l408_40874

theorem square_sum_from_linear_and_product (x y : ℝ) 
  (h1 : x + 3 * y = 3) (h2 : x * y = -6) : 
  x^2 + 9 * y^2 = 45 := by
  sorry

end square_sum_from_linear_and_product_l408_40874


namespace cube_surface_area_equal_volume_l408_40827

/-- Given a rectangular prism with dimensions 10 × 5 × 24 inches, 
    prove that a cube with the same volume has a surface area of approximately 678 square inches. -/
theorem cube_surface_area_equal_volume (ε : ℝ) (hε : ε > 0) : ∃ (s : ℝ), 
  s^3 = 10 * 5 * 24 ∧ 
  abs (6 * s^2 - 678) < ε :=
by sorry

end cube_surface_area_equal_volume_l408_40827


namespace coffee_shop_revenue_l408_40872

/-- The number of customers who ordered coffee -/
def coffee_customers : ℕ := 7

/-- The price of a cup of coffee in dollars -/
def coffee_price : ℕ := 5

/-- The number of customers who ordered tea -/
def tea_customers : ℕ := 8

/-- The price of a cup of tea in dollars -/
def tea_price : ℕ := 4

/-- The total revenue of the coffee shop in dollars -/
def total_revenue : ℕ := 67

theorem coffee_shop_revenue :
  coffee_customers * coffee_price + tea_customers * tea_price = total_revenue :=
by sorry

end coffee_shop_revenue_l408_40872


namespace alternate_arrangement_count_l408_40854

def number_of_men : ℕ := 2
def number_of_women : ℕ := 2

theorem alternate_arrangement_count :
  (number_of_men = 2 ∧ number_of_women = 2) →
  (∃ (count : ℕ), count = 8 ∧
    count = (number_of_men * number_of_women * 1 * 1) +
            (number_of_women * number_of_men * 1 * 1)) :=
by sorry

end alternate_arrangement_count_l408_40854


namespace money_left_is_five_l408_40804

/-- The cost of the gift in dollars -/
def gift_cost : ℕ := 250

/-- Erika's savings in dollars -/
def erika_savings : ℕ := 155

/-- The cost of the cake in dollars -/
def cake_cost : ℕ := 25

/-- Rick's savings in dollars, defined as half of the gift cost -/
def rick_savings : ℕ := gift_cost / 2

/-- The total savings of Erika and Rick -/
def total_savings : ℕ := erika_savings + rick_savings

/-- The total cost of the gift and cake -/
def total_cost : ℕ := gift_cost + cake_cost

/-- The amount of money left after buying the gift and cake -/
def money_left : ℕ := total_savings - total_cost

theorem money_left_is_five : money_left = 5 := by
  sorry

end money_left_is_five_l408_40804


namespace symmetry_and_rotation_sum_l408_40836

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry 
    and its smallest positive angle of rotational symmetry (in degrees) 
    is equal to 17 + 360/17 -/
theorem symmetry_and_rotation_sum : 
  (L n : ℚ) + R n = 17 + 360 / 17 := by
  sorry

end symmetry_and_rotation_sum_l408_40836


namespace second_number_proof_l408_40887

theorem second_number_proof : ∃! x : ℤ, 22030 = (555 + x) * (2 * (x - 555)) + 30 := by
  sorry

end second_number_proof_l408_40887


namespace interior_cubes_6_5_4_l408_40844

/-- Represents a rectangular prism -/
structure RectangularPrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of interior cubes in a rectangular prism -/
def interiorCubes (prism : RectangularPrism) : ℕ :=
  (prism.width - 2) * (prism.length - 2) * (prism.height - 2)

/-- Theorem: A 6x5x4 rectangular prism cut into 1x1x1 cubes has 24 interior cubes -/
theorem interior_cubes_6_5_4 :
  interiorCubes { width := 6, length := 5, height := 4 } = 24 := by
  sorry

#eval interiorCubes { width := 6, length := 5, height := 4 }

end interior_cubes_6_5_4_l408_40844


namespace new_average_weight_l408_40803

theorem new_average_weight (initial_count : ℕ) (initial_average : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_average = 28 →
  new_student_weight = 22 →
  (initial_count * initial_average + new_student_weight) / (initial_count + 1) = 27.8 := by
  sorry

end new_average_weight_l408_40803


namespace seeds_planted_equals_85_l408_40868

/-- Calculates the total number of seeds planted given the number of seeds per bed,
    flowers per bed, and total flowers grown. -/
def total_seeds_planted (seeds_per_bed : ℕ) (flowers_per_bed : ℕ) (total_flowers : ℕ) : ℕ :=
  let full_beds := total_flowers / flowers_per_bed
  let seeds_in_full_beds := full_beds * seeds_per_bed
  let flowers_in_partial_bed := total_flowers % flowers_per_bed
  seeds_in_full_beds + flowers_in_partial_bed

/-- Theorem stating that given the specific conditions, the total seeds planted is 85. -/
theorem seeds_planted_equals_85 :
  total_seeds_planted 15 60 220 = 85 := by
  sorry

end seeds_planted_equals_85_l408_40868


namespace diophantine_equation_solutions_l408_40835

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, y^2 = x^3 + 2*x^2 + 2*x + 1 ↔ (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
by sorry

end diophantine_equation_solutions_l408_40835


namespace pizza_combinations_l408_40832

theorem pizza_combinations (n : ℕ) (h : n = 7) : 
  n + (n.choose 2) + (n.choose 3) = 63 := by sorry

end pizza_combinations_l408_40832


namespace arc_measure_is_sixty_l408_40876

/-- An equilateral triangle with a circle rolling along its side -/
structure TriangleWithCircle where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Assumption that the side length is positive -/
  a_pos : 0 < a

/-- The angular measure of the arc intercepted on the circle -/
def arcMeasure (t : TriangleWithCircle) : ℝ := 60

/-- Theorem stating that the arc measure is always 60 degrees -/
theorem arc_measure_is_sixty (t : TriangleWithCircle) : arcMeasure t = 60 := by
  sorry

end arc_measure_is_sixty_l408_40876


namespace sample_xy_value_l408_40853

theorem sample_xy_value (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 →
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 = 5 →
  x * y = -4 := by
sorry

end sample_xy_value_l408_40853


namespace concentration_a_is_45_percent_l408_40819

/-- The concentration of spirit in vessel a -/
def concentration_a : ℝ := 45

/-- The concentration of spirit in vessel b -/
def concentration_b : ℝ := 30

/-- The concentration of spirit in vessel c -/
def concentration_c : ℝ := 10

/-- The volume taken from vessel a -/
def volume_a : ℝ := 4

/-- The volume taken from vessel b -/
def volume_b : ℝ := 5

/-- The volume taken from vessel c -/
def volume_c : ℝ := 6

/-- The concentration of spirit in the resultant solution -/
def concentration_result : ℝ := 26

/-- Theorem stating that the concentration of spirit in vessel a is 45% -/
theorem concentration_a_is_45_percent :
  (volume_a * concentration_a / 100 + 
   volume_b * concentration_b / 100 + 
   volume_c * concentration_c / 100) / 
  (volume_a + volume_b + volume_c) * 100 = concentration_result :=
by sorry

end concentration_a_is_45_percent_l408_40819
