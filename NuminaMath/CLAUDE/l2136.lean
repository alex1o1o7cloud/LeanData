import Mathlib

namespace sausage_division_ratio_l2136_213661

/-- Represents the length of the sausage after each bite -/
def remaining_length (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then 3/4 * remaining_length n else 2/3 * remaining_length n

/-- Theorem stating that the sausage should be divided in a 1:1 ratio -/
theorem sausage_division_ratio :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |remaining_length n - 1/2| < ε :=
sorry

end sausage_division_ratio_l2136_213661


namespace triangle_angle_sum_l2136_213674

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 40) (h2 : B = 80) : C = 60 := by
  sorry

end triangle_angle_sum_l2136_213674


namespace cone_radius_from_slant_height_and_surface_area_l2136_213667

/-- Given a cone with slant height 10 cm and curved surface area 157.07963267948966 cm²,
    the radius of the base is 5 cm. -/
theorem cone_radius_from_slant_height_and_surface_area :
  let slant_height : ℝ := 10
  let curved_surface_area : ℝ := 157.07963267948966
  let radius : ℝ := curved_surface_area / (Real.pi * slant_height)
  radius = 5 := by sorry

end cone_radius_from_slant_height_and_surface_area_l2136_213667


namespace median_is_55_l2136_213682

/-- A set of consecutive integers with a specific property --/
structure ConsecutiveIntegerSet where
  first : ℤ  -- The first integer in the set
  count : ℕ  -- The number of integers in the set
  sum_property : ∀ n : ℕ, n ≤ count → first + (n - 1) + (first + (count - n)) = 110

/-- The median of a set of consecutive integers --/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + (s.first + (s.count - 1))) / 2

/-- Theorem: The median of the ConsecutiveIntegerSet is always 55 --/
theorem median_is_55 (s : ConsecutiveIntegerSet) : median s = 55 := by
  sorry


end median_is_55_l2136_213682


namespace quadratic_equation_unique_solution_l2136_213622

theorem quadratic_equation_unique_solution :
  ∃! x : ℝ, x^2 + 2*x + 1 = 0 := by sorry

end quadratic_equation_unique_solution_l2136_213622


namespace opposite_signs_abs_sum_less_abs_diff_l2136_213615

theorem opposite_signs_abs_sum_less_abs_diff
  (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end opposite_signs_abs_sum_less_abs_diff_l2136_213615


namespace curve_self_intersection_l2136_213634

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 3, t^4 - t^2 - 9*t + 6)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (6, 6)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end curve_self_intersection_l2136_213634


namespace multiple_of_two_three_five_l2136_213619

theorem multiple_of_two_three_five : ∃ n : ℕ, 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n :=
  by
  use 30
  sorry

end multiple_of_two_three_five_l2136_213619


namespace six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2136_213678

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) : Nat :=
  by
  have h1 : total_six_digit = 900000 := by sorry
  have h2 : six_digit_no_zero = 531441 := by sorry
  have h3 : total_six_digit ≥ six_digit_no_zero := by sorry
  exact total_six_digit - six_digit_no_zero

theorem six_digit_numbers_with_zero_count :
    six_digit_numbers_with_zero 900000 531441 = 368559 :=
  by sorry

end six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2136_213678


namespace root_exists_in_interval_l2136_213657

/-- The function f(x) = x^2 + 12x - 15 -/
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem root_exists_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  sorry

end root_exists_in_interval_l2136_213657


namespace two_digit_sum_square_property_l2136_213639

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The set of numbers satisfying the condition -/
def valid_numbers : Set ℕ := {10, 20, 11, 30, 21, 12, 31, 22, 13}

/-- Main theorem -/
theorem two_digit_sum_square_property (A : ℕ) :
  is_two_digit A →
  (((sum_of_digits A) ^ 2 = sum_of_digits (A ^ 2)) ↔ A ∈ valid_numbers) := by
  sorry

end two_digit_sum_square_property_l2136_213639


namespace cell_division_genetic_info_l2136_213672

-- Define the types for cells and genetic information
variable (Cell GeneticInfo : Type)

-- Define the functions for getting genetic information from a cell
variable (genetic_info : Cell → GeneticInfo)

-- Define the cells
variable (C₁ C₂ S₁ S₂ : Cell)

-- Define the property of being daughter cells from mitosis
variable (mitosis_daughter_cells : Cell → Cell → Prop)

-- Define the property of being secondary spermatocytes from meiosis I
variable (meiosis_I_secondary_spermatocytes : Cell → Cell → Prop)

-- State the theorem
theorem cell_division_genetic_info :
  mitosis_daughter_cells C₁ C₂ →
  meiosis_I_secondary_spermatocytes S₁ S₂ →
  (genetic_info C₁ = genetic_info C₂) ∧
  (genetic_info S₁ ≠ genetic_info S₂) :=
by sorry

end cell_division_genetic_info_l2136_213672


namespace cube_cutting_l2136_213601

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end cube_cutting_l2136_213601


namespace minimum_pipes_needed_l2136_213627

/-- Represents the cutting methods for a 6m steel pipe -/
inductive CuttingMethod
  | method2 -- 4 pieces of 0.8m and 1 piece of 2.5m
  | method3 -- 1 piece of 0.8m and 2 pieces of 2.5m

/-- Represents the number of pieces obtained from each cutting method -/
def piecesObtained (m : CuttingMethod) : (ℕ × ℕ) :=
  match m with
  | CuttingMethod.method2 => (4, 1)
  | CuttingMethod.method3 => (1, 2)

theorem minimum_pipes_needed :
  ∃ (x y : ℕ),
    x * (piecesObtained CuttingMethod.method2).1 + y * (piecesObtained CuttingMethod.method3).1 = 100 ∧
    x * (piecesObtained CuttingMethod.method2).2 + y * (piecesObtained CuttingMethod.method3).2 = 32 ∧
    x + y = 28 ∧
    ∀ (a b : ℕ),
      a * (piecesObtained CuttingMethod.method2).1 + b * (piecesObtained CuttingMethod.method3).1 = 100 →
      a * (piecesObtained CuttingMethod.method2).2 + b * (piecesObtained CuttingMethod.method3).2 = 32 →
      a + b ≥ 28 := by
  sorry

end minimum_pipes_needed_l2136_213627


namespace approximation_place_l2136_213666

/-- A function that returns the number of decimal places in a given number -/
def decimal_places (x : ℚ) : ℕ := sorry

/-- A function that returns the name of the decimal place given its position -/
def place_name (n : ℕ) : String := sorry

theorem approximation_place (x : ℚ) (h : decimal_places x = 2) :
  place_name (decimal_places x) = "hundredths" := by sorry

end approximation_place_l2136_213666


namespace temperature_at_midnight_l2136_213632

/-- Given temperature changes throughout a day, calculate the temperature at midnight. -/
theorem temperature_at_midnight 
  (morning_temp : Int) 
  (noon_rise : Int) 
  (midnight_drop : Int) 
  (h1 : morning_temp = -2)
  (h2 : noon_rise = 13)
  (h3 : midnight_drop = 8) : 
  morning_temp + noon_rise - midnight_drop = 3 :=
by sorry

end temperature_at_midnight_l2136_213632


namespace roots_of_equation_l2136_213611

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * x * (x - 5)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {0, 2, 3, 5} := by
  sorry

end roots_of_equation_l2136_213611


namespace minimum_cost_for_25_apples_l2136_213670

/-- Represents a group of apples with its cost -/
structure AppleGroup where
  count : Nat
  cost : Nat

/-- Calculates the total number of apples from a list of apple groups -/
def totalApples (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.count) 0

/-- Calculates the total cost from a list of apple groups -/
def totalCost (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.cost) 0

/-- Represents the store's apple pricing policy -/
def applePricing : List AppleGroup := [
  { count := 4, cost := 15 },
  { count := 7, cost := 25 }
]

theorem minimum_cost_for_25_apples :
  ∃ (purchase : List AppleGroup),
    totalApples purchase = 25 ∧
    purchase.length ≥ 3 ∧
    (∀ group ∈ purchase, group ∈ applePricing) ∧
    totalCost purchase = 90 ∧
    (∀ (other : List AppleGroup),
      totalApples other = 25 →
      other.length ≥ 3 →
      (∀ group ∈ other, group ∈ applePricing) →
      totalCost purchase ≤ totalCost other) :=
by
  sorry

end minimum_cost_for_25_apples_l2136_213670


namespace blue_balls_in_jar_l2136_213679

theorem blue_balls_in_jar (total : ℕ) (blue : ℕ) (prob : ℚ) : 
  total = 12 →
  blue ≤ total →
  prob = 1 / 55 →
  (blue.choose 3 : ℚ) / (total.choose 3 : ℚ) = prob →
  blue = 4 := by
  sorry

end blue_balls_in_jar_l2136_213679


namespace min_students_class_5_7_l2136_213614

theorem min_students_class_5_7 (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 3) ∧ 
  (∃ m : ℕ, n = 8 * m + 3) → 
  n ≥ 59 :=
sorry

end min_students_class_5_7_l2136_213614


namespace anthony_tax_deduction_l2136_213640

/-- Calculates the total tax deduction in cents given an hourly wage and tax rates -/
def totalTaxDeduction (hourlyWage : ℚ) (federalTaxRate : ℚ) (stateTaxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (federalTaxRate + stateTaxRate)

/-- Theorem: Given Anthony's wage and tax rates, the total tax deduction is 62.5 cents -/
theorem anthony_tax_deduction :
  totalTaxDeduction 25 (2/100) (1/200) = 125/2 := by
  sorry

end anthony_tax_deduction_l2136_213640


namespace quadratic_function_property_l2136_213635

/-- 
Given a quadratic function f(x) = ax^2 + bx + 5 with a ≠ 0,
if there exist two distinct points (x₁, 2023) and (x₂, 2023) on the graph of f,
then f(x₁ + x₂) = 5
-/
theorem quadratic_function_property (a b x₁ x₂ : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + 5
  (f x₁ = 2023) → (f x₂ = 2023) → (x₁ ≠ x₂) → f (x₁ + x₂) = 5 := by
  sorry

end quadratic_function_property_l2136_213635


namespace problem_statement_l2136_213623

theorem problem_statement (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (4/x + 1/y ≥ 4/m + 1/n)) ∧
  (4/m + 1/n ≥ 9) ∧
  (Real.sqrt m + Real.sqrt n ≤ Real.sqrt 2) ∧
  (m > n → 1/(m-1) < 1/(n-1)) := by
sorry

end problem_statement_l2136_213623


namespace base6_addition_l2136_213643

/-- Convert a number from base 6 to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Definition of the first number in base 6 --/
def num1 : List Nat := [2, 3, 5, 4]

/-- Definition of the second number in base 6 --/
def num2 : List Nat := [3, 5, 2, 4, 2]

/-- Theorem stating that the sum of the two numbers in base 6 equals the result --/
theorem base6_addition :
  base10ToBase6 (base6ToBase10 num1 + base6ToBase10 num2) = [5, 2, 2, 2, 3, 3] := by sorry

end base6_addition_l2136_213643


namespace min_product_positive_numbers_l2136_213689

theorem min_product_positive_numbers (x₁ x₂ x₃ : ℝ) :
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 →
  x₁ + x₂ + x₃ = 4 →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₁^2 + 2 * x₂^2 - 5 * x₁ * x₂ ≤ 0) →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₁^2 + 2 * x₃^2 - 5 * x₁ * x₃ ≤ 0) →
  (∀ (i j : Fin 3), i ≠ j → 2 * x₂^2 + 2 * x₃^2 - 5 * x₂ * x₃ ≤ 0) →
  x₁ * x₂ * x₃ ≥ 2 ∧ ∃ (y₁ y₂ y₃ : ℝ), y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 ∧
    y₁ + y₂ + y₃ = 4 ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₁^2 + 2 * y₂^2 - 5 * y₁ * y₂ ≤ 0) ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₁^2 + 2 * y₃^2 - 5 * y₁ * y₃ ≤ 0) ∧
    (∀ (i j : Fin 3), i ≠ j → 2 * y₂^2 + 2 * y₃^2 - 5 * y₂ * y₃ ≤ 0) ∧
    y₁ * y₂ * y₃ = 2 :=
by sorry

end min_product_positive_numbers_l2136_213689


namespace min_value_expression_l2136_213686

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) : 
  (6 * z) / (x + 2 * y + k) + (6 * x) / (2 * z + y + k) + (3 * y) / (x + z + k) ≥ 4.5 := by
  sorry

end min_value_expression_l2136_213686


namespace cubic_difference_l2136_213604

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 47) : 
  a^3 - b^3 = 322 := by
  sorry

end cubic_difference_l2136_213604


namespace sum_of_2001_and_1015_l2136_213606

theorem sum_of_2001_and_1015 : 2001 + 1015 = 3016 := by
  sorry

end sum_of_2001_and_1015_l2136_213606


namespace rectangle_area_diagonal_l2136_213691

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d,
    prove that its area A can be expressed as A = (10/29) * d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l^2 + w^2 = d^2 ∧ l * w = (10/29) * d^2 := by
  sorry

end rectangle_area_diagonal_l2136_213691


namespace sphere_radius_from_cone_volume_l2136_213658

/-- Given a cone with radius 2 inches and height 8 inches, 
    prove that a sphere with twice the volume of this cone 
    has a radius of 2^(4/3) inches. -/
theorem sphere_radius_from_cone_volume 
  (cone_radius : ℝ) 
  (cone_height : ℝ) 
  (sphere_radius : ℝ) :
  cone_radius = 2 ∧ 
  cone_height = 8 ∧ 
  (4/3) * π * sphere_radius^3 = 2 * ((1/3) * π * cone_radius^2 * cone_height) →
  sphere_radius = 2^(4/3) :=
by
  sorry

#check sphere_radius_from_cone_volume

end sphere_radius_from_cone_volume_l2136_213658


namespace sqrt_product_plus_one_l2136_213626

theorem sqrt_product_plus_one : Real.sqrt ((43 : ℝ) * 42 * 41 * 40 + 1) = 1721 := by
  sorry

end sqrt_product_plus_one_l2136_213626


namespace vector_magnitude_proof_l2136_213617

/-- Given a plane vector a = (2,0), |b| = 2, and a ⋅ b = 2, prove |a - 2b| = 2√3 -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 0)
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = 2) →
  norm (a - 2 • b) = 2 * Real.sqrt 3 :=
by sorry

end vector_magnitude_proof_l2136_213617


namespace remaining_money_is_29_l2136_213688

/-- Calculates the remaining money after spending on a novel and lunch -/
def remaining_money (initial_amount novel_cost : ℕ) : ℕ :=
  initial_amount - (novel_cost + 2 * novel_cost)

/-- Theorem: Given $50 initial amount and $7 novel cost, the remaining money is $29 -/
theorem remaining_money_is_29 :
  remaining_money 50 7 = 29 := by
  sorry

end remaining_money_is_29_l2136_213688


namespace f_range_l2136_213629

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = π / 4 := by
  sorry

end f_range_l2136_213629


namespace intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2136_213607

-- Define the sets A, B, and E
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Theorem for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
sorry

-- Theorem for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 :=
sorry

end intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2136_213607


namespace arithmetic_mean_of_fractions_l2136_213664

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (((2 * x + b) / x + (2 * x - b) / x) / 2) = 2 := by
sorry

end arithmetic_mean_of_fractions_l2136_213664


namespace equation_roots_opposite_signs_l2136_213618

theorem equation_roots_opposite_signs (a b c d m : ℝ) (hd : d ≠ 0) :
  (∀ x, (x^2 - (b+1)*x) / ((a-1)*x - (c+d)) = (m-2) / (m+2)) →
  (∃ r : ℝ, r ≠ 0 ∧ (r^2 - (b+1)*r = 0) ∧ (-r^2 - (b+1)*(-r) = 0)) →
  m = 2*(a-b-2) / (a+b) :=
by sorry

end equation_roots_opposite_signs_l2136_213618


namespace oliver_tickets_l2136_213687

def carnival_tickets (ferris_wheel_rides : ℕ) (bumper_car_rides : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride

theorem oliver_tickets : carnival_tickets 5 4 7 = 63 := by
  sorry

end oliver_tickets_l2136_213687


namespace book_sale_loss_percentage_l2136_213648

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

/-- Proves that the percentage loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage
  (sellingPrice : ℚ)
  (gainPrice : ℚ)
  (gainPercentage : ℚ)
  (h1 : sellingPrice = 540)
  (h2 : gainPrice = 660)
  (h3 : gainPercentage = 10)
  (h4 : gainPrice = (100 + gainPercentage) / 100 * (gainPrice / (1 + gainPercentage / 100))) :
  percentageLoss (gainPrice / (1 + gainPercentage / 100)) sellingPrice = 10 := by
  sorry

#eval percentageLoss (660 / (1 + 10 / 100)) 540

end book_sale_loss_percentage_l2136_213648


namespace inverse_as_linear_combination_l2136_213671

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 4, -2]

theorem inverse_as_linear_combination :
  N⁻¹ = (1 / 10 : ℚ) • N + (-1 / 10 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end inverse_as_linear_combination_l2136_213671


namespace circles_tangent_to_ellipse_l2136_213685

theorem circles_tangent_to_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x-r)^2 + y^2 = r^2) ∧ 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x+r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end circles_tangent_to_ellipse_l2136_213685


namespace signal_count_is_324_l2136_213663

/-- Represents the number of indicator lights in a row -/
def total_lights : Nat := 6

/-- Represents the number of lights displayed at a time -/
def displayed_lights : Nat := 3

/-- Represents the number of possible colors for each light -/
def color_options : Nat := 3

/-- Calculates the number of different signals that can be displayed -/
def signal_count : Nat :=
  let adjacent_pair_positions := total_lights - 1
  let non_adjacent_positions := total_lights - 2
  (adjacent_pair_positions * non_adjacent_positions) * color_options^displayed_lights

/-- Theorem stating that the number of different signals is 324 -/
theorem signal_count_is_324 : signal_count = 324 := by
  sorry

end signal_count_is_324_l2136_213663


namespace unfair_coin_prob_theorem_l2136_213655

/-- An unfair coin with probabilities of heads and tails -/
structure UnfairCoin where
  pH : ℝ  -- Probability of heads
  pT : ℝ  -- Probability of tails
  sum_one : pH + pT = 1
  unfair : pH ≠ pT

/-- The probability of getting one head and one tail in two tosses -/
def prob_one_head_one_tail (c : UnfairCoin) : ℝ :=
  2 * c.pH * c.pT

/-- The probability of getting two heads and two tails in four tosses -/
def prob_two_heads_two_tails (c : UnfairCoin) : ℝ :=
  6 * c.pH * c.pH * c.pT * c.pT

/-- Theorem: For an unfair coin where the probability of getting one head and one tail
    in two tosses is 1/2, the probability of getting two heads and two tails in four tosses is 3/8 -/
theorem unfair_coin_prob_theorem (c : UnfairCoin) 
    (h : prob_one_head_one_tail c = 1/2) : 
    prob_two_heads_two_tails c = 3/8 := by
  sorry

end unfair_coin_prob_theorem_l2136_213655


namespace circle_intersection_theorem_l2136_213637

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the condition for the equation to represent a circle
def is_circle (m : ℝ) : Prop :=
  m < 5/4

-- Define the intersection condition
def intersects_at_mn (m : ℝ) : Prop :=
  ∃ (M N : ℝ × ℝ),
    circle_equation M.1 M.2 m ∧
    circle_equation N.1 N.2 m ∧
    line_equation M.1 M.2 ∧
    line_equation N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4/5 * Real.sqrt 5)^2

theorem circle_intersection_theorem :
  ∀ m : ℝ, is_circle m → intersects_at_mn m → m = 3.62 :=
by sorry

end circle_intersection_theorem_l2136_213637


namespace eight_power_fifteen_divided_by_sixtyfour_power_seven_l2136_213603

theorem eight_power_fifteen_divided_by_sixtyfour_power_seven :
  8^15 / 64^7 = 8 := by sorry

end eight_power_fifteen_divided_by_sixtyfour_power_seven_l2136_213603


namespace original_number_proof_l2136_213638

theorem original_number_proof (x : ℝ) : 
  268 * x = 19832 ∧ 2.68 * x = 1.9832 → x = 74 := by
  sorry

end original_number_proof_l2136_213638


namespace sample_size_correct_l2136_213630

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem stating that the sample size is correct -/
theorem sample_size_correct (pop : Population) (samp : Sample) : 
  pop.size = 8000 → samp.size = 400 → samp.size = 400 := by sorry

end sample_size_correct_l2136_213630


namespace smallest_sum_of_sequences_l2136_213668

theorem smallest_sum_of_sequences (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C = (7 * B) / 4) →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (7 * B') / 4) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end smallest_sum_of_sequences_l2136_213668


namespace equation_solutions_l2136_213652

noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

def is_solution (x : ℝ) : Prop :=
  x ≠ 0.5 ∧ (floor x : ℝ) - Real.sqrt ((floor x : ℝ) / (x - 0.5)) - 6 / (x - 0.5) = 0

theorem equation_solutions :
  ∀ x : ℝ, is_solution x ↔ (x = -1.5 ∨ x = 3.5) :=
by sorry

end equation_solutions_l2136_213652


namespace parallel_vectors_t_value_l2136_213647

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.2 = a.2 * b.1)

/-- Given vectors a and b, prove that if they are parallel, then t = 9 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (3, t)
  are_parallel a b → t = 9 := by
  sorry

end parallel_vectors_t_value_l2136_213647


namespace right_triangle_sine_inequality_l2136_213654

/-- Given a right-angled triangle with hypotenuse parallel to plane α, and angles θ₁ and θ₂
    between the lines containing the two legs of the triangle and plane α,
    prove that sin²θ₁ + sin²θ₂ ≤ 1 -/
theorem right_triangle_sine_inequality (θ₁ θ₂ : Real) 
    (h₁ : 0 ≤ θ₁ ∧ θ₁ ≤ π / 2) 
    (h₂ : 0 ≤ θ₂ ∧ θ₂ ≤ π / 2) 
    (h_right_angle : θ₁ + θ₂ ≤ π / 2) : 
    Real.sin θ₁ ^ 2 + Real.sin θ₂ ^ 2 ≤ 1 := by
  sorry

end right_triangle_sine_inequality_l2136_213654


namespace f_expression_for_x_gt_1_l2136_213681

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-x + 1)

theorem f_expression_for_x_gt_1 (f : ℝ → ℝ) 
  (h1 : is_even_shifted f) 
  (h2 : ∀ x, x < 1 → f x = x^2 + 1) :
  ∀ x, x > 1 → f x = x^2 - 4*x + 5 := by
  sorry

end f_expression_for_x_gt_1_l2136_213681


namespace total_bottle_caps_l2136_213676

theorem total_bottle_caps (bottle_caps_per_child : ℕ) (number_of_children : ℕ) 
  (h1 : bottle_caps_per_child = 5) 
  (h2 : number_of_children = 9) : 
  bottle_caps_per_child * number_of_children = 45 := by
  sorry

end total_bottle_caps_l2136_213676


namespace value_of_a_l2136_213696

theorem value_of_a : 
  let a := Real.sqrt ((19.19^2) + (39.19^2) - (38.38 * 39.19))
  a = 20 := by sorry

end value_of_a_l2136_213696


namespace line_equation_through_points_l2136_213694

/-- The equation of a line passing through two points. -/
def line_equation (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  λ x y => y - y1 = m * (x - x1)

/-- Theorem: The equation of the line passing through P(-2, 5) and Q(4, 1/2) is 3x + 4y - 14 = 0. -/
theorem line_equation_through_points :
  let p1 : ℝ × ℝ := (-2, 5)
  let p2 : ℝ × ℝ := (4, 1/2)
  ∀ x y : ℝ, line_equation p1 p2 x y ↔ 3 * x + 4 * y - 14 = 0 := by
  sorry

end line_equation_through_points_l2136_213694


namespace candy_cost_l2136_213641

/-- 
Given that each piece of bulk candy costs 8 cents and 28 gumdrops can be bought,
prove that the total amount of cents is 224.
-/
theorem candy_cost (cost_per_piece : ℕ) (num_gumdrops : ℕ) (h1 : cost_per_piece = 8) (h2 : num_gumdrops = 28) :
  cost_per_piece * num_gumdrops = 224 := by
  sorry

end candy_cost_l2136_213641


namespace tangent_length_right_triangle_l2136_213660

/-- Given a right triangle with legs a and b, and hypotenuse c,
    the length of the tangent to the circumcircle drawn parallel
    to the hypotenuse is equal to c(a + b)²/(2ab) -/
theorem tangent_length_right_triangle (a b c : ℝ) 
  (h_right : c^2 = a^2 + b^2) (h_pos : a > 0 ∧ b > 0) :
  let x := c * (a + b)^2 / (2 * a * b)
  ∃ (m : ℝ), m > 0 ∧ 
    (c / x = m / (m + c/2)) ∧
    (m * c = a * b) :=
by sorry

end tangent_length_right_triangle_l2136_213660


namespace complex_number_parts_opposite_l2136_213665

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (3 + Complex.I)
  (z.re = -z.im) → b = 1 := by
  sorry

end complex_number_parts_opposite_l2136_213665


namespace fraction_equality_l2136_213677

theorem fraction_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end fraction_equality_l2136_213677


namespace f_2015_value_l2136_213610

def f (x : ℝ) : ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def piecewise_def (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x, -2 ≤ x ∧ x < 0 → f x = a * x + b) ∧
  (∀ x, 0 < x ∧ x ≤ 2 → f x = a * x - 1)

theorem f_2015_value (a b : ℝ) :
  is_odd f ∧ has_period f 4 ∧ piecewise_def f a b →
  f 2015 = 3/2 := by sorry

end f_2015_value_l2136_213610


namespace custom_equation_solution_l2136_213628

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℝ, star 2 (star 6 x) = 2 :=
by
  -- The proof goes here
  sorry

end custom_equation_solution_l2136_213628


namespace divisibility_condition_l2136_213602

theorem divisibility_condition (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end divisibility_condition_l2136_213602


namespace wage_calculation_l2136_213645

/-- A worker's wage calculation problem -/
theorem wage_calculation 
  (total_days : ℕ) 
  (absent_days : ℕ) 
  (fine_per_day : ℕ) 
  (total_pay : ℕ) 
  (h1 : total_days = 30)
  (h2 : absent_days = 7)
  (h3 : fine_per_day = 2)
  (h4 : total_pay = 216) :
  ∃ (daily_wage : ℕ),
    (total_days - absent_days) * daily_wage - absent_days * fine_per_day = total_pay ∧
    daily_wage = 10 :=
by
  sorry

end wage_calculation_l2136_213645


namespace infinite_triples_exist_l2136_213692

theorem infinite_triples_exist : 
  ∀ y : ℝ, ∃ x z : ℝ, 
    (x^2 + y = y^2 + z) ∧ 
    (y^2 + z = z^2 + x) ∧ 
    (z^2 + x = x^2 + y) ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x :=
by sorry

end infinite_triples_exist_l2136_213692


namespace isosceles_triangle_area_l2136_213651

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  altitude : ℝ
  perimeter : ℝ
  base_to_side_ratio : ℚ
  is_isosceles : base ≠ side
  altitude_value : altitude = 10
  perimeter_value : perimeter = 40
  ratio_value : base_to_side_ratio = 2 / 3

/-- The area of an isosceles triangle with the given properties is 80 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : t.base * t.altitude / 2 = 80 := by
  sorry

end isosceles_triangle_area_l2136_213651


namespace pyramid_base_edge_length_l2136_213684

/-- The configuration of five identical balls and a circumscribing square pyramid. -/
structure BallPyramidConfig where
  -- Radius of each ball
  ball_radius : ℝ
  -- Distance between centers of adjacent bottom balls
  bottom_center_distance : ℝ
  -- Height from floor to center of top ball
  top_ball_height : ℝ
  -- Edge length of the square base of the pyramid
  pyramid_base_edge : ℝ

/-- The theorem stating the edge length of the square base of the pyramid. -/
theorem pyramid_base_edge_length 
  (config : BallPyramidConfig) 
  (h1 : config.ball_radius = 2)
  (h2 : config.bottom_center_distance = 2 * config.ball_radius)
  (h3 : config.top_ball_height = 3 * config.ball_radius)
  (h4 : config.pyramid_base_edge = config.bottom_center_distance * Real.sqrt 2) :
  config.pyramid_base_edge = 4 * Real.sqrt 2 :=
by sorry

end pyramid_base_edge_length_l2136_213684


namespace starting_number_proof_l2136_213646

theorem starting_number_proof (x : ℕ) : 
  (x ≤ 26) → 
  (x % 2 = 0) → 
  ((x + 26) / 2 = 19) → 
  x = 12 := by
sorry

end starting_number_proof_l2136_213646


namespace perpendicular_bisector_m_value_l2136_213605

/-- Given points A and B, if the equation of the perpendicular bisector of segment AB is x + 2y - 2 = 0, then m = 3 -/
theorem perpendicular_bisector_m_value (m : ℝ) : 
  let A : ℝ × ℝ := (1, -2)
  let B : ℝ × ℝ := (m, 2)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + 2 * midpoint.2 - 2 = 0) → m = 3 := by
sorry


end perpendicular_bisector_m_value_l2136_213605


namespace circle_ratio_l2136_213613

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 6 := by
  sorry

end circle_ratio_l2136_213613


namespace carly_to_lisa_tshirt_ratio_l2136_213669

def lisa_tshirts : ℚ := 40
def lisa_jeans : ℚ := lisa_tshirts / 2
def lisa_coats : ℚ := lisa_tshirts * 2

def carly_jeans : ℚ := lisa_jeans * 3
def carly_coats : ℚ := lisa_coats / 4

def total_spending : ℚ := 230

theorem carly_to_lisa_tshirt_ratio :
  ∃ (carly_tshirts : ℚ),
    lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats = total_spending ∧
    carly_tshirts / lisa_tshirts = 1 / 4 :=
by sorry

end carly_to_lisa_tshirt_ratio_l2136_213669


namespace rooster_stamps_count_l2136_213636

theorem rooster_stamps_count (daffodil_stamps : ℕ) (rooster_stamps : ℕ) 
  (h1 : daffodil_stamps = 2) 
  (h2 : rooster_stamps - daffodil_stamps = 0) : 
  rooster_stamps = 2 := by
  sorry

end rooster_stamps_count_l2136_213636


namespace geometric_series_first_term_l2136_213620

/-- For an infinite geometric series with common ratio 1/4 and sum 40, the first term is 30 -/
theorem geometric_series_first_term (a : ℝ) : 
  (∀ n : ℕ, ∑' k, a * (1/4)^k = 40) → a = 30 := by
  sorry

end geometric_series_first_term_l2136_213620


namespace quadratic_factorization_l2136_213690

theorem quadratic_factorization (a b : ℕ) (h1 : a ≥ b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  4*b - a = 27 := by sorry

end quadratic_factorization_l2136_213690


namespace old_soldiers_participation_l2136_213653

/-- Represents the distribution of soldiers in different age groups for a parade. -/
structure ParadeDistribution where
  total_soldiers : ℕ
  young_soldiers : ℕ
  middle_soldiers : ℕ
  old_soldiers : ℕ
  parade_spots : ℕ
  young_soldiers_le : young_soldiers ≤ total_soldiers
  middle_soldiers_le : middle_soldiers ≤ total_soldiers
  old_soldiers_le : old_soldiers ≤ total_soldiers
  total_sum : young_soldiers + middle_soldiers + old_soldiers = total_soldiers

/-- The number of soldiers aged over 23 participating in the parade. -/
def old_soldiers_in_parade (d : ParadeDistribution) : ℕ :=
  min d.old_soldiers (d.parade_spots - (d.parade_spots / 3 * 2))

/-- Theorem stating that for the given distribution, 2 soldiers aged over 23 will participate. -/
theorem old_soldiers_participation (d : ParadeDistribution) 
  (h1 : d.total_soldiers = 45)
  (h2 : d.young_soldiers = 15)
  (h3 : d.middle_soldiers = 20)
  (h4 : d.old_soldiers = 10)
  (h5 : d.parade_spots = 9) :
  old_soldiers_in_parade d = 2 := by
  sorry


end old_soldiers_participation_l2136_213653


namespace fourth_grade_students_l2136_213642

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 31 → left = 5 → new = 11 → final = initial - left + new → final = 37 := by
  sorry

end fourth_grade_students_l2136_213642


namespace subtraction_result_l2136_213612

/-- The value obtained when 20² is subtracted from the square of 68.70953354520753 is approximately 4321 -/
theorem subtraction_result : 
  let x : ℝ := 68.70953354520753
  ∃ ε > 0, abs ((x^2 - 20^2) - 4321) < ε :=
by sorry

end subtraction_result_l2136_213612


namespace tangent_speed_l2136_213650

/-- Given the equation (a * T) / (a * T - R) = (L + x) / x, where x represents a distance,
    prove that the speed of a point determined by x is equal to a * L / R. -/
theorem tangent_speed (a R L T : ℝ) (x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R := by
  sorry

end tangent_speed_l2136_213650


namespace quadratic_equation_roots_l2136_213675

/-- Given constants a, b, c, where P(a,c) is in the fourth quadrant,
    prove that ax^2 + bx + c = 0 has two distinct real roots -/
theorem quadratic_equation_roots (a b c : ℝ) 
  (h1 : a > 0) (h2 : c < 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end quadratic_equation_roots_l2136_213675


namespace min_value_of_2x_plus_y_l2136_213608

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≥ 8 :=
sorry

end min_value_of_2x_plus_y_l2136_213608


namespace triangle_sin_C_l2136_213649

theorem triangle_sin_C (a c : ℝ) (A : ℝ) :
  a = 7 →
  c = 3 →
  A = π / 3 →
  Real.sin (Real.arcsin ((c * Real.sin A) / a)) = 3 * Real.sqrt 3 / 14 := by
  sorry

end triangle_sin_C_l2136_213649


namespace consecutive_numbers_sequence_l2136_213616

def is_valid_sequence (a b : ℕ) : Prop :=
  let n := b - a + 1
  let sum := n * (a + b) / 2
  let mean := sum / n
  let sum_without_122_123 := sum - 122 - 123
  let mean_without_122_123 := sum_without_122_123 / (n - 2)
  (mean = 85) ∧
  (mean = (70 + 82 + 103) / 3) ∧
  (mean_without_122_123 + 1 = mean) ∧
  (a = 47) ∧
  (b = 123)

theorem consecutive_numbers_sequence :
  ∃ a b : ℕ, is_valid_sequence a b :=
sorry

end consecutive_numbers_sequence_l2136_213616


namespace journey_speed_theorem_l2136_213631

/-- Given a journey with the following parameters:
  * total_distance: The total distance traveled in miles
  * total_time: The total time of the journey in minutes
  * speed_first_30: The average speed during the first 30 minutes in mph
  * speed_second_30: The average speed during the second 30 minutes in mph

  This function calculates the average speed during the last 60 minutes of the journey. -/
def average_speed_last_60 (total_distance : ℝ) (total_time : ℝ) (speed_first_30 : ℝ) (speed_second_30 : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given journey parameters, 
    the average speed during the last 60 minutes is 77.5 mph -/
theorem journey_speed_theorem :
  average_speed_last_60 150 120 75 70 = 77.5 := by
  sorry

end journey_speed_theorem_l2136_213631


namespace exists_larger_area_same_perimeter_l2136_213600

-- Define a convex figure
structure ConvexFigure where
  perimeter : ℝ
  area : ℝ

-- Define a property for a figure to be a circle
def isCircle (f : ConvexFigure) : Prop := sorry

-- Theorem statement
theorem exists_larger_area_same_perimeter 
  (Φ : ConvexFigure) 
  (h_not_circle : ¬ isCircle Φ) : 
  ∃ (Ψ : ConvexFigure), 
    Ψ.perimeter = Φ.perimeter ∧ 
    Ψ.area > Φ.area := by
  sorry

end exists_larger_area_same_perimeter_l2136_213600


namespace polynomial_inequality_l2136_213656

theorem polynomial_inequality (x : ℝ) : (x + 2) * (x - 8) * (x - 3) > 0 ↔ x ∈ Set.Ioo (-2 : ℝ) 3 ∪ Set.Ioi 8 := by
  sorry

end polynomial_inequality_l2136_213656


namespace gcf_lcm_product_l2136_213659

def numbers : List Nat := [6, 18, 24]

theorem gcf_lcm_product (A B : Nat) 
  (h1 : A = Nat.gcd 6 (Nat.gcd 18 24))
  (h2 : B = Nat.lcm 6 (Nat.lcm 18 24)) :
  A * B = 432 := by
  sorry

end gcf_lcm_product_l2136_213659


namespace may_fourth_is_sunday_l2136_213644

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  fridayCount : Nat
  fridayDatesSum : Nat

/-- Returns the day of the week for a given date in the month -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem may_fourth_is_sunday (m : Month) 
  (h1 : m.fridayCount = 5) 
  (h2 : m.fridayDatesSum = 80) : 
  dayOfWeek m 4 = DayOfWeek.Sunday := by sorry

end may_fourth_is_sunday_l2136_213644


namespace cylinder_volume_unit_dimensions_l2136_213621

/-- The volume of a cylinder with base radius 1 and height 1 is π. -/
theorem cylinder_volume_unit_dimensions : 
  let r : ℝ := 1
  let h : ℝ := 1
  let V := π * r^2 * h
  V = π := by sorry

end cylinder_volume_unit_dimensions_l2136_213621


namespace probability_one_black_one_white_l2136_213624

def total_balls : ℕ := 5
def black_balls : ℕ := 3
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_one_white :
  (black_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose drawn_balls = 3 / 5 := by
  sorry

end probability_one_black_one_white_l2136_213624


namespace largest_angle_is_80_l2136_213683

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.angle1 + t.angle2 = (4/3) * right_angle ∧
  t.angle2 = t.angle1 + 40 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem largest_angle_is_80 (t : Triangle) :
  triangle_conditions t → (max t.angle1 (max t.angle2 t.angle3) = 80) :=
by sorry

end largest_angle_is_80_l2136_213683


namespace game_outcome_for_similar_numbers_l2136_213697

/-- The game outcome for a given number -/
inductive Outcome
| Good
| Bad

/-- Definition of the game -/
def game (k : ℕ) (n : ℕ) : Outcome :=
  sorry

/-- Two numbers are similar if they are divisible by the same primes up to k -/
def similar (k : ℕ) (n n' : ℕ) : Prop :=
  ∀ p, p.Prime → p ≤ k → (p ∣ n ↔ p ∣ n')

theorem game_outcome_for_similar_numbers (k : ℕ) (n n' : ℕ) (h_k : k ≥ 2) (h_n : n ≥ k) (h_n' : n' ≥ k) (h_similar : similar k n n') :
  game k n = game k n' :=
sorry

end game_outcome_for_similar_numbers_l2136_213697


namespace coefficient_d_nonzero_l2136_213625

-- Define the polynomial Q(x)
def Q (a b c d e : ℂ) (x : ℂ) : ℂ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- State the theorem
theorem coefficient_d_nonzero 
  (a b c d e : ℂ) 
  (h1 : ∃ u v : ℂ, ∀ x : ℂ, Q a b c d e x = x * (x - (2 + 3*I)) * (x - (2 - 3*I)) * (x - u) * (x - v))
  (h2 : Q a b c d e 0 = 0)
  (h3 : Q a b c d e (2 + 3*I) = 0)
  (h4 : ∀ x : ℂ, Q a b c d e x = 0 → x = 0 ∨ x = 2 + 3*I ∨ x = 2 - 3*I ∨ (∃ y : ℂ, y ≠ 0 ∧ y ≠ 2 + 3*I ∧ y ≠ 2 - 3*I ∧ x = y)) :
  d ≠ 0 := by
  sorry


end coefficient_d_nonzero_l2136_213625


namespace wire_length_from_sphere_l2136_213693

/-- The length of a wire drawn from a metallic sphere -/
theorem wire_length_from_sphere (r_sphere r_wire : ℝ) (h : r_sphere = 24 ∧ r_wire = 0.16) :
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let l_wire := v_sphere / (Real.pi * r_wire ^ 2)
  l_wire = 675000 := by
  sorry

end wire_length_from_sphere_l2136_213693


namespace final_apartments_can_be_less_l2136_213662

/-- Represents the structure of an apartment building project -/
structure ApartmentProject where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a project -/
def total_apartments (p : ApartmentProject) : ℕ :=
  p.entrances * p.floors * p.apartments_per_floor

/-- Applies the architect's adjustments to a project -/
def adjust_project (p : ApartmentProject) (removed_entrances floors_added : ℕ) : ApartmentProject :=
  { entrances := p.entrances - removed_entrances,
    floors := p.floors + floors_added,
    apartments_per_floor := p.apartments_per_floor }

/-- The main theorem stating that the final number of apartments can be less than the initial number -/
theorem final_apartments_can_be_less :
  ∃ (initial : ApartmentProject)
    (removed_entrances1 floors_added1 removed_entrances2 floors_added2 : ℕ),
    initial.entrances = 5 ∧
    initial.floors = 2 ∧
    initial.apartments_per_floor = 1 ∧
    removed_entrances1 = 2 ∧
    floors_added1 = 3 ∧
    removed_entrances2 = 2 ∧
    floors_added2 = 3 ∧
    let first_adjustment := adjust_project initial removed_entrances1 floors_added1
    let final_project := adjust_project first_adjustment removed_entrances2 floors_added2
    total_apartments final_project < total_apartments initial :=
by
  sorry

end final_apartments_can_be_less_l2136_213662


namespace amy_avocado_business_l2136_213698

/-- Proves that given Amy's avocado business conditions, n = 50 --/
theorem amy_avocado_business (n : ℕ+) : 
  (15 * n : ℕ) = (15 * n : ℕ) ∧  -- Amy bought and sold 15n avocados
  (12 * n - 10 * n : ℕ) = 100 ∧  -- She made a profit of $100
  (2 : ℕ) = (2 : ℕ) ∧            -- She paid $2 for every 3 avocados
  (4 : ℕ) = (4 : ℕ)              -- She sold every 5 avocados for $4
  → n = 50 := by
sorry

end amy_avocado_business_l2136_213698


namespace negative_quartic_count_l2136_213695

theorem negative_quartic_count : ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 62*x^2 + 60 < 0) ∧ S.card = 12 ∧ 
  ∀ x : ℤ, x^4 - 62*x^2 + 60 < 0 → x ∈ S :=
sorry

end negative_quartic_count_l2136_213695


namespace ellipse_right_triangle_x_coordinate_l2136_213673

/-- The x-coordinate of a point on an ellipse forming a right triangle with the foci -/
theorem ellipse_right_triangle_x_coordinate 
  (x y : ℝ) 
  (h_ellipse : x^2/16 + y^2/25 = 1) 
  (h_on_ellipse : ∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y)
  (h_foci : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0)
  (h_right_triangle : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0 ∧ 
    (F₁.2 - y)^2 + x^2 + (F₂.2 - y)^2 + x^2 = (F₂.2 - F₁.2)^2) :
  x = 16/5 := by
sorry

end ellipse_right_triangle_x_coordinate_l2136_213673


namespace fish_size_difference_l2136_213680

/-- The size difference between Seongjun's and Sungwoo's fish given the conditions -/
theorem fish_size_difference (S J W : ℝ) 
  (h1 : S = J + 21.52)
  (h2 : J = W - 12.64) :
  S - W = 8.88 := by
  sorry

end fish_size_difference_l2136_213680


namespace smallest_multiple_l2136_213633

theorem smallest_multiple (n : ℕ) : n = 255 ↔ 
  (∃ k : ℕ, n = 15 * k) ∧ 
  (∃ m : ℕ, n = 65 * m + 7) ∧ 
  (∃ p : ℕ, n = 5 * p) ∧ 
  (∀ x : ℕ, x < n → 
    (¬(∃ k : ℕ, x = 15 * k) ∨ 
     ¬(∃ m : ℕ, x = 65 * m + 7) ∨ 
     ¬(∃ p : ℕ, x = 5 * p))) :=
by sorry

end smallest_multiple_l2136_213633


namespace senior_policy_more_profitable_l2136_213609

/-- Represents a customer group with their characteristics -/
structure CustomerGroup where
  repaymentReliability : ℝ
  incomeStability : ℝ
  savingInclination : ℝ
  longTermPreference : ℝ

/-- Represents the bank's policy for a customer group -/
structure BankPolicy where
  depositRate : ℝ
  loanRate : ℝ

/-- Calculates the bank's profit from a customer group under a given policy -/
def bankProfit (group : CustomerGroup) (policy : BankPolicy) : ℝ :=
  group.repaymentReliability * policy.loanRate +
  group.savingInclination * group.longTermPreference * policy.depositRate

/-- Theorem: Under certain conditions, a bank can achieve higher profit 
    by offering better rates to seniors -/
theorem senior_policy_more_profitable 
  (seniors : CustomerGroup) 
  (others : CustomerGroup)
  (seniorPolicy : BankPolicy)
  (otherPolicy : BankPolicy)
  (h1 : seniors.repaymentReliability > others.repaymentReliability)
  (h2 : seniors.incomeStability > others.incomeStability)
  (h3 : seniors.savingInclination > others.savingInclination)
  (h4 : seniors.longTermPreference > others.longTermPreference)
  (h5 : seniorPolicy.depositRate > otherPolicy.depositRate)
  (h6 : seniorPolicy.loanRate < otherPolicy.loanRate) :
  bankProfit seniors seniorPolicy > bankProfit others otherPolicy :=
sorry

end senior_policy_more_profitable_l2136_213609


namespace oddSum_not_prime_l2136_213699

def oddSum (n : Nat) : Nat :=
  List.sum (List.map (fun i => 2 * i - 1) (List.range n))

theorem oddSum_not_prime (n : Nat) (h : 2 ≤ n ∧ n ≤ 5) : ¬ Nat.Prime (oddSum n) := by
  sorry

end oddSum_not_prime_l2136_213699
