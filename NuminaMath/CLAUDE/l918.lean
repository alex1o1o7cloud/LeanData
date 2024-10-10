import Mathlib

namespace existence_of_equal_elements_l918_91897

theorem existence_of_equal_elements (n p q : ℕ) (x : ℕ → ℤ)
  (h_pos : 0 < n ∧ 0 < p ∧ 0 < q)
  (h_n_gt : n > p + q)
  (h_x_bounds : x 0 = 0 ∧ x n = 0)
  (h_x_diff : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (x i - x (i-1) = p ∨ x i - x (i-1) = -q)) :
  ∃ (i j : ℕ), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j :=
by sorry

end existence_of_equal_elements_l918_91897


namespace survey_theorem_l918_91888

def survey_problem (total : ℕ) (high_bp : ℕ) (heart : ℕ) (diabetes : ℕ) 
                   (bp_heart : ℕ) (bp_diabetes : ℕ) (heart_diabetes : ℕ) 
                   (all_three : ℕ) : Prop :=
  let teachers_with_condition := 
    (high_bp - bp_heart - bp_diabetes + all_three) +
    (heart - bp_heart - heart_diabetes + all_three) +
    (diabetes - bp_diabetes - heart_diabetes + all_three) +
    (bp_heart - all_three) + (bp_diabetes - all_three) + 
    (heart_diabetes - all_three) + all_three
  let teachers_without_condition := total - teachers_with_condition
  (teachers_without_condition : ℚ) / (total : ℚ) * 100 = 50/3

theorem survey_theorem : 
  survey_problem 150 90 50 30 25 10 15 5 :=
by
  sorry

end survey_theorem_l918_91888


namespace algebraic_expression_symmetry_l918_91807

/-- Given an algebraic expression ax^5 + bx^3 + cx - 8, if its value is 6 when x = 5,
    then its value is -22 when x = -5 -/
theorem algebraic_expression_symmetry (a b c : ℝ) :
  (5^5 * a + 5^3 * b + 5 * c - 8 = 6) →
  ((-5)^5 * a + (-5)^3 * b + (-5) * c - 8 = -22) :=
by sorry

end algebraic_expression_symmetry_l918_91807


namespace charity_donation_proof_l918_91802

/-- Calculates the donation amount for a charity draw ticket given initial amount,
    winnings, purchases, and final amount. -/
def calculate_donation (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                       (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) : ℤ :=
  initial_amount + prize + lottery_win - water_cost - lottery_cost - final_amount

/-- Proves that the donation for the charity draw ticket was $4 given the problem conditions. -/
theorem charity_donation_proof (initial_amount : ℤ) (prize : ℤ) (lottery_win : ℤ) 
                               (water_cost : ℤ) (lottery_cost : ℤ) (final_amount : ℤ) 
                               (h1 : initial_amount = 10)
                               (h2 : prize = 90)
                               (h3 : lottery_win = 65)
                               (h4 : water_cost = 1)
                               (h5 : lottery_cost = 1)
                               (h6 : final_amount = 94) :
  calculate_donation initial_amount prize lottery_win water_cost lottery_cost final_amount = 4 :=
by sorry

#eval calculate_donation 10 90 65 1 1 94

end charity_donation_proof_l918_91802


namespace monthly_expenses_calculation_l918_91882

-- Define the monthly deposit
def monthly_deposit : ℕ := 5000

-- Define the annual savings
def annual_savings : ℕ := 4800

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem to prove
theorem monthly_expenses_calculation :
  (monthly_deposit * months_in_year - annual_savings) / months_in_year = 4600 :=
by sorry

end monthly_expenses_calculation_l918_91882


namespace expression_simplification_l918_91815

theorem expression_simplification (a c x y : ℝ) (h : c*x^2 + c*y^2 ≠ 0) :
  (c*x^2*(a^2*x^3 + 3*a^2*y^3 + c^2*y^3) + c*y^2*(a^2*x^3 + 3*c^2*x^3 + c^2*y^3)) / (c*x^2 + c*y^2)
  = a^2*x^3 + 3*c*x^3 + c^2*y^3 :=
by sorry

end expression_simplification_l918_91815


namespace sphere_surface_area_tangent_to_cube_l918_91814

/-- The surface area of a sphere tangent to all six faces of a cube with edge length 2 is 4π. -/
theorem sphere_surface_area_tangent_to_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 → 
  sphere_radius = cube_edge_length / 2 → 
  4 * Real.pi * sphere_radius^2 = 4 * Real.pi := by
  sorry

end sphere_surface_area_tangent_to_cube_l918_91814


namespace quadratic_roots_property_l918_91855

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
  sorry

end quadratic_roots_property_l918_91855


namespace complement_intersection_theorem_l918_91873

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x ≤ 3}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl M ∩ N) = {x | x < -1 ∨ (2 < x ∧ x ≤ 3)} :=
sorry

end complement_intersection_theorem_l918_91873


namespace rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l918_91883

/-- The time difference between a rabbit digging a hole and a beaver building a dam -/
theorem rabbit_beaver_time_difference : ℝ → Prop :=
  fun time_difference =>
    ∀ (rabbit_count rabbit_time hole_count : ℝ)
      (beaver_count beaver_time dam_count : ℝ),
    rabbit_count > 0 →
    rabbit_time > 0 →
    hole_count > 0 →
    beaver_count > 0 →
    beaver_time > 0 →
    dam_count > 0 →
    rabbit_count * rabbit_time * 60 / hole_count = 100 →
    beaver_count * beaver_time / dam_count = 90 →
    rabbit_count = 3 →
    rabbit_time = 5 →
    hole_count = 9 →
    beaver_count = 5 →
    beaver_time = 36 / 60 →
    dam_count = 2 →
    time_difference = 10

theorem rabbit_beaver_time_difference_holds : rabbit_beaver_time_difference 10 := by
  sorry

end rabbit_beaver_time_difference_rabbit_beaver_time_difference_holds_l918_91883


namespace ladies_walking_group_miles_l918_91887

/-- Calculates the total miles walked by a group of ladies over a number of days -/
def totalMilesWalked (groupSize : ℕ) (daysWalked : ℕ) (groupMiles : ℕ) (jamieExtra : ℕ) (sueExtra : ℕ) : ℕ :=
  groupSize * groupMiles * daysWalked + (jamieExtra + sueExtra) * daysWalked

/-- Theorem stating the total miles walked by the group under given conditions -/
theorem ladies_walking_group_miles :
  ∀ d : ℕ, totalMilesWalked 5 d 3 2 1 = 18 * d :=
by
  sorry

#check ladies_walking_group_miles

end ladies_walking_group_miles_l918_91887


namespace no_integer_solution_l918_91839

theorem no_integer_solution : ∀ x y : ℤ, (x + 7) * (x + 6) ≠ 8 * y + 3 := by
  sorry

end no_integer_solution_l918_91839


namespace green_hats_count_l918_91854

/-- Proves that the number of green hats is 28 given the problem conditions --/
theorem green_hats_count : ∀ (blue green red : ℕ),
  blue + green + red = 85 →
  6 * blue + 7 * green + 8 * red = 600 →
  blue = 3 * green ∧ green = 2 * red →
  green = 28 := by
  sorry

end green_hats_count_l918_91854


namespace square_land_area_l918_91851

/-- The area of a square land plot with side length 32 units is 1024 square units. -/
theorem square_land_area (side_length : ℝ) (h : side_length = 32) : 
  side_length * side_length = 1024 := by
  sorry

end square_land_area_l918_91851


namespace power_product_equals_negative_eighth_l918_91804

theorem power_product_equals_negative_eighth (x : ℝ) (n : ℕ) :
  x = -0.125 → (x^(n+1) * 8^n = -0.125) := by
  sorry

end power_product_equals_negative_eighth_l918_91804


namespace product_of_fractions_l918_91824

theorem product_of_fractions : (1/2 : ℚ) * (3/5 : ℚ) * (5/6 : ℚ) = (1/4 : ℚ) := by
  sorry

end product_of_fractions_l918_91824


namespace odd_function_a_value_l918_91827

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_a_value :
  ∃ a : ℝ, IsOdd (fun x ↦ lg ((2 / (1 - x)) + a)) ↔ a = -1 := by
  sorry

end odd_function_a_value_l918_91827


namespace power_plus_one_div_square_int_l918_91835

theorem power_plus_one_div_square_int (n : ℕ) : n > 1 →
  (∃ k : ℤ, (2^n + 1 : ℤ) = k * n^2) ↔ n = 3 :=
by sorry

end power_plus_one_div_square_int_l918_91835


namespace algebraic_expression_equality_l918_91861

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 8 = 7 → 3*x^2 + 9*x - 2 = -5 := by
  sorry

end algebraic_expression_equality_l918_91861


namespace bag_balls_count_l918_91878

theorem bag_balls_count (red_balls : ℕ) (white_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 → 
  prob_red = 1/4 → 
  prob_red = red_balls / (red_balls + white_balls) →
  white_balls = 12 := by
sorry

end bag_balls_count_l918_91878


namespace circle_has_longest_perimeter_l918_91889

/-- The perimeter of a square with side length 7 cm -/
def square_perimeter : ℝ := 4 * 7

/-- The perimeter of an equilateral triangle with side length 9 cm -/
def triangle_perimeter : ℝ := 3 * 9

/-- The perimeter of a circle with radius 5 cm, using π = 3 -/
def circle_perimeter : ℝ := 2 * 3 * 5

theorem circle_has_longest_perimeter :
  circle_perimeter > square_perimeter ∧ circle_perimeter > triangle_perimeter :=
sorry

end circle_has_longest_perimeter_l918_91889


namespace smallest_number_with_properties_l918_91891

def ends_in_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * (10 ^ d) + (n / 10)

theorem smallest_number_with_properties :
  ∃ (n : ℕ),
    ends_in_6 n ∧
    move_6_to_front n = 4 * n ∧
    ∀ (m : ℕ), (ends_in_6 m ∧ move_6_to_front m = 4 * m) → n ≤ m ∧
    n = 153846 :=
by sorry

end smallest_number_with_properties_l918_91891


namespace prime_relations_l918_91833

theorem prime_relations (p : ℕ) : 
  (Prime p ∧ Prime (8*p - 1)) → (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8*p + 1) ∧
  (Prime p ∧ Prime (8*p^2 + 1)) → Prime (8*p^2 - 1) :=
sorry

end prime_relations_l918_91833


namespace distance_ratios_sum_to_one_l918_91801

theorem distance_ratios_sum_to_one (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2 + z^2)
  let c := x / r
  let s := y / r
  let z_r := z / r
  s^2 - c^2 + z_r^2 = 1 := by
  sorry

end distance_ratios_sum_to_one_l918_91801


namespace sugar_amount_proof_l918_91893

/-- The price of a kilogram of sugar in dollars -/
def sugar_price : ℝ := 1.50

/-- The number of kilograms of sugar bought -/
def sugar_bought : ℝ := 2

/-- The price of a kilogram of salt in dollars -/
def salt_price : ℝ := (5 - 3 * sugar_price)

theorem sugar_amount_proof :
  sugar_bought * sugar_price + 5 * salt_price = 5.50 ∧
  3 * sugar_price + salt_price = 5 →
  sugar_bought = 2 :=
by sorry

end sugar_amount_proof_l918_91893


namespace pythagorean_triple_divisibility_l918_91868

theorem pythagorean_triple_divisibility (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (k l m : ℤ), 
    (a = 3*k ∨ b = 3*k ∨ c = 3*k) ∧
    (a = 4*l ∨ b = 4*l ∨ c = 4*l) ∧
    (a = 5*m ∨ b = 5*m ∨ c = 5*m) := by
  sorry

end pythagorean_triple_divisibility_l918_91868


namespace salesman_pear_sales_l918_91832

/-- A salesman's pear sales problem -/
theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  morning_sales = 120 →
  afternoon_sales = 240 →
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 360 := by
  sorry

#check salesman_pear_sales

end salesman_pear_sales_l918_91832


namespace election_winner_margin_l918_91816

theorem election_winner_margin (total_votes : ℕ) (winner_votes : ℕ) :
  (winner_votes : ℝ) = 0.56 * total_votes →
  winner_votes = 1344 →
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end election_winner_margin_l918_91816


namespace smallest_gcd_ef_l918_91820

theorem smallest_gcd_ef (d e f : ℕ+) (h1 : Nat.gcd d e = 210) (h2 : Nat.gcd d f = 770) :
  ∃ (e' f' : ℕ+), Nat.gcd d e' = 210 ∧ Nat.gcd d f' = 770 ∧ 
  Nat.gcd e' f' = 10 ∧ ∀ (e'' f'' : ℕ+), Nat.gcd d e'' = 210 → Nat.gcd d f'' = 770 → 
  Nat.gcd e'' f'' ≥ 10 :=
sorry

end smallest_gcd_ef_l918_91820


namespace goose_egg_calculation_l918_91811

theorem goose_egg_calculation (total_survived : ℕ) 
  (hatch_rate : ℚ) (first_month_survival : ℚ) 
  (first_year_death : ℚ) (migration_rate : ℚ) 
  (predator_survival : ℚ) :
  hatch_rate = 1/3 →
  first_month_survival = 4/5 →
  first_year_death = 3/5 →
  migration_rate = 1/4 →
  predator_survival = 2/3 →
  total_survived = 140 →
  ∃ (total_eggs : ℕ), 
    total_eggs = 1050 ∧
    (total_eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_year_death) * 
    (1 - migration_rate) * predator_survival = total_survived := by
  sorry

#eval 1050

end goose_egg_calculation_l918_91811


namespace savings_account_growth_l918_91870

/-- Calculates the final amount in a savings account with compound interest. -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem savings_account_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 3046.28| < ε :=
by sorry

end savings_account_growth_l918_91870


namespace tank_length_calculation_l918_91800

/-- Given a rectangular field and a tank dug within it, this theorem proves
    the length of the tank when the excavated earth raises the field level. -/
theorem tank_length_calculation (field_length field_width tank_width tank_depth level_rise : ℝ)
  (h1 : field_length = 90)
  (h2 : field_width = 50)
  (h3 : tank_width = 20)
  (h4 : tank_depth = 4)
  (h5 : level_rise = 0.5)
  (h6 : tank_width < field_width)
  (h7 : ∀ tank_length, tank_length > 0 → tank_length < field_length) :
  ∃ tank_length : ℝ,
    tank_length > 0 ∧
    tank_length < field_length ∧
    tank_length * tank_width * tank_depth =
      (field_length * field_width - tank_length * tank_width) * level_rise ∧
    tank_length = 25 := by
  sorry


end tank_length_calculation_l918_91800


namespace hyperbola_foci_distance_l918_91830

-- Define the asymptotes
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 1

-- Define the point the hyperbola passes through
def point : ℝ × ℝ := (5, 7)

-- Define the hyperbola (implicitly)
def is_on_hyperbola (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ((y - 2)^2 / a^2) - ((x + 1/2)^2 / b^2) = 1

-- Theorem statement
theorem hyperbola_foci_distance :
  ∃ (f1 f2 : ℝ × ℝ),
    is_on_hyperbola point.1 point.2 ∧
    ‖f1 - f2‖ = 15 := by
  sorry

end hyperbola_foci_distance_l918_91830


namespace rock_collection_inconsistency_l918_91808

theorem rock_collection_inconsistency (J : ℤ) : ¬ (∃ (jose albert : ℤ),
  jose = J - 14 ∧
  albert = jose + 20 ∧
  albert = J + 6) := by
  sorry

end rock_collection_inconsistency_l918_91808


namespace arithmetic_mean_arrangement_l918_91825

theorem arithmetic_mean_arrangement (n : ℕ+) :
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ (i j k : Fin n), i < k ∧ k < j →
      (p i + p j : ℚ) / 2 ≠ p k := by
  sorry

end arithmetic_mean_arrangement_l918_91825


namespace tan_sum_reciprocal_implies_sin_double_angle_l918_91886

theorem tan_sum_reciprocal_implies_sin_double_angle (θ : Real) 
  (h : Real.tan θ + (Real.tan θ)⁻¹ = 4) : 
  Real.sin (2 * θ) = 1/2 := by sorry

end tan_sum_reciprocal_implies_sin_double_angle_l918_91886


namespace derivative_of_f_at_1_l918_91841

def f (x : ℝ) := x^2

theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end derivative_of_f_at_1_l918_91841


namespace book_loss_percentage_l918_91809

/-- Proves the loss percentage on a book given specific conditions --/
theorem book_loss_percentage
  (total_cost : ℝ)
  (cost_book1 : ℝ)
  (gain_percentage : ℝ)
  (h1 : total_cost = 540)
  (h2 : cost_book1 = 315)
  (h3 : gain_percentage = 19)
  (h4 : ∃ (selling_price : ℝ),
    selling_price = cost_book1 * (1 - loss_percentage / 100) ∧
    selling_price = (total_cost - cost_book1) * (1 + gain_percentage / 100)) :
  ∃ (loss_percentage : ℝ), loss_percentage = 15 := by
sorry


end book_loss_percentage_l918_91809


namespace numerator_of_x_l918_91896

theorem numerator_of_x (x y a : ℝ) : 
  x + y = -10 → 
  x^2 + y^2 = 50 → 
  x = a / y → 
  a = 25 := by sorry

end numerator_of_x_l918_91896


namespace four_digit_square_palindromes_l918_91860

/-- A function that checks if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that checks if a number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  digits = digits.reverse

/-- The main theorem stating that there are exactly 3 numbers satisfying all conditions -/
theorem four_digit_square_palindromes :
  ∃! (s : Finset ℕ), s.card = 3 ∧ 
  (∀ n ∈ s, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n) ∧
  (∀ n, is_four_digit n → is_perfect_square n → is_palindrome n → n ∈ s) :=
sorry

end four_digit_square_palindromes_l918_91860


namespace perfect_square_between_prime_sums_l918_91844

/-- Represents the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Represents the sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ :=
  (List.range n).map nthPrime |>.sum

/-- Theorem: For any n, there exists a perfect square between the sum of the first n primes
    and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ m : ℕ, sumFirstNPrimes n ≤ m^2 ∧ m^2 ≤ sumFirstNPrimes (n+1) := by sorry

end perfect_square_between_prime_sums_l918_91844


namespace final_amount_is_correct_l918_91857

-- Define the quantities and prices of fruits
def grapes_quantity : ℝ := 15
def grapes_price : ℝ := 98
def mangoes_quantity : ℝ := 8
def mangoes_price : ℝ := 120
def pineapples_quantity : ℝ := 5
def pineapples_price : ℝ := 75
def oranges_quantity : ℝ := 10
def oranges_price : ℝ := 60

-- Define the discount and tax rates
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08

-- Define the function to calculate the final amount
def calculate_final_amount : ℝ :=
  let total_cost := grapes_quantity * grapes_price + 
                    mangoes_quantity * mangoes_price + 
                    pineapples_quantity * pineapples_price + 
                    oranges_quantity * oranges_price
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

-- Theorem statement
theorem final_amount_is_correct : 
  calculate_final_amount = 3309.66 := by sorry

end final_amount_is_correct_l918_91857


namespace square_plus_one_positive_l918_91869

theorem square_plus_one_positive (a : ℚ) : 0 < a^2 + 1 := by
  sorry

end square_plus_one_positive_l918_91869


namespace find_number_l918_91847

theorem find_number : ∃ x : ℝ, x - (3/5) * x = 56 ∧ x = 140 := by
  sorry

end find_number_l918_91847


namespace quadrilateral_area_is_40_l918_91828

/-- Represents a triangle divided into four triangles and one quadrilateral -/
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  triangle4_area : ℝ
  quadrilateral_area : ℝ

/-- The sum of all areas equals the total area of the triangle -/
def area_sum (dt : DividedTriangle) : Prop :=
  dt.total_area = dt.triangle1_area + dt.triangle2_area + dt.triangle3_area + dt.triangle4_area + dt.quadrilateral_area

/-- The theorem stating that given the areas of the four triangles, the area of the quadrilateral is 40 -/
theorem quadrilateral_area_is_40 (dt : DividedTriangle) 
  (h1 : dt.triangle1_area = 5)
  (h2 : dt.triangle2_area = 10)
  (h3 : dt.triangle3_area = 10)
  (h4 : dt.triangle4_area = 15)
  (h_sum : area_sum dt) : 
  dt.quadrilateral_area = 40 := by
  sorry

end quadrilateral_area_is_40_l918_91828


namespace train_crossing_time_l918_91849

/-- Calculates the time for a train to cross a signal pole given its length, 
    the length of a platform it crosses, and the time it takes to cross the platform. -/
theorem train_crossing_time (train_length platform_length : ℝ) (platform_crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : platform_length = 175)
  (h3 : platform_crossing_time = 39) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end train_crossing_time_l918_91849


namespace hyperbola_equation_l918_91853

def ellipse_equation (x y : ℝ) : Prop := x^2 + y^2/2 = 1

def hyperbola_vertices (h_vertices : ℝ × ℝ) (e_vertices : ℝ × ℝ) : Prop :=
  h_vertices = e_vertices

def eccentricity_product (e_hyperbola e_ellipse : ℝ) : Prop :=
  e_hyperbola * e_ellipse = 1

theorem hyperbola_equation 
  (h_vertices : ℝ × ℝ) 
  (e_vertices : ℝ × ℝ) 
  (e_hyperbola e_ellipse : ℝ) :
  hyperbola_vertices h_vertices e_vertices →
  eccentricity_product e_hyperbola e_ellipse →
  ∃ (x y : ℝ), y^2 - x^2 = 2 :=
sorry

end hyperbola_equation_l918_91853


namespace xy_value_l918_91821

theorem xy_value (x y : ℝ) (h : |3*x + y - 2| + (2*x + 3*y + 1)^2 = 0) : x * y = -1 := by
  sorry

end xy_value_l918_91821


namespace percentage_increase_men_is_twenty_percent_l918_91862

/-- Represents the population data and conditions --/
structure PopulationData where
  men_1990 : ℕ
  women_1990 : ℕ
  boys_1990 : ℕ
  total_1994 : ℕ
  boys_1994 : ℕ

/-- Calculates the percentage increase in men given the population data --/
def percentageIncreaseMen (data : PopulationData) : ℚ :=
  let women_1994 := data.women_1990 + data.boys_1990 * data.women_1990 / data.women_1990
  let men_1994 := data.total_1994 - women_1994 - data.boys_1994
  (men_1994 - data.men_1990) * 100 / data.men_1990

/-- Theorem stating that the percentage increase in men is 20% --/
theorem percentage_increase_men_is_twenty_percent (data : PopulationData) 
  (h1 : data.men_1990 = 5000)
  (h2 : data.women_1990 = 3000)
  (h3 : data.boys_1990 = 2000)
  (h4 : data.total_1994 = 13000)
  (h5 : data.boys_1994 = data.boys_1990) :
  percentageIncreaseMen data = 20 := by
  sorry

#eval percentageIncreaseMen {
  men_1990 := 5000,
  women_1990 := 3000,
  boys_1990 := 2000,
  total_1994 := 13000,
  boys_1994 := 2000
}

end percentage_increase_men_is_twenty_percent_l918_91862


namespace price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l918_91843

-- Define constants
def cost_price : ℝ := 240
def original_price : ℝ := 400
def initial_sales : ℝ := 200
def sales_increase_rate : ℝ := 4
def target_profit : ℝ := 41600
def impossible_profit : ℝ := 50000

-- Define function for weekly profit based on price reduction
def weekly_profit (price_reduction : ℝ) : ℝ :=
  (original_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: Price reduction options
theorem price_reduction_options :
  ∃ (x y : ℝ), x ≠ y ∧ weekly_profit x = target_profit ∧ weekly_profit y = target_profit :=
sorry

-- Theorem 2: Best discount percentage
theorem best_discount_percentage :
  ∃ (best_reduction : ℝ), weekly_profit best_reduction = target_profit ∧
    ∀ (other_reduction : ℝ), weekly_profit other_reduction = target_profit →
      best_reduction ≥ other_reduction ∧
      (original_price - best_reduction) / original_price = 0.8 :=
sorry

-- Theorem 3: Impossibility of higher profit
theorem impossibility_of_higher_profit :
  ∀ (price_reduction : ℝ), weekly_profit price_reduction ≠ impossible_profit :=
sorry

end price_reduction_options_best_discount_percentage_impossibility_of_higher_profit_l918_91843


namespace sqrt_x_div_sqrt_y_l918_91817

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  ((1/3)^2 + (1/4)^2) / ((1/5)^2 + (1/6)^2) = 21*x / (65*y) →
  Real.sqrt x / Real.sqrt y = 25 * Real.sqrt 65 / (2 * Real.sqrt 1281) := by
  sorry

end sqrt_x_div_sqrt_y_l918_91817


namespace perpendicular_vectors_m_equals_two_l918_91805

/-- Two vectors a and b in 2D space -/
def a : Fin 2 → ℝ := ![(-2 : ℝ), 3]
def b : ℝ → Fin 2 → ℝ := λ m => ![3, m]

/-- The dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- Theorem: If vectors a and b are perpendicular, then m = 2 -/
theorem perpendicular_vectors_m_equals_two :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = 2 := by
  sorry

end perpendicular_vectors_m_equals_two_l918_91805


namespace prime_power_sum_l918_91898

theorem prime_power_sum (w x y z : ℕ) :
  2^w * 3^x * 5^y * 7^z = 840 →
  2*w + 3*x + 5*y + 7*z = 21 := by
  sorry

end prime_power_sum_l918_91898


namespace greatest_valid_integer_l918_91892

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 18 = 6

theorem greatest_valid_integer : 
  (∀ m, is_valid m → m ≤ 174) ∧ is_valid 174 := by sorry

end greatest_valid_integer_l918_91892


namespace previous_year_300th_day_l918_91842

/-- Represents the days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Returns the day of the week after n days -/
def addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => nextDay (addDays d n)

theorem previous_year_300th_day 
  (current_year_200th_day : DayOfWeek)
  (next_year_100th_day : DayOfWeek)
  (h1 : current_year_200th_day = DayOfWeek.sunday)
  (h2 : next_year_100th_day = DayOfWeek.sunday)
  : addDays DayOfWeek.monday 299 = current_year_200th_day :=
by sorry

#check previous_year_300th_day

end previous_year_300th_day_l918_91842


namespace tommys_nickels_l918_91867

/-- The number of coins Tommy has in his collection. -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ
  dollar_coins : ℕ

/-- Tommy's coin collection satisfies the given conditions. -/
def valid_collection (c : CoinCollection) : Prop :=
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.quarters = 4 ∧
  c.pennies = 10 * c.quarters ∧
  c.half_dollars = c.quarters + 5 ∧
  c.dollar_coins = 3 * c.half_dollars

/-- The number of nickels in Tommy's collection is 100. -/
theorem tommys_nickels (c : CoinCollection) (h : valid_collection c) : c.nickels = 100 := by
  sorry

end tommys_nickels_l918_91867


namespace halfway_between_one_fourth_and_one_seventh_l918_91874

theorem halfway_between_one_fourth_and_one_seventh :
  let x : ℚ := 11 / 56
  (x - 1 / 4 : ℚ) = (1 / 7 - x : ℚ) ∧ 
  x = (1 / 4 + 1 / 7) / 2 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l918_91874


namespace john_phone_cost_l918_91831

theorem john_phone_cost (alan_price : ℝ) (john_percentage : ℝ) : 
  alan_price = 2000 → john_percentage = 0.02 → 
  alan_price * (1 + john_percentage) = 2040 := by
  sorry

end john_phone_cost_l918_91831


namespace cloth_sale_theorem_l918_91858

/-- Represents the sale of cloth with given parameters. -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold. -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given parameters, 600 metres of cloth were sold. -/
theorem cloth_sale_theorem (sale : ClothSale) 
  (h1 : sale.totalSellingPrice = 18000)
  (h2 : sale.lossPerMetre = 5)
  (h3 : sale.costPricePerMetre = 35) :
  metresSold sale = 600 := by
  sorry

#eval metresSold { totalSellingPrice := 18000, lossPerMetre := 5, costPricePerMetre := 35 }

end cloth_sale_theorem_l918_91858


namespace prime_cube_plus_five_prime_l918_91829

theorem prime_cube_plus_five_prime (p : ℕ) 
  (hp : Nat.Prime p) 
  (hp_cube : Nat.Prime (p^3 + 5)) : 
  p^5 - 7 = 25 := by
sorry

end prime_cube_plus_five_prime_l918_91829


namespace parallel_line_to_plane_transitive_parallel_planes_skew_lines_parallel_planes_l918_91834

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β γ : Plane)

-- Axioms
axiom different_lines : m ≠ n
axiom different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem A
theorem parallel_line_to_plane 
  (h1 : parallel_lines l m) 
  (h2 : line_in_plane m α) 
  (h3 : ¬ line_in_plane l α) : 
  parallel_line_plane l α :=
sorry

-- Theorem C
theorem transitive_parallel_planes 
  (h1 : parallel_planes α β) 
  (h2 : parallel_planes β γ) : 
  parallel_planes α γ :=
sorry

-- Theorem D
theorem skew_lines_parallel_planes 
  (h1 : skew_lines l m)
  (h2 : parallel_line_plane l α)
  (h3 : parallel_line_plane m α)
  (h4 : parallel_line_plane l β)
  (h5 : parallel_line_plane m β) :
  parallel_planes α β :=
sorry

end parallel_line_to_plane_transitive_parallel_planes_skew_lines_parallel_planes_l918_91834


namespace five_digit_multiple_of_nine_l918_91813

theorem five_digit_multiple_of_nine : ∃ (d : ℕ), d < 10 ∧ (56780 + d) % 9 = 0 := by
  -- The proof goes here
  sorry

end five_digit_multiple_of_nine_l918_91813


namespace framed_photo_border_area_l918_91866

/-- The area of the border of a framed rectangular photograph. -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 12) 
  (h2 : photo_width = 15) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

end framed_photo_border_area_l918_91866


namespace harrys_travel_time_l918_91810

/-- Harry's travel time calculation -/
theorem harrys_travel_time :
  let bus_time_so_far : ℕ := 15
  let remaining_bus_time : ℕ := 25
  let total_bus_time : ℕ := bus_time_so_far + remaining_bus_time
  let walking_time : ℕ := total_bus_time / 2
  total_bus_time + walking_time = 60 := by sorry

end harrys_travel_time_l918_91810


namespace delta_max_success_ratio_l918_91856

/-- Represents a participant's score in a single day of the competition -/
structure DailyScore where
  scored : ℕ
  attempted : ℕ
  success_ratio : scored ≤ attempted

/-- Represents a participant's total score across two days -/
structure TotalScore where
  day1 : DailyScore
  day2 : DailyScore
  total_attempted : day1.attempted + day2.attempted = 500

/-- Gamma's scores for the two days -/
def gamma : TotalScore := {
  day1 := { scored := 180, attempted := 280, success_ratio := by sorry },
  day2 := { scored := 120, attempted := 220, success_ratio := by sorry },
  total_attempted := by sorry
}

/-- Delta's scores for the two days -/
structure DeltaScore extends TotalScore where
  day1_ratio_less : (day1.scored : ℚ) / day1.attempted < (gamma.day1.scored : ℚ) / gamma.day1.attempted
  day2_ratio_less : (day2.scored : ℚ) / day2.attempted < (gamma.day2.scored : ℚ) / gamma.day2.attempted

theorem delta_max_success_ratio :
  ∀ delta : DeltaScore,
    (delta.day1.scored + delta.day2.scored : ℚ) / 500 ≤ 409 / 500 := by sorry

end delta_max_success_ratio_l918_91856


namespace cold_drink_pitcher_l918_91871

/-- Represents the recipe for a cold drink -/
structure Recipe where
  iced_tea : Rat
  lemonade : Rat

/-- Calculates the total amount of drink for a given recipe -/
def total_drink (r : Recipe) : Rat :=
  r.iced_tea + r.lemonade

/-- Represents the contents of a pitcher -/
structure Pitcher where
  lemonade : Rat
  total : Rat

/-- The theorem to be proved -/
theorem cold_drink_pitcher (r : Recipe) (p : Pitcher) :
  r.iced_tea = 1/4 →
  r.lemonade = 5/4 →
  p.lemonade = 15 →
  p.total = 18 :=
by sorry

end cold_drink_pitcher_l918_91871


namespace special_ellipse_properties_l918_91872

/-- An ellipse with eccentricity √3/2 and maximum triangle area of 1 -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_max_area : a * b = 2

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : SpecialEllipse) where
  k : ℝ
  m : ℝ
  h_slope_product : (5 : ℝ) / 4 = k^2 + m^2

/-- The theorem statement -/
theorem special_ellipse_properties (E : SpecialEllipse) (L : IntersectingLine E) :
  (E.a = 2 ∧ E.b = 1) ∧
  (∃ (S : ℝ), S = 1 ∧ ∀ (k m : ℝ), (5 : ℝ) / 4 = k^2 + m^2 → S ≥ 
    ((5 - 4*k^2) * (20*k^2 - 1)) / (2 * (4*k^2 + 1))) :=
  sorry

end special_ellipse_properties_l918_91872


namespace operation_result_l918_91819

def universal_set : Set ℝ := Set.univ

def operation (M N : Set ℝ) : Set ℝ := M ∩ (universal_set \ N)

def set_M : Set ℝ := {x : ℝ | |x| ≤ 2}

def set_N : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}

theorem operation_result :
  operation set_M set_N = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end operation_result_l918_91819


namespace gcd_of_repeated_five_digit_integers_l918_91881

theorem gcd_of_repeated_five_digit_integers : 
  ∃ (g : ℕ), 
    (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → g ∣ (n * 10000100001)) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), 10000 ≤ n ∧ n < 100000 → d ∣ (n * 10000100001)) → d ∣ g) ∧
    g = 10000100001 := by
  sorry

end gcd_of_repeated_five_digit_integers_l918_91881


namespace three_digit_number_theorem_l918_91859

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10
  h_a_pos : 0 < a

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.c

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ℕ :=
  100 * n.c + 10 * n.b + n.a

def ThreeDigitNumber.sumDigits (n : ThreeDigitNumber) : ℕ :=
  n.a + n.b + n.c

theorem three_digit_number_theorem (n : ThreeDigitNumber) :
  (n.toNat / n.reverse = 3 ∧ n.toNat % n.reverse = n.sumDigits) →
  (n.toNat = 441 ∨ n.toNat = 882) := by
  sorry

end three_digit_number_theorem_l918_91859


namespace triangle_problem_l918_91865

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_problem (t : Triangle) 
  (h1 : 2 * Real.sin t.A * Real.cos t.B + Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.a = 4 * Real.sqrt 3)
  (h3 : t.b + t.c = 8) : 
  t.A = Real.pi / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : Real) = 4 * Real.sqrt 3 / 3 := by
  sorry

end triangle_problem_l918_91865


namespace girls_in_class_l918_91875

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (girls : ℕ) :
  total = 35 →
  ratio_girls = 3 →
  ratio_boys = 4 →
  girls * ratio_boys = (total - girls) * ratio_girls →
  girls = 15 := by
sorry

end girls_in_class_l918_91875


namespace union_of_sets_l918_91818

open Set

theorem union_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | 1 < x ∧ x ≤ 3} → 
  N = {x : ℝ | 2 < x ∧ x ≤ 5} → 
  M ∪ N = {x : ℝ | 1 < x ∧ x ≤ 5} := by
  sorry

end union_of_sets_l918_91818


namespace vector_subtraction_and_scalar_multiplication_l918_91850

theorem vector_subtraction_and_scalar_multiplication :
  let v₁ : Fin 2 → ℝ := ![3, -8]
  let v₂ : Fin 2 → ℝ := ![2, -6]
  v₁ - 5 • v₂ = ![-7, 22] := by sorry

end vector_subtraction_and_scalar_multiplication_l918_91850


namespace triangle_inequality_l918_91837

/-- 
Given a triangle ABC with circumradius R = 1 and area S = 1/4, 
prove that sqrt(a) + sqrt(b) + sqrt(c) < 1/a + 1/b + 1/c, 
where a, b, and c are the side lengths of the triangle.
-/
theorem triangle_inequality (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_circumradius : (a * b * c) / (4 * (1/4)) = 1) 
  (h_area : (1/4) > 0) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end triangle_inequality_l918_91837


namespace geometric_sequence_ratio_l918_91803

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_a2 : a 2 = Real.sqrt 2)
  (h_a3 : a 3 = Real.rpow 4 (1/3)) :
  (a 1 + a 15) / (a 7 + a 21) = 1/2 := by
sorry

end geometric_sequence_ratio_l918_91803


namespace problem_polygon_area_l918_91885

/-- A polygon in 2D space defined by a list of points --/
def Polygon : Type := List (ℝ × ℝ)

/-- The polygon described in the problem --/
def problemPolygon : Polygon :=
  [(0,0), (5,0), (5,5), (0,5), (0,3), (3,3), (3,0), (0,0)]

/-- Calculates the area of a polygon --/
def polygonArea (p : Polygon) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The area of the problem polygon is 19 square units --/
theorem problem_polygon_area :
  polygonArea problemPolygon = 19 := by
  sorry

end problem_polygon_area_l918_91885


namespace total_stripes_eq_22_l918_91826

/-- The number of stripes on one of Olga's shoes -/
def olga_stripes_per_shoe : ℕ := 3

/-- The number of stripes on one of Rick's shoes -/
def rick_stripes_per_shoe : ℕ := olga_stripes_per_shoe - 1

/-- The number of stripes on one of Hortense's shoes -/
def hortense_stripes_per_shoe : ℕ := olga_stripes_per_shoe * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all pairs of tennis shoes -/
def total_stripes : ℕ := 
  (olga_stripes_per_shoe * shoes_per_person) + 
  (rick_stripes_per_shoe * shoes_per_person) + 
  (hortense_stripes_per_shoe * shoes_per_person)

theorem total_stripes_eq_22 : total_stripes = 22 := by
  sorry

end total_stripes_eq_22_l918_91826


namespace range_of_a_theorem_l918_91823

-- Define the propositions P and q as functions of a
def P (a : ℝ) : Prop := a ≤ -1 ∨ a ≥ 2

def q (a : ℝ) : Prop := a > 3

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 3)

-- Theorem statement
theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ q a) ∧ (P a ∨ q a)) → range_of_a a :=
by
  sorry

end range_of_a_theorem_l918_91823


namespace marble_groups_l918_91836

theorem marble_groups (total_marbles : ℕ) (marbles_per_group : ℕ) (num_groups : ℕ) :
  total_marbles = 64 →
  marbles_per_group = 2 →
  num_groups * marbles_per_group = total_marbles →
  num_groups = 32 := by
  sorry

end marble_groups_l918_91836


namespace combined_swim_time_l918_91895

def freestyle_time : ℕ := 48
def backstroke_time : ℕ := freestyle_time + 4
def butterfly_time : ℕ := backstroke_time + 3
def breaststroke_time : ℕ := butterfly_time + 2

theorem combined_swim_time : 
  freestyle_time + backstroke_time + butterfly_time + breaststroke_time = 212 := by
  sorry

end combined_swim_time_l918_91895


namespace sin_22_5_deg_identity_l918_91852

theorem sin_22_5_deg_identity : 1 - 2 * (Real.sin (22.5 * π / 180))^2 = Real.sqrt 2 / 2 := by
  sorry

end sin_22_5_deg_identity_l918_91852


namespace polynomial_simplification_l918_91863

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 + 6*x^3 
  = 4*x^3 - x^2 + 23*x - 3 := by
sorry

end polynomial_simplification_l918_91863


namespace rhombus_area_l918_91846

/-- The area of a rhombus with side length 20 and one diagonal of length 16 is 64√21 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) : 
  side = 20 → diagonal1 = 16 → diagonal2 = 8 * Real.sqrt 21 →
  (1/2) * diagonal1 * diagonal2 = 64 * Real.sqrt 21 := by
  sorry

end rhombus_area_l918_91846


namespace tv_show_total_watch_time_l918_91884

theorem tv_show_total_watch_time :
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let extra_episodes_in_final_season : ℕ := 4
  let hours_per_episode : ℚ := 1/2

  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + 
    (episodes_per_regular_season + extra_episodes_in_final_season)

  let total_watch_time : ℚ := total_episodes * hours_per_episode

  total_watch_time = 112 := by sorry

end tv_show_total_watch_time_l918_91884


namespace function_decreasing_implies_a_range_l918_91890

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) →
  a ≤ -3 :=
by sorry

end function_decreasing_implies_a_range_l918_91890


namespace base_representation_theorem_l918_91845

theorem base_representation_theorem :
  (∃ b : ℕ, 1 < b ∧ b < 1993 ∧ 1994 = 2 * (1 + b)) ∧
  (∀ b : ℕ, 1 < b → b < 1992 → ¬∃ n : ℕ, n ≥ 2 ∧ 1993 * (b - 1) = b^n - 1) :=
by sorry

end base_representation_theorem_l918_91845


namespace jack_first_half_time_l918_91877

/-- Jack and Jill's hill race problem -/
theorem jack_first_half_time (jill_finish_time jack_second_half_time : ℕ)
  (h1 : jill_finish_time = 32)
  (h2 : jack_second_half_time = 6) :
  let jack_finish_time := jill_finish_time - 7
  jack_finish_time - jack_second_half_time = 19 := by
  sorry

end jack_first_half_time_l918_91877


namespace broadway_show_attendance_l918_91840

/-- The number of children attending a Broadway show -/
def num_children : ℕ := 200

/-- The number of adults attending the Broadway show -/
def num_adults : ℕ := 400

/-- The price of a child's ticket in dollars -/
def child_ticket_price : ℕ := 16

/-- The price of an adult ticket in dollars -/
def adult_ticket_price : ℕ := 32

/-- The total amount collected from ticket sales in dollars -/
def total_amount : ℕ := 16000

theorem broadway_show_attendance :
  num_children = 200 ∧
  num_adults = 400 ∧
  adult_ticket_price = 2 * child_ticket_price ∧
  adult_ticket_price = 32 ∧
  total_amount = num_adults * adult_ticket_price + num_children * child_ticket_price :=
by sorry

end broadway_show_attendance_l918_91840


namespace textbook_cost_decrease_l918_91864

theorem textbook_cost_decrease (original_cost new_cost : ℝ) 
  (h1 : original_cost = 75)
  (h2 : new_cost = 60) :
  (1 - new_cost / original_cost) * 100 = 20 := by
sorry

end textbook_cost_decrease_l918_91864


namespace volume_of_cube_with_triple_surface_area_l918_91806

def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem volume_of_cube_with_triple_surface_area (cube_a_side : ℝ) (cube_b_side : ℝ) :
  cube_volume cube_a_side = 8 →
  cube_surface_area cube_b_side = 3 * cube_surface_area cube_a_side →
  cube_volume cube_b_side = 24 * Real.sqrt 3 := by
  sorry

end volume_of_cube_with_triple_surface_area_l918_91806


namespace sum_distances_focus_to_points_l918_91848

/-- The parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- Theorem: Sum of distances from focus to three points on parabola -/
theorem sum_distances_focus_to_points
  (A B C : ℝ × ℝ)
  (hA : A ∈ Parabola)
  (hB : B ∈ Parabola)
  (hC : C ∈ Parabola)
  (h_sum : F.1 * 3 = A.1 + B.1 + C.1) :
  dist F A + dist F B + dist F C = 12 :=
by
  sorry

end sum_distances_focus_to_points_l918_91848


namespace banana_arrangements_l918_91838

/-- The number of unique arrangements of letters in a word -/
def uniqueArrangements (totalLetters : Nat) (repetitions : List Nat) : Nat :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- The word "BANANA" has 6 letters -/
def totalLetters : Nat := 6

/-- The repetitions of letters in "BANANA": 3 A's, 2 N's, and 1 B (which we don't need to include) -/
def letterRepetitions : List Nat := [3, 2]

/-- Theorem: The number of unique arrangements of letters in "BANANA" is 60 -/
theorem banana_arrangements :
  uniqueArrangements totalLetters letterRepetitions = 60 := by
  sorry

end banana_arrangements_l918_91838


namespace max_discarded_grapes_proof_l918_91822

/-- The number of children among whom the grapes are to be distributed. -/
def num_children : ℕ := 8

/-- The maximum number of grapes that could be discarded. -/
def max_discarded_grapes : ℕ := num_children - 1

/-- Theorem stating that the maximum number of discarded grapes is one less than the number of children. -/
theorem max_discarded_grapes_proof :
  max_discarded_grapes = num_children - 1 :=
by sorry

end max_discarded_grapes_proof_l918_91822


namespace range_of_a_in_linear_inequality_l918_91899

/-- The range of values for parameter 'a' in the inequality 2x - y + a > 0,
    given that only one point among (0,0) and (1,1) is inside the region. -/
theorem range_of_a_in_linear_inequality :
  ∃ (a : ℝ), (∀ x y : ℝ, 2*x - y + a > 0 →
    ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
    (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) ∧
  (-1 < a ∧ a ≤ 0) :=
by sorry

end range_of_a_in_linear_inequality_l918_91899


namespace butter_price_is_correct_l918_91812

/-- Represents the milk and butter sales problem --/
structure MilkButterSales where
  milk_price : ℚ
  milk_to_butter_ratio : ℚ
  num_cows : ℕ
  milk_per_cow : ℚ
  num_customers : ℕ
  milk_per_customer : ℚ
  total_earnings : ℚ

/-- Calculates the price per stick of butter --/
def butter_price (s : MilkButterSales) : ℚ :=
  let total_milk := s.num_cows * s.milk_per_cow
  let milk_sold := s.num_customers * s.milk_per_customer
  let milk_for_butter := total_milk - milk_sold
  let butter_sticks := milk_for_butter * s.milk_to_butter_ratio
  let milk_earnings := milk_sold * s.milk_price
  let butter_earnings := s.total_earnings - milk_earnings
  butter_earnings / butter_sticks

/-- Theorem stating that the butter price is $1.50 given the problem conditions --/
theorem butter_price_is_correct (s : MilkButterSales) 
  (h1 : s.milk_price = 3)
  (h2 : s.milk_to_butter_ratio = 2)
  (h3 : s.num_cows = 12)
  (h4 : s.milk_per_cow = 4)
  (h5 : s.num_customers = 6)
  (h6 : s.milk_per_customer = 6)
  (h7 : s.total_earnings = 144) :
  butter_price s = 3/2 := by
  sorry

end butter_price_is_correct_l918_91812


namespace quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l918_91880

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := k * x^2 - (2*k + 4) * x + k - 6

-- Theorem for the range of k
theorem quadratic_two_roots_range (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ 
  (k > -2/5 ∧ k ≠ 0) :=
sorry

-- Theorem for solutions when k = 1
theorem quadratic_solutions_when_k_is_one :
  ∃ x y : ℝ, x ≠ y ∧ 
  quadratic 1 x = 0 ∧ quadratic 1 y = 0 ∧
  x = 3 + Real.sqrt 14 ∧ y = 3 - Real.sqrt 14 :=
sorry

end quadratic_two_roots_range_quadratic_solutions_when_k_is_one_l918_91880


namespace value_of_a_l918_91876

theorem value_of_a (x a : ℝ) : (x + 1) * (x - 3) = x^2 + a*x - 3 → a = -2 := by
  sorry

end value_of_a_l918_91876


namespace sin_two_theta_l918_91879

theorem sin_two_theta (θ : Real) 
  (h : Real.exp (Real.log 2 * (-2 + 3 * Real.sin θ)) + 1 = Real.exp (Real.log 2 * (1/2 + Real.sin θ))) : 
  Real.sin (2 * θ) = 4 * Real.sqrt 2 / 9 := by
sorry

end sin_two_theta_l918_91879


namespace sine_cosine_product_l918_91894

theorem sine_cosine_product (α : Real) (h : (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end sine_cosine_product_l918_91894
