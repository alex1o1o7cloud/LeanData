import Mathlib

namespace age_ratio_in_two_years_l943_94324

def brother_age : ℕ := 10
def man_age : ℕ := brother_age + 12

theorem age_ratio_in_two_years :
  (man_age + 2) / (brother_age + 2) = 2 := by sorry

end age_ratio_in_two_years_l943_94324


namespace correct_inequality_l943_94360

theorem correct_inequality : 
  (-3 > -5) ∧ 
  ¬(-3 > -2) ∧ 
  ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ 
  ¬(-(1:ℚ)/2 < -(2:ℚ)/3) := by
  sorry

end correct_inequality_l943_94360


namespace min_expression_l943_94317

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x * y) / 2 + 18 / (x * y) ≥ 6 ∧
  ((x * y) / 2 + 18 / (x * y) = 6 → y / 2 + x / 3 ≥ 2) ∧
  ((x * y) / 2 + 18 / (x * y) = 6 ∧ y / 2 + x / 3 = 2 → x = 3 ∧ y = 2) :=
by sorry

end min_expression_l943_94317


namespace horseshoe_cost_per_set_l943_94335

/-- Proves that the cost per set of horseshoes is $20.75 given the initial outlay,
    selling price, number of sets sold, and profit. -/
theorem horseshoe_cost_per_set 
  (initial_outlay : ℝ)
  (selling_price : ℝ)
  (sets_sold : ℕ)
  (profit : ℝ)
  (h1 : initial_outlay = 12450)
  (h2 : selling_price = 50)
  (h3 : sets_sold = 950)
  (h4 : profit = 15337.5)
  (h5 : profit = selling_price * sets_sold - (initial_outlay + cost_per_set * sets_sold)) :
  cost_per_set = 20.75 :=
by
  sorry

#check horseshoe_cost_per_set

end horseshoe_cost_per_set_l943_94335


namespace sequence_matches_l943_94357

def a (n : ℕ) : ℤ := (-1)^n * (1 - 2*n)

theorem sequence_matches : 
  (a 1 = 1) ∧ (a 2 = -3) ∧ (a 3 = 5) ∧ (a 4 = -7) ∧ (a 5 = 9) := by
  sorry

end sequence_matches_l943_94357


namespace power_of_17_mod_26_l943_94390

theorem power_of_17_mod_26 : 17^1999 % 26 = 17 := by
  sorry

end power_of_17_mod_26_l943_94390


namespace cubic_value_l943_94311

theorem cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2010 = 2011 := by
  sorry

end cubic_value_l943_94311


namespace total_soak_time_l943_94327

def grass_soak_time : ℕ := 3
def marinara_soak_time : ℕ := 7
def ink_soak_time : ℕ := 5
def coffee_soak_time : ℕ := 10

def num_grass_stains : ℕ := 3
def num_marinara_stains : ℕ := 1
def num_ink_stains : ℕ := 2
def num_coffee_stains : ℕ := 1

theorem total_soak_time :
  grass_soak_time * num_grass_stains +
  marinara_soak_time * num_marinara_stains +
  ink_soak_time * num_ink_stains +
  coffee_soak_time * num_coffee_stains = 36 := by
  sorry

end total_soak_time_l943_94327


namespace triangle_parallelogram_altitude_l943_94339

theorem triangle_parallelogram_altitude (base : ℝ) (triangle_altitude parallelogram_altitude : ℝ) :
  base > 0 →
  parallelogram_altitude > 0 →
  parallelogram_altitude = 100 →
  (1 / 2 * base * triangle_altitude) = (base * parallelogram_altitude) →
  triangle_altitude = 200 := by
  sorry

end triangle_parallelogram_altitude_l943_94339


namespace percentage_of_five_digit_numbers_with_repeats_l943_94378

def five_digit_numbers : ℕ := 90000

def numbers_without_repeats : ℕ := 9 * 9 * 8 * 7 * 6

def numbers_with_repeats : ℕ := five_digit_numbers - numbers_without_repeats

def percentage_with_repeats : ℚ := numbers_with_repeats / five_digit_numbers

theorem percentage_of_five_digit_numbers_with_repeats :
  (percentage_with_repeats * 100).floor / 10 = 698 / 10 := by sorry

end percentage_of_five_digit_numbers_with_repeats_l943_94378


namespace arithmetic_sequence_middle_term_l943_94370

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first : a 0 = 13) 
  (h_last : a 4 = 37) : 
  a 2 = 25 := by
sorry

end arithmetic_sequence_middle_term_l943_94370


namespace hard_hats_remaining_is_51_l943_94364

/-- Calculates the remaining hard hats after transactions --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ) 
  (carl_pink_taken john_pink_taken : ℕ) : ℕ :=
  let john_green_taken := 2 * john_pink_taken
  let pink_after_taken := pink_initial - carl_pink_taken - john_pink_taken
  let green_after_taken := green_initial - john_green_taken
  let carl_pink_returned := carl_pink_taken / 2
  let john_pink_returned := john_pink_taken / 3
  let john_green_returned := john_green_taken / 3
  let pink_final := pink_after_taken + carl_pink_returned + john_pink_returned
  let green_final := green_after_taken + john_green_returned
  pink_final + green_final + yellow_initial

/-- Theorem stating that the total number of hard hats remaining is 51 --/
theorem hard_hats_remaining_is_51 : 
  remaining_hard_hats 26 15 24 4 6 = 51 := by
  sorry

end hard_hats_remaining_is_51_l943_94364


namespace inequality_proof_l943_94305

theorem inequality_proof (t : Real) (h : 0 ≤ t ∧ t ≤ π / 2) :
  Real.sqrt 2 * (Real.sin t + Real.cos t) ≥ 2 * (Real.sin (2 * t))^(1/4) := by
  sorry

end inequality_proof_l943_94305


namespace minimum_value_implies_a_l943_94396

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0,
    if f takes its minimum value at x = 3, then a = 36 -/
theorem minimum_value_implies_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x > 0 → ∃ f : ℝ → ℝ, f x = 4*x + a/x) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, x > 0 → f x ≥ f 3) →
  a = 36 := by
sorry

end minimum_value_implies_a_l943_94396


namespace max_m_value_l943_94398

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Theorem statement
theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, f x > g x m) → m < 5 := by
  sorry

end max_m_value_l943_94398


namespace solve_equation_l943_94348

/-- Given an equation 19(x + y) + 17 = 19(-x + y) - n where x = 1, prove that n = -55 -/
theorem solve_equation (y : ℝ) : 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n) → 
  (∃ (n : ℝ), 19 * (1 + y) + 17 = 19 * (-1 + y) - n ∧ n = -55) :=
by sorry

end solve_equation_l943_94348


namespace school_construction_problem_l943_94362

/-- School construction problem -/
theorem school_construction_problem
  (total_area : ℝ)
  (demolition_cost : ℝ)
  (construction_cost : ℝ)
  (actual_demolition_ratio : ℝ)
  (actual_construction_ratio : ℝ)
  (greening_cost : ℝ)
  (h1 : total_area = 7200)
  (h2 : demolition_cost = 80)
  (h3 : construction_cost = 700)
  (h4 : actual_demolition_ratio = 1.1)
  (h5 : actual_construction_ratio = 0.8)
  (h6 : greening_cost = 200) :
  ∃ (planned_demolition planned_construction greening_area : ℝ),
    planned_demolition + planned_construction = total_area ∧
    actual_demolition_ratio * planned_demolition + actual_construction_ratio * planned_construction = total_area ∧
    planned_demolition = 4800 ∧
    planned_construction = 2400 ∧
    greening_area = 1488 ∧
    greening_area * greening_cost = 
      (planned_demolition * demolition_cost + planned_construction * construction_cost) -
      (actual_demolition_ratio * planned_demolition * demolition_cost + 
       actual_construction_ratio * planned_construction * construction_cost) :=
by sorry

end school_construction_problem_l943_94362


namespace order_of_operations_4_times_20_plus_30_l943_94314

theorem order_of_operations_4_times_20_plus_30 : 
  let expression := 4 * (20 + 30)
  let correct_order := ["addition", "multiplication"]
  correct_order = ["addition", "multiplication"] := by sorry

end order_of_operations_4_times_20_plus_30_l943_94314


namespace chord_length_perpendicular_bisector_l943_94371

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 15) :
  let c := 2 * r * Real.sqrt 3 / 2
  c = 26 * Real.sqrt 3 := by
sorry

end chord_length_perpendicular_bisector_l943_94371


namespace inequality_count_l943_94301

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
  (n = (ite (x + y < a + b) 1 0 : ℕ) + 
       (ite (x + y^2 < a + b^2) 1 0 : ℕ) + 
       (ite (x * y < a * b) 1 0 : ℕ) + 
       (ite (|x / y| < |a / b|) 1 0 : ℕ)) :=
by sorry

end inequality_count_l943_94301


namespace broken_seashells_l943_94374

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 7) (h2 : unbroken = 3) :
  total - unbroken = 4 := by
  sorry

end broken_seashells_l943_94374


namespace expand_product_l943_94387

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) * (x + 6) = x^3 + 5*x^2 - 26*x - 120 := by
  sorry

end expand_product_l943_94387


namespace prime_triplet_with_perfect_square_sum_l943_94329

theorem prime_triplet_with_perfect_square_sum (p₁ p₂ p₃ : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ → 
  p₂ ≠ p₃ → 
  ∃ x y : ℕ, x^2 = 4 + p₁ * p₂ ∧ y^2 = 4 + p₁ * p₃ → 
  ((p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 3) ∨ (p₁ = 7 ∧ p₂ = 3 ∧ p₃ = 11)) := by
sorry

end prime_triplet_with_perfect_square_sum_l943_94329


namespace smallest_positive_number_l943_94315

theorem smallest_positive_number :
  let a := 8 - 3 * Real.sqrt 10
  let b := 3 * Real.sqrt 10 - 8
  let c := 23 - 6 * Real.sqrt 15
  let d := 58 - 12 * Real.sqrt 30
  let e := 12 * Real.sqrt 30 - 58
  (0 < b) ∧
  (a ≤ 0 ∨ b < a) ∧
  (c ≤ 0 ∨ b < c) ∧
  (d ≤ 0 ∨ b < d) ∧
  (e ≤ 0 ∨ b < e) :=
by sorry

end smallest_positive_number_l943_94315


namespace carly_swimming_time_l943_94344

/-- Carly's swimming practice schedule and total time calculation -/
theorem carly_swimming_time :
  let butterfly_hours_per_day : ℕ := 3
  let butterfly_days_per_week : ℕ := 4
  let backstroke_hours_per_day : ℕ := 2
  let backstroke_days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  
  let butterfly_hours_per_week : ℕ := butterfly_hours_per_day * butterfly_days_per_week
  let backstroke_hours_per_week : ℕ := backstroke_hours_per_day * backstroke_days_per_week
  let total_hours_per_week : ℕ := butterfly_hours_per_week + backstroke_hours_per_week
  let total_hours_per_month : ℕ := total_hours_per_week * weeks_per_month
  
  total_hours_per_month = 96 :=
by
  sorry


end carly_swimming_time_l943_94344


namespace total_current_ages_l943_94341

theorem total_current_ages (amar akbar anthony : ℕ) : 
  (amar - 4) + (akbar - 4) + (anthony - 4) = 54 → amar + akbar + anthony = 66 := by
  sorry

end total_current_ages_l943_94341


namespace tan_problem_l943_94380

theorem tan_problem (α : Real) (h : Real.tan (α + π/3) = 2) :
  (Real.sin (α + 4*π/3) + Real.cos (2*π/3 - α)) /
  (Real.cos (π/6 - α) - Real.sin (α + 5*π/6)) = -3 := by
  sorry

end tan_problem_l943_94380


namespace trains_return_to_initial_positions_l943_94320

/-- Represents a train on a circular track -/
structure Train where
  period : ℕ
  position : ℕ

/-- The state of the metro system -/
structure MetroSystem where
  trains : List Train

/-- Calculates the position of a train after a given number of minutes -/
def trainPosition (t : Train) (minutes : ℕ) : ℕ :=
  minutes % t.period

/-- Checks if all trains are at their initial positions -/
def allTrainsAtInitial (ms : MetroSystem) (minutes : ℕ) : Prop :=
  ∀ t ∈ ms.trains, trainPosition t minutes = 0

/-- The main theorem -/
theorem trains_return_to_initial_positions (ms : MetroSystem) : 
  ms.trains = [⟨14, 0⟩, ⟨16, 0⟩, ⟨18, 0⟩] → allTrainsAtInitial ms 2016 := by
  sorry


end trains_return_to_initial_positions_l943_94320


namespace m_range_characterization_l943_94326

def f (x : ℝ) : ℝ := x^2 + 3

theorem m_range_characterization (m : ℝ) : 
  (∀ x ≥ 1, f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔ 
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end m_range_characterization_l943_94326


namespace car_highway_efficiency_l943_94343

/-- The number of miles the car can travel on the highway with one gallon of gasoline. -/
def highway_miles_per_gallon : ℝ := 38

/-- The number of miles the car can travel in the city with one gallon of gasoline. -/
def city_miles_per_gallon : ℝ := 20

/-- Proves that the car can travel 38 miles on the highway with one gallon of gasoline,
    given the conditions stated in the problem. -/
theorem car_highway_efficiency :
  highway_miles_per_gallon = 38 ∧
  (4 / highway_miles_per_gallon + 4 / city_miles_per_gallon =
   8 / highway_miles_per_gallon * (1 + 0.45000000000000014)) :=
by sorry

end car_highway_efficiency_l943_94343


namespace boutique_hats_count_l943_94331

/-- The total number of hats in the shipment -/
def total_hats : ℕ := 120

/-- The number of hats stored -/
def stored_hats : ℕ := 90

/-- The percentage of hats displayed -/
def displayed_percentage : ℚ := 25 / 100

theorem boutique_hats_count :
  total_hats = stored_hats / (1 - displayed_percentage) := by sorry

end boutique_hats_count_l943_94331


namespace negative_sqrt_three_squared_equals_negative_three_l943_94349

theorem negative_sqrt_three_squared_equals_negative_three :
  -Real.sqrt (3^2) = -3 := by
  sorry

end negative_sqrt_three_squared_equals_negative_three_l943_94349


namespace expression_value_l943_94359

theorem expression_value : 2^2 + (-3)^2 - 7^2 - 2*2*(-3) + 3*7 = -15 := by
  sorry

end expression_value_l943_94359


namespace number_times_five_equals_hundred_l943_94372

theorem number_times_five_equals_hundred (x : ℝ) : x * 5 = 100 → x = 20 := by
  sorry

end number_times_five_equals_hundred_l943_94372


namespace octal_726_to_binary_l943_94379

/-- Converts a single digit from base 8 to its 3-digit binary representation -/
def octalToBinary (digit : Nat) : Fin 8 → Fin 2 × Fin 2 × Fin 2 := sorry

/-- Converts a 3-digit octal number to its 9-digit binary representation -/
def octalToBinaryThreeDigits (a b c : Fin 8) : Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 := sorry

theorem octal_726_to_binary :
  octalToBinaryThreeDigits 7 2 6 = (1, 1, 1, 0, 1, 0, 1, 1, 0) := by sorry

end octal_726_to_binary_l943_94379


namespace root_exists_in_interval_l943_94328

def f (x : ℝ) := x^3 + 3*x - 1

theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 := by
  sorry

end root_exists_in_interval_l943_94328


namespace problem_statement_l943_94310

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) :
  a^2008 - b^2008 = -1 := by
  sorry

end problem_statement_l943_94310


namespace max_consecutive_funny_numbers_l943_94388

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is funny if it's divisible by the sum of its digits plus one -/
def isFunny (n : ℕ) : Prop := n % (sumOfDigits n + 1) = 0

/-- The maximum number of consecutive funny numbers is 1 -/
theorem max_consecutive_funny_numbers :
  ∀ n : ℕ, isFunny n → isFunny (n + 1) → False := by sorry

end max_consecutive_funny_numbers_l943_94388


namespace perpendicular_lines_m_value_l943_94330

/-- Given two lines in the general form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The problem statement --/
theorem perpendicular_lines_m_value :
  ∀ m : ℝ, are_perpendicular m 4 (-2) 2 (-5) 1 → m = 10 := by
  sorry

end perpendicular_lines_m_value_l943_94330


namespace sum_of_coefficients_is_negative_seven_l943_94351

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x : ℝ | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients_is_negative_seven 
  (h_union : A ∪ B = Set.univ)
  (h_intersection : A ∩ B = Set.Ioc 3 4)
  : ∃ (a b : ℝ), B = {x : ℝ | x^2 + a*x + b ≤ 0} ∧ a + b = -7 := by
  sorry

end sum_of_coefficients_is_negative_seven_l943_94351


namespace zoo_animals_count_l943_94340

/-- The number of penguins in the zoo -/
def num_penguins : ℕ := 21

/-- The number of polar bears in the zoo -/
def num_polar_bears : ℕ := 2 * num_penguins

/-- The total number of animals in the zoo -/
def total_animals : ℕ := num_penguins + num_polar_bears

theorem zoo_animals_count : total_animals = 63 := by
  sorry

end zoo_animals_count_l943_94340


namespace souvenir_spending_l943_94333

/-- Given the total spending on souvenirs and the difference between
    key chains & bracelets and t-shirts, proves the amount spent on
    key chains and bracelets. -/
theorem souvenir_spending
  (total : ℚ)
  (difference : ℚ)
  (h1 : total = 548)
  (h2 : difference = 146) :
  let tshirts := (total - difference) / 2
  let keychains_bracelets := tshirts + difference
  keychains_bracelets = 347 := by
sorry

end souvenir_spending_l943_94333


namespace fraction_division_equality_l943_94381

theorem fraction_division_equality : (-1/12 + 1/3 - 1/2) / (-1/18) = 9/2 := by
  sorry

end fraction_division_equality_l943_94381


namespace no_infinite_sqrt_sequence_l943_94375

theorem no_infinite_sqrt_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), n ≥ 1 → (a (n + 2)).val = Int.sqrt ((a (n + 1)).val) + (a n).val) :=
by sorry

end no_infinite_sqrt_sequence_l943_94375


namespace base_2_representation_of_75_l943_94397

theorem base_2_representation_of_75 :
  ∃ (a b c d e f g : ℕ),
    a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 ∧
    75 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_75_l943_94397


namespace power_of_power_l943_94307

-- Define the problem statement
theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end power_of_power_l943_94307


namespace complex_fraction_evaluation_l943_94350

theorem complex_fraction_evaluation :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 := by
  sorry

end complex_fraction_evaluation_l943_94350


namespace intersection_point_of_two_lines_l943_94342

/-- Two lines in 2D space -/
structure Line2D where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

/-- The point lies on the given line -/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.origin.1 + t * l.direction.1, l.origin.2 + t * l.direction.2)

theorem intersection_point_of_two_lines :
  let l1 : Line2D := { origin := (2, 3), direction := (-1, 5) }
  let l2 : Line2D := { origin := (0, 7), direction := (-1, 4) }
  let p : ℝ × ℝ := (6, -17)
  (pointOnLine p l1 ∧ pointOnLine p l2) ∧
  ∀ q : ℝ × ℝ, pointOnLine q l1 ∧ pointOnLine q l2 → q = p :=
by sorry

end intersection_point_of_two_lines_l943_94342


namespace probability_sum_seven_is_one_sixth_l943_94369

/-- The number of faces on each cubic die -/
def dice_faces : ℕ := 6

/-- The number of ways to obtain a sum of 7 -/
def favorable_outcomes : ℕ := 6

/-- The probability of obtaining a sum of 7 when throwing two cubic dice -/
def probability_sum_seven : ℚ := favorable_outcomes / (dice_faces * dice_faces)

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by
sorry

end probability_sum_seven_is_one_sixth_l943_94369


namespace employee_new_salary_is_35000_l943_94366

/-- Calculates the new salary of employees after a salary redistribution --/
def new_employee_salary (emily_original_salary emily_new_salary num_employees employee_original_salary : ℕ) : ℕ :=
  let salary_reduction := emily_original_salary - emily_new_salary
  let additional_per_employee := salary_reduction / num_employees
  employee_original_salary + additional_per_employee

/-- Proves that the new employee salary is $35,000 given the problem conditions --/
theorem employee_new_salary_is_35000 :
  new_employee_salary 1000000 850000 10 20000 = 35000 := by
  sorry

end employee_new_salary_is_35000_l943_94366


namespace gum_cost_800_l943_94325

/-- The cost of gum pieces with a bulk discount -/
def gum_cost (pieces : ℕ) : ℚ :=
  let base_cost := pieces
  let discount_threshold := 500
  let discount_rate := 1 / 10
  let total_cents :=
    if pieces > discount_threshold
    then base_cost * (1 - discount_rate)
    else base_cost
  total_cents / 100

/-- The cost of 800 pieces of gum is $7.20 -/
theorem gum_cost_800 : gum_cost 800 = 72 / 10 := by
  sorry

end gum_cost_800_l943_94325


namespace dividend_problem_l943_94354

theorem dividend_problem (dividend divisor quotient : ℕ) : 
  dividend + divisor + quotient = 103 →
  quotient = 3 →
  dividend % divisor = 0 →
  dividend / divisor = quotient →
  dividend = 75 := by
sorry

end dividend_problem_l943_94354


namespace product_inequality_l943_94312

theorem product_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a + 1) * (b + 1) * (a + c) * (b + c) > 16 * a * b * c := by
  sorry

end product_inequality_l943_94312


namespace parabola_focus_l943_94363

/-- The parabola defined by the equation y^2 + 4x = 0 -/
def parabola (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop := 
  ∃ (a b : ℝ), p = (a, b) ∧ 
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (x + a)^2

theorem parabola_focus :
  focus (-1, 0) parabola := by sorry

end parabola_focus_l943_94363


namespace oliver_battle_gremlins_count_l943_94356

/-- Oliver's card collection -/
structure CardCollection where
  monster_club : ℕ
  alien_baseball : ℕ
  battle_gremlins : ℕ

/-- Oliver's card collection satisfies the given conditions -/
def oliver_collection : CardCollection where
  monster_club := 32
  alien_baseball := 16
  battle_gremlins := 48

/-- Theorem: Oliver has 48 Battle Gremlins cards given the conditions -/
theorem oliver_battle_gremlins_count : 
  oliver_collection.battle_gremlins = 48 ∧
  oliver_collection.monster_club = 2 * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 3 * oliver_collection.alien_baseball :=
by sorry

end oliver_battle_gremlins_count_l943_94356


namespace total_rods_for_fence_l943_94304

/-- Represents the types of metal used in the fence. -/
inductive Metal
| A  -- Aluminum
| B  -- Bronze
| C  -- Copper

/-- Represents the components of a fence panel. -/
inductive Component
| Sheet
| Beam

/-- The number of rods needed for each type of metal and component. -/
def rods_needed (m : Metal) (c : Component) : ℕ :=
  match m, c with
  | Metal.A, Component.Sheet => 10
  | Metal.B, Component.Sheet => 8
  | Metal.C, Component.Sheet => 12
  | Metal.A, Component.Beam => 6
  | Metal.B, Component.Beam => 4
  | Metal.C, Component.Beam => 5

/-- Represents a fence pattern. -/
structure Pattern :=
  (a_sheets : ℕ)
  (b_sheets : ℕ)
  (c_sheets : ℕ)
  (a_beams : ℕ)
  (b_beams : ℕ)
  (c_beams : ℕ)

/-- The composition of Pattern X. -/
def pattern_x : Pattern :=
  { a_sheets := 2
  , b_sheets := 1
  , c_sheets := 0
  , a_beams := 0
  , b_beams := 0
  , c_beams := 2 }

/-- The composition of Pattern Y. -/
def pattern_y : Pattern :=
  { a_sheets := 0
  , b_sheets := 2
  , c_sheets := 1
  , a_beams := 3
  , b_beams := 1
  , c_beams := 0 }

/-- Calculate the total number of rods needed for a given pattern and number of panels. -/
def total_rods (p : Pattern) (panels : ℕ) : ℕ :=
  (p.a_sheets * rods_needed Metal.A Component.Sheet +
   p.b_sheets * rods_needed Metal.B Component.Sheet +
   p.c_sheets * rods_needed Metal.C Component.Sheet +
   p.a_beams * rods_needed Metal.A Component.Beam +
   p.b_beams * rods_needed Metal.B Component.Beam +
   p.c_beams * rods_needed Metal.C Component.Beam) * panels

/-- The main theorem stating that the total number of rods needed is 416. -/
theorem total_rods_for_fence : 
  total_rods pattern_x 7 + total_rods pattern_y 3 = 416 := by
  sorry


end total_rods_for_fence_l943_94304


namespace perpendicular_line_through_intersection_l943_94303

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := 2*x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 1)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 3*x - 2*y - 4 = 0

-- Theorem statement
theorem perpendicular_line_through_intersection :
  ∃ (x y : ℝ), 
    line1 x y ∧ 
    line2 x y ∧ 
    perpendicular_line x y ∧
    (∀ (m : ℝ), line3 x y → m = -2/3) :=
sorry

end perpendicular_line_through_intersection_l943_94303


namespace quadratic_completion_l943_94346

theorem quadratic_completion (c : ℝ) (n : ℝ) : 
  c < 0 → 
  (∀ x, x^2 + c*x + (1/4 : ℝ) = (x + n)^2 + (1/8 : ℝ)) → 
  c = -Real.sqrt 2 / 2 := by
sorry

end quadratic_completion_l943_94346


namespace product_xyz_equals_negative_one_l943_94316

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (h3 : z + 1/x = 2) : 
  x * y * z = -1 := by
sorry

end product_xyz_equals_negative_one_l943_94316


namespace reciprocal_of_negative_fraction_l943_94385

theorem reciprocal_of_negative_fraction (n : ℤ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n :=
by sorry

end reciprocal_of_negative_fraction_l943_94385


namespace a_gt_b_necessary_not_sufficient_l943_94338

/-- Curve C defined by the equation x²/a + y²/b = 1 -/
structure CurveC (a b : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / a + y^2 / b = 1

/-- Predicate for C being an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (a b : ℝ) : Prop :=
  ∃ (c : ℝ), a > b ∧ b > 0 ∧ c^2 = a^2 - b^2

/-- Main theorem: "a > b" is necessary but not sufficient for C to be an ellipse with foci on x-axis -/
theorem a_gt_b_necessary_not_sufficient (a b : ℝ) :
  (is_ellipse_x_foci a b → a > b) ∧
  ¬(a > b → is_ellipse_x_foci a b) :=
sorry

end a_gt_b_necessary_not_sufficient_l943_94338


namespace sallys_raise_l943_94391

/-- Given Sally's earnings last month and the total for two months, calculate her percentage raise. -/
theorem sallys_raise (last_month : ℝ) (total_two_months : ℝ) : 
  last_month = 1000 → total_two_months = 2100 → 
  (total_two_months - last_month) / last_month * 100 = 10 := by
  sorry

end sallys_raise_l943_94391


namespace consecutive_even_integers_sum_l943_94308

theorem consecutive_even_integers_sum (n : ℤ) : 
  (∃ k : ℤ, n = k^2) →
  (n - 2) + (n + 2) = 162 →
  (n - 2) + n + (n + 2) = 243 := by
sorry

end consecutive_even_integers_sum_l943_94308


namespace triangle_angle_calculation_l943_94376

theorem triangle_angle_calculation (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  A = 60 →           -- Angle A is 60°
  C = 2 * B →        -- Angle C is twice Angle B
  C = 80 :=          -- Conclusion: Angle C is 80°
by
  sorry

end triangle_angle_calculation_l943_94376


namespace complex_fraction_simplification_l943_94383

theorem complex_fraction_simplification :
  let z : ℂ := (5 - 3*I) / (2 - 3*I)
  z = -19/5 - 9/5*I :=
by sorry

end complex_fraction_simplification_l943_94383


namespace smallest_four_digit_mod_9_l943_94392

theorem smallest_four_digit_mod_9 : 
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 8 [MOD 9] → n ≥ 1007 :=
by sorry

end smallest_four_digit_mod_9_l943_94392


namespace conditional_probability_rain_given_east_wind_l943_94399

-- Define the probabilities
def prob_east_wind : ℚ := 3/10
def prob_rain : ℚ := 11/30
def prob_both : ℚ := 4/15

-- State the theorem
theorem conditional_probability_rain_given_east_wind :
  (prob_both / prob_east_wind : ℚ) = 8/9 := by
sorry

end conditional_probability_rain_given_east_wind_l943_94399


namespace f_inequality_A_is_solution_set_l943_94353

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem f_inequality (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  f (a * b) > f a - f b := by
  sorry

-- Prove that A is indeed the solution set to f(x) < 3 - |2x + 1|
theorem A_is_solution_set (x : ℝ) :
  x ∈ A ↔ f x < 3 - |2 * x + 1| := by
  sorry

end f_inequality_A_is_solution_set_l943_94353


namespace unique_pair_l943_94337

/-- A function that returns the last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A theorem stating that the only pair of positive integers (a, b) satisfying
    all the given conditions is (9, 4) -/
theorem unique_pair : ∀ a b : ℕ+, 
  (lastDigit (a.val + b.val) = 3) →
  (∃ p : ℕ, Nat.Prime p ∧ a.val - b.val = p) →
  isPerfectSquare (a.val * b.val) →
  (a.val = 9 ∧ b.val = 4) ∨ (a.val = 4 ∧ b.val = 9) := by
  sorry

#check unique_pair

end unique_pair_l943_94337


namespace restaurant_pies_theorem_l943_94358

/-- The number of pies sold in a week by a restaurant that sells 8 pies per day -/
def pies_sold_in_week (pies_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  pies_per_day * days_in_week

/-- Proof that a restaurant selling 8 pies per day for a week sells 56 pies in total -/
theorem restaurant_pies_theorem :
  pies_sold_in_week 8 7 = 56 := by
  sorry

end restaurant_pies_theorem_l943_94358


namespace sufficient_not_necessary_l943_94345

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x > 0, x^2 + 1/x^2 ≥ 2) ∧
  (∃ x ≤ 0, x ≠ 0 ∧ x^2 + 1/x^2 ≥ 2) :=
by sorry

end sufficient_not_necessary_l943_94345


namespace root_in_interval_l943_94393

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 5 6 ∧ log10 x = x - 5 :=
sorry

end root_in_interval_l943_94393


namespace max_zombies_after_four_days_l943_94352

/-- The maximum number of zombies in a mall after 4 days of doubling, given initial constraints -/
theorem max_zombies_after_four_days (initial_zombies : ℕ) : 
  initial_zombies < 50 → 
  (initial_zombies * 2^4 : ℕ) ≤ 48 :=
by sorry

end max_zombies_after_four_days_l943_94352


namespace cube_volume_scaling_l943_94336

theorem cube_volume_scaling (v : ℝ) (s : ℝ) :
  v > 0 →
  s > 0 →
  let original_side := v ^ (1/3)
  let scaled_side := s * original_side
  let scaled_volume := scaled_side ^ 3
  v = 64 ∧ s = 2 → scaled_volume = 512 :=
by
  sorry

end cube_volume_scaling_l943_94336


namespace tangent_line_and_minimum_value_and_a_range_l943_94321

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + (1 - x) * Real.exp x

noncomputable def g (a x : ℝ) : ℝ := x - (1 + a) * Real.log x - a / x

theorem tangent_line_and_minimum_value_and_a_range 
  (a : ℝ) 
  (h_a : a < 1) :
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 0, ∃ x₂ ∈ Set.Icc (Real.exp 1) 3, f x₁ > g a x₂) →
  (Real.exp 2 - 2 * Real.exp 1) / (Real.exp 1 + 1) < a ∧ a < 1 :=
by sorry

end tangent_line_and_minimum_value_and_a_range_l943_94321


namespace rihanna_remaining_money_l943_94306

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℚ) 
  (mango_price : ℚ) (mango_count : ℕ)
  (juice_price : ℚ) (juice_count : ℕ)
  (chips_price : ℚ) (chips_count : ℕ)
  (chocolate_price : ℚ) (chocolate_count : ℕ) : ℚ :=
  initial_amount - 
  (mango_price * mango_count + 
   juice_price * juice_count + 
   chips_price * chips_count + 
   chocolate_price * chocolate_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money : 
  remaining_money 50 3 6 3.5 4 2.25 2 1.75 3 = 8.25 := by
  sorry

end rihanna_remaining_money_l943_94306


namespace pencil_and_pen_cost_l943_94300

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: four pencils and three pens cost $3.70 -/
axiom condition1 : 4 * pencil_cost + 3 * pen_cost = 3.70

/-- The second condition: three pencils and four pens cost $4.20 -/
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = 4.20

/-- Theorem: The cost of one pencil and one pen is $1.1286 -/
theorem pencil_and_pen_cost : pencil_cost + pen_cost = 1.1286 := by
  sorry

end pencil_and_pen_cost_l943_94300


namespace inequality_proof_l943_94332

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end inequality_proof_l943_94332


namespace repeating_decimal_equals_fraction_l943_94347

/-- Represents a repeating decimal with a non-repeating part and a repeating part -/
structure RepeatingDecimal where
  nonRepeating : ℚ
  repeating : ℚ
  repeatingDigits : ℕ

/-- Converts a RepeatingDecimal to its fraction representation -/
def repeatingDecimalToFraction (x : RepeatingDecimal) : ℚ :=
  x.nonRepeating + x.repeating / (1 - (1 / 10 ^ x.repeatingDigits))

theorem repeating_decimal_equals_fraction :
  let x : RepeatingDecimal := {
    nonRepeating := 1/2,
    repeating := 23/1000,
    repeatingDigits := 3
  }
  repeatingDecimalToFraction x = 1045/1998 := by sorry

end repeating_decimal_equals_fraction_l943_94347


namespace cubic_sum_divisible_by_nine_l943_94323

theorem cubic_sum_divisible_by_nine (n : ℕ) :
  9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := by
  sorry

end cubic_sum_divisible_by_nine_l943_94323


namespace quadratic_rewrite_l943_94373

theorem quadratic_rewrite (b : ℕ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 68 = (x + n)^2 + 32) →
  b % 2 = 0 →
  b > 0 →
  b = 12 := by
sorry

end quadratic_rewrite_l943_94373


namespace square_diff_sum_eq_three_l943_94322

theorem square_diff_sum_eq_three (a b c : ℤ) 
  (ha : a = 2011) (hb : b = 2012) (hc : c = 2013) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := by
  sorry

end square_diff_sum_eq_three_l943_94322


namespace a_upper_bound_l943_94318

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0}

-- State the theorem
theorem a_upper_bound (a : ℝ) (h : A ⊆ B a) : a ≤ -2 := by
  sorry

end a_upper_bound_l943_94318


namespace four_solutions_to_g_composition_l943_94334

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem four_solutions_to_g_composition :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 5) ∧ s.card = 4 :=
sorry

end four_solutions_to_g_composition_l943_94334


namespace sum_of_coefficients_l943_94394

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^6 - 3 * x + 9) + 5 * (x^7 - 3 * x^3 + 2 * x^2 - 4)

theorem sum_of_coefficients : polynomial 1 = 32 := by
  sorry

end sum_of_coefficients_l943_94394


namespace sum_of_cubes_l943_94382

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 := by
  sorry

end sum_of_cubes_l943_94382


namespace negation_of_existential_proposition_l943_94302

theorem negation_of_existential_proposition :
  (¬ ∃ x₀ : ℝ, x₀ < 0 ∧ Real.exp x₀ - x₀ > 1) ↔ (∀ x : ℝ, x < 0 → Real.exp x - x ≤ 1) := by
  sorry

end negation_of_existential_proposition_l943_94302


namespace car_rental_cost_l943_94367

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost of renting a car for 5 days at $30 per day and driving 500 miles at $0.25 per mile is $275. -/
theorem car_rental_cost : rental_cost 30 (1/4) 5 500 = 275 := by
  sorry

end car_rental_cost_l943_94367


namespace sum_of_coefficients_in_factorization_l943_94384

theorem sum_of_coefficients_in_factorization (x y : ℝ) : 
  ∃ (a b c d e f : ℤ), 
    (8 * x^8 - 243 * y^8 = (a * x^2 + b * y^2) * (c * x^2 + d * y^2) * (e * x^4 + f * y^4)) ∧
    (a + b + c + d + e + f = 17) := by
  sorry

end sum_of_coefficients_in_factorization_l943_94384


namespace purple_marbles_fraction_l943_94355

theorem purple_marbles_fraction (total : ℚ) (h1 : total > 0) : 
  let yellow := (4/7) * total
  let green := (2/7) * total
  let initial_purple := total - yellow - green
  let new_purple := 3 * initial_purple
  let new_total := yellow + green + new_purple
  new_purple / new_total = 1/3 := by
  sorry

end purple_marbles_fraction_l943_94355


namespace school_population_theorem_l943_94319

theorem school_population_theorem (b g t s : ℕ) :
  b = 4 * g ∧ g = 8 * t ∧ t = 2 * s →
  b + g + t + s = (83 * g) / 16 := by
  sorry

end school_population_theorem_l943_94319


namespace abs_x_lt_2_sufficient_not_necessary_l943_94313

theorem abs_x_lt_2_sufficient_not_necessary :
  (∃ x : ℝ, (abs x < 2 → x^2 - x - 6 < 0) ∧ 
            ¬(x^2 - x - 6 < 0 → abs x < 2)) :=
sorry

end abs_x_lt_2_sufficient_not_necessary_l943_94313


namespace animals_in_field_l943_94386

/-- The number of animals running through a field -/
def total_animals (dog : ℕ) (cats : ℕ) (rabbits_per_cat : ℕ) (hares_per_rabbit : ℕ) : ℕ :=
  dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit)

/-- Theorem stating the total number of animals in the field -/
theorem animals_in_field : total_animals 1 4 2 3 = 37 := by
  sorry

end animals_in_field_l943_94386


namespace other_number_proof_l943_94389

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 33)
  (h3 : a = 231) : 
  b = 660 := by
  sorry

end other_number_proof_l943_94389


namespace trivia_team_groups_l943_94377

theorem trivia_team_groups 
  (total_students : ℕ) 
  (not_picked : ℕ) 
  (students_per_group : ℕ) 
  (h1 : total_students = 36) 
  (h2 : not_picked = 9) 
  (h3 : students_per_group = 9) : 
  (total_students - not_picked) / students_per_group = 3 := by
  sorry

end trivia_team_groups_l943_94377


namespace solution_characterization_l943_94395

def satisfies_equation (a b c d : ℝ) : Prop :=
  a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b)

def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(k, 0, 0, 0) | k : ℝ} ∪
  {(0, k, 0, 0) | k : ℝ} ∪
  {(0, 0, k, 0) | k : ℝ} ∪
  {(0, 0, 0, k) | k : ℝ} ∪
  {(k, k, k, k) | k : ℝ} ∪
  {(k, -k, k, -k) | k : ℝ} ∪
  {(k, k*(-1 + Real.sqrt 2), -k, k*(1 - Real.sqrt 2)) | k : ℝ} ∪
  {(k, k*(-1 - Real.sqrt 2), -k, k*(1 + Real.sqrt 2)) | k : ℝ}

theorem solution_characterization :
  ∀ (a b c d : ℝ), satisfies_equation a b c d ↔ (a, b, c, d) ∈ solution_set :=
sorry

end solution_characterization_l943_94395


namespace movie_theater_shows_24_movies_l943_94361

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

/-- Theorem: A movie theater with 6 screens, open for 8 hours, where each movie lasts 2 hours,
    shows 24 movies throughout the day. -/
theorem movie_theater_shows_24_movies :
  total_movies 6 8 2 = 24 := by
  sorry

end movie_theater_shows_24_movies_l943_94361


namespace game_price_calculation_l943_94309

theorem game_price_calculation (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) : 
  total_games = 15 → non_working_games = 6 → total_earnings = 63 →
  total_earnings / (total_games - non_working_games) = 7 := by
sorry

end game_price_calculation_l943_94309


namespace qr_equals_b_l943_94368

-- Define the curve
def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem qr_equals_b (a b c : ℝ) (P Q R : Point) : 
  curve c P.x P.y →  -- P is on the curve
  curve c Q.x Q.y →  -- Q is on the curve
  P = Point.mk a b →  -- P has coordinates (a, b)
  Q = Point.mk 0 c →  -- Q has coordinates (0, c)
  R.y = 0 →  -- R is on the x-axis
  (∃ k : ℝ, R.x = k * Real.sinh (a / c)) →  -- R.x is proportional to sinh(a/c)
  (Q.y - R.y) / (Q.x - R.x) = -1 / Real.sinh (a / c) →  -- QR is parallel to normal at P
  Real.sqrt ((R.x - Q.x)^2 + (R.y - Q.y)^2) = b  -- Distance QR equals b
  := by sorry

end qr_equals_b_l943_94368


namespace wire_shapes_area_difference_l943_94365

theorem wire_shapes_area_difference :
  let wire_length : ℝ := 52
  let square_side : ℝ := wire_length / 4
  let rect_width : ℝ := 15
  let rect_length : ℝ := (wire_length / 2) - rect_width
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_width * rect_length
  square_area - rect_area = 4 := by
  sorry

end wire_shapes_area_difference_l943_94365
