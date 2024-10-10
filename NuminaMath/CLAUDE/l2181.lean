import Mathlib

namespace fib_1960_1988_gcd_l2181_218149

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The greatest common divisor of the 1960th and 1988th Fibonacci numbers is 317811 -/
theorem fib_1960_1988_gcd : Nat.gcd (fib 1988) (fib 1960) = 317811 := by
  sorry

end fib_1960_1988_gcd_l2181_218149


namespace probability_of_white_ball_l2181_218183

-- Define the number of white and black balls
def num_white_balls : ℕ := 1
def num_black_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_white_balls + num_black_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 1 / 3 := by sorry

end probability_of_white_ball_l2181_218183


namespace complex_fraction_simplification_l2181_218172

theorem complex_fraction_simplification :
  ((-4 : ℂ) - 6*I) / (5 - 2*I) = -32/21 - 38/21*I :=
by sorry

end complex_fraction_simplification_l2181_218172


namespace gcd_factorial_problem_l2181_218123

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end gcd_factorial_problem_l2181_218123


namespace reflection_over_x_axis_of_P_l2181_218186

/-- Reflects a point over the x-axis -/
def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def P : ℝ × ℝ := (2, -3)

theorem reflection_over_x_axis_of_P :
  reflect_over_x_axis P = (2, 3) := by sorry

end reflection_over_x_axis_of_P_l2181_218186


namespace product_of_roots_l2181_218161

theorem product_of_roots (x : ℝ) : 
  let a : ℝ := 24
  let b : ℝ := 36
  let c : ℝ := -648
  let equation := a * x^2 + b * x + c
  let root_product := c / a
  equation = 0 → root_product = -27 :=
by
  sorry

end product_of_roots_l2181_218161


namespace circle_elimination_count_l2181_218147

/-- Calculates the total number of counts in a circle elimination game. -/
def totalCounts (initialPeople : ℕ) : ℕ :=
  let rec countRounds (remaining : ℕ) (acc : ℕ) : ℕ :=
    if remaining ≤ 2 then acc
    else
      let eliminated := remaining / 3
      let newRemaining := remaining - eliminated
      countRounds newRemaining (acc + remaining)
  countRounds initialPeople 0

/-- Theorem stating that for 21 initial people, the total count is 64. -/
theorem circle_elimination_count :
  totalCounts 21 = 64 := by
  sorry

end circle_elimination_count_l2181_218147


namespace playground_length_l2181_218110

theorem playground_length (garden_width garden_perimeter playground_width : ℝ) 
  (hw : garden_width = 24)
  (hp : garden_perimeter = 64)
  (pw : playground_width = 12)
  (area_eq : garden_width * ((garden_perimeter / 2) - garden_width) = playground_width * (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width)) :
  (garden_width * ((garden_perimeter / 2) - garden_width) / playground_width) = 16 := by
sorry

end playground_length_l2181_218110


namespace initial_cargo_calculation_l2181_218136

theorem initial_cargo_calculation (cargo_loaded : ℕ) (total_cargo : ℕ) 
  (h1 : cargo_loaded = 8723)
  (h2 : total_cargo = 14696) :
  total_cargo - cargo_loaded = 5973 := by
  sorry

end initial_cargo_calculation_l2181_218136


namespace school_trip_students_l2181_218152

/-- The number of students in a school given the number of buses and seats per bus -/
def number_of_students (buses : ℕ) (seats_per_bus : ℕ) : ℕ :=
  buses * seats_per_bus

/-- Theorem stating that the number of students in the school is 111 -/
theorem school_trip_students :
  let buses : ℕ := 37
  let seats_per_bus : ℕ := 3
  number_of_students buses seats_per_bus = 111 := by
  sorry

#eval number_of_students 37 3

end school_trip_students_l2181_218152


namespace system_solution_l2181_218144

-- Define the system of equations
def system_equations (a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
  (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
  (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
  (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1)

-- Theorem statement
theorem system_solution (a₁ a₂ a₃ a₄ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), system_equations a₁ a₂ a₃ a₄ x₁ x₂ x₃ x₄ ∧ 
    x₁ = 1 / |a₁ - a₄| ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / |a₁ - a₄| :=
by sorry

end system_solution_l2181_218144


namespace katie_pastries_left_l2181_218101

/-- Represents the number of pastries Katie had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Proves that Katie had 8 pastries left after the bake sale -/
theorem katie_pastries_left : pastries_left 7 5 4 = 8 := by
  sorry

end katie_pastries_left_l2181_218101


namespace equal_sum_sequence_a8_l2181_218177

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. --/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

/-- The common sum of an equal sum sequence. --/
def CommonSum (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n : ℕ, a n + a (n + 1) = k

theorem equal_sum_sequence_a8 (a : ℕ → ℝ) (h1 : EqualSumSequence a) (h2 : a 1 = 2) (h3 : CommonSum a 5) :
  a 8 = 3 := by
  sorry

end equal_sum_sequence_a8_l2181_218177


namespace negative_three_to_zero_power_l2181_218194

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by
  sorry

end negative_three_to_zero_power_l2181_218194


namespace square_product_inequality_l2181_218104

theorem square_product_inequality (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end square_product_inequality_l2181_218104


namespace yogurt_combinations_l2181_218132

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end yogurt_combinations_l2181_218132


namespace sufficient_not_necessary_condition_l2181_218179

theorem sufficient_not_necessary_condition (A B : Set α) 
  (h1 : A ∩ B = A) (h2 : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by sorry

end sufficient_not_necessary_condition_l2181_218179


namespace larger_number_ratio_l2181_218113

theorem larger_number_ratio (a b : ℕ+) (k : ℚ) (s : ℤ) 
  (h1 : (a : ℚ) / (b : ℚ) = k)
  (h2 : k < 1)
  (h3 : (a : ℤ) + (b : ℤ) = s) :
  max a b = |s| / (1 + k) :=
sorry

end larger_number_ratio_l2181_218113


namespace subtraction_of_integers_l2181_218127

theorem subtraction_of_integers : -1 - 3 = -4 := by
  sorry

end subtraction_of_integers_l2181_218127


namespace square_side_length_l2181_218197

/-- Given six identical squares arranged to form a larger rectangle ABCD with an area of 3456,
    the side length of each square is 24. -/
theorem square_side_length (s : ℝ) : s > 0 → s * s * 6 = 3456 → s = 24 := by
  sorry

end square_side_length_l2181_218197


namespace max_xy_value_l2181_218174

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 16 ∧ x₀ * y₀ = 32 :=
by sorry

end max_xy_value_l2181_218174


namespace muffin_cost_savings_l2181_218184

/-- Represents the cost savings when choosing raspberries over blueberries for muffins -/
def cost_savings (num_batches : ℕ) (ounces_per_batch : ℕ) 
  (blueberry_price : ℚ) (blueberry_ounces : ℕ) 
  (raspberry_price : ℚ) (raspberry_ounces : ℕ) : ℚ :=
  let total_ounces := num_batches * ounces_per_batch
  let blueberry_cartons := (total_ounces + blueberry_ounces - 1) / blueberry_ounces
  let raspberry_cartons := (total_ounces + raspberry_ounces - 1) / raspberry_ounces
  blueberry_cartons * blueberry_price - raspberry_cartons * raspberry_price

/-- The cost savings when choosing raspberries over blueberries for 4 batches of muffins -/
theorem muffin_cost_savings : 
  cost_savings 4 12 (5 / 1) 6 (3 / 1) 8 = 22 := by
  sorry

end muffin_cost_savings_l2181_218184


namespace intersection_S_T_l2181_218176

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0 ∧ x ≠ 6}
def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by sorry

end intersection_S_T_l2181_218176


namespace coconut_jelly_beans_count_l2181_218122

def total_jelly_beans : ℕ := 4000
def red_fraction : ℚ := 3/4
def coconut_fraction : ℚ := 1/4

theorem coconut_jelly_beans_count : 
  (red_fraction * total_jelly_beans : ℚ) * coconut_fraction = 750 := by
  sorry

end coconut_jelly_beans_count_l2181_218122


namespace article_cost_price_l2181_218178

theorem article_cost_price (marked_price : ℝ) (cost_price : ℝ) : 
  marked_price = 112.5 →
  0.95 * marked_price = 1.25 * cost_price →
  cost_price = 85.5 := by
sorry

end article_cost_price_l2181_218178


namespace solution_implies_k_value_l2181_218198

theorem solution_implies_k_value (k : ℝ) : 
  (k * (-3 + 4) - 2 * k - (-3) = 5) → k = -2 := by
  sorry

end solution_implies_k_value_l2181_218198


namespace quadratic_inequality_proof_l2181_218153

theorem quadratic_inequality_proof (x : ℝ) : 
  x^2 + 6*x + 8 ≥ -(x + 4)*(x + 6) ∧ 
  (x^2 + 6*x + 8 = -(x + 4)*(x + 6) ↔ x = -4) := by
sorry

end quadratic_inequality_proof_l2181_218153


namespace total_snake_owners_l2181_218167

/- Define the total number of pet owners -/
def total_pet_owners : ℕ := 120

/- Define the number of people owning specific combinations of pets -/
def only_dogs : ℕ := 25
def only_cats : ℕ := 18
def only_birds : ℕ := 12
def only_snakes : ℕ := 15
def only_hamsters : ℕ := 7
def cats_and_dogs : ℕ := 8
def dogs_and_birds : ℕ := 5
def cats_and_birds : ℕ := 6
def cats_and_snakes : ℕ := 7
def dogs_and_snakes : ℕ := 10
def dogs_and_hamsters : ℕ := 4
def cats_and_hamsters : ℕ := 3
def birds_and_hamsters : ℕ := 5
def birds_and_snakes : ℕ := 2
def snakes_and_hamsters : ℕ := 3
def cats_dogs_birds : ℕ := 3
def cats_dogs_snakes : ℕ := 4
def cats_snakes_hamsters : ℕ := 2
def all_pets : ℕ := 1

/- Theorem stating the total number of snake owners -/
theorem total_snake_owners : 
  only_snakes + cats_and_snakes + dogs_and_snakes + birds_and_snakes + 
  snakes_and_hamsters + cats_dogs_snakes + cats_snakes_hamsters + all_pets = 44 :=
by
  sorry

end total_snake_owners_l2181_218167


namespace initial_discount_percentage_l2181_218190

/-- Given a dress with original price d and an initial discount percentage x,
    prove that x = 65 when a staff member pays 0.14d after an additional 60% discount. -/
theorem initial_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  0.40 * (1 - x / 100) * d = 0.14 * d → x = 65 := by
  sorry

end initial_discount_percentage_l2181_218190


namespace special_function_value_at_one_l2181_218107

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y, and f(2) = 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ f 2 = 4

/-- Theorem: If f is a special function, then f(1) = 2 -/
theorem special_function_value_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end special_function_value_at_one_l2181_218107


namespace derivative_of_f_l2181_218195

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 10) / x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = (1 - Real.log 10 * (Real.log x / Real.log 10)) / (x^2 * Real.log 10) :=
by sorry

end derivative_of_f_l2181_218195


namespace alcohol_water_mixture_ratio_l2181_218148

theorem alcohol_water_mixture_ratio 
  (p q r : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hr : r > 0) :
  let jar1_ratio := p / (p + 1)
  let jar2_ratio := q / (q + 1)
  let jar3_ratio := r / (r + 1)
  let total_alcohol := jar1_ratio + jar2_ratio + jar3_ratio
  let total_water := 1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1)
  total_alcohol / total_water = (p*q*r + p*q + p*r + q*r + p + q + r) / (p*q + p*r + q*r + p + q + r + 1) :=
by sorry

end alcohol_water_mixture_ratio_l2181_218148


namespace floral_arrangement_daisies_percentage_l2181_218192

theorem floral_arrangement_daisies_percentage
  (total : ℝ)
  (yellow_flowers : ℝ)
  (blue_flowers : ℝ)
  (yellow_tulips : ℝ)
  (blue_tulips : ℝ)
  (yellow_daisies : ℝ)
  (blue_daisies : ℝ)
  (h1 : yellow_flowers = 7 / 10 * total)
  (h2 : blue_flowers = 3 / 10 * total)
  (h3 : yellow_tulips = 1 / 2 * yellow_flowers)
  (h4 : blue_daisies = 1 / 3 * blue_flowers)
  (h5 : yellow_flowers + blue_flowers = total)
  (h6 : yellow_tulips + blue_tulips + yellow_daisies + blue_daisies = total)
  : (yellow_daisies + blue_daisies) / total = 9 / 20 :=
by sorry

end floral_arrangement_daisies_percentage_l2181_218192


namespace problem_solution_l2181_218196

theorem problem_solution (a b : ℝ) (h : a^2 + |b+1| = 0) : (a+b)^2015 = -1 := by
  sorry

end problem_solution_l2181_218196


namespace complement_of_union_l2181_218133

def I : Finset Int := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 0, 2}

theorem complement_of_union :
  (I \ (A ∪ B)) = {4, 5} := by sorry

end complement_of_union_l2181_218133


namespace rachel_envelope_stuffing_l2181_218137

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_time : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour : ℕ) 
  (second_hour : ℕ) 
  (h1 : total_time = 8)
  (h2 : total_envelopes = 1500)
  (h3 : first_hour = 135)
  (h4 : second_hour = 141) : 
  (total_envelopes - first_hour - second_hour) / (total_time - 2) = 204 := by
sorry

end rachel_envelope_stuffing_l2181_218137


namespace corn_acreage_l2181_218125

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end corn_acreage_l2181_218125


namespace distance_z1z2_to_origin_l2181_218146

open Complex

theorem distance_z1z2_to_origin : 
  let z₁ : ℂ := I
  let z₂ : ℂ := 1 + I
  let z : ℂ := z₁ * z₂
  abs z = Real.sqrt 2 := by sorry

end distance_z1z2_to_origin_l2181_218146


namespace complex_power_modulus_l2181_218142

theorem complex_power_modulus : Complex.abs ((2 : ℂ) + Complex.I * Real.sqrt 11) ^ 4 = 225 := by
  sorry

end complex_power_modulus_l2181_218142


namespace prob_A_truth_l2181_218119

-- Define the probabilities
def prob_B_truth : ℝ := 0.60
def prob_both_truth : ℝ := 0.45

-- Theorem statement
theorem prob_A_truth :
  ∃ (prob_A : ℝ),
    prob_A * prob_B_truth = prob_both_truth ∧
    prob_A = 0.75 :=
by sorry

end prob_A_truth_l2181_218119


namespace burger_sharing_l2181_218171

theorem burger_sharing (burger_length : ℝ) (brother_fraction : ℝ) (friend1_fraction : ℝ) (friend2_fraction : ℝ) :
  burger_length = 12 →
  brother_fraction = 1/3 →
  friend1_fraction = 1/4 →
  friend2_fraction = 1/2 →
  ∃ (brother_share friend1_share friend2_share valentina_share : ℝ),
    brother_share = burger_length * brother_fraction ∧
    friend1_share = (burger_length - brother_share) * friend1_fraction ∧
    friend2_share = (burger_length - brother_share - friend1_share) * friend2_fraction ∧
    valentina_share = burger_length - brother_share - friend1_share - friend2_share ∧
    brother_share = 4 ∧
    friend1_share = 2 ∧
    friend2_share = 3 ∧
    valentina_share = 3 :=
by
  sorry

end burger_sharing_l2181_218171


namespace stating_kevin_vanessa_age_multiple_l2181_218112

/-- Represents the age difference between Kevin and Vanessa -/
def age_difference : ℕ := 14

/-- Represents Kevin's initial age -/
def kevin_initial_age : ℕ := 16

/-- Represents Vanessa's initial age -/
def vanessa_initial_age : ℕ := 2

/-- 
Theorem stating that the first time Kevin's age becomes a multiple of Vanessa's age, 
Kevin will be 4.5 times older than Vanessa.
-/
theorem kevin_vanessa_age_multiple :
  ∃ (years : ℕ), 
    (kevin_initial_age + years) % (vanessa_initial_age + years) = 0 ∧
    (kevin_initial_age + years : ℚ) / (vanessa_initial_age + years : ℚ) = 4.5 ∧
    ∀ (y : ℕ), y < years → (kevin_initial_age + y) % (vanessa_initial_age + y) ≠ 0 :=
sorry

end stating_kevin_vanessa_age_multiple_l2181_218112


namespace min_value_expression_min_value_achieved_l2181_218168

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 :=
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 6 * Real.sqrt (a * b) + 3 / a + 3 / b < 12 + ε :=
sorry

end min_value_expression_min_value_achieved_l2181_218168


namespace probability_different_tens_digits_l2181_218158

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_integers : ℕ := 70

/-- The number of different tens digits in the range 10 to 79. -/
def different_tens_digits : ℕ := 7

/-- The number of integers to be chosen. -/
def chosen_integers : ℕ := 6

/-- The number of integers for each tens digit. -/
def integers_per_tens : ℕ := 10

theorem probability_different_tens_digits :
  (different_tens_digits.choose chosen_integers * integers_per_tens ^ chosen_integers : ℚ) /
  (total_integers.choose chosen_integers) = 1750 / 2980131 := by sorry

end probability_different_tens_digits_l2181_218158


namespace fraction_nonnegative_iff_l2181_218185

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 3) / (x^2 + 5*x + 11) ≥ 0 ↔ x ≥ 3 := by sorry

end fraction_nonnegative_iff_l2181_218185


namespace fraction_without_finite_decimal_l2181_218170

def has_finite_decimal_expansion (n d : ℕ) : Prop :=
  ∃ k : ℕ, d * (10 ^ k) % n = 0

theorem fraction_without_finite_decimal : 
  has_finite_decimal_expansion 9 10 ∧ 
  has_finite_decimal_expansion 3 5 ∧ 
  ¬ has_finite_decimal_expansion 3 7 ∧ 
  has_finite_decimal_expansion 7 8 :=
sorry

end fraction_without_finite_decimal_l2181_218170


namespace largest_n_for_factorization_l2181_218115

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ (m : ℤ), (∃ (a b c d : ℤ), 3 * X^2 + m * X + 108 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 108 = (a * X + b) * (c * X + d)) ∧
  n = 325 :=
sorry

end largest_n_for_factorization_l2181_218115


namespace average_decrease_l2181_218138

def initial_observations : ℕ := 6
def initial_average : ℚ := 12
def new_observation : ℕ := 5

theorem average_decrease :
  let initial_sum := initial_observations * initial_average
  let new_sum := initial_sum + new_observation
  let new_average := new_sum / (initial_observations + 1)
  initial_average - new_average = 1 := by
  sorry

end average_decrease_l2181_218138


namespace min_value_given_condition_l2181_218126

theorem min_value_given_condition (a b : ℝ) : 
  (|a - 2| + (b + 3)^2 = 0) → 
  (min (min (min (a + b) (a - b)) (b^a)) (a * b) = a * b) :=
by sorry

end min_value_given_condition_l2181_218126


namespace complex_function_from_real_part_l2181_218166

open Complex

/-- Given that u(x, y) = x^2 - y^2 + 2x is the real part of a differentiable complex function f(z),
    prove that f(z) = z^2 + 2z + c for some constant c. -/
theorem complex_function_from_real_part
  (f : ℂ → ℂ)
  (h_diff : Differentiable ℂ f)
  (h_real_part : ∀ z : ℂ, (f z).re = z.re^2 - z.im^2 + 2*z.re) :
  ∃ c : ℂ, ∀ z : ℂ, f z = z^2 + 2*z + c :=
sorry

end complex_function_from_real_part_l2181_218166


namespace difference_of_squares_l2181_218143

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 6) : 
  x^2 - y^2 = 120 := by
sorry

end difference_of_squares_l2181_218143


namespace monogram_count_l2181_218130

/-- The number of letters in the alphabet before 'M' -/
def letters_before_m : Nat := 12

/-- The number of letters in the alphabet after 'M' -/
def letters_after_m : Nat := 13

/-- A monogram is valid if it satisfies the given conditions -/
def is_valid_monogram (f m l : Char) : Prop :=
  f < m ∧ m < l ∧ f ≠ m ∧ m ≠ l ∧ f ≠ l ∧ m = 'M'

/-- The total number of valid monograms -/
def total_valid_monograms : Nat := letters_before_m * letters_after_m

theorem monogram_count :
  total_valid_monograms = 156 :=
sorry

end monogram_count_l2181_218130


namespace tangent_circles_concyclic_points_l2181_218193

/-- Four circles are tangent consecutively if each circle is tangent to the next one in the sequence. -/
def ConsecutivelyTangentCircles (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are the tangent points of consecutively tangent circles if they are the points where each pair of consecutive circles touch. -/
def TangentPoints (A B C D : ℝ × ℝ) (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) : Prop := sorry

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : ℝ × ℝ) : Prop := sorry

/-- Theorem: If four circles are tangent to each other consecutively at four points, then these four points are concyclic. -/
theorem tangent_circles_concyclic_points
  (Γ₁ Γ₂ Γ₃ Γ₄ : Set ℝ × ℝ) (A B C D : ℝ × ℝ) :
  ConsecutivelyTangentCircles Γ₁ Γ₂ Γ₃ Γ₄ →
  TangentPoints A B C D Γ₁ Γ₂ Γ₃ Γ₄ →
  Concyclic A B C D :=
by sorry

end tangent_circles_concyclic_points_l2181_218193


namespace trigonometric_equalities_l2181_218175

theorem trigonometric_equalities
  (α β γ a b c : ℝ)
  (h_alpha : 0 < α ∧ α < π)
  (h_beta : 0 < β ∧ β < π)
  (h_gamma : 0 < γ ∧ γ < π)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_c : c > 0)
  (h_b_eq : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (h_a_eq : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2)
  (h_identity : 1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0) :
  (Real.cos α + Real.cos β * Real.cos γ = Real.sin α * Real.sin β) ∧
  (Real.cos β + Real.cos α * Real.cos γ = Real.sin α * Real.sin γ) ∧
  (Real.cos γ + Real.cos α * Real.cos β = Real.sin β * Real.sin γ) ∧
  (a * Real.sin γ = c * Real.sin α) ∧
  (b * Real.sin γ = c * Real.sin β) ∧
  (c * Real.sin α = a * Real.sin γ) := by
  sorry

end trigonometric_equalities_l2181_218175


namespace range_of_independent_variable_l2181_218124

theorem range_of_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - 3 * x)) ↔ x < 2 / 3 := by
sorry

end range_of_independent_variable_l2181_218124


namespace classroom_key_probability_is_two_sevenths_l2181_218131

/-- The probability of selecting a key that opens the classroom door -/
def classroom_key_probability (total_keys : ℕ) (classroom_keys : ℕ) : ℚ :=
  classroom_keys / total_keys

/-- Theorem: The probability of randomly selecting a key that can open the classroom door lock is 2/7 -/
theorem classroom_key_probability_is_two_sevenths :
  classroom_key_probability 7 2 = 2 / 7 := by
  sorry

end classroom_key_probability_is_two_sevenths_l2181_218131


namespace expression_equality_l2181_218116

theorem expression_equality : 
  Real.sqrt (4/3) * Real.sqrt 15 + ((-8) ^ (1/3 : ℝ)) + (π - 3) ^ (0 : ℝ) = 2 * Real.sqrt 5 - 1 :=
by sorry

end expression_equality_l2181_218116


namespace semicircle_problem_l2181_218150

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 18 → N = 19 := by
  sorry

end semicircle_problem_l2181_218150


namespace john_roommates_l2181_218128

theorem john_roommates (bob_roommates : ℕ) (h1 : bob_roommates = 10) :
  let john_roommates := 2 * bob_roommates + 5
  john_roommates = 25 := by sorry

end john_roommates_l2181_218128


namespace square_division_perimeter_l2181_218105

theorem square_division_perimeter 
  (original_perimeter : ℝ) 
  (h_original_perimeter : original_perimeter = 200) : 
  ∃ (smaller_square_perimeter : ℝ), 
    smaller_square_perimeter = 100 ∧
    ∃ (original_side : ℝ), 
      4 * original_side = original_perimeter ∧
      ∃ (rectangle_width rectangle_height : ℝ),
        rectangle_width = original_side ∧
        rectangle_height = original_side / 2 ∧
        smaller_square_perimeter = 4 * rectangle_height :=
by sorry

end square_division_perimeter_l2181_218105


namespace scout_hourly_rate_l2181_218169

/-- Represents Scout's weekend earnings --/
def weekend_earnings (hourly_rate : ℚ) : ℚ :=
  -- Saturday earnings
  (4 * hourly_rate + 5 * 5) +
  -- Sunday earnings
  (5 * hourly_rate + 8 * 5)

/-- Theorem stating that Scout's hourly rate is $10.00 --/
theorem scout_hourly_rate :
  ∃ (rate : ℚ), weekend_earnings rate = 155 ∧ rate = 10 := by
  sorry

end scout_hourly_rate_l2181_218169


namespace arithmetic_sequence_sum_l2181_218135

theorem arithmetic_sequence_sum (a₁ a_n d : ℚ) (n : ℕ) (h1 : a₁ = 2/7) (h2 : a_n = 20/7) (h3 : d = 2/7) (h4 : n = 10) :
  (n : ℚ) / 2 * (a₁ + a_n) = 110/7 := by
  sorry

end arithmetic_sequence_sum_l2181_218135


namespace M_union_N_eq_N_l2181_218118

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem M_union_N_eq_N : M ∪ N = N := by sorry

end M_union_N_eq_N_l2181_218118


namespace optimal_orange_purchase_l2181_218157

-- Define the pricing options
def price_option_1 : ℕ × ℕ := (4, 15)  -- 4 oranges for 15 cents
def price_option_2 : ℕ × ℕ := (7, 25)  -- 7 oranges for 25 cents

-- Define the number of oranges to purchase
def total_oranges : ℕ := 28

-- Theorem statement
theorem optimal_orange_purchase :
  ∃ (n m : ℕ),
    n * price_option_1.1 + m * price_option_2.1 = total_oranges ∧
    n * price_option_1.2 + m * price_option_2.2 = 100 ∧
    (n * price_option_1.2 + m * price_option_2.2) / total_oranges = 25 / 7 :=
sorry

end optimal_orange_purchase_l2181_218157


namespace complex_product_theorem_l2181_218120

theorem complex_product_theorem :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 - i
  let z₂ : ℂ := 2 + i
  z₁ * z₂ = 3 - i :=
by sorry

end complex_product_theorem_l2181_218120


namespace calculate_original_nes_price_l2181_218140

/-- Calculates the original price of an NES given trade-in values, discounts, and final payment -/
theorem calculate_original_nes_price
  (snes_value : ℝ)
  (snes_credit_rate : ℝ)
  (gameboy_value : ℝ)
  (gameboy_credit_rate : ℝ)
  (ps2_value : ℝ)
  (ps2_credit_rate : ℝ)
  (nes_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (payment : ℝ)
  (change : ℝ)
  (h1 : snes_value = 150)
  (h2 : snes_credit_rate = 0.8)
  (h3 : gameboy_value = 50)
  (h4 : gameboy_credit_rate = 0.75)
  (h5 : ps2_value = 100)
  (h6 : ps2_credit_rate = 0.6)
  (h7 : nes_discount_rate = 0.2)
  (h8 : sales_tax_rate = 0.08)
  (h9 : payment = 100)
  (h10 : change = 12) :
  ∃ (original_price : ℝ), abs (original_price - 101.85) < 0.01 :=
by sorry

end calculate_original_nes_price_l2181_218140


namespace cos_squared_fifteen_degrees_l2181_218154

theorem cos_squared_fifteen_degrees :
  2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by sorry

end cos_squared_fifteen_degrees_l2181_218154


namespace product_inequality_l2181_218199

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end product_inequality_l2181_218199


namespace factorization_proof_l2181_218108

theorem factorization_proof (x y : ℝ) : 2 * x^3 - 18 * x * y^2 = 2 * x * (x + 3 * y) * (x - 3 * y) := by
  sorry

end factorization_proof_l2181_218108


namespace rachels_budget_l2181_218181

/-- Rachel's budget for a beauty and modeling contest -/
theorem rachels_budget (sara_shoes : ℕ) (sara_dress : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → 2 * (sara_shoes + sara_dress) = 500 := by
  sorry

end rachels_budget_l2181_218181


namespace diagonal_intersections_count_l2181_218164

/-- Represents a convex polygon with n sides where no two diagonals are parallel
    and no three diagonals intersect at the same point. -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3
  no_parallel_diagonals : True
  no_triple_intersections : True

/-- The number of intersection points of diagonals outside a convex polygon. -/
def diagonal_intersections_outside (n : ℕ) (p : ConvexPolygon n) : ℚ :=
  (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5)

theorem diagonal_intersections_count (n : ℕ) (p : ConvexPolygon n) :
  diagonal_intersections_outside n p = (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5) := by
  sorry

end diagonal_intersections_count_l2181_218164


namespace prime_squared_minus_one_divisible_by_24_l2181_218129

theorem prime_squared_minus_one_divisible_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end prime_squared_minus_one_divisible_by_24_l2181_218129


namespace distance_AB_is_600_l2181_218121

/-- The distance between city A and city B -/
def distance_AB : ℝ := 600

/-- The time taken by Eddy to travel from A to B -/
def time_Eddy : ℝ := 3

/-- The time taken by Freddy to travel from A to C -/
def time_Freddy : ℝ := 4

/-- The distance between city A and city C -/
def distance_AC : ℝ := 460

/-- The ratio of Eddy's average speed to Freddy's average speed -/
def speed_ratio : ℝ := 1.7391304347826086

theorem distance_AB_is_600 :
  distance_AB = (speed_ratio * distance_AC * time_Eddy) / time_Freddy :=
sorry

end distance_AB_is_600_l2181_218121


namespace no_solution_exists_l2181_218173

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem no_solution_exists : ¬∃ (x y : ℕ), sumOfDigits ((10^x)^y - 64) = 279 := by
  sorry

end no_solution_exists_l2181_218173


namespace ferris_wheel_capacity_l2181_218162

theorem ferris_wheel_capacity 
  (total_people : ℕ) 
  (total_seats : ℕ) 
  (h1 : total_people = 16) 
  (h2 : total_seats = 4) 
  : total_people / total_seats = 4 := by
  sorry

end ferris_wheel_capacity_l2181_218162


namespace parabola_tangents_and_triangle_l2181_218109

/-- Parabola defined by the equation 8y = (x-3)^2 -/
def parabola (x y : ℝ) : Prop := 8 * y = (x - 3)^2

/-- Point M -/
def M : ℝ × ℝ := (0, -2)

/-- Tangent line equation -/
def is_tangent_line (m b : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ 
    (∀ x y, y = m * x + b ↔ (x = x₀ ∧ y = y₀ ∨ (y - y₀) = (x - x₀) * ((x₀ - 3) / 4)))

/-- Theorem stating the properties of the tangent lines and the triangle -/
theorem parabola_tangents_and_triangle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent lines equations
    is_tangent_line (-2) (-2) ∧
    is_tangent_line (1/2) (-2) ∧
    -- Points A and B are on the parabola
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    -- A and B are on the tangent lines
    y₁ = -2 * x₁ - 2 ∧
    y₂ = 1/2 * x₂ - 2 ∧
    -- Tangent lines are perpendicular
    (-2) * (1/2) = -1 ∧
    -- Area of triangle ABM
    abs ((x₁ - 0) * (y₂ - (-2)) - (x₂ - 0) * (y₁ - (-2))) / 2 = 125/4 := by
  sorry

end parabola_tangents_and_triangle_l2181_218109


namespace problem_solution_l2181_218100

theorem problem_solution (x y z : ℚ) : 
  x / (y + 1) = 4 / 5 → 
  3 * z = 2 * x + y → 
  y = 10 → 
  z = 46 / 5 := by
sorry

end problem_solution_l2181_218100


namespace triangle_345_is_acute_l2181_218163

/-- A triangle with sides 3, 4, and 4.5 is acute. -/
theorem triangle_345_is_acute : 
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 4.5 → 
  (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) := by
  sorry

end triangle_345_is_acute_l2181_218163


namespace power_division_equality_l2181_218111

theorem power_division_equality : (3 : ℕ)^12 / (27 : ℕ)^2 = 729 := by sorry

end power_division_equality_l2181_218111


namespace hyperbola_to_ellipse_l2181_218141

/-- Given a hyperbola with equation x^2/4 - y^2/12 = -1, 
    prove that the equation of the ellipse with its vertices at the foci of the hyperbola 
    and its foci at the vertices of the hyperbola is x^2/4 + y^2/16 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (x' y' : ℝ), (x'^2 / 4 + y'^2 / 16 = 1 ∧ 
    (∀ (a b c : ℝ), (a > b ∧ b > 0 ∧ c > 0) → 
      (y'^2 / a^2 + x'^2 / b^2 = 1 ↔ 
        (a = 4 ∧ b^2 = 4 ∧ c = 2 * Real.sqrt 3)))) :=
by sorry

end hyperbola_to_ellipse_l2181_218141


namespace parabola_translation_l2181_218145

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola --/
def original : Parabola := { a := -2, h := -2, k := 3 }

/-- The translated parabola --/
def translated : Parabola := { a := -2, h := 1, k := -1 }

/-- The translation that moves the original parabola to the translated parabola --/
def translation : Translation := { dx := 3, dy := -4 }

theorem parabola_translation : 
  ∀ (x y : ℝ), 
  (y = -2 * (x - translated.h)^2 + translated.k) ↔ 
  (y + translation.dy = -2 * ((x - translation.dx) - original.h)^2 + original.k) :=
sorry

end parabola_translation_l2181_218145


namespace complex_equality_sum_l2181_218114

theorem complex_equality_sum (a b : ℝ) : a - 3*I = 2 + b*I → a + b = -1 := by
  sorry

end complex_equality_sum_l2181_218114


namespace playground_children_count_l2181_218180

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
  sorry

end playground_children_count_l2181_218180


namespace helen_gas_usage_l2181_218103

-- Define the number of months with 2 cuts and 4 cuts
def months_with_two_cuts : ℕ := 4
def months_with_four_cuts : ℕ := 4

-- Define the number of cuts per month for each category
def cuts_per_month_low : ℕ := 2
def cuts_per_month_high : ℕ := 4

-- Define the gas usage
def gas_per_fourth_cut : ℕ := 2
def cuts_per_gas_usage : ℕ := 4

-- Theorem statement
theorem helen_gas_usage :
  let total_cuts := months_with_two_cuts * cuts_per_month_low + months_with_four_cuts * cuts_per_month_high
  let gas_fill_ups := total_cuts / cuts_per_gas_usage
  gas_fill_ups * gas_per_fourth_cut = 12 := by
  sorry

end helen_gas_usage_l2181_218103


namespace fraction_equality_l2181_218188

theorem fraction_equality (u v : ℝ) (h : (1/u + 1/v) / (1/u - 1/v) = 2024) :
  (u + v) / (u - v) = 2024 := by
  sorry

end fraction_equality_l2181_218188


namespace min_value_of_expression_l2181_218160

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 →
    1/(x' - 1) + 4/(y' - 1) ≥ min_val :=
by sorry

end min_value_of_expression_l2181_218160


namespace shoes_sold_l2181_218159

theorem shoes_sold (large medium small left : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_left : left = 13) :
  large + medium + small - left = 83 := by
  sorry

end shoes_sold_l2181_218159


namespace custom_operation_equality_l2181_218106

/-- Custom operation $ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end custom_operation_equality_l2181_218106


namespace travel_distance_ratio_l2181_218139

/-- Given a total distance traveled, with specified portions by plane and bus,
    calculate the ratio of train distance to bus distance. -/
theorem travel_distance_ratio
  (total_distance : ℝ)
  (plane_fraction : ℝ)
  (bus_distance : ℝ)
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1 / 3)
  (h3 : bus_distance = 720)
  : (total_distance - plane_fraction * total_distance - bus_distance) / bus_distance = 2 / 3 := by
  sorry

#check travel_distance_ratio

end travel_distance_ratio_l2181_218139


namespace animals_left_after_sale_l2181_218187

/-- Calculates the number of animals left in a pet store after a sale --/
theorem animals_left_after_sale (siamese_cats house_cats dogs birds cats_sold dogs_sold birds_sold : ℕ) :
  siamese_cats = 25 →
  house_cats = 55 →
  dogs = 30 →
  birds = 20 →
  cats_sold = 45 →
  dogs_sold = 25 →
  birds_sold = 10 →
  (siamese_cats + house_cats - cats_sold) + (dogs - dogs_sold) + (birds - birds_sold) = 50 := by
sorry

end animals_left_after_sale_l2181_218187


namespace tangent_line_to_circle_l2181_218165

theorem tangent_line_to_circle (a : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, x^2 + a^2 + 2*x - 2*a - 2 = 0 ∧ 
   ∀ y : ℝ, y ≠ a → x^2 + y^2 + 2*x - 2*y - 2 > 0) → 
  a = 3 := by sorry

end tangent_line_to_circle_l2181_218165


namespace equation_solution_l2181_218182

theorem equation_solution :
  ∀ y : ℝ, (5 + 3.2 * y = 2.1 * y - 25) ↔ (y = -300 / 11) :=
by sorry

end equation_solution_l2181_218182


namespace gumball_probability_l2181_218156

theorem gumball_probability (orange green yellow : ℕ) 
  (h_orange : orange = 10)
  (h_green : green = 6)
  (h_yellow : yellow = 9) :
  let total := orange + green + yellow
  let p_first_orange := orange / total
  let p_second_not_orange := (green + yellow) / (total - 1)
  let p_third_orange := (orange - 1) / (total - 2)
  p_first_orange * p_second_not_orange * p_third_orange = 9 / 92 := by
  sorry

end gumball_probability_l2181_218156


namespace expenditure_problem_l2181_218117

/-- Proves that given the conditions of the expenditure problem, the number of days in the next part of the week is 4. -/
theorem expenditure_problem (first_part_days : ℕ) (second_part_days : ℕ) 
  (first_part_avg : ℚ) (second_part_avg : ℚ) (total_avg : ℚ) :
  first_part_days = 3 →
  first_part_avg = 350 →
  second_part_avg = 420 →
  total_avg = 390 →
  first_part_days + second_part_days = 7 →
  (first_part_days * first_part_avg + second_part_days * second_part_avg) / 7 = total_avg →
  second_part_days = 4 := by
sorry

end expenditure_problem_l2181_218117


namespace city_population_problem_l2181_218155

theorem city_population_problem (population_b : ℕ) : 
  let population_a := (3 * population_b) / 5
  let population_c := 27500
  let total_population := population_a + population_b + population_c
  (population_c = (5 * population_b) / 4 + 4000) →
  (total_population % 250 = 0) →
  total_population = 57500 :=
by sorry

end city_population_problem_l2181_218155


namespace area_of_awesome_points_l2181_218151

/-- A right triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 3 ∧ b = 4 ∧ c = 5

/-- A point is awesome if it's the center of a parallelogram with vertices on the triangle's boundary -/
def is_awesome (T : RightTriangle) (P : ℝ × ℝ) : Prop := sorry

/-- The set of awesome points -/
def awesome_points (T : RightTriangle) : Set (ℝ × ℝ) :=
  {P | is_awesome T P}

/-- The area of a set of points in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The area of awesome points in the 3-4-5 right triangle is 3/2 -/
theorem area_of_awesome_points (T : RightTriangle) :
  area (awesome_points T) = 3/2 := by sorry

end area_of_awesome_points_l2181_218151


namespace production_rates_correct_l2181_218189

/-- Represents the production data for a company in November and March --/
structure ProductionData where
  nov_production : ℕ
  mar_production : ℕ
  time_difference : ℕ
  efficiency_ratio : Rat

/-- Calculates the production rates given the production data --/
def calculate_production_rates (data : ProductionData) : ℚ × ℚ :=
  let nov_rate := 2 * data.efficiency_ratio
  let mar_rate := 3 * data.efficiency_ratio
  (nov_rate, mar_rate)

theorem production_rates_correct (data : ProductionData) 
  (h1 : data.nov_production = 1400)
  (h2 : data.mar_production = 2400)
  (h3 : data.time_difference = 50)
  (h4 : data.efficiency_ratio = 2/3) :
  calculate_production_rates data = (4, 6) := by
  sorry

#eval calculate_production_rates {
  nov_production := 1400,
  mar_production := 2400,
  time_difference := 50,
  efficiency_ratio := 2/3
}

end production_rates_correct_l2181_218189


namespace apple_grape_equivalence_l2181_218191

/-- Given that 3/4 of 12 apples are worth 9 grapes, 
    prove that 1/2 of 6 apples are worth 3 grapes -/
theorem apple_grape_equivalence : 
  (3/4 : ℚ) * 12 * (1 : ℚ) = 9 * (1 : ℚ) → 
  (1/2 : ℚ) * 6 * (1 : ℚ) = 3 * (1 : ℚ) :=
by
  sorry

end apple_grape_equivalence_l2181_218191


namespace average_increase_l2181_218102

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 1.6 * x + 2

-- Theorem statement
theorem average_increase (x : ℝ) : 
  linear_regression (x + 1) - linear_regression x = 1.6 := by
  sorry

end average_increase_l2181_218102


namespace watch_price_proof_l2181_218134

/-- Represents the original cost price of the watch in Rupees. -/
def original_price : ℝ := 1800

/-- The selling price after discounts and loss. -/
def selling_price (price : ℝ) : ℝ := price * (1 - 0.05) * (1 - 0.03) * (1 - 0.10)

/-- The selling price for an 8% gain with 12% tax. -/
def selling_price_with_gain_and_tax (price : ℝ) : ℝ := price * (1 + 0.08) + price * 0.12

theorem watch_price_proof :
  selling_price original_price = original_price * 0.90 ∧
  selling_price_with_gain_and_tax original_price = selling_price original_price + 540 :=
by sorry

end watch_price_proof_l2181_218134
