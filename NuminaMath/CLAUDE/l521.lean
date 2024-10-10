import Mathlib

namespace max_value_expression_l521_52150

theorem max_value_expression (x y : ℝ) : 2 * y^2 - y^4 - x^2 - 3 * x ≤ 13/4 := by sorry

end max_value_expression_l521_52150


namespace tuesday_sales_fifty_l521_52145

/-- Calculates the number of books sold on Tuesday given the initial stock,
    sales on other days, and the percentage of unsold books. -/
def books_sold_tuesday (initial_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ)
    (unsold_percentage : ℚ) : ℕ :=
  let unsold_books := (initial_stock : ℚ) * unsold_percentage / 100
  let other_days_sales := monday_sales + wednesday_sales + thursday_sales + friday_sales
  initial_stock - (other_days_sales + unsold_books.ceil.toNat)

/-- Theorem stating that the number of books sold on Tuesday is 50. -/
theorem tuesday_sales_fifty :
  books_sold_tuesday 1100 75 64 78 135 (63945/1000) = 50 := by
  sorry

end tuesday_sales_fifty_l521_52145


namespace smaller_number_in_ratio_l521_52148

theorem smaller_number_in_ratio (a b : ℕ) : 
  a > 0 → b > 0 → a * 3 = b * 2 → lcm a b = 120 → a = 80 := by
  sorry

end smaller_number_in_ratio_l521_52148


namespace kyle_bottles_l521_52179

theorem kyle_bottles (bottle_capacity : ℕ) (additional_bottles : ℕ) (total_stars : ℕ) :
  bottle_capacity = 15 →
  additional_bottles = 3 →
  total_stars = 75 →
  ∃ (initial_bottles : ℕ), initial_bottles = 2 ∧ 
    (initial_bottles + additional_bottles) * bottle_capacity = total_stars :=
by sorry

end kyle_bottles_l521_52179


namespace matrix_equality_l521_52193

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -2, 3]) : 
  B * A = !![5, 2; -2, 3] := by sorry

end matrix_equality_l521_52193


namespace gold_silver_price_ratio_l521_52127

/-- Proves that the ratio of gold price to silver price is 50 given the problem conditions --/
theorem gold_silver_price_ratio :
  let silver_amount : ℝ := 1.5
  let gold_amount : ℝ := 2 * silver_amount
  let silver_price : ℝ := 20
  let total_spent : ℝ := 3030
  let gold_price : ℝ := (total_spent - silver_amount * silver_price) / gold_amount
  gold_price / silver_price = 50 := by sorry

end gold_silver_price_ratio_l521_52127


namespace modular_arithmetic_problem_l521_52108

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 97) + 5 * (13⁻¹ : ZMod 97)) = (73 : ZMod 97) := by
  sorry

end modular_arithmetic_problem_l521_52108


namespace don_max_bottles_l521_52187

/-- The number of bottles Shop A sells to Don -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Shop B sells to Don -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Shop C sells to Don -/
def shop_c_bottles : ℕ := 220

/-- The maximum number of bottles Don can buy -/
def max_bottles : ℕ := shop_a_bottles + shop_b_bottles + shop_c_bottles

theorem don_max_bottles : max_bottles = 550 := by sorry

end don_max_bottles_l521_52187


namespace unique_solution_l521_52167

/-- Represents the letters used in the triangle puzzle -/
inductive Letter
| A | B | C | D | E | F

/-- Represents the mapping of letters to numbers -/
def LetterMapping := Letter → Fin 6

/-- Checks if a mapping is valid according to the puzzle rules -/
def is_valid_mapping (m : LetterMapping) : Prop :=
  m Letter.A ≠ m Letter.B ∧ m Letter.A ≠ m Letter.C ∧ m Letter.A ≠ m Letter.D ∧ m Letter.A ≠ m Letter.E ∧ m Letter.A ≠ m Letter.F ∧
  m Letter.B ≠ m Letter.C ∧ m Letter.B ≠ m Letter.D ∧ m Letter.B ≠ m Letter.E ∧ m Letter.B ≠ m Letter.F ∧
  m Letter.C ≠ m Letter.D ∧ m Letter.C ≠ m Letter.E ∧ m Letter.C ≠ m Letter.F ∧
  m Letter.D ≠ m Letter.E ∧ m Letter.D ≠ m Letter.F ∧
  m Letter.E ≠ m Letter.F ∧
  (m Letter.B).val + (m Letter.D).val + (m Letter.E).val = 14 ∧
  (m Letter.C).val + (m Letter.E).val + (m Letter.F).val = 12

/-- The unique solution to the puzzle -/
def solution : LetterMapping :=
  fun l => match l with
  | Letter.A => 0
  | Letter.B => 2
  | Letter.C => 1
  | Letter.D => 4
  | Letter.E => 5
  | Letter.F => 3

/-- Theorem stating that the solution is the only valid mapping -/
theorem unique_solution :
  is_valid_mapping solution ∧ ∀ m : LetterMapping, is_valid_mapping m → m = solution := by
  sorry


end unique_solution_l521_52167


namespace meaningful_expression_l521_52130

/-- The expression sqrt(a+1)/(a-2) is meaningful iff a ≥ -1 and a ≠ 2 -/
theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x^2 = a + 1) ∧ (a ≠ 2) ↔ a ≥ -1 ∧ a ≠ 2 :=
by sorry

end meaningful_expression_l521_52130


namespace unique_campers_difference_l521_52146

def rowing_problem (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ) : Prop :=
  let total_afternoon := morning_and_afternoon + afternoon_and_evening + (afternoon - morning_and_afternoon - afternoon_and_evening)
  let total_evening := afternoon_and_evening + evening_only
  morning = 33 ∧
  morning_and_afternoon = 11 ∧
  afternoon = 34 ∧
  afternoon_and_evening = 20 ∧
  evening_only = 10 ∧
  total_afternoon - total_evening = 4

theorem unique_campers_difference :
  ∃ (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ),
    rowing_problem morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only :=
by
  sorry

end unique_campers_difference_l521_52146


namespace intersection_M_N_l521_52113

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def N : Set ℝ := {x | 3 * x + 2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 3} := by sorry

end intersection_M_N_l521_52113


namespace power_fraction_evaluation_l521_52168

theorem power_fraction_evaluation (a b : ℕ) : 
  (2^a : ℕ) ∣ 360 ∧ 
  (3^b : ℕ) ∣ 360 ∧ 
  ∀ k > a, ¬((2^k : ℕ) ∣ 360) ∧ 
  ∀ l > b, ¬((3^l : ℕ) ∣ 360) →
  ((1/4 : ℚ) ^ (b - a) : ℚ) = 4 := by
sorry

end power_fraction_evaluation_l521_52168


namespace total_people_is_123_l521_52110

/-- Calculates the total number of people on a bus given the number of boys and additional information. -/
def total_people_on_bus (num_boys : ℕ) : ℕ :=
  let num_girls := num_boys + (2 * num_boys) / 5
  let num_students := num_boys + num_girls
  let num_adults := 3  -- driver, assistant, and teacher
  num_students + num_adults

/-- Theorem stating that given the conditions, the total number of people on the bus is 123. -/
theorem total_people_is_123 : total_people_on_bus 50 = 123 := by
  sorry

#eval total_people_on_bus 50

end total_people_is_123_l521_52110


namespace age_problem_l521_52144

/-- Theorem: Given the age relationships and total age, prove b's age --/
theorem age_problem (a b c d e : ℝ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  e = d / 2 + 3 →
  a + b + c + d + e = 70 →
  b = 16.625 := by
sorry

end age_problem_l521_52144


namespace total_spent_is_72_l521_52192

/-- The cost of a single trick deck in dollars -/
def deck_cost : ℕ := 9

/-- The number of decks Edward bought -/
def edward_decks : ℕ := 4

/-- The number of decks Edward's friend bought -/
def friend_decks : ℕ := 4

/-- The total amount spent by Edward and his friend -/
def total_spent : ℕ := deck_cost * (edward_decks + friend_decks)

theorem total_spent_is_72 : total_spent = 72 := by
  sorry

end total_spent_is_72_l521_52192


namespace father_age_problem_l521_52166

/-- The age problem -/
theorem father_age_problem (father_age son_age : ℕ) : 
  father_age = 3 * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 10 →
  father_age = 33 := by
sorry

end father_age_problem_l521_52166


namespace double_counted_page_number_l521_52197

theorem double_counted_page_number :
  ∃! (n : ℕ) (x : ℕ), 
    1 ≤ x ∧ 
    x ≤ n ∧ 
    n * (n + 1) / 2 + x = 2550 ∧ 
    x = 65 := by
  sorry

end double_counted_page_number_l521_52197


namespace intersection_M_N_l521_52154

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end intersection_M_N_l521_52154


namespace square_of_product_l521_52184

theorem square_of_product (p q : ℝ) : (-3 * p * q)^2 = 9 * p^2 * q^2 := by
  sorry

end square_of_product_l521_52184


namespace total_cost_calculation_l521_52124

/-- The total cost of buying pens and exercise books -/
def total_cost (m n : ℝ) : ℝ := 2 * m + 3 * n

/-- Theorem: The total cost of 2 pens at m yuan each and 3 exercise books at n yuan each is 2m + 3n yuan -/
theorem total_cost_calculation (m n : ℝ) :
  total_cost m n = 2 * m + 3 * n :=
by sorry

end total_cost_calculation_l521_52124


namespace black_cards_taken_out_l521_52186

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (remaining_black : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    black_cards := 26,
    remaining_black := 22 }

/-- Theorem: The number of black cards taken out is 4 -/
theorem black_cards_taken_out (d : Deck) (h1 : d = standard_deck) :
  d.black_cards - d.remaining_black = 4 := by
  sorry

end black_cards_taken_out_l521_52186


namespace a_99_value_l521_52164

def is_increasing (s : ℕ → ℝ) := ∀ n, s n ≤ s (n + 1)
def is_decreasing (s : ℕ → ℝ) := ∀ n, s n ≥ s (n + 1)

theorem a_99_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, |a n - a (n-1)| = (n : ℝ)^2)
  (h3 : is_increasing (λ n => a (2*n - 1)))
  (h4 : is_decreasing (λ n => a (2*n)))
  (h5 : a 1 > a 2) :
  a 99 = 4950 := by sorry

end a_99_value_l521_52164


namespace union_of_A_and_B_l521_52131

def A : Set ℤ := {-1, 3}
def B : Set ℤ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 3} := by sorry

end union_of_A_and_B_l521_52131


namespace least_common_multiple_3_4_6_7_8_l521_52165

theorem least_common_multiple_3_4_6_7_8 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) ∧ (8 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) ∧ (8 ∣ n) :=
by
  -- The proof goes here
  sorry

end least_common_multiple_3_4_6_7_8_l521_52165


namespace even_function_inequality_l521_52120

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x < f y

theorem even_function_inequality (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (h_even : is_even_function f)
  (h_increasing : increasing_on_negative f)
  (h_x₁_neg : x₁ < 0)
  (h_x₂_pos : 0 < x₂)
  (h_sum_pos : 0 < x₁ + x₂) :
  f (-x₁) > f x₂ := by
  sorry

end even_function_inequality_l521_52120


namespace amandas_family_size_l521_52123

theorem amandas_family_size :
  let total_rooms : ℕ := 9
  let rooms_with_four_walls : ℕ := 5
  let rooms_with_five_walls : ℕ := 4
  let walls_per_person : ℕ := 8
  let total_walls : ℕ := rooms_with_four_walls * 4 + rooms_with_five_walls * 5
  total_rooms = rooms_with_four_walls + rooms_with_five_walls →
  total_walls % walls_per_person = 0 →
  total_walls / walls_per_person = 5 :=
by sorry

end amandas_family_size_l521_52123


namespace inequality_system_solution_l521_52137

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 5 > 3 ∧ x > a) ↔ x > -2) → a ≤ -2 := by
  sorry

end inequality_system_solution_l521_52137


namespace divisibility_by_1961_l521_52171

theorem divisibility_by_1961 (n : ℕ) : ∃ k : ℤ, 5^(2*n) * 3^(4*n) - 2^(6*n) = k * 1961 := by
  sorry

end divisibility_by_1961_l521_52171


namespace average_age_after_leaving_l521_52121

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 8 →
  initial_avg = 27 →
  leaving_age = 21 →
  remaining_people = 7 →
  (initial_people : ℚ) * initial_avg - leaving_age ≥ 0 →
  (((initial_people : ℚ) * initial_avg - leaving_age) / remaining_people : ℚ) = 28 :=
by
  sorry

end average_age_after_leaving_l521_52121


namespace intersection_A_complement_B_l521_52116

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end intersection_A_complement_B_l521_52116


namespace part_one_part_two_l521_52159

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one (m : ℝ) :
  (∀ x : ℝ, f m x < 0) → -4 < m ∧ m ≤ 0 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → f m x > -m + x - 1) → m > 1 := by sorry

end part_one_part_two_l521_52159


namespace sin_beta_value_l521_52161

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = 4 / 5)
  (h4 : Real.cos (α + β) = 3 / 5) : 
  Real.sin β = 7 / 25 := by
  sorry

end sin_beta_value_l521_52161


namespace pure_imaginary_product_theorem_l521_52126

theorem pure_imaginary_product_theorem (z : ℂ) (a : ℝ) : 
  (∃ b : ℝ, z = b * Complex.I) → 
  (3 - Complex.I) * z = a + Complex.I → 
  a = 1/3 := by
sorry

end pure_imaginary_product_theorem_l521_52126


namespace max_value_implies_ratio_l521_52140

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

theorem max_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧
  (f a b 1 = 10) →
  a / b = -2/3 :=
sorry

end max_value_implies_ratio_l521_52140


namespace sky_diving_company_total_amount_l521_52141

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem sky_diving_company_total_amount :
  individual_bookings + group_bookings - refunds = 26400 := by
  sorry

end sky_diving_company_total_amount_l521_52141


namespace rajesh_savings_l521_52172

def monthly_salary : ℕ := 15000
def food_percentage : ℚ := 40 / 100
def medicine_percentage : ℚ := 20 / 100
def savings_percentage : ℚ := 60 / 100

theorem rajesh_savings : 
  let remaining := monthly_salary - (monthly_salary * food_percentage + monthly_salary * medicine_percentage)
  ↑(remaining * savings_percentage) = 3600 := by sorry

end rajesh_savings_l521_52172


namespace base_conversion_equivalence_l521_52189

theorem base_conversion_equivalence :
  ∀ (C B : ℕ),
    C < 9 →
    B < 6 →
    9 * C + B = 6 * B + C →
    C = 0 ∧ B = 0 :=
by sorry

end base_conversion_equivalence_l521_52189


namespace multiplication_addition_equality_l521_52183

theorem multiplication_addition_equality : 45 * 52 + 78 * 45 = 5850 := by
  sorry

end multiplication_addition_equality_l521_52183


namespace sqrt_a_power_b_equals_three_l521_52162

theorem sqrt_a_power_b_equals_three (a b : ℝ) 
  (h : a^2 - 6*a + Real.sqrt (2*b - 4) = -9) : 
  Real.sqrt (a^b) = 3 := by
sorry

end sqrt_a_power_b_equals_three_l521_52162


namespace faster_train_speed_l521_52149

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (overtake_time : ℝ)
  (h1 : train_length = 50)
  (h2 : slower_speed = 36)
  (h3 : overtake_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 46 :=
by
  sorry

#check faster_train_speed

end faster_train_speed_l521_52149


namespace sufficient_not_necessary_condition_l521_52190

/-- Given a quadratic function f(x) = x^2 + bx + c, 
    prove that c < 0 is sufficient but not necessary for f(x) < 0 to have a real solution -/
theorem sufficient_not_necessary_condition (b c : ℝ) :
  (∀ x, (x : ℝ)^2 + b*x + c < 0 → c < 0) ∧
  ¬(∀ b c : ℝ, (∃ x, (x : ℝ)^2 + b*x + c < 0) → c < 0) :=
by sorry

end sufficient_not_necessary_condition_l521_52190


namespace volume_of_rotated_composite_shape_l521_52118

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let lower_rectangle_height : ℝ := 4
  let lower_rectangle_width : ℝ := 1
  let upper_rectangle_height : ℝ := 1
  let upper_rectangle_width : ℝ := 5
  let volume_lower := π * lower_rectangle_height^2 * lower_rectangle_width
  let volume_upper := π * upper_rectangle_height^2 * upper_rectangle_width
  volume_lower + volume_upper = 21 * π := by
  sorry

end volume_of_rotated_composite_shape_l521_52118


namespace computer_discount_theorem_l521_52104

theorem computer_discount_theorem (saved : ℝ) (paid : ℝ) (additional_discount : ℝ) :
  saved = 120 →
  paid = 1080 →
  additional_discount = 0.05 →
  let original_price := saved + paid
  let first_discount_percentage := saved / original_price
  let second_discount_amount := additional_discount * paid
  let total_saved := saved + second_discount_amount
  let total_percentage_saved := total_saved / original_price
  total_percentage_saved = 0.145 := by
  sorry

end computer_discount_theorem_l521_52104


namespace garland_theorem_l521_52199

/-- The number of ways to arrange light bulbs in a garland -/
def garland_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red + 1) white * Nat.choose (blue + red) blue

/-- Theorem: The number of ways to arrange 9 blue, 7 red, and 14 white light bulbs
    in a garland, such that no two white light bulbs are adjacent, is 7,779,200 -/
theorem garland_theorem :
  garland_arrangements 9 7 14 = 7779200 := by
  sorry

#eval garland_arrangements 9 7 14

end garland_theorem_l521_52199


namespace quadratic_factor_condition_l521_52152

theorem quadratic_factor_condition (a b p q : ℝ) : 
  (∀ x, (x + a) * (x + b) = x^2 + p*x + q) →
  p > 0 →
  q < 0 →
  ((a > 0 ∧ b < 0 ∧ a > -b) ∨ (a < 0 ∧ b > 0 ∧ b > -a)) :=
by sorry

end quadratic_factor_condition_l521_52152


namespace not_function_B_but_others_are_l521_52153

-- Define the concept of a function
def is_function (f : ℝ → Set ℝ) : Prop :=
  ∀ x : ℝ, ∃! y : ℝ, y ∈ f x

-- Define the relationships
def rel_A (x : ℝ) : Set ℝ := {y | y = 1 / x}
def rel_B (x : ℝ) : Set ℝ := {y | |y| = 2 * x}
def rel_C (x : ℝ) : Set ℝ := {y | y = 2 * x^2}
def rel_D (x : ℝ) : Set ℝ := {y | y = 3 * x^3}

-- Theorem statement
theorem not_function_B_but_others_are :
  (¬ is_function rel_B) ∧ 
  (is_function rel_A) ∧ 
  (is_function rel_C) ∧ 
  (is_function rel_D) :=
sorry

end not_function_B_but_others_are_l521_52153


namespace solution_comparison_l521_52178

theorem solution_comparison (a a' b b' c : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  ((c - b) / a > (c - b') / a') ↔ ((c - b') / a' < (c - b) / a) := by sorry

end solution_comparison_l521_52178


namespace gcd_product_l521_52163

theorem gcd_product (a b n : ℕ) (ha : Nat.gcd a n = 1) (hb : Nat.gcd b n = 1) : 
  Nat.gcd (a * b) n = 1 := by
sorry

end gcd_product_l521_52163


namespace bugs_farthest_apart_l521_52181

/-- Two circles with a common point and bugs moving on them -/
structure TwoCirclesWithBugs where
  /-- Diameter of the larger circle in cm -/
  d_large : ℝ
  /-- Diameter of the smaller circle in cm -/
  d_small : ℝ
  /-- The two circles have exactly one common point -/
  common_point : Prop
  /-- Bugs start at the common point and move at the same speed -/
  bugs_same_speed : Prop

/-- The number of laps completed by the bug on the smaller circle when the bugs are farthest apart -/
def farthest_apart_laps (circles : TwoCirclesWithBugs) : ℕ :=
  4

/-- Theorem stating that the bugs are farthest apart after 4 laps on the smaller circle -/
theorem bugs_farthest_apart (circles : TwoCirclesWithBugs) 
    (h1 : circles.d_large = 48) 
    (h2 : circles.d_small = 30) : 
  farthest_apart_laps circles = 4 := by
  sorry

end bugs_farthest_apart_l521_52181


namespace car_travel_inequality_l521_52155

/-- Represents the daily distance traveled by a car -/
def daily_distance : ℝ → ℝ
| x => x + 19

/-- Represents the total distance traveled in 8 days -/
def total_distance (x : ℝ) : ℝ := 8 * (daily_distance x)

/-- Theorem stating the inequality representing the car's travel -/
theorem car_travel_inequality (x : ℝ) :
  total_distance x > 2200 ↔ 8 * (x + 19) > 2200 := by sorry

end car_travel_inequality_l521_52155


namespace fourth_vertex_location_l521_52134

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle defined by its four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a rectangle ABCD with specific constraints, the fourth vertex C is at (x, -y) -/
theorem fourth_vertex_location (a y d : ℝ) :
  let O : Point := ⟨0, 0⟩
  let A : Point := ⟨a, 0⟩
  let B : Point := ⟨a, y⟩
  let D : Point := ⟨0, d⟩
  ∀ (rect : Rectangle),
    rect.A = A →
    rect.B = B →
    rect.D = D →
    (O.x - rect.C.x) * (A.x - B.x) + (O.y - rect.C.y) * (A.y - B.y) = 0 →
    (O.x - D.x) * (A.x - rect.C.x) + (O.y - D.y) * (A.y - rect.C.y) = 0 →
    rect.C = ⟨a, -y⟩ :=
by
  sorry


end fourth_vertex_location_l521_52134


namespace lower_limit_of_set_D_l521_52176

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetD : Set ℕ := {n : ℕ | isPrime n ∧ n ≤ 25}

theorem lower_limit_of_set_D (rangeD : ℕ) (h_range : rangeD = 12) :
  ∃ (lower : ℕ), lower = 13 ∧ 
    (∀ n ∈ SetD, n ≥ lower) ∧
    (∃ m ∈ SetD, m = lower) ∧
    (∃ max ∈ SetD, max - lower = rangeD) :=
sorry

end lower_limit_of_set_D_l521_52176


namespace class_average_height_l521_52173

def average_height_problem (total_students : ℕ) (group1_count : ℕ) (group1_avg : ℝ) (group2_avg : ℝ) : Prop :=
  let group2_count : ℕ := total_students - group1_count
  let total_height : ℝ := group1_count * group1_avg + group2_count * group2_avg
  let class_avg : ℝ := total_height / total_students
  class_avg = 168.6

theorem class_average_height :
  average_height_problem 50 40 169 167 := by
  sorry

end class_average_height_l521_52173


namespace angle_expression_value_l521_52100

theorem angle_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3/2 := by
  sorry

end angle_expression_value_l521_52100


namespace rotation_result_l521_52170

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a vector 180° about the y-axis -/
def rotateY180 (v : Vector3D) : Vector3D :=
  { x := -v.x, y := v.y, z := -v.z }

/-- The given vector -/
def givenVector : Vector3D :=
  { x := 2, y := -1, z := 1 }

/-- The expected result after rotation -/
def expectedResult : Vector3D :=
  { x := -2, y := -1, z := -1 }

theorem rotation_result :
  rotateY180 givenVector = expectedResult := by sorry

end rotation_result_l521_52170


namespace acid_dilution_l521_52191

/-- Given an initial acid solution of m ounces at m% concentration, 
    prove that adding x ounces of water to reach (m-15)% concentration
    results in x = 15m / (m-15) for m > 30 -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h₁ : m > 30) :
  (m * m / 100 = (m - 15) * (m + x) / 100) → x = 15 * m / (m - 15) := by
  sorry


end acid_dilution_l521_52191


namespace speed_limit_inequality_l521_52128

/-- Given a speed limit of 40 km/h, prove that it can be expressed as v ≤ 40, where v is the speed of a vehicle. -/
theorem speed_limit_inequality (v : ℝ) (speed_limit : ℝ) (h : speed_limit = 40) :
  v ≤ speed_limit ↔ v ≤ 40 := by sorry

end speed_limit_inequality_l521_52128


namespace min_cost_for_ten_boxes_l521_52174

/-- Calculates the minimum cost for buying a given number of yogurt boxes under a "buy two get one free" promotion. -/
def min_cost (box_price : ℕ) (num_boxes : ℕ) : ℕ :=
  let full_price_boxes := (num_boxes + 2) / 3 * 2
  full_price_boxes * box_price

/-- Theorem stating that the minimum cost for 10 boxes of yogurt at 4 yuan each under the promotion is 28 yuan. -/
theorem min_cost_for_ten_boxes : min_cost 4 10 = 28 := by
  sorry

end min_cost_for_ten_boxes_l521_52174


namespace smallest_possible_b_l521_52194

theorem smallest_possible_b : ∃ (b : ℝ), b = 2 ∧ 
  (∀ (a : ℝ), (2 < a ∧ a < b) → 
    (2 + a ≤ b ∧ 1/a + 1/b ≤ 1)) ∧
  (∀ (b' : ℝ), 2 < b' → 
    (∃ (a : ℝ), (2 < a ∧ a < b') ∧ 
      (2 + a > b' ∨ 1/a + 1/b' > 1))) :=
sorry

end smallest_possible_b_l521_52194


namespace trapezoid_wings_area_l521_52196

/-- A trapezoid divided into four triangles -/
structure Trapezoid :=
  (A₁ : ℝ) -- Area of first triangle
  (A₂ : ℝ) -- Area of second triangle
  (A₃ : ℝ) -- Area of third triangle
  (A₄ : ℝ) -- Area of fourth triangle

/-- The theorem stating that if two triangles in the trapezoid have areas 4 and 9,
    then the sum of the areas of the other two triangles is 12 -/
theorem trapezoid_wings_area (T : Trapezoid) 
  (h₁ : T.A₁ = 4) 
  (h₂ : T.A₂ = 9) : 
  T.A₃ + T.A₄ = 12 :=
sorry

end trapezoid_wings_area_l521_52196


namespace triangular_sum_congruence_l521_52115

theorem triangular_sum_congruence (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := by
  sorry

end triangular_sum_congruence_l521_52115


namespace dans_remaining_cards_l521_52157

/-- Given Dan's initial number of baseball cards, the number of torn cards,
    and the number of cards sold to Sam, prove that Dan now has 82 baseball cards. -/
theorem dans_remaining_cards
  (initial_cards : ℕ)
  (torn_cards : ℕ)
  (sold_cards : ℕ)
  (h1 : initial_cards = 97)
  (h2 : torn_cards = 8)
  (h3 : sold_cards = 15) :
  initial_cards - sold_cards = 82 := by
  sorry

end dans_remaining_cards_l521_52157


namespace imaginary_part_of_z_l521_52195

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 3) = -1 + 3 * Complex.I) : 
  z.im = 1 := by
sorry

end imaginary_part_of_z_l521_52195


namespace unique_solution_of_equation_l521_52111

theorem unique_solution_of_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔
  x = 1 := by sorry

end unique_solution_of_equation_l521_52111


namespace convex_polygon_partition_l521_52158

/-- A convex polygon represented by its side lengths -/
structure ConvexPolygon where
  sides : List ℝ
  sides_positive : ∀ s ∈ sides, s > 0
  convexity : ∀ s ∈ sides, s ≤ (sides.sum / 2)

/-- The perimeter of a convex polygon -/
def perimeter (p : ConvexPolygon) : ℝ := p.sides.sum

/-- A partition of the sides of a polygon into two sets -/
structure Partition (p : ConvexPolygon) where
  set1 : List ℝ
  set2 : List ℝ
  partition_complete : set1 ∪ set2 = p.sides
  partition_disjoint : set1 ∩ set2 = ∅

theorem convex_polygon_partition (p : ConvexPolygon) :
  ∃ (part : Partition p), |part.set1.sum - part.set2.sum| ≤ (perimeter p) / 3 := by
  sorry

end convex_polygon_partition_l521_52158


namespace equation_condition_l521_52188

theorem equation_condition (x y z : ℤ) :
  x * (x - y) + y * (y - z) + z * (z - x) = 0 → x = y ∧ y = z := by
  sorry

end equation_condition_l521_52188


namespace intersection_sum_coordinates_l521_52147

/-- The quartic equation -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x + 1

/-- The linear equation -/
def g (x y : ℝ) : ℝ := 2*x - 3*y - 6

/-- The intersection points of f and g -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2 = 0}

theorem intersection_sum_coordinates :
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    x₁ + x₂ + x₃ + x₄ = 3 ∧
    y₁ + y₂ + y₃ + y₄ = -6 :=
by sorry

end intersection_sum_coordinates_l521_52147


namespace taxi_charge_calculation_l521_52135

/-- Calculates the taxi charge per mile given the initial fee, total distance, and total payment. -/
def taxi_charge_per_mile (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ) : ℚ :=
  (total_payment - initial_fee) / total_distance

/-- Theorem stating that the taxi charge per mile is $2.50 given the specific conditions. -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_payment : ℚ)
    (h1 : initial_fee = 2)
    (h2 : total_distance = 4)
    (h3 : total_payment = 12) :
    taxi_charge_per_mile initial_fee total_distance total_payment = 2.5 := by
  sorry

#eval taxi_charge_per_mile 2 4 12

end taxi_charge_calculation_l521_52135


namespace fun_run_participation_l521_52136

/-- Fun Run Participation Theorem -/
theorem fun_run_participation (signed_up_last_year : ℕ) (no_show_last_year : ℕ) : 
  signed_up_last_year = 200 →
  no_show_last_year = 40 →
  (signed_up_last_year - no_show_last_year) * 2 = 320 := by
  sorry

#check fun_run_participation

end fun_run_participation_l521_52136


namespace parabola_focus_directrix_distance_l521_52105

/-- A parabola is defined by its equation y² = x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  is_parabola : equation = fun y x => y^2 = x

/-- The distance between the focus and directrix of a parabola -/
def focus_directrix_distance (p : Parabola) : ℝ := sorry

/-- Theorem: The distance between the focus and directrix of the parabola y² = x is 0.5 -/
theorem parabola_focus_directrix_distance :
  ∀ p : Parabola, p.equation = fun y x => y^2 = x → focus_directrix_distance p = 0.5 := by sorry

end parabola_focus_directrix_distance_l521_52105


namespace root_implies_a_range_l521_52117

def f (a x : ℝ) : ℝ := 2 * a * x^2 + 2 * x - 3 - a

theorem root_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0) →
  a ≥ 1 ∨ a ≤ -(3 + Real.sqrt 7) / 2 :=
by sorry

end root_implies_a_range_l521_52117


namespace inequality_proof_l521_52101

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(a+c)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end inequality_proof_l521_52101


namespace unique_solution_quadratic_l521_52107

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
sorry

end unique_solution_quadratic_l521_52107


namespace special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l521_52177

/-- An ellipse passing through five given points with specific properties -/
structure SpecialEllipse where
  -- The five points the ellipse passes through
  p₁ : ℝ × ℝ := (-1, -1)
  p₂ : ℝ × ℝ := (0, 0)
  p₃ : ℝ × ℝ := (0, 4)
  p₄ : ℝ × ℝ := (4, 0)
  p₅ : ℝ × ℝ := (4, 4)
  -- The center of the ellipse
  center : ℝ × ℝ := (2, 2)
  -- The ellipse has axes parallel to the coordinate axes
  axes_parallel : Bool

/-- The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length (e : SpecialEllipse) : ℝ :=
  4

/-- The main theorem: The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length_is_four (e : SpecialEllipse) :
  special_ellipse_minor_axis_length e = 4 := by
  sorry

end special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l521_52177


namespace two_valid_colorings_l521_52109

/-- Represents the three possible colors for a hexagon. -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a position in the hexagonal grid. -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents the hexagonal grid. -/
def HexGrid := Position → Color

/-- Checks if two positions are adjacent in the hexagonal grid. -/
def are_adjacent (p1 p2 : Position) : Bool :=
  sorry

/-- Checks if a coloring of the hexagonal grid is valid. -/
def is_valid_coloring (grid : HexGrid) : Prop :=
  (grid ⟨1, 1⟩ = Color.Red) ∧
  (∀ p1 p2, are_adjacent p1 p2 → grid p1 ≠ grid p2)

/-- The number of valid colorings for the hexagonal grid. -/
def num_valid_colorings : ℕ :=
  sorry

/-- Theorem stating that there are exactly 2 valid colorings of the hexagonal grid. -/
theorem two_valid_colorings : num_valid_colorings = 2 := by
  sorry

end two_valid_colorings_l521_52109


namespace special_collection_loans_l521_52143

theorem special_collection_loans (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_books = 67) :
  (initial_books - final_books : ℚ) / (1 - return_rate) = 40 := by
  sorry

end special_collection_loans_l521_52143


namespace average_of_combined_sets_l521_52125

theorem average_of_combined_sets (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) 
  (h1 : n1 = 60) (h2 : n2 = 40) (h3 : avg1 = 40) (h4 : avg2 = 60) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 48 := by
  sorry

end average_of_combined_sets_l521_52125


namespace next_two_terms_of_sequence_l521_52112

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ - d * (n - 1)

theorem next_two_terms_of_sequence :
  let a₁ := 19.8
  let d := 1.2
  (arithmetic_sequence a₁ d 4 = 16.2) ∧ (arithmetic_sequence a₁ d 5 = 15) := by
  sorry

end next_two_terms_of_sequence_l521_52112


namespace acorn_price_multiple_l521_52185

theorem acorn_price_multiple :
  let alice_acorns : ℕ := 3600
  let alice_price_per_acorn : ℕ := 15
  let bob_total_payment : ℕ := 6000
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  (alice_total_payment : ℚ) / bob_total_payment = 9 := by
  sorry

end acorn_price_multiple_l521_52185


namespace sqrt_cube_root_equality_l521_52133

theorem sqrt_cube_root_equality (a : ℝ) (h : a > 0) : 
  Real.sqrt (a * Real.rpow a (1/3)) = Real.rpow a (2/3) := by
  sorry

end sqrt_cube_root_equality_l521_52133


namespace arithmetic_sequence_properties_l521_52119

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  first_positive : a 1 > 0
  sum_condition : a 1 + a 3 + a 5 = 6
  product_condition : a 1 * a 3 * a 5 = 0
  is_arithmetic : ∀ n m : ℕ+, a (n + m) - a n = m * (a 2 - a 1)

/-- The general term of the sequence -/
def general_term (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  5 - n

/-- The b_n term -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  1 / (n * (seq.a n - 6))

/-- The sum of the first n terms of b_n -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  -n / (n + 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = general_term seq n) ∧
  (∀ n : ℕ+, S seq n = -n / (n + 1)) := by
  sorry

end arithmetic_sequence_properties_l521_52119


namespace product_of_three_numbers_l521_52103

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 5 * c) : 
  a * b * c = 22500 / 343 := by
sorry

end product_of_three_numbers_l521_52103


namespace complex_fraction_equality_l521_52169

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i^3 / (1 + i) = -1/2 - 1/2 * i := by
  sorry

end complex_fraction_equality_l521_52169


namespace sum_of_divisors_57_l521_52180

/-- The sum of all positive divisors of 57 is 80. -/
theorem sum_of_divisors_57 : (Finset.filter (λ x => 57 % x = 0) (Finset.range 58)).sum id = 80 := by
  sorry

end sum_of_divisors_57_l521_52180


namespace min_value_quadratic_form_l521_52182

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end min_value_quadratic_form_l521_52182


namespace smallest_staircase_length_l521_52122

theorem smallest_staircase_length (n : ℕ) : 
  n > 30 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧
  (∀ m : ℕ, m > 30 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end smallest_staircase_length_l521_52122


namespace michelle_initial_crayons_l521_52175

/-- Given that Janet has 2 crayons and the sum of Michelle's initial crayons
    and Janet's crayons is 4, prove that Michelle initially has 2 crayons. -/
theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end michelle_initial_crayons_l521_52175


namespace adjusted_retail_price_l521_52151

/-- The adjusted retail price of a shirt given its cost price and price adjustments -/
theorem adjusted_retail_price 
  (a : ℝ) -- Cost price per shirt in yuan
  (m : ℝ) -- Initial markup percentage
  (n : ℝ) -- Price adjustment percentage
  : ℝ := by
  -- The adjusted retail price is a(1+m%)n% yuan
  sorry

#check adjusted_retail_price

end adjusted_retail_price_l521_52151


namespace simplify_and_rationalize_l521_52106

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l521_52106


namespace avery_egg_cartons_l521_52129

theorem avery_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : 
  num_chickens = 20 →
  eggs_per_chicken = 6 →
  eggs_per_carton = 12 →
  (num_chickens * eggs_per_chicken) / eggs_per_carton = 10 := by
sorry

end avery_egg_cartons_l521_52129


namespace job_completion_time_l521_52102

/-- Given that:
  * A can do a job in 15 days
  * A and B working together for 4 days complete 0.4666666666666667 of the job
  Prove that B can do the job alone in 20 days -/
theorem job_completion_time (a_time : ℝ) (together_time : ℝ) (together_completion : ℝ) (b_time : ℝ) :
  a_time = 15 →
  together_time = 4 →
  together_completion = 0.4666666666666667 →
  together_completion = together_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end job_completion_time_l521_52102


namespace geometric_sequence_divisibility_l521_52114

theorem geometric_sequence_divisibility (a₁ a₂ : ℚ) (n : ℕ) : 
  a₁ = 5/8 → a₂ = 25 → 
  (∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, m < k → ¬(∃ q : ℚ, a₁ * (a₂ / a₁)^(m-1) = 2000000 * q)) ∧
              (∃ q : ℚ, a₁ * (a₂ / a₁)^(k-1) = 2000000 * q)) → 
  n = 7 := by
sorry

end geometric_sequence_divisibility_l521_52114


namespace song_ratio_is_two_to_one_l521_52160

/-- Represents the number of songs on Aisha's mp3 player at different stages --/
structure SongCount where
  initial : ℕ
  afterWeek : ℕ
  added : ℕ
  removed : ℕ
  final : ℕ

/-- Calculates the ratio of added songs to songs after the first two weeks --/
def songRatio (s : SongCount) : ℚ :=
  s.added / (s.initial + s.afterWeek)

/-- Theorem stating the ratio of added songs to songs after the first two weeks --/
theorem song_ratio_is_two_to_one (s : SongCount)
  (h1 : s.initial = 500)
  (h2 : s.afterWeek = 500)
  (h3 : s.removed = 50)
  (h4 : s.final = 2950)
  (h5 : s.initial + s.afterWeek + s.added - s.removed = s.final) :
  songRatio s = 2 := by
  sorry

#check song_ratio_is_two_to_one

end song_ratio_is_two_to_one_l521_52160


namespace fifth_power_last_digit_l521_52138

theorem fifth_power_last_digit (n : ℕ) : n % 10 = (n^5) % 10 := by
  sorry

end fifth_power_last_digit_l521_52138


namespace quadratic_sum_l521_52142

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ :=
  fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c (-3) = 0) →
  (QuadraticFunction a b c 5 = 0) →
  (∀ x, QuadraticFunction a b c x ≥ -36) →
  (∃ x, QuadraticFunction a b c x = -36) →
  a + b + c = -36 := by
  sorry

end quadratic_sum_l521_52142


namespace fibonacci_divisibility_sequence_l521_52132

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility_sequence (m n : ℕ) (h : m > 0) (h' : n > 0) :
  m ∣ n → fib m ∣ fib n := by
  sorry

end fibonacci_divisibility_sequence_l521_52132


namespace eric_marbles_l521_52198

/-- The number of marbles Eric has -/
def total_marbles (white blue green : ℕ) : ℕ := white + blue + green

/-- Proof that Eric has 20 marbles in total -/
theorem eric_marbles : total_marbles 12 6 2 = 20 := by
  sorry

end eric_marbles_l521_52198


namespace digit_equation_solution_l521_52156

theorem digit_equation_solution (A M C : ℕ) : 
  A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9 →
  (100 * A + 10 * M + C) * (A + M + C) = 2040 →
  Even (A + M + C) →
  M = 7 :=
by sorry

end digit_equation_solution_l521_52156


namespace x_cubed_coefficient_l521_52139

/-- The coefficient of x³ in the expansion of (3x³ + 2x² + 4x + 5)(4x³ + 3x² + 5x + 6) is 32 -/
theorem x_cubed_coefficient (x : ℝ) : 
  (3*x^3 + 2*x^2 + 4*x + 5) * (4*x^3 + 3*x^2 + 5*x + 6) = 
  32*x^3 + (12*x^5 + 15*x^4 + 23*x^2 + 34*x + 30) := by
sorry

end x_cubed_coefficient_l521_52139
