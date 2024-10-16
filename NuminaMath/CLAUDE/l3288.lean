import Mathlib

namespace NUMINAMATH_CALUDE_external_diagonal_inequality_l3288_328824

theorem external_diagonal_inequality (a b c x y z : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  x^2 = a^2 + b^2 ∧ y^2 = b^2 + c^2 ∧ z^2 = a^2 + c^2 →
  x^2 + y^2 ≥ z^2 ∧ y^2 + z^2 ≥ x^2 ∧ z^2 + x^2 ≥ y^2 := by sorry

end NUMINAMATH_CALUDE_external_diagonal_inequality_l3288_328824


namespace NUMINAMATH_CALUDE_summer_discount_is_fifty_percent_l3288_328874

def original_price : ℝ := 49
def final_price : ℝ := 14.50
def additional_discount : ℝ := 10

def summer_discount_percentage (d : ℝ) : Prop :=
  original_price * (1 - d / 100) - additional_discount = final_price

theorem summer_discount_is_fifty_percent : 
  summer_discount_percentage 50 := by sorry

end NUMINAMATH_CALUDE_summer_discount_is_fifty_percent_l3288_328874


namespace NUMINAMATH_CALUDE_final_shape_independent_of_initial_fold_l3288_328893

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the folded state of the paper -/
inductive FoldedState
  | Unfolded
  | FoldedOnce
  | FoldedTwice
  | FoldedThrice

/-- Represents the initial fold direction -/
inductive FoldDirection
  | MN
  | AB

/-- Represents the final shape after unfolding -/
structure FinalShape :=
  (shape : Set (ℝ × ℝ))

/-- Function to fold the paper -/
def fold (s : Square) (state : FoldedState) : FoldedState :=
  match state with
  | FoldedState.Unfolded => FoldedState.FoldedOnce
  | FoldedState.FoldedOnce => FoldedState.FoldedTwice
  | FoldedState.FoldedTwice => FoldedState.FoldedThrice
  | FoldedState.FoldedThrice => FoldedState.FoldedThrice

/-- Function to cut and unfold the paper -/
def cutAndUnfold (s : Square) (state : FoldedState) (dir : FoldDirection) : FinalShape :=
  sorry

/-- Theorem stating that the final shape is independent of initial fold direction -/
theorem final_shape_independent_of_initial_fold (s : Square) :
  ∀ (dir1 dir2 : FoldDirection),
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir1 =
    cutAndUnfold s (fold s (fold s (fold s FoldedState.Unfolded))) dir2 :=
  sorry

end NUMINAMATH_CALUDE_final_shape_independent_of_initial_fold_l3288_328893


namespace NUMINAMATH_CALUDE_students_from_other_communities_l3288_328828

/-- Given a school with 1000 students and the percentages of students belonging to different communities,
    prove that the number of students from other communities is 90. -/
theorem students_from_other_communities
  (total_students : ℕ)
  (muslim_percent : ℚ)
  (hindu_percent : ℚ)
  (sikh_percent : ℚ)
  (christian_percent : ℚ)
  (buddhist_percent : ℚ)
  (h1 : total_students = 1000)
  (h2 : muslim_percent = 36 / 100)
  (h3 : hindu_percent = 24 / 100)
  (h4 : sikh_percent = 15 / 100)
  (h5 : christian_percent = 10 / 100)
  (h6 : buddhist_percent = 6 / 100) :
  ↑total_students * (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent)) = 90 :=
by sorry

end NUMINAMATH_CALUDE_students_from_other_communities_l3288_328828


namespace NUMINAMATH_CALUDE_vampire_daily_victims_l3288_328861

-- Define the vampire's weekly blood requirement in gallons
def weekly_blood_requirement : ℚ := 7

-- Define the amount of blood sucked per person in pints
def blood_per_person : ℚ := 2

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Define the number of pints in a gallon
def pints_per_gallon : ℕ := 8

-- Theorem: The vampire needs to suck blood from 4 people per day
theorem vampire_daily_victims : 
  (weekly_blood_requirement / days_per_week * pints_per_gallon) / blood_per_person = 4 := by
  sorry


end NUMINAMATH_CALUDE_vampire_daily_victims_l3288_328861


namespace NUMINAMATH_CALUDE_number_of_girls_l3288_328826

/-- The number of girls in the group -/
def n : ℕ := sorry

/-- The average weight of the group before the new girl arrives -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def W : ℝ := 80

theorem number_of_girls :
  (n * A = n * A - 55 + W) ∧ (n * (A + 1) = n * A - 55 + W) → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l3288_328826


namespace NUMINAMATH_CALUDE_paint_calculation_l3288_328844

/-- The total amount of paint needed for finishing touches -/
def total_paint_needed (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) : ℕ :=
  initial + purchased + additional_needed

/-- Theorem stating that the total paint needed is the sum of initial, purchased, and additional needed paint -/
theorem paint_calculation (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) :
  total_paint_needed initial purchased additional_needed =
  initial + purchased + additional_needed :=
by
  sorry

#eval total_paint_needed 36 23 11

end NUMINAMATH_CALUDE_paint_calculation_l3288_328844


namespace NUMINAMATH_CALUDE_largest_sum_is_923_l3288_328825

def digits : List Nat := [3, 5, 7, 8, 0]

def is_valid_partition (a b : List Nat) : Prop :=
  a.length = 3 ∧ b.length = 2 ∧ (a ++ b).toFinset = digits.toFinset

def to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem largest_sum_is_923 :
  ∀ a b : List Nat,
    is_valid_partition a b →
    to_number a + to_number b ≤ 923 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_923_l3288_328825


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3288_328802

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, if a₁ + a₂ + a₃ = 32 and a₁₁ + a₁₂ + a₁₃ = 118, then a₄ + a₁₀ = 50. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum1 : a 1 + a 2 + a 3 = 32) 
    (h_sum2 : a 11 + a 12 + a 13 = 118) : 
  a 4 + a 10 = 50 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3288_328802


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_100_l3288_328879

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_prime_divisor_of_sum_100 :
  let sum := sum_of_first_n 100
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ sum ∧ ∀ q < p, Nat.Prime q → ¬(q ∣ sum) ∧ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_100_l3288_328879


namespace NUMINAMATH_CALUDE_distilled_water_calculation_l3288_328869

/-- The amount of distilled water (in liters) used in the original mixture -/
def original_water : ℝ := 0.03

/-- The total amount of growth medium (in liters) in the original mixture -/
def original_total : ℝ := 0.08

/-- The amount of growth medium (in liters) needed for the experiment -/
def required_total : ℝ := 0.64

/-- The amount of distilled water (in liters) needed for the experiment -/
def required_water : ℝ := 0.24

theorem distilled_water_calculation :
  (required_total / original_total) * original_water = required_water :=
sorry

end NUMINAMATH_CALUDE_distilled_water_calculation_l3288_328869


namespace NUMINAMATH_CALUDE_greatest_npmm_l3288_328846

/-- Represents a three-digit number with equal digits -/
def ThreeEqualDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- Represents a one-digit number -/
def OneDigit (n : ℕ) : Prop := n < 10 ∧ n > 0

/-- Represents a four-digit number -/
def FourDigits (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

/-- The main theorem -/
theorem greatest_npmm :
  ∀ MMM M NPMM : ℕ,
    ThreeEqualDigits MMM →
    OneDigit M →
    FourDigits NPMM →
    MMM * M = NPMM →
    NPMM ≤ 3996 :=
by
  sorry

#check greatest_npmm

end NUMINAMATH_CALUDE_greatest_npmm_l3288_328846


namespace NUMINAMATH_CALUDE_discount_profit_equivalence_l3288_328837

theorem discount_profit_equivalence (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ)
  (h1 : discount_rate = 0.04)
  (h2 : profit_rate = 0.38) :
  let selling_price := cost_price * (1 + profit_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit_with_discount := discounted_price - cost_price
  let profit_without_discount := selling_price - cost_price
  profit_without_discount / cost_price = profit_rate :=
by
  sorry

end NUMINAMATH_CALUDE_discount_profit_equivalence_l3288_328837


namespace NUMINAMATH_CALUDE_sandwich_price_calculation_l3288_328873

/-- The price of a single sandwich -/
def sandwich_price : ℝ := 5

/-- The number of sandwiches ordered -/
def num_sandwiches : ℕ := 18

/-- The delivery fee -/
def delivery_fee : ℝ := 20

/-- The tip percentage -/
def tip_percent : ℝ := 0.1

/-- The total amount received -/
def total_received : ℝ := 121

theorem sandwich_price_calculation :
  sandwich_price * num_sandwiches + delivery_fee +
  (sandwich_price * num_sandwiches + delivery_fee) * tip_percent = total_received :=
by sorry

end NUMINAMATH_CALUDE_sandwich_price_calculation_l3288_328873


namespace NUMINAMATH_CALUDE_cost_decrease_l3288_328862

theorem cost_decrease (original_cost : ℝ) (decrease_percentage : ℝ) (new_cost : ℝ) : 
  original_cost = 200 →
  decrease_percentage = 50 →
  new_cost = original_cost * (1 - decrease_percentage / 100) →
  new_cost = 100 := by
sorry

end NUMINAMATH_CALUDE_cost_decrease_l3288_328862


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_l3288_328804

theorem cubic_factorization_sum (a b c d e : ℤ) : 
  (∀ x, 1728 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 132 := by
sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_l3288_328804


namespace NUMINAMATH_CALUDE_unbounded_fraction_value_l3288_328850

theorem unbounded_fraction_value (M : ℝ) :
  ∃ (x y : ℝ), -3 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 ∧ 1 ≤ y ∧ y ≤ 3 ∧ (x + y + 1) / x > M :=
by sorry

end NUMINAMATH_CALUDE_unbounded_fraction_value_l3288_328850


namespace NUMINAMATH_CALUDE_gcf_of_104_and_156_l3288_328812

theorem gcf_of_104_and_156 : Nat.gcd 104 156 = 52 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_104_and_156_l3288_328812


namespace NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3288_328831

/-- The sum of the series Σ(k=0 to ∞) (3^(2^k) / (6^(2^k) - 2)) is equal to 3/4 -/
theorem series_sum_equals_three_fourths : 
  ∑' k : ℕ, (3 : ℝ)^(2^k) / ((6 : ℝ)^(2^k) - 2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_fourths_l3288_328831


namespace NUMINAMATH_CALUDE_secret_organization_membership_l3288_328884

theorem secret_organization_membership (total_cents : ℕ) (max_members : ℕ) : 
  total_cents = 300737 ∧ max_members = 500 →
  ∃! (members : ℕ) (fee_cents : ℕ),
    members ≤ max_members ∧
    members * fee_cents = total_cents ∧
    members = 311 ∧
    fee_cents = 967 := by
  sorry

end NUMINAMATH_CALUDE_secret_organization_membership_l3288_328884


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l3288_328853

theorem sin_ratio_comparison : 
  (Real.sin (2014 * π / 180)) / (Real.sin (2015 * π / 180)) < 
  (Real.sin (2016 * π / 180)) / (Real.sin (2017 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l3288_328853


namespace NUMINAMATH_CALUDE_kolya_purchase_l3288_328866

/-- Represents the price of an item in kopecks -/
def ItemPrice : ℕ → Prop :=
  λ p => ∃ a : ℕ, p = 100 * a + 99

/-- The total cost of the purchase in kopecks -/
def TotalCost : ℕ := 20083

/-- Represents the number of items bought -/
def NumberOfItems : ℕ → Prop :=
  λ n => ∃ p : ℕ, ItemPrice p ∧ n * p = TotalCost

theorem kolya_purchase :
  ∀ n : ℕ, NumberOfItems n ↔ (n = 17 ∨ n = 117) :=
sorry

end NUMINAMATH_CALUDE_kolya_purchase_l3288_328866


namespace NUMINAMATH_CALUDE_art_class_gender_difference_l3288_328887

theorem art_class_gender_difference (total_students : ℕ) 
  (boy_ratio girl_ratio : ℕ) (h1 : total_students = 42) 
  (h2 : boy_ratio = 3) (h3 : girl_ratio = 4) : 
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boy_ratio * girls = girl_ratio * boys ∧ 
    girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_art_class_gender_difference_l3288_328887


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_16_plus_9_l3288_328859

theorem abs_neg_sqrt_16_plus_9 : |-(Real.sqrt 16) + 9| = 5 := by sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_16_plus_9_l3288_328859


namespace NUMINAMATH_CALUDE_net_growth_rate_calculation_l3288_328860

/-- Given birth and death rates per certain number of people and an initial population,
    calculate the net growth rate as a percentage. -/
theorem net_growth_rate_calculation 
  (birth_rate death_rate : ℕ) 
  (initial_population : ℕ) 
  (birth_rate_val : birth_rate = 32)
  (death_rate_val : death_rate = 11)
  (initial_population_val : initial_population = 1000) :
  (birth_rate - death_rate : ℝ) / initial_population * 100 = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_net_growth_rate_calculation_l3288_328860


namespace NUMINAMATH_CALUDE_men_at_first_stop_l3288_328857

/-- Represents the number of people on a subway --/
structure SubwayPopulation where
  women : ℕ
  men : ℕ

/-- The subway population after the first stop --/
def first_stop : SubwayPopulation → Prop
  | ⟨w, m⟩ => m = w - 17

/-- The change in subway population at the second stop --/
def second_stop (pop : SubwayPopulation) : ℕ := 
  pop.women + pop.men + (57 + 18 - 44)

/-- The theorem stating the number of men who got on at the first stop --/
theorem men_at_first_stop (pop : SubwayPopulation) : 
  first_stop pop → second_stop pop = 502 → pop.men = 227 := by
  sorry

end NUMINAMATH_CALUDE_men_at_first_stop_l3288_328857


namespace NUMINAMATH_CALUDE_roots_sum_product_l3288_328830

theorem roots_sum_product (α' β' : ℝ) : 
  (α' + β' = 5) → (α' * β' = 6) → 3 * α'^3 + 4 * β'^2 = 271 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_product_l3288_328830


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l3288_328835

-- Define the equation
def equation (x : ℝ) : Prop := (9 - x^2 / 3)^(1/3) = 3

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 18 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l3288_328835


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_lower_bound_l3288_328883

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 2} = Set.Ici (4/3) ∪ Set.Iic 0 := by sorry

-- Theorem for part II
theorem f_lower_bound (a x : ℝ) :
  f a x ≥ |a - 1/2| := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_f_lower_bound_l3288_328883


namespace NUMINAMATH_CALUDE_fran_travel_time_l3288_328809

/-- Proves that given Joann's speed and time, and Fran's speed, Fran will take 3 hours to travel the same distance as Joann. -/
theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_speed = 20) :
  (joann_speed * joann_time) / fran_speed = 3 := by
  sorry

#check fran_travel_time

end NUMINAMATH_CALUDE_fran_travel_time_l3288_328809


namespace NUMINAMATH_CALUDE_f_expression_l3288_328872

-- Define the function f
def f : ℝ → ℝ := λ x => 2 * (x - 1) - 1

-- Theorem statement
theorem f_expression : ∀ x : ℝ, f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_l3288_328872


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l3288_328843

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) = -x^2 + 24*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l3288_328843


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_sqrt_three_only_irrational_l3288_328832

theorem sqrt_three_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ) :=
by
  sorry

-- Definitions for rational numbers in the problem
def zero_rational : ℚ := 0
def one_point_five_rational : ℚ := 3/2
def negative_two_rational : ℚ := -2

-- Theorem stating that √3 is the only irrational number among the given options
theorem sqrt_three_only_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ zero_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ one_point_five_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ negative_two_rational = (p : ℚ) / (q : ℚ)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_sqrt_three_only_irrational_l3288_328832


namespace NUMINAMATH_CALUDE_baking_cookies_theorem_l3288_328836

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

theorem baking_cookies_theorem (total_time minutes_per_pan : ℕ) 
  (h1 : total_time = 28) (h2 : minutes_per_pan = 7) : 
  pans_of_cookies total_time minutes_per_pan = 4 := by
  sorry

end NUMINAMATH_CALUDE_baking_cookies_theorem_l3288_328836


namespace NUMINAMATH_CALUDE_min_value_of_function_solution_set_inequality_l3288_328867

-- Part 1
theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (1 / Real.sin x ^ 2) + (4 / Real.cos x ^ 2) ≥ 9 := sorry

-- Part 2
theorem solution_set_inequality (a b c α β : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β) 
  (h2 : 0 < α) (h3 : α < β) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1/β ∨ x > 1/α := sorry

end NUMINAMATH_CALUDE_min_value_of_function_solution_set_inequality_l3288_328867


namespace NUMINAMATH_CALUDE_susie_piggy_bank_total_l3288_328871

/-- Calculates the total amount in Susie's piggy bank after two years -/
def piggy_bank_total (initial_amount : ℝ) (first_year_addition : ℝ) (second_year_addition : ℝ) (interest_rate : ℝ) : ℝ :=
  let first_year_total := (initial_amount + initial_amount * first_year_addition) * (1 + interest_rate)
  let second_year_total := (first_year_total + first_year_total * second_year_addition) * (1 + interest_rate)
  second_year_total

/-- Theorem stating that Susie's piggy bank total after two years is $343.98 -/
theorem susie_piggy_bank_total :
  piggy_bank_total 200 0.2 0.3 0.05 = 343.98 := by
  sorry

end NUMINAMATH_CALUDE_susie_piggy_bank_total_l3288_328871


namespace NUMINAMATH_CALUDE_binomial_15_choose_4_l3288_328819

theorem binomial_15_choose_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_choose_4_l3288_328819


namespace NUMINAMATH_CALUDE_farm_tax_collection_l3288_328842

theorem farm_tax_collection (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 0.25) : 
  william_tax / william_land_percentage = 1920 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_collection_l3288_328842


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l3288_328888

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x * Real.sin y = 0.75) →
  (Real.tan x * Real.tan y = 3) →
  ∃ (k n : ℤ), 
    (x = π/3 + π*(k + n : ℝ) ∨ x = -π/3 + π*(k + n : ℝ)) ∧
    (y = π/3 + π*(n - k : ℝ) ∨ y = -π/3 + π*(n - k : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l3288_328888


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3288_328806

theorem fraction_to_decimal : 58 / 200 = 1.16 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3288_328806


namespace NUMINAMATH_CALUDE_triangle_side_ratio_range_l3288_328878

theorem triangle_side_ratio_range (a b c : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < Real.pi / 2 →
  a^2 = b^2 + b*c →
  Real.sqrt 2 < a/b ∧ a/b < 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_range_l3288_328878


namespace NUMINAMATH_CALUDE_box_face_ratio_l3288_328851

/-- Given a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Properties of the box -/
def BoxProperties (box : Box) : Prop :=
  box.l > 0 ∧ box.w > 0 ∧ box.h > 0 ∧
  box.l * box.w * box.h = 5184 ∧
  box.l * box.h = 288 ∧
  box.w * box.h = (1/2) * box.l * box.w

theorem box_face_ratio (box : Box) (hp : BoxProperties box) :
  (box.l * box.w) / (box.l * box.h) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_face_ratio_l3288_328851


namespace NUMINAMATH_CALUDE_parallel_line_equation_distance_to_origin_equal_intercepts_l3288_328870

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x - y + 4 * a = 0

-- Part 1
theorem parallel_line_equation :
  ∃ (m b : ℝ), line_l 1 m b ∧ m = 1 ∧ b = 0 ∧ 
  ∀ (x y : ℝ), 2 * x - y - 2 = 0 ↔ (m + 1) * x - y + 4 * m = 0 :=
sorry

-- Part 2
theorem distance_to_origin :
  ∃ (a : ℝ), a = -1 ∧ 
  (4 : ℝ) = |4 * a| / Real.sqrt ((a + 1)^2 + 1) :=
sorry

-- Part 3
theorem equal_intercepts :
  ∃ (a : ℝ), (a = 0 ∨ a = -2) ∧
  -(4 * a) / (a + 1) = 4 * a :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_distance_to_origin_equal_intercepts_l3288_328870


namespace NUMINAMATH_CALUDE_three_bus_interval_l3288_328892

/-- Given a circular bus route with two buses operating at an interval of 21 minutes,
    this theorem proves that when three buses operate on the same route at the same speed,
    the new interval between consecutive buses is 14 minutes. -/
theorem three_bus_interval (interval_two_buses : ℕ) (h : interval_two_buses = 21) :
  let total_time := 2 * interval_two_buses
  let interval_three_buses := total_time / 3
  interval_three_buses = 14 := by
sorry

end NUMINAMATH_CALUDE_three_bus_interval_l3288_328892


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3288_328848

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a1, a3, and a4 form a geometric sequence
  a 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3288_328848


namespace NUMINAMATH_CALUDE_max_crayfish_revenue_l3288_328800

/-- The revenue function for selling crayfish -/
def revenue (total : ℕ) (sold : ℕ) : ℝ :=
  (total - sold : ℝ) * ((total - sold : ℝ) - 4.5) * sold

/-- The statement that proves the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  let total := 32
  ∃ (max_sold : ℕ) (max_revenue : ℝ),
    max_sold = 14 ∧
    max_revenue = 189 ∧
    ∀ (sold : ℕ), sold ≤ total → revenue total sold ≤ max_revenue :=
by sorry

end NUMINAMATH_CALUDE_max_crayfish_revenue_l3288_328800


namespace NUMINAMATH_CALUDE_k_value_l3288_328821

theorem k_value (k : ℝ) : (5 + k) * (5 - k) = 5^2 - 2^3 → k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3288_328821


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3288_328811

theorem polynomial_division_remainder (t : ℚ) :
  (∀ x, (6 * x^2 - 7 * x + 8) = (5 * x^2 + t * x + 12) * (4 * x^2 - 9 * x + 12)) →
  t = -7/12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3288_328811


namespace NUMINAMATH_CALUDE_combined_teaching_experience_l3288_328840

theorem combined_teaching_experience 
  (james_experience : ℕ) 
  (partner_experience : ℕ) 
  (h1 : james_experience = 40)
  (h2 : partner_experience = james_experience - 10) : 
  james_experience + partner_experience = 70 := by
sorry

end NUMINAMATH_CALUDE_combined_teaching_experience_l3288_328840


namespace NUMINAMATH_CALUDE_sum_seventh_eighth_l3288_328864

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 16
  sum_third_fourth : a 3 + a 4 = 32

/-- The sum of the 7th and 8th terms is 128 -/
theorem sum_seventh_eighth (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_sum_seventh_eighth_l3288_328864


namespace NUMINAMATH_CALUDE_square_sum_value_l3288_328891

theorem square_sum_value (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3288_328891


namespace NUMINAMATH_CALUDE_extreme_value_at_zero_decreasing_on_interval_l3288_328810

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (3 * x^2 + a * x) / Real.exp x

-- Theorem for the first part of the problem
theorem extreme_value_at_zero (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≥ f a x) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≤ f a x) ↔
  a = 0 :=
sorry

-- Theorem for the second part of the problem
theorem decreasing_on_interval (a : ℝ) :
  (∀ (x y : ℝ), 3 ≤ x ∧ x < y → f a x > f a y) ↔
  a ≥ -9/2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_at_zero_decreasing_on_interval_l3288_328810


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3288_328896

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x * (x + 2) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3288_328896


namespace NUMINAMATH_CALUDE_mirror_pieces_l3288_328847

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked : ℕ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked = (total - swept - stolen) / 3 →
  picked = 9 := by
sorry

end NUMINAMATH_CALUDE_mirror_pieces_l3288_328847


namespace NUMINAMATH_CALUDE_fewer_girls_than_boys_l3288_328889

theorem fewer_girls_than_boys (total_students : ℕ) (girls_ratio boys_ratio : ℕ) : 
  total_students = 24 →
  girls_ratio = 3 →
  boys_ratio = 5 →
  total_students * girls_ratio / (girls_ratio + boys_ratio) = 9 ∧
  total_students * boys_ratio / (girls_ratio + boys_ratio) = 15 ∧
  15 - 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_fewer_girls_than_boys_l3288_328889


namespace NUMINAMATH_CALUDE_opposite_of_one_l3288_328808

/-- Two real numbers are opposites if their sum is zero -/
def IsOpposite (x y : ℝ) : Prop := x + y = 0

/-- If a is the opposite of 1, then a = -1 -/
theorem opposite_of_one (a : ℝ) (h : IsOpposite a 1) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_l3288_328808


namespace NUMINAMATH_CALUDE_train_speed_problem_l3288_328820

def train_journey (x : ℝ) (v : ℝ) : Prop :=
  let first_part_distance := x
  let first_part_speed := 40
  let second_part_distance := 2 * x
  let second_part_speed := v
  let total_distance := 3 * x
  let average_speed := 24
  (first_part_distance / first_part_speed + second_part_distance / second_part_speed) * average_speed = total_distance

theorem train_speed_problem (x : ℝ) (hx : x > 0) :
  ∃ v : ℝ, train_journey x v ∧ v = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3288_328820


namespace NUMINAMATH_CALUDE_contractor_hourly_rate_l3288_328814

/-- Contractor's hourly rate calculation -/
theorem contractor_hourly_rate 
  (total_cost : ℝ) 
  (permit_cost : ℝ) 
  (contractor_hours : ℝ) 
  (inspector_rate_ratio : ℝ) :
  total_cost = 2950 →
  permit_cost = 250 →
  contractor_hours = 15 →
  inspector_rate_ratio = 0.2 →
  ∃ (contractor_rate : ℝ),
    contractor_rate = 150 ∧
    total_cost = permit_cost + contractor_hours * contractor_rate * (1 + inspector_rate_ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_hourly_rate_l3288_328814


namespace NUMINAMATH_CALUDE_factorial_sum_power_of_two_solutions_l3288_328807

def is_solution (a b c n : ℕ) : Prop :=
  Nat.factorial a + Nat.factorial b + Nat.factorial c = 2^n

theorem factorial_sum_power_of_two_solutions :
  ∀ a b c n : ℕ,
    is_solution a b c n ↔
      ((a, b, c) = (1, 1, 2) ∧ n = 2) ∨
      ((a, b, c) = (1, 1, 3) ∧ n = 3) ∨
      ((a, b, c) = (2, 3, 4) ∧ n = 5) ∨
      ((a, b, c) = (2, 3, 5) ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_power_of_two_solutions_l3288_328807


namespace NUMINAMATH_CALUDE_shirt_cost_l3288_328877

-- Define the number of $10 bills
def num_10_bills : ℕ := 2

-- Define the number of $20 bills
def num_20_bills : ℕ := num_10_bills + 1

-- Define the value of a $10 bill
def value_10_bill : ℕ := 10

-- Define the value of a $20 bill
def value_20_bill : ℕ := 20

-- Theorem: The cost of the shirt is $80
theorem shirt_cost : 
  num_10_bills * value_10_bill + num_20_bills * value_20_bill = 80 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3288_328877


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3288_328855

theorem infinitely_many_primes_of_form (m n : ℤ) : 
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ p ∈ S, Prime p ∧ ∃ m n : ℤ, p = m^2 + m*n + n^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3288_328855


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l3288_328841

/-- A binary sequence of length 6 -/
def BinarySeq := Fin 6 → Bool

/-- The set of all possible 6-digit binary sequences -/
def AllBinarySeqs : Set BinarySeq :=
  {seq | seq ∈ Set.univ}

/-- Two binary sequences differ by exactly one digit -/
def differByOne (seq1 seq2 : BinarySeq) : Prop :=
  ∃! i : Fin 6, seq1 i ≠ seq2 i

/-- A valid arrangement of binary sequences in an 8x8 grid -/
def ValidArrangement (arrangement : Fin 8 → Fin 8 → BinarySeq) : Prop :=
  (∀ i j, arrangement i j ∈ AllBinarySeqs) ∧
  (∀ i j, i + 1 < 8 → differByOne (arrangement i j) (arrangement (i + 1) j)) ∧
  (∀ i j, j + 1 < 8 → differByOne (arrangement i j) (arrangement i (j + 1))) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)

/-- The main theorem: a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ arrangement, ValidArrangement arrangement := by
  sorry


end NUMINAMATH_CALUDE_valid_arrangement_exists_l3288_328841


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3288_328816

/-- A geometric sequence with negative terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = -5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3288_328816


namespace NUMINAMATH_CALUDE_sequence_14th_term_is_9_l3288_328833

theorem sequence_14th_term_is_9 :
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by sorry

end NUMINAMATH_CALUDE_sequence_14th_term_is_9_l3288_328833


namespace NUMINAMATH_CALUDE_tom_rare_cards_l3288_328885

/-- The number of rare cards in Tom's deck -/
def rare_cards : ℕ := 19

/-- The number of uncommon cards in Tom's deck -/
def uncommon_cards : ℕ := 11

/-- The number of common cards in Tom's deck -/
def common_cards : ℕ := 30

/-- The cost of a rare card in dollars -/
def rare_cost : ℚ := 1

/-- The cost of an uncommon card in dollars -/
def uncommon_cost : ℚ := 1/2

/-- The cost of a common card in dollars -/
def common_cost : ℚ := 1/4

/-- The total cost of Tom's deck in dollars -/
def total_cost : ℚ := 32

theorem tom_rare_cards : 
  rare_cards * rare_cost + 
  uncommon_cards * uncommon_cost + 
  common_cards * common_cost = total_cost := by sorry

end NUMINAMATH_CALUDE_tom_rare_cards_l3288_328885


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l3288_328886

theorem washing_machine_capacity (pounds_per_machine : ℕ) (num_machines : ℕ) 
  (h1 : pounds_per_machine = 28) 
  (h2 : num_machines = 8) : 
  pounds_per_machine * num_machines = 224 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l3288_328886


namespace NUMINAMATH_CALUDE_half_minus_third_equals_sixth_l3288_328823

theorem half_minus_third_equals_sixth : (1/2 : ℚ) - (1/3 : ℚ) = 1/6 := by sorry

end NUMINAMATH_CALUDE_half_minus_third_equals_sixth_l3288_328823


namespace NUMINAMATH_CALUDE_function_value_at_two_l3288_328868

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 0, prove that f(2) = -16 -/
theorem function_value_at_two (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^5 + a*x^3 + b*x - 8
  f (-2) = 0 → f 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3288_328868


namespace NUMINAMATH_CALUDE_cube_coplanar_probability_l3288_328801

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices we choose -/
def chosen_vertices : ℕ := 4

/-- The number of ways to choose 4 vertices that lie in the same plane -/
def coplanar_choices : ℕ := 12

/-- The total number of ways to choose 4 vertices from 8 -/
def total_choices : ℕ := Nat.choose cube_vertices chosen_vertices

/-- The probability that 4 randomly chosen vertices of a cube lie in the same plane -/
theorem cube_coplanar_probability : 
  (coplanar_choices : ℚ) / total_choices = 6 / 35 := by sorry

end NUMINAMATH_CALUDE_cube_coplanar_probability_l3288_328801


namespace NUMINAMATH_CALUDE_increasing_order_x_y_z_l3288_328822

theorem increasing_order_x_y_z (x : ℝ) (hx : 1.1 < x ∧ x < 1.2) :
  x < x^x ∧ x^x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_y_z_l3288_328822


namespace NUMINAMATH_CALUDE_odd_function_parallelicity_l3288_328882

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (-x) = -f x

/-- A function has parallelicity if there exist two distinct points with parallel tangent lines -/
def HasParallelicity (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    DifferentiableAt ℝ f x₁ ∧ DifferentiableAt ℝ f x₂ ∧
    deriv f x₁ = deriv f x₂

/-- Theorem: Any odd function defined on (-∞,0)∪(0,+∞) has parallelicity -/
theorem odd_function_parallelicity (f : ℝ → ℝ) (hf : IsOdd f) : HasParallelicity f := by
  sorry


end NUMINAMATH_CALUDE_odd_function_parallelicity_l3288_328882


namespace NUMINAMATH_CALUDE_parabola_properties_l3288_328854

-- Define the parabola and its properties
def parabola (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ a < 0 ∧ -2 < m ∧ m < -1 ∧
  a * 1^2 + b * 1 + c = 0 ∧
  a * m^2 + b * m + c = 0

-- State the theorem
theorem parabola_properties (a b c m : ℝ) (h : parabola a b c m) :
  a * b * c > 0 ∧ a - b + c > 0 ∧ a * (m + 1) - b + c > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3288_328854


namespace NUMINAMATH_CALUDE_f_properties_l3288_328899

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, deriv f x > 0) ∧
  (∀ k, (∀ x, f (x^2) + f (k*x + 1) > 0) ↔ -2 < k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3288_328899


namespace NUMINAMATH_CALUDE_non_multiples_count_is_546_l3288_328829

/-- The count of three-digit numbers that are not multiples of 3 or 11 -/
def non_multiples_count : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_11 := (990 - 110) / 11 + 1
  let multiples_of_33 := (990 - 132) / 33 + 1
  total_three_digit - (multiples_of_3 + multiples_of_11 - multiples_of_33)

theorem non_multiples_count_is_546 : non_multiples_count = 546 := by
  sorry

end NUMINAMATH_CALUDE_non_multiples_count_is_546_l3288_328829


namespace NUMINAMATH_CALUDE_lunch_cost_distribution_l3288_328856

theorem lunch_cost_distribution (total_cost : ℕ) 
  (your_cost first_friend_extra second_friend_less third_friend_multiplier : ℕ) :
  total_cost = 100 ∧ 
  first_friend_extra = 15 ∧ 
  second_friend_less = 20 ∧ 
  third_friend_multiplier = 2 →
  ∃ (your_amount : ℕ),
    your_amount = 21 ∧
    your_amount + (your_amount + first_friend_extra) + 
    (your_amount - second_friend_less) + (your_amount * third_friend_multiplier) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_lunch_cost_distribution_l3288_328856


namespace NUMINAMATH_CALUDE_can_identification_theorem_l3288_328858

-- Define the type for our weighing results
inductive WeighResult
| Heavy
| Medium
| Light

def WeighSequence := List WeighResult

theorem can_identification_theorem (n : ℕ) (weights : Fin n → ℝ) 
  (h_n : n = 80) (h_distinct : ∀ i j : Fin n, i ≠ j → weights i ≠ weights j) :
  (∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 4) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) ∧ 
  (¬ ∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 3) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) := by
  sorry


end NUMINAMATH_CALUDE_can_identification_theorem_l3288_328858


namespace NUMINAMATH_CALUDE_happy_boys_count_l3288_328805

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) (happy_boys_exist : Prop) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 19 →
  total_girls = 41 →
  sad_girls = 4 →
  neutral_boys = 7 →
  happy_boys_exist →
  ∃ (happy_boys : ℕ), happy_boys = 6 ∧ 
    happy_boys + (sad_children - sad_girls) + neutral_boys = total_boys :=
by sorry

end NUMINAMATH_CALUDE_happy_boys_count_l3288_328805


namespace NUMINAMATH_CALUDE_f_6_equals_37_l3288_328897

def f : ℕ → ℤ
| 0 => 0  -- Arbitrary base case
| n + 1 =>
  if 1 ≤ n + 1 ∧ n + 1 ≤ 4 then f n - (n + 1)
  else if 5 ≤ n + 1 ∧ n + 1 ≤ 8 then f n + 2 * (n + 1)
  else f n * (n + 1)

theorem f_6_equals_37 (h : f 4 = 15) : f 6 = 37 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_37_l3288_328897


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3288_328875

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 80 - (1/4) * 80 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l3288_328875


namespace NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l3288_328894

/-- Represents the number of home runs scored by a team in each inning -/
structure HomeRuns :=
  (third : ℕ)
  (fifth : ℕ)
  (eighth : ℕ)

/-- Represents the number of home runs scored by the opposing team in each inning -/
structure OpponentHomeRuns :=
  (second : ℕ)
  (fifth : ℕ)

/-- The difference in home runs between the Cubs and the Cardinals -/
def homRunDifference (cubs : HomeRuns) (cardinals : OpponentHomeRuns) : ℕ :=
  (cubs.third + cubs.fifth + cubs.eighth) - (cardinals.second + cardinals.fifth)

theorem cubs_cardinals_home_run_difference :
  ∀ (cubs : HomeRuns) (cardinals : OpponentHomeRuns),
    cubs.third = 2 → cubs.fifth = 1 → cubs.eighth = 2 →
    cardinals.second = 1 → cardinals.fifth = 1 →
    homRunDifference cubs cardinals = 3 :=
by
  sorry

#check cubs_cardinals_home_run_difference

end NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l3288_328894


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3288_328880

theorem book_sale_loss_percentage (selling_price_loss : ℝ) (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) (loss_percentage : ℝ) :
  selling_price_loss = 450 →
  selling_price_gain = 550 →
  gain_percentage = 10 →
  (selling_price_gain = (100 + gain_percentage) / 100 * (100 / (100 + gain_percentage) * selling_price_gain)) →
  (loss_percentage = (((100 / (100 + gain_percentage) * selling_price_gain) - selling_price_loss) / 
    (100 / (100 + gain_percentage) * selling_price_gain)) * 100) →
  loss_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3288_328880


namespace NUMINAMATH_CALUDE_smallest_square_cover_l3288_328863

/-- The side length of the smallest square that can be covered by 3x4 rectangles -/
def minSquareSide : ℕ := 12

/-- The number of 3x4 rectangles needed to cover the square -/
def numRectangles : ℕ := 12

/-- The area of a 3x4 rectangle -/
def rectangleArea : ℕ := 3 * 4

theorem smallest_square_cover :
  (minSquareSide * minSquareSide) % rectangleArea = 0 ∧
  numRectangles * rectangleArea = minSquareSide * minSquareSide ∧
  ∀ n : ℕ, n < minSquareSide → (n * n) % rectangleArea ≠ 0 :=
by sorry

#check smallest_square_cover

end NUMINAMATH_CALUDE_smallest_square_cover_l3288_328863


namespace NUMINAMATH_CALUDE_solve_for_b_and_c_l3288_328852

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_for_b_and_c (a b c : ℝ) : 
  A a ≠ B b c →
  A a ∪ B b c = {-3, 4} →
  A a ∩ B b c = {-3} →
  b = 3 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_solve_for_b_and_c_l3288_328852


namespace NUMINAMATH_CALUDE_integer_solution_condition_non_negative_condition_l3288_328834

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + (3 + m)*x^2 - 12*x + 12

/-- Theorem for the first part of the problem -/
theorem integer_solution_condition (m : ℝ) :
  (∃ x : ℤ, f m x - f m (1 - x) + 4*x^3 = 0) ↔ (m = 8 ∨ m = 12) := by sorry

/-- Theorem for the second part of the problem -/
theorem non_negative_condition (m : ℝ) :
  (∀ x : ℝ, f m x ≥ 0) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_integer_solution_condition_non_negative_condition_l3288_328834


namespace NUMINAMATH_CALUDE_square_of_odd_is_sum_of_consecutive_integers_l3288_328838

theorem square_of_odd_is_sum_of_consecutive_integers :
  ∀ n : ℕ, n > 1 → Odd n → ∃ j : ℕ, n^2 = j + (j + 1) := by sorry

end NUMINAMATH_CALUDE_square_of_odd_is_sum_of_consecutive_integers_l3288_328838


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3288_328895

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 13 ∧ (7538 - x) % 14 = 0 ∧ ∀ y : ℕ, y < x → (7538 - y) % 14 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3288_328895


namespace NUMINAMATH_CALUDE_lucy_share_l3288_328815

/-- Proves that Lucy's share is $2000 given the conditions of the problem -/
theorem lucy_share (total : ℝ) (natalie_fraction : ℝ) (rick_fraction : ℝ) 
  (h_total : total = 10000)
  (h_natalie : natalie_fraction = 1/2)
  (h_rick : rick_fraction = 3/5) : 
  total * (1 - natalie_fraction) * (1 - rick_fraction) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lucy_share_l3288_328815


namespace NUMINAMATH_CALUDE_pencil_sharpening_hours_l3288_328881

/-- The number of times Jenine can sharpen a pencil before it runs out -/
def sharpen_times : ℕ := 5

/-- The number of pencils Jenine already has -/
def initial_pencils : ℕ := 10

/-- The total number of hours Jenine needs to write -/
def total_writing_hours : ℕ := 105

/-- The cost of a new pencil in dollars -/
def pencil_cost : ℕ := 2

/-- The amount Jenine needs to spend on more pencils in dollars -/
def additional_pencil_cost : ℕ := 8

/-- The number of hours of use Jenine gets from sharpening a pencil once -/
def hours_per_sharpen : ℚ := 1.5

theorem pencil_sharpening_hours :
  let total_pencils := initial_pencils + additional_pencil_cost / pencil_cost
  total_pencils * sharpen_times * hours_per_sharpen = total_writing_hours :=
by sorry

end NUMINAMATH_CALUDE_pencil_sharpening_hours_l3288_328881


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l3288_328803

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  area : ℝ

/-- Theorem stating the length of EG in the given trapezoid -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = 14)
  (h3 : t.area = 72)
  (h4 : t.EG = (((t.GH - t.EF) / 2) ^ 2 + (2 * t.area / (t.EF + t.GH)) ^ 2).sqrt) :
  t.EG = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l3288_328803


namespace NUMINAMATH_CALUDE_count_triplets_eq_30_l3288_328813

/-- Count of ordered triplets (a, b, c) of positive integers satisfying 30a + 50b + 70c ≤ 343 -/
def count_triplets : ℕ :=
  (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 30 * a + 50 * b + 70 * c ≤ 343)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 7) (Finset.range 5)))).card

theorem count_triplets_eq_30 : count_triplets = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_triplets_eq_30_l3288_328813


namespace NUMINAMATH_CALUDE_triangle_area_side_a_value_l3288_328890

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  cosA : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.cosA = 1/2 ∧ t.b * t.c = 3

-- Theorem 1: Area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2 : ℝ) * t.b * t.c * Real.sqrt (1 - t.cosA^2) = Real.sqrt 3 / 2 := by
  sorry

-- Theorem 2: Value of side a when c = 1
theorem side_a_value (t : Triangle) (h : triangle_conditions t) (h_c : t.c = 1) :
  t.a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_side_a_value_l3288_328890


namespace NUMINAMATH_CALUDE_area_of_triangle_DBC_l3288_328818

/-- Given points A, B, C, D, and E in a coordinate plane, prove that the area of triangle DBC is 20 -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = (B.1 + (C.1 - B.1) / 3, B.2 + (C.2 - B.2) / 3) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DBC_l3288_328818


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3288_328876

theorem inequality_solution_set (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  (2 * x) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ioc 0 (1/5) ∪ Set.Ioc 2 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3288_328876


namespace NUMINAMATH_CALUDE_aria_cookie_expense_is_2356_l3288_328849

/-- The amount Aria spent on cookies in March -/
def aria_cookie_expense : ℕ :=
  let cookies_per_day : ℕ := 4
  let cost_per_cookie : ℕ := 19
  let days_in_march : ℕ := 31
  cookies_per_day * cost_per_cookie * days_in_march

/-- Theorem stating that Aria spent 2356 dollars on cookies in March -/
theorem aria_cookie_expense_is_2356 : aria_cookie_expense = 2356 := by
  sorry

end NUMINAMATH_CALUDE_aria_cookie_expense_is_2356_l3288_328849


namespace NUMINAMATH_CALUDE_ages_solution_l3288_328865

/-- Represents the ages of Henry, Jill, and Alex -/
structure Ages where
  henry : ℕ
  jill : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.henry + ages.jill + ages.alex = 90 ∧
  ages.henry - 5 = 2 * (ages.jill - 5) ∧
  ages.henry + ages.jill - 10 = ages.alex

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.henry = 32 ∧ ages.jill = 18 ∧ ages.alex = 40 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l3288_328865


namespace NUMINAMATH_CALUDE_quadratic_root_implies_positive_triangle_l3288_328817

theorem quadratic_root_implies_positive_triangle (a b c : ℝ) 
  (h_root : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ Complex.I * Complex.I = -1 ∧ 
    (α + Complex.I * β) ^ 2 - (a + b + c) * (α + Complex.I * β) + (a * b + b * c + c * a) = 0) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧ 
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧ 
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_positive_triangle_l3288_328817


namespace NUMINAMATH_CALUDE_d_must_be_positive_l3288_328898

theorem d_must_be_positive
  (a b c d e f : ℤ)
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : c < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  d > 0 := by
sorry

end NUMINAMATH_CALUDE_d_must_be_positive_l3288_328898


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l3288_328839

/-- Calculates the total time for a round trip by boat given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 15)
  (h2 : stream_speed = 3)
  (h3 : distance = 180)
  : (distance / (boat_speed + stream_speed)) + (distance / (boat_speed - stream_speed)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_round_trip_time_l3288_328839


namespace NUMINAMATH_CALUDE_power_of_product_l3288_328845

theorem power_of_product (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3288_328845


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_one_l3288_328827

theorem sqrt_meaningful_iff_geq_one (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_geq_one_l3288_328827
