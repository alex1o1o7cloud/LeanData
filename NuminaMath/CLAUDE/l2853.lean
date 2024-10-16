import Mathlib

namespace NUMINAMATH_CALUDE_can_distribution_properties_l2853_285330

/-- Represents the distribution of cans across bags -/
structure CanDistribution where
  total_cans : ℕ
  num_bags : ℕ
  first_bags_limit : ℕ
  last_bags_limit : ℕ

/-- Calculates the number of cans in each of the last bags -/
def cans_in_last_bags (d : CanDistribution) : ℕ :=
  let cans_in_first_bags := d.first_bags_limit * (d.num_bags / 2)
  let remaining_cans := d.total_cans - cans_in_first_bags
  remaining_cans / (d.num_bags / 2)

/-- Calculates the difference between cans in first and last bag -/
def cans_difference (d : CanDistribution) : ℕ :=
  d.first_bags_limit - cans_in_last_bags d

/-- Theorem stating the properties of the can distribution -/
theorem can_distribution_properties (d : CanDistribution) 
    (h1 : d.total_cans = 200)
    (h2 : d.num_bags = 6)
    (h3 : d.first_bags_limit = 40)
    (h4 : d.last_bags_limit = 30) :
    cans_in_last_bags d = 26 ∧ cans_difference d = 14 := by
  sorry

#eval cans_in_last_bags { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }
#eval cans_difference { total_cans := 200, num_bags := 6, first_bags_limit := 40, last_bags_limit := 30 }

end NUMINAMATH_CALUDE_can_distribution_properties_l2853_285330


namespace NUMINAMATH_CALUDE_stating_max_equations_theorem_l2853_285316

/-- 
Represents the maximum number of equations without real roots 
that the first player can guarantee in a game with n equations.
-/
def max_equations_without_real_roots (n : ℕ) : ℕ :=
  if n % 2 = 0 then 0 else (n + 1) / 2

/-- 
Theorem stating the maximum number of equations without real roots 
that the first player can guarantee in the game.
-/
theorem max_equations_theorem (n : ℕ) :
  max_equations_without_real_roots n = 
    if n % 2 = 0 then 0 else (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_equations_theorem_l2853_285316


namespace NUMINAMATH_CALUDE_least_integer_in_ratio_l2853_285329

theorem least_integer_in_ratio (a b c : ℕ+) : 
  (a : ℚ) + (b : ℚ) + (c : ℚ) = 90 →
  (b : ℚ) = 2 * (a : ℚ) →
  (c : ℚ) = 5 * (a : ℚ) →
  (a : ℚ) = 45 / 4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_in_ratio_l2853_285329


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2853_285355

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2853_285355


namespace NUMINAMATH_CALUDE_first_year_rate_is_two_percent_l2853_285357

/-- Given an initial amount, time period, second year interest rate, and final amount,
    calculate the first year interest rate. -/
def calculate_first_year_rate (initial_amount : ℝ) (time_period : ℕ) 
                               (second_year_rate : ℝ) (final_amount : ℝ) : ℝ :=
  sorry

/-- Theorem: Given the specific conditions, the first year interest rate is 2% -/
theorem first_year_rate_is_two_percent :
  let initial_amount : ℝ := 5000
  let time_period : ℕ := 2
  let second_year_rate : ℝ := 0.03
  let final_amount : ℝ := 5253
  calculate_first_year_rate initial_amount time_period second_year_rate final_amount = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_first_year_rate_is_two_percent_l2853_285357


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l2853_285350

/-- Given a sphere of radius 6 cm and a right circular cylinder with equal height and diameter,
    if their surface areas are equal, then the height and diameter of the cylinder are both 12 cm. -/
theorem cylinder_dimensions (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ) :
  r_sphere = 6 →
  h_cylinder = 2 * r_cylinder →
  4 * Real.pi * r_sphere^2 = 2 * Real.pi * r_cylinder * h_cylinder →
  h_cylinder = 12 ∧ (2 * r_cylinder) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_l2853_285350


namespace NUMINAMATH_CALUDE_trig_fraction_equals_four_fifths_l2853_285321

theorem trig_fraction_equals_four_fifths (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equals_four_fifths_l2853_285321


namespace NUMINAMATH_CALUDE_total_candidates_l2853_285391

theorem total_candidates (avg_all : ℝ) (avg_passed : ℝ) (avg_failed : ℝ) (num_passed : ℕ) :
  avg_all = 35 →
  avg_passed = 39 →
  avg_failed = 15 →
  num_passed = 100 →
  ∃ (total : ℕ), total = 120 ∧ 
    (avg_all * total : ℝ) = avg_passed * num_passed + avg_failed * (total - num_passed) :=
by sorry

end NUMINAMATH_CALUDE_total_candidates_l2853_285391


namespace NUMINAMATH_CALUDE_time_per_furniture_piece_l2853_285360

theorem time_per_furniture_piece (chairs tables total_time : ℕ) 
  (h1 : chairs = 7)
  (h2 : tables = 3)
  (h3 : total_time = 40) : 
  total_time / (chairs + tables) = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_per_furniture_piece_l2853_285360


namespace NUMINAMATH_CALUDE_correct_book_prices_optimal_purchasing_plan_l2853_285336

-- Define the prices of the books
def price_analects : ℕ := 20
def price_standards : ℕ := 15

-- Define the quantities in the given information
def qty_analects1 : ℕ := 40
def qty_standards1 : ℕ := 30
def total_cost1 : ℕ := 1250

def qty_analects2 : ℕ := 50
def qty_standards2 : ℕ := 20
def total_cost2 : ℕ := 1300

-- Define the optimal quantities for the purchasing plan
def optimal_qty_analects : ℕ := 34
def optimal_qty_standards : ℕ := 66
def optimal_total_cost : ℕ := 1670

-- Theorem to prove the correctness of the book prices
theorem correct_book_prices :
  qty_analects1 * price_analects + qty_standards1 * price_standards = total_cost1 ∧
  qty_analects2 * price_analects + qty_standards2 * price_standards = total_cost2 :=
sorry

-- Theorem to prove the optimal purchasing plan
theorem optimal_purchasing_plan :
  optimal_qty_analects + optimal_qty_standards = 100 ∧
  optimal_qty_standards ≤ 2 * optimal_qty_analects ∧
  optimal_qty_analects * price_analects + optimal_qty_standards * price_standards = optimal_total_cost ∧
  ∀ (a s : ℕ), a + s = 100 → s ≤ 2 * a →
    a * price_analects + s * price_standards ≥ optimal_total_cost :=
sorry

end NUMINAMATH_CALUDE_correct_book_prices_optimal_purchasing_plan_l2853_285336


namespace NUMINAMATH_CALUDE_product_in_base5_l2853_285337

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else convert (m / 5) ((m % 5) :: acc)
    convert n []

theorem product_in_base5 :
  let a := [4, 1, 3, 2]  -- 2314₅ in reverse order
  let b := [3, 2]        -- 23₅ in reverse order
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b) = [2, 3, 3, 8, 6] :=
by sorry

end NUMINAMATH_CALUDE_product_in_base5_l2853_285337


namespace NUMINAMATH_CALUDE_union_of_sets_l2853_285392

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2853_285392


namespace NUMINAMATH_CALUDE_regression_y_change_l2853_285345

/-- Represents a simple linear regression equation -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- Represents the change in y for a unit change in x -/
def yChange (reg : LinearRegression) : ℝ := -reg.slope

theorem regression_y_change (reg : LinearRegression) 
  (h : reg = { intercept := 3, slope := 5 }) : 
  yChange reg = -5 := by sorry

end NUMINAMATH_CALUDE_regression_y_change_l2853_285345


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_l2853_285314

/-- The perimeter of a regular octagon with side length 2 is 16 -/
theorem regular_octagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 2 → 
  (8 : ℝ) * side_length = 16 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_l2853_285314


namespace NUMINAMATH_CALUDE_range_of_a_for_local_max_l2853_285375

noncomputable def f (a b x : ℝ) : ℝ := Real.log x + a * x^2 + b * x

theorem range_of_a_for_local_max (a b : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f a b x ≤ f a b 1) →
  a < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_local_max_l2853_285375


namespace NUMINAMATH_CALUDE_gcd_of_sum_is_222_l2853_285302

def is_consecutive_even (a b c d : ℕ) : Prop :=
  b = a + 2 ∧ c = a + 4 ∧ d = a + 6 ∧ a % 2 = 0

def e_sum (a d : ℕ) : ℕ := a + d

def abcde (a b c d e : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * d + e

def edcba (a b c d e : ℕ) : ℕ := 10000 * e + 1000 * d + 100 * c + 10 * b + a

theorem gcd_of_sum_is_222 (a b c d : ℕ) (h : is_consecutive_even a b c d) :
  Nat.gcd (abcde a b c d (e_sum a d) + edcba a b c d (e_sum a d))
          (abcde (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2)) +
           edcba (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2))) = 222 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_sum_is_222_l2853_285302


namespace NUMINAMATH_CALUDE_younger_brother_age_after_30_years_l2853_285348

/-- Given two brothers with an age difference of 10 years, where the elder is 40 years old now,
    prove that the younger brother will be 60 years old after 30 years. -/
theorem younger_brother_age_after_30_years
  (age_difference : ℕ)
  (elder_brother_current_age : ℕ)
  (years_from_now : ℕ)
  (h1 : age_difference = 10)
  (h2 : elder_brother_current_age = 40)
  (h3 : years_from_now = 30) :
  elder_brother_current_age - age_difference + years_from_now = 60 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_age_after_30_years_l2853_285348


namespace NUMINAMATH_CALUDE_correct_writers_l2853_285346

/-- Represents the group of students and their writing task -/
structure StudentGroup where
  total : Nat
  cat_writers : Nat
  rat_writers : Nat
  crocodile_writers : Nat
  correct_cat : Nat
  correct_rat : Nat

/-- Theorem stating the number of students who wrote their word correctly -/
theorem correct_writers (group : StudentGroup) 
  (h1 : group.total = 50)
  (h2 : group.cat_writers = 10)
  (h3 : group.rat_writers = 18)
  (h4 : group.crocodile_writers = group.total - group.cat_writers - group.rat_writers)
  (h5 : group.correct_cat = 15)
  (h6 : group.correct_rat = 15) :
  group.correct_cat + group.correct_rat - (group.cat_writers + group.rat_writers) + group.crocodile_writers = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_writers_l2853_285346


namespace NUMINAMATH_CALUDE_fifth_score_calculation_l2853_285389

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 80) (h5 : avg = 76.6) :
  ∃ (s5 : ℕ), s5 = 95 ∧ (s1 + s2 + s3 + s4 + s5 : ℚ) / 5 = avg :=
sorry

end NUMINAMATH_CALUDE_fifth_score_calculation_l2853_285389


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l2853_285331

/-- A function that generates all valid eight-digit numbers using the digits 4, 0, 2, 6 twice each -/
def validNumbers : List Nat := sorry

/-- The largest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def largestNumber : Nat := sorry

/-- The smallest eight-digit number that can be formed using the digits 4, 0, 2, 6 twice each -/
def smallestNumber : Nat := sorry

/-- Theorem stating that the sum of the largest and smallest valid numbers is 86,466,666 -/
theorem sum_of_largest_and_smallest :
  largestNumber + smallestNumber = 86466666 := by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_l2853_285331


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2853_285388

theorem lcm_factor_proof (A B : ℕ+) (Y : ℕ+) : 
  Nat.gcd A B = 63 →
  Nat.lcm A B = 63 * 11 * Y →
  A = 1071 →
  Y = 17 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2853_285388


namespace NUMINAMATH_CALUDE_base4_1302_equals_base5_424_l2853_285342

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

theorem base4_1302_equals_base5_424 :
  base10ToBase5 (base4ToBase10 [2, 0, 3, 1]) = [4, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_base4_1302_equals_base5_424_l2853_285342


namespace NUMINAMATH_CALUDE_oak_trees_in_park_l2853_285332

theorem oak_trees_in_park (current_trees : ℕ) 
  (h1 : current_trees + 4 = 9) : current_trees = 5 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_in_park_l2853_285332


namespace NUMINAMATH_CALUDE_parallelogram_base_proof_l2853_285325

/-- The area of a parallelogram -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_base_proof (area height : ℝ) (h1 : area = 96) (h2 : height = 8) :
  parallelogram_area (area / height) height = area → area / height = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_proof_l2853_285325


namespace NUMINAMATH_CALUDE_production_growth_equation_l2853_285324

/-- Represents the production growth scenario of Dream Enterprise --/
def production_growth_scenario (initial_value : ℝ) (growth_rate : ℝ) : Prop :=
  let feb_value := initial_value * (1 + growth_rate)
  let mar_value := initial_value * (1 + growth_rate)^2
  mar_value - feb_value = 220000

/-- Theorem stating the correct equation for the production growth scenario --/
theorem production_growth_equation :
  production_growth_scenario 2000000 x ↔ 2000000 * (1 + x)^2 - 2000000 * (1 + x) = 220000 :=
sorry

end NUMINAMATH_CALUDE_production_growth_equation_l2853_285324


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_irrational_l2853_285303

-- Define the irrationality of √6
def sqrt6_irrational : Irrational (Real.sqrt 6) := sorry

-- Theorem to prove
theorem sqrt_two_thirds_irrational : Irrational (Real.sqrt (2/3)) := by sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_irrational_l2853_285303


namespace NUMINAMATH_CALUDE_train_distance_l2853_285399

/-- Calculates the distance traveled by a train given its speed and time. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a train traveling at 85 km/h for 4 hours covers 340 km. -/
theorem train_distance : distance_traveled 85 4 = 340 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2853_285399


namespace NUMINAMATH_CALUDE_investments_sum_to_22000_l2853_285363

/-- Represents the initial investment amounts of five individuals --/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ
  alok : ℝ
  harshit : ℝ

/-- Calculates the total sum of investments --/
def total_investment (i : Investments) : ℝ :=
  i.raghu + i.trishul + i.vishal + i.alok + i.harshit

/-- Theorem stating that the investments satisfy the given conditions and sum to 22000 --/
theorem investments_sum_to_22000 :
  ∃ (i : Investments),
    i.trishul = 0.9 * i.raghu ∧
    i.vishal = 1.1 * i.trishul ∧
    i.alok = 1.15 * i.trishul ∧
    i.harshit = 0.95 * i.vishal ∧
    total_investment i = 22000 :=
  sorry

end NUMINAMATH_CALUDE_investments_sum_to_22000_l2853_285363


namespace NUMINAMATH_CALUDE_least_sum_with_equation_l2853_285312

theorem least_sum_with_equation (x y z : ℕ+) 
  (eq : 4 * x.val = 5 * y.val) 
  (least_sum : ∀ (a b c : ℕ+), 4 * a.val = 5 * b.val → a.val + b.val + c.val ≥ x.val + y.val + z.val) 
  (sum_37 : x.val + y.val + z.val = 37) : 
  z.val = 28 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_equation_l2853_285312


namespace NUMINAMATH_CALUDE_roberts_extra_chocolates_l2853_285344

theorem roberts_extra_chocolates (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 9) 
  (h2 : nickel_chocolates = 2) : 
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end NUMINAMATH_CALUDE_roberts_extra_chocolates_l2853_285344


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l2853_285385

/-- Given a triangle with inradius 2.0 cm and area 28 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter_from_inradius_and_area :
  ∀ (p : ℝ), 
    (2.0 : ℝ) * p / 2 = 28 →
    p = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_inradius_and_area_l2853_285385


namespace NUMINAMATH_CALUDE_alyssa_total_games_l2853_285390

/-- The total number of soccer games Alyssa will attend -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Proof that Alyssa will attend 39 soccer games in total -/
theorem alyssa_total_games : 
  ∃ (this_year last_year next_year : ℕ),
    this_year = 11 ∧ 
    last_year = 13 ∧ 
    next_year = 15 ∧ 
    total_games this_year last_year next_year = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l2853_285390


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2853_285365

theorem square_root_equation_solution (A C : ℝ) (hA : A ≥ 0) (hC : C ≥ 0) :
  ∃ x : ℝ, x > 0 ∧
    Real.sqrt (2 + A * C + 2 * C * x) + Real.sqrt (A * C - 2 + 2 * A * x) =
    Real.sqrt (2 * (A + C) * x + 2 * A * C) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2853_285365


namespace NUMINAMATH_CALUDE_third_grade_trees_l2853_285311

theorem third_grade_trees (total_students : ℕ) (total_trees : ℕ) 
  (trees_per_third : ℕ) (trees_per_fourth : ℕ) (trees_per_fifth : ℚ) :
  total_students = 100 →
  total_trees = 566 →
  trees_per_third = 4 →
  trees_per_fourth = 5 →
  trees_per_fifth = 13/2 →
  ∃ (third_students fourth_students fifth_students : ℕ),
    third_students = fourth_students ∧
    third_students + fourth_students + fifth_students = total_students ∧
    third_students * trees_per_third + fourth_students * trees_per_fourth + 
      (fifth_students : ℚ) * trees_per_fifth = total_trees ∧
    third_students * trees_per_third = 84 :=
by
  sorry

#check third_grade_trees

end NUMINAMATH_CALUDE_third_grade_trees_l2853_285311


namespace NUMINAMATH_CALUDE_charity_event_probability_l2853_285382

/-- The number of students participating in the charity event -/
def num_students : ℕ := 4

/-- The number of days students can choose from (Saturday and Sunday) -/
def num_days : ℕ := 2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := num_days ^ num_students

/-- The number of outcomes where students participate on both days -/
def both_days_outcomes : ℕ := total_outcomes - num_days

/-- The probability of students participating on both days -/
def probability_both_days : ℚ := both_days_outcomes / total_outcomes

theorem charity_event_probability : probability_both_days = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_probability_l2853_285382


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l2853_285379

/-- Represents the pet shop inventory and pricing --/
structure PetShop where
  num_puppies : ℕ
  puppy_price : ℕ
  kitten_price : ℕ
  total_value : ℕ

/-- Calculates the number of kittens in the pet shop --/
def num_kittens (shop : PetShop) : ℕ :=
  (shop.total_value - shop.num_puppies * shop.puppy_price) / shop.kitten_price

/-- Theorem stating that the number of kittens in the given pet shop is 4 --/
theorem pet_shop_kittens :
  let shop : PetShop := {
    num_puppies := 2,
    puppy_price := 20,
    kitten_price := 15,
    total_value := 100
  }
  num_kittens shop = 4 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l2853_285379


namespace NUMINAMATH_CALUDE_minnows_per_prize_bowl_l2853_285334

theorem minnows_per_prize_bowl (total_minnows : ℕ) (total_players : ℕ) (winner_percentage : ℚ) (leftover_minnows : ℕ) :
  total_minnows = 600 →
  total_players = 800 →
  winner_percentage = 15 / 100 →
  leftover_minnows = 240 →
  (total_minnows - leftover_minnows) / (total_players * winner_percentage) = 3 :=
by sorry

end NUMINAMATH_CALUDE_minnows_per_prize_bowl_l2853_285334


namespace NUMINAMATH_CALUDE_inverse_proposition_true_l2853_285317

theorem inverse_proposition_true : ∃ x y : ℝ, (x ≤ y ∧ x ≤ |y|) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_true_l2853_285317


namespace NUMINAMATH_CALUDE_inequality_proof_l2853_285341

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8/(x*y) + y^2 ≥ 8 ∧
  (x^2 + 8/(x*y) + y^2 = 8 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2853_285341


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l2853_285369

/-- The total amount Sara spent on movies -/
def total_spent : ℚ :=
  let theater_ticket_price : ℚ := 10.62
  let theater_ticket_count : ℕ := 2
  let rented_movie_price : ℚ := 1.59
  let bought_movie_price : ℚ := 13.95
  theater_ticket_price * theater_ticket_count + rented_movie_price + bought_movie_price

/-- Theorem stating that Sara spent $36.78 on movies -/
theorem sara_movie_expenses : total_spent = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l2853_285369


namespace NUMINAMATH_CALUDE_sequence_properties_l2853_285398

def a (n : ℕ+) : ℤ := n * (n - 8) - 20

theorem sequence_properties :
  (∃ (k : ℕ), k = 9 ∧ ∀ n : ℕ+, a n < 0 ↔ n.val ≤ k) ∧
  (∀ n : ℕ+, n ≥ 4 → a (n + 1) > a n) ∧
  (∀ n : ℕ+, a n ≥ a 4 ∧ a 4 = -36) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2853_285398


namespace NUMINAMATH_CALUDE_harry_scores_l2853_285308

/-- Harry's basketball scores -/
def first_10_games : List ℕ := [9, 5, 4, 7, 11, 4, 2, 8, 5, 7]

/-- Sum of scores in the first 10 games -/
def sum_first_10 : ℕ := first_10_games.sum

/-- Proposition: Harry's 11th and 12th game scores -/
theorem harry_scores : ∃ (score_11 score_12 : ℕ),
  (score_11 < 15 ∧ score_12 < 15) ∧
  (sum_first_10 + score_11) % 11 = 0 ∧
  (sum_first_10 + score_11 + score_12) % 12 = 0 ∧
  score_11 * score_12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_harry_scores_l2853_285308


namespace NUMINAMATH_CALUDE_smallest_positive_t_value_l2853_285351

theorem smallest_positive_t_value (p q r s t : ℤ) : 
  (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) →
  t > 0 →
  (∀ t' : ℤ, t' > 0 ∧ (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t' = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) → t' ≥ t) →
  t = 72 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_t_value_l2853_285351


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2853_285376

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Area of triangle ABC
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15 ∧
  -- Relationship between b and c
  b - c = 2 ∧
  -- Given cosine of A
  Real.cos A = -(1/4) →
  -- Conclusions
  a = 8 ∧
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2853_285376


namespace NUMINAMATH_CALUDE_g_difference_l2853_285364

/-- The function g defined as g(n) = n^3 + 3n^2 + 3n + 1 -/
def g (n : ℝ) : ℝ := n^3 + 3*n^2 + 3*n + 1

/-- Theorem stating that g(s) - g(s-2) = 6s^2 + 2 for any real number s -/
theorem g_difference (s : ℝ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2853_285364


namespace NUMINAMATH_CALUDE_group_capacity_for_given_conditions_l2853_285335

/-- The capacity of each group in a systematic sampling method -/
def group_capacity (population : ℕ) (sample_size : ℕ) : ℕ :=
  (population - (population % sample_size)) / sample_size

/-- Theorem: The capacity of each group is 25 for the given conditions -/
theorem group_capacity_for_given_conditions :
  group_capacity 5008 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_group_capacity_for_given_conditions_l2853_285335


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2853_285377

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define set N
def N : Set ℝ := {x | ∃ a : ℝ, x = 2*a^2 - 4*a + 1}

-- Theorem statement
theorem M_intersect_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l2853_285377


namespace NUMINAMATH_CALUDE_set_equality_l2853_285305

def set_a : Set ℕ := {x : ℕ | 2 * x + 3 ≥ 3 * x}
def set_b : Set ℕ := {0, 1, 2, 3}

theorem set_equality : set_a = set_b := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2853_285305


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2853_285366

theorem sum_of_decimals : (4.3 : ℝ) + 3.88 = 8.18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2853_285366


namespace NUMINAMATH_CALUDE_megan_folders_l2853_285368

/-- Calculates the number of full folders given the initial number of files, 
    number of deleted files, and number of files per folder. -/
def fullFolders (initialFiles : ℕ) (deletedFiles : ℕ) (filesPerFolder : ℕ) : ℕ :=
  ((initialFiles - deletedFiles) / filesPerFolder : ℕ)

/-- Proves that Megan ends up with 15 full folders given the initial conditions. -/
theorem megan_folders : fullFolders 256 67 12 = 15 := by
  sorry

#eval fullFolders 256 67 12

end NUMINAMATH_CALUDE_megan_folders_l2853_285368


namespace NUMINAMATH_CALUDE_value_of_n_l2853_285301

theorem value_of_n : ∃ n : ℕ, 5^3 - 7 = 2^2 + n ∧ n = 114 := by sorry

end NUMINAMATH_CALUDE_value_of_n_l2853_285301


namespace NUMINAMATH_CALUDE_vacation_payment_difference_is_zero_l2853_285304

/-- Represents the vacation expenses and payments of four friends -/
structure VacationExpenses where
  alice_paid : ℝ
  bob_paid : ℝ
  charlie_paid : ℝ
  donna_paid : ℝ
  alice_to_charlie : ℝ
  bob_to_donna : ℝ

/-- Theorem stating that the difference between Alice's payment to Charlie
    and Bob's payment to Donna is zero, given the vacation expenses -/
theorem vacation_payment_difference_is_zero
  (expenses : VacationExpenses)
  (h1 : expenses.alice_paid = 90)
  (h2 : expenses.bob_paid = 150)
  (h3 : expenses.charlie_paid = 120)
  (h4 : expenses.donna_paid = 240)
  (h5 : expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid = 600)
  (h6 : (expenses.alice_paid + expenses.bob_paid + expenses.charlie_paid + expenses.donna_paid) / 4 = 150)
  (h7 : expenses.alice_to_charlie = 150 - expenses.alice_paid)
  (h8 : expenses.bob_to_donna = 150 - expenses.bob_paid)
  : expenses.alice_to_charlie - expenses.bob_to_donna = 0 := by
  sorry

#check vacation_payment_difference_is_zero

end NUMINAMATH_CALUDE_vacation_payment_difference_is_zero_l2853_285304


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2853_285347

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2853_285347


namespace NUMINAMATH_CALUDE_dana_beth_same_money_l2853_285340

-- Define the set of individuals
inductive Person : Type
  | Abby : Person
  | Beth : Person
  | Cindy : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q ∨ (p = Person.Dana ∧ q = Person.Beth) ∨ (p = Person.Beth ∧ q = Person.Dana)
axiom abby_more_than_cindy : money Person.Abby > money Person.Cindy
axiom beth_less_than_eve : money Person.Beth < money Person.Eve
axiom beth_more_than_dana : money Person.Beth > money Person.Dana
axiom abby_less_than_dana : money Person.Abby < money Person.Dana
axiom dana_not_most : ∃ (p : Person), money p > money Person.Dana
axiom cindy_more_than_beth : money Person.Cindy > money Person.Beth

-- Theorem to prove
theorem dana_beth_same_money : money Person.Dana = money Person.Beth :=
  sorry

end NUMINAMATH_CALUDE_dana_beth_same_money_l2853_285340


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2853_285384

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2853_285384


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l2853_285322

/-- Proves that adding a specific amount of pure salt to a given salt solution results in the desired concentration -/
theorem salt_solution_concentration 
  (initial_weight : Real) 
  (initial_concentration : Real) 
  (added_salt : Real) 
  (final_concentration : Real) : 
  initial_weight = 100 ∧ 
  initial_concentration = 0.1 ∧ 
  added_salt = 28.571428571428573 ∧ 
  final_concentration = 0.3 →
  (initial_concentration * initial_weight + added_salt) / (initial_weight + added_salt) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l2853_285322


namespace NUMINAMATH_CALUDE_male_cousins_count_l2853_285354

/-- Represents the Martin family structure -/
structure MartinFamily where
  michael_sisters : ℕ
  michael_brothers : ℕ
  total_cousins : ℕ

/-- The number of male cousins counted by each female cousin in the Martin family -/
def male_cousins_per_female (family : MartinFamily) : ℕ :=
  family.michael_brothers + 1

/-- Theorem stating the number of male cousins counted by each female cousin -/
theorem male_cousins_count (family : MartinFamily) 
  (h1 : family.michael_sisters = 4)
  (h2 : family.michael_brothers = 6)
  (h3 : family.total_cousins = family.michael_sisters + family.michael_brothers + 2) 
  (h4 : ∃ n : ℕ, 2 * n = family.total_cousins) :
  male_cousins_per_female family = 8 := by
  sorry

#eval male_cousins_per_female { michael_sisters := 4, michael_brothers := 6, total_cousins := 14 }

end NUMINAMATH_CALUDE_male_cousins_count_l2853_285354


namespace NUMINAMATH_CALUDE_alex_painting_time_l2853_285372

/-- Given Jose's painting rate and the combined painting rate of Jose and Alex,
    calculate Alex's individual painting rate. -/
theorem alex_painting_time (jose_time : ℝ) (combined_time : ℝ) (alex_time : ℝ) : 
  jose_time = 7 → combined_time = 7 / 3 → alex_time = 7 / 2 := by
  sorry

#check alex_painting_time

end NUMINAMATH_CALUDE_alex_painting_time_l2853_285372


namespace NUMINAMATH_CALUDE_f_image_is_closed_interval_l2853_285362

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- Define the domain
def domain : Set ℝ := Set.Ioc 2 5

-- Theorem statement
theorem f_image_is_closed_interval :
  Set.image f domain = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_f_image_is_closed_interval_l2853_285362


namespace NUMINAMATH_CALUDE_set_A_is_correct_l2853_285313

-- Define the set A
def A : Set ℝ := {x : ℝ | x = -3 ∨ x = -1/2 ∨ x = 1/3 ∨ x = 2}

-- Define the property that if a ∈ A, then (1+a)/(1-a) ∈ A
def closure_property (S : Set ℝ) : Prop :=
  ∀ a ∈ S, (1 + a) / (1 - a) ∈ S

-- Theorem statement
theorem set_A_is_correct :
  -3 ∈ A ∧ closure_property A → A = {-3, -1/2, 1/3, 2} := by sorry

end NUMINAMATH_CALUDE_set_A_is_correct_l2853_285313


namespace NUMINAMATH_CALUDE_intersection_sum_l2853_285386

theorem intersection_sum (c d : ℝ) : 
  (2 = (1/5) * 3 + c) → 
  (3 = (1/5) * 2 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2853_285386


namespace NUMINAMATH_CALUDE_wades_food_truck_l2853_285380

/-- Wade's hot dog food truck problem -/
theorem wades_food_truck (tips_per_customer : ℚ) 
  (friday_customers sunday_customers : ℕ) (total_tips : ℚ) :
  tips_per_customer = 2 →
  friday_customers = 28 →
  sunday_customers = 36 →
  total_tips = 296 →
  let saturday_customers := (total_tips - tips_per_customer * (friday_customers + sunday_customers)) / tips_per_customer
  (saturday_customers : ℚ) / friday_customers = 3 := by
  sorry

end NUMINAMATH_CALUDE_wades_food_truck_l2853_285380


namespace NUMINAMATH_CALUDE_birdseed_mix_cost_l2853_285381

theorem birdseed_mix_cost (millet_weight : ℝ) (millet_cost : ℝ) (sunflower_weight : ℝ) (mixture_cost : ℝ) :
  millet_weight = 100 →
  millet_cost = 0.60 →
  sunflower_weight = 25 →
  mixture_cost = 0.70 →
  let total_weight := millet_weight + sunflower_weight
  let total_cost := mixture_cost * total_weight
  let millet_total_cost := millet_weight * millet_cost
  let sunflower_total_cost := total_cost - millet_total_cost
  sunflower_total_cost / sunflower_weight = 1.10 := by
sorry

end NUMINAMATH_CALUDE_birdseed_mix_cost_l2853_285381


namespace NUMINAMATH_CALUDE_license_plate_count_l2853_285396

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of license plate combinations with the given conditions -/
def license_plate_combinations : ℕ :=
  alphabet_size *  -- Choose the repeated letter
  (alphabet_size - 1).choose 2 *  -- Choose the other two distinct letters
  letter_positions.choose 2 *  -- Arrange the repeated letters
  2 *  -- Arrange the remaining two letters
  digit_count *  -- Choose the digit to repeat
  digit_positions.choose 2 *  -- Choose positions for the repeated digit
  (digit_count - 1)  -- Choose the second, different digit

/-- Theorem stating the number of possible license plate combinations -/
theorem license_plate_count : license_plate_combinations = 4212000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2853_285396


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l2853_285315

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -3/7 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l2853_285315


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2853_285306

/-- Proves that given a jogger running at 9 kmph, 230 meters ahead of a 120-meter long train,
    if the train passes the jogger in 35 seconds, then the speed of the train is 19 kmph. -/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 →
  initial_distance = 230 →
  train_length = 120 →
  passing_time = 35 / 3600 →
  ∃ (train_speed : ℝ), train_speed = 19 ∧
    (initial_distance + train_length) / passing_time = train_speed - jogger_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2853_285306


namespace NUMINAMATH_CALUDE_substitution_remainder_l2853_285359

/-- Number of players in a soccer team --/
def total_players : ℕ := 22

/-- Number of starting players --/
def starting_players : ℕ := 11

/-- Number of substitute players --/
def substitute_players : ℕ := 11

/-- Maximum number of substitutions allowed --/
def max_substitutions : ℕ := 4

/-- Function to calculate the number of ways to make k substitutions --/
def substitution_ways (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => starting_players * substitute_players
  | k+1 => starting_players * (substitute_players - k) * substitution_ways k

/-- Total number of substitution scenarios --/
def total_scenarios : ℕ :=
  List.sum (List.map substitution_ways (List.range (max_substitutions + 1)))

/-- Theorem stating the remainder when total scenarios is divided by 2000 --/
theorem substitution_remainder :
  total_scenarios % 2000 = 942 := by sorry

end NUMINAMATH_CALUDE_substitution_remainder_l2853_285359


namespace NUMINAMATH_CALUDE_smallest_divisor_of_S_l2853_285318

def S : Finset ℕ → ℕ := λ x => (x.sum (λ i => i^2)) + 8^2

theorem smallest_divisor_of_S : 
  ∀ (x : Finset ℕ), x.card = 6 ∧ x ⊆ Finset.range 7 → 2 ∣ S x ∧ 
  ∀ (k : ℕ), 0 < k ∧ k < 2 → ¬(∀ (y : Finset ℕ), y.card = 6 ∧ y ⊆ Finset.range 7 → k ∣ S y) := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_S_l2853_285318


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2853_285374

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 65)
  : ∃ (speed_second_hour : ℝ),
    speed_second_hour = 40 ∧
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l2853_285374


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2853_285352

/-- Given a line y = (5/3)x + 10, prove that a parallel line L
    that is 5 units away from it has the equation
    y = (5/3)x + (10 ± (5√34)/3) -/
theorem parallel_line_equation (x y : ℝ) :
  let original_line := fun x => (5/3) * x + 10
  let distance := 5
  let slope := 5/3
  let perpendicular_slope := -3/5
  let c := 10
  ∃ L : ℝ → ℝ,
    (∀ x, L x = slope * x + (c + distance * Real.sqrt (slope^2 + 1))) ∨
    (∀ x, L x = slope * x + (c - distance * Real.sqrt (slope^2 + 1))) ∧
    (∀ x, |L x - original_line x| / Real.sqrt (1 + perpendicular_slope^2) = distance) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2853_285352


namespace NUMINAMATH_CALUDE_max_triangles_theorem_l2853_285343

/-- Represents a convex n-gon with diagonals drawn such that no three or more intersect at a single point. -/
structure ConvexPolygonWithDiagonals where
  n : ℕ
  is_convex : Bool
  no_triple_intersection : Bool

/-- Calculates the maximum number of triangles formed by diagonals in a convex n-gon. -/
def max_triangles (polygon : ConvexPolygonWithDiagonals) : ℕ :=
  if polygon.n % 2 = 0 then
    2 * polygon.n - 4
  else
    2 * polygon.n - 5

/-- Theorem stating the maximum number of triangles formed by diagonals in a convex n-gon. -/
theorem max_triangles_theorem (polygon : ConvexPolygonWithDiagonals) :
  polygon.is_convex ∧ polygon.no_triple_intersection →
  max_triangles polygon = if polygon.n % 2 = 0 then 2 * polygon.n - 4 else 2 * polygon.n - 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_triangles_theorem_l2853_285343


namespace NUMINAMATH_CALUDE_complement_intersection_equals_five_l2853_285358

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

theorem complement_intersection_equals_five :
  (I \ A) ∩ B = {5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_five_l2853_285358


namespace NUMINAMATH_CALUDE_triangle_free_edge_bound_l2853_285320

/-- A graph with n vertices and k edges, where no three edges form a triangle -/
structure TriangleFreeGraph where
  n : ℕ  -- number of vertices
  k : ℕ  -- number of edges
  no_triangle : True  -- represents the condition that no three edges form a triangle

/-- Theorem: In a triangle-free graph, the number of edges is at most ⌊n²/4⌋ -/
theorem triangle_free_edge_bound (G : TriangleFreeGraph) : G.k ≤ (G.n^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_free_edge_bound_l2853_285320


namespace NUMINAMATH_CALUDE_product_prices_and_savings_l2853_285387

-- Define the discount rates
def discount_A : ℚ := 0.2
def discount_B : ℚ := 0.25

-- Define the equations from the conditions
def equation1 (x y : ℚ) : Prop := 6 * x + 3 * y = 600
def equation2 (x y : ℚ) : Prop := 50 * (1 - discount_A) * x + 40 * (1 - discount_B) * y = 5200

-- Define the prices we want to prove
def price_A : ℚ := 40
def price_B : ℚ := 120

-- Define the savings calculation
def savings (x y : ℚ) : ℚ :=
  80 * x + 100 * y - (80 * (1 - discount_A) * x + 100 * (1 - discount_B) * y)

-- Theorem statement
theorem product_prices_and_savings :
  equation1 price_A price_B ∧
  equation2 price_A price_B ∧
  savings price_A price_B = 3640 := by
  sorry

end NUMINAMATH_CALUDE_product_prices_and_savings_l2853_285387


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l2853_285300

theorem complete_square_equivalence :
  ∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x - 1)^2 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l2853_285300


namespace NUMINAMATH_CALUDE_page_lines_increase_percentage_l2853_285383

theorem page_lines_increase_percentage : 
  ∀ (original_lines : ℕ), 
  original_lines + 200 = 350 → 
  (200 : ℝ) / original_lines * 100 = 400 / 3 := by
sorry

end NUMINAMATH_CALUDE_page_lines_increase_percentage_l2853_285383


namespace NUMINAMATH_CALUDE_problem_solution_l2853_285393

theorem problem_solution (A B C : ℕ) 
  (h_diff1 : A ≠ B) (h_diff2 : B ≠ C) (h_diff3 : A ≠ C)
  (h1 : A + B = 84)
  (h2 : B + C = 60)
  (h3 : A = 6 * B) :
  A - C = 24 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2853_285393


namespace NUMINAMATH_CALUDE_sequence_of_primes_l2853_285326

theorem sequence_of_primes (a p : ℕ → ℕ) 
  (h_increasing : ∀ n m, n < m → a n < a m)
  (h_positive : ∀ n, 0 < a n)
  (h_prime : ∀ n, Nat.Prime (p n))
  (h_distinct : ∀ n m, n ≠ m → p n ≠ p m)
  (h_divides : ∀ n, p n ∣ a n)
  (h_difference : ∀ n k, a n - a k = p n - p k) :
  ∀ n, a n = p n :=
sorry

end NUMINAMATH_CALUDE_sequence_of_primes_l2853_285326


namespace NUMINAMATH_CALUDE_double_square_root_simplification_l2853_285307

theorem double_square_root_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 2 * Real.sqrt b > 0) 
  (hm : m > 0) (hn : n > 0)
  (h1 : Real.sqrt m ^ 2 + Real.sqrt n ^ 2 = a)
  (h2 : Real.sqrt m * Real.sqrt n = Real.sqrt b) :
  Real.sqrt (a + 2 * Real.sqrt b) = |Real.sqrt m + Real.sqrt n| ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end NUMINAMATH_CALUDE_double_square_root_simplification_l2853_285307


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l2853_285397

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l2853_285397


namespace NUMINAMATH_CALUDE_number_equals_eight_l2853_285353

theorem number_equals_eight (x : ℝ) : 0.75 * x + 2 = 8 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_eight_l2853_285353


namespace NUMINAMATH_CALUDE_oblique_drawing_area_relation_original_triangle_area_l2853_285339

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Represents the oblique drawing method transformation -/
def obliqueDrawing (t : Triangle) : Triangle := sorry

theorem oblique_drawing_area_relation (t : Triangle) :
  area (obliqueDrawing t) / area t = Real.sqrt 2 / 4 := sorry

/-- The main theorem proving the area of the original triangle -/
theorem original_triangle_area (t : Triangle) 
  (h1 : area (obliqueDrawing t) = 3) :
  area t = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_oblique_drawing_area_relation_original_triangle_area_l2853_285339


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2853_285394

theorem two_digit_number_problem :
  ∀ (x y : ℕ),
    x ≤ 9 ∧ y ≤ 9 ∧  -- Ensuring x and y are single digits
    x + y = 8 ∧  -- Sum of digits is 8
    (10 * x + y) * (10 * y + x) = 1855 →  -- Product condition
    (10 * x + y = 35) ∨ (10 * x + y = 53) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2853_285394


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l2853_285327

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l2853_285327


namespace NUMINAMATH_CALUDE_good_goods_sufficient_for_not_cheap_l2853_285373

-- Define the propositions
def good_goods : Prop := sorry
def not_cheap : Prop := sorry

-- Define Sister Qian's statement
def sister_qian_statement : Prop := good_goods → not_cheap

-- Theorem to prove
theorem good_goods_sufficient_for_not_cheap :
  sister_qian_statement → (∃ p q : Prop, (p → q) ∧ (p = good_goods) ∧ (q = not_cheap)) :=
by sorry

end NUMINAMATH_CALUDE_good_goods_sufficient_for_not_cheap_l2853_285373


namespace NUMINAMATH_CALUDE_percent_of_a_l2853_285319

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10/3) * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l2853_285319


namespace NUMINAMATH_CALUDE_tourists_knowing_both_languages_l2853_285367

theorem tourists_knowing_both_languages 
  (total : ℕ) 
  (neither : ℕ) 
  (german : ℕ) 
  (french : ℕ) 
  (h1 : total = 100) 
  (h2 : neither = 10) 
  (h3 : german = 76) 
  (h4 : french = 83) : 
  total - neither = german + french - 69 := by
sorry

end NUMINAMATH_CALUDE_tourists_knowing_both_languages_l2853_285367


namespace NUMINAMATH_CALUDE_periodic_function_value_l2853_285356

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  f 2015 = 5 → f 2016 = 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l2853_285356


namespace NUMINAMATH_CALUDE_total_luggage_calculation_l2853_285378

def passengers : ℕ := 4
def luggage_per_passenger : ℕ := 8

theorem total_luggage_calculation : passengers * luggage_per_passenger = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_luggage_calculation_l2853_285378


namespace NUMINAMATH_CALUDE_help_user_hours_l2853_285328

theorem help_user_hours (total_hours : Real) (software_hours : Real) (other_services_percentage : Real) :
  total_hours = 68.33333333333333 →
  software_hours = 24 →
  other_services_percentage = 0.40 →
  ∃ help_user_hours : Real,
    help_user_hours = total_hours - software_hours - (other_services_percentage * total_hours) ∧
    help_user_hours = 17 := by
  sorry

end NUMINAMATH_CALUDE_help_user_hours_l2853_285328


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2853_285361

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2853_285361


namespace NUMINAMATH_CALUDE_tammy_climbing_speed_l2853_285371

/-- Tammy's mountain climbing problem -/
theorem tammy_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * ((total_time + time_difference) / 2) + 
    (v + speed_difference) * ((total_time - time_difference) / 2) = total_distance ∧
    v + speed_difference = 4 := by
  sorry


end NUMINAMATH_CALUDE_tammy_climbing_speed_l2853_285371


namespace NUMINAMATH_CALUDE_third_day_temperature_l2853_285395

/-- Given three temperatures in Fahrenheit, calculates their average -/
def average (t1 t2 t3 : ℚ) : ℚ := (t1 + t2 + t3) / 3

/-- Proves that given an average temperature of -7°F for three days, 
    with temperatures of -8°F and +1°F on two of the days, 
    the temperature on the third day must be -14°F -/
theorem third_day_temperature 
  (t1 t2 t3 : ℚ) 
  (h1 : t1 = -8)
  (h2 : t2 = 1)
  (h_avg : average t1 t2 t3 = -7) :
  t3 = -14 := by
  sorry

#eval average (-8) 1 (-14) -- Should output -7

end NUMINAMATH_CALUDE_third_day_temperature_l2853_285395


namespace NUMINAMATH_CALUDE_probability_two_colored_is_four_ninths_l2853_285349

/-- Represents a cube divided into smaller cubes -/
structure DividedCube where
  total_small_cubes : ℕ
  two_colored_faces : ℕ

/-- The probability of selecting a cube with exactly 2 colored faces -/
def probability_two_colored (cube : DividedCube) : ℚ :=
  cube.two_colored_faces / cube.total_small_cubes

/-- Theorem stating the probability of selecting a cube with exactly 2 colored faces -/
theorem probability_two_colored_is_four_ninths (cube : DividedCube) 
    (h1 : cube.total_small_cubes = 27)
    (h2 : cube.two_colored_faces = 12) : 
  probability_two_colored cube = 4/9 := by
  sorry

#check probability_two_colored_is_four_ninths

end NUMINAMATH_CALUDE_probability_two_colored_is_four_ninths_l2853_285349


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2853_285370

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

/-- The asymptotic line equations of the hyperbola -/
def asymptotic_lines (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- Theorem: The asymptotic line equations of the given hyperbola are y = ±√3x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptotic_lines x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2853_285370


namespace NUMINAMATH_CALUDE_range_of_m_l2853_285309

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the proposition q
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (¬p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2853_285309


namespace NUMINAMATH_CALUDE_phi_difference_squared_l2853_285333

theorem phi_difference_squared : ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 2*Φ - 1 = 0 → 
  φ^2 - 2*φ - 1 = 0 → 
  (Φ - φ)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_phi_difference_squared_l2853_285333


namespace NUMINAMATH_CALUDE_function_properties_and_range_l2853_285338

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties_and_range 
  (A ω φ : ℝ) 
  (h_A : A > 0) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (h_max : f A ω φ (π/6) = 2)
  (h_roots : ∃ x₁ x₂, f A ω φ x₁ = 0 ∧ f A ω φ x₂ = 0 ∧ 
    ∀ y₁ y₂, f A ω φ y₁ = 0 → f A ω φ y₂ = 0 → |y₁ - y₂| ≥ π) :
  (∀ x, f A ω φ x = 2 * Real.sin (x + π/3)) ∧
  (∀ x ∈ Set.Icc (-π/4) (π/4), 
    2 * Real.sin (2*x + π/3) ∈ Set.Icc (-1) 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_range_l2853_285338


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l2853_285310

def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

theorem parallel_vectors_imply_x_equals_two :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ (a + b x) = k • (4 • b x - 2 • a)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_two_l2853_285310


namespace NUMINAMATH_CALUDE_rotation_surface_area_theorem_l2853_285323

/-- Represents a plane curve -/
structure PlaneCurve where
  -- Add necessary fields for a plane curve

/-- Calculates the length of a plane curve -/
def curveLength (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the distance of the center of gravity from the axis of rotation -/
def centerOfGravityDistance (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the surface area generated by rotating a plane curve around an axis -/
def rotationSurfaceArea (c : PlaneCurve) : ℝ :=
  sorry

/-- Theorem: The surface area generated by rotating an arbitrary plane curve around an axis
    is equal to 2π times the distance of the center of gravity from the axis
    times the length of the curve -/
theorem rotation_surface_area_theorem (c : PlaneCurve) :
  rotationSurfaceArea c = 2 * Real.pi * centerOfGravityDistance c * curveLength c :=
sorry

end NUMINAMATH_CALUDE_rotation_surface_area_theorem_l2853_285323
