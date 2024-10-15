import Mathlib

namespace NUMINAMATH_CALUDE_simplified_expression_l2953_295349

theorem simplified_expression (a : ℝ) (h : a ≠ 1/2) :
  1 - (2 / (1 + (2*a / (1 - 2*a)))) = 4*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l2953_295349


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2953_295312

theorem sum_of_three_numbers : 1.35 + 0.123 + 0.321 = 1.794 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2953_295312


namespace NUMINAMATH_CALUDE_total_visits_equals_39_l2953_295324

/-- Calculates the total number of doctor visits in a year -/
def total_visits (visits_per_month_doc1 : ℕ) 
                 (months_between_visits_doc2 : ℕ) 
                 (visits_per_period_doc3 : ℕ) 
                 (months_per_period_doc3 : ℕ) : ℕ :=
  let months_in_year := 12
  let visits_doc1 := visits_per_month_doc1 * months_in_year
  let visits_doc2 := months_in_year / months_between_visits_doc2
  let periods_in_year := months_in_year / months_per_period_doc3
  let visits_doc3 := visits_per_period_doc3 * periods_in_year
  visits_doc1 + visits_doc2 + visits_doc3

/-- Theorem stating that the total visits in a year is 39 -/
theorem total_visits_equals_39 : 
  total_visits 2 2 3 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_visits_equals_39_l2953_295324


namespace NUMINAMATH_CALUDE_savings_calculation_l2953_295338

def thomas_monthly_savings : ℕ := 40
def saving_years : ℕ := 6
def months_per_year : ℕ := 12
def joseph_savings_ratio : ℚ := 3 / 5  -- Joseph saves 2/5 less, so he saves 3/5 of what Thomas saves

def total_savings : ℕ := 4608

theorem savings_calculation :
  thomas_monthly_savings * saving_years * months_per_year +
  (thomas_monthly_savings * joseph_savings_ratio).floor * saving_years * months_per_year = total_savings :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l2953_295338


namespace NUMINAMATH_CALUDE_cup_sales_problem_l2953_295302

/-- Proves that the number of additional days is 11, given the conditions of the cup sales problem -/
theorem cup_sales_problem (first_day_sales : ℕ) (daily_sales : ℕ) (average_sales : ℚ) : 
  first_day_sales = 86 →
  daily_sales = 50 →
  average_sales = 53 →
  ∃ d : ℕ, 
    (first_day_sales + d * daily_sales : ℚ) / (d + 1 : ℚ) = average_sales ∧
    d = 11 := by
  sorry


end NUMINAMATH_CALUDE_cup_sales_problem_l2953_295302


namespace NUMINAMATH_CALUDE_eden_bears_count_eden_final_bears_count_l2953_295376

theorem eden_bears_count (initial_bears : ℕ) (favorite_bears : ℕ) (sisters : ℕ) (eden_initial_bears : ℕ) : ℕ :=
  let remaining_bears := initial_bears - favorite_bears
  let bears_per_sister := remaining_bears / sisters
  eden_initial_bears + bears_per_sister

theorem eden_final_bears_count :
  eden_bears_count 20 8 3 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_eden_bears_count_eden_final_bears_count_l2953_295376


namespace NUMINAMATH_CALUDE_range_of_m_l2953_295309

/-- Given points A(1,0) and B(4,0) in the Cartesian plane, and a point P on the line x-y+m=0 
    such that 2PA = PB, the range of possible values for m is [-2√2, 2√2]. -/
theorem range_of_m (m : ℝ) : 
  ∃ (x y : ℝ), 
    (x - y + m = 0) ∧ 
    (2 * ((x - 1)^2 + y^2) = (x - 4)^2 + y^2) →
    -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2953_295309


namespace NUMINAMATH_CALUDE_sum_of_digits_879_times_492_l2953_295300

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits in the product of 879 and 492 is 27 -/
theorem sum_of_digits_879_times_492 :
  sum_of_digits (879 * 492) = 27 := by
  sorry

#eval sum_of_digits (879 * 492)  -- This line is optional, for verification

end NUMINAMATH_CALUDE_sum_of_digits_879_times_492_l2953_295300


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2953_295344

/-- A number is prime or the sum of two consecutive primes -/
def IsPrimeOrSumOfConsecutivePrimes (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ q = p + 1 ∧ n = p + q

/-- The theorem statement -/
theorem rectangular_solid_surface_area 
  (a b c : ℕ) 
  (ha : IsPrimeOrSumOfConsecutivePrimes a) 
  (hb : IsPrimeOrSumOfConsecutivePrimes b)
  (hc : IsPrimeOrSumOfConsecutivePrimes c)
  (hv : a * b * c = 399) : 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2953_295344


namespace NUMINAMATH_CALUDE_women_count_l2953_295355

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan_ratio : ℚ
  women_with_plan_ratio : ℚ
  total_men : ℕ

/-- Calculates the number of women in the company -/
def number_of_women (c : Company) : ℚ :=
  c.women_without_plan_ratio * c.workers_without_plan +
  c.women_with_plan_ratio * (c.total_workers - c.workers_without_plan)

/-- Theorem stating the number of women in the company -/
theorem women_count (c : Company) 
  (h1 : c.total_workers = 200)
  (h2 : c.workers_without_plan = c.total_workers / 3)
  (h3 : c.women_without_plan_ratio = 2/5)
  (h4 : c.women_with_plan_ratio = 3/5)
  (h5 : c.total_men = 120) :
  ∃ (n : ℕ), n ≤ number_of_women c ∧ number_of_women c < n + 1 ∧ n = 107 := by
  sorry

end NUMINAMATH_CALUDE_women_count_l2953_295355


namespace NUMINAMATH_CALUDE_corporation_employees_l2953_295377

theorem corporation_employees (part_time : ℕ) (full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_corporation_employees_l2953_295377


namespace NUMINAMATH_CALUDE_base7_multiplication_l2953_295378

/-- Converts a base 7 number (represented as a list of digits) to a natural number. -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation (as a list of digits). -/
def natToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The main theorem stating that 356₇ * 4₇ = 21323₇ in base 7. -/
theorem base7_multiplication :
  natToBase7 (base7ToNat [3, 5, 6] * base7ToNat [4]) = [2, 1, 3, 2, 3] := by
  sorry

#eval base7ToNat [3, 5, 6] -- Should output 188
#eval base7ToNat [4] -- Should output 4
#eval natToBase7 (188 * 4) -- Should output [2, 1, 3, 2, 3]

end NUMINAMATH_CALUDE_base7_multiplication_l2953_295378


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l2953_295391

/-- Checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments --/
def set_A : List ℝ := [2, 5, 7]
def set_B : List ℝ := [4, 4, 8]
def set_C : List ℝ := [4, 5, 6]
def set_D : List ℝ := [4, 5, 10]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l2953_295391


namespace NUMINAMATH_CALUDE_express_c_in_terms_of_a_and_b_l2953_295341

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ × ℝ := (4, 1)

theorem express_c_in_terms_of_a_and_b :
  c = 2 • a - b := by sorry

end NUMINAMATH_CALUDE_express_c_in_terms_of_a_and_b_l2953_295341


namespace NUMINAMATH_CALUDE_sara_bouquets_l2953_295360

theorem sara_bouquets (red_flowers yellow_flowers : ℕ) 
  (h1 : red_flowers = 16) 
  (h2 : yellow_flowers = 24) : 
  (Nat.gcd red_flowers yellow_flowers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sara_bouquets_l2953_295360


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_factorials_l2953_295306

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 5 + factorial 6

theorem largest_prime_factor_of_sum_factorials :
  (Nat.factors sum_of_factorials).maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_factorials_l2953_295306


namespace NUMINAMATH_CALUDE_marta_textbook_problem_l2953_295367

theorem marta_textbook_problem :
  ∀ (sale_books online_books bookstore_books : ℕ)
    (sale_price online_total bookstore_total total_spent : ℚ),
    sale_books = 5 →
    sale_price = 10 →
    online_books = 2 →
    online_total = 40 →
    bookstore_total = 3 * online_total →
    total_spent = 210 →
    sale_books * sale_price + online_total + bookstore_total = total_spent →
    bookstore_books = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_marta_textbook_problem_l2953_295367


namespace NUMINAMATH_CALUDE_popcorn_tablespoons_needed_l2953_295372

/-- The number of cups of popcorn produced by 2 tablespoons of kernels -/
def cups_per_two_tablespoons : ℕ := 4

/-- The number of cups of popcorn Joanie wants -/
def joanie_cups : ℕ := 3

/-- The number of cups of popcorn Mitchell wants -/
def mitchell_cups : ℕ := 4

/-- The number of cups of popcorn Miles and Davis will split -/
def miles_davis_cups : ℕ := 6

/-- The number of cups of popcorn Cliff wants -/
def cliff_cups : ℕ := 3

/-- The total number of cups of popcorn wanted -/
def total_cups : ℕ := joanie_cups + mitchell_cups + miles_davis_cups + cliff_cups

/-- Theorem stating the number of tablespoons of popcorn kernels needed -/
theorem popcorn_tablespoons_needed : 
  (total_cups / cups_per_two_tablespoons) * 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_tablespoons_needed_l2953_295372


namespace NUMINAMATH_CALUDE_cats_sold_l2953_295311

theorem cats_sold (ratio : ℚ) (dogs : ℕ) (cats : ℕ) : 
  ratio = 2 / 1 → dogs = 8 → cats = 16 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_l2953_295311


namespace NUMINAMATH_CALUDE_rachels_homework_l2953_295383

theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 → 
  math_pages = reading_pages + 4 → 
  reading_pages = 3 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l2953_295383


namespace NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l2953_295398

theorem rectangle_triangle_perimeter (d : ℕ) : 
  let triangle_side := 3 * w - d
  let rectangle_width := w
  let rectangle_length := 3 * w
  let triangle_perimeter := 3 * triangle_side
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  (∀ w : ℕ, 
    triangle_perimeter > 0 ∧ 
    rectangle_perimeter = triangle_perimeter + 1950 ∧
    rectangle_length - triangle_side = d ∧
    rectangle_length = 3 * rectangle_width) →
  d > 650 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l2953_295398


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2953_295304

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2953_295304


namespace NUMINAMATH_CALUDE_wendy_pastries_left_l2953_295389

/-- The number of pastries Wendy had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Theorem stating that Wendy had 24 pastries left after the bake sale -/
theorem wendy_pastries_left : pastries_left 4 29 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_wendy_pastries_left_l2953_295389


namespace NUMINAMATH_CALUDE_vessel_combination_theorem_l2953_295358

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℚ
  denominator : ℚ

/-- Represents a vessel containing a mixture of milk and water -/
structure Vessel where
  volume : ℚ
  milkWaterRatio : Ratio

/-- Combines the contents of two vessels -/
def combineVessels (v1 v2 : Vessel) : Ratio :=
  let totalMilk := v1.volume * v1.milkWaterRatio.numerator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                   v2.volume * v2.milkWaterRatio.numerator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  let totalWater := v1.volume * v1.milkWaterRatio.denominator / (v1.milkWaterRatio.numerator + v1.milkWaterRatio.denominator) +
                    v2.volume * v2.milkWaterRatio.denominator / (v2.milkWaterRatio.numerator + v2.milkWaterRatio.denominator)
  { numerator := totalMilk, denominator := totalWater }

theorem vessel_combination_theorem :
  let v1 : Vessel := { volume := 3, milkWaterRatio := { numerator := 1, denominator := 2 } }
  let v2 : Vessel := { volume := 5, milkWaterRatio := { numerator := 3, denominator := 2 } }
  let combinedRatio := combineVessels v1 v2
  combinedRatio.numerator = combinedRatio.denominator :=
by
  sorry

end NUMINAMATH_CALUDE_vessel_combination_theorem_l2953_295358


namespace NUMINAMATH_CALUDE_selection_probabilities_l2953_295356

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 4

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the total number of people in the group -/
def total_people : ℕ := num_boys + num_girls

/-- Represents the number of people to be selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting two boys -/
def prob_two_boys : ℚ := (num_boys.choose 2) / (total_people.choose 2)

/-- Calculates the probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (num_boys.choose 1 * num_girls.choose 1) / (total_people.choose 2)

/-- Calculates the probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_two_boys

theorem selection_probabilities :
  prob_two_boys = 2/7 ∧
  prob_one_girl = 4/7 ∧
  prob_at_least_one_girl = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_l2953_295356


namespace NUMINAMATH_CALUDE_apple_cost_price_l2953_295371

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 20 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - (loss_fraction * cost_price) ∧ 
    cost_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l2953_295371


namespace NUMINAMATH_CALUDE_mary_green_crayons_mary_green_crayons_correct_l2953_295326

theorem mary_green_crayons 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : ℕ :=
  
  let initial_total := remaining + green_given + blue_given
  let initial_green := initial_total - initial_blue

  initial_green

theorem mary_green_crayons_correct 
  (initial_blue : ℕ) 
  (green_given : ℕ) 
  (blue_given : ℕ) 
  (remaining : ℕ) : 
  mary_green_crayons initial_blue green_given blue_given remaining = 5 :=
by
  -- Given conditions
  have h1 : initial_blue = 8 := by sorry
  have h2 : green_given = 3 := by sorry
  have h3 : blue_given = 1 := by sorry
  have h4 : remaining = 9 := by sorry

  -- Proof
  sorry

end NUMINAMATH_CALUDE_mary_green_crayons_mary_green_crayons_correct_l2953_295326


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_triangle_area_l2953_295329

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  3 * t.b^2 = 2 * t.a * t.c * (1 + Real.cos t.B)

-- Define arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Theorem 1
theorem triangle_arithmetic_sequence (t : Triangle) 
  (h : satisfiesCondition t) : isArithmeticSequence t.a t.b t.c := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 3) (h2 : t.b = 5) (h3 : satisfiesCondition t) : 
  (1/2 * t.a * t.b * Real.sin t.C) = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_triangle_area_l2953_295329


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l2953_295303

def original_list : List Int := List.range 15 |>.map (λ x => x - 2)

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (λ x => x ≠ n)

def count_pairs_sum_11 (list : List Int) : Nat :=
  list.filterMap (λ x => 
    if x < 11 ∧ list.contains (11 - x) ∧ x ≠ 11 - x
    then some (x, 11 - x)
    else none
  ) |>.length

theorem remove_six_maximizes_probability :
  ∀ n ∈ original_list, n ≠ 6 →
    count_pairs_sum_11 (remove_number original_list 6) ≥ 
    count_pairs_sum_11 (remove_number original_list n) :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l2953_295303


namespace NUMINAMATH_CALUDE_towel_area_decrease_l2953_295375

theorem towel_area_decrease : 
  ∀ (original_length original_width : ℝ),
  original_length > 0 → original_width > 0 →
  let new_length := original_length * 0.8
  let new_width := original_width * 0.9
  let original_area := original_length * original_width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l2953_295375


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2953_295333

theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2953_295333


namespace NUMINAMATH_CALUDE_f_properties_l2953_295346

noncomputable def f (a x : ℝ) : ℝ := Real.log x - (a + 2) * x + a * x^2

theorem f_properties (a : ℝ) :
  -- Part I: Tangent line equation when a = 0
  (∀ x y : ℝ, f 0 1 = -2 ∧ x + y + 1 = 0 ↔ y = f 0 x ∧ (x - 1) * (f 0 x - f 0 1) = (y - f 0 1) * (x - 1)) ∧
  -- Part II: Monotonicity intervals
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) > f a x)) ∧
  (∀ x : ℝ, x > 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) < f a x)) ∧
  -- Part III: Condition for exactly two zeros
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔ a < -4 * Real.log 2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2953_295346


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l2953_295361

/-- The area of the shaded region between a circle circumscribing two overlapping circles
    and the two smaller circles. -/
theorem shaded_area_between_circles (r₁ r₂ d R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : d = 6)
    (h₄ : R = r₁ + r₂ + d) : π * R^2 - (π * r₁^2 + π * r₂^2) = 184 * π := by
  sorry

#check shaded_area_between_circles

end NUMINAMATH_CALUDE_shaded_area_between_circles_l2953_295361


namespace NUMINAMATH_CALUDE_refrigerator_price_calculation_l2953_295390

def refrigerator_purchase_price (labelled_price : ℝ) (discount_rate : ℝ) (additional_costs : ℝ) (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  (1 - discount_rate) * labelled_price + additional_costs

theorem refrigerator_price_calculation :
  let labelled_price : ℝ := 18400 / 1.15
  let discount_rate : ℝ := 0.20
  let additional_costs : ℝ := 125 + 250
  let selling_price : ℝ := 18400
  let profit_rate : ℝ := 0.15
  refrigerator_purchase_price labelled_price discount_rate additional_costs selling_price profit_rate = 13175 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_calculation_l2953_295390


namespace NUMINAMATH_CALUDE_opposite_of_eight_l2953_295353

theorem opposite_of_eight : 
  -(8 : ℤ) = -8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l2953_295353


namespace NUMINAMATH_CALUDE_ship_length_l2953_295322

/-- The length of a ship given its speed and time to cross a lighthouse -/
theorem ship_length (speed : ℝ) (time : ℝ) : 
  speed = 18 → time = 20 → speed * time * (1000 / 3600) = 100 := by
  sorry

#check ship_length

end NUMINAMATH_CALUDE_ship_length_l2953_295322


namespace NUMINAMATH_CALUDE_xy_value_l2953_295345

theorem xy_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y + 4*y + 4 = 0) : x^y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2953_295345


namespace NUMINAMATH_CALUDE_pants_pricing_l2953_295354

theorem pants_pricing (S P : ℝ) 
  (h1 : S = P + 0.25 * S)
  (h2 : 14 = 0.8 * S - P) :
  P = 210 := by sorry

end NUMINAMATH_CALUDE_pants_pricing_l2953_295354


namespace NUMINAMATH_CALUDE_candy_store_spending_l2953_295369

def weekly_allowance : ℚ := 4.5

def arcade_spending (allowance : ℚ) : ℚ := (3 / 5) * allowance

def remaining_after_arcade (allowance : ℚ) : ℚ := allowance - arcade_spending allowance

def toy_store_spending (remaining : ℚ) : ℚ := (1 / 3) * remaining

def remaining_after_toy_store (remaining : ℚ) : ℚ := remaining - toy_store_spending remaining

theorem candy_store_spending :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_candy_store_spending_l2953_295369


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l2953_295381

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n has exactly 16 different positive integer divisors -/
def has_16_divisors (n : ℕ) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ), has_16_divisors n ∧ ∀ (m : ℕ), has_16_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l2953_295381


namespace NUMINAMATH_CALUDE_landmark_distance_set_l2953_295373

def distance_to_landmark (d : ℝ) : Prop :=
  d > 0 ∧ (d < 7 ∨ d > 7) ∧ (d ≤ 8 ∨ d > 8) ∧ (d ≤ 10 ∨ d > 10)

theorem landmark_distance_set :
  ∀ d : ℝ, distance_to_landmark d ↔ d > 10 :=
sorry

end NUMINAMATH_CALUDE_landmark_distance_set_l2953_295373


namespace NUMINAMATH_CALUDE_checkerboard_probability_l2953_295393

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square that doesn't touch the outer edge -/
def probabilityInnerSquare : ℚ := innerSquares / totalSquares

theorem checkerboard_probability :
  probabilityInnerSquare = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l2953_295393


namespace NUMINAMATH_CALUDE_students_opted_for_math_and_science_l2953_295330

/-- Given a class with the following properties:
  * There are 40 students in total.
  * 10 students did not opt for math.
  * 15 students did not opt for science.
  * 20 students did not opt for history.
  * 5 students did not opt for geography.
  * 2 students did not opt for either math or science.
  * 3 students did not opt for either math or history.
  * 4 students did not opt for either math or geography.
  * 7 students did not opt for either science or history.
  * 8 students did not opt for either science or geography.
  * 10 students did not opt for either history or geography.

  Prove that the number of students who opted for both math and science is 17. -/
theorem students_opted_for_math_and_science
  (total : ℕ) (not_math : ℕ) (not_science : ℕ) (not_history : ℕ) (not_geography : ℕ)
  (not_math_or_science : ℕ) (not_math_or_history : ℕ) (not_math_or_geography : ℕ)
  (not_science_or_history : ℕ) (not_science_or_geography : ℕ) (not_history_or_geography : ℕ)
  (h_total : total = 40)
  (h_not_math : not_math = 10)
  (h_not_science : not_science = 15)
  (h_not_history : not_history = 20)
  (h_not_geography : not_geography = 5)
  (h_not_math_or_science : not_math_or_science = 2)
  (h_not_math_or_history : not_math_or_history = 3)
  (h_not_math_or_geography : not_math_or_geography = 4)
  (h_not_science_or_history : not_science_or_history = 7)
  (h_not_science_or_geography : not_science_or_geography = 8)
  (h_not_history_or_geography : not_history_or_geography = 10) :
  (total - not_math) + (total - not_science) - (total - not_math_or_science) = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_opted_for_math_and_science_l2953_295330


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2953_295339

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2953_295339


namespace NUMINAMATH_CALUDE_money_left_after_spending_l2953_295382

theorem money_left_after_spending (initial_amount spent_on_sweets given_to_friend number_of_friends : ℚ) :
  initial_amount = 8.5 ∧ 
  spent_on_sweets = 1.25 ∧ 
  given_to_friend = 1.2 ∧ 
  number_of_friends = 2 →
  initial_amount - (spent_on_sweets + given_to_friend * number_of_friends) = 4.85 := by
sorry

end NUMINAMATH_CALUDE_money_left_after_spending_l2953_295382


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l2953_295384

/-- Given a rectangle EFGH and an ellipse, prove the perimeter of the rectangle -/
theorem rectangle_ellipse_perimeter (p q c d : ℝ) : 
  p > 0 → q > 0 → c > 0 → d > 0 →
  p * q = 4032 →
  π * c * d = 2016 * π →
  p + q = 2 * c →
  p^2 + q^2 = 4 * (c^2 - d^2) →
  2 * (p + q) = 8 * Real.sqrt 2016 := by
sorry


end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l2953_295384


namespace NUMINAMATH_CALUDE_fencing_cost_is_225_rupees_l2953_295379

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculate the total fencing cost for a rectangular park -/
def calculate_fencing_cost (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width) * park.fencing_cost_per_meter

/-- Theorem: The fencing cost for the given rectangular park is 225 rupees -/
theorem fencing_cost_is_225_rupees :
  ∀ (park : RectangularPark),
    park.length / park.width = 3 / 2 →
    park.area = 3750 →
    park.fencing_cost_per_meter = 0.9 →
    calculate_fencing_cost park = 225 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_is_225_rupees_l2953_295379


namespace NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l2953_295323

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ satisfies_condition n ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_digit_sum_10_l2953_295323


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2953_295397

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 5 = 0) ∧
  (∀ (x : ℝ), x^2 + b*x + 5 ≠ 0) ∧
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2953_295397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2953_295368

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2953_295368


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2953_295321

theorem inequality_solution_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ m : ℝ, (4 / (x + 1) + 1 / y < m^2 + 3/2 * m)) →
  (∃ m : ℝ, m < -3 ∨ m > 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2953_295321


namespace NUMINAMATH_CALUDE_distance_center_to_point_l2953_295387

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the point
def point : ℝ × ℝ := (8, 3)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l2953_295387


namespace NUMINAMATH_CALUDE_determinant_scaling_l2953_295351

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 7 →
  Matrix.det ![![3*a, 3*b], ![3*c, 3*d]] = 63 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l2953_295351


namespace NUMINAMATH_CALUDE_range_of_g_l2953_295336

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  π/4 ≤ Real.arcsin x + Real.arccos x - Real.arctan x ∧ 
  Real.arcsin x + Real.arccos x - Real.arctan x ≤ 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2953_295336


namespace NUMINAMATH_CALUDE_k_value_proof_l2953_295366

theorem k_value_proof (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_k_value_proof_l2953_295366


namespace NUMINAMATH_CALUDE_min_ellipse_eccentricity_l2953_295328

/-- Given an ellipse C: x²/a² + y²/b² = 1 where a > b > 0, with foci F₁ and F₂,
    and right vertex A. A line l passing through F₁ intersects C at P and Q.
    AP · AQ = (1/2)(a+c)². The minimum eccentricity of C is 1 - √2/2. -/
theorem min_ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  ∀ P Q : ℝ × ℝ,
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
  (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
  ∃ m : ℝ, (P.1 = m * P.2 - c) ∧ (Q.1 = m * Q.2 - c) →
  ((P.1 - a) * (Q.1 - a) + P.2 * Q.2 = (1/2) * (a + c)^2) →
  e ≥ 1 - Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_ellipse_eccentricity_l2953_295328


namespace NUMINAMATH_CALUDE_b_absolute_value_l2953_295315

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ := a * x^5 + b * x^4 + c * x^3 + b * x + a

-- State the theorem
theorem b_absolute_value (a b c : ℤ) : 
  (g a b c (3 + i) = 0) →  -- Condition 1
  (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c) = 1) →  -- Condition 2 and 3
  (Int.natAbs b = 60) :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_b_absolute_value_l2953_295315


namespace NUMINAMATH_CALUDE_trig_simplification_l2953_295313

theorem trig_simplification (x y z : ℝ) :
  Real.sin (x - y + z) * Real.cos y - Real.cos (x - y + z) * Real.sin y = Real.sin (x - 2*y + z) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2953_295313


namespace NUMINAMATH_CALUDE_xiaoming_home_most_precise_l2953_295325

-- Define the possible descriptions of location
inductive LocationDescription
  | RightSide
  | Distance (d : ℝ)
  | WestSide
  | WestSideWithDistance (d : ℝ)

-- Define a function to check if a description is complete (has both direction and distance)
def isCompleteDescription (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.WestSideWithDistance _ => True
  | _ => False

-- Define Xiao Ming's home location
def xiaomingHome : LocationDescription := LocationDescription.WestSideWithDistance 900

-- Theorem: Xiao Ming's home location is the most precise description
theorem xiaoming_home_most_precise :
  isCompleteDescription xiaomingHome ∧
  ∀ (desc : LocationDescription), isCompleteDescription desc → desc = xiaomingHome :=
sorry

end NUMINAMATH_CALUDE_xiaoming_home_most_precise_l2953_295325


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2953_295332

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -1)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 1)

theorem reflection_across_x_axis :
  reflect_x original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2953_295332


namespace NUMINAMATH_CALUDE_train_journey_time_l2953_295334

/-- Proves that if a train moving at 6/7 of its usual speed arrives 30 minutes late, then its usual journey time is 3 hours. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 1/2) = usual_speed * usual_time) : 
  usual_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2953_295334


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l2953_295362

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := countOnes n / totalElements n

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l2953_295362


namespace NUMINAMATH_CALUDE_not_p_equiv_p_and_q_equiv_l2953_295396

-- Define propositions p and q
def p (x : ℝ) := x * (x - 2) ≥ 0
def q (x : ℝ) := |x - 2| < 1

-- Theorem 1: Negation of p is equivalent to 0 < x < 2
theorem not_p_equiv (x : ℝ) : ¬(p x) ↔ 0 < x ∧ x < 2 := by sorry

-- Theorem 2: p and q together are equivalent to 2 ≤ x < 3
theorem p_and_q_equiv (x : ℝ) : p x ∧ q x ↔ 2 ≤ x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_not_p_equiv_p_and_q_equiv_l2953_295396


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2953_295395

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 8 * x - 18 = 0 ↔ x = 3 ∨ x = -4/3) → k = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2953_295395


namespace NUMINAMATH_CALUDE_savings_difference_is_250_l2953_295307

def window_price : ℕ := 125
def offer_purchase : ℕ := 6
def offer_free : ℕ := 2
def dave_windows : ℕ := 9
def doug_windows : ℕ := 11

def calculate_cost (num_windows : ℕ) : ℕ :=
  let sets := num_windows / (offer_purchase + offer_free)
  let remainder := num_windows % (offer_purchase + offer_free)
  (sets * offer_purchase + min remainder offer_purchase) * window_price

def savings_difference : ℕ :=
  let separate_cost := calculate_cost dave_windows + calculate_cost doug_windows
  let combined_cost := calculate_cost (dave_windows + doug_windows)
  let separate_savings := dave_windows * window_price + doug_windows * window_price - separate_cost
  let combined_savings := (dave_windows + doug_windows) * window_price - combined_cost
  combined_savings - separate_savings

theorem savings_difference_is_250 : savings_difference = 250 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_is_250_l2953_295307


namespace NUMINAMATH_CALUDE_tan_sum_greater_than_three_l2953_295327

theorem tan_sum_greater_than_three :
  Real.tan (40 * π / 180) + Real.tan (45 * π / 180) + Real.tan (50 * π / 180) > 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_greater_than_three_l2953_295327


namespace NUMINAMATH_CALUDE_expression_evaluation_l2953_295316

theorem expression_evaluation : (2.1 * (49.7 + 0.3) + 15 : ℝ) = 120 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2953_295316


namespace NUMINAMATH_CALUDE_first_book_pictures_correct_l2953_295352

/-- The number of pictures in the first coloring book -/
def pictures_in_first_book : ℕ := 23

/-- The number of pictures in the second coloring book -/
def pictures_in_second_book : ℕ := 32

/-- The total number of pictures in both coloring books -/
def total_pictures : ℕ := 55

/-- Theorem stating that the number of pictures in the first coloring book is correct -/
theorem first_book_pictures_correct :
  pictures_in_first_book + pictures_in_second_book = total_pictures :=
by sorry

end NUMINAMATH_CALUDE_first_book_pictures_correct_l2953_295352


namespace NUMINAMATH_CALUDE_northwest_molded_handle_cost_l2953_295342

/-- Northwest Molded's handle production problem -/
theorem northwest_molded_handle_cost 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 7640)
  (h2 : selling_price = 4.60)
  (h3 : break_even_quantity = 1910) :
  ∃ (cost_per_handle : ℝ), 
    cost_per_handle = 0.60 ∧ 
    (selling_price * break_even_quantity : ℝ) = fixed_cost + (break_even_quantity : ℝ) * cost_per_handle :=
by sorry

end NUMINAMATH_CALUDE_northwest_molded_handle_cost_l2953_295342


namespace NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2953_295399

/-- The number of family members -/
def family_members : ℕ := 5

/-- The number of days Sheila can place cinnamon balls in socks -/
def days : ℕ := 10

/-- The number of cinnamon balls Sheila bought -/
def cinnamon_balls : ℕ := family_members * days

theorem sheila_cinnamon_balls : cinnamon_balls = 50 := by
  sorry

end NUMINAMATH_CALUDE_sheila_cinnamon_balls_l2953_295399


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2953_295385

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square garden -/
def Garden :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a frog jump -/
def jump (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Down => ⟨p.x, p.y - 1⟩
  | Direction.Left => ⟨p.x - 1, p.y⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Checks if a point is on the vertical sides of the garden -/
def isOnVerticalSide (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = 5) ∧ 0 ≤ p.y ∧ p.y ≤ 5

/-- The probability of ending on a vertical side from a given point -/
noncomputable def probabilityVerticalSide (p : Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 1⟩ = 13 / 20 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2953_295385


namespace NUMINAMATH_CALUDE_yard_width_calculation_l2953_295348

/-- The width of a rectangular yard with a row of trees --/
def yard_width (num_trees : ℕ) (edge_distance : ℝ) (center_distance : ℝ) (end_space : ℝ) : ℝ :=
  let tree_diameter := center_distance - edge_distance
  let total_center_distance := (num_trees - 1) * center_distance
  let total_tree_width := tree_diameter * num_trees
  let total_end_space := 2 * end_space
  total_center_distance + total_tree_width + total_end_space

/-- Theorem stating the width of the yard given the specific conditions --/
theorem yard_width_calculation :
  yard_width 6 12 15 2 = 82 := by
  sorry

end NUMINAMATH_CALUDE_yard_width_calculation_l2953_295348


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l2953_295350

/-- Given a rectangle ABCD with area a + 4√3, where the lines joining the centers of 
    circles inscribed in its corners form an equilateral triangle with side length 2, 
    prove that a = 8. -/
theorem rectangle_area_proof (a : ℝ) : 
  let triangle_side_length : ℝ := 2
  let rectangle_width : ℝ := triangle_side_length + triangle_side_length * Real.sqrt 3 / 2
  let rectangle_height : ℝ := 4
  let rectangle_area : ℝ := a + 4 * Real.sqrt 3
  rectangle_area = rectangle_width * rectangle_height → a = 8 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l2953_295350


namespace NUMINAMATH_CALUDE_function_composition_problem_l2953_295301

/-- Given a function f(x) = ax - b where a > 0 and f(f(x)) = 4x - 3, prove that f(2) = 3 -/
theorem function_composition_problem (a b : ℝ) (h1 : a > 0) :
  (∀ x, ∃ f : ℝ → ℝ, f x = a * x - b) →
  (∀ x, ∃ f : ℝ → ℝ, f (f x) = 4 * x - 3) →
  ∃ f : ℝ → ℝ, f 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_function_composition_problem_l2953_295301


namespace NUMINAMATH_CALUDE_angle_calculation_l2953_295363

/-- Given three angles, proves that if angle 1 and angle 2 are complementary, 
    angle 2 and angle 3 are supplementary, and angle 3 equals 18°, 
    then angle 1 equals 108°. -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 = 90 →
  angle2 + angle3 = 180 →
  angle3 = 18 →
  angle1 = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l2953_295363


namespace NUMINAMATH_CALUDE_number_added_to_product_l2953_295310

theorem number_added_to_product (a b : Int) (h1 : a = -2) (h2 : b = -3) :
  ∃ x : Int, a * b + x = 65 ∧ x = 59 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_product_l2953_295310


namespace NUMINAMATH_CALUDE_output_is_six_l2953_295357

def program_output (a : ℕ) : ℕ :=
  if a < 10 then 2 * a else a * a

theorem output_is_six : program_output 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_output_is_six_l2953_295357


namespace NUMINAMATH_CALUDE_fraction_addition_l2953_295308

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2953_295308


namespace NUMINAMATH_CALUDE_caesars_meal_cost_proof_l2953_295374

/-- The cost per meal at Caesar's banquet hall -/
def caesars_meal_cost : ℝ := 30

/-- The number of guests attending the prom -/
def num_guests : ℕ := 60

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's cost per meal -/
def venus_meal_cost : ℝ := 35

theorem caesars_meal_cost_proof :
  caesars_room_fee + num_guests * caesars_meal_cost =
  venus_room_fee + num_guests * venus_meal_cost :=
by sorry

end NUMINAMATH_CALUDE_caesars_meal_cost_proof_l2953_295374


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2953_295317

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := Real.pi * r₁ ^ 2  -- area of smaller circle
  let A₂ : ℝ := Real.pi * r₂ ^ 2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l2953_295317


namespace NUMINAMATH_CALUDE_equation_with_geometric_progression_roots_l2953_295388

theorem equation_with_geometric_progression_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ) (q : ℝ),
  (x₁ ≠ x₂) ∧ (x₁ ≠ x₃) ∧ (x₁ ≠ x₄) ∧ (x₂ ≠ x₃) ∧ (x₂ ≠ x₄) ∧ (x₃ ≠ x₄) ∧
  (q ≠ 1) ∧ (q > 0) ∧
  (x₂ = q * x₁) ∧ (x₃ = q * x₂) ∧ (x₄ = q * x₃) ∧
  (16 * x₁^4 - 170 * x₁^3 + 357 * x₁^2 - 170 * x₁ + 16 = 0) ∧
  (16 * x₂^4 - 170 * x₂^3 + 357 * x₂^2 - 170 * x₂ + 16 = 0) ∧
  (16 * x₃^4 - 170 * x₃^3 + 357 * x₃^2 - 170 * x₃ + 16 = 0) ∧
  (16 * x₄^4 - 170 * x₄^3 + 357 * x₄^2 - 170 * x₄ + 16 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_with_geometric_progression_roots_l2953_295388


namespace NUMINAMATH_CALUDE_band_gigs_played_l2953_295343

theorem band_gigs_played (earnings_per_member : ℕ) (num_members : ℕ) (total_earnings : ℕ) : 
  earnings_per_member = 20 →
  num_members = 4 →
  total_earnings = 400 →
  total_earnings / (earnings_per_member * num_members) = 5 := by
sorry

end NUMINAMATH_CALUDE_band_gigs_played_l2953_295343


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2953_295394

theorem perfect_square_trinomial_factorization (x : ℝ) :
  x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_factorization_l2953_295394


namespace NUMINAMATH_CALUDE_traffic_light_color_change_probability_l2953_295335

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total time when color changes occur -/
def colorChangeDuration (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds for each color change

/-- Theorem: The probability of observing a color change is 3/20 -/
theorem traffic_light_color_change_probability
  (cycle : TrafficLightCycle)
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 50)
  (h4 : colorChangeDuration cycle = 15) :
  (colorChangeDuration cycle : ℚ) / (cycleDuration cycle : ℚ) = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_color_change_probability_l2953_295335


namespace NUMINAMATH_CALUDE_equation_solution_l2953_295380

theorem equation_solution : ∀ x : ℝ, (9 / x^2 = x / 25) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2953_295380


namespace NUMINAMATH_CALUDE_product_divisible_by_twelve_l2953_295331

theorem product_divisible_by_twelve (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_twelve_l2953_295331


namespace NUMINAMATH_CALUDE_f_minimum_value_l2953_295318

def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem f_minimum_value :
  (∀ x y : ℝ, f x y ≥ 21.2) ∧ f 8 1 = 21.2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2953_295318


namespace NUMINAMATH_CALUDE_product_purchase_discount_l2953_295364

/-- Proves that if a product is sold for $439.99999999999966 with a 10% profit, 
    and if buying it for x% less and selling at 30% profit would yield $28 more, 
    then x = 10%. -/
theorem product_purchase_discount (x : Real) : 
  (1.1 * (439.99999999999966 / 1.1) = 439.99999999999966) →
  (1.3 * (1 - x/100) * (439.99999999999966 / 1.1) = 439.99999999999966 + 28) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_discount_l2953_295364


namespace NUMINAMATH_CALUDE_geometric_relations_l2953_295319

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → perpendicular_planes α β) ∧
  (perpendicular_plane m α ∧ parallel n β ∧ parallel_planes α β → perpendicular m n) := by
  sorry

end NUMINAMATH_CALUDE_geometric_relations_l2953_295319


namespace NUMINAMATH_CALUDE_tea_cups_problem_l2953_295370

theorem tea_cups_problem (total_tea : ℕ) (cup_capacity : ℕ) (h1 : total_tea = 1050) (h2 : cup_capacity = 65) :
  (total_tea / cup_capacity : ℕ) = 16 :=
by sorry

end NUMINAMATH_CALUDE_tea_cups_problem_l2953_295370


namespace NUMINAMATH_CALUDE_A_power_98_l2953_295359

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A^98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by
  sorry

end NUMINAMATH_CALUDE_A_power_98_l2953_295359


namespace NUMINAMATH_CALUDE_smallest_a_for_two_zeros_in_unit_interval_l2953_295314

theorem smallest_a_for_two_zeros_in_unit_interval :
  ∃ (a b c : ℤ), 
    a = 5 ∧
    (∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
      a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∀ (a' b' c' : ℤ), a' > 0 ∧ a' < 5 →
      ¬(∃ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≠ y ∧
        a' * x^2 - b' * x + c' = 0 ∧ a' * y^2 - b' * y + c' = 0)) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_zeros_in_unit_interval_l2953_295314


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_15_18_l2953_295337

theorem smallest_divisible_by_12_15_18 : ∃ n : ℕ+, (∀ m : ℕ+, 12 ∣ m ∧ 15 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_15_18_l2953_295337


namespace NUMINAMATH_CALUDE_tan_range_problem_l2953_295340

open Real Set

theorem tan_range_problem (m : ℝ) : 
  (∃ x ∈ Icc 0 (π/4), ¬(tan x < m)) ↔ m ∈ Iic 1 :=
sorry

end NUMINAMATH_CALUDE_tan_range_problem_l2953_295340


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2953_295365

/-- A geometric sequence with first term 3 and sum of first, third, and fifth terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = 3 * q ^ (n - 1)) ∧
  a 1 + a 3 + a 5 = 21

/-- The sum of the third, fifth, and seventh terms of the sequence is 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 3 + a 5 + a 7 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2953_295365


namespace NUMINAMATH_CALUDE_binomial_coefficient_even_l2953_295347

theorem binomial_coefficient_even (n : ℕ) (h : Even n) (h2 : n > 0) : 
  Nat.choose n 2 = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_even_l2953_295347


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2953_295305

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n + (n + 4) = 150) → (n + (n + 2) + (n + 4) = 225) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2953_295305


namespace NUMINAMATH_CALUDE_balloon_permutations_count_l2953_295320

/-- The number of distinct permutations of a 7-letter word with two pairs of repeated letters -/
def balloon_permutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "balloon" is 1260 -/
theorem balloon_permutations_count : balloon_permutations = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_count_l2953_295320


namespace NUMINAMATH_CALUDE_floor_negative_two_point_eight_l2953_295386

theorem floor_negative_two_point_eight :
  ⌊(-2.8 : ℝ)⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_negative_two_point_eight_l2953_295386


namespace NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l2953_295392

theorem greatest_prime_factor_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 18) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 18) → q ≤ p :=
by
  -- The proof would go here
  sorry

#eval 4897 -- This will output the expected result

end NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l2953_295392
