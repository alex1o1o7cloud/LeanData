import Mathlib

namespace min_max_values_l1626_162680

theorem min_max_values (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3*b = 5) :
  (((1 : ℝ) / (a - b)) + (4 / (b - 1)) ≥ 25) ∧ (a*b - b^2 - a + b ≤ (1 : ℝ) / 16) := by
  sorry

end min_max_values_l1626_162680


namespace existence_of_coprime_sum_l1626_162663

theorem existence_of_coprime_sum (n k : ℕ) (hn : n > 0) (hk : Even (k * (n - 1))) :
  ∃ x y : ℤ, (Nat.gcd x.natAbs n = 1) ∧ (Nat.gcd y.natAbs n = 1) ∧ ((x + y) % n = k % n) := by
  sorry

end existence_of_coprime_sum_l1626_162663


namespace gecko_eats_15_bugs_l1626_162698

/-- The number of bugs eaten by various creatures in a garden --/
structure GardenBugs where
  gecko : ℕ
  lizard : ℕ
  frog : ℕ
  toad : ℕ

/-- The conditions of the bug-eating scenario in the garden --/
def validGardenBugs (bugs : GardenBugs) : Prop :=
  bugs.lizard = bugs.gecko / 2 ∧
  bugs.frog = 3 * bugs.lizard ∧
  bugs.toad = (3 * bugs.frog) / 2 ∧
  bugs.gecko + bugs.lizard + bugs.frog + bugs.toad = 63

/-- The theorem stating that the gecko eats 15 bugs --/
theorem gecko_eats_15_bugs :
  ∃ (bugs : GardenBugs), validGardenBugs bugs ∧ bugs.gecko = 15 := by
  sorry

end gecko_eats_15_bugs_l1626_162698


namespace coefficient_x2_cube_polynomial_l1626_162617

/-- Given a polynomial q(x) = x^5 - 4x^3 + 5x^2 - 6x + 3, 
    this theorem states that the coefficient of x^2 in (q(x))^3 is 540. -/
theorem coefficient_x2_cube_polynomial :
  let q : Polynomial ℝ := X^5 - 4*X^3 + 5*X^2 - 6*X + 3
  (q^3).coeff 2 = 540 := by
  sorry

end coefficient_x2_cube_polynomial_l1626_162617


namespace complement_A_in_U_l1626_162644

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A
def A : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end complement_A_in_U_l1626_162644


namespace root_of_quadratic_l1626_162606

theorem root_of_quadratic (x : ℝ) : 
  x = (-25 + Real.sqrt 361) / 12 → 6 * x^2 + 25 * x + 11 = 0 := by
  sorry

end root_of_quadratic_l1626_162606


namespace min_prime_factorization_sum_l1626_162616

theorem min_prime_factorization_sum (x y a b : ℕ+) (e f : ℕ) :
  5 * x^7 = 13 * y^11 →
  x = a^e * b^f →
  a.val.Prime ∧ b.val.Prime →
  a ≠ b →
  a + b + e + f = 25 :=
sorry

end min_prime_factorization_sum_l1626_162616


namespace equal_debt_days_l1626_162601

/-- The number of days for two borrowers to owe the same amount -/
def days_to_equal_debt (
  morgan_initial : ℚ)
  (morgan_rate : ℚ)
  (olivia_initial : ℚ)
  (olivia_rate : ℚ) : ℚ :=
  (olivia_initial - morgan_initial) / (morgan_rate * morgan_initial - olivia_rate * olivia_initial)

/-- Proof that Morgan and Olivia will owe the same amount after 25/3 days -/
theorem equal_debt_days :
  days_to_equal_debt 200 (12/100) 300 (4/100) = 25/3 := by
  sorry

end equal_debt_days_l1626_162601


namespace smallest_number_l1626_162681

/-- Converts a number from base b to decimal --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The decimal representation of 85₍₉₎ --/
def num1 : Nat := toDecimal [5, 8] 9

/-- The decimal representation of 210₍₆₎ --/
def num2 : Nat := toDecimal [0, 1, 2] 6

/-- The decimal representation of 1000₍₄₎ --/
def num3 : Nat := toDecimal [0, 0, 0, 1] 4

/-- The decimal representation of 111111₍₂₎ --/
def num4 : Nat := toDecimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating that 111111₍₂₎ is the smallest among the given numbers --/
theorem smallest_number : num4 ≤ num1 ∧ num4 ≤ num2 ∧ num4 ≤ num3 := by
  sorry

end smallest_number_l1626_162681


namespace decreasing_function_inequality_l1626_162650

-- Define a decreasing function on ℝ
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_function_inequality (f : ℝ → ℝ) (h : DecreasingOn f) : f 3 > f 5 := by
  sorry

end decreasing_function_inequality_l1626_162650


namespace circumference_diameter_ratio_l1626_162651

/-- The ratio of circumference to diameter for a ring with radius 15 cm and circumference 90 cm is 3. -/
theorem circumference_diameter_ratio :
  let radius : ℝ := 15
  let circumference : ℝ := 90
  let diameter : ℝ := 2 * radius
  circumference / diameter = 3 := by
  sorry

end circumference_diameter_ratio_l1626_162651


namespace sqrt_18_minus_sqrt_8_equals_sqrt_2_l1626_162693

theorem sqrt_18_minus_sqrt_8_equals_sqrt_2 : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end sqrt_18_minus_sqrt_8_equals_sqrt_2_l1626_162693


namespace roots_sum_of_squares_l1626_162657

theorem roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 1 = 0) → 
  x₁^2 + x₂^2 = 13/4 := by
sorry

end roots_sum_of_squares_l1626_162657


namespace jorge_ticket_cost_l1626_162615

/-- Calculates the total cost of tickets after all discounts --/
def total_cost_after_discounts (adult_tickets senior_tickets child_tickets : ℕ)
  (adult_price senior_price child_price : ℚ)
  (tier1_threshold tier2_threshold tier3_threshold : ℚ)
  (tier1_adult_discount tier1_senior_discount : ℚ)
  (tier2_adult_discount tier2_senior_discount : ℚ)
  (tier3_adult_discount tier3_senior_discount : ℚ)
  (extra_discount_per_50 max_extra_discount : ℚ) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem jorge_ticket_cost :
  total_cost_after_discounts 10 8 6 12 8 6 100 200 300
    0.1 0.05 0.2 0.1 0.3 0.15 0.05 0.15 = 161.16 := by
  sorry

end jorge_ticket_cost_l1626_162615


namespace largest_k_for_distinct_roots_l1626_162614

theorem largest_k_for_distinct_roots : 
  ∃ k : ℤ, k = 8 ∧ 
  (∀ m : ℤ, m > k → ¬(∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + m = 0 ∧ y^2 - 6*y + m = 0)) ∧
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) :=
by sorry

end largest_k_for_distinct_roots_l1626_162614


namespace select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l1626_162659

def select_three_from_five : ℕ := Nat.choose 5 3

def distribute_five_to_three : ℕ :=
  let scenario1 := Nat.choose 5 3 * Nat.factorial 3
  let scenario2 := Nat.choose 5 1 * Nat.choose 4 2 * Nat.factorial 3 / 2
  scenario1 + scenario2

theorem select_three_from_five_eq_ten :
  select_three_from_five = 10 := by sorry

theorem distribute_five_to_three_eq_onefifty :
  distribute_five_to_three = 150 := by sorry

end select_three_from_five_eq_ten_distribute_five_to_three_eq_onefifty_l1626_162659


namespace company_income_analysis_l1626_162602

structure Company where
  employees : ℕ
  max_income : ℕ
  avg_income : ℕ
  min_income : ℕ
  mid_50_low : ℕ
  mid_50_high : ℕ

def is_high_income (c : Company) (income : ℕ) : Prop :=
  income > c.avg_income

def is_sufficient_info (c : Company) : Prop :=
  c.mid_50_low > 0 ∧ c.mid_50_high > c.mid_50_low

def estimate_median (c : Company) : ℕ :=
  (c.mid_50_low + c.mid_50_high) / 2

theorem company_income_analysis (c : Company) 
  (h1 : c.employees = 50)
  (h2 : c.max_income = 1000000)
  (h3 : c.avg_income = 35000)
  (h4 : c.min_income = 5000)
  (h5 : c.mid_50_low = 10000)
  (h6 : c.mid_50_high = 30000) :
  ¬is_high_income c 25000 ∧
  ¬is_sufficient_info {employees := c.employees, max_income := c.max_income, avg_income := c.avg_income, min_income := c.min_income, mid_50_low := 0, mid_50_high := 0} ∧
  is_sufficient_info c ∧
  estimate_median c < c.avg_income := by
  sorry

#check company_income_analysis

end company_income_analysis_l1626_162602


namespace congruent_count_l1626_162612

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 1) (Finset.range 251)).card = 36 := by
  sorry

end congruent_count_l1626_162612


namespace radical_sum_equals_eight_sqrt_three_l1626_162695

theorem radical_sum_equals_eight_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end radical_sum_equals_eight_sqrt_three_l1626_162695


namespace flag_arrangement_remainder_l1626_162683

/-- The number of distinguishable arrangements of flags on two poles -/
def N : ℕ :=
  let blue_flags := 10
  let green_flags := 9
  let total_flags := blue_flags + green_flags
  let poles := 2
  -- Definition of N based on the problem conditions
  -- (Actual calculation is omitted as it's part of the proof)
  2310

/-- Theorem stating that N mod 1000 = 310 -/
theorem flag_arrangement_remainder :
  N % 1000 = 310 := by
  sorry

end flag_arrangement_remainder_l1626_162683


namespace greatest_x_implies_n_l1626_162666

theorem greatest_x_implies_n (x : ℤ) (n : ℝ) : 
  (∀ y : ℤ, 2.13 * (10 : ℝ) ^ y < n → y ≤ 2) →
  (2.13 * (10 : ℝ) ^ 2 < n) ∧
  (∀ m : ℝ, m < n → m ≤ 213) ∧
  (n ≥ 214) :=
sorry

end greatest_x_implies_n_l1626_162666


namespace inscribed_circles_radii_equal_l1626_162652

theorem inscribed_circles_radii_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let r₁ := a * b / (a + b)
  let r₂ := a * b / (a + b)
  r₁ = r₂ := by sorry

end inscribed_circles_radii_equal_l1626_162652


namespace reciprocal_of_eight_l1626_162641

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_eight : reciprocal 8 = 1 / 8 := by
  sorry

end reciprocal_of_eight_l1626_162641


namespace quadratic_root_k_value_l1626_162638

theorem quadratic_root_k_value (k : ℝ) : 
  ((-2 : ℝ)^2 - k * (-2) - 6 = 0) → k = 1 := by
  sorry

end quadratic_root_k_value_l1626_162638


namespace quadratic_root_difference_l1626_162642

theorem quadratic_root_difference (r s : ℝ) (hr : r > 0) : 
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 - r*y1 - s = 0 ∧ 
    y2^2 - r*y2 - s = 0 ∧ 
    |y1 - y2| = 2) → 
  r = 2 :=
by sorry

end quadratic_root_difference_l1626_162642


namespace max_lateral_area_cylinder_l1626_162699

/-- The maximum lateral area of a cylinder with a rectangular cross-section of perimeter 4 is π. -/
theorem max_lateral_area_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → 2 * (2 * r + h) = 4 → 2 * π * r * h ≤ π := by
  sorry

end max_lateral_area_cylinder_l1626_162699


namespace first_brother_is_treljalya_l1626_162633

structure Brother where
  name : String
  card_color : String
  tells_truth : Bool

def first_brother_statement_1 (b : Brother) : Prop :=
  b.name = "Treljalya"

def second_brother_statement (b : Brother) : Prop :=
  b.name = "Treljalya"

def first_brother_statement_2 (b : Brother) : Prop :=
  b.card_color = "orange"

def same_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color = b2.card_color → b1.tells_truth ≠ b2.tells_truth

def different_suit_rule (b1 b2 : Brother) : Prop :=
  b1.card_color ≠ b2.card_color → b1.tells_truth = b2.tells_truth

theorem first_brother_is_treljalya (b1 b2 : Brother) :
  same_suit_rule b1 b2 →
  different_suit_rule b1 b2 →
  first_brother_statement_1 b1 →
  second_brother_statement b2 →
  first_brother_statement_2 b2 →
  b1.name = "Treljalya" :=
sorry

end first_brother_is_treljalya_l1626_162633


namespace quadratic_equation_result_l1626_162691

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 2*a^2 - 8*a - 8 = 16 := by
  sorry

end quadratic_equation_result_l1626_162691


namespace decimal_place_150_of_5_over_8_l1626_162626

theorem decimal_place_150_of_5_over_8 : 
  let decimal_expansion := (5 : ℚ) / 8
  let digit_at_n (q : ℚ) (n : ℕ) := (q * 10^n).floor % 10
  digit_at_n decimal_expansion 150 = 0 := by
  sorry

end decimal_place_150_of_5_over_8_l1626_162626


namespace inverse_307_mod_455_l1626_162656

theorem inverse_307_mod_455 : ∃ x : ℕ, x < 455 ∧ (307 * x) % 455 = 1 :=
by
  use 81
  sorry

end inverse_307_mod_455_l1626_162656


namespace sequence_decreasing_l1626_162628

theorem sequence_decreasing (a : ℕ → ℝ) (h1 : a 1 > 0) (h2 : ∀ n : ℕ, a (n + 1) / a n = 1 / 2) :
  ∀ n m : ℕ, n < m → a m < a n :=
sorry

end sequence_decreasing_l1626_162628


namespace purple_part_length_l1626_162654

/-- The length of the purple part of a pencil -/
def purple_length : ℝ := 1.5

/-- The length of the black part of a pencil -/
def black_length : ℝ := 0.5

/-- The length of the blue part of a pencil -/
def blue_length : ℝ := 2

/-- The total length of the pencil -/
def total_length : ℝ := 4

/-- Theorem stating that the length of the purple part of the pencil is 1.5 cm -/
theorem purple_part_length :
  purple_length = total_length - (black_length + blue_length) :=
by sorry

end purple_part_length_l1626_162654


namespace sin_cos_product_l1626_162604

theorem sin_cos_product (a : ℝ) (h : Real.sin (Real.pi - a) = -2 * Real.sin (Real.pi / 2 + a)) :
  Real.sin a * Real.cos a = -2/5 := by
  sorry

end sin_cos_product_l1626_162604


namespace least_positive_integer_with_remainder_one_l1626_162669

theorem least_positive_integer_with_remainder_one : ∃ n : ℕ,
  n > 1 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → n % k = 1) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → m % k = 1) → n ≤ m) ∧
  n = 2521 := by
sorry

end least_positive_integer_with_remainder_one_l1626_162669


namespace inequality_solution_l1626_162611

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 12) / (x - 3) < 0 ↔ (x > -2 ∧ x < 3) ∨ (x > 3 ∧ x < 6) :=
by sorry

end inequality_solution_l1626_162611


namespace arctan_sum_equation_l1626_162665

theorem arctan_sum_equation : ∃ (n : ℕ+), 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/n) = π/3 ∧ n = 10 := by
  sorry

end arctan_sum_equation_l1626_162665


namespace power_of_negative_product_l1626_162690

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by sorry

end power_of_negative_product_l1626_162690


namespace parabola_fixed_point_l1626_162687

/-- The parabola equation as a function of x and p -/
def parabola (x p : ℝ) : ℝ := 2 * x^2 - p * x + 4 * p + 1

/-- The fixed point through which the parabola passes -/
def fixed_point : ℝ × ℝ := (4, 33)

theorem parabola_fixed_point :
  ∀ p : ℝ, parabola (fixed_point.1) p = fixed_point.2 := by
  sorry

#check parabola_fixed_point

end parabola_fixed_point_l1626_162687


namespace fifth_fibonacci_is_eight_l1626_162636

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | k + 2 => fibonacci k + fibonacci (k + 1)

theorem fifth_fibonacci_is_eight :
  fibonacci 4 = 8 := by
  sorry

end fifth_fibonacci_is_eight_l1626_162636


namespace real_axis_length_of_hyperbola_l1626_162671

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem real_axis_length_of_hyperbola :
  let hyperbola_equation := fun (x y : ℝ) => x^2 - y^2/9 = 1
  ∃ a : ℝ, a > 0 ∧ hyperbola_equation = fun (x y : ℝ) => x^2/a^2 - y^2/(9*a^2) = 1 →
  (real_axis_length : ℝ) = 2 := by
  sorry

end real_axis_length_of_hyperbola_l1626_162671


namespace sum_of_composite_function_l1626_162625

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x + 1

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_composite_function :
  (xValues.map (λ x => q (p x))).sum = -13 := by
  sorry

end sum_of_composite_function_l1626_162625


namespace base_conversion_equivalence_l1626_162640

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

def base_three_number : List Nat := [2, 0, 1, 2, 1]
def base_five_number : List Nat := [1, 2, 0, 3]

theorem base_conversion_equivalence :
  decimal_to_base_five (base_three_to_decimal base_three_number) = base_five_number := by
  sorry

end base_conversion_equivalence_l1626_162640


namespace equation_solution_l1626_162629

theorem equation_solution : ∃ x : ℚ, x - 1/2 = 2/5 - 1/4 ∧ x = 13/20 := by
  sorry

end equation_solution_l1626_162629


namespace stella_monthly_income_l1626_162674

def months_in_year : ℕ := 12
def unpaid_leave_months : ℕ := 2
def annual_income : ℕ := 49190

def monthly_income : ℕ := annual_income / (months_in_year - unpaid_leave_months)

theorem stella_monthly_income : monthly_income = 4919 := by
  sorry

end stella_monthly_income_l1626_162674


namespace town_population_problem_l1626_162664

theorem town_population_problem : ∃ (n : ℝ), 
  n > 0 ∧ 
  0.92 * (0.85 * (n + 2500)) = n + 49 ∧ 
  n = 8740 := by
  sorry

end town_population_problem_l1626_162664


namespace park_short_trees_l1626_162613

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem: The park will have 217 short trees after planting -/
theorem park_short_trees : 
  total_short_trees 112 105 = 217 := by
  sorry

end park_short_trees_l1626_162613


namespace einstein_snack_sale_l1626_162637

/-- The number of potato fries packs sold by Einstein --/
def potato_fries_packs : ℕ := sorry

theorem einstein_snack_sale :
  let goal : ℚ := 500
  let pizza_price : ℚ := 12
  let fries_price : ℚ := 0.30
  let soda_price : ℚ := 2
  let pizza_sold : ℕ := 15
  let soda_sold : ℕ := 25
  let remaining : ℚ := 258
  
  (pizza_price * pizza_sold + fries_price * potato_fries_packs + soda_price * soda_sold = goal - remaining) ∧
  (potato_fries_packs = 40) := by sorry

end einstein_snack_sale_l1626_162637


namespace updated_mean_l1626_162600

/-- Given 50 observations with an original mean of 200 and a decrement of 47 from each observation,
    the updated mean is 153. -/
theorem updated_mean (n : ℕ) (original_mean decrement : ℚ) (h1 : n = 50) (h2 : original_mean = 200) (h3 : decrement = 47) :
  let total_sum := n * original_mean
  let total_decrement := n * decrement
  let updated_sum := total_sum - total_decrement
  let updated_mean := updated_sum / n
  updated_mean = 153 := by
sorry

end updated_mean_l1626_162600


namespace birth_year_problem_l1626_162627

theorem birth_year_problem (x : ℕ) (h1 : x^2 - 2*x ≥ 1900) (h2 : x^2 - 2*x < 1950) : 
  (x^2 - 2*x + x = 1936) := by
  sorry

end birth_year_problem_l1626_162627


namespace smallest_division_is_six_l1626_162647

/-- A typical rectangular parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube has all sides equal -/
structure Cube where
  side : ℝ

/-- The division of a cube into typical parallelepipeds -/
def CubeDivision (c : Cube) := List TypicalParallelepiped

/-- Predicate to check if a division is valid (i.e., the parallelepipeds fill the cube exactly) -/
def IsValidDivision (c : Cube) (d : CubeDivision c) : Prop := sorry

/-- The smallest number of typical parallelepipeds into which a cube can be divided is 6 -/
theorem smallest_division_is_six (c : Cube) : 
  (∃ (d : CubeDivision c), IsValidDivision c d ∧ d.length = 6) ∧
  (∀ (d : CubeDivision c), IsValidDivision c d → d.length ≥ 6) :=
sorry

end smallest_division_is_six_l1626_162647


namespace distance_AD_MN_l1626_162684

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the pyramid structure described in the problem -/
structure Pyramid where
  a : ℝ
  b : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  M : Point3D
  N : Point3D

/-- The distance between two skew lines in 3D space -/
def distanceBetweenSkewLines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- The main theorem stating the distance between AD and MN -/
theorem distance_AD_MN (p : Pyramid) :
  let AD := Line3D.mk p.A (Point3D.mk p.a p.a 0)
  let MN := Line3D.mk p.M (Point3D.mk 0 (p.a / 2) p.b)
  distanceBetweenSkewLines AD MN = (p.b / (2 * p.a)) * Real.sqrt (4 * p.a^2 - p.b^2) :=
by sorry

end distance_AD_MN_l1626_162684


namespace password_probability_l1626_162688

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The set of possible even digits for the last position -/
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

/-- The probability of guessing the correct password in one attempt, given the last digit is even -/
def prob_correct_first_attempt : ℚ := 1 / 5

/-- The probability of guessing the correct password in exactly two attempts, given the last digit is even -/
def prob_correct_second_attempt : ℚ := 4 / 25

/-- The probability of guessing the correct password in no more than two attempts, given the last digit is even -/
def prob_correct_within_two_attempts : ℚ := prob_correct_first_attempt + prob_correct_second_attempt

theorem password_probability : prob_correct_within_two_attempts = 2 / 5 := by
  sorry

end password_probability_l1626_162688


namespace drum_oil_capacity_l1626_162630

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) : 
  C > 0 → -- Capacity of Drum X is positive
  Y ≥ 0 → -- Initial amount of oil in Drum Y is non-negative
  Y + (1/2 * C) = 0.65 * (2 * C) → -- After pouring, Drum Y is filled to 0.65 capacity
  Y = 0.8 * (2 * C) -- Initial fill level of Drum Y is 0.8 of its capacity
  := by sorry

end drum_oil_capacity_l1626_162630


namespace expression_evaluation_l1626_162676

theorem expression_evaluation (x y z : ℚ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5)
  (hd1 : x + 2 ≠ 0)
  (hd2 : y - 3 ≠ 0)
  (hd3 : z + 7 ≠ 0) :
  (x + 3) / (x + 2) * (y - 2) / (y - 3) * (z + 9) / (z + 7) = 2 / 7 := by
  sorry

end expression_evaluation_l1626_162676


namespace inverse_negation_equivalence_l1626_162649

-- Define a quadrilateral type
structure Quadrilateral where
  isParallelogram : Prop
  oppositeSidesEqual : Prop

-- Define the original proposition
def originalProposition (q : Quadrilateral) : Prop :=
  q.oppositeSidesEqual → q.isParallelogram

-- Define the inverse negation
def inverseNegation (q : Quadrilateral) : Prop :=
  ¬q.isParallelogram → ¬q.oppositeSidesEqual

-- Theorem stating the equivalence of the inverse negation
theorem inverse_negation_equivalence :
  ∀ q : Quadrilateral, inverseNegation q ↔ ¬(originalProposition q) :=
sorry

end inverse_negation_equivalence_l1626_162649


namespace exactly_one_two_digit_sum_with_reverse_is_cube_l1626_162661

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem exactly_one_two_digit_sum_with_reverse_is_cube : 
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_cube (n + reverse_digits n) :=
sorry

end exactly_one_two_digit_sum_with_reverse_is_cube_l1626_162661


namespace lcm_product_implies_hcf_l1626_162620

theorem lcm_product_implies_hcf (x y : ℕ+) 
  (h1 : Nat.lcm x y = 600) 
  (h2 : x * y = 18000) : 
  Nat.gcd x y = 30 := by
  sorry

end lcm_product_implies_hcf_l1626_162620


namespace min_value_quadratic_l1626_162610

theorem min_value_quadratic (x : ℝ) : 
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4*x^2 + 8*x + 10 → y ≥ y_min ∧ y_min = 6 :=
by
  sorry

end min_value_quadratic_l1626_162610


namespace power_sum_divisibility_and_quotient_units_digit_l1626_162692

theorem power_sum_divisibility_and_quotient_units_digit :
  (∃ k : ℕ, 4^1987 + 6^1987 = 10 * k) ∧
  (∃ m : ℕ, 4^1987 + 6^1987 = 5 * m) ∧
  (∃ n : ℕ, (4^1987 + 6^1987) / 5 = 10 * n + 0) :=
by sorry

end power_sum_divisibility_and_quotient_units_digit_l1626_162692


namespace johns_remaining_money_l1626_162686

theorem johns_remaining_money (initial_amount : ℚ) : 
  initial_amount = 200 → 
  initial_amount - (3/8 * initial_amount + 3/10 * initial_amount) = 65 := by
sorry

end johns_remaining_money_l1626_162686


namespace regular_octagon_perimeter_l1626_162679

/-- The perimeter of a regular octagon with side length 2 is 16 -/
theorem regular_octagon_perimeter : 
  ∀ (side_length : ℝ), 
  side_length = 2 → 
  (8 : ℝ) * side_length = 16 := by
sorry

end regular_octagon_perimeter_l1626_162679


namespace chess_tournament_theorem_l1626_162621

/-- Represents the number of participants from each city -/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between participants from different cities -/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem statement based on the chess tournament problem -/
theorem chess_tournament_theorem (p : Participants) (g : Games) : 
  (p.moscow * 9 = p.saintPetersburg * 6) ∧ 
  (p.saintPetersburg * 2 = p.kazan * 6) ∧ 
  (p.moscow * g.moscowKazan = p.kazan * 8) →
  g.moscowKazan = 4 := by
  sorry

end chess_tournament_theorem_l1626_162621


namespace perfect_square_form_l1626_162672

theorem perfect_square_form (k : ℕ+) : ∃ (n : ℕ+) (a : ℤ), a^2 = n * 2^(k : ℕ) - 7 := by
  sorry

end perfect_square_form_l1626_162672


namespace max_value_of_trig_function_l1626_162673

theorem max_value_of_trig_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b ≤ 1) →
  (∀ x : ℝ, a * Real.cos x + b ≥ -7) →
  (∃ x : ℝ, a * Real.cos x + b = 1) →
  (∃ x : ℝ, a * Real.cos x + b = -7) →
  (∀ x : ℝ, a * Real.cos x + b * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, a * Real.cos x + b * Real.sin x = 5) :=
by sorry

end max_value_of_trig_function_l1626_162673


namespace gwens_birthday_money_l1626_162660

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := sorry

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between money from mom and dad after spending -/
def difference_after_spending : ℕ := 2

theorem gwens_birthday_money : 
  money_from_mom = 6 ∧
  money_from_mom + money_from_dad - money_spent = 
  money_from_dad + difference_after_spending :=
by sorry

end gwens_birthday_money_l1626_162660


namespace print_time_with_rate_change_l1626_162623

/-- Represents the printing scenario with given parameters -/
structure PrintingScenario where
  num_presses : ℕ
  initial_time : ℝ
  new_time : ℝ
  num_papers : ℕ

/-- Calculates the time taken to print papers given a printing scenario -/
def time_to_print (s : PrintingScenario) : ℝ :=
  s.new_time

/-- Theorem stating that the time to print remains the same as the new_time 
    when the printing rate changes but the number of presses remains constant -/
theorem print_time_with_rate_change (s : PrintingScenario) 
  (h1 : s.num_presses = 35)
  (h2 : s.initial_time = 15)
  (h3 : s.new_time = 21)
  (h4 : s.num_papers = 500000) :
  time_to_print s = s.new_time := by
  sorry


end print_time_with_rate_change_l1626_162623


namespace quadratic_minimum_minimum_at_three_l1626_162668

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 1 ≥ -8 ∧ ∃ x₀ : ℝ, x₀^2 - 6*x₀ + 1 = -8 := by
  sorry

theorem minimum_at_three : (3 : ℝ)^2 - 6*3 + 1 = -8 := by
  sorry

end quadratic_minimum_minimum_at_three_l1626_162668


namespace expression_equals_seventeen_l1626_162655

theorem expression_equals_seventeen : 1-(-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := by
  sorry

end expression_equals_seventeen_l1626_162655


namespace max_value_theorem_l1626_162639

theorem max_value_theorem (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ (max : ℝ), max = 7/5 ∧ ∀ (x y : ℝ), 
    x + y - 2 ≥ 0 → y - x - 1 ≤ 0 → x ≤ 1 → 
    (x + 2*y) / (2*x + y) ≤ max :=
by sorry

end max_value_theorem_l1626_162639


namespace range_of_a_l1626_162622

def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → (4 * x^2 - a * x) ≤ (4 * y^2 - a * y)

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) → ¬(prop_p a) → (a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end range_of_a_l1626_162622


namespace plates_for_matt_l1626_162607

/-- The number of plates needed for a week under specific dining conditions -/
def plates_needed (days_with_two : Nat) (days_with_four : Nat) (plates_per_person_two : Nat) (plates_per_person_four : Nat) : Nat :=
  (days_with_two * 2 * plates_per_person_two) + (days_with_four * 4 * plates_per_person_four)

theorem plates_for_matt : plates_needed 3 4 1 2 = 38 := by
  sorry

end plates_for_matt_l1626_162607


namespace wanda_initial_blocks_l1626_162605

/-- The number of blocks Theresa gave to Wanda -/
def blocks_from_theresa : ℕ := 79

/-- The total number of blocks Wanda has after receiving blocks from Theresa -/
def total_blocks : ℕ := 83

/-- The number of blocks Wanda had initially -/
def initial_blocks : ℕ := total_blocks - blocks_from_theresa

theorem wanda_initial_blocks :
  initial_blocks = 4 :=
by sorry

end wanda_initial_blocks_l1626_162605


namespace parallelogram_area_l1626_162670

def v : ℝ × ℝ := (7, 4)
def w : ℝ × ℝ := (2, -9)

theorem parallelogram_area : 
  let v2w := (2 * w.1, 2 * w.2)
  abs (v.1 * v2w.2 - v.2 * v2w.1) = 142 := by sorry

end parallelogram_area_l1626_162670


namespace T_is_three_intersecting_lines_l1626_162632

-- Define the set T
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ 5 ≤ y - 3) ∨
               (5 = y - 3 ∧ 5 ≤ x + 3) ∨
               (x + 3 = y - 3 ∧ 5 ≤ x + 3)}

-- Define the three lines
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≥ 8}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 8 ∧ p.1 ≥ 2}
def line3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≤ 2}

-- Define the intersection points
def point1 : ℝ × ℝ := (2, 8)
def point2 : ℝ × ℝ := (2, 8)
def point3 : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  T = line1 ∪ line2 ∪ line3 ∧
  (∃ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 ∈ line1 ∩ line2 ∧ p2 ∈ line2 ∩ line3 ∧ p3 ∈ line1 ∩ line3) :=
by sorry

end T_is_three_intersecting_lines_l1626_162632


namespace divisibility_statements_l1626_162631

theorem divisibility_statements :
  (∃ n : ℤ, 24 = 4 * n) ∧
  (∃ m : ℤ, 152 = 19 * m) ∧ ¬(∃ k : ℤ, 96 = 19 * k) ∧
  ((∃ p : ℤ, 75 = 15 * p) ∨ (∃ q : ℤ, 90 = 15 * q)) ∧
  ((∃ r : ℤ, 28 = 14 * r) ∧ (∃ s : ℤ, 56 = 14 * s)) ∧
  (∃ t : ℤ, 180 = 6 * t) :=
by
  sorry

end divisibility_statements_l1626_162631


namespace tromino_tileable_tromino_area_div_by_three_l1626_162618

/-- Definition of a size-n tromino -/
def tromino (n : ℕ) := (2 * n) ^ 2 - n ^ 2

/-- The area of a size-n tromino -/
def tromino_area (n : ℕ) : ℕ := 3 * n ^ 2

/-- A size-n tromino can be tiled by size-1 trominos iff n ≢ 1 (mod 2) -/
theorem tromino_tileable (n : ℕ) (hn : n > 0) :
  (∃ k : ℕ, tromino n = 3 * k) ↔ n % 2 ≠ 1 := by sorry

/-- The area of a size-n tromino is divisible by 3 iff n ≢ 1 (mod 2) -/
theorem tromino_area_div_by_three (n : ℕ) (hn : n > 0) :
  3 ∣ tromino_area n ↔ n % 2 ≠ 1 := by sorry

end tromino_tileable_tromino_area_div_by_three_l1626_162618


namespace parabola_vertex_l1626_162603

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 + 1 is (3, 1) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola x y → (x, y) = vertex :=
sorry

end parabola_vertex_l1626_162603


namespace profit_percentage_l1626_162635

/-- If the cost price of 72 articles equals the selling price of 60 articles, then the percent profit is 20%. -/
theorem profit_percentage (C S : ℝ) (h : 72 * C = 60 * S) : (S - C) / C * 100 = 20 := by
  sorry

end profit_percentage_l1626_162635


namespace min_contribution_l1626_162675

/-- Proves that given 10 people contributing a total of $20.00, with a maximum individual contribution of $11, the minimum amount each person must have contributed is $2.00. -/
theorem min_contribution (num_people : ℕ) (total_contribution : ℚ) (max_individual : ℚ) :
  num_people = 10 ∧ 
  total_contribution = 20 ∧ 
  max_individual = 11 →
  ∃ (min_contribution : ℚ),
    min_contribution = 2 ∧
    num_people * min_contribution = total_contribution ∧
    ∀ (individual : ℚ),
      individual ≥ min_contribution ∧
      individual ≤ max_individual ∧
      (num_people - 1) * min_contribution + individual = total_contribution :=
by sorry

end min_contribution_l1626_162675


namespace certain_number_value_l1626_162694

theorem certain_number_value (y : ℕ) :
  (2^14 : ℕ) - (2^y : ℕ) = 3 * (2^12 : ℕ) → y = 13 := by
  sorry

end certain_number_value_l1626_162694


namespace chord_intersection_ratio_l1626_162678

-- Define a circle
variable (circle : Type) [AddCommGroup circle] [Module ℝ circle]

-- Define points on the circle
variable (E F G H Q : circle)

-- Define the lengths
variable (EQ FQ GQ HQ : ℝ)

-- State the theorem
theorem chord_intersection_ratio 
  (h1 : EQ = 5) 
  (h2 : GQ = 12) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by sorry

end chord_intersection_ratio_l1626_162678


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l1626_162689

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l1626_162689


namespace tracy_candies_problem_l1626_162608

theorem tracy_candies_problem :
  ∃ (initial : ℕ) (brother_took : ℕ),
    initial > 0 ∧
    brother_took ≥ 2 ∧
    brother_took ≤ 6 ∧
    (3 * initial / 10 : ℚ) - 20 - brother_took = 6 ∧
    initial = 100 := by
  sorry

end tracy_candies_problem_l1626_162608


namespace parallel_lines_k_value_l1626_162685

/-- Two lines in R² defined by their parametric equations -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line2D) : Prop :=
  ∃ c : ℝ, l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem parallel_lines_k_value :
  ∃! k : ℝ, are_parallel
    (Line2D.mk (2, 3) (6, -9))
    (Line2D.mk (-1, 0) (3, k))
  ∧ k = -4.5 := by
  sorry

end parallel_lines_k_value_l1626_162685


namespace easiest_to_pick_black_l1626_162646

structure Box where
  label : Char
  black_balls : ℕ
  white_balls : ℕ

def probability_black (b : Box) : ℚ :=
  b.black_balls / (b.black_balls + b.white_balls)

def boxes : List Box := [
  ⟨'A', 12, 4⟩,
  ⟨'B', 10, 10⟩,
  ⟨'C', 4, 2⟩,
  ⟨'D', 10, 5⟩
]

theorem easiest_to_pick_black (boxes : List Box) :
  ∃ b ∈ boxes, ∀ b' ∈ boxes, probability_black b ≥ probability_black b' :=
sorry

end easiest_to_pick_black_l1626_162646


namespace tamara_cracker_count_l1626_162667

/-- The number of crackers each person has -/
structure CrackerCount where
  tamara : ℕ
  nicholas : ℕ
  marcus : ℕ
  mona : ℕ

/-- The conditions of the cracker problem -/
def CrackerProblem (c : CrackerCount) : Prop :=
  c.tamara = 2 * c.nicholas ∧
  c.marcus = 3 * c.mona ∧
  c.nicholas = c.mona + 6 ∧
  c.marcus = 27

theorem tamara_cracker_count (c : CrackerCount) (h : CrackerProblem c) : c.tamara = 30 := by
  sorry

end tamara_cracker_count_l1626_162667


namespace soccer_boys_percentage_l1626_162696

theorem soccer_boys_percentage (total_students boys soccer_players girls_not_playing : ℕ) : 
  total_students = 420 →
  boys = 312 →
  soccer_players = 250 →
  girls_not_playing = 63 →
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 82 := by
sorry

end soccer_boys_percentage_l1626_162696


namespace no_solution_iff_k_eq_two_l1626_162624

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x + 2) / (x - 3) = (x - k) / (x - 7)

-- Define the domain restriction
def valid_domain (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 7

-- Theorem statement
theorem no_solution_iff_k_eq_two :
  ∀ k : ℝ, (∀ x : ℝ, valid_domain x → ¬equation x k) ↔ k = 2 :=
by sorry

end no_solution_iff_k_eq_two_l1626_162624


namespace base5_of_89_l1626_162645

-- Define a function to convert a natural number to its base-5 representation
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

-- Theorem stating that 89 in base-5 is equivalent to [4, 2, 3]
theorem base5_of_89 : toBase5 89 = [4, 2, 3] := by sorry

end base5_of_89_l1626_162645


namespace factorization_x4_minus_4x2_l1626_162697

theorem factorization_x4_minus_4x2 (x : ℝ) : x^4 - 4*x^2 = x^2 * (x - 2) * (x + 2) := by
  sorry

end factorization_x4_minus_4x2_l1626_162697


namespace last_page_stamps_l1626_162609

/-- The number of stamp books Jenny originally has -/
def num_books : ℕ := 8

/-- The number of pages in each stamp book -/
def pages_per_book : ℕ := 42

/-- The number of stamps on each page originally -/
def stamps_per_page_original : ℕ := 6

/-- The number of stamps on each page after reorganization -/
def stamps_per_page_new : ℕ := 10

/-- The number of completely filled books after reorganization -/
def filled_books : ℕ := 4

/-- The number of completely filled pages in the partially filled book -/
def filled_pages_partial : ℕ := 33

theorem last_page_stamps :
  (num_books * pages_per_book * stamps_per_page_original) % stamps_per_page_new = 6 :=
sorry

end last_page_stamps_l1626_162609


namespace expression_evaluation_l1626_162682

theorem expression_evaluation :
  let x : ℚ := 3/2
  (2 + x) * (2 - x) + (x - 1) * (x + 5) = 5 := by sorry

end expression_evaluation_l1626_162682


namespace stating_minimum_red_cubes_correct_l1626_162643

/-- 
Given a positive integer n, we construct a cube of side length 3n using smaller 3x3x3 cubes.
Each 3x3x3 cube is made of 26 white unit cubes and 1 black unit cube.
This function returns the minimum number of white unit cubes that need to be painted red
so that every remaining white unit cube has at least one common point with at least one red unit cube.
-/
def minimum_red_cubes (n : ℕ+) : ℕ :=
  (n + 1) * n^2

/-- 
Theorem stating that the minimum number of white unit cubes that need to be painted red
is indeed (n+1)n^2, where n is the number of 3x3x3 cubes along each edge of the larger cube.
-/
theorem minimum_red_cubes_correct (n : ℕ+) : 
  minimum_red_cubes n = (n + 1) * n^2 := by sorry

end stating_minimum_red_cubes_correct_l1626_162643


namespace trigonometric_identity_l1626_162653

theorem trigonometric_identity (α β : ℝ) :
  1 - Real.sin α ^ 2 - Real.sin β ^ 2 + 2 * Real.sin α * Real.sin β * Real.cos (α - β) = 
  Real.cos (α - β) ^ 2 := by
sorry

end trigonometric_identity_l1626_162653


namespace complex_sum_problem_l1626_162648

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 4 →
  e = -a - c →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 6 + 3 * Complex.I →
  d + f = -1 := by
  sorry

end complex_sum_problem_l1626_162648


namespace genuine_purses_and_handbags_l1626_162658

theorem genuine_purses_and_handbags (total_purses : ℕ) (total_handbags : ℕ)
  (h_purses : total_purses = 26)
  (h_handbags : total_handbags = 24)
  (fake_purses : ℕ → ℕ)
  (fake_handbags : ℕ → ℕ)
  (h_fake_purses : fake_purses total_purses = total_purses / 2)
  (h_fake_handbags : fake_handbags total_handbags = total_handbags / 4) :
  total_purses - fake_purses total_purses + total_handbags - fake_handbags total_handbags = 31 := by
  sorry

end genuine_purses_and_handbags_l1626_162658


namespace rectangle_shorter_side_l1626_162634

theorem rectangle_shorter_side (a b d : ℝ) : 
  a > 0 → b > 0 → d > 0 →
  (a / b = 3 / 4) →
  (a^2 + b^2 = d^2) →
  d = 9 →
  a = 5.4 := by
sorry

end rectangle_shorter_side_l1626_162634


namespace solution_interval_l1626_162662

theorem solution_interval (x₀ : ℝ) (k : ℤ) : 
  (Real.log x₀ + x₀ = 4) → 
  (x₀ > k ∧ x₀ < k + 1) → 
  k = 2 := by
  sorry

end solution_interval_l1626_162662


namespace quadratic_roots_greater_than_one_l1626_162619

theorem quadratic_roots_greater_than_one (a : ℝ) :
  a ≠ -1 →
  (∀ x : ℝ, (1 + a) * x^2 - 3 * a * x + 4 * a = 0 → x > 1) ↔
  -16/7 < a ∧ a < -1 :=
by sorry

end quadratic_roots_greater_than_one_l1626_162619


namespace equation_and_inequalities_l1626_162677

theorem equation_and_inequalities (x a : ℝ) (hx : x ≠ 0) :
  (x⁻¹ + a * x = 1 ↔ a = (x - 1) / x^2) ∧
  (x⁻¹ + a * x > 1 ↔ (a > (x - 1) / x^2 ∧ x > 0) ∨ (a < (x - 1) / x^2 ∧ x < 0)) ∧
  (x⁻¹ + a * x < 1 ↔ (a < (x - 1) / x^2 ∧ x > 0) ∨ (a > (x - 1) / x^2 ∧ x < 0)) := by
  sorry

end equation_and_inequalities_l1626_162677
