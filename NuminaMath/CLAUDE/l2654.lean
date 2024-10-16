import Mathlib

namespace NUMINAMATH_CALUDE_kids_bike_wheels_count_l2654_265461

/-- The number of wheels on a kid's bike -/
def kids_bike_wheels : ℕ := 4

/-- The number of regular bikes -/
def regular_bikes : ℕ := 7

/-- The number of children's bikes -/
def children_bikes : ℕ := 11

/-- The number of wheels on a regular bike -/
def regular_bike_wheels : ℕ := 2

/-- The total number of wheels observed -/
def total_wheels : ℕ := 58

theorem kids_bike_wheels_count :
  regular_bikes * regular_bike_wheels + children_bikes * kids_bike_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_kids_bike_wheels_count_l2654_265461


namespace NUMINAMATH_CALUDE_no_separable_representation_l2654_265425

theorem no_separable_representation :
  ¬ ∃ (f g : ℝ → ℝ), ∀ x y : ℝ, 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end NUMINAMATH_CALUDE_no_separable_representation_l2654_265425


namespace NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5082_l2654_265411

theorem smallest_two_digit_factor_of_5082 (a b : ℕ) 
  (h1 : 10 ≤ a) (h2 : a < b) (h3 : b ≤ 99) (h4 : a * b = 5082) : a = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_factor_of_5082_l2654_265411


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l2654_265419

theorem quadratic_roots_sum_bound (a b : ℤ) 
  (ha : a ≠ -1) (hb : b ≠ -1) 
  (h_roots : ∃ x y : ℤ, x ≠ y ∧ x^2 + a*b*x + (a + b) = 0 ∧ y^2 + a*b*y + (a + b) = 0) : 
  a + b ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l2654_265419


namespace NUMINAMATH_CALUDE_eighteenth_prime_l2654_265468

-- Define a function that returns the nth prime number
def nthPrime (n : ℕ) : ℕ :=
  sorry

-- State the theorem
theorem eighteenth_prime :
  (nthPrime 7 = 17) → (nthPrime 18 = 67) :=
by sorry

end NUMINAMATH_CALUDE_eighteenth_prime_l2654_265468


namespace NUMINAMATH_CALUDE_inequality_proof_l2654_265430

theorem inequality_proof (a b c : ℝ) 
  (h : a + b + c + a*b + b*c + a*c + a*b*c ≥ 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l2654_265430


namespace NUMINAMATH_CALUDE_extravagant_gift_bags_carl_extravagant_bags_l2654_265402

/-- The number of extravagant gift bags Carl created for his open house -/
theorem extravagant_gift_bags 
  (confirmed_attendees : ℕ) 
  (potential_attendees : ℕ) 
  (average_bags_made : ℕ) 
  (additional_bags_needed : ℕ) : ℕ :=
  let total_expected_attendees := confirmed_attendees + potential_attendees
  let total_average_bags := average_bags_made + additional_bags_needed
  total_expected_attendees - total_average_bags

/-- Proof that Carl created 10 extravagant gift bags -/
theorem carl_extravagant_bags : extravagant_gift_bags 50 40 20 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_extravagant_gift_bags_carl_extravagant_bags_l2654_265402


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2654_265451

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2654_265451


namespace NUMINAMATH_CALUDE_inequality_solution_l2654_265466

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x - 20) * Real.log (2 * a / x) ≤ 0) ↔ a = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2654_265466


namespace NUMINAMATH_CALUDE_well_volume_l2654_265407

/-- The volume of a cylindrical well with diameter 2 meters and depth 14 meters is π * 14 cubic meters -/
theorem well_volume (π : ℝ) (h : π = Real.pi) :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 14 := by sorry

end NUMINAMATH_CALUDE_well_volume_l2654_265407


namespace NUMINAMATH_CALUDE_special_triangle_relation_l2654_265479

/-- Represents a triangle with angles A, B, C and parts C₁, C₂, C₃ -/
structure SpecialTriangle where
  A : Real
  B : Real
  C₁ : Real
  C₂ : Real
  C₃ : Real
  ang_sum : A + B + C₁ + C₂ + C₃ = 180
  B_gt_A : B > A
  C₂_largest : C₂ ≥ C₁ ∧ C₂ ≥ C₃
  C₂_between : C₁ + C₂ + C₃ = C₁ + C₃ + C₂

/-- The main theorem stating the relationship between angles and parts -/
theorem special_triangle_relation (t : SpecialTriangle) : t.C₁ - t.C₃ = t.B - t.A := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_relation_l2654_265479


namespace NUMINAMATH_CALUDE_functions_characterization_l2654_265406

variable (f g : ℚ → ℚ)

-- Define the conditions
axiom condition1 : ∀ x y : ℚ, f (g x + g y) = f (g x) + y
axiom condition2 : ∀ x y : ℚ, g (f x + f y) = g (f x) + y

-- Define the theorem
theorem functions_characterization :
  ∃ a b : ℚ, (a * b = 1) ∧ (∀ x : ℚ, f x = a * x ∧ g x = b * x) :=
sorry

end NUMINAMATH_CALUDE_functions_characterization_l2654_265406


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l2654_265412

theorem pure_imaginary_square_root (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ b : ℝ, (1 + a * Complex.I)^2 = b * Complex.I) →
  (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l2654_265412


namespace NUMINAMATH_CALUDE_function_equality_l2654_265496

theorem function_equality (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2654_265496


namespace NUMINAMATH_CALUDE_exists_k_all_multiples_contain_all_digits_l2654_265485

/-- For a given positive integer, check if its decimal representation contains all digits from 0 to 9 -/
def containsAllDigits (n : ℕ+) : Prop := sorry

/-- For a given positive integer k and a set of positive integers, check if k*i contains all digits for all i in the set -/
def allMultiplesContainAllDigits (k : ℕ+) (s : Set ℕ+) : Prop :=
  ∀ i ∈ s, containsAllDigits (i * k)

/-- Main theorem: For all positive integers n, there exists a positive integer k such that
    k, 2k, ..., nk all contain all digits from 0 to 9 in their decimal representations -/
theorem exists_k_all_multiples_contain_all_digits (n : ℕ+) :
  ∃ k : ℕ+, allMultiplesContainAllDigits k (Set.Icc 1 n) := by sorry

end NUMINAMATH_CALUDE_exists_k_all_multiples_contain_all_digits_l2654_265485


namespace NUMINAMATH_CALUDE_windy_driving_time_l2654_265491

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  non_windy_speed : ℝ  -- Speed in non-windy conditions (miles per hour)
  windy_speed : ℝ      -- Speed in windy conditions (miles per hour)
  total_distance : ℝ   -- Total distance covered (miles)
  total_time : ℝ       -- Total time spent driving (minutes)

/-- Calculates the time spent driving in windy conditions -/
def time_in_windy_conditions (scenario : DrivingScenario) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that the time spent in windy conditions is 20 minutes -/
theorem windy_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.non_windy_speed = 40)
  (h2 : scenario.windy_speed = 25)
  (h3 : scenario.total_distance = 25)
  (h4 : scenario.total_time = 45) :
  time_in_windy_conditions scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_windy_driving_time_l2654_265491


namespace NUMINAMATH_CALUDE_gargamel_tire_savings_l2654_265478

/-- The total amount saved when buying tires on sale -/
def total_savings (num_tires : ℕ) (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) * num_tires

/-- Proof that Gargamel saved $36 on his tire purchase -/
theorem gargamel_tire_savings :
  let num_tires : ℕ := 4
  let original_price : ℚ := 84
  let sale_price : ℚ := 75
  total_savings num_tires original_price sale_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_gargamel_tire_savings_l2654_265478


namespace NUMINAMATH_CALUDE_max_fraction_value_l2654_265410

theorem max_fraction_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 500) (hb : 500 ≤ b ∧ b ≤ 1500) : 
  (b - 100) / (a + 50) ≤ 28/3 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l2654_265410


namespace NUMINAMATH_CALUDE_john_needs_72_strings_l2654_265400

/-- Calculates the total number of strings needed for restringing instruments --/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string * strings_per_8string

theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_72_strings_l2654_265400


namespace NUMINAMATH_CALUDE_variance_of_successes_l2654_265486

/-- The number of experiments -/
def n : ℕ := 30

/-- The probability of success in a single experiment -/
def p : ℚ := 5/9

/-- The variance of the number of successes in n independent experiments 
    with probability of success p -/
def variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_successes : variance n p = 200/27 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_successes_l2654_265486


namespace NUMINAMATH_CALUDE_nails_to_buy_proof_l2654_265445

/-- Given the total number of nails needed, the number of nails already owned,
    and the number of nails found in the toolshed, calculate the number of nails
    that need to be bought. -/
def nails_to_buy (total_needed : ℕ) (already_owned : ℕ) (found_in_toolshed : ℕ) : ℕ :=
  total_needed - (already_owned + found_in_toolshed)

/-- Prove that the number of nails needed to buy is 109 given the specific quantities. -/
theorem nails_to_buy_proof :
  nails_to_buy 500 247 144 = 109 := by
  sorry

end NUMINAMATH_CALUDE_nails_to_buy_proof_l2654_265445


namespace NUMINAMATH_CALUDE_g_of_f_minus_x_l2654_265470

theorem g_of_f_minus_x (x : ℝ) (hx : x^2 ≠ 1) :
  let f (x : ℝ) := (x^2 + 2*x + 1) / (x^2 - 2*x + 1)
  let g (x : ℝ) := x^2
  g (f (-x)) = (x^2 - 2*x + 1)^2 / (x^2 + 2*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_minus_x_l2654_265470


namespace NUMINAMATH_CALUDE_min_value_expression_l2654_265446

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 ∧
  (1/a + a/(b^2) + b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2654_265446


namespace NUMINAMATH_CALUDE_cookie_revenue_l2654_265408

/-- Calculates the total revenue from selling chocolate and vanilla cookies -/
theorem cookie_revenue (chocolate_count : ℕ) (vanilla_count : ℕ) 
  (chocolate_price : ℚ) (vanilla_price : ℚ) : 
  chocolate_count * chocolate_price + vanilla_count * vanilla_price = 360 :=
by
  -- Assuming chocolate_count = 220, vanilla_count = 70, 
  -- chocolate_price = 1, and vanilla_price = 2
  have h1 : chocolate_count = 220 := by sorry
  have h2 : vanilla_count = 70 := by sorry
  have h3 : chocolate_price = 1 := by sorry
  have h4 : vanilla_price = 2 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_cookie_revenue_l2654_265408


namespace NUMINAMATH_CALUDE_simons_raft_sticks_l2654_265434

theorem simons_raft_sticks (S : ℕ) : 
  S + (2 * S / 3) + (S + (2 * S / 3) + 9) = 129 → S = 51 := by
  sorry

end NUMINAMATH_CALUDE_simons_raft_sticks_l2654_265434


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2654_265455

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_cond : x + y + z = 2)
  (x_cond : x ≥ -1)
  (y_cond : y ≥ -3/2)
  (z_cond : z ≥ -2) :
  ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ + y₀ + z₀ = 2 ∧ 
    x₀ ≥ -1 ∧ 
    y₀ ≥ -3/2 ∧ 
    z₀ ≥ -2 ∧
    Real.sqrt (4 * x₀ + 2) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (4 * z₀ + 8) = 2 * Real.sqrt 30 ∧
    ∀ (x y z : ℝ), 
      x + y + z = 2 → 
      x ≥ -1 → 
      y ≥ -3/2 → 
      z ≥ -2 → 
      Real.sqrt (4 * x + 2) + Real.sqrt (4 * y + 6) + Real.sqrt (4 * z + 8) ≤ 2 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2654_265455


namespace NUMINAMATH_CALUDE_number_relationship_theorem_l2654_265463

theorem number_relationship_theorem :
  ∃ (x y a : ℝ), x = 6 * y - a ∧ x + y = 38 ∧
  (∀ (x' y' : ℝ), x' = 6 * y' - a ∧ x' + y' = 38 → x' = x ∧ y' = y → a = a) := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_theorem_l2654_265463


namespace NUMINAMATH_CALUDE_difference_of_products_l2654_265426

theorem difference_of_products : 20132014 * 20142013 - 20132013 * 20142014 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_products_l2654_265426


namespace NUMINAMATH_CALUDE_factorization_200_perfect_square_factors_200_l2654_265459

/-- A function that returns the number of positive factors of n that are perfect squares -/
def perfect_square_factors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 200 is 2^3 * 5^2 -/
theorem factorization_200 : 200 = 2^3 * 5^2 := sorry

/-- Theorem stating that the number of positive factors of 200 that are perfect squares is 4 -/
theorem perfect_square_factors_200 : perfect_square_factors 200 = 4 := by sorry

end NUMINAMATH_CALUDE_factorization_200_perfect_square_factors_200_l2654_265459


namespace NUMINAMATH_CALUDE_range_of_A_l2654_265460

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_A : ∀ a : ℝ, a ∈ A ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_A_l2654_265460


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2654_265482

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h : d ≠ 0)
  (h' : ∀ n, a (n + 1) = a n + d)

/-- A geometric sequence -/
structure GeometricSequence :=
  (b : ℕ → ℝ)
  (r : ℝ)
  (h : ∀ n, b (n + 1) = r * b n)

theorem arithmetic_geometric_sequence_property
  (as : ArithmeticSequence)
  (gs : GeometricSequence)
  (h1 : 2 * as.a 3 - (as.a 7)^2 + 2 * as.a 11 = 0)
  (h2 : gs.b 7 = as.a 7) :
  gs.b 6 * gs.b 8 = 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2654_265482


namespace NUMINAMATH_CALUDE_domain_of_h_l2654_265457

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-10) 3

-- Define the function h
def h (x : ℝ) : ℝ := f (-3 * x)

-- Define the domain of h
def domain_h : Set ℝ := Set.Ici (10/3)

-- Theorem statement
theorem domain_of_h :
  ∀ x : ℝ, x ∈ domain_h ↔ -3 * x ∈ domain_f :=
sorry

end NUMINAMATH_CALUDE_domain_of_h_l2654_265457


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2654_265458

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2654_265458


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2654_265413

theorem sum_of_x_and_y (x y : ℚ) 
  (eq1 : 5 * x - 7 * y = 17) 
  (eq2 : 3 * x + 5 * y = 11) : 
  x + y = 83 / 23 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2654_265413


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l2654_265442

theorem paper_I_maximum_mark :
  ∀ (max_mark : ℕ) (passing_percentage : ℚ) (scored_marks failed_by : ℕ),
    passing_percentage = 52 / 100 →
    scored_marks = 45 →
    failed_by = 35 →
    (scored_marks + failed_by : ℚ) = passing_percentage * max_mark →
    max_mark = 154 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l2654_265442


namespace NUMINAMATH_CALUDE_remainder_problem_l2654_265423

theorem remainder_problem (d : ℤ) (r : ℤ) 
  (h1 : d > 1)
  (h2 : 1250 % d = r)
  (h3 : 1890 % d = r)
  (h4 : 2500 % d = r) :
  d - r = 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2654_265423


namespace NUMINAMATH_CALUDE_gails_wallet_total_l2654_265421

/-- Represents the count of bills of a specific denomination in Gail's wallet. -/
structure BillCount where
  fives : Nat
  tens : Nat
  twenties : Nat

/-- Calculates the total amount of money in Gail's wallet given the bill counts. -/
def totalMoney (bills : BillCount) : Nat :=
  5 * bills.fives + 10 * bills.tens + 20 * bills.twenties

/-- Theorem stating that the total amount of money in Gail's wallet is $100. -/
theorem gails_wallet_total :
  ∃ (bills : BillCount),
    bills.fives = 4 ∧
    bills.tens = 2 ∧
    bills.twenties = 3 ∧
    totalMoney bills = 100 := by
  sorry

end NUMINAMATH_CALUDE_gails_wallet_total_l2654_265421


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l2654_265474

theorem prime_pairs_congruence (p : Nat) : Prime p →
  (∃! n : Nat, n = (Finset.filter (fun pair : Nat × Nat =>
    0 ≤ pair.1 ∧ pair.1 ≤ p ∧
    0 ≤ pair.2 ∧ pair.2 ≤ p ∧
    (pair.2 ^ 2) % p = ((pair.1 ^ 3) - pair.1) % p)
    (Finset.product (Finset.range (p + 1)) (Finset.range (p + 1)))).card ∧ n = p) ↔
  (p = 2 ∨ p % 4 = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l2654_265474


namespace NUMINAMATH_CALUDE_best_method_for_pedestrian_phone_use_data_l2654_265424

/-- Represents a data collection method -/
structure DataCollectionMethod where
  name : String
  target_group : String
  is_random : Bool

/-- Represents the characteristics of a good data collection method -/
structure MethodCharacteristics where
  is_representative : Bool
  is_extensive : Bool

/-- Defines the criteria for evaluating a data collection method -/
def evaluate_method (method : DataCollectionMethod) : MethodCharacteristics :=
  { is_representative := method.is_random && method.target_group = "pedestrians on roadside",
    is_extensive := method.is_random && method.target_group = "pedestrians on roadside" }

/-- The theorem stating that randomly distributing questionnaires to pedestrians on the roadside
    is the most representative and extensive method for collecting data on pedestrians
    walking on the roadside while looking down at their phones -/
theorem best_method_for_pedestrian_phone_use_data :
  let method := { name := "Random questionnaires to roadside pedestrians",
                  target_group := "pedestrians on roadside",
                  is_random := true : DataCollectionMethod }
  let evaluation := evaluate_method method
  evaluation.is_representative ∧ evaluation.is_extensive :=
by
  sorry


end NUMINAMATH_CALUDE_best_method_for_pedestrian_phone_use_data_l2654_265424


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l2654_265437

/-- Given that Kayla and Kylie picked a total of 200 apples, and Kayla picked 40 apples,
    prove that the ratio of apples Kayla picked to apples Kylie picked is 1:4. -/
theorem apple_picking_ratio :
  ∀ (total_apples kayla_apples : ℕ),
    total_apples = 200 →
    kayla_apples = 40 →
    ∃ (kylie_apples : ℕ),
      kylie_apples = total_apples - kayla_apples ∧
      kayla_apples * 4 = kylie_apples :=
by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l2654_265437


namespace NUMINAMATH_CALUDE_probability_at_least_one_contract_probability_at_least_one_contract_proof_l2654_265492

theorem probability_at_least_one_contract 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180) -- 0.3944444444444444 ≈ 71/180
  : ℚ :=
  29/36

theorem probability_at_least_one_contract_proof 
  (p_hardware : ℚ) 
  (p_not_software : ℚ) 
  (p_both : ℚ) 
  (h1 : p_hardware = 3/4) 
  (h2 : p_not_software = 5/9) 
  (h3 : p_both = 71/180)
  : probability_at_least_one_contract p_hardware p_not_software p_both h1 h2 h3 = 29/36 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_contract_probability_at_least_one_contract_proof_l2654_265492


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l2654_265436

-- Part 1: Equation solution
theorem equation_solution :
  ∀ x : ℚ, (1 / (x + 1) - 1 / (x + 2) = 1 / (x + 3) - 1 / (x + 4)) ↔ (x = -5/2) :=
sorry

-- Part 2: Inequality solution
theorem inequality_solution (a : ℚ) (x : ℚ) :
  x^2 - (a + 1) * x + a ≤ 0 ↔
    (a = 1 ∧ x = 1) ∨
    (a < 1 ∧ a ≤ x ∧ x ≤ 1) ∨
    (a > 1 ∧ 1 ≤ x ∧ x ≤ a) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l2654_265436


namespace NUMINAMATH_CALUDE_inequality_proof_l2654_265471

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  Real.sqrt ((b + c) / (2 * a + b + c)) + 
  Real.sqrt ((c + a) / (2 * b + c + a)) + 
  Real.sqrt ((a + b) / (2 * c + a + b)) ≤ 1 + 2 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2654_265471


namespace NUMINAMATH_CALUDE_outfits_count_l2654_265489

/-- The number of different outfits that can be made from a given number of shirts, ties, and shoes. -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (shoes : ℕ) : ℕ :=
  shirts * ties * shoes

/-- Theorem stating that the number of outfits is 192 given 8 shirts, 6 ties, and 4 pairs of shoes. -/
theorem outfits_count : number_of_outfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2654_265489


namespace NUMINAMATH_CALUDE_A_3_2_equals_19_l2654_265450

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_19 : A 3 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_A_3_2_equals_19_l2654_265450


namespace NUMINAMATH_CALUDE_prove_not_p_or_not_q_l2654_265448

theorem prove_not_p_or_not_q (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prove_not_p_or_not_q_l2654_265448


namespace NUMINAMATH_CALUDE_equation_solution_l2654_265415

theorem equation_solution : ∃ x : ℚ, (5 + 3.5 * x = 2.1 * x - 25) ∧ (x = -150/7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2654_265415


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l2654_265422

-- Define the piecewise function
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then 2*a*x + 1
  else if x ≥ -1 then 3*x - c
  else 3*x + b

-- State the theorem
theorem continuous_piecewise_function_sum (a b c : ℝ) :
  (Continuous (f a b c)) →
  a + b - c = 5*a - 4 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l2654_265422


namespace NUMINAMATH_CALUDE_linear_equation_exponent_sum_l2654_265414

/-- If x^(a-1) - 3y^(b-2) = 7 is a linear equation in x and y, then a + b = 5 -/
theorem linear_equation_exponent_sum (a b : ℝ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-1) - 3*y^(b-2) = m*x + n*y + c) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_sum_l2654_265414


namespace NUMINAMATH_CALUDE_last_three_digits_sum_sum_of_last_three_digits_l2654_265432

theorem last_three_digits_sum (C : ℕ) : ∃ (k : ℕ), 7^(4+C) = 1000 * k + 601 := by sorry

theorem sum_of_last_three_digits (C : ℕ) : (6 + 0 + 1 : ℕ) = 7 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_sum_sum_of_last_three_digits_l2654_265432


namespace NUMINAMATH_CALUDE_person_age_l2654_265462

theorem person_age : ∃ (age : ℕ), 
  (4 * (age + 4) - 4 * (age - 4) = age) ∧ (age = 32) := by
  sorry

end NUMINAMATH_CALUDE_person_age_l2654_265462


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2654_265476

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2654_265476


namespace NUMINAMATH_CALUDE_derivative_of_e_squared_l2654_265427

theorem derivative_of_e_squared :
  (deriv (λ _ : ℝ => Real.exp 2)) = (λ _ => 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_e_squared_l2654_265427


namespace NUMINAMATH_CALUDE_negation_equivalence_l2654_265493

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2654_265493


namespace NUMINAMATH_CALUDE_harper_mineral_water_cost_l2654_265477

/-- The amount Harper spends on mineral water for 240 days -/
def mineral_water_cost (daily_consumption : ℚ) (bottles_per_case : ℕ) (case_cost : ℚ) (total_days : ℕ) : ℚ :=
  let cases_needed := (total_days : ℚ) * daily_consumption / bottles_per_case
  cases_needed.ceil * case_cost

/-- Theorem stating the cost of mineral water for Harper -/
theorem harper_mineral_water_cost :
  mineral_water_cost (1/2) 24 12 240 = 60 := by
  sorry

end NUMINAMATH_CALUDE_harper_mineral_water_cost_l2654_265477


namespace NUMINAMATH_CALUDE_exactly_one_absent_probability_l2654_265429

/-- The probability of an employee being absent on a given day -/
def p_absent : ℚ := 1 / 30

/-- The probability of an employee being present on a given day -/
def p_present : ℚ := 1 - p_absent

/-- The number of employees selected -/
def n : ℕ := 3

/-- The number of employees that should be absent -/
def k : ℕ := 1

theorem exactly_one_absent_probability :
  (n.choose k : ℚ) * p_absent^k * p_present^(n - k) = 841 / 9000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_absent_probability_l2654_265429


namespace NUMINAMATH_CALUDE_g_fifty_l2654_265480

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 40 -/
def g : ℝ → ℝ := sorry

/-- The property that g(xy) = xg(y) for all real x and y -/
axiom g_prop (x y : ℝ) : g (x * y) = x * g y

/-- The property that g(1) = 40 -/
axiom g_one : g 1 = 40

/-- Theorem: g(50) = 2000 -/
theorem g_fifty : g 50 = 2000 := by sorry

end NUMINAMATH_CALUDE_g_fifty_l2654_265480


namespace NUMINAMATH_CALUDE_license_plate_difference_l2654_265495

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of letters at the beginning of a California license plate -/
def ca_prefix_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def ca_digits : ℕ := 3

/-- The number of letters at the end of a California license plate -/
def ca_suffix_letters : ℕ := 2

/-- The number of letters in a Texas license plate -/
def tx_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def tx_digits : ℕ := 4

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : 
  (num_letters ^ (ca_prefix_letters + ca_suffix_letters) * num_digits ^ ca_digits) - 
  (num_letters ^ tx_letters * num_digits ^ tx_digits) = 301093376000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l2654_265495


namespace NUMINAMATH_CALUDE_wendys_cookies_l2654_265469

/-- Represents the number of pastries in various categories -/
structure Pastries where
  cupcakes : ℕ
  cookies : ℕ
  taken_home : ℕ
  sold : ℕ

/-- The theorem statement for Wendy's bake sale problem -/
theorem wendys_cookies (w : Pastries) 
  (h1 : w.cupcakes = 4)
  (h2 : w.taken_home = 24)
  (h3 : w.sold = 9)
  (h4 : w.cupcakes + w.cookies = w.taken_home + w.sold) :
  w.cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_wendys_cookies_l2654_265469


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_two_l2654_265433

theorem logarithm_sum_equals_two : Real.log 25 / Real.log 10 + (Real.log 2 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_two_l2654_265433


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2654_265431

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y r : ℝ) : Prop := (x+4)^2 + (y-3)^2 = r^2

-- Define the condition of external tangency
def externally_tangent (r : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y r ∧
  (∀ (x' y' : ℝ), circle1 x' y' → circle2 x' y' r → (x = x' ∧ y = y'))

-- Theorem statement
theorem tangent_circles_radius :
  ∀ r : ℝ, externally_tangent r → (r = 3 ∨ r = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2654_265431


namespace NUMINAMATH_CALUDE_sprinkler_evening_usage_l2654_265418

/-- Represents the water usage of a desert garden's sprinkler system -/
structure SprinklerSystem where
  morning_usage : ℝ
  evening_usage : ℝ
  days : ℕ
  total_usage : ℝ

/-- Theorem stating the evening water usage of the sprinkler system -/
theorem sprinkler_evening_usage 
  (s : SprinklerSystem) 
  (h1 : s.morning_usage = 4)
  (h2 : s.days = 5)
  (h3 : s.total_usage = 50)
  : s.evening_usage = 6 := by
  sorry

end NUMINAMATH_CALUDE_sprinkler_evening_usage_l2654_265418


namespace NUMINAMATH_CALUDE_rod_cutting_l2654_265405

theorem rod_cutting (total_length : Real) (num_pieces : Nat) :
  total_length = 42.5 → num_pieces = 50 →
  (total_length / num_pieces) * 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_rod_cutting_l2654_265405


namespace NUMINAMATH_CALUDE_point_on_number_line_l2654_265403

theorem point_on_number_line (B C : ℝ) : 
  B = 3 → abs (C - B) = 2 → (C = 1 ∨ C = 5) :=
by sorry

end NUMINAMATH_CALUDE_point_on_number_line_l2654_265403


namespace NUMINAMATH_CALUDE_werewolf_unreachable_l2654_265465

def is_black (x y : Int) : Bool :=
  x % 2 = y % 2

def possible_moves : List (Int × Int) :=
  [(1, 2), (2, -1), (-1, -2), (-2, 1)]

def reachable (start_x start_y end_x end_y : Int) : Prop :=
  ∃ (n : Nat), ∃ (moves : List (Int × Int)),
    moves.length = n ∧
    moves.all (λ m => m ∈ possible_moves) ∧
    (moves.foldl (λ (x, y) (dx, dy) => (x + dx, y + dy)) (start_x, start_y) = (end_x, end_y))

theorem werewolf_unreachable :
  ¬(reachable 26 10 42 2017) :=
by sorry

end NUMINAMATH_CALUDE_werewolf_unreachable_l2654_265465


namespace NUMINAMATH_CALUDE_tina_customers_l2654_265447

/-- Calculates the number of customers Tina sold books to -/
def number_of_customers (selling_price cost_price total_profit books_per_customer : ℚ) : ℚ :=
  (total_profit / (selling_price - cost_price)) / books_per_customer

/-- Theorem: Given the conditions, Tina sold books to 4 customers -/
theorem tina_customers :
  let selling_price : ℚ := 20
  let cost_price : ℚ := 5
  let total_profit : ℚ := 120
  let books_per_customer : ℚ := 2
  number_of_customers selling_price cost_price total_profit books_per_customer = 4 := by
  sorry

#eval number_of_customers 20 5 120 2

end NUMINAMATH_CALUDE_tina_customers_l2654_265447


namespace NUMINAMATH_CALUDE_paul_weed_eating_money_l2654_265420

/-- The amount of money Paul made weed eating -/
def weed_eating_money (mowing_money weekly_spending weeks_lasted : ℕ) : ℕ :=
  weekly_spending * weeks_lasted - mowing_money

/-- Theorem stating that Paul made $28 weed eating -/
theorem paul_weed_eating_money :
  weed_eating_money 44 9 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_weed_eating_money_l2654_265420


namespace NUMINAMATH_CALUDE_album_photos_l2654_265490

theorem album_photos (n : ℕ) 
  (h1 : ∀ (album : ℕ), album > 0 → ∃ (page : ℕ), page > 0 ∧ page ≤ n)
  (h2 : ∀ (page : ℕ), page > 0 → page ≤ n → ∃ (photos : Fin 4), True)
  (h3 : ∃ (album : ℕ), album > 0 ∧ 81 ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 5) - 3 + i) → j ≤ 4*n*album))
  (h4 : ∃ (album : ℕ), album > 0 ∧ 171 ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) ∧ (∀ j, j ∈ Set.range (λ i => 4*(n*(album-1) + 3) - 3 + i) → j ≤ 4*n*album))
  : n = 8 ∧ 4*n = 32 := by
  sorry

end NUMINAMATH_CALUDE_album_photos_l2654_265490


namespace NUMINAMATH_CALUDE_triangle_inequality_l2654_265499

/-- A triangle with heights and an internal point -/
structure Triangle where
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  l_a : ℝ
  l_b : ℝ
  l_c : ℝ
  h_a_pos : h_a > 0
  h_b_pos : h_b > 0
  h_c_pos : h_c > 0
  l_a_pos : l_a > 0
  l_b_pos : l_b > 0
  l_c_pos : l_c > 0

/-- The inequality holds for any triangle -/
theorem triangle_inequality (t : Triangle) :
  t.h_a / t.l_a + t.h_b / t.l_b + t.h_c / t.l_c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2654_265499


namespace NUMINAMATH_CALUDE_melissa_games_played_l2654_265483

-- Define the total points scored
def total_points : ℕ := 91

-- Define the points scored per game
def points_per_game : ℕ := 7

-- Theorem statement
theorem melissa_games_played : 
  total_points / points_per_game = 13 := by
  sorry

end NUMINAMATH_CALUDE_melissa_games_played_l2654_265483


namespace NUMINAMATH_CALUDE_tenth_term_is_three_point_five_l2654_265456

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

/-- The first term of our sequence -/
def a₁ : ℚ := 1/2

/-- The second term of our sequence -/
def a₂ : ℚ := 5/6

/-- The common difference of our sequence -/
def d : ℚ := a₂ - a₁

theorem tenth_term_is_three_point_five :
  arithmetic_sequence a₁ d 10 = 7/2 := by sorry

end NUMINAMATH_CALUDE_tenth_term_is_three_point_five_l2654_265456


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l2654_265440

theorem gcd_of_polynomial_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1428 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 30)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l2654_265440


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l2654_265497

theorem sum_of_squares_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l2654_265497


namespace NUMINAMATH_CALUDE_not_always_intersects_x_axis_l2654_265428

/-- Represents a circle in a 2D plane -/
structure Circle where
  a : ℝ  -- x-coordinate of the center
  b : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Predicate to check if a circle intersects the x-axis -/
def intersects_x_axis (c : Circle) : Prop :=
  ∃ x : ℝ, (x - c.a)^2 + c.b^2 = c.r^2

/-- Theorem stating that b < r does not always imply intersection with x-axis -/
theorem not_always_intersects_x_axis :
  ¬ (∀ c : Circle, c.b < c.r → intersects_x_axis c) :=
sorry

end NUMINAMATH_CALUDE_not_always_intersects_x_axis_l2654_265428


namespace NUMINAMATH_CALUDE_expression_evaluation_l2654_265488

theorem expression_evaluation : 
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (10 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2654_265488


namespace NUMINAMATH_CALUDE_no_fib_rectangle_decomposition_l2654_265481

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- A square with side length that is a Fibonacci number -/
structure FibSquare where
  side : ℕ
  is_fib : ∃ n, fib n = side

/-- A rectangle composed of Fibonacci squares -/
structure FibRectangle where
  squares : List FibSquare
  different_sizes : ∀ i j, i ≠ j → (squares.get i).side ≠ (squares.get j).side
  at_least_two : squares.length ≥ 2

/-- The theorem stating that a rectangle cannot be composed of different-sized Fibonacci squares -/
theorem no_fib_rectangle_decomposition : ¬ ∃ (r : FibRectangle), True := by
  sorry

end NUMINAMATH_CALUDE_no_fib_rectangle_decomposition_l2654_265481


namespace NUMINAMATH_CALUDE_taehyungs_calculation_l2654_265438

theorem taehyungs_calculation : ∃ x : ℝ, (x / 5 = 30) ∧ (8 * x = 1200) := by
  sorry

end NUMINAMATH_CALUDE_taehyungs_calculation_l2654_265438


namespace NUMINAMATH_CALUDE_plot_size_in_acres_l2654_265401

/-- Represents the scale factor for converting centimeters to miles -/
def scale : ℝ := 3

/-- Represents the conversion factor from square miles to acres -/
def milesSquareToAcres : ℝ := 640

/-- Represents the length of one side of the right triangle in the scaled drawing -/
def side1 : ℝ := 20

/-- Represents the length of the other side of the right triangle in the scaled drawing -/
def side2 : ℝ := 15

/-- Theorem stating that the actual size of the plot is 864000 acres -/
theorem plot_size_in_acres :
  let scaledArea := (side1 * side2) / 2
  let actualAreaInMilesSquare := scaledArea * scale^2
  let actualAreaInAcres := actualAreaInMilesSquare * milesSquareToAcres
  actualAreaInAcres = 864000 := by sorry

end NUMINAMATH_CALUDE_plot_size_in_acres_l2654_265401


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2654_265464

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Icc (-2 : ℝ) 1 = {x : ℝ | a * x^2 - x + b ≥ 0}) : 
  Set.Icc (-1/2 : ℝ) 1 = {x : ℝ | b * x^2 - x + a ≤ 0} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l2654_265464


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2654_265487

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (q > 1) ∧ (a 1 > 0)

/-- The theorem stating the common ratio of the geometric sequence -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : IsIncreasingGeometricSequence a q) 
  (h_sum : a 1 + a 4 = 9) 
  (h_prod : a 2 * a 3 = 8) : 
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2654_265487


namespace NUMINAMATH_CALUDE_min_ab_value_l2654_265409

theorem min_ab_value (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) :
  (a * b : ℕ) ≥ 60 ∧ ∃ (a₀ b₀ : ℕ+), (a₀ : ℚ)⁻¹ + (3 * b₀ : ℚ)⁻¹ = (9 : ℚ)⁻¹ ∧ (a₀ * b₀ : ℕ) = 60 :=
by sorry

end NUMINAMATH_CALUDE_min_ab_value_l2654_265409


namespace NUMINAMATH_CALUDE_oil_production_per_capita_correct_l2654_265443

/-- Oil production per capita for a region -/
structure OilProductionPerCapita where
  region : String
  value : Float

/-- Given oil production per capita data -/
def given_data : List OilProductionPerCapita := [
  ⟨"West", 55.084⟩,
  ⟨"Non-West", 214.59⟩,
  ⟨"Russia", 1038.33⟩
]

/-- Theorem: The oil production per capita for West, Non-West, and Russia are as given -/
theorem oil_production_per_capita_correct :
  ∀ region value, OilProductionPerCapita.mk region value ∈ given_data →
  (region = "West" → value = 55.084) ∧
  (region = "Non-West" → value = 214.59) ∧
  (region = "Russia" → value = 1038.33) :=
by sorry

end NUMINAMATH_CALUDE_oil_production_per_capita_correct_l2654_265443


namespace NUMINAMATH_CALUDE_locus_of_sine_zero_l2654_265473

theorem locus_of_sine_zero (x y : ℝ) : 
  Real.sin (x + y) = 0 ↔ ∃ k : ℤ, x + y = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_locus_of_sine_zero_l2654_265473


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_first_term_of_geometric_series_l2654_265435

/-- Given an infinite geometric series with sum 18 and sum of squares 72, 
    prove that the first term of the series is 72/11 -/
theorem first_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (r : ℝ) -- Common ratio of the series
  (h1 : a / (1 - r) = 18) -- Sum of the series is 18
  (h2 : a^2 / (1 - r^2) = 72) -- Sum of squares is 72
  : a = 72 / 11 := by
sorry

/-- Alternative formulation using a function for the series -/
theorem first_term_of_geometric_series' 
  (S : ℕ → ℝ) -- Geometric series as a function
  (h1 : ∃ r : ℝ, ∀ n : ℕ, S (n + 1) = r * S n) -- S is a geometric series
  (h2 : ∑' n, S n = 18) -- Sum of the series is 18
  (h3 : ∑' n, (S n)^2 = 72) -- Sum of squares is 72
  : S 0 = 72 / 11 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_first_term_of_geometric_series_l2654_265435


namespace NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2654_265453

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- Add necessary fields here
  convex : Bool

/-- Represents an orientation-preserving movement (rotation or translation). -/
structure OrientationPreservingMovement where
  -- Add necessary fields here

/-- Represents a division of a polygon into two parts. -/
structure PolygonDivision (P : ConvexPolygon) where
  part1 : Set (ℝ × ℝ)
  part2 : Set (ℝ × ℝ)
  is_valid : part1 ∪ part2 = Set.univ -- The union of parts equals the whole polygon

/-- Predicate to check if a division is by a broken line. -/
def is_broken_line_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of broken line division

/-- Predicate to check if a division is by a straight line segment. -/
def is_segment_division (P : ConvexPolygon) (d : PolygonDivision P) : Prop :=
  sorry -- Definition of straight line segment division

/-- Predicate to check if two parts of a division can be transformed into each other
    by an orientation-preserving movement. -/
def parts_transformable (P : ConvexPolygon) (d : PolygonDivision P) 
    (m : OrientationPreservingMovement) : Prop :=
  sorry -- Definition of transformability

/-- Main theorem statement -/
theorem convex_polygon_division_theorem (P : ConvexPolygon) 
    (h_convex : P.convex = true) :
    (∃ (d : PolygonDivision P) (m : OrientationPreservingMovement), 
      is_broken_line_division P d ∧ parts_transformable P d m) →
    (∃ (d' : PolygonDivision P) (m' : OrientationPreservingMovement),
      is_segment_division P d' ∧ parts_transformable P d' m') :=
  sorry

end NUMINAMATH_CALUDE_convex_polygon_division_theorem_l2654_265453


namespace NUMINAMATH_CALUDE_smallest_odd_four_primes_with_13_l2654_265494

def is_prime (n : ℕ) : Prop := sorry

def prime_factors (n : ℕ) : Finset ℕ := sorry

theorem smallest_odd_four_primes_with_13 :
  ∀ n : ℕ,
  n % 2 = 1 →
  (prime_factors n).card = 4 →
  13 ∈ prime_factors n →
  n ≥ 1365 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_four_primes_with_13_l2654_265494


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2654_265416

theorem triangle_angle_sum (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a = 20 →  -- Smallest angle is 20 degrees
  b = 3 * a →  -- Middle angle is 3 times the smallest
  c = 5 * a →  -- Largest angle is 5 times the smallest
  a + b + c = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2654_265416


namespace NUMINAMATH_CALUDE_second_side_length_l2654_265439

/-- A triangle with a perimeter of 55 centimeters and two sides measuring 5 and 30 centimeters. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  perimeter_eq : side1 + side2 + side3 = 55
  side1_eq : side1 = 5
  side3_eq : side3 = 30

/-- The second side of the triangle measures 20 centimeters. -/
theorem second_side_length (t : Triangle) : t.side2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_side_length_l2654_265439


namespace NUMINAMATH_CALUDE_harpers_rubber_bands_l2654_265484

/-- Harper's rubber band problem -/
theorem harpers_rubber_bands :
  ∀ (h : ℕ),                        -- h represents Harper's number of rubber bands
  (h + (h - 6) = 24) →              -- Total rubber bands condition
  h = 15 := by                      -- Prove that Harper has 15 rubber bands
sorry


end NUMINAMATH_CALUDE_harpers_rubber_bands_l2654_265484


namespace NUMINAMATH_CALUDE_project_completion_time_l2654_265475

theorem project_completion_time (a b c : ℝ) 
  (h1 : a + b = 1/2)   -- A and B together complete in 2 days
  (h2 : b + c = 1/4)   -- B and C together complete in 4 days
  (h3 : c + a = 1/2.4) -- C and A together complete in 2.4 days
  : 1/a = 3 :=         -- A alone completes in 3 days
by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l2654_265475


namespace NUMINAMATH_CALUDE_fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l2654_265498

theorem fraction_sum_and_reciprocal_sum_integer :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

theorem fraction_sum_and_reciprocal_sum_integer_distinct_numerators :
  ∃ (a b c d e f : ℕ), 
    (0 < a ∧ a < b) ∧ 
    (0 < c ∧ c < d) ∧ 
    (0 < e ∧ e < f) ∧
    (Nat.gcd a b = 1) ∧
    (Nat.gcd c d = 1) ∧
    (Nat.gcd e f = 1) ∧
    a ≠ c ∧ a ≠ e ∧ c ≠ e ∧
    (a : ℚ) / b + (c : ℚ) / d + (e : ℚ) / f = 1 ∧
    ∃ (n : ℕ), (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e = n :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_and_reciprocal_sum_integer_fraction_sum_and_reciprocal_sum_integer_distinct_numerators_l2654_265498


namespace NUMINAMATH_CALUDE_sum_of_seven_place_values_l2654_265454

/-- Given the number 87953.0727, this theorem states that the sum of the place values
    of the three 7's in this number is equal to 7,000.0707. -/
theorem sum_of_seven_place_values (n : ℝ) (h : n = 87953.0727) :
  7000 + 0.07 + 0.0007 = 7000.0707 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_place_values_l2654_265454


namespace NUMINAMATH_CALUDE_divisors_of_factorial_8_l2654_265441

theorem divisors_of_factorial_8 : (Nat.divisors (Nat.factorial 8)).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_factorial_8_l2654_265441


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2654_265472

theorem cubic_roots_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 8 = 0 →
  s^3 - 15*s^2 + 13*s - 8 = 0 →
  t^3 - 15*t^2 + 13*t - 8 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2654_265472


namespace NUMINAMATH_CALUDE_expression_evaluation_l2654_265452

theorem expression_evaluation :
  let y : ℚ := -3
  let numerator := 4 + y * (4 + y) - 4^2
  let denominator := y - 4 + y^2
  numerator / denominator = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2654_265452


namespace NUMINAMATH_CALUDE_remainder_sum_l2654_265417

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 90 = 84) 
  (hb : b % 120 = 114) 
  (hc : c % 150 = 144) : 
  (a + b + c) % 30 = 12 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2654_265417


namespace NUMINAMATH_CALUDE_ternary_57_has_four_digits_l2654_265404

/-- Converts a natural number to its ternary (base 3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The theorem stating that the ternary representation of 57 has exactly 4 digits -/
theorem ternary_57_has_four_digits : (to_ternary 57).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ternary_57_has_four_digits_l2654_265404


namespace NUMINAMATH_CALUDE_alicia_local_taxes_l2654_265449

theorem alicia_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.02 → hourly_wage * tax_rate * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_alicia_local_taxes_l2654_265449


namespace NUMINAMATH_CALUDE_arrangement_inequality_l2654_265467

-- Define the arrangement function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the set of valid x values
def valid_x : Set ℕ := {x | 3 ≤ x ∧ x ≤ 7}

-- State the theorem
theorem arrangement_inequality (x : ℕ) (h1 : 2 < x) (h2 : x ≤ 9) :
  A 9 x > 6 * A 9 (x - 2) ↔ x ∈ valid_x :=
sorry

end NUMINAMATH_CALUDE_arrangement_inequality_l2654_265467


namespace NUMINAMATH_CALUDE_algebraic_identities_l2654_265444

theorem algebraic_identities (x y : ℝ) : 
  ((2*x - 3*y)^2 = 4*x^2 - 12*x*y + 9*y^2) ∧ 
  ((x + y)*(x + y)*(x^2 + y^2) = x^4 + 2*x^2*y^2 + y^4 + 2*x^3*y + 2*x*y^3) := by
sorry

end NUMINAMATH_CALUDE_algebraic_identities_l2654_265444
