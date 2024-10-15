import Mathlib

namespace NUMINAMATH_CALUDE_reporters_not_covering_politics_l1276_127649

theorem reporters_not_covering_politics 
  (local_politics_coverage : Real) 
  (non_local_politics_ratio : Real) 
  (h1 : local_politics_coverage = 0.12)
  (h2 : non_local_politics_ratio = 0.4) :
  1 - (local_politics_coverage / (1 - non_local_politics_ratio)) = 0.8 := by
sorry

end NUMINAMATH_CALUDE_reporters_not_covering_politics_l1276_127649


namespace NUMINAMATH_CALUDE_average_difference_l1276_127675

theorem average_difference (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1276_127675


namespace NUMINAMATH_CALUDE_coprime_in_ten_consecutive_integers_l1276_127618

theorem coprime_in_ten_consecutive_integers (k : ℤ) :
  ∃ n ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ n → Int.gcd (k + n) (k + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_in_ten_consecutive_integers_l1276_127618


namespace NUMINAMATH_CALUDE_power_of_product_l1276_127692

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l1276_127692


namespace NUMINAMATH_CALUDE_language_spoken_by_three_scientists_l1276_127662

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  languages_bound : languages.card ≤ 3

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientists_count : Scientists.card = 9

/-- No scientist speaks more than 3 languages -/
axiom max_languages (s : Scientist) : s.languages.card ≤ 3

/-- Among any three scientists, there are two who speak a common language -/
axiom common_language_exists (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- There exists a language spoken by at least three scientists -/
theorem language_spoken_by_three_scientists :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_language_spoken_by_three_scientists_l1276_127662


namespace NUMINAMATH_CALUDE_six_hamburgers_left_over_l1276_127671

/-- Given a restaurant that made hamburgers and served some, calculate the number left over. -/
def hamburgers_left_over (made served : ℕ) : ℕ := made - served

/-- Prove that when 9 hamburgers are made and 3 are served, 6 are left over. -/
theorem six_hamburgers_left_over :
  hamburgers_left_over 9 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_hamburgers_left_over_l1276_127671


namespace NUMINAMATH_CALUDE_g_of_f_minus_x_l1276_127672

theorem g_of_f_minus_x (x : ℝ) (hx : x^2 ≠ 1) :
  let f (x : ℝ) := (x^2 + 2*x + 1) / (x^2 - 2*x + 1)
  let g (x : ℝ) := x^2
  g (f (-x)) = (x^2 - 2*x + 1)^2 / (x^2 + 2*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_g_of_f_minus_x_l1276_127672


namespace NUMINAMATH_CALUDE_prob_at_least_one_girl_l1276_127603

/-- The probability of selecting at least one girl when randomly choosing 2 people from a group of 3 boys and 2 girls is 7/10 -/
theorem prob_at_least_one_girl (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 2) :
  let total := boys + girls
  let prob_at_least_one_girl := 1 - (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ)
  prob_at_least_one_girl = 7/10 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_girl_l1276_127603


namespace NUMINAMATH_CALUDE_total_car_production_l1276_127624

theorem total_car_production (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_car_production_l1276_127624


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l1276_127652

/-- Given a square divided into 5 identical rectangles, where each rectangle has a perimeter of 120
    and a length that is 5 times its width, the area of the original square is 2500 -/
theorem square_area_from_rectangles (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 5 * width →
  2 * (length + width) = perimeter →
  (5 * width)^2 = 2500 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangles_l1276_127652


namespace NUMINAMATH_CALUDE_cat_weight_ratio_l1276_127620

theorem cat_weight_ratio (female_weight male_weight : ℝ) : 
  female_weight = 2 →
  male_weight > female_weight →
  female_weight + male_weight = 6 →
  male_weight / female_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_cat_weight_ratio_l1276_127620


namespace NUMINAMATH_CALUDE_custom_mul_equality_l1276_127663

/-- Custom multiplication operation for real numbers -/
def custom_mul (x y : ℝ) : ℝ := (x - y)^2

/-- Theorem stating the equality for the given expression using custom multiplication -/
theorem custom_mul_equality (x y z : ℝ) : 
  custom_mul (x - y) (y - z) = (x - 2*y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_equality_l1276_127663


namespace NUMINAMATH_CALUDE_fraction_equality_l1276_127629

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a/b = 8)
  (h2 : c/b = 4)
  (h3 : c/d = 2/3) :
  d/a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1276_127629


namespace NUMINAMATH_CALUDE_journey_distance_l1276_127626

/-- Calculates the total distance of a journey with multiple parts and a detour -/
theorem journey_distance (speed1 speed2 speed3 : ℝ) 
                         (time1 time2 time3 : ℝ) 
                         (detour_distance : ℝ) : 
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 30 →
  time1 = 1.5 →
  time2 = 1 →
  time3 = 2.25 →
  detour_distance = 10 →
  speed1 * time1 + speed2 * time2 + detour_distance + speed3 * time3 = 187.5 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l1276_127626


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_max_l1276_127650

/-- An arithmetic sequence -/
def ArithmeticSequence := ℕ+ → ℝ

/-- Sum of the first n terms of an arithmetic sequence -/
def SumOfTerms (a : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_sum_max (a : ArithmeticSequence) :
  (SumOfTerms a 10 > 0) →
  (SumOfTerms a 11 = 0) →
  (∀ n : ℕ+, ∃ k : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) →
  ∃ k : ℕ+, (k = 5 ∨ k = 6) ∧ 
    (∀ n : ℕ+, SumOfTerms a n ≤ SumOfTerms a k) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_max_l1276_127650


namespace NUMINAMATH_CALUDE_max_profit_at_16_l1276_127667

/-- Represents the daily sales quantity as a function of selling price -/
def sales_quantity (k b x : ℝ) : ℝ := k * x + b

/-- Represents the daily profit as a function of selling price -/
def daily_profit (k b x : ℝ) : ℝ := (x - 12) * (sales_quantity k b x)

theorem max_profit_at_16 (k b : ℝ) :
  sales_quantity k b 15 = 50 →
  sales_quantity k b 17 = 30 →
  (∀ x, 12 ≤ x → x ≤ 18 → daily_profit k b x ≤ daily_profit k b 16) ∧
  daily_profit k b 16 = 160 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_16_l1276_127667


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l1276_127691

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

theorem probability_no_consecutive_ones (n : ℕ) :
  n = 12 →
  (valid_sequences n : ℚ) / (total_sequences n : ℚ) = 377 / 4096 := by
  sorry

#eval valid_sequences 12
#eval total_sequences 12

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l1276_127691


namespace NUMINAMATH_CALUDE_product_of_distances_to_asymptotes_l1276_127628

/-- Represents a hyperbola with equation y²/2 - x²/b = 1 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : y^2 / 2 - x^2 / h.b = 1

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The distance from a point to an asymptote of the hyperbola -/
def distance_to_asymptote (h : Hyperbola) (p : PointOnHyperbola h) : ℝ := sorry

/-- The theorem stating the product of distances to asymptotes -/
theorem product_of_distances_to_asymptotes (h : Hyperbola) 
  (h_ecc : eccentricity h = 2) (p : PointOnHyperbola h) : 
  (distance_to_asymptote h p) * (distance_to_asymptote h p) = 3/2 := 
sorry

end NUMINAMATH_CALUDE_product_of_distances_to_asymptotes_l1276_127628


namespace NUMINAMATH_CALUDE_cos_max_value_l1276_127608

theorem cos_max_value (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  ∃ (max : ℝ), max = Real.sqrt 3 - 1 ∧ Real.cos a ≤ max ∧ 
  ∃ (a₀ b₀ : ℝ), Real.cos (a₀ + b₀) = Real.cos a₀ + Real.cos b₀ ∧ Real.cos a₀ = max :=
by sorry

end NUMINAMATH_CALUDE_cos_max_value_l1276_127608


namespace NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l1276_127682

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 8)

-- Define vector CD
def CD (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem for parallel case
theorem parallel_case : 
  (∃ k : ℝ, AB = k • CD 1) → 1 = 1 := by sorry

-- Theorem for perpendicular case
theorem perpendicular_case :
  (AB.1 * (CD (-9)).1 + AB.2 * (CD (-9)).2 = 0) → -9 = -9 := by sorry

end NUMINAMATH_CALUDE_parallel_case_perpendicular_case_l1276_127682


namespace NUMINAMATH_CALUDE_bus_stop_interval_l1276_127640

/-- Proves that the time interval between bus stops is 6 minutes -/
theorem bus_stop_interval (average_speed : ℝ) (total_distance : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60)
  (h2 : total_distance = 30)
  (h3 : num_stops = 6) :
  (total_distance / average_speed) * 60 / (num_stops - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_interval_l1276_127640


namespace NUMINAMATH_CALUDE_prob_one_defective_out_of_two_l1276_127696

/-- The probability of selecting exactly one defective product when randomly choosing 2 out of 5 products, where 2 are defective and 3 are qualified. -/
theorem prob_one_defective_out_of_two (total : Nat) (defective : Nat) (selected : Nat) : 
  total = 5 → defective = 2 → selected = 2 → 
  (Nat.choose defective 1 * Nat.choose (total - defective) (selected - 1)) / Nat.choose total selected = 3/5 := by
sorry

end NUMINAMATH_CALUDE_prob_one_defective_out_of_two_l1276_127696


namespace NUMINAMATH_CALUDE_max_profit_is_21600_l1276_127600

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the total profit for a given production quantity -/
def totalProfit (p : Production) : ℝ :=
  2100 * p.a + 900 * p.b

/-- Checks if a production quantity satisfies all constraints -/
def isValid (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1.5 * p.a + 0.5 * p.b ≤ 150 ∧
  1 * p.a + 0.3 * p.b ≤ 90 ∧
  5 * p.a + 3 * p.b ≤ 600

/-- Theorem stating that the maximum total profit is 21600 yuan -/
theorem max_profit_is_21600 :
  ∃ (p : Production), isValid p ∧
    totalProfit p = 21600 ∧
    ∀ (q : Production), isValid q → totalProfit q ≤ 21600 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_21600_l1276_127600


namespace NUMINAMATH_CALUDE_coles_fence_payment_l1276_127637

/-- Calculates Cole's payment for fencing his backyard -/
theorem coles_fence_payment
  (side_length : ℝ)
  (back_length : ℝ)
  (fence_cost_per_foot : ℝ)
  (back_neighbor_contribution_ratio : ℝ)
  (left_neighbor_contribution_ratio : ℝ)
  (h1 : side_length = 9)
  (h2 : back_length = 18)
  (h3 : fence_cost_per_foot = 3)
  (h4 : back_neighbor_contribution_ratio = 1/2)
  (h5 : left_neighbor_contribution_ratio = 1/3) :
  side_length * 2 + back_length * fence_cost_per_foot -
  (back_length * back_neighbor_contribution_ratio * fence_cost_per_foot +
   side_length * left_neighbor_contribution_ratio * fence_cost_per_foot) = 72 :=
by sorry

end NUMINAMATH_CALUDE_coles_fence_payment_l1276_127637


namespace NUMINAMATH_CALUDE_therapy_cost_difference_l1276_127690

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  first_hour_more_expensive : firstHourCost > additionalHourCost

/-- Theorem: Given the conditions, the difference in cost between the first hour
    and each additional hour is $30 -/
theorem therapy_cost_difference (p : PricingScheme) 
  (five_hour_cost : p.firstHourCost + 4 * p.additionalHourCost = 400)
  (three_hour_cost : p.firstHourCost + 2 * p.additionalHourCost = 252) :
  p.firstHourCost - p.additionalHourCost = 30 := by
  sorry

end NUMINAMATH_CALUDE_therapy_cost_difference_l1276_127690


namespace NUMINAMATH_CALUDE_particular_innings_number_l1276_127630

/-- Represents the statistics of a cricket player -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding runs -/
def newAverage (stats : CricketStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem particular_innings_number
  (initialStats : CricketStats)
  (h1 : initialStats.innings = 16)
  (h2 : newAverage initialStats 112 = initialStats.average + 6)
  (h3 : newAverage initialStats 112 = 16) :
  initialStats.innings + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_particular_innings_number_l1276_127630


namespace NUMINAMATH_CALUDE_unique_solution_system_l1276_127680

theorem unique_solution_system (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (2 * x₁ = x₅^2 - 23) ∧
  (4 * x₂ = x₁^2 + 7) ∧
  (6 * x₃ = x₂^2 + 14) ∧
  (8 * x₄ = x₃^2 + 23) ∧
  (10 * x₅ = x₄^2 + 34) →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = 5 :=
by sorry

#check unique_solution_system

end NUMINAMATH_CALUDE_unique_solution_system_l1276_127680


namespace NUMINAMATH_CALUDE_chalk_inventory_theorem_l1276_127656

-- Define the types of chalk
inductive ChalkType
  | Regular
  | Unusual
  | Excellent

-- Define the store's chalk inventory
structure ChalkInventory where
  regular : ℕ
  unusual : ℕ
  excellent : ℕ

def initial_ratio : Fin 3 → ℕ
  | 0 => 3  -- Regular
  | 1 => 4  -- Unusual
  | 2 => 6  -- Excellent

def new_ratio : Fin 3 → ℕ
  | 0 => 2  -- Regular
  | 1 => 5  -- Unusual
  | 2 => 8  -- Excellent

theorem chalk_inventory_theorem (initial : ChalkInventory) (final : ChalkInventory) :
  -- Initial ratio condition
  initial.regular * initial_ratio 1 = initial.unusual * initial_ratio 0 ∧
  initial.regular * initial_ratio 2 = initial.excellent * initial_ratio 0 ∧
  -- New ratio condition
  final.regular * new_ratio 1 = final.unusual * new_ratio 0 ∧
  final.regular * new_ratio 2 = final.excellent * new_ratio 0 ∧
  -- Excellent chalk increase condition
  final.excellent = initial.excellent * 180 / 100 ∧
  -- Regular chalk decrease condition
  initial.regular - final.regular ≤ 10 ∧
  -- Total initial packs
  initial.regular + initial.unusual + initial.excellent = 390 :=
by sorry

end NUMINAMATH_CALUDE_chalk_inventory_theorem_l1276_127656


namespace NUMINAMATH_CALUDE_range_of_a_in_first_quadrant_l1276_127644

-- Define a complex number z with real part a and imaginary part (a-1)
def z (a : ℝ) : ℂ := Complex.mk a (a - 1)

-- Define what it means for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- State the theorem
theorem range_of_a_in_first_quadrant :
  ∀ a : ℝ, in_first_quadrant (z a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_first_quadrant_l1276_127644


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1276_127631

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-5 : ℝ) (4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1276_127631


namespace NUMINAMATH_CALUDE_enthalpy_change_proof_l1276_127651

-- Define the sum of standard formation enthalpies for products
def sum_enthalpy_products : ℝ := -286.0 - 297.0

-- Define the sum of standard formation enthalpies for reactants
def sum_enthalpy_reactants : ℝ := -20.17

-- Define Hess's Law
def hess_law (products reactants : ℝ) : ℝ := products - reactants

-- Theorem statement
theorem enthalpy_change_proof :
  hess_law sum_enthalpy_products sum_enthalpy_reactants = -1125.66 := by
  sorry

end NUMINAMATH_CALUDE_enthalpy_change_proof_l1276_127651


namespace NUMINAMATH_CALUDE_choose_three_from_nine_l1276_127664

theorem choose_three_from_nine (n : ℕ) (r : ℕ) (h1 : n = 9) (h2 : r = 3) :
  Nat.choose n r = 84 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_nine_l1276_127664


namespace NUMINAMATH_CALUDE_chocolate_kisses_bags_l1276_127616

theorem chocolate_kisses_bags (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (non_chocolate_pieces : ℕ) :
  total_candy = 63 →
  total_bags = 9 →
  heart_bags = 2 →
  non_chocolate_pieces = 28 →
  total_candy % total_bags = 0 →
  ∃ (kisses_bags : ℕ),
    kisses_bags = total_bags - heart_bags - (non_chocolate_pieces / (total_candy / total_bags)) ∧
    kisses_bags = 3 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_kisses_bags_l1276_127616


namespace NUMINAMATH_CALUDE_special_number_not_perfect_square_l1276_127647

/-- A number composed of exactly 100 zeros, 100 ones, and 100 twos -/
def special_number : ℕ :=
  -- We don't need to define the exact number, just its properties
  sorry

/-- The sum of digits of the special number -/
def sum_of_digits : ℕ := 300

/-- Theorem: The special number is not a perfect square -/
theorem special_number_not_perfect_square :
  ∀ n : ℕ, n ^ 2 ≠ special_number := by
  sorry

end NUMINAMATH_CALUDE_special_number_not_perfect_square_l1276_127647


namespace NUMINAMATH_CALUDE_map_scale_l1276_127645

/-- Given a map where 10 cm represents 50 km, prove that 15 cm represents 75 km -/
theorem map_scale (scale : ℝ → ℝ) : 
  (scale 10 = 50) → (scale 15 = 75) := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l1276_127645


namespace NUMINAMATH_CALUDE_arrangement_inequality_l1276_127648

-- Define the arrangement function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the set of valid x values
def valid_x : Set ℕ := {x | 3 ≤ x ∧ x ≤ 7}

-- State the theorem
theorem arrangement_inequality (x : ℕ) (h1 : 2 < x) (h2 : x ≤ 9) :
  A 9 x > 6 * A 9 (x - 2) ↔ x ∈ valid_x :=
sorry

end NUMINAMATH_CALUDE_arrangement_inequality_l1276_127648


namespace NUMINAMATH_CALUDE_patricia_lemon_heads_l1276_127606

/-- The number of Lemon Heads Patricia ate -/
def eaten : ℕ := 15

/-- The number of Lemon Heads Patricia gave to her friend -/
def given : ℕ := 5

/-- The number of Lemon Heads in each package -/
def per_package : ℕ := 3

/-- The function to calculate the number of packages -/
def calculate_packages (total : ℕ) : ℕ :=
  (total + per_package - 1) / per_package

/-- Theorem stating that Patricia originally had 7 packages of Lemon Heads -/
theorem patricia_lemon_heads : calculate_packages (eaten + given) = 7 := by
  sorry

end NUMINAMATH_CALUDE_patricia_lemon_heads_l1276_127606


namespace NUMINAMATH_CALUDE_equation_solution_l1276_127678

theorem equation_solution : ∃! x : ℝ, (1 + x) / 4 - (x - 2) / 8 = 1 := by
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1276_127678


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1276_127623

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2) * (z + 2)^2) ≤ 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1276_127623


namespace NUMINAMATH_CALUDE_polynomial_not_factorizable_l1276_127621

theorem polynomial_not_factorizable :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ),
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = x^200 * y^200 + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_not_factorizable_l1276_127621


namespace NUMINAMATH_CALUDE_min_value_of_f_f_is_even_f_monotone_increasing_l1276_127613

noncomputable section

-- Define the operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Define the function f
def f (x : ℝ) : ℝ := 1 + Real.exp x + 1 / Real.exp x

-- Theorem statements
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 3 ∧ ∃ x₀ : ℝ, f x₀ = 3 := by sorry

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

theorem f_monotone_increasing : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ x → f y ≥ f x := by sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_f_is_even_f_monotone_increasing_l1276_127613


namespace NUMINAMATH_CALUDE_function_expression_l1276_127654

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_expression (f : ℝ → ℝ) :
  is_periodic f 2 →
  is_even f →
  (∀ x ∈ Set.Icc 2 3, f x = x) →
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
by sorry

end NUMINAMATH_CALUDE_function_expression_l1276_127654


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1276_127681

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1276_127681


namespace NUMINAMATH_CALUDE_exam_results_l1276_127694

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l1276_127694


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1276_127633

/-- An arithmetic sequence (b_n) with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  x^2 + (n - 1) * x

theorem arithmetic_sequence_eighth_term (x : ℝ) :
  arithmetic_sequence x 8 = 2 * x^2 + 7 * x := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1276_127633


namespace NUMINAMATH_CALUDE_cube_greater_than_l1276_127612

theorem cube_greater_than (a b : ℝ) : a > b → ¬(a^3 ≤ b^3) := by
  sorry

end NUMINAMATH_CALUDE_cube_greater_than_l1276_127612


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_and_c_are_2_l1276_127689

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions given in the problem
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - t.b * t.c

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.a = 2 ∧ t.b + t.c = 4

-- Theorem 1: If the first condition is satisfied, then angle A is 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : satisfiesCondition1 t) :
  t.A = 60 * (π / 180) := by sorry

-- Theorem 2: If both conditions are satisfied, then b = 2 and c = 2
theorem b_and_c_are_2 (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) :
  t.b = 2 ∧ t.c = 2 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_and_c_are_2_l1276_127689


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1276_127639

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax + 3y + 3 = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 3, c := 3 }

/-- The second line x + (a-2)y + l = 0 -/
def line2 (a l : ℝ) : Line :=
  { a := 1, b := a - 2, c := l }

/-- Theorem stating that the lines are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one (a l : ℝ) :
  parallel (line1 a) (line2 a l) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1276_127639


namespace NUMINAMATH_CALUDE_binary_101101_to_octal_55_l1276_127643

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to octal
def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

-- Theorem statement
theorem binary_101101_to_octal_55 :
  let binary := [true, false, true, true, false, true]
  decimal_to_octal (binary_to_decimal binary) = [5, 5] := by sorry

end NUMINAMATH_CALUDE_binary_101101_to_octal_55_l1276_127643


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1276_127679

theorem tan_alpha_plus_pi_sixth (α : Real) 
  (h : Real.cos α + 2 * Real.cos (α + π/3) = 0) : 
  Real.tan (α + π/6) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l1276_127679


namespace NUMINAMATH_CALUDE_closest_integer_to_thirteen_minus_sqrt_thirteen_l1276_127627

theorem closest_integer_to_thirteen_minus_sqrt_thirteen : 
  ∃ (n : ℤ), ∀ (m : ℤ), |13 - Real.sqrt 13 - n| ≤ |13 - Real.sqrt 13 - m| → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_thirteen_minus_sqrt_thirteen_l1276_127627


namespace NUMINAMATH_CALUDE_book_magazine_cost_l1276_127611

theorem book_magazine_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 18.40)
  (h2 : 2 * x + 3 * y = 17.60) :
  2 * x + y = 11.20 := by
  sorry

end NUMINAMATH_CALUDE_book_magazine_cost_l1276_127611


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l1276_127632

theorem paper_I_maximum_mark :
  ∀ (max_mark : ℕ) (passing_percentage : ℚ) (scored_marks failed_by : ℕ),
    passing_percentage = 52 / 100 →
    scored_marks = 45 →
    failed_by = 35 →
    (scored_marks + failed_by : ℚ) = passing_percentage * max_mark →
    max_mark = 154 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l1276_127632


namespace NUMINAMATH_CALUDE_yoongi_behind_count_l1276_127674

/-- Given a line of students, calculate the number of students behind a specific position. -/
def studentsBehindinLine (totalStudents : ℕ) (position : ℕ) : ℕ :=
  totalStudents - position

theorem yoongi_behind_count :
  let totalStudents : ℕ := 20
  let jungkookPosition : ℕ := 3
  let yoongiPosition : ℕ := jungkookPosition + 1
  studentsBehindinLine totalStudents yoongiPosition = 16 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_behind_count_l1276_127674


namespace NUMINAMATH_CALUDE_factors_of_81_l1276_127653

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l1276_127653


namespace NUMINAMATH_CALUDE_arbelos_equal_segments_l1276_127695

/-- Arbelos type representing the geometric figure --/
structure Arbelos where
  -- Define necessary components of an arbelos
  -- (placeholder for actual definition)

/-- Point type representing a point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line in the plane --/
structure Line where
  -- Define necessary components of a line
  -- (placeholder for actual definition)

/-- Function to check if a point is inside an arbelos --/
def isInsideArbelos (a : Arbelos) (p : Point) : Prop :=
  -- Define the condition for a point to be inside an arbelos
  sorry

/-- Function to check if two lines make equal angles with a given line --/
def makeEqualAngles (l1 l2 base : Line) : Prop :=
  -- Define the condition for two lines to make equal angles with a base line
  sorry

/-- Function to get the segment cut by an arbelos on a line --/
def segmentCutByArbelos (a : Arbelos) (l : Line) : ℝ :=
  -- Define how to calculate the segment cut by an arbelos on a line
  sorry

/-- Theorem statement --/
theorem arbelos_equal_segments 
  (a : Arbelos) (ac : Line) (d : Point) (l1 l2 : Line) :
  isInsideArbelos a d →
  makeEqualAngles l1 l2 ac →
  segmentCutByArbelos a l1 = segmentCutByArbelos a l2 :=
sorry

end NUMINAMATH_CALUDE_arbelos_equal_segments_l1276_127695


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1276_127697

theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
fun equilateral_side isosceles_perimeter isosceles_base =>
  let isosceles_side := equilateral_side
  let equilateral_perimeter := 3 * equilateral_side
  isosceles_perimeter = 2 * isosceles_side + isosceles_base ∧
  isosceles_perimeter = 40 ∧
  isosceles_base = 10 →
  equilateral_perimeter = 45

-- The proof would go here, but we'll skip it as requested
theorem equilateral_triangle_perimeter_proof :
  equilateral_triangle_perimeter 15 40 10 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1276_127697


namespace NUMINAMATH_CALUDE_first_loan_amount_l1276_127614

/-- Represents a student loan -/
structure Loan where
  amount : ℝ
  rate : ℝ

/-- Calculates the interest paid on a loan -/
def interest_paid (loan : Loan) : ℝ :=
  loan.amount * loan.rate

theorem first_loan_amount
  (loan1 loan2 : Loan)
  (h1 : loan2.rate = 0.09)
  (h2 : loan1.amount = loan2.amount + 1500)
  (h3 : interest_paid loan1 + interest_paid loan2 = 617)
  (h4 : loan2.amount = 4700) :
  loan1.amount = 6200 := by
  sorry

end NUMINAMATH_CALUDE_first_loan_amount_l1276_127614


namespace NUMINAMATH_CALUDE_first_player_wins_l1276_127622

/-- Represents a board in the game -/
structure Board :=
  (m : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents a move in the game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Represents the game state -/
structure GameState :=
  (board : Board)
  (currentPosition : Position)
  (usedSegments : List (Position × Position))

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Up => state.currentPosition.y < state.board.m
  | Move.Down => state.currentPosition.y > 0
  | Move.Left => state.currentPosition.x > 0
  | Move.Right => state.currentPosition.x < state.board.m - 1

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Represents a winning strategy for the first player -/
def winningStrategy (board : Board) : Prop :=
  ∃ (strategy : List Move),
    ∀ (opponentMoves : List Move),
      let finalState := (strategy ++ opponentMoves).foldl applyMove
        { board := board
        , currentPosition := ⟨0, 0⟩
        , usedSegments := []
        }
      ¬∃ (move : Move), isValidMove finalState move

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins (m : ℕ) (h : m > 1) :
  winningStrategy { m := m } :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l1276_127622


namespace NUMINAMATH_CALUDE_charity_event_selection_l1276_127670

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding A and B -/
def remaining_students : ℕ := total_students - 2

theorem charity_event_selection :
  choose total_students selected_students - choose remaining_students selected_students = 140 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_selection_l1276_127670


namespace NUMINAMATH_CALUDE_three_circles_cross_ratio_invariance_l1276_127684

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line by two points
structure Line where
  p1 : Point
  p2 : Point

-- Define the cross-ratio of four points on a line
def cross_ratio (p1 p2 p3 p4 : Point) : ℝ := sorry

-- Define a function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define a function to find the intersection points of a line and a circle
def line_circle_intersection (l : Line) (c : Circle) : Set Point := sorry

theorem three_circles_cross_ratio_invariance 
  (c1 c2 c3 : Circle) 
  (A B : Point) 
  (h1 : point_on_circle A c1 ∧ point_on_circle A c2 ∧ point_on_circle A c3)
  (h2 : point_on_circle B c1 ∧ point_on_circle B c2 ∧ point_on_circle B c3)
  (h3 : A ≠ B) :
  ∀ (l1 l2 : Line), 
  (point_on_line A l1 ∧ point_on_line A l2) →
  ∃ (P1 Q1 R1 P2 Q2 R2 : Point),
  (P1 ∈ line_circle_intersection l1 c1 ∧ 
   Q1 ∈ line_circle_intersection l1 c2 ∧ 
   R1 ∈ line_circle_intersection l1 c3 ∧
   P2 ∈ line_circle_intersection l2 c1 ∧ 
   Q2 ∈ line_circle_intersection l2 c2 ∧ 
   R2 ∈ line_circle_intersection l2 c3) →
  cross_ratio A P1 Q1 R1 = cross_ratio A P2 Q2 R2 :=
sorry

end NUMINAMATH_CALUDE_three_circles_cross_ratio_invariance_l1276_127684


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1276_127604

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 3) (h2 : p 3 = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (6 * x - 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1276_127604


namespace NUMINAMATH_CALUDE_average_of_squares_first_11_even_l1276_127668

/-- The first 11 consecutive even numbers -/
def first_11_even_numbers : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

/-- The average of squares of the first 11 consecutive even numbers -/
theorem average_of_squares_first_11_even : 
  (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length = 184 := by
  sorry

#eval (first_11_even_numbers.map (λ x => x^2)).sum / first_11_even_numbers.length

end NUMINAMATH_CALUDE_average_of_squares_first_11_even_l1276_127668


namespace NUMINAMATH_CALUDE_fair_coin_head_is_random_event_l1276_127619

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- Represents different types of events -/
inductive EventType
  | Impossible
  | Certain
  | Random
  | Definite

/-- A fair coin toss -/
structure FairCoinToss where
  /-- The coin has two possible outcomes -/
  outcome : CoinOutcome
  /-- The probability of getting heads is 0.5 -/
  prob_head : ℝ := 0.5
  /-- The probability of getting tails is 0.5 -/
  prob_tail : ℝ := 0.5
  /-- The probabilities sum to 1 -/
  prob_sum : prob_head + prob_tail = 1

/-- The theorem stating that tossing a fair coin with the head facing up is a random event -/
theorem fair_coin_head_is_random_event (toss : FairCoinToss) : 
  EventType.Random = 
    match toss.outcome with
    | CoinOutcome.Head => EventType.Random
    | CoinOutcome.Tail => EventType.Random :=
by
  sorry


end NUMINAMATH_CALUDE_fair_coin_head_is_random_event_l1276_127619


namespace NUMINAMATH_CALUDE_car_price_difference_car_price_difference_proof_l1276_127646

/-- The price difference between a new car and an old car, given specific conditions --/
theorem car_price_difference : ℝ → Prop :=
  fun price_difference =>
    ∃ (old_car_price : ℝ),
      -- New car costs $30,000
      let new_car_price : ℝ := 30000
      -- Down payment is 25% of new car price
      let down_payment : ℝ := 0.25 * new_car_price
      -- Old car sold at 80% of original price
      let old_car_sale_price : ℝ := 0.8 * old_car_price
      -- After selling old car and making down payment, $4000 more is needed
      old_car_sale_price + down_payment + 4000 = new_car_price ∧
      -- Price difference is the difference between new and old car prices
      price_difference = new_car_price - old_car_price ∧
      -- The price difference is $6875
      price_difference = 6875

/-- Proof of the car price difference theorem --/
theorem car_price_difference_proof : car_price_difference 6875 := by
  sorry

end NUMINAMATH_CALUDE_car_price_difference_car_price_difference_proof_l1276_127646


namespace NUMINAMATH_CALUDE_volume_of_four_cubes_l1276_127665

theorem volume_of_four_cubes (edge_length : ℝ) (num_boxes : ℕ) : 
  edge_length = 5 → num_boxes = 4 → num_boxes * (edge_length ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_four_cubes_l1276_127665


namespace NUMINAMATH_CALUDE_testes_mice_most_suitable_for_meiosis_l1276_127655

/-- Represents different types of biological materials --/
inductive BiologicalMaterial
  | FertilizedEggsAscaris
  | TestesMice
  | SpermLocusts
  | BloodChickens

/-- Represents different types of cell division --/
inductive CellDivision
  | Mitosis
  | Meiosis
  | Amitosis

/-- Defines the property of a biological material undergoing a specific type of cell division --/
def undergoes (m : BiologicalMaterial) (d : CellDivision) : Prop := sorry

/-- Defines the property of a biological material being suitable for observing a specific type of cell division --/
def suitableForObserving (m : BiologicalMaterial) (d : CellDivision) : Prop := sorry

/-- Defines the property of a biological material producing a large number of cells --/
def producesLargeNumberOfCells (m : BiologicalMaterial) : Prop := sorry

/-- Theorem stating that testes of mice are the most suitable material for observing meiosis among the given options --/
theorem testes_mice_most_suitable_for_meiosis :
  suitableForObserving BiologicalMaterial.TestesMice CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.FertilizedEggsAscaris CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.SpermLocusts CellDivision.Meiosis ∧
  ¬suitableForObserving BiologicalMaterial.BloodChickens CellDivision.Meiosis :=
by sorry

end NUMINAMATH_CALUDE_testes_mice_most_suitable_for_meiosis_l1276_127655


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1276_127617

theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, a k > 0) →  -- positive terms
  (∀ k, a (k + 1) / a k = a (k + 2) / a (k + 1)) →  -- geometric sequence
  (a 3 = 4) →
  (a 4 * a 5 * a 6 = 2^12) →
  (S n = 2^10 - 1) →
  (S n = (a 1 * (1 - (a 2 / a 1)^n)) / (1 - (a 2 / a 1))) →  -- sum formula for geometric sequence
  (a 1 = 1 ∧ a 2 / a 1 = 2 ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1276_127617


namespace NUMINAMATH_CALUDE_max_cookies_without_ingredients_l1276_127635

/-- Given a set of cookies with specific ingredient distributions, 
    prove the maximum number of cookies without any of the ingredients. -/
theorem max_cookies_without_ingredients (total_cookies : ℕ) 
    (h_total : total_cookies = 48)
    (h_choc_chips : (total_cookies / 2 : ℕ) = 24)
    (h_peanut_butter : (total_cookies * 3 / 4 : ℕ) = 36)
    (h_white_choc : (total_cookies / 3 : ℕ) = 16)
    (h_coconut : (total_cookies / 8 : ℕ) = 6) :
    ∃ (max_without : ℕ), max_without ≤ 12 ∧ 
    max_without = total_cookies - (total_cookies * 3 / 4 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_max_cookies_without_ingredients_l1276_127635


namespace NUMINAMATH_CALUDE_jimmys_bet_l1276_127666

/-- Represents a fan with equally spaced blades -/
structure Fan where
  num_blades : ℕ
  revolutions_per_second : ℝ

/-- Represents a bullet shot -/
structure Bullet where
  shot_time : ℝ
  speed : ℝ

/-- Predicate that determines if a bullet can hit all blades of a fan -/
def can_hit_all_blades (f : Fan) (b : Bullet) : Prop :=
  ∃ t : ℝ, ∀ i : Fin f.num_blades, 
    ∃ k : ℤ, b.shot_time + (i : ℝ) * (1 / f.num_blades) = t + k / f.revolutions_per_second

/-- Theorem stating that for a fan with 4 blades rotating at 50 revolutions per second,
    there exists a bullet that can hit all blades -/
theorem jimmys_bet : 
  ∃ b : Bullet, can_hit_all_blades ⟨4, 50⟩ b :=
sorry

end NUMINAMATH_CALUDE_jimmys_bet_l1276_127666


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l1276_127609

/-- Given a line intersecting a circle, prove the range of the parameter a -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.1 + A.2 + a = 0) ∧ (B.1 + B.2 + a = 0) ∧
   (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧
   (‖(A.1, A.2)‖ + ‖(B.1, B.2)‖)^2 ≥ ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  a ∈ Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Icc 1 (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l1276_127609


namespace NUMINAMATH_CALUDE_fans_with_all_items_l1276_127693

/-- The maximum capacity of the stadium --/
def stadium_capacity : ℕ := 3000

/-- The interval at which t-shirts are given --/
def tshirt_interval : ℕ := 50

/-- The interval at which caps are given --/
def cap_interval : ℕ := 25

/-- The interval at which wristbands are given --/
def wristband_interval : ℕ := 60

/-- Theorem stating that the number of fans receiving all three items is 10 --/
theorem fans_with_all_items : 
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l1276_127693


namespace NUMINAMATH_CALUDE_gcd_11121_12012_l1276_127698

theorem gcd_11121_12012 : Nat.gcd 11121 12012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_11121_12012_l1276_127698


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1276_127625

-- Define the side length of the small squares
def small_square_side : ℝ := sorry

-- Define the side length of the large square
def large_square_side : ℝ := 3 * small_square_side

-- Define the length of the rectangle
def rectangle_length : ℝ := large_square_side

-- Define the width of the rectangle
def rectangle_width : ℝ := small_square_side

-- Theorem stating that the ratio of rectangle's length to width is 3
theorem rectangle_ratio : rectangle_length / rectangle_width = 3 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1276_127625


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1276_127636

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃! x : ℕ, x < d ∧ (n - x) % d = 0 :=
by sorry

theorem problem_solution :
  let n := 42739
  let d := 15
  (least_subtraction_for_divisibility n d (by norm_num)).choose = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1276_127636


namespace NUMINAMATH_CALUDE_complex_equation_implies_difference_l1276_127685

theorem complex_equation_implies_difference (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by sorry

end NUMINAMATH_CALUDE_complex_equation_implies_difference_l1276_127685


namespace NUMINAMATH_CALUDE_nap_time_calculation_l1276_127660

/-- Calculates the remaining time for a nap given flight duration and time spent on activities -/
def remaining_nap_time (
  flight_hours : ℕ
) (
  flight_minutes : ℕ
) (
  reading_hours : ℕ
) (
  movie_hours : ℕ
) (
  dinner_minutes : ℕ
) (
  radio_minutes : ℕ
) (
  game_hours : ℕ
) (
  game_minutes : ℕ
) : ℕ :=
  let total_flight_minutes := flight_hours * 60 + flight_minutes
  let total_activity_minutes := 
    reading_hours * 60 + 
    movie_hours * 60 + 
    dinner_minutes + 
    radio_minutes + 
    game_hours * 60 + 
    game_minutes
  (total_flight_minutes - total_activity_minutes) / 60

theorem nap_time_calculation : 
  remaining_nap_time 11 20 2 4 30 40 1 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nap_time_calculation_l1276_127660


namespace NUMINAMATH_CALUDE_molly_current_age_l1276_127610

/-- Represents Molly's age and candle information --/
structure MollysBirthday where
  last_year_candles : ℕ
  additional_candles : ℕ
  friend_gift_candles : ℕ

/-- Calculates Molly's current age based on her birthday information --/
def current_age (mb : MollysBirthday) : ℕ :=
  mb.last_year_candles + 1

/-- Theorem stating Molly's current age --/
theorem molly_current_age (mb : MollysBirthday)
  (h1 : mb.last_year_candles = 14)
  (h2 : mb.additional_candles = 6)
  (h3 : mb.friend_gift_candles = 3) :
  current_age mb = 15 := by
  sorry

end NUMINAMATH_CALUDE_molly_current_age_l1276_127610


namespace NUMINAMATH_CALUDE_special_triangle_relation_l1276_127699

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

end NUMINAMATH_CALUDE_special_triangle_relation_l1276_127699


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1276_127602

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : subset l α)
  (h2 : perp l β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1276_127602


namespace NUMINAMATH_CALUDE_fraction_simplification_l1276_127638

theorem fraction_simplification :
  (1/2 + 1/3) / (3/4 - 1/5) = 50/33 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1276_127638


namespace NUMINAMATH_CALUDE_brothers_identity_l1276_127607

-- Define the two brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define a function to represent a brother's statement
def statement (b : Brother) : Brother :=
  match b with
  | Brother.Tweedledum => Brother.Tweedledum
  | Brother.Tweedledee => Brother.Tweedledee

-- Define the consistency of statements
def consistent (first second : Brother) : Prop :=
  (statement first = Brother.Tweedledum) ∧ (statement second = Brother.Tweedledee)

-- Theorem: The only consistent scenario is when both brothers tell the truth
theorem brothers_identity :
  ∀ (first second : Brother),
    consistent first second →
    (first = Brother.Tweedledum ∧ second = Brother.Tweedledee) :=
by sorry


end NUMINAMATH_CALUDE_brothers_identity_l1276_127607


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1276_127677

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1276_127677


namespace NUMINAMATH_CALUDE_jogger_train_distance_l1276_127641

/-- Proves the distance a jogger is ahead of a train engine given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time = train_length + 230 := by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l1276_127641


namespace NUMINAMATH_CALUDE_four_points_plane_count_l1276_127659

/-- A set of four points in three-dimensional space -/
structure FourPoints where
  points : Fin 4 → ℝ × ℝ × ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : ℝ × ℝ × ℝ) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (fp : FourPoints) : Prop :=
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ¬(are_collinear (fp.points i) (fp.points j) (fp.points k))

/-- The number of planes determined by four points -/
def num_planes (fp : FourPoints) : ℕ := sorry

/-- Theorem: Given four points in space where no three points are collinear, 
    the number of planes these points can determine is either 1 or 4 -/
theorem four_points_plane_count (fp : FourPoints) 
  (h : no_three_collinear fp) : 
  num_planes fp = 1 ∨ num_planes fp = 4 := by sorry

end NUMINAMATH_CALUDE_four_points_plane_count_l1276_127659


namespace NUMINAMATH_CALUDE_point_sqrt_6_away_from_origin_l1276_127634

-- Define a point on the number line
def Point := ℝ

-- Define the distance function
def distance (p : Point) : ℝ := |p|

-- State the theorem
theorem point_sqrt_6_away_from_origin (M : Point) 
  (h : distance M = Real.sqrt 6) : M = Real.sqrt 6 ∨ M = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_point_sqrt_6_away_from_origin_l1276_127634


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1276_127642

theorem batsman_average_increase 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (new_average : ℚ) :
  total_innings = 17 →
  last_innings_score = 85 →
  new_average = 37 →
  (total_innings * new_average - last_innings_score) / (total_innings - 1) + 3 = new_average :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1276_127642


namespace NUMINAMATH_CALUDE_candy_pack_cost_l1276_127657

theorem candy_pack_cost (cory_has : ℝ) (cory_needs : ℝ) (num_packs : ℕ) :
  cory_has = 20 →
  cory_needs = 78 →
  num_packs = 2 →
  (cory_has + cory_needs) / num_packs = 49 := by
  sorry

end NUMINAMATH_CALUDE_candy_pack_cost_l1276_127657


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l1276_127601

theorem polynomial_expansion_theorem (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1/2 → 
  (28 : ℝ) * a^6 * b^2 = (56 : ℝ) * a^5 * b^3 → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l1276_127601


namespace NUMINAMATH_CALUDE_circle_equation_l1276_127658

/-- The general equation of a circle with specific properties -/
theorem circle_equation (x y : ℝ) : 
  ∃ (h k : ℝ), 
    (k = -4 * h) ∧ 
    ((3 - h)^2 + (-2 - k)^2 = (3 + (-2) - 1)^2) ∧
    (∀ (a b : ℝ), (a + b - 1 = 0) → ((a - h)^2 + (b - k)^2 ≥ (3 + (-2) - 1)^2)) →
    x^2 + y^2 - 2*x + 8*y + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1276_127658


namespace NUMINAMATH_CALUDE_calculate_expression_l1276_127688

theorem calculate_expression : (-3)^2 - (1/5)⁻¹ - Real.sqrt 8 * Real.sqrt 2 + (-2)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1276_127688


namespace NUMINAMATH_CALUDE_find_S_l1276_127676

-- Define the relationship between R, S, and T
def relationship (c R S T : ℝ) : Prop :=
  R = c * (S / T)

-- Define the theorem
theorem find_S (c : ℝ) :
  relationship c (4/3) (3/7) (9/14) →
  relationship c (Real.sqrt 98) S (Real.sqrt 32) →
  S = 28 := by
  sorry


end NUMINAMATH_CALUDE_find_S_l1276_127676


namespace NUMINAMATH_CALUDE_jericho_debt_ratio_l1276_127687

theorem jericho_debt_ratio :
  ∀ (jericho_money annika_debt manny_debt : ℚ),
    2 * jericho_money = 60 →
    annika_debt = 14 →
    jericho_money - annika_debt - manny_debt = 9 →
    manny_debt / annika_debt = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jericho_debt_ratio_l1276_127687


namespace NUMINAMATH_CALUDE_marcella_lost_shoes_l1276_127669

/-- Given the initial number of shoe pairs and the final number of matching pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * final_pairs

/-- Theorem stating that Marcella lost 10 individual shoes. -/
theorem marcella_lost_shoes : shoes_lost 23 18 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marcella_lost_shoes_l1276_127669


namespace NUMINAMATH_CALUDE_min_value_expression_l1276_127673

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
  (2 * x^2 + 1) / (x * y) - 2 ≥ min_val ∧
  (2 * a^2 + 1) / (a * b) - 2 = min_val ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1276_127673


namespace NUMINAMATH_CALUDE_min_value_expression_l1276_127661

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -1640.25 ∧
  ∀ y : ℝ, (15 - y) * (12 - y) * (15 + y) * (12 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1276_127661


namespace NUMINAMATH_CALUDE_remainder_problem_l1276_127686

theorem remainder_problem (x : ℤ) : 
  x % 82 = 5 → (x + 13) % 41 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1276_127686


namespace NUMINAMATH_CALUDE_jake_paid_forty_l1276_127683

/-- Calculates the amount paid before working given initial debt, hourly rate, hours worked, and that the remaining debt was paid off by working. -/
def amount_paid_before_working (initial_debt : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  initial_debt - (hourly_rate * hours_worked)

/-- Proves that Jake paid $40 before working, given the problem conditions. -/
theorem jake_paid_forty :
  amount_paid_before_working 100 15 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jake_paid_forty_l1276_127683


namespace NUMINAMATH_CALUDE_certain_number_value_l1276_127615

/-- Given that the average of 100, 200, 300, and x is 250,
    and the average of 300, 150, 100, x, and y is 200,
    prove that y = 50 -/
theorem certain_number_value (x : ℝ) (y : ℝ) 
    (h1 : (100 + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : 
  y = 50 := by sorry

end NUMINAMATH_CALUDE_certain_number_value_l1276_127615


namespace NUMINAMATH_CALUDE_kelly_initial_games_l1276_127605

/-- The number of games Kelly needs to give away -/
def games_to_give : ℕ := 15

/-- The number of games Kelly will have left after giving away some games -/
def games_left : ℕ := 35

/-- Kelly's initial number of games -/
def initial_games : ℕ := games_left + games_to_give

theorem kelly_initial_games : initial_games = 50 := by sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l1276_127605
