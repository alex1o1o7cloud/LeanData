import Mathlib

namespace NUMINAMATH_CALUDE_inclination_angle_range_l3811_381122

theorem inclination_angle_range (α : Real) (h : α ∈ Set.Icc (π / 6) (2 * π / 3)) :
  let θ := Real.arctan (2 * Real.cos α)
  θ ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3811_381122


namespace NUMINAMATH_CALUDE_brother_scores_double_l3811_381114

/-- Represents the hockey goal scoring scenario of Louie and his brother -/
structure HockeyScenario where
  louie_last_match : ℕ
  louie_previous : ℕ
  brother_seasons : ℕ
  games_per_season : ℕ
  total_goals : ℕ

/-- The ratio of Louie's brother's goals per game to Louie's goals in the last match -/
def brother_to_louie_ratio (h : HockeyScenario) : ℚ :=
  let brother_total_games := h.brother_seasons * h.games_per_season
  let brother_total_goals := h.total_goals - (h.louie_last_match + h.louie_previous)
  (brother_total_goals / brother_total_games : ℚ) / h.louie_last_match

/-- The main theorem stating the ratio is 2:1 -/
theorem brother_scores_double (h : HockeyScenario) 
    (h_louie_last : h.louie_last_match = 4)
    (h_louie_prev : h.louie_previous = 40)
    (h_seasons : h.brother_seasons = 3)
    (h_games : h.games_per_season = 50)
    (h_total : h.total_goals = 1244) : 
  brother_to_louie_ratio h = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_scores_double_l3811_381114


namespace NUMINAMATH_CALUDE_max_viewership_l3811_381140

structure Series where
  runtime : ℕ
  commercials : ℕ
  viewers : ℕ

def seriesA : Series := { runtime := 80, commercials := 1, viewers := 600000 }
def seriesB : Series := { runtime := 40, commercials := 1, viewers := 200000 }

def totalProgramTime : ℕ := 320
def minCommercials : ℕ := 6

def Schedule := ℕ × ℕ  -- (number of A episodes, number of B episodes)

def isValidSchedule (s : Schedule) : Prop :=
  s.1 * seriesA.runtime + s.2 * seriesB.runtime ≤ totalProgramTime ∧
  s.1 * seriesA.commercials + s.2 * seriesB.commercials ≥ minCommercials

def viewership (s : Schedule) : ℕ :=
  s.1 * seriesA.viewers + s.2 * seriesB.viewers

theorem max_viewership :
  ∃ (s : Schedule), isValidSchedule s ∧
    ∀ (s' : Schedule), isValidSchedule s' → viewership s' ≤ viewership s ∧
    viewership s = 2000000 :=
  sorry

end NUMINAMATH_CALUDE_max_viewership_l3811_381140


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l3811_381115

theorem typhoon_tree_difference (initial_trees : ℕ) (survival_rate : ℚ) : 
  initial_trees = 25 → 
  survival_rate = 2/5 → 
  (initial_trees - (survival_rate * initial_trees).floor) - (survival_rate * initial_trees).floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l3811_381115


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_l3811_381159

theorem binomial_coefficient_third_term (a b : ℝ) : 
  Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_l3811_381159


namespace NUMINAMATH_CALUDE_barrels_for_remaining_road_l3811_381157

/-- Represents the road paving problem -/
structure RoadPaving where
  total_length : ℝ
  truckloads_per_mile : ℝ
  day1_paved : ℝ
  day2_paved : ℝ
  pitch_per_truckload : ℝ

/-- Calculates the barrels of pitch needed for the remaining road -/
def barrels_needed (rp : RoadPaving) : ℝ :=
  (rp.total_length - (rp.day1_paved + rp.day2_paved)) * rp.truckloads_per_mile * rp.pitch_per_truckload

/-- Theorem stating the number of barrels needed for the given scenario -/
theorem barrels_for_remaining_road :
  let rp : RoadPaving := {
    total_length := 16,
    truckloads_per_mile := 3,
    day1_paved := 4,
    day2_paved := 7,
    pitch_per_truckload := 0.4
  }
  barrels_needed rp = 6 := by sorry

end NUMINAMATH_CALUDE_barrels_for_remaining_road_l3811_381157


namespace NUMINAMATH_CALUDE_paintings_from_C_l3811_381180

-- Define the number of paintings from each school
variable (A B C : ℕ)

-- Define the total number of paintings
def T : ℕ := A + B + C

-- State the conditions
axiom not_from_A : B + C = 41
axiom not_from_B : A + C = 38
axiom from_A_and_B : A + B = 43

-- State the theorem to be proved
theorem paintings_from_C : C = 18 := by sorry

end NUMINAMATH_CALUDE_paintings_from_C_l3811_381180


namespace NUMINAMATH_CALUDE_garbage_collection_difference_l3811_381150

theorem garbage_collection_difference (daliah_amount dewei_amount zane_amount : ℝ) : 
  daliah_amount = 17.5 →
  zane_amount = 62 →
  zane_amount = 4 * dewei_amount →
  dewei_amount < daliah_amount →
  daliah_amount - dewei_amount = 2 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_difference_l3811_381150


namespace NUMINAMATH_CALUDE_collinear_complex_points_l3811_381113

theorem collinear_complex_points (z : ℂ) : 
  (∃ (t : ℝ), z = 1 + t * (Complex.I - 1)) → Complex.abs z = 5 → 
  (z = 4 - 3 * Complex.I ∨ z = -3 + 4 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_collinear_complex_points_l3811_381113


namespace NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_for_inequality_l3811_381174

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

-- Theorem for part I
theorem min_value_of_f_over_x (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (y : ℝ), y > 0 → f 2 y / y ≥ min_val :=
sorry

-- Theorem for part II
theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f a x ≤ a) ↔ a ≥ -2 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_over_x_range_of_a_for_inequality_l3811_381174


namespace NUMINAMATH_CALUDE_symmetry_axis_l3811_381199

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry property of f
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → f (2 * a - x) = y

-- Theorem stating that x = 1.5 is an axis of symmetry
theorem symmetry_axis :
  is_axis_of_symmetry 1.5 :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l3811_381199


namespace NUMINAMATH_CALUDE_factory_production_l3811_381168

/-- Calculates the total television production in the second year given the daily production rate in the first year and the reduction percentage. -/
def secondYearProduction (dailyRate : ℕ) (reductionPercent : ℕ) : ℕ :=
  let firstYearTotal := dailyRate * 365
  let reduction := firstYearTotal * reductionPercent / 100
  firstYearTotal - reduction

/-- Theorem stating that for a factory producing 10 televisions per day in the first year
    and reducing production by 10% in the second year, the total production in the second year is 3285. -/
theorem factory_production :
  secondYearProduction 10 10 = 3285 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l3811_381168


namespace NUMINAMATH_CALUDE_students_not_liking_sports_l3811_381167

theorem students_not_liking_sports (total : ℕ) (basketball : ℕ) (tableTennis : ℕ) (both : ℕ) :
  total = 30 →
  basketball = 15 →
  tableTennis = 10 →
  both = 3 →
  total - (basketball + tableTennis - both) = 8 :=
by sorry

end NUMINAMATH_CALUDE_students_not_liking_sports_l3811_381167


namespace NUMINAMATH_CALUDE_reducible_factorial_fraction_l3811_381131

theorem reducible_factorial_fraction (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k ∣ n.factorial ∧ k ∣ (n + 1)) ↔
  (n % 2 = 1 ∧ n > 1) ∨ (n % 2 = 0 ∧ ¬(Nat.Prime (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_reducible_factorial_fraction_l3811_381131


namespace NUMINAMATH_CALUDE_hiker_catches_cyclist_l3811_381197

/-- Proves that a hiker catches up to a cyclist in 15 minutes under specific conditions --/
theorem hiker_catches_cyclist (hiker_speed cyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 7 →
  cyclist_speed = 28 →
  stop_time = 5 / 60 →
  let distance_cyclist := cyclist_speed * stop_time
  let distance_hiker := hiker_speed * stop_time
  let distance_difference := distance_cyclist - distance_hiker
  let catch_up_time := distance_difference / hiker_speed
  catch_up_time * 60 = 15 := by
  sorry

#check hiker_catches_cyclist

end NUMINAMATH_CALUDE_hiker_catches_cyclist_l3811_381197


namespace NUMINAMATH_CALUDE_log_like_function_72_l3811_381171

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = f a + f b

/-- Theorem: If f is a LogLikeFunction with f(2) = m and f(3) = n, then f(72) = 3m + 2n -/
theorem log_like_function_72 (f : ℝ → ℝ) (m n : ℝ) 
  (h_log_like : LogLikeFunction f) (h_2 : f 2 = m) (h_3 : f 3 = n) : 
  f 72 = 3 * m + 2 * n := by
sorry

end NUMINAMATH_CALUDE_log_like_function_72_l3811_381171


namespace NUMINAMATH_CALUDE_cafeteria_pies_correct_l3811_381184

def cafeteria_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies_correct : cafeteria_pies 47 27 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_correct_l3811_381184


namespace NUMINAMATH_CALUDE_exists_four_unacquainted_l3811_381185

/-- A type representing a person in the group -/
def Person : Type := Fin 10

/-- The acquaintance relation between people -/
def acquainted : Person → Person → Prop := sorry

theorem exists_four_unacquainted 
  (h1 : ∀ p : Person, ∃! (q r : Person), q ≠ r ∧ acquainted p q ∧ acquainted p r)
  (h2 : ∀ p q : Person, acquainted p q → acquainted q p) :
  ∃ (a b c d : Person), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ¬acquainted a b ∧ ¬acquainted a c ∧ ¬acquainted a d ∧
    ¬acquainted b c ∧ ¬acquainted b d ∧ ¬acquainted c d :=
sorry

end NUMINAMATH_CALUDE_exists_four_unacquainted_l3811_381185


namespace NUMINAMATH_CALUDE_product_nine_consecutive_divisible_by_ten_l3811_381164

theorem product_nine_consecutive_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) = 10 * k :=
by sorry

end NUMINAMATH_CALUDE_product_nine_consecutive_divisible_by_ten_l3811_381164


namespace NUMINAMATH_CALUDE_gain_percent_for_80_and_58_l3811_381147

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles sellingArticles : ℕ) : ℚ :=
  let ratio : ℚ := costArticles / sellingArticles
  (ratio - 1) / ratio * 100

theorem gain_percent_for_80_and_58 :
  gainPercent 80 58 = 11 / 29 * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_for_80_and_58_l3811_381147


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3811_381161

theorem algebraic_expression_symmetry (a b c : ℝ) :
  (a * (-5)^4 + b * (-5)^2 + c = 3) →
  (a * 5^4 + b * 5^2 + c = 3) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3811_381161


namespace NUMINAMATH_CALUDE_squares_ending_in_76_l3811_381179

theorem squares_ending_in_76 : 
  {x : ℕ | x^2 % 100 = 76} = {24, 26, 74, 76} := by sorry

end NUMINAMATH_CALUDE_squares_ending_in_76_l3811_381179


namespace NUMINAMATH_CALUDE_ratio_transformation_l3811_381142

theorem ratio_transformation (x : ℚ) : 
  ((2 : ℚ) + 2) / (x + 2) = 4 / 5 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l3811_381142


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l3811_381110

theorem marble_selection_theorem (total_marbles special_marbles marbles_to_choose : ℕ) 
  (h1 : total_marbles = 18)
  (h2 : special_marbles = 6)
  (h3 : marbles_to_choose = 4) :
  (Nat.choose special_marbles 2) * (Nat.choose (total_marbles - special_marbles) 2) = 990 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_theorem_l3811_381110


namespace NUMINAMATH_CALUDE_f_leq_one_iff_a_range_l3811_381134

-- Define the function f
def f (a x : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- State the theorem
theorem f_leq_one_iff_a_range (a : ℝ) :
  (∀ x, f a x ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_f_leq_one_iff_a_range_l3811_381134


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l3811_381104

/-- An arithmetic sequence with a_5 = 15 -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 5 = 15

/-- Theorem: In an arithmetic sequence where a_5 = 15, the sum of a_2, a_4, a_6, and a_8 is 60 -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (h : arithmeticSequence a) :
  a 2 + a 4 + a 6 + a 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l3811_381104


namespace NUMINAMATH_CALUDE_equilateral_triangle_pq_l3811_381178

/-- Given an equilateral triangle with vertices at (0,0), (p, 13), and (q, 41),
    prove that the product pq equals -2123/3 -/
theorem equilateral_triangle_pq (p q : ℝ) : 
  (∃ (z : ℂ), z^3 = 1 ∧ z ≠ 1 ∧ z * (p + 13*I) = q + 41*I) →
  p * q = -2123/3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_pq_l3811_381178


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_function_l3811_381112

theorem max_value_of_sin_cos_function :
  ∃ (M : ℝ), M = 17 ∧ ∀ x, 8 * Real.sin x + 15 * Real.cos x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_function_l3811_381112


namespace NUMINAMATH_CALUDE_expand_expression_l3811_381153

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x + 6) + x = 2 * x^2 + 19 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3811_381153


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3811_381169

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the first prime after 7 consecutive non-primes
def isFirstPrimeAfter7NonPrimes (p : ℕ) : Prop :=
  isPrime p ∧
  ∀ k : ℕ, k ∈ Finset.range 7 → ¬isPrime (p - k - 1) ∧
  ∀ q : ℕ, q < p → isFirstPrimeAfter7NonPrimes q → False

-- State the theorem
theorem smallest_prime_after_seven_nonprimes :
  isFirstPrimeAfter7NonPrimes 97 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3811_381169


namespace NUMINAMATH_CALUDE_initial_payment_calculation_l3811_381103

theorem initial_payment_calculation (car_cost installment_amount : ℕ) (num_installments : ℕ) 
  (h1 : car_cost = 18000)
  (h2 : installment_amount = 2500)
  (h3 : num_installments = 6) :
  car_cost - (num_installments * installment_amount) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_initial_payment_calculation_l3811_381103


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3811_381111

theorem negative_fraction_comparison : -6/5 > -5/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3811_381111


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l3811_381193

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 :=
by sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 :=
by sorry

theorem less_than_150 : 145 < 150 :=
by sorry

theorem is_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 :=
by sorry

theorem main_result : (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n)) ∧ 
  (∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ 
  (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) ∧ n = 145) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_gcf_of_145_and_30_less_than_150_is_greatest_main_result_l3811_381193


namespace NUMINAMATH_CALUDE_min_transportation_cost_min_cost_at_ten_l3811_381109

/-- Represents the total transportation cost function --/
def transportation_cost (x : ℝ) : ℝ := 4 * x + 1980

/-- Theorem stating the minimum transportation cost --/
theorem min_transportation_cost :
  ∀ x : ℝ, 10 ≤ x ∧ x ≤ 50 → transportation_cost x ≥ 2020 :=
by
  sorry

/-- Theorem stating that the minimum cost occurs at x = 10 --/
theorem min_cost_at_ten :
  transportation_cost 10 = 2020 :=
by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_min_cost_at_ten_l3811_381109


namespace NUMINAMATH_CALUDE_sum_in_base4_l3811_381172

/-- Represents a number in base 4 -/
def Base4 : Type := List (Fin 4)

/-- Addition of two Base4 numbers -/
def add_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from a natural number to Base4 -/
def nat_to_base4 : ℕ → Base4 := sorry

/-- Conversion from Base4 to a natural number -/
def base4_to_nat : Base4 → ℕ := sorry

theorem sum_in_base4 :
  let a : Base4 := nat_to_base4 211
  let b : Base4 := nat_to_base4 332
  let c : Base4 := nat_to_base4 123
  let result : Base4 := nat_to_base4 1120
  add_base4 (add_base4 a b) c = result := by sorry

end NUMINAMATH_CALUDE_sum_in_base4_l3811_381172


namespace NUMINAMATH_CALUDE_orange_cost_24_pounds_l3811_381188

/-- The cost of oranges given a rate and a quantity -/
def orange_cost (rate_price : ℚ) (rate_weight : ℚ) (weight : ℚ) : ℚ :=
  (rate_price / rate_weight) * weight

/-- Theorem: The cost of 24 pounds of oranges at a rate of $6 per 8 pounds is $18 -/
theorem orange_cost_24_pounds : orange_cost 6 8 24 = 18 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_24_pounds_l3811_381188


namespace NUMINAMATH_CALUDE_dean_taller_than_ron_l3811_381177

theorem dean_taller_than_ron (water_depth : ℝ) (ron_height : ℝ) (dean_height : ℝ) :
  water_depth = 255 ∧ ron_height = 13 ∧ water_depth = 15 * dean_height →
  dean_height - ron_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_dean_taller_than_ron_l3811_381177


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l3811_381195

theorem quadratic_equations_common_root (a b : ℝ) : 
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l3811_381195


namespace NUMINAMATH_CALUDE_book_arrangements_eq_34560_l3811_381145

/-- The number of ways to arrange 11 books (3 Arabic, 2 English, 4 Spanish, and 2 French) on a shelf,
    keeping Arabic, Spanish, and English books together respectively. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let english_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let group_arrangements : ℕ := Nat.factorial 5
  let arabic_internal_arrangements : ℕ := Nat.factorial arabic_books
  let english_internal_arrangements : ℕ := Nat.factorial english_books
  let spanish_internal_arrangements : ℕ := Nat.factorial spanish_books
  group_arrangements * arabic_internal_arrangements * english_internal_arrangements * spanish_internal_arrangements

theorem book_arrangements_eq_34560 : book_arrangements = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_34560_l3811_381145


namespace NUMINAMATH_CALUDE_digit_multiplication_puzzle_l3811_381107

def is_single_digit (n : ℕ) : Prop := n < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def number_from_digits (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem digit_multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    is_single_digit e ∧
    is_five_digit_number (number_from_digits a b c d e) ∧
    4 * (number_from_digits a b c d e) = number_from_digits e d c b a →
    a = 2 ∧ b = 1 ∧ c = 9 ∧ d = 7 ∧ e = 8 :=
by sorry

end NUMINAMATH_CALUDE_digit_multiplication_puzzle_l3811_381107


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l3811_381144

/-- The weight of one liter of vegetable ghee for brand 'b' in grams -/
def weight_b : ℝ := 850

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def mixture_ratio : ℚ := 3/2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The weight of one liter of vegetable ghee for brand 'a' in grams -/
def weight_a : ℝ := 950

theorem vegetable_ghee_weight : 
  (weight_a * (mixture_ratio / (mixture_ratio + 1)) * total_volume) + 
  (weight_b * (1 / (mixture_ratio + 1)) * total_volume) = total_weight :=
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l3811_381144


namespace NUMINAMATH_CALUDE_N_swaps_rows_l3811_381190

/-- The matrix that swaps rows of a 2x2 matrix -/
def N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

/-- Theorem: N swaps the rows of any 2x2 matrix -/
theorem N_swaps_rows (a b c d : ℝ) :
  N • !![a, b; c, d] = !![c, d; a, b] := by
  sorry

end NUMINAMATH_CALUDE_N_swaps_rows_l3811_381190


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3811_381137

theorem fraction_sum_equality : (1 : ℚ) / 5 * 3 / 7 + 1 / 2 = 41 / 70 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3811_381137


namespace NUMINAMATH_CALUDE_twelve_numbers_divisible_by_three_l3811_381101

theorem twelve_numbers_divisible_by_three (n : ℕ) : 
  (n ≥ 10) ∧ 
  (∃ (seq : List ℕ), seq.length = 12 ∧ 
    (∀ x ∈ seq, x ≥ 10 ∧ x ≤ n ∧ x % 3 = 0) ∧
    (∀ y, y ≥ 10 ∧ y ≤ n ∧ y % 3 = 0 → y ∈ seq)) →
  n = 45 :=
by sorry

end NUMINAMATH_CALUDE_twelve_numbers_divisible_by_three_l3811_381101


namespace NUMINAMATH_CALUDE_remainder_13_pow_21_mod_1000_l3811_381175

theorem remainder_13_pow_21_mod_1000 : 13^21 % 1000 = 413 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_21_mod_1000_l3811_381175


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l3811_381187

theorem unique_solution_of_equation :
  ∃! x : ℝ, (x^16 + 1) * (x^12 + x^8 + x^4 + 1) = 18 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l3811_381187


namespace NUMINAMATH_CALUDE_prime_divides_power_difference_l3811_381183

theorem prime_divides_power_difference (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  p ∣ (3^(n+p) - 3^(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_power_difference_l3811_381183


namespace NUMINAMATH_CALUDE_exist_positive_integers_satisfying_equation_l3811_381160

theorem exist_positive_integers_satisfying_equation : 
  ∃ (x y z : ℕ+), x^2006 + y^2006 = z^2007 := by
  sorry

end NUMINAMATH_CALUDE_exist_positive_integers_satisfying_equation_l3811_381160


namespace NUMINAMATH_CALUDE_tetrahedron_edges_lengths_l3811_381165

-- Define the tetrahedron and its circumscribed sphere
structure Tetrahedron :=
  (base_edge1 : ℝ)
  (base_edge2 : ℝ)
  (base_edge3 : ℝ)
  (inclined_edge : ℝ)
  (sphere_radius : ℝ)
  (volume : ℝ)

-- Define the conditions
def tetrahedron_conditions (t : Tetrahedron) : Prop :=
  t.base_edge1 = 2 * t.sphere_radius ∧
  t.base_edge2 / t.base_edge3 = 4 / 3 ∧
  t.volume = 40 ∧
  t.base_edge1^2 = t.base_edge2^2 + t.base_edge3^2 ∧
  t.inclined_edge^2 = t.sphere_radius^2 + (t.base_edge2 / 2)^2

-- Theorem statement
theorem tetrahedron_edges_lengths 
  (t : Tetrahedron) 
  (h : tetrahedron_conditions t) : 
  t.base_edge1 = 10 ∧ 
  t.base_edge2 = 8 ∧ 
  t.base_edge3 = 6 ∧ 
  t.inclined_edge = Real.sqrt 50 := 
sorry

end NUMINAMATH_CALUDE_tetrahedron_edges_lengths_l3811_381165


namespace NUMINAMATH_CALUDE_root_in_interval_l3811_381119

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- State the theorem
theorem root_in_interval :
  Continuous f →
  f 2 < 0 →
  f 2.5 > 0 →
  ∃ x ∈ Set.Ioo 2 2.5, f x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l3811_381119


namespace NUMINAMATH_CALUDE_horner_method_v3_l3811_381129

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 - 3x^3 + 2x^2 + 5x + 1 -/
def f : List ℝ := [4, 0, -3, 2, 5, 1]

theorem horner_method_v3 :
  let v3 := (horner (f.take 4) 3)
  v3 = 101 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3811_381129


namespace NUMINAMATH_CALUDE_negation_equivalence_l3811_381130

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3811_381130


namespace NUMINAMATH_CALUDE_isosceles_triangle_34_perimeter_l3811_381154

/-- An isosceles triangle with sides 3 and 4 -/
structure IsoscelesTriangle34 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 = 3) ∨ (side1 = side3 ∧ side2 = 3)
  has_side_4 : side1 = 4 ∨ side2 = 4 ∨ side3 = 4

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle34) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of an isosceles triangle with sides 3 and 4 is either 10 or 11 -/
theorem isosceles_triangle_34_perimeter (t : IsoscelesTriangle34) : 
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_34_perimeter_l3811_381154


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3811_381181

/-- Given a function f: ℝ → ℝ satisfying the specified condition,
    proves that the tangent line to y = f(x) at (1, f(1)) has the equation x - y - 2 = 0 -/
theorem tangent_line_at_one (f : ℝ → ℝ) 
    (h : ∀ x, f (1 + x) = 2 * f (1 - x) - x^2 + 3*x + 1) : 
    ∃ m b, (∀ x, m * (x - 1) + f 1 = m * x + b) ∧ m = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3811_381181


namespace NUMINAMATH_CALUDE_total_miles_on_wednesdays_l3811_381135

/-- The total miles flown on Wednesdays over a 4-week period, given that a pilot flies
    the same number of miles each week and x miles each Wednesday. -/
theorem total_miles_on_wednesdays
  (x : ℕ)  -- Miles flown on Wednesday
  (h1 : ∀ week : Fin 4, ∃ miles : ℕ, miles = x)  -- Same miles flown each Wednesday for 4 weeks
  : ∃ total : ℕ, total = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_total_miles_on_wednesdays_l3811_381135


namespace NUMINAMATH_CALUDE_sine_of_angle_plus_three_half_pi_l3811_381127

theorem sine_of_angle_plus_three_half_pi (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = -12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin (3 * Real.pi / 2 + α) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_plus_three_half_pi_l3811_381127


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l3811_381163

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def circleP : Circle := { center := (0, 1), radius := 1 }
def circleQ : Circle := { center := (3, 2), radius := 2 }
def circleR : Circle := { center := (4, 3), radius := 3 }

-- Define the line l (implicitly defined by the tangent points)

-- Define the theorem
theorem area_of_triangle_PQR :
  let P := circleP.center
  let Q := circleQ.center
  let R := circleR.center
  let area := abs ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2)
  area = Real.sqrt 6 - Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_area_of_triangle_PQR_l3811_381163


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3811_381146

theorem triangle_equilateral (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2)
  (h5 : b^4 = c^4 + a^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3811_381146


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l3811_381120

def A : Set ℝ := {x | x ≤ 2}
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem set_intersection_theorem :
  (Set.univ \ A) ∩ B = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l3811_381120


namespace NUMINAMATH_CALUDE_black_car_speed_l3811_381176

/-- Proves that given the conditions of the car problem, the black car's speed is 50 mph -/
theorem black_car_speed (red_speed : ℝ) (initial_gap : ℝ) (overtake_time : ℝ) : ℝ :=
  let black_speed := (initial_gap + red_speed * overtake_time) / overtake_time
  by
    sorry

#check black_car_speed 40 30 3 = 50

end NUMINAMATH_CALUDE_black_car_speed_l3811_381176


namespace NUMINAMATH_CALUDE_base_b_sum_product_l3811_381105

/-- Given a base b, this function converts a number from base b to decimal --/
def toDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a decimal number to base b --/
def fromDecimal (b : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the given product and sum in base b --/
theorem base_b_sum_product (b : ℕ) : 
  (toDecimal b 14) * (toDecimal b 17) * (toDecimal b 18) = toDecimal b 6274 →
  (toDecimal b 14) + (toDecimal b 17) + (toDecimal b 18) = 49 := by
  sorry

end NUMINAMATH_CALUDE_base_b_sum_product_l3811_381105


namespace NUMINAMATH_CALUDE_divisibility_by_two_iff_last_digit_even_l3811_381128

theorem divisibility_by_two_iff_last_digit_even (a : ℕ) : 
  ∃ b c : ℕ, a = 10 * b + c ∧ c < 10 → (∃ k : ℕ, a = 2 * k ↔ ∃ m : ℕ, c = 2 * m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_two_iff_last_digit_even_l3811_381128


namespace NUMINAMATH_CALUDE_reciprocal_of_i_l3811_381132

theorem reciprocal_of_i : Complex.I⁻¹ = -Complex.I := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_i_l3811_381132


namespace NUMINAMATH_CALUDE_a_2000_mod_9_l3811_381189

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1995
  | n + 1 => (n + 1) * a n + 1

-- State the theorem
theorem a_2000_mod_9 : a 2000 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_2000_mod_9_l3811_381189


namespace NUMINAMATH_CALUDE_correct_calculation_l3811_381152

theorem correct_calculation (x : ℝ) : 2 * (3 * x + 14) = 946 → 2 * (x / 3 + 14) = 130 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3811_381152


namespace NUMINAMATH_CALUDE_jenny_cat_expenses_l3811_381139

theorem jenny_cat_expenses (adoption_fee : ℝ) (vet_visits : ℝ) (monthly_food_cost : ℝ) (toy_cost : ℝ) :
  adoption_fee = 50 →
  vet_visits = 500 →
  monthly_food_cost = 25 →
  toy_cost = 200 →
  (adoption_fee + vet_visits + 12 * monthly_food_cost) / 2 + toy_cost = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expenses_l3811_381139


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3811_381117

/-- Represents the number of people in each stratum -/
structure Strata :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Represents the sample sizes for each stratum -/
structure Sample :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)
  (remaining : ℕ)

/-- Calculates the total population size -/
def totalPopulation (s : Strata) : ℕ :=
  s.senior + s.intermediate + s.junior + s.remaining

/-- Checks if the sample sizes are proportional to the strata sizes -/
def isProportionalSample (strata : Strata) (sample : Sample) (totalSampleSize : ℕ) : Prop :=
  let total := totalPopulation strata
  sample.senior * total = strata.senior * totalSampleSize ∧
  sample.intermediate * total = strata.intermediate * totalSampleSize ∧
  sample.junior * total = strata.junior * totalSampleSize ∧
  sample.remaining * total = strata.remaining * totalSampleSize

/-- Theorem: The given sample sizes are proportional for the given strata -/
theorem correct_stratified_sample :
  let strata : Strata := ⟨160, 320, 200, 120⟩
  let sample : Sample := ⟨8, 16, 10, 6⟩
  let totalSampleSize : ℕ := 40
  totalPopulation strata = 800 →
  isProportionalSample strata sample totalSampleSize :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l3811_381117


namespace NUMINAMATH_CALUDE_inequality_theorem_l3811_381158

theorem inequality_theorem (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3811_381158


namespace NUMINAMATH_CALUDE_base4_calculation_l3811_381162

/-- Converts a base-4 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base-10 number to base-4 --/
def toBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_calculation :
  let a := toBase10 [1, 3, 2]  -- 231₄
  let b := toBase10 [1, 2]     -- 21₄
  let c := toBase10 [2, 3]     -- 32₄
  let d := toBase10 [2]        -- 2₄
  toBase4 (a * b + c / d) = [0, 3, 1, 6] := by
  sorry

end NUMINAMATH_CALUDE_base4_calculation_l3811_381162


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_mod_18_l3811_381121

def consecutive_odd_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + 2 * i)

theorem sum_consecutive_odd_integers_mod_18 (start : ℕ) (h : start = 11065) :
  (consecutive_odd_integers start 9).sum % 18 =
  ([1, 3, 5, 7, 9, 11, 13, 15, 17].map (λ x => x % 18)).sum % 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_mod_18_l3811_381121


namespace NUMINAMATH_CALUDE_susan_works_four_days_per_week_l3811_381156

/-- Represents Susan's work schedule and vacation details -/
structure WorkSchedule where
  hourlyRate : ℚ
  hoursPerDay : ℕ
  vacationDays : ℕ
  paidVacationDays : ℕ
  missedPay : ℚ

/-- Calculates the number of days Susan works per week -/
def daysWorkedPerWeek (schedule : WorkSchedule) : ℚ :=
  let totalVacationDays := 2 * 7
  let unpaidVacationDays := totalVacationDays - schedule.paidVacationDays
  unpaidVacationDays / 2

/-- Theorem stating that Susan works 4 days a week -/
theorem susan_works_four_days_per_week (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate = 15)
  (h2 : schedule.hoursPerDay = 8)
  (h3 : schedule.vacationDays = 14)
  (h4 : schedule.paidVacationDays = 6)
  (h5 : schedule.missedPay = 480) :
  daysWorkedPerWeek schedule = 4 := by
  sorry


end NUMINAMATH_CALUDE_susan_works_four_days_per_week_l3811_381156


namespace NUMINAMATH_CALUDE_least_n_with_k_ge_10_M_mod_500_l3811_381155

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation of h(n) -/
def j (n : ℕ) : ℕ := sorry

/-- Sum of squares of digits in base 12 representation of j(n) -/
def k (n : ℕ) : ℕ := sorry

/-- The least value of n such that k(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_k_ge_10 : M = 31 := by sorry

theorem M_mod_500 : M % 500 = 31 := by sorry

end NUMINAMATH_CALUDE_least_n_with_k_ge_10_M_mod_500_l3811_381155


namespace NUMINAMATH_CALUDE_solve_equation_l3811_381108

theorem solve_equation : 48 / (7 - 3/4) = 192/25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3811_381108


namespace NUMINAMATH_CALUDE_eccentricity_of_ellipse_through_roots_l3811_381192

-- Define the complex equation
def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + 3*z + 5) * (z^2 + 5*z + 8) = 0

-- Define the set of roots
def roots : Set ℂ :=
  {z : ℂ | complex_equation z}

-- Define the ellipse passing through the roots
def ellipse_through_roots (E : Set (ℝ × ℝ)) : Prop :=
  ∀ z ∈ roots, (z.re, z.im) ∈ E

-- Define the eccentricity of an ellipse
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem eccentricity_of_ellipse_through_roots :
  ∃ E : Set (ℝ × ℝ), ellipse_through_roots E ∧ eccentricity E = Real.sqrt (1/7) :=
sorry

end NUMINAMATH_CALUDE_eccentricity_of_ellipse_through_roots_l3811_381192


namespace NUMINAMATH_CALUDE_betty_cupcake_rate_l3811_381149

theorem betty_cupcake_rate : 
  ∀ (B : ℕ), -- Betty's cupcake rate per hour
  (5 * 8 - 3 * B = 10) → -- Difference in cupcakes after 5 hours
  B = 10 := by
sorry

end NUMINAMATH_CALUDE_betty_cupcake_rate_l3811_381149


namespace NUMINAMATH_CALUDE_quadratic_properties_l3811_381148

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 10

-- State the theorem
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 0 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3811_381148


namespace NUMINAMATH_CALUDE_expression_simplification_l3811_381125

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = 2 - Real.sqrt 2) : 
  (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3811_381125


namespace NUMINAMATH_CALUDE_strawberry_jam_money_l3811_381186

-- Define the given conditions
def betty_strawberries : ℕ := 16
def matthew_strawberries : ℕ := betty_strawberries + 20
def natalie_strawberries : ℕ := matthew_strawberries / 2
def strawberries_per_jar : ℕ := 7
def price_per_jar : ℕ := 4

-- Define the theorem
theorem strawberry_jam_money : 
  (betty_strawberries + matthew_strawberries + natalie_strawberries) / strawberries_per_jar * price_per_jar = 40 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_jam_money_l3811_381186


namespace NUMINAMATH_CALUDE_travel_group_average_age_l3811_381100

theorem travel_group_average_age 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_age_men : ℚ) 
  (avg_age_women : ℚ) 
  (h1 : num_men = 6) 
  (h2 : num_women = 9) 
  (h3 : avg_age_men = 57) 
  (h4 : avg_age_women = 52) :
  (num_men * avg_age_men + num_women * avg_age_women) / (num_men + num_women) = 54 := by
sorry

end NUMINAMATH_CALUDE_travel_group_average_age_l3811_381100


namespace NUMINAMATH_CALUDE_percentage_sum_equality_l3811_381118

theorem percentage_sum_equality : 
  (25 / 100 * 2018) + (2018 / 100 * 25) = 1009 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_equality_l3811_381118


namespace NUMINAMATH_CALUDE_house_selling_price_l3811_381182

def commission_rate : ℝ := 0.06
def commission_amount : ℝ := 8880

theorem house_selling_price :
  ∃ (selling_price : ℝ),
    selling_price * commission_rate = commission_amount ∧
    selling_price = 148000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l3811_381182


namespace NUMINAMATH_CALUDE_present_age_of_b_l3811_381191

theorem present_age_of_b (a b : ℕ) : 
  (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
  (a = b + 5) →              -- A is now 5 years older than B
  b = 95 :=                  -- The present age of B is 95
by sorry

end NUMINAMATH_CALUDE_present_age_of_b_l3811_381191


namespace NUMINAMATH_CALUDE_max_N_is_six_l3811_381141

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+1) + 32

/-- Definition of N(k) -/
def N (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of N(k) is 6 -/
theorem max_N_is_six :
  (∀ k : ℕ, N k ≤ 6) ∧ (∃ k : ℕ, N k = 6) := by sorry

end NUMINAMATH_CALUDE_max_N_is_six_l3811_381141


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3811_381138

def DigitSet : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

theorem digit_sum_puzzle (a b c x z : Nat) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ a ≠ z ∧
                b ≠ c ∧ b ≠ x ∧ b ≠ z ∧
                c ≠ x ∧ c ≠ z ∧
                x ≠ z)
  (h_in_set : a ∈ DigitSet ∧ b ∈ DigitSet ∧ c ∈ DigitSet ∧ x ∈ DigitSet ∧ z ∈ DigitSet)
  (h_vertical_sum : a + b + c = 17)
  (h_horizontal_sum : x + b + z = 14) :
  a + b + c + x + z = 26 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3811_381138


namespace NUMINAMATH_CALUDE_angle_C_value_angle_C_range_l3811_381123

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides

-- Define the function f
def f (x : Real) : Real := a^2 * x^2 - (a^2 - b^2) * x - 4 * c^2

-- Theorem 1
theorem angle_C_value (h1 : f 1 = 0) (h2 : B - C = π/3) : C = π/6 := by
  sorry

-- Theorem 2
theorem angle_C_range (h : f 2 = 0) : 0 < C ∧ C ≤ π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_angle_C_value_angle_C_range_l3811_381123


namespace NUMINAMATH_CALUDE_max_salary_is_400000_l3811_381126

/-- Represents a baseball team -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem: The maximum salary for a single player in the given conditions is $400,000 -/
theorem max_salary_is_400000 (team : BaseballTeam)
  (h1 : team.players = 21)
  (h2 : team.minSalary = 15000)
  (h3 : team.maxTotalSalary = 700000) :
  maxSinglePlayerSalary team = 400000 := by
  sorry

#eval maxSinglePlayerSalary { players := 21, minSalary := 15000, maxTotalSalary := 700000 }

end NUMINAMATH_CALUDE_max_salary_is_400000_l3811_381126


namespace NUMINAMATH_CALUDE_construction_materials_l3811_381151

theorem construction_materials (concrete stone total : Real) 
  (h1 : concrete = 0.17)
  (h2 : stone = 0.5)
  (h3 : total = 0.83) :
  total - (concrete + stone) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_l3811_381151


namespace NUMINAMATH_CALUDE_complement_of_union_theorem_l3811_381136

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_theorem :
  (U \ (A ∪ B)) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_theorem_l3811_381136


namespace NUMINAMATH_CALUDE_sector_area_l3811_381124

/-- Given a sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : Real) (l : Real) (A : Real) : 
  θ = 2 → l = 4 → A = (1/2) * (l/θ)^2 * θ → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3811_381124


namespace NUMINAMATH_CALUDE_chessboard_numbering_exists_l3811_381196

theorem chessboard_numbering_exists : 
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 → f i j ∈ Finset.range 64) ∧ 
    (∀ i j, i ∈ Finset.range 7 ∧ j ∈ Finset.range 7 → 
      (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 4 = 0) ∧
    (∀ n, n ∈ Finset.range 64 → ∃ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 ∧ f i j = n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_chessboard_numbering_exists_l3811_381196


namespace NUMINAMATH_CALUDE_total_spent_is_450_l3811_381143

/-- The total amount spent by Leonard and Michael on gifts for their father -/
def total_spent (leonard_wallet : ℕ) (leonard_sneakers : ℕ) (leonard_sneakers_count : ℕ)
                (michael_backpack : ℕ) (michael_jeans : ℕ) (michael_jeans_count : ℕ) : ℕ :=
  leonard_wallet + leonard_sneakers * leonard_sneakers_count +
  michael_backpack + michael_jeans * michael_jeans_count

/-- Theorem stating that the total amount spent by Leonard and Michael is $450 -/
theorem total_spent_is_450 :
  total_spent 50 100 2 100 50 2 = 450 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_is_450_l3811_381143


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3811_381116

/-- Represents the probability of selecting non-defective pens from a box -/
def probability_non_defective (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective.choose selected_pens : ℚ) / (total_pens.choose selected_pens)

/-- Theorem stating the probability of selecting 2 non-defective pens from a box of 16 pens with 3 defective pens -/
theorem probability_two_non_defective_pens :
  probability_non_defective 16 3 2 = 13/20 := by
  sorry

#eval probability_non_defective 16 3 2

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l3811_381116


namespace NUMINAMATH_CALUDE_estimate_white_balls_l3811_381166

/-- Represents the contents of the box -/
structure Box where
  black : ℕ
  white : ℕ

/-- Represents the result of the drawing experiment -/
structure DrawResult where
  total : ℕ
  black : ℕ

/-- Calculates the expected number of white balls given the box contents and draw results -/
def expectedWhiteBalls (box : Box) (result : DrawResult) : ℚ :=
  (box.black : ℚ) * (result.total - result.black : ℚ) / result.black

/-- The main theorem statement -/
theorem estimate_white_balls (box : Box) (result : DrawResult) :
  box.black = 4 ∧ result.total = 40 ∧ result.black = 10 →
  expectedWhiteBalls box result = 12 := by
  sorry

end NUMINAMATH_CALUDE_estimate_white_balls_l3811_381166


namespace NUMINAMATH_CALUDE_two_digit_number_with_remainders_l3811_381194

theorem two_digit_number_with_remainders : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 9 = 7 ∧ 
  n % 7 = 5 ∧ 
  n % 3 = 1 ∧ 
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_with_remainders_l3811_381194


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3811_381133

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (y = 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3811_381133


namespace NUMINAMATH_CALUDE_teena_speed_is_55_l3811_381198

-- Define the given conditions
def yoe_speed : ℝ := 40
def initial_distance : ℝ := 7.5
def final_relative_distance : ℝ := 15
def time : ℝ := 1.5  -- 90 minutes in hours

-- Define Teena's speed as a variable
def teena_speed : ℝ := 55

-- Theorem statement
theorem teena_speed_is_55 :
  yoe_speed * time + initial_distance + final_relative_distance = teena_speed * time :=
by sorry

end NUMINAMATH_CALUDE_teena_speed_is_55_l3811_381198


namespace NUMINAMATH_CALUDE_tan_5040_degrees_equals_zero_l3811_381106

theorem tan_5040_degrees_equals_zero : Real.tan (5040 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_5040_degrees_equals_zero_l3811_381106


namespace NUMINAMATH_CALUDE_yvonne_probability_l3811_381102

theorem yvonne_probability (xavier_prob zelda_prob joint_prob : ℝ) :
  xavier_prob = 1/4 →
  zelda_prob = 5/8 →
  joint_prob = 0.0625 →
  ∃ yvonne_prob : ℝ,
    yvonne_prob = 1/16 ∧
    xavier_prob * yvonne_prob * (1 - zelda_prob) = joint_prob :=
by sorry

end NUMINAMATH_CALUDE_yvonne_probability_l3811_381102


namespace NUMINAMATH_CALUDE_student_arrangements_l3811_381170

theorem student_arrangements (n : ℕ) (h : n = 5) : 
  (n - 1) * Nat.factorial (n - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l3811_381170


namespace NUMINAMATH_CALUDE_vacation_pictures_deleted_l3811_381173

theorem vacation_pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (remaining_pics : ℕ) : 
  zoo_pics = 41 → museum_pics = 29 → remaining_pics = 55 → 
  zoo_pics + museum_pics - remaining_pics = 15 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_deleted_l3811_381173
