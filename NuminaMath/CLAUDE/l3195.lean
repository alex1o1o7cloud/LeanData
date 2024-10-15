import Mathlib

namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3195_319502

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (z : ℂ), z = m^2 - 1 + (m + 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3195_319502


namespace NUMINAMATH_CALUDE_square_sequence_20th_figure_l3195_319529

theorem square_sequence_20th_figure :
  let square_count : ℕ → ℕ := λ n => 2 * n^2 - 2 * n + 1
  (square_count 1 = 1) ∧
  (square_count 2 = 5) ∧
  (square_count 3 = 13) ∧
  (square_count 4 = 25) →
  square_count 20 = 761 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_20th_figure_l3195_319529


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3195_319513

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  s : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, s n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with given conditions, S_6 = 6 -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 6)
    (h2 : seq.a 3 + seq.a 5 = 0) : 
  seq.s 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3195_319513


namespace NUMINAMATH_CALUDE_exists_set_with_square_diff_divides_product_l3195_319563

theorem exists_set_with_square_diff_divides_product (n : ℕ+) :
  ∃ (S : Finset ℕ+), 
    S.card = n ∧ 
    ∀ (a b : ℕ+), a ∈ S → b ∈ S → (a - b)^2 ∣ (a * b) :=
sorry

end NUMINAMATH_CALUDE_exists_set_with_square_diff_divides_product_l3195_319563


namespace NUMINAMATH_CALUDE_cottage_rental_cost_l3195_319587

theorem cottage_rental_cost (cost_per_hour : ℝ) (rental_hours : ℝ) (num_friends : ℕ) :
  cost_per_hour = 5 →
  rental_hours = 8 →
  num_friends = 2 →
  (cost_per_hour * rental_hours) / num_friends = 20 := by
  sorry

end NUMINAMATH_CALUDE_cottage_rental_cost_l3195_319587


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3195_319512

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3195_319512


namespace NUMINAMATH_CALUDE_initial_money_theorem_l3195_319527

def meat_cost : ℝ := 17
def chicken_cost : ℝ := 22
def veggie_cost : ℝ := 43
def egg_cost : ℝ := 5
def dog_food_cost : ℝ := 45
def cat_food_cost : ℝ := 18
def discount_rate : ℝ := 0.1
def money_left : ℝ := 35

def total_spent : ℝ := meat_cost + chicken_cost + veggie_cost + egg_cost + dog_food_cost + (cat_food_cost * (1 - discount_rate))

theorem initial_money_theorem :
  total_spent + money_left = 183.20 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_theorem_l3195_319527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3195_319576

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_product (seq : ArithmeticSequence) 
  (h1 : seq.nthTerm 8 = 20)
  (h2 : seq.d = 2) :
  seq.nthTerm 2 * seq.nthTerm 3 = 80 := by
  sorry

#check arithmetic_sequence_product

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3195_319576


namespace NUMINAMATH_CALUDE_original_number_is_five_l3195_319598

theorem original_number_is_five : 
  ∃ x : ℚ, 3 * (2 * x + 9) = 57 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_five_l3195_319598


namespace NUMINAMATH_CALUDE_integer_solutions_equation_l3195_319514

theorem integer_solutions_equation :
  ∀ (a b c : ℤ),
    c ≤ 94 →
    (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c →
    ((a = 3 ∧ b = 7 ∧ c = 41) ∨
     (a = 4 ∧ b = 6 ∧ c = 44) ∨
     (a = 5 ∧ b = 5 ∧ c = 45) ∨
     (a = 6 ∧ b = 4 ∧ c = 44) ∨
     (a = 7 ∧ b = 3 ∧ c = 41)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_equation_l3195_319514


namespace NUMINAMATH_CALUDE_probability_red_or_white_marble_l3195_319507

theorem probability_red_or_white_marble (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h1 : total = 60) 
  (h2 : blue = 5) 
  (h3 : red = 9) :
  (red : ℚ) / total + ((total - blue - red) : ℚ) / total = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_marble_l3195_319507


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l3195_319567

/-- Represents the number of residents in different age groups and the sampling size for one group -/
structure CommunityData where
  residents_35_to_45 : ℕ
  residents_46_to_55 : ℕ
  residents_56_to_65 : ℕ
  sampled_46_to_55 : ℕ

/-- Calculates the total number of people selected in a stratified sampling survey -/
def totalSampled (data : CommunityData) : ℕ :=
  (data.residents_35_to_45 + data.residents_46_to_55 + data.residents_56_to_65) / 
  (data.residents_46_to_55 / data.sampled_46_to_55)

/-- Theorem: Given the community data, the total number of people selected in the sampling survey is 140 -/
theorem stratified_sampling_result (data : CommunityData) 
  (h1 : data.residents_35_to_45 = 450)
  (h2 : data.residents_46_to_55 = 750)
  (h3 : data.residents_56_to_65 = 900)
  (h4 : data.sampled_46_to_55 = 50) :
  totalSampled data = 140 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l3195_319567


namespace NUMINAMATH_CALUDE_marble_probability_l3195_319547

/-- Given a bag of marbles with 5 red, 4 blue, and 6 yellow marbles,
    the probability of drawing one marble that is either red or blue is 3/5. -/
theorem marble_probability : 
  let red : ℕ := 5
  let blue : ℕ := 4
  let yellow : ℕ := 6
  let total : ℕ := red + blue + yellow
  let target : ℕ := red + blue
  (target : ℚ) / total = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l3195_319547


namespace NUMINAMATH_CALUDE_certain_number_is_sixty_l3195_319516

theorem certain_number_is_sixty : 
  ∃ x : ℝ, (10 + 20 + x) / 3 = (10 + 40 + 25) / 3 + 5 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_sixty_l3195_319516


namespace NUMINAMATH_CALUDE_zou_win_probability_l3195_319580

/-- Represents the outcome of a race -/
inductive RaceOutcome
| Win
| Loss

/-- Calculates the probability of winning a race given the previous outcome -/
def winProbability (previousOutcome : RaceOutcome) : ℚ :=
  match previousOutcome with
  | RaceOutcome.Win => 2/3
  | RaceOutcome.Loss => 1/3

/-- Represents a sequence of race outcomes -/
def RaceSequence := List RaceOutcome

/-- Calculates the probability of a given race sequence -/
def sequenceProbability (sequence : RaceSequence) : ℚ :=
  sequence.foldl (fun acc outcome => acc * winProbability outcome) 1

/-- Generates all possible race sequences where Zou wins exactly 5 out of 6 races -/
def winningSequences : List RaceSequence := sorry

theorem zou_win_probability :
  let totalProbability := (winningSequences.map sequenceProbability).sum
  totalProbability = 80/243 := by sorry

end NUMINAMATH_CALUDE_zou_win_probability_l3195_319580


namespace NUMINAMATH_CALUDE_triangle_side_length_l3195_319593

-- Define a triangle XYZ
structure Triangle :=
  (x y z : ℝ)
  (X Y Z : ℝ)

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (hy : t.y = 7)
  (hz : t.z = 6)
  (hcos : Real.cos (t.Y - t.Z) = 1/2) :
  t.x = Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3195_319593


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l3195_319559

/-- Calculates the cost of transmitting a number using Option 1 (decimal representation) -/
def option1Cost (n : Nat) : Nat :=
  sorry

/-- Calculates the cost of transmitting a number using Option 2 (binary representation) -/
def option2Cost (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def costsEqual (n : Nat) : Bool :=
  option1Cost n = option2Cost n

/-- Theorem stating that 1118 is the largest number less than 2000 with equal costs -/
theorem largest_equal_cost_number :
  (∀ n : Nat, n < 2000 → costsEqual n → n ≤ 1118) ∧ costsEqual 1118 := by
  sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l3195_319559


namespace NUMINAMATH_CALUDE_at_least_one_angle_not_greater_than_60_l3195_319526

-- Define a triangle as a triple of angles
def Triangle := (ℝ × ℝ × ℝ)

-- Define a predicate for a valid triangle (sum of angles is 180°)
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_not_greater_than_60 (t : Triangle) 
  (h : is_valid_triangle t) : 
  ∃ θ, θ ∈ [t.1, t.2.1, t.2.2] ∧ θ ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_angle_not_greater_than_60_l3195_319526


namespace NUMINAMATH_CALUDE_kates_savings_l3195_319533

/-- Kate's savings and purchases problem -/
theorem kates_savings (march april may june : ℕ) 
  (keyboard mouse headset video_game : ℕ) : 
  march = 27 → 
  april = 13 → 
  may = 28 → 
  june = 35 → 
  keyboard = 49 → 
  mouse = 5 → 
  headset = 15 → 
  video_game = 25 → 
  (march + april + may + june + 2 * april) - 
  (keyboard + mouse + headset + video_game) = 35 := by
  sorry

end NUMINAMATH_CALUDE_kates_savings_l3195_319533


namespace NUMINAMATH_CALUDE_missing_village_population_l3195_319574

def village_population_problem (total_villages : Nat) 
                               (average_population : Nat) 
                               (known_populations : List Nat) : Prop :=
  total_villages = 7 ∧
  average_population = 1000 ∧
  known_populations = [803, 900, 1023, 945, 980, 1249] ∧
  known_populations.length = 6 ∧
  (List.sum known_populations + 1100) / total_villages = average_population

theorem missing_village_population :
  ∀ (total_villages : Nat) (average_population : Nat) (known_populations : List Nat),
  village_population_problem total_villages average_population known_populations →
  1100 = total_villages * average_population - List.sum known_populations :=
by
  sorry

end NUMINAMATH_CALUDE_missing_village_population_l3195_319574


namespace NUMINAMATH_CALUDE_sixty_percent_high_profit_puppies_l3195_319515

/-- Represents a litter of puppies with their spot counts -/
structure PuppyLitter where
  total : Nat
  fiveSpots : Nat
  fourSpots : Nat
  twoSpots : Nat

/-- Calculates the percentage of puppies with more than 4 spots -/
def percentageHighProfitPuppies (litter : PuppyLitter) : Rat :=
  (litter.fiveSpots : Rat) / (litter.total : Rat) * 100

/-- The theorem stating that for the given litter, 60% of puppies can be sold for greater profit -/
theorem sixty_percent_high_profit_puppies (litter : PuppyLitter)
    (h1 : litter.total = 10)
    (h2 : litter.fiveSpots = 6)
    (h3 : litter.fourSpots = 3)
    (h4 : litter.twoSpots = 1)
    (h5 : litter.fiveSpots + litter.fourSpots + litter.twoSpots = litter.total) :
    percentageHighProfitPuppies litter = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_high_profit_puppies_l3195_319515


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l3195_319549

theorem quadratic_root_sum (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 5 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 5 = 0) → 
  (x₁ ≠ x₂) →
  (x₁ + x₂ = 3/2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l3195_319549


namespace NUMINAMATH_CALUDE_combined_share_proof_l3195_319520

/-- Proves that given $12,000 to be distributed among 5 children in the ratio 2 : 4 : 3 : 1 : 5,
    the combined share of the children with ratios 1 and 5 is $4,800. -/
theorem combined_share_proof (total_money : ℕ) (num_children : ℕ) (ratio : List ℕ) :
  total_money = 12000 →
  num_children = 5 →
  ratio = [2, 4, 3, 1, 5] →
  (ratio.sum * 800 = total_money) →
  (List.get! ratio 3 * 800 + List.get! ratio 4 * 800 = 4800) :=
by sorry

end NUMINAMATH_CALUDE_combined_share_proof_l3195_319520


namespace NUMINAMATH_CALUDE_number_properties_l3195_319511

def is_even (n : ℕ) := n % 2 = 0
def is_odd (n : ℕ) := n % 2 ≠ 0
def is_prime (n : ℕ) := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)
def is_composite (n : ℕ) := n > 1 ∧ ¬(is_prime n)

theorem number_properties :
  (∀ n : ℕ, n ≤ 10 → (is_even n ∧ ¬is_composite n) → n = 2) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ ¬is_prime n) → n = 1) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ is_composite n) → n = 9) ∧
  (∀ n : ℕ, is_prime n → n ≥ 2) ∧
  (∀ n : ℕ, is_composite n → n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l3195_319511


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3195_319521

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) : 
  1 / (a - 1) - a + 1 = (2*a - a^2) / (a - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 2) / (x^2 - 2*x) - 1 / (x - 2)) / (2 / x) = 1 / (x - 2) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3195_319521


namespace NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l3195_319572

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : f a = f b) : 
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_one_when_equal_absolute_log_l3195_319572


namespace NUMINAMATH_CALUDE_sector_central_angle_l3195_319570

theorem sector_central_angle (circumference area : ℝ) (h1 : circumference = 4) (h2 : area = 1) :
  let r := (4 - circumference) / 2
  let l := circumference - 2 * r
  l / r = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3195_319570


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l3195_319582

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) :
  ∃ k : Nat, (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) = 143 * k := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l3195_319582


namespace NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l3195_319595

theorem complex_square_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I) ^ 2 = Complex.I * y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l3195_319595


namespace NUMINAMATH_CALUDE_candy_cost_proof_l3195_319590

/-- Represents the cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- Represents the weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- Represents the cost per pound of the first type of candy -/
def first_candy_cost : ℝ := 8

/-- Represents the weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- Represents the cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_cost_proof :
  first_candy_weight * first_candy_cost + second_candy_weight * second_candy_cost =
  total_weight * mixture_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l3195_319590


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3195_319545

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3195_319545


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3195_319569

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3195_319569


namespace NUMINAMATH_CALUDE_arc_length_parametric_curve_l3195_319589

open Real MeasureTheory

/-- The arc length of the curve given by the parametric equations
    x = e^t(cos t + sin t) and y = e^t(cos t - sin t) for 0 ≤ t ≤ 2π -/
theorem arc_length_parametric_curve :
  let x : ℝ → ℝ := fun t ↦ exp t * (cos t + sin t)
  let y : ℝ → ℝ := fun t ↦ exp t * (cos t - sin t)
  let curve_length := ∫ t in Set.Icc 0 (2 * π), sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)
  curve_length = 2 * (exp (2 * π) - 1) := by
sorry

end NUMINAMATH_CALUDE_arc_length_parametric_curve_l3195_319589


namespace NUMINAMATH_CALUDE_second_agency_daily_charge_proof_l3195_319565

/-- The daily charge of the first agency in dollars -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency in dollars -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency in dollars -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25.0

/-- The daily charge of the second agency in dollars -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_proof :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_second_agency_daily_charge_proof_l3195_319565


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_29_l3195_319541

theorem largest_five_digit_congruent_to_17_mod_29 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 17 [MOD 29] → n ≤ 99982 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_29_l3195_319541


namespace NUMINAMATH_CALUDE_afternoon_campers_l3195_319539

theorem afternoon_campers (morning_campers : ℕ) (total_campers : ℕ) 
  (h1 : morning_campers = 15) 
  (h2 : total_campers = 32) : 
  total_campers - morning_campers = 17 := by
sorry

end NUMINAMATH_CALUDE_afternoon_campers_l3195_319539


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l3195_319568

/-- Given three numbers in ratio 5:7:9 with LCM 6300, their sum is 14700 -/
theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  (a : ℚ) / 5 = (b : ℚ) / 7 ∧ (b : ℚ) / 7 = (c : ℚ) / 9 →
  Nat.lcm a (Nat.lcm b c) = 6300 →
  a + b + c = 14700 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l3195_319568


namespace NUMINAMATH_CALUDE_watch_angle_difference_l3195_319518

/-- Represents the angle between the hour and minute hands of a watch -/
def watchAngle (hours minutes : ℝ) : ℝ :=
  |30 * hours - 5.5 * minutes|

/-- Theorem stating that the time difference between two 120° angles of watch hands between 7:00 PM and 8:00 PM is 30 minutes -/
theorem watch_angle_difference : ∃ (t₁ t₂ : ℝ),
  0 < t₁ ∧ t₁ < t₂ ∧ t₂ < 60 ∧
  watchAngle (7 + t₁ / 60) t₁ = 120 ∧
  watchAngle (7 + t₂ / 60) t₂ = 120 ∧
  t₂ - t₁ = 30 := by
  sorry

end NUMINAMATH_CALUDE_watch_angle_difference_l3195_319518


namespace NUMINAMATH_CALUDE_augmented_matrix_solution_sum_l3195_319584

/-- Given an augmented matrix representing a system of linear equations,
    if the solution exists, then the sum of certain elements in the matrix is determined. -/
theorem augmented_matrix_solution_sum (m n : ℝ) : 
  (∃ x y : ℝ, m * x = 6 ∧ 3 * y = n ∧ x = -3 ∧ y = 4) → m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_solution_sum_l3195_319584


namespace NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l3195_319531

/-- The function f(x) = 3 sin x + 4 cos x has a maximum value of 5 -/
theorem max_value_of_sin_cos_combination :
  ∃ (M : ℝ), M = 5 ∧ ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sin_cos_combination_l3195_319531


namespace NUMINAMATH_CALUDE_current_year_is_2021_l3195_319599

/-- The year Aziz's parents moved to America -/
def parents_move_year : ℕ := 1982

/-- Aziz's current age -/
def aziz_age : ℕ := 36

/-- Years Aziz's parents lived in America before his birth -/
def years_before_birth : ℕ := 3

/-- The current year -/
def current_year : ℕ := parents_move_year + aziz_age + years_before_birth

theorem current_year_is_2021 : current_year = 2021 := by
  sorry

end NUMINAMATH_CALUDE_current_year_is_2021_l3195_319599


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3195_319556

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3195_319556


namespace NUMINAMATH_CALUDE_marker_sale_savings_l3195_319532

/-- Calculates the savings when buying markers during a sale --/
def calculate_savings (original_price : ℚ) (num_markers : ℕ) (discount_rate : ℚ) : ℚ :=
  let original_total := original_price * num_markers
  let discounted_price := original_price * (1 - discount_rate)
  let free_markers := num_markers / 4
  let effective_markers := num_markers + free_markers
  let sale_total := discounted_price * num_markers
  original_total - sale_total

theorem marker_sale_savings :
  let original_price : ℚ := 3
  let num_markers : ℕ := 8
  let discount_rate : ℚ := 0.3
  calculate_savings original_price num_markers discount_rate = 36/5 := by sorry

end NUMINAMATH_CALUDE_marker_sale_savings_l3195_319532


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3195_319564

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^2 = k) →  -- a^3 varies inversely with b^2
  (4^3 * 2^2 = k) →         -- a = 4 when b = 2
  (a^3 * 8^2 = k) →         -- condition for b = 8
  a = 4^(1/3) :=            -- prove that a = 4^(1/3) when b = 8
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3195_319564


namespace NUMINAMATH_CALUDE_angle_measure_l3195_319523

theorem angle_measure (α : Real) (h : α > 0 ∧ α < π/2) :
  1 / Real.sqrt (Real.tan (α/2)) = Real.sqrt (2 * Real.sqrt 3) * Real.sqrt (Real.tan (π/18)) + Real.sqrt (Real.tan (α/2)) →
  α = π/3.6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3195_319523


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l3195_319546

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + bc + c², then the measure of angle A is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a^2 = b^2 + b*c + c^2 →
  A = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l3195_319546


namespace NUMINAMATH_CALUDE_lizette_overall_average_l3195_319577

def average_first_two_quizzes : ℝ := 95
def third_quiz_score : ℝ := 92
def number_of_quizzes : ℕ := 3

theorem lizette_overall_average :
  (average_first_two_quizzes * 2 + third_quiz_score) / number_of_quizzes = 94 := by
  sorry

end NUMINAMATH_CALUDE_lizette_overall_average_l3195_319577


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_phi_l3195_319551

theorem symmetry_axis_implies_phi (φ : ℝ) : 
  (∀ x, 2 * Real.sin (3 * x + φ) = 2 * Real.sin (3 * (π / 6 - x) + φ)) →
  |φ| < π / 2 →
  φ = π / 4 := by
sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_phi_l3195_319551


namespace NUMINAMATH_CALUDE_soap_brand_usage_l3195_319594

/-- The number of households using both brand R and brand B soap -/
def households_using_both : ℕ := 15

/-- The total number of households surveyed -/
def total_households : ℕ := 200

/-- The number of households using neither brand R nor brand B -/
def households_using_neither : ℕ := 80

/-- The number of households using only brand R -/
def households_using_only_R : ℕ := 60

/-- For every household using both brands, this many use only brand B -/
def ratio_B_to_both : ℕ := 3

theorem soap_brand_usage :
  households_using_both * (ratio_B_to_both + 1) + 
  households_using_neither + 
  households_using_only_R = 
  total_households := by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l3195_319594


namespace NUMINAMATH_CALUDE_clever_cat_academy_count_l3195_319579

theorem clever_cat_academy_count :
  let jump : ℕ := 60
  let spin : ℕ := 35
  let fetch : ℕ := 40
  let jump_and_spin : ℕ := 25
  let spin_and_fetch : ℕ := 20
  let jump_and_fetch : ℕ := 22
  let all_three : ℕ := 12
  let none : ℕ := 10
  jump + spin + fetch - jump_and_spin - spin_and_fetch - jump_and_fetch + all_three + none = 92 :=
by sorry

end NUMINAMATH_CALUDE_clever_cat_academy_count_l3195_319579


namespace NUMINAMATH_CALUDE_bicolored_angles_bound_l3195_319538

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A coloring of segments -/
def Coloring (n : ℕ) := Fin (n + 1) → Fin (n + 1) → Fin n

/-- The number of bicolored angles for a given coloring -/
def bicoloredAngles (n k : ℕ) (points : Fin (n + 1) → Point) (coloring : Coloring k) : ℕ :=
  sorry

/-- Three points are collinear if they lie on the same line -/
def collinear (p q r : Point) : Prop :=
  sorry

theorem bicolored_angles_bound (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 3) 
  (points : Fin (n + 1) → Point) 
  (h3 : ∀ (i j l : Fin (n + 1)), i ≠ j → j ≠ l → i ≠ l → ¬collinear (points i) (points j) (points l)) :
  ∃ (coloring : Coloring k), bicoloredAngles n k points coloring > n * (n / k)^2 * (k.choose 2) :=
sorry

end NUMINAMATH_CALUDE_bicolored_angles_bound_l3195_319538


namespace NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l3195_319542

theorem abs_five_implies_plus_minus_five (a : ℝ) : |a| = 5 → a = 5 ∨ a = -5 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l3195_319542


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l3195_319505

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x, f x = m) ∧ 
               (∀ x, n ≤ f x) ∧ (∃ x, f x = n) ∧
               (m + n = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l3195_319505


namespace NUMINAMATH_CALUDE_muffin_buyers_count_l3195_319560

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_buyers : ℕ := 50

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- Define the probability of selecting a buyer who purchases neither cake nor muffin mix
def prob_neither : ℚ := 29/100

-- Theorem to prove
theorem muffin_buyers_count : 
  ∃ (muffin_buyers : ℕ), 
    muffin_buyers = total_buyers - cake_buyers - (total_buyers * prob_neither).num + both_buyers := by
  sorry

end NUMINAMATH_CALUDE_muffin_buyers_count_l3195_319560


namespace NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l3195_319562

theorem no_solution_implies_b_bounded (a b : ℝ) :
  (∀ x : ℝ, ¬(a * Real.cos x + b * Real.cos (3 * x) > 1)) →
  |b| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_b_bounded_l3195_319562


namespace NUMINAMATH_CALUDE_cara_charge_account_l3195_319573

/-- Represents the simple interest calculation for Cara's charge account --/
theorem cara_charge_account (initial_charge : ℝ) : 
  initial_charge * (1 + 0.05) = 56.7 → initial_charge = 54 := by
  sorry

end NUMINAMATH_CALUDE_cara_charge_account_l3195_319573


namespace NUMINAMATH_CALUDE_phi_expression_l3195_319508

/-- A function that is directly proportional to x -/
def DirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- A function that is inversely proportional to x -/
def InverselyProportional (g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → g x = k / x

/-- The main theorem -/
theorem phi_expression (f g φ : ℝ → ℝ) 
    (h1 : DirectlyProportional f)
    (h2 : InverselyProportional g)
    (h3 : ∀ x : ℝ, φ x = f x + g x)
    (h4 : φ (1/3) = 16)
    (h5 : φ 1 = 8) :
    ∀ x : ℝ, x ≠ 0 → φ x = 3 * x + 5 / x := by
  sorry

end NUMINAMATH_CALUDE_phi_expression_l3195_319508


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3195_319588

/-- The perimeter of an equilateral triangle with side length 23 centimeters is 69 centimeters. -/
theorem equilateral_triangle_perimeter :
  ∀ (triangle : Set ℝ × Set ℝ),
    (∀ side : ℝ, side ∈ (triangle.1 ∪ triangle.2) → side = 23) →
    (∃ (a b c : ℝ), a ∈ triangle.1 ∧ b ∈ triangle.1 ∧ c ∈ triangle.2 ∧
      a + b + c = 69) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l3195_319588


namespace NUMINAMATH_CALUDE_p_and_q_true_l3195_319504

theorem p_and_q_true : 
  (∃ x₀ : ℝ, Real.tan x₀ = Real.sqrt 3) ∧ 
  (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l3195_319504


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3195_319550

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 - 6*x^2 + 9 : ℝ) = (x^2 - 3) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3195_319550


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3195_319535

theorem fraction_subtraction : 
  (((3 : ℚ) + 6 + 9) / ((2 : ℚ) + 5 + 8) - ((2 : ℚ) + 5 + 8) / ((3 : ℚ) + 6 + 9)) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3195_319535


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l3195_319525

theorem difference_of_squares_special_case : (532 * 532) - (531 * 533) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l3195_319525


namespace NUMINAMATH_CALUDE_jacket_final_price_l3195_319536

/-- The final price of a jacket after applying two successive discounts --/
theorem jacket_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 250 → 
  discount1 = 0.4 → 
  discount2 = 0.25 → 
  original_price * (1 - discount1) * (1 - discount2) = 112.5 := by
  sorry


end NUMINAMATH_CALUDE_jacket_final_price_l3195_319536


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3195_319530

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + y * f x

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) (h2 : f 1 = 3) : f 501 = 503 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3195_319530


namespace NUMINAMATH_CALUDE_equal_diagonal_quadrilateral_multiple_shapes_l3195_319585

/-- A quadrilateral with equal-length diagonals -/
structure EqualDiagonalQuadrilateral where
  /-- The length of the diagonals -/
  diagonal_length : ℝ
  /-- The quadrilateral has positive area -/
  positive_area : ℝ
  area_pos : positive_area > 0

/-- Possible shapes of a quadrilateral -/
inductive QuadrilateralShape
  | Square
  | Rectangle
  | IsoscelesTrapezoid
  | Other

/-- A function that determines if a given shape is possible for an equal-diagonal quadrilateral -/
def is_possible_shape (q : EqualDiagonalQuadrilateral) (shape : QuadrilateralShape) : Prop :=
  ∃ (quad : EqualDiagonalQuadrilateral), quad.diagonal_length = q.diagonal_length ∧ 
    quad.positive_area = q.positive_area ∧ 
    (match shape with
      | QuadrilateralShape.Square => true
      | QuadrilateralShape.Rectangle => true
      | QuadrilateralShape.IsoscelesTrapezoid => true
      | QuadrilateralShape.Other => true)

theorem equal_diagonal_quadrilateral_multiple_shapes (q : EqualDiagonalQuadrilateral) :
  (is_possible_shape q QuadrilateralShape.Square) ∧
  (is_possible_shape q QuadrilateralShape.Rectangle) ∧
  (is_possible_shape q QuadrilateralShape.IsoscelesTrapezoid) :=
sorry

end NUMINAMATH_CALUDE_equal_diagonal_quadrilateral_multiple_shapes_l3195_319585


namespace NUMINAMATH_CALUDE_davids_biology_marks_l3195_319566

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (average : ℕ) 
  (h1 : english = 96) 
  (h2 : mathematics = 95) 
  (h3 : physics = 82) 
  (h4 : chemistry = 97) 
  (h5 : average = 93) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 95 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l3195_319566


namespace NUMINAMATH_CALUDE_square_of_number_ending_in_five_l3195_319575

theorem square_of_number_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_number_ending_in_five_l3195_319575


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l3195_319592

/-- A function f is decreasing on ℝ if for all x y, x < y implies f x > f y -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingOn f) (h_inequality : f (3 * a) < f (-2 * a + 10)) : 
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l3195_319592


namespace NUMINAMATH_CALUDE_perfume_price_decrease_l3195_319561

theorem perfume_price_decrease (original_price increased_price final_price : ℝ) : 
  original_price = 1200 →
  increased_price = original_price * 1.1 →
  final_price = original_price - 78 →
  (increased_price - final_price) / increased_price = 0.15 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_decrease_l3195_319561


namespace NUMINAMATH_CALUDE_bea_highest_profit_l3195_319543

/-- Represents a lemonade seller with their sales information -/
structure LemonadeSeller where
  name : String
  price : ℕ
  soldGlasses : ℕ
  variableCost : ℕ

/-- Calculates the profit for a lemonade seller -/
def calculateProfit (seller : LemonadeSeller) : ℕ :=
  seller.price * seller.soldGlasses - seller.variableCost * seller.soldGlasses

/-- Theorem stating that Bea makes the most profit -/
theorem bea_highest_profit (bea dawn carla : LemonadeSeller)
  (h_bea : bea = { name := "Bea", price := 25, soldGlasses := 10, variableCost := 10 })
  (h_dawn : dawn = { name := "Dawn", price := 28, soldGlasses := 8, variableCost := 12 })
  (h_carla : carla = { name := "Carla", price := 35, soldGlasses := 6, variableCost := 15 }) :
  calculateProfit bea ≥ calculateProfit dawn ∧ calculateProfit bea ≥ calculateProfit carla :=
by sorry

end NUMINAMATH_CALUDE_bea_highest_profit_l3195_319543


namespace NUMINAMATH_CALUDE_right_triangle_area_l3195_319591

theorem right_triangle_area (h : ℝ) (h_pos : h > 0) : ∃ (a b : ℝ),
  a > 0 ∧ b > 0 ∧
  a / b = 3 / 4 ∧
  a^2 + b^2 = h^2 ∧
  (1/2) * a * b = (6/25) * h^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3195_319591


namespace NUMINAMATH_CALUDE_quadratic_constraint_l3195_319540

/-- Quadratic function defined by a parameter a -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

/-- Theorem stating the condition on a for the given constraints -/
theorem quadratic_constraint (a : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : a ≠ 0)
  (h₂ : x₁ + x₂ = 2)
  (h₃ : x₁ < x₂)
  (h₄ : y₁ > y₂)
  (h₅ : quadratic_function a x₁ = y₁)
  (h₆ : quadratic_function a x₂ = y₂) :
  a < -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_constraint_l3195_319540


namespace NUMINAMATH_CALUDE_max_concave_polygons_in_square_l3195_319509

/-- A concave polygon with sides parallel to a square's sides -/
structure ConcavePolygon where
  vertices : List (ℝ × ℝ)
  is_concave : Bool
  sides_parallel_to_square : Bool

/-- A square divided into concave polygons -/
structure DividedSquare where
  polygons : List ConcavePolygon
  no_parallel_translation : Bool

/-- The maximum number of equal concave polygons a square can be divided into -/
def max_concave_polygons : ℕ := 8

/-- Theorem stating the maximum number of equal concave polygons a square can be divided into -/
theorem max_concave_polygons_in_square :
  ∀ (d : DividedSquare),
    d.no_parallel_translation →
    (∀ p ∈ d.polygons, p.is_concave ∧ p.sides_parallel_to_square) →
    (List.length d.polygons ≤ max_concave_polygons) :=
by sorry

end NUMINAMATH_CALUDE_max_concave_polygons_in_square_l3195_319509


namespace NUMINAMATH_CALUDE_sunflower_height_l3195_319581

-- Define the height of Marissa's sister in inches
def sister_height_inches : ℕ := 4 * 12 + 3

-- Define the height difference between the sunflower and Marissa's sister
def height_difference : ℕ := 21

-- Theorem to prove the height of the sunflower
theorem sunflower_height :
  (sister_height_inches + height_difference) / 12 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_height_l3195_319581


namespace NUMINAMATH_CALUDE_remainder_after_adding_3006_l3195_319555

theorem remainder_after_adding_3006 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_3006_l3195_319555


namespace NUMINAMATH_CALUDE_company_n_profit_change_l3195_319544

def CompanyN (R : ℝ) : Prop :=
  let profit1998 := 0.10 * R
  let revenue1999 := 0.70 * R
  let profit1999 := 0.15 * revenue1999
  let revenue2000 := 1.20 * revenue1999
  let profit2000 := 0.18 * revenue2000
  let percentageChange := (profit2000 - profit1998) / profit1998 * 100
  percentageChange = 51.2

theorem company_n_profit_change (R : ℝ) (h : R > 0) : CompanyN R := by
  sorry

end NUMINAMATH_CALUDE_company_n_profit_change_l3195_319544


namespace NUMINAMATH_CALUDE_thursday_tea_consumption_l3195_319506

/-- Represents the relationship between hours grading and liters of tea consumed -/
structure TeaGrading where
  hours : ℝ
  liters : ℝ
  inv_prop : hours * liters = hours * liters

/-- The constant of proportionality derived from Wednesday's data -/
def wednesday_constant : ℝ := 5 * 4

/-- Theorem stating that given Wednesday's data and Thursday's hours, the teacher drinks 2.5 liters of tea on Thursday -/
theorem thursday_tea_consumption (wednesday : TeaGrading) (thursday : TeaGrading) 
    (h_wednesday : wednesday.hours = 5 ∧ wednesday.liters = 4)
    (h_thursday : thursday.hours = 8)
    (h_constant : wednesday.hours * wednesday.liters = thursday.hours * thursday.liters) :
    thursday.liters = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_thursday_tea_consumption_l3195_319506


namespace NUMINAMATH_CALUDE_complex_product_equality_complex_product_equality_proof_l3195_319503

theorem complex_product_equality : Complex → Prop :=
  fun i =>
    i * i = -1 →
    (1 + i) * (2 - i) = 3 + i

-- The proof is omitted
theorem complex_product_equality_proof : complex_product_equality Complex.I :=
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_complex_product_equality_proof_l3195_319503


namespace NUMINAMATH_CALUDE_transfer_equation_l3195_319554

theorem transfer_equation (x : ℤ) : 
  let initial_A : ℤ := 232
  let initial_B : ℤ := 146
  let final_A : ℤ := initial_A + x
  let final_B : ℤ := initial_B - x
  (final_A = 3 * final_B) ↔ (232 + x = 3 * (146 - x)) :=
by sorry

end NUMINAMATH_CALUDE_transfer_equation_l3195_319554


namespace NUMINAMATH_CALUDE_exists_odd_power_function_l3195_319586

/-- A function satisfying the given conditions -/
def special_function (f : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ, (m + n) ∣ (f m + f n))

/-- The main theorem -/
theorem exists_odd_power_function (f : ℕ → ℕ) (hf : special_function f) :
  ∃ k : ℕ, Odd k ∧ ∀ n : ℕ, f n = n^k :=
sorry

end NUMINAMATH_CALUDE_exists_odd_power_function_l3195_319586


namespace NUMINAMATH_CALUDE_rectangular_shingle_area_l3195_319501

/-- The area of a rectangular roof shingle -/
theorem rectangular_shingle_area (length width : ℝ) (h1 : length = 10) (h2 : width = 7) :
  length * width = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_shingle_area_l3195_319501


namespace NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l3195_319596

/-- The amount of money made per t-shirt in dollars -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := arkansas_shirts_sold * profit_per_shirt

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 :=
sorry

end NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l3195_319596


namespace NUMINAMATH_CALUDE_no_nonzero_solution_for_equation_l3195_319583

theorem no_nonzero_solution_for_equation (a b c d : ℤ) :
  a^2 + b^2 = 3*(c^2 + d^2) → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_for_equation_l3195_319583


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l3195_319510

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_award_ways (total_sprinters : ℕ) (canadian_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_canadian_sprinters := total_sprinters - canadian_sprinters
  let no_canadian_medal := non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)
  let one_canadian_medal := canadian_sprinters * medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1)
  no_canadian_medal + one_canadian_medal

/-- Theorem: The number of ways to award medals in the given scenario is 480 -/
theorem olympic_medal_awards : medal_award_ways 10 4 3 = 480 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l3195_319510


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3195_319524

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3195_319524


namespace NUMINAMATH_CALUDE_parallel_segment_length_l3195_319519

theorem parallel_segment_length (base : ℝ) (a b c : ℝ) :
  base = 18 →
  a + b + c = 1 →
  a = (1/4 : ℝ) →
  b = (1/2 : ℝ) →
  c = (1/4 : ℝ) →
  ∃ (middle_segment : ℝ), middle_segment = 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l3195_319519


namespace NUMINAMATH_CALUDE_mandy_coin_value_l3195_319548

/-- Represents the number of cents in a coin -/
inductive Coin
| Dime : Coin
| Quarter : Coin

def coin_value : Coin → Nat
| Coin.Dime => 10
| Coin.Quarter => 25

/-- Represents Mandy's coin collection -/
structure CoinCollection where
  dimes : Nat
  quarters : Nat
  total_coins : Nat
  coin_balance : dimes + quarters = total_coins
  dime_quarter_relation : dimes + 2 = quarters

def collection_value (c : CoinCollection) : Nat :=
  c.dimes * coin_value Coin.Dime + c.quarters * coin_value Coin.Quarter

theorem mandy_coin_value :
  ∃ c : CoinCollection, c.total_coins = 17 ∧ collection_value c = 320 := by
  sorry

end NUMINAMATH_CALUDE_mandy_coin_value_l3195_319548


namespace NUMINAMATH_CALUDE_problem_solution_l3195_319558

theorem problem_solution (a : ℝ) : a = 1 / (Real.sqrt 2 - 1) → 4 * a^2 - 8 * a + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3195_319558


namespace NUMINAMATH_CALUDE_theta_range_l3195_319597

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l3195_319597


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l3195_319552

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the region enclosed by the curve x^2 + y^2 = 2(|x| + |y|) is 2π -/
theorem area_of_enclosed_region : area enclosed_region = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l3195_319552


namespace NUMINAMATH_CALUDE_complex_square_sum_l3195_319557

theorem complex_square_sum (a b : ℝ) (h : (1 + Complex.I)^2 = Complex.mk a b) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l3195_319557


namespace NUMINAMATH_CALUDE_cos_negative_45_degrees_l3195_319571

theorem cos_negative_45_degrees : Real.cos (-(45 * π / 180)) = 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_45_degrees_l3195_319571


namespace NUMINAMATH_CALUDE_total_pages_in_book_l3195_319537

theorem total_pages_in_book (pages_read : ℕ) (pages_left : ℕ) : 
  pages_read = 147 → pages_left = 416 → pages_read + pages_left = 563 := by
sorry

end NUMINAMATH_CALUDE_total_pages_in_book_l3195_319537


namespace NUMINAMATH_CALUDE_game_draw_fraction_l3195_319534

theorem game_draw_fraction (jack_win : ℚ) (emma_win : ℚ) 
  (h1 : jack_win = 4/9) (h2 : emma_win = 5/14) : 
  1 - (jack_win + emma_win) = 25/126 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l3195_319534


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l3195_319553

/-- Permutation function: number of ways to arrange k items out of m items -/
def A (m : ℕ) (k : ℕ) : ℕ := m.factorial / (m - k).factorial

/-- The theorem states that the equation 3A₈ⁿ⁻¹ = 4A₉ⁿ⁻² is satisfied when n = 9 -/
theorem permutation_equation_solution :
  ∃ n : ℕ, 3 * A 8 (n - 1) = 4 * A 9 (n - 2) ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l3195_319553


namespace NUMINAMATH_CALUDE_card_drawing_theorem_l3195_319528

/-- The number of cards of each color -/
def cards_per_color : ℕ := 4

/-- The total number of cards -/
def total_cards : ℕ := 4 * cards_per_color

/-- The number of cards to be drawn -/
def cards_drawn : ℕ := 3

/-- The number of ways to draw cards satisfying the conditions -/
def valid_draws : ℕ := 472

theorem card_drawing_theorem : 
  (Nat.choose total_cards cards_drawn) - 
  (4 * Nat.choose cards_per_color cards_drawn) - 
  (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) = 
  valid_draws := by sorry

end NUMINAMATH_CALUDE_card_drawing_theorem_l3195_319528


namespace NUMINAMATH_CALUDE_school_teachers_calculation_l3195_319578

/-- Calculates the number of teachers required in a school given specific conditions -/
theorem school_teachers_calculation (total_students : ℕ) (lessons_per_student : ℕ) 
  (lessons_per_teacher : ℕ) (students_per_class : ℕ) : 
  total_students = 1200 →
  lessons_per_student = 5 →
  lessons_per_teacher = 4 →
  students_per_class = 30 →
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher) = 50 := by
  sorry

#check school_teachers_calculation

end NUMINAMATH_CALUDE_school_teachers_calculation_l3195_319578


namespace NUMINAMATH_CALUDE_angelina_walk_speeds_l3195_319500

theorem angelina_walk_speeds (v : ℝ) :
  v > 0 ∧
  960 / v - 40 = 480 / (2 * v) ∧
  480 / (2 * v) - 20 = 720 / (3 * v) →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by sorry

end NUMINAMATH_CALUDE_angelina_walk_speeds_l3195_319500


namespace NUMINAMATH_CALUDE_problem_solution_l3195_319517

/-- Given that (k-1)x^|k| + 3 ≥ 0 is a one-variable linear inequality about x and (k-1) ≠ 0, prove that k = -1 -/
theorem problem_solution (k : ℝ) : 
  (∀ x, ∃ a b, (k - 1) * x^(|k|) + 3 = a * x + b) → -- Linear inequality condition
  (k - 1 ≠ 0) →                                     -- Non-zero coefficient condition
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3195_319517


namespace NUMINAMATH_CALUDE_triangle_problem_l3195_319522

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) : 
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3195_319522
