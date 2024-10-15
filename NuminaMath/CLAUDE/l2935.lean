import Mathlib

namespace NUMINAMATH_CALUDE_correct_average_after_mark_correction_l2935_293550

theorem correct_average_after_mark_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark correct_mark : ℚ) :
  n = 25 →
  initial_average = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  (n * initial_average - (wrong_mark - correct_mark)) / n = 98 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_mark_correction_l2935_293550


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l2935_293534

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 15) :
  let c := 2 * r * Real.sqrt 3 / 2
  c = 26 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l2935_293534


namespace NUMINAMATH_CALUDE_sallys_raise_l2935_293508

/-- Given Sally's earnings last month and the total for two months, calculate her percentage raise. -/
theorem sallys_raise (last_month : ℝ) (total_two_months : ℝ) : 
  last_month = 1000 → total_two_months = 2100 → 
  (total_two_months - last_month) / last_month * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sallys_raise_l2935_293508


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l2935_293579

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_range (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l2935_293579


namespace NUMINAMATH_CALUDE_root_relation_implies_coefficient_ratio_l2935_293569

/-- Given two quadratic equations with roots related by a factor of 3, prove the ratio of coefficients -/
theorem root_relation_implies_coefficient_ratio
  (m n p : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hp : p ≠ 0)
  (h_root_relation : ∀ x, x^2 + p*x + m = 0 → (3*x)^2 + m*(3*x) + n = 0) :
  n / p = -27 := by
  sorry

end NUMINAMATH_CALUDE_root_relation_implies_coefficient_ratio_l2935_293569


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2935_293552

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  x^2 / 121 - y^2 / 49 = 1 →
  ∃ (v1 v2 : ℝ × ℝ),
    v1 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v2 ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} ∧
    v1 ≠ v2 ∧
    ∀ (v : ℝ × ℝ),
      v ∈ {p : ℝ × ℝ | p.1^2 / 121 - p.2^2 / 49 = 1} →
      v.2 = 0 →
      v = v1 ∨ v = v2 ∧
    Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2935_293552


namespace NUMINAMATH_CALUDE_number_times_five_equals_hundred_l2935_293522

theorem number_times_five_equals_hundred (x : ℝ) : x * 5 = 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_times_five_equals_hundred_l2935_293522


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_problem_2_l2935_293529

/-- Two quantities are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 4 = 36 →
  x 9 = 16 :=
by sorry

theorem inverse_proportion_problem_2 :
  ∀ a b : ℝ → ℝ,
  InverselyProportional a b →
  a 5 = 50 →
  a 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_problem_2_l2935_293529


namespace NUMINAMATH_CALUDE_science_book_page_count_l2935_293571

def history_book_pages : ℕ := 300

def novel_pages (history : ℕ) : ℕ := history / 2

def science_book_pages (novel : ℕ) : ℕ := 4 * novel

theorem science_book_page_count : 
  science_book_pages (novel_pages history_book_pages) = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_page_count_l2935_293571


namespace NUMINAMATH_CALUDE_initial_phone_price_l2935_293574

/-- The initial price of a phone given negotiation conditions. -/
theorem initial_phone_price (negotiated_price : ℝ) (negotiation_percentage : ℝ) 
  (h1 : negotiated_price = 480)
  (h2 : negotiation_percentage = 0.20)
  (h3 : negotiated_price = negotiation_percentage * initial_price) : 
  initial_price = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_phone_price_l2935_293574


namespace NUMINAMATH_CALUDE_min_odd_correct_answers_l2935_293567

/-- Represents the number of correct answers a student can give -/
inductive CorrectAnswers
  | zero
  | one
  | two
  | three
  | four

/-- Represents the distribution of correct answers among students -/
structure AnswerDistribution where
  total : Nat
  zero : Nat
  one : Nat
  two : Nat
  three : Nat
  four : Nat
  sum_constraint : total = zero + one + two + three + four

/-- Checks if a distribution satisfies the problem constraints -/
def satisfies_constraints (d : AnswerDistribution) : Prop :=
  d.total = 50 ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.three)).card ≥ 1) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.two)).card ≥ 2) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.one)).card ≥ 3) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.zero)).card ≥ 4)

/-- The main theorem to prove -/
theorem min_odd_correct_answers (d : AnswerDistribution) 
  (h : satisfies_constraints d) : d.one + d.three ≥ 23 := by
  sorry


end NUMINAMATH_CALUDE_min_odd_correct_answers_l2935_293567


namespace NUMINAMATH_CALUDE_octal_726_to_binary_l2935_293510

/-- Converts a single digit from base 8 to its 3-digit binary representation -/
def octalToBinary (digit : Nat) : Fin 8 → Fin 2 × Fin 2 × Fin 2 := sorry

/-- Converts a 3-digit octal number to its 9-digit binary representation -/
def octalToBinaryThreeDigits (a b c : Fin 8) : Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 × Fin 2 := sorry

theorem octal_726_to_binary :
  octalToBinaryThreeDigits 7 2 6 = (1, 1, 1, 0, 1, 0, 1, 1, 0) := by sorry

end NUMINAMATH_CALUDE_octal_726_to_binary_l2935_293510


namespace NUMINAMATH_CALUDE_loraine_wax_usage_l2935_293532

/-- The number of wax sticks used for all animals -/
def total_wax_sticks (large_animal_wax small_animal_wax : ℕ) 
  (small_animal_ratio : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  small_animal_total_wax + 
  (small_animal_total_wax / small_animal_wax) / small_animal_ratio * large_animal_wax

/-- Proof that Loraine used 20 sticks of wax for all animals -/
theorem loraine_wax_usage : 
  total_wax_sticks 4 2 3 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_loraine_wax_usage_l2935_293532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2935_293533

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first : a 0 = 13) 
  (h_last : a 4 = 37) : 
  a 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2935_293533


namespace NUMINAMATH_CALUDE_seventeen_in_binary_l2935_293556

theorem seventeen_in_binary : 17 = 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_binary_l2935_293556


namespace NUMINAMATH_CALUDE_keith_picked_six_apples_l2935_293561

/-- Given the number of apples picked by Mike, eaten by Nancy, and left in total,
    calculate the number of apples picked by Keith. -/
def keith_apples (mike_picked : ℝ) (nancy_ate : ℝ) (total_left : ℝ) : ℝ :=
  total_left - (mike_picked - nancy_ate)

/-- Theorem stating that Keith picked 6.0 apples given the problem conditions. -/
theorem keith_picked_six_apples :
  keith_apples 7.0 3.0 10 = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_six_apples_l2935_293561


namespace NUMINAMATH_CALUDE_max_consecutive_funny_numbers_l2935_293526

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is funny if it's divisible by the sum of its digits plus one -/
def isFunny (n : ℕ) : Prop := n % (sumOfDigits n + 1) = 0

/-- The maximum number of consecutive funny numbers is 1 -/
theorem max_consecutive_funny_numbers :
  ∀ n : ℕ, isFunny n → isFunny (n + 1) → False := by sorry

end NUMINAMATH_CALUDE_max_consecutive_funny_numbers_l2935_293526


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2935_293501

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2935_293501


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l2935_293513

/-- Proves that given a mixture of zinc and copper in the ratio 9:11, 
    where 27 kg of zinc is used, the total weight of the mixture is 60 kg. -/
theorem zinc_copper_mixture_weight (zinc_weight : ℝ) (copper_weight : ℝ) :
  zinc_weight = 27 →
  zinc_weight / copper_weight = 9 / 11 →
  zinc_weight + copper_weight = 60 := by
sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l2935_293513


namespace NUMINAMATH_CALUDE_roots_in_interval_l2935_293559

theorem roots_in_interval (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ 
  -3/2 < m ∧ m < 12/7 :=
by sorry

end NUMINAMATH_CALUDE_roots_in_interval_l2935_293559


namespace NUMINAMATH_CALUDE_altered_detergent_theorem_l2935_293583

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let original_detergent_water_ratio := original.detergent / original.water
  let new_detergent_water_ratio := original_detergent_water_ratio / 2
  water_amount * new_detergent_water_ratio

/-- Theorem stating the amount of detergent in the altered solution -/
theorem altered_detergent_theorem (original : CleaningSolution) 
    (h1 : original.bleach = 2)
    (h2 : original.detergent = 25)
    (h3 : original.water = 100)
    (h4 : altered_detergent_amount original 300 = 37.5) : 
  altered_detergent_amount original 300 = 37.5 := by
  sorry

#check altered_detergent_theorem

end NUMINAMATH_CALUDE_altered_detergent_theorem_l2935_293583


namespace NUMINAMATH_CALUDE_lindsey_september_savings_l2935_293566

/-- The amount of money Lindsey saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- The amount of money Lindsey saved in November -/
def november_savings : ℕ := 11

/-- The amount of money Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- The cost of the video game Lindsey bought -/
def video_game_cost : ℕ := 87

/-- The amount of money Lindsey had left after buying the video game -/
def money_left : ℕ := 36

/-- Theorem stating that Lindsey saved $50 in September -/
theorem lindsey_september_savings :
  september_savings = 50 ∧
  september_savings + october_savings + november_savings > 75 ∧
  september_savings + october_savings + november_savings + mom_gift = video_game_cost + money_left :=
sorry

end NUMINAMATH_CALUDE_lindsey_september_savings_l2935_293566


namespace NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l2935_293544

theorem tan_neg_five_pi_fourth : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_five_pi_fourth_l2935_293544


namespace NUMINAMATH_CALUDE_semi_annual_compounding_l2935_293568

noncomputable def compound_interest_frequency 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : ℝ :=
  let r := annual_rate / 100
  ((final_amount / initial_investment) ^ (1 / (r * years)) - 1) / (r / years)

theorem semi_annual_compounding 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) 
  (h1 : initial_investment = 10000) 
  (h2 : annual_rate = 3.96) 
  (h3 : final_amount = 10815.83) 
  (h4 : years = 2) :
  ∃ ε > 0, |compound_interest_frequency initial_investment annual_rate final_amount years - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_semi_annual_compounding_l2935_293568


namespace NUMINAMATH_CALUDE_no_solutions_to_inequality_system_l2935_293573

theorem no_solutions_to_inequality_system :
  ¬ ∃ (x y : ℝ), 11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_inequality_system_l2935_293573


namespace NUMINAMATH_CALUDE_article_price_decrease_l2935_293563

theorem article_price_decrease (price_after_decrease : ℝ) (decrease_percentage : ℝ) :
  price_after_decrease = 200 ∧ decrease_percentage = 20 →
  (price_after_decrease / (1 - decrease_percentage / 100)) = 250 :=
by sorry

end NUMINAMATH_CALUDE_article_price_decrease_l2935_293563


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l2935_293543

theorem complex_sum_to_polar : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 8)) + 
  5 * Complex.exp (Complex.I * (17 * Real.pi / 16)) = 
  10 * Real.cos (5 * Real.pi / 32) * Complex.exp (Complex.I * (23 * Real.pi / 32)) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l2935_293543


namespace NUMINAMATH_CALUDE_unbroken_seashells_l2935_293575

theorem unbroken_seashells (total_seashells : ℕ) (broken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) : 
  total_seashells - broken_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l2935_293575


namespace NUMINAMATH_CALUDE_least_addend_proof_l2935_293542

/-- The least non-negative integer that, when added to 11002, results in a number divisible by 11 -/
def least_addend : ℕ := 9

/-- The original number we start with -/
def original_number : ℕ := 11002

theorem least_addend_proof :
  (∀ k : ℕ, k < least_addend → ¬((original_number + k) % 11 = 0)) ∧
  ((original_number + least_addend) % 11 = 0) :=
sorry

end NUMINAMATH_CALUDE_least_addend_proof_l2935_293542


namespace NUMINAMATH_CALUDE_expand_product_l2935_293530

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) * (x + 6) = x^3 + 5*x^2 - 26*x - 120 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2935_293530


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l2935_293596

theorem birthday_celebration_attendance (total_guests : ℕ) 
  (women_ratio : ℚ) (men_count : ℕ) (men_left_ratio : ℚ) (children_left : ℕ) : 
  total_guests = 60 →
  women_ratio = 1/2 →
  men_count = 15 →
  men_left_ratio = 1/3 →
  children_left = 5 →
  ∃ (stayed : ℕ), stayed = 50 := by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendance_l2935_293596


namespace NUMINAMATH_CALUDE_problem_polygon_area_l2935_293515

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a polygon on a 2D grid --/
structure GridPolygon where
  vertices : List GridPoint

/-- Calculates the area of a polygon on a grid --/
def calculateGridPolygonArea (polygon : GridPolygon) : ℕ :=
  sorry

/-- The specific polygon from the problem --/
def problemPolygon : GridPolygon :=
  { vertices := [
    {x := 0, y := 0}, {x := 1, y := 0}, {x := 1, y := 1}, {x := 2, y := 1},
    {x := 3, y := 0}, {x := 3, y := 1}, {x := 4, y := 0}, {x := 4, y := 1},
    {x := 4, y := 3}, {x := 3, y := 3}, {x := 4, y := 4}, {x := 3, y := 4},
    {x := 2, y := 4}, {x := 0, y := 4}, {x := 0, y := 2}, {x := 0, y := 0}
  ] }

theorem problem_polygon_area : calculateGridPolygonArea problemPolygon = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l2935_293515


namespace NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l2935_293535

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l2935_293535


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l2935_293528

def smaller_number : ℝ := 20

def larger_number : ℝ := 6 * smaller_number

theorem ratio_of_numbers : larger_number / smaller_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l2935_293528


namespace NUMINAMATH_CALUDE_angle_representation_l2935_293525

theorem angle_representation (given_angle : ℝ) : 
  given_angle = -1485 → 
  ∃ (α k : ℝ), 
    given_angle = α + k * 360 ∧ 
    0 ≤ α ∧ α < 360 ∧ 
    k = -5 ∧
    α = 315 := by
  sorry

end NUMINAMATH_CALUDE_angle_representation_l2935_293525


namespace NUMINAMATH_CALUDE_penny_pudding_grains_l2935_293594

-- Define the given conditions
def cans_per_tonne : ℕ := 25000
def grains_per_tonne : ℕ := 50000000

-- Define the function to calculate grains per can
def grains_per_can : ℕ := grains_per_tonne / cans_per_tonne

-- Theorem statement
theorem penny_pudding_grains :
  grains_per_can = 2000 :=
sorry

end NUMINAMATH_CALUDE_penny_pudding_grains_l2935_293594


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2935_293549

/-- A function satisfying the given properties -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x - y) = g x * g y) ∧
  (∀ x : ℝ, g x ≠ 0) ∧
  (g 0 = 1)

/-- Theorem stating that g(5) = e^5 for functions satisfying the given properties -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  g 5 = Real.exp 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2935_293549


namespace NUMINAMATH_CALUDE_units_digit_of_product_is_8_l2935_293546

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_product_is_8 :
  (first_four_composites.prod % 10 = 8) := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_is_8_l2935_293546


namespace NUMINAMATH_CALUDE_f_composition_equals_226_l2935_293593

def f (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

theorem f_composition_equals_226 : f (f (f 1)) = 226 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_226_l2935_293593


namespace NUMINAMATH_CALUDE_apple_sharing_ways_l2935_293580

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute apples among people with a minimum requirement --/
def apple_distribution (total min_per_person people : ℕ) : ℕ :=
  stars_and_bars (total - min_per_person * people) people

theorem apple_sharing_ways :
  apple_distribution 24 2 3 = 190 := by
  sorry

end NUMINAMATH_CALUDE_apple_sharing_ways_l2935_293580


namespace NUMINAMATH_CALUDE_smallest_n_for_powers_l2935_293503

theorem smallest_n_for_powers : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 3^n = a^4) ∧ 
  (∃ (b : ℕ), 2^n = b^6) ∧ 
  (∀ (m : ℕ), m > 0 → (∃ (c : ℕ), 3^m = c^4) → (∃ (d : ℕ), 2^m = d^6) → m ≥ n) ∧
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_powers_l2935_293503


namespace NUMINAMATH_CALUDE_fraction_division_equality_l2935_293518

theorem fraction_division_equality : (-1/12 + 1/3 - 1/2) / (-1/18) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l2935_293518


namespace NUMINAMATH_CALUDE_remainder_theorem_l2935_293545

theorem remainder_theorem (P D Q R D' Q' R' D'' S T : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R' = S + T)
  (h4 : S = D'' * T) :
  P % (D * D' * D'') = D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2935_293545


namespace NUMINAMATH_CALUDE_expected_heads_is_94_l2935_293548

/-- The probability of a coin landing on heads after at most four flips -/
def prob_heads : ℚ := 15 / 16

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The expected number of coins landing on heads -/
def expected_heads : ℚ := num_coins * prob_heads

theorem expected_heads_is_94 :
  ⌊expected_heads⌋ = 94 := by sorry

end NUMINAMATH_CALUDE_expected_heads_is_94_l2935_293548


namespace NUMINAMATH_CALUDE_sequence_ratio_density_l2935_293586

theorem sequence_ratio_density (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, 0 < a (n + 1) - a n ∧ a (n + 1) - a n < Real.sqrt (a n)) :
  ∀ x y : ℝ, 0 < x → x < y → y < 1 → 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ x < (a k : ℝ) / (a m : ℝ) ∧ (a k : ℝ) / (a m : ℝ) < y :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_density_l2935_293586


namespace NUMINAMATH_CALUDE_correct_inequality_l2935_293504

theorem correct_inequality : 
  (-3 > -5) ∧ 
  ¬(-3 > -2) ∧ 
  ¬((1:ℚ)/2 < (1:ℚ)/3) ∧ 
  ¬(-(1:ℚ)/2 < -(2:ℚ)/3) := by
  sorry

end NUMINAMATH_CALUDE_correct_inequality_l2935_293504


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2935_293507

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2935_293507


namespace NUMINAMATH_CALUDE_x_squared_greater_than_x_root_l2935_293516

theorem x_squared_greater_than_x_root (x : ℝ) : x ^ 2 > x ^ (1 / 2) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_greater_than_x_root_l2935_293516


namespace NUMINAMATH_CALUDE_other_number_proof_l2935_293527

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 33)
  (h3 : a = 231) : 
  b = 660 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l2935_293527


namespace NUMINAMATH_CALUDE_distinct_arithmetic_sequences_l2935_293588

/-- The largest prime power factor of a positive integer -/
def largest_prime_power_factor (n : ℕ+) : ℕ+ := sorry

/-- Check if two positive integers have the same largest prime power factor -/
def same_largest_prime_power_factor (m n : ℕ+) : Prop := 
  largest_prime_power_factor m = largest_prime_power_factor n

theorem distinct_arithmetic_sequences 
  (n : Fin 10000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_same_factor : ∀ i j, same_largest_prime_power_factor (n i) (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → a i + k * (n i : ℤ) ≠ a j + l * (n j : ℤ) := by
    sorry

end NUMINAMATH_CALUDE_distinct_arithmetic_sequences_l2935_293588


namespace NUMINAMATH_CALUDE_midpoint_locus_l2935_293565

/-- Given a circle x^2 + y^2 = 1, point A(1,0), and triangle ABC inscribed in the circle
    with angle BAC = 60°, the locus of the midpoint of BC as BC moves on the circle
    is described by the equation x^2 + y^2 = 1/4 for x < 1/4 -/
theorem midpoint_locus (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ),
    x1^2 + y1^2 = 1 ∧
    x2^2 + y2^2 = 1 ∧
    x = (x1 + x2) / 2 ∧
    y = (y1 + y2) / 2 ∧
    (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + y2^2 - (x1 - x2)^2 - (y1 - y2)^2 = 1) →
  x < 1/4 →
  x^2 + y^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2935_293565


namespace NUMINAMATH_CALUDE_weight_measurement_l2935_293578

def weights : List ℕ := [1, 3, 9, 27]

theorem weight_measurement (w : List ℕ := weights) :
  (∃ (S : List ℕ), S.sum = (List.sum w)) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ (List.sum w) → 
    ∃ (S : List ℕ), (∀ x ∈ S, x ∈ w) ∧ S.sum = n) :=
by sorry

end NUMINAMATH_CALUDE_weight_measurement_l2935_293578


namespace NUMINAMATH_CALUDE_solve_equation_l2935_293541

theorem solve_equation (m : ℝ) : m + (m + 2) + (m + 4) = 21 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2935_293541


namespace NUMINAMATH_CALUDE_hotel_meal_spending_l2935_293599

theorem hotel_meal_spending (total_persons : ℕ) (regular_spenders : ℕ) (regular_amount : ℕ) 
  (extra_amount : ℕ) (total_spent : ℕ) :
  total_persons = 9 →
  regular_spenders = 8 →
  regular_amount = 12 →
  extra_amount = 8 →
  total_spent = 117 →
  ∃ x : ℕ, (regular_spenders * regular_amount) + (x + extra_amount) = total_spent ∧ x = 13 :=
by sorry

end NUMINAMATH_CALUDE_hotel_meal_spending_l2935_293599


namespace NUMINAMATH_CALUDE_parabola_focus_l2935_293502

/-- The parabola defined by the equation y^2 + 4x = 0 -/
def parabola (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop := 
  ∃ (a b : ℝ), p = (a, b) ∧ 
  ∀ (x y : ℝ), parabola x y → (x - a)^2 + (y - b)^2 = (x + a)^2

theorem parabola_focus :
  focus (-1, 0) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2935_293502


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2935_293519

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2935_293519


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2935_293500

theorem reciprocal_of_negative_fraction (n : ℤ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n)⁻¹ = -n :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l2935_293500


namespace NUMINAMATH_CALUDE_becky_new_necklaces_l2935_293582

/-- The number of new necklaces Becky bought -/
def new_necklaces (initial : ℕ) (broken : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - broken - given_away)

/-- Theorem stating that Becky bought 5 new necklaces -/
theorem becky_new_necklaces :
  new_necklaces 50 3 15 37 = 5 := by
  sorry

end NUMINAMATH_CALUDE_becky_new_necklaces_l2935_293582


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2935_293553

theorem other_root_of_quadratic (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - (a+b+c)*x + ab + bc + ca
  let ab := a + b
  let ab_bc_ca := ab + bc + ca
  f ab = 0 →
  ∃ k, f k = 0 ∧ k = (ab + bc + ca) / (a + b) :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2935_293553


namespace NUMINAMATH_CALUDE_chord_intersection_sum_l2935_293538

-- Define the sphere and point S
variable (sphere : Type) (S : sphere)

-- Define the chords
variable (A A' B B' C C' : sphere)

-- Define the lengths
variable (AS BS CS : ℝ)

-- Define the volume ratio
variable (volume_ratio : ℝ)

-- State the theorem
theorem chord_intersection_sum (h1 : AS = 6) (h2 : BS = 3) (h3 : CS = 2)
  (h4 : volume_ratio = 2/9) :
  ∃ (SA' SB' SC' : ℝ), SA' + SB' + SC' = 18 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_sum_l2935_293538


namespace NUMINAMATH_CALUDE_train_journey_time_l2935_293572

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0)
  (h3 : (6 / 7 * usual_speed) * (usual_time + 20) = usual_speed * usual_time) :
  usual_time = 140 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l2935_293572


namespace NUMINAMATH_CALUDE_employee_new_salary_is_35000_l2935_293521

/-- Calculates the new salary of employees after a salary redistribution --/
def new_employee_salary (emily_original_salary emily_new_salary num_employees employee_original_salary : ℕ) : ℕ :=
  let salary_reduction := emily_original_salary - emily_new_salary
  let additional_per_employee := salary_reduction / num_employees
  employee_original_salary + additional_per_employee

/-- Proves that the new employee salary is $35,000 given the problem conditions --/
theorem employee_new_salary_is_35000 :
  new_employee_salary 1000000 850000 10 20000 = 35000 := by
  sorry

end NUMINAMATH_CALUDE_employee_new_salary_is_35000_l2935_293521


namespace NUMINAMATH_CALUDE_cubic_sum_equality_l2935_293540

theorem cubic_sum_equality (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 2*y^3 + 2*z^3 + 6*x*y*z = 24 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equality_l2935_293540


namespace NUMINAMATH_CALUDE_inequality_solution_l2935_293524

theorem inequality_solution (x : ℝ) : x - 1 / x > 0 ↔ (-1 < x ∧ x < 0) ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2935_293524


namespace NUMINAMATH_CALUDE_rotation_equivalence_l2935_293558

theorem rotation_equivalence (x : ℝ) : 
  (420 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l2935_293558


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2935_293562

/-- Given point A and vector AB, find the coordinates of point B -/
theorem point_B_coordinates (A B : ℝ × ℝ × ℝ) (AB : ℝ × ℝ × ℝ) :
  A = (3, -1, 0) →
  AB = (2, 5, -3) →
  B = (5, 4, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2935_293562


namespace NUMINAMATH_CALUDE_checkers_placement_divisibility_l2935_293587

theorem checkers_placement_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  (Nat.choose (p^2) p) % (p^5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_checkers_placement_divisibility_l2935_293587


namespace NUMINAMATH_CALUDE_cash_percentage_proof_l2935_293506

/-- Calculates the percentage of total amount spent as cash given the total amount and amounts spent on raw materials and machinery. -/
def percentage_spent_as_cash (total_amount raw_materials machinery : ℚ) : ℚ :=
  ((total_amount - (raw_materials + machinery)) / total_amount) * 100

/-- Proves that given a total amount of $250, with $100 spent on raw materials and $125 spent on machinery, the percentage of the total amount spent as cash is 10%. -/
theorem cash_percentage_proof :
  percentage_spent_as_cash 250 100 125 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_proof_l2935_293506


namespace NUMINAMATH_CALUDE_investment_growth_l2935_293591

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that the given investment scenario results in the expected amount -/
theorem investment_growth (principal : ℝ) (rate : ℝ) (years : ℕ) 
  (h1 : principal = 2000)
  (h2 : rate = 0.05)
  (h3 : years = 18) :
  ∃ ε > 0, |compound_interest principal rate years - 4813.24| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_growth_l2935_293591


namespace NUMINAMATH_CALUDE_nice_people_count_l2935_293547

/-- Represents the number of nice people for a given name and total count -/
def nice_count (name : String) (total : ℕ) : ℕ :=
  match name with
  | "Barry" => total
  | "Kevin" => total / 2
  | "Julie" => total * 3 / 4
  | "Joe" => total / 10
  | _ => 0

/-- The total number of nice people in the crowd -/
def total_nice_people : ℕ :=
  nice_count "Barry" 24 + nice_count "Kevin" 20 + nice_count "Julie" 80 + nice_count "Joe" 50

theorem nice_people_count : total_nice_people = 99 := by
  sorry

end NUMINAMATH_CALUDE_nice_people_count_l2935_293547


namespace NUMINAMATH_CALUDE_movie_theater_shows_24_movies_l2935_293505

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

/-- Theorem: A movie theater with 6 screens, open for 8 hours, where each movie lasts 2 hours,
    shows 24 movies throughout the day. -/
theorem movie_theater_shows_24_movies :
  total_movies 6 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_shows_24_movies_l2935_293505


namespace NUMINAMATH_CALUDE_bottom_layer_lights_for_specific_tower_l2935_293520

/-- Represents a tower with a geometric progression of lights -/
structure LightTower where
  layers : ℕ
  total_lights : ℕ
  ratio : ℕ

/-- Calculates the number of lights on the bottom layer of a tower -/
def bottom_layer_lights (tower : LightTower) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The theorem stating the number of lights on the bottom layer of the specific tower -/
theorem bottom_layer_lights_for_specific_tower :
  let tower : LightTower := ⟨5, 242, 3⟩
  bottom_layer_lights tower = 162 := by
  sorry

end NUMINAMATH_CALUDE_bottom_layer_lights_for_specific_tower_l2935_293520


namespace NUMINAMATH_CALUDE_fraction_inequality_l2935_293514

theorem fraction_inequality (a b : ℝ) :
  (b > 0 ∧ 0 > a → 1/a < 1/b) ∧
  (0 > a ∧ a > b → 1/a < 1/b) ∧
  (a > b ∧ b > 0 → 1/a < 1/b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2935_293514


namespace NUMINAMATH_CALUDE_incorrect_denominator_clearing_l2935_293589

theorem incorrect_denominator_clearing (x : ℝ) : 
  ¬((-((3*x+1)/2) - ((2*x-5)/6) > 1) ↔ (3*(3*x+1)+(2*x-5) > -6)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_denominator_clearing_l2935_293589


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2935_293554

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y → y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2935_293554


namespace NUMINAMATH_CALUDE_cube_coloring_count_l2935_293592

/-- The number of distinct colorings of a cube with 6 faces using m colors -/
def g (m : ℕ) : ℚ :=
  (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2)

/-- Theorem: The number of distinct colorings of a cube with 6 faces,
    using m colors, where each face is painted one color,
    is equal to (1/24)(m^6 + 3m^4 + 12m^3 + 8m^2) -/
theorem cube_coloring_count (m : ℕ) :
  g m = (1 / 24) * (m^6 + 3*m^4 + 12*m^3 + 8*m^2) :=
by sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l2935_293592


namespace NUMINAMATH_CALUDE_unique_m_value_l2935_293551

theorem unique_m_value : ∃! m : ℝ, ∀ y : ℝ, 
  (y - 2 = 1) → (m * y - 2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l2935_293551


namespace NUMINAMATH_CALUDE_combined_degrees_theorem_l2935_293595

/-- Represents the budget allocation for Megatech Corporation's research and development --/
structure BudgetAllocation where
  microphotonics : Float
  homeElectronics : Float
  foodAdditives : Float
  geneticallyModifiedMicroorganisms : Float
  industrialLubricants : Float
  nanotechnology : Float

/-- Calculates the degrees in a circle graph for a given percentage --/
def percentageToDegrees (percentage : Float) : Float :=
  percentage * 360 / 100

/-- Calculates the combined degrees for basic astrophysics and nanotechnology --/
def combinedDegrees (budget : BudgetAllocation) : Float :=
  let basicAstrophysics := 100 - (budget.microphotonics + budget.homeElectronics + 
                                  budget.foodAdditives + budget.geneticallyModifiedMicroorganisms + 
                                  budget.industrialLubricants + budget.nanotechnology)
  percentageToDegrees (basicAstrophysics + budget.nanotechnology)

/-- Theorem: The combined degrees for basic astrophysics and nanotechnology is 50.4 --/
theorem combined_degrees_theorem (budget : BudgetAllocation) 
  (h1 : budget.microphotonics = 10)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.geneticallyModifiedMicroorganisms = 29)
  (h5 : budget.industrialLubricants = 8)
  (h6 : budget.nanotechnology = 7) :
  combinedDegrees budget = 50.4 := by
  sorry


end NUMINAMATH_CALUDE_combined_degrees_theorem_l2935_293595


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2935_293570

/-- The dimensions of the larger box -/
def box_dimensions : Fin 3 → ℕ
| 0 => 3  -- length
| 1 => 2  -- width
| 2 => 3  -- height
| _ => 0

/-- The dimensions of the smaller block -/
def block_dimensions : Fin 3 → ℕ
| 0 => 2  -- length
| 1 => 2  -- width
| 2 => 1  -- height
| _ => 0

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (dimensions : Fin 3 → ℕ) : ℕ :=
  dimensions 0 * dimensions 1 * dimensions 2

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 4

/-- Theorem stating that the maximum number of blocks that can fit in the box is 4 -/
theorem max_blocks_fit :
  (volume box_dimensions ≥ max_blocks * volume block_dimensions) ∧
  (∀ n : ℕ, n > max_blocks → volume box_dimensions < n * volume block_dimensions) :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l2935_293570


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2935_293597

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (age_15th_student : ℝ), 
    age_15th_student = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧ 
    age_15th_student = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2935_293597


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2935_293511

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop := 2 * x^2 + 4 * x + 6 * y = 24

/-- The slope of the tangent line at a given point -/
noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -(2/3 * x + 2/3)

/-- Theorem: The slope of the tangent line to the curve at x = 1 is -4/3 -/
theorem tangent_slope_at_one : 
  tangent_slope 1 = -4/3 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2935_293511


namespace NUMINAMATH_CALUDE_painting_progress_l2935_293560

/-- Represents the fraction of a wall painted in a given time -/
def fraction_painted (total_time minutes : ℕ) : ℚ :=
  minutes / total_time

theorem painting_progress (heidi_time karl_time minutes : ℕ) 
  (h1 : heidi_time = 60)
  (h2 : karl_time = heidi_time / 2)
  (h3 : minutes = 20) :
  (fraction_painted heidi_time minutes = 1/3) ∧ 
  (fraction_painted karl_time minutes = 2/3) := by
  sorry

#check painting_progress

end NUMINAMATH_CALUDE_painting_progress_l2935_293560


namespace NUMINAMATH_CALUDE_percentage_of_five_digit_numbers_with_repeats_l2935_293509

def five_digit_numbers : ℕ := 90000

def numbers_without_repeats : ℕ := 9 * 9 * 8 * 7 * 6

def numbers_with_repeats : ℕ := five_digit_numbers - numbers_without_repeats

def percentage_with_repeats : ℚ := numbers_with_repeats / five_digit_numbers

theorem percentage_of_five_digit_numbers_with_repeats :
  (percentage_with_repeats * 100).floor / 10 = 698 / 10 := by sorry

end NUMINAMATH_CALUDE_percentage_of_five_digit_numbers_with_repeats_l2935_293509


namespace NUMINAMATH_CALUDE_g_of_3_equals_5_l2935_293590

-- Define the function g
def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

-- State the theorem
theorem g_of_3_equals_5 : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_5_l2935_293590


namespace NUMINAMATH_CALUDE_picnic_watermelon_slices_l2935_293581

/-- The number of watermelon slices at a family picnic -/
def total_watermelon_slices : ℕ :=
  let danny_watermelons : ℕ := 3
  let danny_slices_per_watermelon : ℕ := 10
  let sister_watermelons : ℕ := 1
  let sister_slices_per_watermelon : ℕ := 15
  (danny_watermelons * danny_slices_per_watermelon) + (sister_watermelons * sister_slices_per_watermelon)

theorem picnic_watermelon_slices : total_watermelon_slices = 45 := by
  sorry

end NUMINAMATH_CALUDE_picnic_watermelon_slices_l2935_293581


namespace NUMINAMATH_CALUDE_incorrect_calculation_l2935_293536

theorem incorrect_calculation : ¬(3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l2935_293536


namespace NUMINAMATH_CALUDE_exponent_calculation_correct_and_uses_operations_l2935_293564

-- Define the exponent operations
inductive ExponentOperation
  | multiplication
  | division
  | exponentiation
  | productExponentiation

-- Define a function to represent the calculation
def exponentCalculation (a : ℝ) : ℝ := (a^3 * a^2)^2

-- Define a function to represent the result of the calculation
def exponentResult (a : ℝ) : ℝ := a^10

-- Define a function to check if an operation is used in the calculation
def isOperationUsed (op : ExponentOperation) : Prop :=
  match op with
  | ExponentOperation.multiplication => True
  | ExponentOperation.exponentiation => True
  | ExponentOperation.productExponentiation => True
  | _ => False

-- Theorem stating that the calculation is correct and uses the specified operations
theorem exponent_calculation_correct_and_uses_operations (a : ℝ) :
  exponentCalculation a = exponentResult a ∧
  isOperationUsed ExponentOperation.multiplication ∧
  isOperationUsed ExponentOperation.exponentiation ∧
  isOperationUsed ExponentOperation.productExponentiation :=
by sorry

end NUMINAMATH_CALUDE_exponent_calculation_correct_and_uses_operations_l2935_293564


namespace NUMINAMATH_CALUDE_winnie_balloon_distribution_l2935_293598

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 242) (h2 : num_friends = 12) : 
  total_balloons % num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_winnie_balloon_distribution_l2935_293598


namespace NUMINAMATH_CALUDE_frog_jumps_theorem_l2935_293576

/-- Represents a hexagon with vertices A, B, C, D, E, F -/
inductive Vertex
| A | B | C | D | E | F

/-- Represents the number of paths from A to C in n jumps -/
def paths_to_C (n : ℕ) : ℕ := (2^n - 1) / 3

/-- Represents the number of paths from A to C in n jumps avoiding D -/
def paths_to_C_avoiding_D (n : ℕ) : ℕ := 3^(n/2 - 1)

/-- Represents the probability of survival after n jumps with a mine at D -/
def survival_probability (n : ℕ) : ℚ := (3/4)^((n + 1)/2 - 1)

/-- The average lifespan of frogs -/
def average_lifespan : ℕ := 9

/-- Main theorem stating the properties of frog jumps on a hexagon -/
theorem frog_jumps_theorem :
  ∀ n : ℕ,
  (paths_to_C n = (2^n - 1) / 3) ∧
  (paths_to_C_avoiding_D n = 3^(n/2 - 1)) ∧
  (survival_probability n = (3/4)^((n + 1)/2 - 1)) ∧
  (average_lifespan = 9) :=
by sorry

end NUMINAMATH_CALUDE_frog_jumps_theorem_l2935_293576


namespace NUMINAMATH_CALUDE_matrix_equals_five_l2935_293585

-- Define the matrix
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 2*x, 4*x]

-- Define the determinant of a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem matrix_equals_five (x : ℝ) : 
  det2x2 (matrix x 0 0) (matrix x 0 1) (matrix x 1 0) (matrix x 1 1) = 5 ↔ 
  x = 5/6 ∨ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_matrix_equals_five_l2935_293585


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2935_293584

theorem no_positive_integer_solution (f : ℕ+ → ℕ+) (a b : ℕ+) : 
  (∀ x, f x = x^2 + x) → 4 * (f a) ≠ f b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2935_293584


namespace NUMINAMATH_CALUDE_combined_yellow_ratio_approx_31_percent_l2935_293531

/-- Represents a bag of jelly beans -/
structure JellyBeanBag where
  total : ℕ
  yellow_ratio : ℚ

/-- Calculates the total number of yellow jelly beans in a bag -/
def yellow_count (bag : JellyBeanBag) : ℚ :=
  bag.total * bag.yellow_ratio

/-- Theorem: The ratio of yellow jelly beans to all beans when three bags are combined -/
theorem combined_yellow_ratio_approx_31_percent 
  (bag1 bag2 bag3 : JellyBeanBag)
  (h1 : bag1 = ⟨26, 1/2⟩)
  (h2 : bag2 = ⟨28, 1/4⟩)
  (h3 : bag3 = ⟨30, 1/5⟩) :
  let total_yellow := yellow_count bag1 + yellow_count bag2 + yellow_count bag3
  let total_beans := bag1.total + bag2.total + bag3.total
  abs ((total_yellow / total_beans) - 31/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_combined_yellow_ratio_approx_31_percent_l2935_293531


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2935_293523

theorem quadratic_rewrite (b : ℕ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 68 = (x + n)^2 + 32) →
  b % 2 = 0 →
  b > 0 →
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2935_293523


namespace NUMINAMATH_CALUDE_chopped_cube_height_chopped_cube_height_value_l2935_293517

/-- The height of a 2x2x2 cube with a corner chopped off -/
theorem chopped_cube_height : ℝ :=
  let cube_side : ℝ := 2
  let cut_face_side : ℝ := 2 * Real.sqrt 2
  let cut_face_area : ℝ := Real.sqrt 3 / 4 * cut_face_side^2
  let removed_pyramid_height : ℝ := Real.sqrt 3 / 9
  cube_side - removed_pyramid_height

/-- Theorem stating that the height of the chopped cube is (17√3)/9 -/
theorem chopped_cube_height_value : chopped_cube_height = (17 * Real.sqrt 3) / 9 := by
  sorry


end NUMINAMATH_CALUDE_chopped_cube_height_chopped_cube_height_value_l2935_293517


namespace NUMINAMATH_CALUDE_no_integer_solution_l2935_293539

theorem no_integer_solution : ¬ ∃ (x y z : ℤ), (x - y)^3 + (y - z)^3 + (z - x)^3 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2935_293539


namespace NUMINAMATH_CALUDE_negation_equivalence_l2935_293557

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- Theorem statement
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2935_293557


namespace NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2935_293537

-- Define the probabilities
def prob_east_wind : ℚ := 3/10
def prob_rain : ℚ := 11/30
def prob_both : ℚ := 4/15

-- State the theorem
theorem conditional_probability_rain_given_east_wind :
  (prob_both / prob_east_wind : ℚ) = 8/9 := by
sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_given_east_wind_l2935_293537


namespace NUMINAMATH_CALUDE_missing_digit_is_one_l2935_293555

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_sum (d : ℕ) : ℕ :=
  3 + 5 + 7 + 2 + d + 9

theorem missing_digit_is_one :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_3 (357200 + d * 10 + 9) ↔ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_one_l2935_293555


namespace NUMINAMATH_CALUDE_root_in_interval_l2935_293512

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 5 6 ∧ log10 x = x - 5 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l2935_293512


namespace NUMINAMATH_CALUDE_range_of_c_l2935_293577

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f y < f x

def has_two_distinct_real_roots (a b c : ℝ) := 
  b^2 - 4*a*c > 0

def proposition_p (c : ℝ) := is_decreasing (fun x ↦ c^x)

def proposition_q (c : ℝ) := has_two_distinct_real_roots 1 (2 * Real.sqrt c) (1/2)

theorem range_of_c (c : ℝ) 
  (h1 : c > 0) 
  (h2 : c ≠ 1) 
  (h3 : proposition_p c ∨ proposition_q c) 
  (h4 : ¬(proposition_p c ∧ proposition_q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l2935_293577
