import Mathlib

namespace NUMINAMATH_CALUDE_fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2337_233793

-- Define the concept of a fixed point
def IsFixedPoint (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Define the concept of a stable point
def IsStablePoint (f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def FixedPoints (f : ℝ → ℝ) : Set ℝ := {x | IsFixedPoint f x}

-- Define the set of stable points
def StablePoints (f : ℝ → ℝ) : Set ℝ := {x | IsStablePoint f x}

-- Statement 1: Fixed points are a subset of stable points
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) :
  FixedPoints f ⊆ StablePoints f := by sorry

-- Statement 2: There exists a function with infinitely many stable points
theorem exists_function_with_infinite_stable_points :
  ∃ f : ℝ → ℝ, ¬(Finite (StablePoints f)) := by sorry

-- Statement 3: For monotonically increasing functions, stable points are fixed points
theorem stable_points_are_fixed_points_for_increasing_functions
  (f : ℝ → ℝ) (h : ∀ x y, x < y → f x < f y) :
  ∀ x, IsStablePoint f x → IsFixedPoint f x := by sorry

end NUMINAMATH_CALUDE_fixed_points_subset_stable_points_exists_function_with_infinite_stable_points_stable_points_are_fixed_points_for_increasing_functions_l2337_233793


namespace NUMINAMATH_CALUDE_carpenter_tables_problem_l2337_233755

theorem carpenter_tables_problem (T : ℕ) : 
  T + (T - 3) = 17 → T = 10 := by sorry

end NUMINAMATH_CALUDE_carpenter_tables_problem_l2337_233755


namespace NUMINAMATH_CALUDE_brothers_age_sum_l2337_233745

theorem brothers_age_sum : 
  ∀ (older_age younger_age : ℕ),
  younger_age = 27 →
  younger_age = older_age / 3 + 10 →
  older_age + younger_age = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l2337_233745


namespace NUMINAMATH_CALUDE_negation_of_implication_l2337_233719

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2337_233719


namespace NUMINAMATH_CALUDE_dans_initial_marbles_l2337_233724

/-- The number of marbles Dan gave to Mary -/
def marbles_given : ℕ := 14

/-- The number of marbles Dan has now -/
def marbles_remaining : ℕ := 50

/-- The initial number of marbles Dan had -/
def initial_marbles : ℕ := marbles_given + marbles_remaining

theorem dans_initial_marbles : initial_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_marbles_l2337_233724


namespace NUMINAMATH_CALUDE_andrena_debelyn_difference_l2337_233785

/-- Represents the number of dolls each person has -/
structure DollCount where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- The initial doll counts before any transfers -/
def initial_count : DollCount :=
  { debelyn := 20, christel := 24, andrena := 0 }

/-- The number of dolls transferred from Debelyn to Andrena -/
def debelyn_transfer : ℕ := 2

/-- The number of dolls transferred from Christel to Andrena -/
def christel_transfer : ℕ := 5

/-- The final doll counts after transfers -/
def final_count : DollCount :=
  { debelyn := initial_count.debelyn - debelyn_transfer,
    christel := initial_count.christel - christel_transfer,
    andrena := initial_count.andrena + debelyn_transfer + christel_transfer }

/-- Andrena has 2 more dolls than Christel after transfers -/
axiom andrena_christel_difference : final_count.andrena = final_count.christel + 2

/-- The theorem to be proved -/
theorem andrena_debelyn_difference :
  final_count.andrena - final_count.debelyn = 3 := by
  sorry

end NUMINAMATH_CALUDE_andrena_debelyn_difference_l2337_233785


namespace NUMINAMATH_CALUDE_junior_score_l2337_233708

theorem junior_score (n : ℕ) (h : n > 0) : 
  let junior_count : ℝ := 0.2 * n
  let senior_count : ℝ := 0.8 * n
  let total_score : ℝ := 85 * n
  let senior_score : ℝ := 82 * senior_count
  let junior_total_score : ℝ := total_score - senior_score
  junior_total_score / junior_count = 97 := by sorry

end NUMINAMATH_CALUDE_junior_score_l2337_233708


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l2337_233765

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 1 / (x - 1)) → x > 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l2337_233765


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2337_233764

/-- Given an article with cost C, selling price S, profit factor p, and margin M,
    prove that M can be expressed in terms of S as (p+n)S / (n(2n + p)). -/
theorem margin_in_terms_of_selling_price
  (C S : ℝ) (p n : ℝ) (h_pos : n > 0)
  (h_margin : ∀ M, M = p * (1/n) * C + C)
  (h_selling : S = C + M) :
  ∃ M, M = (p + n) * S / (n * (2 * n + p)) :=
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2337_233764


namespace NUMINAMATH_CALUDE_tomatoes_left_l2337_233716

theorem tomatoes_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 21 → 
  eaten_fraction = 1/3 →
  left = total - (total * eaten_fraction).floor →
  left = 14 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_left_l2337_233716


namespace NUMINAMATH_CALUDE_secretary_project_hours_l2337_233723

theorem secretary_project_hours (total_hours : ℕ) (ratio_1 ratio_2 ratio_3 ratio_4 : ℕ) :
  total_hours = 2080 →
  ratio_1 = 3 →
  ratio_2 = 5 →
  ratio_3 = 7 →
  ratio_4 = 11 →
  (ratio_1 + ratio_2 + ratio_3 + ratio_4) * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = total_hours →
  ratio_4 * (total_hours / (ratio_1 + ratio_2 + ratio_3 + ratio_4)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_secretary_project_hours_l2337_233723


namespace NUMINAMATH_CALUDE_total_pears_picked_l2337_233718

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l2337_233718


namespace NUMINAMATH_CALUDE_cubic_real_root_l2337_233730

/-- Given a cubic polynomial with real coefficients c and d, 
    if -3 - 4i is a root, then the real root is 25/3 -/
theorem cubic_real_root (c d : ℝ) : 
  (c * (Complex.I ^ 3 + (-3 - 4*Complex.I) ^ 3) + 
   4 * (Complex.I ^ 2 + (-3 - 4*Complex.I) ^ 2) + 
   d * (Complex.I + (-3 - 4*Complex.I)) - 100 = 0) →
  (∃ (x : ℝ), c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = 25/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2337_233730


namespace NUMINAMATH_CALUDE_cube_root_fraction_equality_l2337_233749

theorem cube_root_fraction_equality : 
  (((5 : ℝ) / 6 * 20.25) ^ (1/3 : ℝ)) = (3 * (5 ^ (2/3 : ℝ))) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fraction_equality_l2337_233749


namespace NUMINAMATH_CALUDE_terrier_hush_interval_terrier_hush_interval_is_two_l2337_233751

/-- The interval at which a terrier's owner hushes it, given the following conditions:
  - The poodle barks twice for every one time the terrier barks.
  - The terrier's owner says "hush" six times before the dogs stop barking.
  - The poodle barked 24 times. -/
theorem terrier_hush_interval : ℕ :=
  let poodle_barks : ℕ := 24
  let poodle_to_terrier_ratio : ℕ := 2
  let total_hushes : ℕ := 6
  let terrier_barks : ℕ := poodle_barks / poodle_to_terrier_ratio
  terrier_barks / total_hushes

/-- Proof that the terrier_hush_interval is equal to 2 -/
theorem terrier_hush_interval_is_two : terrier_hush_interval = 2 := by
  sorry

end NUMINAMATH_CALUDE_terrier_hush_interval_terrier_hush_interval_is_two_l2337_233751


namespace NUMINAMATH_CALUDE_zoo_meat_amount_l2337_233700

/-- The amount of meat (in kg) that lasts for a given number of days for a lion and a tiger -/
def meatAmount (lionConsumption tigerConsumption daysLasting : ℕ) : ℕ :=
  (lionConsumption + tigerConsumption) * daysLasting

theorem zoo_meat_amount :
  meatAmount 25 20 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_zoo_meat_amount_l2337_233700


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l2337_233715

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (3*a + 2)^3 - (3*b + 2)^3 = 9*k :=
by sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l2337_233715


namespace NUMINAMATH_CALUDE_functions_equality_l2337_233762

theorem functions_equality (x : ℝ) : 2 * |x| = Real.sqrt (4 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_functions_equality_l2337_233762


namespace NUMINAMATH_CALUDE_archibald_win_percentage_l2337_233727

/-- Calculates the percentage of games won by Archibald given the number of games won by him and his brother -/
def percentage_games_won (archibald_wins : ℕ) (brother_wins : ℕ) : ℚ :=
  (archibald_wins : ℚ) / ((archibald_wins + brother_wins) : ℚ) * 100

/-- Theorem stating that Archibald won 40% of the games -/
theorem archibald_win_percentage :
  percentage_games_won 12 18 = 40 := by
  sorry

end NUMINAMATH_CALUDE_archibald_win_percentage_l2337_233727


namespace NUMINAMATH_CALUDE_fraction_equals_91_when_x_is_3_l2337_233799

theorem fraction_equals_91_when_x_is_3 :
  let x : ℝ := 3
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_91_when_x_is_3_l2337_233799


namespace NUMINAMATH_CALUDE_fiftieth_digit_of_seventh_l2337_233714

/-- The decimal representation of 1/7 as a list of digits -/
def seventhDecimal : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating part in the decimal representation of 1/7 -/
def repeatLength : Nat := 6

/-- The 50th digit after the decimal point in the decimal representation of 1/7 -/
def fiftiethDigit : Nat := seventhDecimal[(50 - 1) % repeatLength]

theorem fiftieth_digit_of_seventh :
  fiftiethDigit = 4 := by sorry

end NUMINAMATH_CALUDE_fiftieth_digit_of_seventh_l2337_233714


namespace NUMINAMATH_CALUDE_range_of_m2_plus_n2_l2337_233725

/-- An increasing function f satisfying f(1-x) + f(1+x) = 0 for all x -/
def IncreasingSymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (1 - x) + f (1 + x) = 0)

theorem range_of_m2_plus_n2 (f : ℝ → ℝ) (m n x : ℝ) 
  (hf : IncreasingSymmetricFunction f) 
  (hm : f m > 0) 
  (hn : f n ≤ 0) 
  (hmn : m^2 + n^2 ≤ x^2) : 
  13 < m^2 + n^2 ∧ m^2 + n^2 < 49 := by
sorry

end NUMINAMATH_CALUDE_range_of_m2_plus_n2_l2337_233725


namespace NUMINAMATH_CALUDE_f_odd_g_even_l2337_233733

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the main property
axiom main_property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y

-- Define f(0) = 0
axiom f_zero : f 0 = 0

-- Define f is not identically zero
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

-- Theorem to prove
theorem f_odd_g_even :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ y : ℝ, g (-y) = g y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_g_even_l2337_233733


namespace NUMINAMATH_CALUDE_marble_count_l2337_233713

theorem marble_count (total : ℕ) (red blue white : ℕ) : 
  total = 108 →
  blue = red / 3 →
  white = blue / 2 →
  red + blue + white = total →
  white < red ∧ white < blue :=
by sorry

end NUMINAMATH_CALUDE_marble_count_l2337_233713


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2337_233792

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = 1 ∨ x = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2337_233792


namespace NUMINAMATH_CALUDE_complex_multiplication_l2337_233744

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (3 - 4*i) * (3 + 4*i) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2337_233744


namespace NUMINAMATH_CALUDE_angle_point_cosine_l2337_233748

/-- Given an angle α and a real number a, proves that if the terminal side of α
    passes through point P(3a, 4) and cos α = -3/5, then a = -1. -/
theorem angle_point_cosine (α : Real) (a : Real) : 
  (∃ r : Real, r > 0 ∧ 3 * a = r * Real.cos α ∧ 4 = r * Real.sin α) → 
  Real.cos α = -3/5 → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_angle_point_cosine_l2337_233748


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2337_233709

theorem solve_linear_equation :
  ∀ x : ℚ, -3 * x - 8 = 4 * x + 3 → x = -11/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2337_233709


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2337_233798

theorem min_value_of_fraction_sum (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : (a - b) * (b - c) * (c - a) = -16) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ x y z : ℝ, 
    x > y → y > z → (x - y) * (y - z) * (z - x) = -16 → 
    1 / (x - y) + 1 / (y - z) - 1 / (z - x) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l2337_233798


namespace NUMINAMATH_CALUDE_rectangle_area_l2337_233732

/-- Given a rectangle PQRS with specified coordinates, prove its area is 40400 -/
theorem rectangle_area (y : ℤ) : 
  let P : ℝ × ℝ := (10, -30)
  let Q : ℝ × ℝ := (2010, 170)
  let S : ℝ × ℝ := (12, y)
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PS := Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2)
  PQ * PS = 40400 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2337_233732


namespace NUMINAMATH_CALUDE_grid_sum_l2337_233746

theorem grid_sum (p q r s : ℕ+) 
  (h_pq : p * q = 6)
  (h_rs : r * s = 8)
  (h_pr : p * r = 4)
  (h_qs : q * s = 12)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  p + q + r + s = 13 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_l2337_233746


namespace NUMINAMATH_CALUDE_inequality_proof_l2337_233788

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2337_233788


namespace NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l2337_233787

theorem quadratic_coefficient_of_equation : ∃ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a = 5 ∧ b = -1 ∧ c = -3 ∧ d = 1 ∧ e = 1 ∧ f = -3) →
  (a - d = 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_of_equation_l2337_233787


namespace NUMINAMATH_CALUDE_total_weekly_batches_l2337_233763

/-- Represents the types of flour --/
inductive FlourType
| Regular
| GlutenFree
| WholeWheat

/-- Represents a day of the week --/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents the flour usage for a single day --/
structure DailyUsage where
  regular : ℕ
  glutenFree : ℕ
  wholeWheat : ℕ
  regularToWholeWheat : ℕ

/-- The number of batches that can be made from one sack of flour --/
def batchesPerSack (t : FlourType) : ℕ :=
  match t with
  | FlourType.Regular => 15
  | FlourType.GlutenFree => 10
  | FlourType.WholeWheat => 12

/-- The conversion rate from regular flour to whole-wheat flour --/
def regularToWholeWheatRate : ℚ := 3/2

/-- The daily flour usage for the week --/
def weekUsage : Day → DailyUsage
| Day.Monday => ⟨4, 3, 2, 0⟩
| Day.Tuesday => ⟨6, 2, 0, 1⟩
| Day.Wednesday => ⟨5, 1, 2, 0⟩
| Day.Thursday => ⟨3, 4, 3, 0⟩
| Day.Friday => ⟨7, 1, 0, 2⟩
| Day.Saturday => ⟨5, 3, 1, 0⟩
| Day.Sunday => ⟨2, 4, 0, 2⟩

/-- Calculates the total number of batches for a given flour type in a week --/
def totalBatches (t : FlourType) : ℕ := sorry

/-- The main theorem: Bruce can make 846 batches of pizza dough in a week --/
theorem total_weekly_batches : (totalBatches FlourType.Regular) + 
                               (totalBatches FlourType.GlutenFree) + 
                               (totalBatches FlourType.WholeWheat) = 846 := sorry

end NUMINAMATH_CALUDE_total_weekly_batches_l2337_233763


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2337_233739

theorem mean_equality_implies_x_value :
  let mean1 := (8 + 12 + 24) / 3
  let mean2 := (16 + x) / 2
  mean1 = mean2 → x = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l2337_233739


namespace NUMINAMATH_CALUDE_students_in_score_range_l2337_233703

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  variance : ℝ
  prob_above_140 : ℝ

/-- Calculates the number of students within a given score range -/
def students_in_range (dist : ScoreDistribution) (lower upper : ℝ) : ℕ :=
  sorry

theorem students_in_score_range (dist : ScoreDistribution) 
  (h1 : dist.total_students = 50)
  (h2 : dist.mean = 120)
  (h3 : dist.prob_above_140 = 0.2) :
  students_in_range dist 100 140 = 30 :=
sorry

end NUMINAMATH_CALUDE_students_in_score_range_l2337_233703


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2337_233704

theorem diophantine_equation_solutions : 
  ∀ x y : ℤ, 5 * x^2 + 5 * x * y + 5 * y^2 = 7 * x + 14 * y ↔ 
  (x = -1 ∧ y = 3) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2337_233704


namespace NUMINAMATH_CALUDE_remainder_calculation_l2337_233731

-- Define the remainder function
def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_calculation : rem (-1/3) (4/7) = 5/21 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l2337_233731


namespace NUMINAMATH_CALUDE_detergent_quarts_in_altered_solution_l2337_233786

/-- Represents the ratio of bleach : detergent : water in a cleaning solution -/
structure CleaningSolution :=
  (bleach : ℚ)
  (detergent : ℚ)
  (water : ℚ)

/-- Calculates the amount of detergent in quarts given the conditions of the problem -/
def calculate_detergent_quarts (original : CleaningSolution) (water_gallons : ℚ) : ℚ :=
  let new_ratio := CleaningSolution.mk 
    (original.bleach * 3) 
    original.detergent
    (original.water / 2)
  let total_parts := new_ratio.bleach + new_ratio.detergent + new_ratio.water
  let detergent_gallons := (new_ratio.detergent / new_ratio.water) * water_gallons
  detergent_gallons * 4

/-- Theorem stating that the altered solution will contain 160 quarts of detergent -/
theorem detergent_quarts_in_altered_solution :
  let original := CleaningSolution.mk 2 25 100
  calculate_detergent_quarts original 80 = 160 := by
  sorry


end NUMINAMATH_CALUDE_detergent_quarts_in_altered_solution_l2337_233786


namespace NUMINAMATH_CALUDE_dog_food_theorem_l2337_233754

/-- The amount of food eaten by Hannah's three dogs -/
def dog_food_problem (first_dog_food second_dog_food third_dog_food : ℝ) : Prop :=
  -- Hannah has three dogs
  -- The first dog eats 1.5 cups of dog food a day
  first_dog_food = 1.5 ∧
  -- The second dog eats twice as much as the first dog
  second_dog_food = 2 * first_dog_food ∧
  -- Hannah prepares 10 cups of dog food in total for her three dogs
  first_dog_food + second_dog_food + third_dog_food = 10 ∧
  -- The difference between the third dog's food and the second dog's food is 2.5 cups
  third_dog_food - second_dog_food = 2.5

theorem dog_food_theorem :
  ∃ (first_dog_food second_dog_food third_dog_food : ℝ),
    dog_food_problem first_dog_food second_dog_food third_dog_food :=
by
  sorry

end NUMINAMATH_CALUDE_dog_food_theorem_l2337_233754


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2337_233760

/-- The polynomial resulting from simplifying 3(x^3 - x^2 + 4) - 5(x^4 - 2x^3 + x - 1) -/
def simplified_polynomial (x : ℝ) : ℝ :=
  -5 * x^4 + 13 * x^3 - 3 * x^2 - 5 * x + 17

/-- The coefficients of the simplified polynomial -/
def coefficients : List ℝ := [-5, 13, -3, -5, 17]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 517 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2337_233760


namespace NUMINAMATH_CALUDE_runner_position_l2337_233721

theorem runner_position (track_circumference : ℝ) (distance_run : ℝ) : 
  track_circumference = 100 →
  distance_run = 10560 →
  ∃ (n : ℕ) (remainder : ℝ), 
    distance_run = n * track_circumference + remainder ∧
    75 < remainder ∧ remainder ≤ 100 :=
by sorry

end NUMINAMATH_CALUDE_runner_position_l2337_233721


namespace NUMINAMATH_CALUDE_length_width_relation_l2337_233774

/-- A rectangle enclosed by a wire -/
structure WireRectangle where
  wireLength : ℝ
  width : ℝ
  length : ℝ
  wireLength_positive : 0 < wireLength
  width_positive : 0 < width
  length_positive : 0 < length
  perimeter_eq_wireLength : 2 * (width + length) = wireLength

/-- The relationship between length and width for a 20-meter wire rectangle -/
theorem length_width_relation (rect : WireRectangle) 
    (h : rect.wireLength = 20) : 
    rect.length = -rect.width + 10 := by
  sorry

end NUMINAMATH_CALUDE_length_width_relation_l2337_233774


namespace NUMINAMATH_CALUDE_triangle_ratio_l2337_233742

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) - b*sin(B) = 4c*sin(C) and cos(A) = -1/4, then b/c = 6 -/
theorem triangle_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l2337_233742


namespace NUMINAMATH_CALUDE_solve_for_q_l2337_233781

theorem solve_for_q (p q : ℝ) 
  (h1 : p > 1)
  (h2 : q > 1)
  (h3 : 1/p + 1/q = 1)
  (h4 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l2337_233781


namespace NUMINAMATH_CALUDE_clock_painting_theorem_l2337_233784

def clock_numbers : ℕ := 12

def paint_interval_a : ℕ := 57
def paint_interval_b : ℕ := 2005

theorem clock_painting_theorem :
  (∃ (painted_numbers : Finset ℕ),
    painted_numbers.card = 4 ∧
    ∀ n : ℕ, n ∈ painted_numbers ↔ n < clock_numbers ∧ ∃ k : ℕ, (paint_interval_a * k) % clock_numbers = n) ∧
  (∀ n : ℕ, n < clock_numbers → ∃ k : ℕ, (paint_interval_b * k) % clock_numbers = n) :=
by sorry

end NUMINAMATH_CALUDE_clock_painting_theorem_l2337_233784


namespace NUMINAMATH_CALUDE_expedition_duration_proof_l2337_233737

theorem expedition_duration_proof (first_expedition : ℕ) 
  (h1 : first_expedition = 3)
  (second_expedition : ℕ) 
  (h2 : second_expedition = first_expedition + 2)
  (third_expedition : ℕ) 
  (h3 : third_expedition = 2 * second_expedition) : 
  (first_expedition + second_expedition + third_expedition) * 7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_expedition_duration_proof_l2337_233737


namespace NUMINAMATH_CALUDE_ten_digit_number_divisibility_l2337_233707

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem ten_digit_number_divisibility (a b : ℕ) :
  a < 10 → b < 10 →
  is_divisible_by_99 (2016 * 10000 + a * 1000 + b * 100 + 2017) →
  a + b = 8 := by sorry

end NUMINAMATH_CALUDE_ten_digit_number_divisibility_l2337_233707


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2337_233741

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 7 * Complex.I
  let z₂ : ℂ := 2 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -90 / 53 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2337_233741


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_5_1000_without_zeros_l2337_233775

theorem exists_number_divisible_by_5_1000_without_zeros : 
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → d ≠ 0 → ∃ k : ℕ, n / 10^k % 10 = d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_5_1000_without_zeros_l2337_233775


namespace NUMINAMATH_CALUDE_product_unit_digit_l2337_233768

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- Define the numbers given in the problem
def a : ℕ := 7858
def b : ℕ := 1086
def c : ℕ := 4582
def d : ℕ := 9783

-- State the theorem
theorem product_unit_digit :
  unitDigit (a * b * c * d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l2337_233768


namespace NUMINAMATH_CALUDE_zebra_sleeps_longer_l2337_233766

/-- Proves that a zebra sleeps 2 hours more per night than a cougar, given the conditions -/
theorem zebra_sleeps_longer (cougar_sleep : ℕ) (total_sleep : ℕ) : 
  cougar_sleep = 4 →
  total_sleep = 70 →
  (total_sleep - 7 * cougar_sleep) / 7 - cougar_sleep = 2 := by
sorry

end NUMINAMATH_CALUDE_zebra_sleeps_longer_l2337_233766


namespace NUMINAMATH_CALUDE_undetermined_disjunction_l2337_233702

theorem undetermined_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (p q : Prop), (¬p ∧ ¬(p ∧ q)) → (p ∨ q) := by
sorry

end NUMINAMATH_CALUDE_undetermined_disjunction_l2337_233702


namespace NUMINAMATH_CALUDE_car_repair_cost_john_car_repair_cost_l2337_233796

/-- Calculates the amount spent on car repairs given savings information -/
theorem car_repair_cost (monthly_savings : ℕ) (savings_months : ℕ) (remaining_amount : ℕ) : ℕ :=
  let total_savings := monthly_savings * savings_months
  total_savings - remaining_amount

/-- Proves that John spent $400 on car repairs -/
theorem john_car_repair_cost : 
  car_repair_cost 25 24 200 = 400 := by
  sorry

end NUMINAMATH_CALUDE_car_repair_cost_john_car_repair_cost_l2337_233796


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l2337_233736

/-- Represents the rental prices and quantities of canoes and kayaks --/
structure RentalInfo where
  canoePrice : ℕ
  kayakPrice : ℕ
  canoeCount : ℕ
  kayakCount : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def totalRevenue (info : RentalInfo) : ℕ :=
  info.canoePrice * info.canoeCount + info.kayakPrice * info.kayakCount

/-- Theorem stating the ratio of canoes to kayaks given the rental conditions --/
theorem canoe_kayak_ratio (info : RentalInfo) :
  info.canoePrice = 15 →
  info.kayakPrice = 18 →
  totalRevenue info = 405 →
  info.canoeCount = info.kayakCount + 5 →
  (info.canoeCount : ℚ) / info.kayakCount = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_canoe_kayak_ratio_l2337_233736


namespace NUMINAMATH_CALUDE_product_ratio_integer_l2337_233712

def divisible_count (seq : List Nat) (d : Nat) : Nat :=
  (seq.filter (fun x => x % d == 0)).length

theorem product_ratio_integer (m n : List Nat) :
  (∀ d : Nat, d > 1 → divisible_count m d ≥ divisible_count n d) →
  m.all (· > 0) →
  n.all (· > 0) →
  n.length > 0 →
  ∃ k : Nat, k > 0 ∧ (m.prod : Int) = k * (n.prod : Int) := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_integer_l2337_233712


namespace NUMINAMATH_CALUDE_operation_does_not_eliminate_variables_l2337_233726

/-- Represents a linear equation ax + by = c -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a system of two linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- Applies the operation ①$-$②$\times 3$ to a linear system -/
def applyOperation (sys : LinearSystem) : LinearEquation :=
  { a := sys.eq1.a - 3 * sys.eq2.a
  , b := sys.eq1.b - 3 * sys.eq2.b
  , c := sys.eq1.c - 3 * sys.eq2.c }

/-- The given system of linear equations -/
def givenSystem : LinearSystem :=
  { eq1 := { a := 1, b := 3, c := 4 }
  , eq2 := { a := 2, b := -1, c := 1 } }

theorem operation_does_not_eliminate_variables :
  let result := applyOperation givenSystem
  result.a ≠ 0 ∧ result.b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_operation_does_not_eliminate_variables_l2337_233726


namespace NUMINAMATH_CALUDE_find_N_l2337_233729

theorem find_N : ∃ N : ℕ+, (22 ^ 2 * 55 ^ 2 : ℕ) = 10 ^ 2 * N ^ 2 ∧ N = 121 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l2337_233729


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2337_233728

/-- Given a geometric sequence where the first term is 1000 and the eighth term is 125,
    prove that the sixth term is 31.25. -/
theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1000)  -- First term is 1000
  (h2 : a 8 = 125)   -- Eighth term is 125
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 6 = 31.25 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2337_233728


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2337_233791

/-- A parallelogram with adjacent sides of length 3 and 5 has a perimeter of 16. -/
theorem parallelogram_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) :
  2 * (a + b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2337_233791


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l2337_233776

/-- Represents the state of the game --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Bool) : GameState :=
  if move
  then { points := state.points * 2, rubles := state.rubles + 2 }
  else { points := state.points + 1, rubles := state.rubles + 1 }

/-- Checks if the game state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Bool :=
  state.points <= 50

/-- Checks if the game is won (exactly 50 points) --/
def isWinningState (state : GameState) : Bool :=
  state.points = 50

/-- Theorem: The minimum number of rubles to win the game is 11 --/
theorem min_rubles_to_win :
  ∃ (moves : List Bool),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Bool),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l2337_233776


namespace NUMINAMATH_CALUDE_ben_catch_count_l2337_233795

/-- The number of fish caught by each family member (except Ben) --/
def family_catch : Fin 4 → ℕ
| 0 => 1  -- Judy
| 1 => 3  -- Billy
| 2 => 2  -- Jim
| 3 => 5  -- Susie

/-- The total number of filets they will have --/
def total_filets : ℕ := 24

/-- The number of fish thrown back --/
def thrown_back : ℕ := 3

/-- The number of filets per fish --/
def filets_per_fish : ℕ := 2

theorem ben_catch_count :
  ∃ (ben_catch : ℕ),
    ben_catch = total_filets / filets_per_fish + thrown_back - (family_catch 0 + family_catch 1 + family_catch 2 + family_catch 3) ∧
    ben_catch = 4 := by
  sorry

end NUMINAMATH_CALUDE_ben_catch_count_l2337_233795


namespace NUMINAMATH_CALUDE_problem_solution_l2337_233771

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y + 6 = 10) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2337_233771


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2337_233790

theorem binomial_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  let p := 2 * k - 1
  Prime p →
  p ∣ (n.choose 2 - k.choose 2) →
  p^2 ∣ (n.choose 2 - k.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2337_233790


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2337_233717

/-- Calculates the speed of trains given their length, crossing time, and direction --/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 →
  crossing_time = 12 →
  (2 * train_length) / crossing_time * 3.6 = 36 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2337_233717


namespace NUMINAMATH_CALUDE_dice_cube_surface_area_l2337_233759

theorem dice_cube_surface_area (num_dice : ℕ) (die_side_length : ℝ) (h1 : num_dice = 27) (h2 : die_side_length = 3) :
  let edge_length : ℝ := die_side_length * (num_dice ^ (1/3 : ℝ))
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 486 :=
by sorry

end NUMINAMATH_CALUDE_dice_cube_surface_area_l2337_233759


namespace NUMINAMATH_CALUDE_continuous_compound_interest_rate_l2337_233722

/-- Continuous compound interest rate calculation -/
theorem continuous_compound_interest_rate 
  (P : ℝ) -- Principal amount
  (A : ℝ) -- Total amount after interest
  (t : ℝ) -- Time in years
  (h1 : P = 600)
  (h2 : A = 760)
  (h3 : t = 4)
  : ∃ r : ℝ, (A = P * Real.exp (r * t)) ∧ (abs (r - 0.05909725) < 0.00000001) :=
sorry

end NUMINAMATH_CALUDE_continuous_compound_interest_rate_l2337_233722


namespace NUMINAMATH_CALUDE_tan_cos_sum_identity_l2337_233750

theorem tan_cos_sum_identity : 
  Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.tan (45 * π / 180) * Real.cos (30 * π / 180) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_sum_identity_l2337_233750


namespace NUMINAMATH_CALUDE_complex_exponential_identity_l2337_233711

theorem complex_exponential_identity :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^40 = Complex.exp (Complex.I * Real.pi * (40 / 180) * (-1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_identity_l2337_233711


namespace NUMINAMATH_CALUDE_max_m_value_l2337_233794

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_m_value : 
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) ∧ 
    (∀ (m' : ℝ), m' > m → 
      ¬(∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m' → f (x + t) ≤ x))) ∧
  (∀ (m : ℝ), 
    (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 1 m → f (x + t) ≤ x) → 
    m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l2337_233794


namespace NUMINAMATH_CALUDE_equation_equivalence_l2337_233752

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 4 / y = 1 / 3) ↔ (9 * y / (y - 12) = x) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2337_233752


namespace NUMINAMATH_CALUDE_fourth_root_power_eight_l2337_233778

theorem fourth_root_power_eight : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_power_eight_l2337_233778


namespace NUMINAMATH_CALUDE_factorization_bound_l2337_233710

/-- The number of ways to factorize k into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n > 1 and any prime factor p of n,
    the number of ways to factorize n is less than or equal to n/p -/
theorem factorization_bound (n : ℕ) (p : ℕ) (h1 : n > 1) (h2 : Nat.Prime p) (h3 : p ∣ n) :
  f n ≤ n / p :=
sorry

end NUMINAMATH_CALUDE_factorization_bound_l2337_233710


namespace NUMINAMATH_CALUDE_field_trip_students_l2337_233756

theorem field_trip_students (van_capacity : ℕ) (num_adults : ℕ) (num_vans : ℕ) : 
  van_capacity = 5 → num_adults = 5 → num_vans = 6 → 
  (num_vans * van_capacity - num_adults : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l2337_233756


namespace NUMINAMATH_CALUDE_principal_booking_l2337_233779

/-- The number of rooms needed to accommodate a class on a field trip -/
def rooms_needed (total_students : ℕ) (students_per_room : ℕ) : ℕ :=
  (total_students + students_per_room - 1) / students_per_room

/-- Theorem: The principal needs to book 6 rooms for 30 students -/
theorem principal_booking : 
  let total_students : ℕ := 30
  let queen_bed_capacity : ℕ := 2
  let pullout_couch_capacity : ℕ := 1
  let room_capacity : ℕ := 2 * queen_bed_capacity + pullout_couch_capacity
  rooms_needed total_students room_capacity = 6 := by
sorry

end NUMINAMATH_CALUDE_principal_booking_l2337_233779


namespace NUMINAMATH_CALUDE_special_cone_vertex_angle_l2337_233701

/-- A right circular cone with three pairwise perpendicular generatrices -/
structure SpecialCone where
  /-- The angle at the vertex of the axial section -/
  vertex_angle : ℝ
  /-- The condition that three generatrices are pairwise perpendicular -/
  perpendicular_generatrices : Prop

/-- Theorem: The angle at the vertex of the axial section of a special cone is 2 * arcsin(√6 / 3) -/
theorem special_cone_vertex_angle (cone : SpecialCone) :
  cone.perpendicular_generatrices →
  cone.vertex_angle = 2 * Real.arcsin (Real.sqrt 6 / 3) := by
  sorry

end NUMINAMATH_CALUDE_special_cone_vertex_angle_l2337_233701


namespace NUMINAMATH_CALUDE_range_of_f_l2337_233789

def f (x : ℤ) : ℤ := x^2 - 2*x

def domain : Set ℤ := {x : ℤ | -2 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3, 8} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2337_233789


namespace NUMINAMATH_CALUDE_storks_on_fence_l2337_233747

/-- The number of storks initially on the fence -/
def initial_storks : ℕ := 4

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

/-- The total number of birds and storks after additional storks joined -/
def total_after : ℕ := 13

theorem storks_on_fence :
  initial_birds + initial_storks + additional_storks = total_after :=
by sorry

end NUMINAMATH_CALUDE_storks_on_fence_l2337_233747


namespace NUMINAMATH_CALUDE_marks_garden_flowers_l2337_233782

theorem marks_garden_flowers (yellow : ℕ) (purple : ℕ) (green : ℕ) 
  (h1 : yellow = 10)
  (h2 : purple = yellow + yellow * 4 / 5)
  (h3 : green = (yellow + purple) / 4) :
  yellow + purple + green = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_garden_flowers_l2337_233782


namespace NUMINAMATH_CALUDE_relay_race_theorem_l2337_233757

def relay_race_length (team_size : ℕ) (standard_distance : ℝ) (long_distance_multiplier : ℝ) : ℝ :=
  (team_size - 1) * standard_distance + long_distance_multiplier * standard_distance

theorem relay_race_theorem :
  relay_race_length 5 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_theorem_l2337_233757


namespace NUMINAMATH_CALUDE_not_perfect_square_polynomial_l2337_233797

theorem not_perfect_square_polynomial (n : ℕ) : ¬∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_polynomial_l2337_233797


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2337_233738

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 5*x + 6 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 5 ∧ s₁ * s₂ = 6 ∧ s₁^2 + s₂^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2337_233738


namespace NUMINAMATH_CALUDE_circle_C_properties_l2337_233777

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Define the line x - y - 1 = 0
def tangent_line (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem stating the properties of circle C
theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) ∧
  (3, 0) ∈ circle_C ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) :=
by
  sorry

#check circle_C_properties

end NUMINAMATH_CALUDE_circle_C_properties_l2337_233777


namespace NUMINAMATH_CALUDE_triangle_is_acute_l2337_233783

theorem triangle_is_acute (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) :
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2 := by
  sorry

#check triangle_is_acute

end NUMINAMATH_CALUDE_triangle_is_acute_l2337_233783


namespace NUMINAMATH_CALUDE_distribution_schemes_count_l2337_233758

/-- The number of ways to distribute 3 people to 7 communities with at most 2 people per community -/
def distribution_schemes : ℕ := sorry

/-- The number of ways to choose 3 communities out of 7 -/
def three_single_communities : ℕ := sorry

/-- The number of ways to choose 2 communities out of 7 and distribute 3 people -/
def one_double_one_single : ℕ := sorry

theorem distribution_schemes_count :
  distribution_schemes = three_single_communities + one_double_one_single ∧
  distribution_schemes = 336 := by sorry

end NUMINAMATH_CALUDE_distribution_schemes_count_l2337_233758


namespace NUMINAMATH_CALUDE_sphere_radius_from_intersection_l2337_233767

theorem sphere_radius_from_intersection (width depth : ℝ) (h_width : width = 30) (h_depth : depth = 10) :
  let r := Real.sqrt ((width / 2) ^ 2 + (width / 4 + depth) ^ 2)
  ∃ ε > 0, abs (r - 22.1129) < ε :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_intersection_l2337_233767


namespace NUMINAMATH_CALUDE_quadratic_root_form_l2337_233770

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 6*x + d = 0 ↔ x = (-6 + Real.sqrt d) / 2 ∨ x = (-6 - Real.sqrt d) / 2) →
  d = 36/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_form_l2337_233770


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2337_233780

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2337_233780


namespace NUMINAMATH_CALUDE_equation_solution_l2337_233772

theorem equation_solution (x : ℝ) : 
  (x + 1)^5 + (x + 1)^4 * (x - 1) + (x + 1)^3 * (x - 1)^2 + 
  (x + 1)^2 * (x - 1)^3 + (x + 1) * (x - 1)^4 + (x - 1)^5 = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2337_233772


namespace NUMINAMATH_CALUDE_bridge_length_l2337_233735

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length ∧
    bridge_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2337_233735


namespace NUMINAMATH_CALUDE_equation_solutions_l2337_233773

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 + 6 * x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x + 2)^2 = 3 * (x + 2)
  let sol1 : Set ℝ := {(-3 + Real.sqrt 3) / 2, (-3 - Real.sqrt 3) / 2}
  let sol2 : Set ℝ := {-2, 1}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2337_233773


namespace NUMINAMATH_CALUDE_total_investment_proof_l2337_233720

def bank_investment : ℝ := 6000
def bond_investment : ℝ := 6000
def bank_interest_rate : ℝ := 0.05
def bond_return_rate : ℝ := 0.09
def annual_income : ℝ := 660

theorem total_investment_proof :
  bank_investment + bond_investment = 12000 :=
by sorry

end NUMINAMATH_CALUDE_total_investment_proof_l2337_233720


namespace NUMINAMATH_CALUDE_power_multiplication_simplification_l2337_233705

theorem power_multiplication_simplification :
  let a : ℝ := 0.25
  let b : ℝ := -4
  let n : ℕ := 16
  let m : ℕ := 17
  (a ^ n) * (b ^ m) = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_simplification_l2337_233705


namespace NUMINAMATH_CALUDE_resort_cost_theorem_l2337_233753

def resort_problem (swimming_pool_cost : ℝ) : Prop :=
  let first_cabin_cost := swimming_pool_cost
  let second_cabin_cost := first_cabin_cost / 2
  let third_cabin_cost := second_cabin_cost / 3
  let land_cost := 4 * swimming_pool_cost
  swimming_pool_cost + first_cabin_cost + second_cabin_cost + third_cabin_cost + land_cost = 150000

theorem resort_cost_theorem :
  ∃ (swimming_pool_cost : ℝ), resort_problem swimming_pool_cost :=
sorry

end NUMINAMATH_CALUDE_resort_cost_theorem_l2337_233753


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l2337_233734

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 30, green := 25, yellow := 25, blue := 18, white := 15, black := 12 }

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_for_twenty_of_one_color :
    minBallsForColor problemCounts 20 = 103 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l2337_233734


namespace NUMINAMATH_CALUDE_math_class_grade_distribution_l2337_233743

theorem math_class_grade_distribution (total_students : ℕ) 
  (prob_A : ℚ) (prob_B : ℚ) (prob_C : ℚ) : 
  total_students = 40 →
  prob_A = 0.8 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  ∃ (num_B : ℕ), num_B = 13 ∧ 
    (↑num_B : ℚ) * prob_B = (total_students : ℚ) * prob_B := by
  sorry

end NUMINAMATH_CALUDE_math_class_grade_distribution_l2337_233743


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l2337_233706

theorem maintenance_check_increase (old_time new_time : ℝ) (h1 : old_time = 45) (h2 : new_time = 60) :
  (new_time - old_time) / old_time * 100 = 33.33 := by
sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l2337_233706


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l2337_233769

theorem lucky_larry_problem (a b c d e : ℤ) : 
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 →
  (a - (b - (c - (d + e))) = a - b - c - d + e) →
  e = 3 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l2337_233769


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l2337_233761

theorem sqrt_two_squared : (Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l2337_233761


namespace NUMINAMATH_CALUDE_quadratic_real_roots_when_ac_negative_l2337_233740

theorem quadratic_real_roots_when_ac_negative 
  (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_when_ac_negative_l2337_233740
