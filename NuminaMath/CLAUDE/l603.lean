import Mathlib

namespace NUMINAMATH_CALUDE_min_value_quadratic_l603_60341

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 3*y^2 - 8*x + 12*y + 40 ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l603_60341


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l603_60381

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_coordinates :
  ∀ (m n : ℝ), symmetric_about_origin (m, 4) (-2, n) → m = 2 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l603_60381


namespace NUMINAMATH_CALUDE_special_sequence_250th_term_l603_60359

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Predicate to check if a number is a multiple of 3 -/
def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 3 * m

/-- The sequence of positive integers omitting perfect squares and multiples of 3 -/
def special_sequence : ℕ → ℕ :=
  sorry

/-- The 250th term of the special sequence is 350 -/
theorem special_sequence_250th_term :
  special_sequence 250 = 350 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_250th_term_l603_60359


namespace NUMINAMATH_CALUDE_dental_cleaning_theorem_l603_60355

/-- Represents the number of teeth for different animals --/
structure AnimalTeeth where
  dog : ℕ
  cat : ℕ
  pig : ℕ

/-- Represents the number of animals to be cleaned --/
structure AnimalsToClean where
  dogs : ℕ
  cats : ℕ
  pigs : ℕ

/-- Calculates the total number of teeth cleaned --/
def totalTeethCleaned (teeth : AnimalTeeth) (animals : AnimalsToClean) : ℕ :=
  teeth.dog * animals.dogs + teeth.cat * animals.cats + teeth.pig * animals.pigs

/-- Theorem stating that given the conditions, 5 dogs result in 706 teeth cleaned --/
theorem dental_cleaning_theorem (teeth : AnimalTeeth) (animals : AnimalsToClean) :
  teeth.dog = 42 →
  teeth.cat = 30 →
  teeth.pig = 28 →
  animals.cats = 10 →
  animals.pigs = 7 →
  totalTeethCleaned teeth { dogs := 5, cats := animals.cats, pigs := animals.pigs } = 706 :=
by
  sorry

#check dental_cleaning_theorem

end NUMINAMATH_CALUDE_dental_cleaning_theorem_l603_60355


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l603_60303

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x : ℝ, x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l603_60303


namespace NUMINAMATH_CALUDE_r_six_times_thirty_l603_60357

/-- The function r as defined in the problem -/
def r (θ : ℚ) : ℚ := 1 / (2 - θ)

/-- The composition of r with itself n times -/
def r_n (n : ℕ) (θ : ℚ) : ℚ :=
  match n with
  | 0 => θ
  | n + 1 => r (r_n n θ)

/-- The main theorem stating that applying r six times to 30 results in 22/23 -/
theorem r_six_times_thirty : r_n 6 30 = 22 / 23 := by
  sorry

end NUMINAMATH_CALUDE_r_six_times_thirty_l603_60357


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l603_60306

def f (x : ℝ) := x^2

theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l603_60306


namespace NUMINAMATH_CALUDE_quadratic_factorization_l603_60340

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x, x^2 + 11*x + 28 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l603_60340


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l603_60386

theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (y = 1 - x ∧ 0 ≤ x ∧ x ≤ 1) ↔
  (ρ = 1 / (Real.cos θ + Real.sin θ) ∧ 0 ≤ θ ∧ θ ≤ Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l603_60386


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l603_60344

theorem tangent_triangle_area (a : ℝ) : 
  a > 0 → 
  (1/2 * a/2 * a^2 = 2) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_triangle_area_l603_60344


namespace NUMINAMATH_CALUDE_square_exterior_points_distance_l603_60395

/-- Given a square ABCD with side length 10 and exterior points E and F,
    prove that EF^2 = 850 + 250√125 when BE = DF = 7 and AE = CF = 15 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 10
  let be_df_length : ℝ := 7
  let ae_cf_length : ℝ := 15
  -- Square ABCD definition
  (A = (0, side_length) ∧ 
   B = (side_length, side_length) ∧ 
   C = (side_length, 0) ∧ 
   D = (0, 0)) →
  -- E and F are exterior points
  (E.1 > side_length ∧ E.2 = side_length) →
  (F.1 = 0 ∧ F.2 < 0) →
  -- BE and DF lengths
  (dist B E = be_df_length ∧ dist D F = be_df_length) →
  -- AE and CF lengths
  (dist A E = ae_cf_length ∧ dist C F = ae_cf_length) →
  -- Conclusion: EF^2 = 850 + 250√125
  dist E F ^ 2 = 850 + 250 * Real.sqrt 125 :=
by sorry


end NUMINAMATH_CALUDE_square_exterior_points_distance_l603_60395


namespace NUMINAMATH_CALUDE_emily_card_collection_l603_60305

/-- Emily's card collection problem -/
theorem emily_card_collection (initial_cards : ℕ) (additional_cards : ℕ) :
  initial_cards = 63 → additional_cards = 7 → initial_cards + additional_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_emily_card_collection_l603_60305


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l603_60383

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l603_60383


namespace NUMINAMATH_CALUDE_joanne_part_time_hours_l603_60342

/-- Calculates the number of hours Joanne works at her part-time job each day -/
def part_time_hours_per_day (main_job_hourly_rate : ℚ) (main_job_hours_per_day : ℚ) 
  (part_time_hourly_rate : ℚ) (days_per_week : ℚ) (total_weekly_earnings : ℚ) : ℚ :=
  let main_job_daily_earnings := main_job_hourly_rate * main_job_hours_per_day
  let main_job_weekly_earnings := main_job_daily_earnings * days_per_week
  let part_time_weekly_earnings := total_weekly_earnings - main_job_weekly_earnings
  let part_time_weekly_hours := part_time_weekly_earnings / part_time_hourly_rate
  part_time_weekly_hours / days_per_week

theorem joanne_part_time_hours : 
  part_time_hours_per_day 16 8 (27/2) 5 775 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joanne_part_time_hours_l603_60342


namespace NUMINAMATH_CALUDE_find_a_l603_60363

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem find_a : ∃ (a : ℝ), A ∩ B a = {3} → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l603_60363


namespace NUMINAMATH_CALUDE_min_sum_a_b_l603_60361

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 2*d = 0) →
  (∃ x : ℝ, x^2 + 2*d*x + c = 0) →
  c + d ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l603_60361


namespace NUMINAMATH_CALUDE_erasers_per_friend_l603_60394

/-- Given 9306 erasers shared among 99 friends, prove that each friend receives 94 erasers. -/
theorem erasers_per_friend :
  let total_erasers : ℕ := 9306
  let num_friends : ℕ := 99
  let erasers_per_friend : ℕ := total_erasers / num_friends
  erasers_per_friend = 94 := by sorry

end NUMINAMATH_CALUDE_erasers_per_friend_l603_60394


namespace NUMINAMATH_CALUDE_maggie_bought_ten_magazines_l603_60396

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The number of books Maggie bought -/
def num_books : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

/-- Proof that Maggie bought 10 science magazines -/
theorem maggie_bought_ten_magazines :
  num_magazines = 10 ∧
  num_books * book_cost + num_magazines * magazine_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_maggie_bought_ten_magazines_l603_60396


namespace NUMINAMATH_CALUDE_equation_solution_l603_60371

theorem equation_solution : ∃ x : ℝ, (x - 5) ^ 4 = (1 / 16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l603_60371


namespace NUMINAMATH_CALUDE_only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l603_60311

-- Define the type for L-like shapes
inductive LLikeShape
| Shape1
| Shape2
| Shape3
| Shape4
| Shape5

-- Define a function to check if a shape is symmetric to the original
def isSymmetric (shape : LLikeShape) : Prop :=
  match shape with
  | LLikeShape.Shape3 => True
  | _ => False

-- Theorem stating that only Shape3 is symmetric
theorem only_shape3_symmetric :
  ∀ (shape : LLikeShape), isSymmetric shape ↔ shape = LLikeShape.Shape3 :=
by sorry

-- Theorem stating that Shape3 is indeed symmetric
theorem shape3_is_symmetric : isSymmetric LLikeShape.Shape3 :=
by sorry

-- Theorem stating that other shapes are not symmetric
theorem other_shapes_not_symmetric :
  ∀ (shape : LLikeShape), shape ≠ LLikeShape.Shape3 → ¬(isSymmetric shape) :=
by sorry

end NUMINAMATH_CALUDE_only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l603_60311


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l603_60356

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l603_60356


namespace NUMINAMATH_CALUDE_domino_trick_l603_60318

theorem domino_trick (x y : ℕ) (hx : x ≤ 6) (hy : y ≤ 6) :
  10 * x + y + 30 = 62 → x = 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_domino_trick_l603_60318


namespace NUMINAMATH_CALUDE_crates_lost_l603_60322

/-- Proves the number of crates lost given initial conditions --/
theorem crates_lost (initial_crates : ℕ) (total_cost : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) : 
  initial_crates = 10 →
  total_cost = 160 →
  selling_price = 25 →
  profit_percentage = 25 / 100 →
  ∃ (lost_crates : ℕ), lost_crates = 2 ∧ 
    selling_price * (initial_crates - lost_crates) = total_cost * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_crates_lost_l603_60322


namespace NUMINAMATH_CALUDE_business_profit_share_l603_60320

/-- Calculates the profit share of a partner given the total capital, partner's capital, and total profit -/
def profitShare (totalCapital : ℚ) (partnerCapital : ℚ) (totalProfit : ℚ) : ℚ :=
  (partnerCapital / totalCapital) * totalProfit

theorem business_profit_share 
  (capitalA capitalB capitalC : ℚ)
  (profitDifferenceAC : ℚ)
  (h1 : capitalA = 8000)
  (h2 : capitalB = 10000)
  (h3 : capitalC = 12000)
  (h4 : profitDifferenceAC = 760) :
  ∃ (totalProfit : ℚ), 
    profitShare (capitalA + capitalB + capitalC) capitalB totalProfit = 1900 :=
by
  sorry

#check business_profit_share

end NUMINAMATH_CALUDE_business_profit_share_l603_60320


namespace NUMINAMATH_CALUDE_mom_tshirt_count_l603_60307

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end NUMINAMATH_CALUDE_mom_tshirt_count_l603_60307


namespace NUMINAMATH_CALUDE_always_available_official_l603_60313

/-- Represents the state of the tennis tournament after n matches. -/
structure TournamentState (n : ℕ) where
  /-- The number of eliminated players after n matches -/
  eliminated : ℕ
  /-- The number of players who have officiated a match -/
  officiated : ℕ
  /-- The first match was officiated by an invited referee -/
  first_match_external : eliminated = n
  /-- Each subsequent match is officiated by an eliminated player -/
  officiated_by_eliminated : officiated ≤ eliminated - 1

/-- There is always someone available to officiate the next match -/
theorem always_available_official (n : ℕ) (state : TournamentState n) :
  ∃ m : ℕ, m < state.eliminated ∧ m ≥ state.officiated :=
sorry

end NUMINAMATH_CALUDE_always_available_official_l603_60313


namespace NUMINAMATH_CALUDE_choose_positions_count_l603_60332

def num_people : ℕ := 6
def num_positions : ℕ := 3

theorem choose_positions_count :
  (num_people.factorial) / ((num_people - num_positions).factorial) = 120 :=
sorry

end NUMINAMATH_CALUDE_choose_positions_count_l603_60332


namespace NUMINAMATH_CALUDE_emma_account_balance_l603_60352

def remaining_balance (initial_balance : ℕ) (daily_spending : ℕ) (days : ℕ) (bill_denomination : ℕ) : ℕ :=
  let balance_after_spending := initial_balance - daily_spending * days
  let withdrawal_amount := (balance_after_spending / bill_denomination) * bill_denomination
  balance_after_spending - withdrawal_amount

theorem emma_account_balance :
  remaining_balance 100 8 7 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emma_account_balance_l603_60352


namespace NUMINAMATH_CALUDE_last_digit_of_even_ten_digit_with_sum_89_l603_60343

/-- A ten-digit integer -/
def TenDigitInt : Type := { n : ℕ // 1000000000 ≤ n ∧ n < 10000000000 }

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem last_digit_of_even_ten_digit_with_sum_89 (n : TenDigitInt) 
  (h_even : Even n.val)
  (h_sum : sum_of_digits n.val = 89) :
  n.val % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_even_ten_digit_with_sum_89_l603_60343


namespace NUMINAMATH_CALUDE_inflation_cost_increase_l603_60391

def original_lumber_cost : ℝ := 450
def original_nails_cost : ℝ := 30
def original_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def total_increased_cost : ℝ :=
  (original_lumber_cost * lumber_inflation_rate) +
  (original_nails_cost * nails_inflation_rate) +
  (original_fabric_cost * fabric_inflation_rate)

theorem inflation_cost_increase :
  total_increased_cost = 97 := by sorry

end NUMINAMATH_CALUDE_inflation_cost_increase_l603_60391


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l603_60339

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : abs (z + 2 + 4*I) = 2)
  (hw : abs (w - 6 - 7*I) = 4) :
  ∃ (d : ℝ), d = Real.sqrt 185 - 6 ∧ ∀ (z' w' : ℂ), 
    abs (z' + 2 + 4*I) = 2 → abs (w' - 6 - 7*I) = 4 → 
    abs (z' - w') ≥ d ∧ ∃ (z'' w'' : ℂ), abs (z'' - w'') = d :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l603_60339


namespace NUMINAMATH_CALUDE_distance_between_points_l603_60302

/-- The distance between points (2,5) and (7,1) is √41. -/
theorem distance_between_points : Real.sqrt 41 = Real.sqrt ((7 - 2)^2 + (1 - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l603_60302


namespace NUMINAMATH_CALUDE_add_36_15_l603_60358

theorem add_36_15 : 36 + 15 = 51 := by
  sorry

end NUMINAMATH_CALUDE_add_36_15_l603_60358


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l603_60360

theorem square_sum_equals_eight (a b c : ℝ) 
  (sum_condition : a + b + c = 4)
  (product_sum_condition : a * b + b * c + a * c = 4) :
  a^2 + b^2 + c^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l603_60360


namespace NUMINAMATH_CALUDE_rationalize_denominator_l603_60385

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l603_60385


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l603_60354

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: B ⊆ A ⇔ m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ ⇔ m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l603_60354


namespace NUMINAMATH_CALUDE_unclaimed_candy_fraction_l603_60377

/-- Represents the order of arrival -/
inductive Participant
| Charlie
| Alice
| Bob

/-- The fraction of candy each participant should receive based on the 4:3:2 ratio -/
def intended_share (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 4/9
  | Participant.Bob => 1/3

/-- The actual amount of candy taken by each participant -/
def actual_take (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 28/81
  | Participant.Bob => 17/81

theorem unclaimed_candy_fraction :
  1 - (actual_take Participant.Charlie + actual_take Participant.Alice + actual_take Participant.Bob) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_fraction_l603_60377


namespace NUMINAMATH_CALUDE_complex_modulus_product_l603_60392

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l603_60392


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l603_60380

-- Problem 1
theorem solve_equation_1 : 
  let f : ℝ → ℝ := λ x => -3*x*(2*x-3)+(2*x-3)
  ∃ x₁ x₂ : ℝ, x₁ = 3/2 ∧ x₂ = 1/3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 2
theorem solve_equation_2 :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 5 ∧ x₂ = 3 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 3
theorem solve_equation_3 :
  let f : ℝ → ℝ := λ x => 4 / (x^2 - 4) + 1 / (x - 2) + 1
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l603_60380


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l603_60309

def ellipse (x y : ℝ) : Prop := y^2/16 + x^2/4 = 1

theorem max_value_on_ellipse :
  ∃ (M : ℝ), ∀ (x y : ℝ), ellipse x y → |2*Real.sqrt 3*x + y - 1| ≤ M ∧
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ |2*Real.sqrt 3*x₀ + y₀ - 1| = M :=
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l603_60309


namespace NUMINAMATH_CALUDE_vector_triangle_l603_60329

/-- Given vectors a and b, if 4a, 3b - 2a, and c form a triangle, then c = (4, -6) -/
theorem vector_triangle (a b c : ℝ × ℝ) : 
  a = (1, -3) → 
  b = (-2, 4) → 
  4 • a + (3 • b - 2 • a) + c = (0, 0) → 
  c = (4, -6) := by
sorry

end NUMINAMATH_CALUDE_vector_triangle_l603_60329


namespace NUMINAMATH_CALUDE_restaurant_tip_percentage_l603_60397

theorem restaurant_tip_percentage : 
  let james_meal : ℚ := 16
  let friend_meal : ℚ := 14
  let total_bill : ℚ := james_meal + friend_meal
  let james_paid : ℚ := 21
  let friend_paid : ℚ := total_bill / 2
  let tip : ℚ := james_paid - friend_paid
  tip / total_bill = 1/5 := by sorry

end NUMINAMATH_CALUDE_restaurant_tip_percentage_l603_60397


namespace NUMINAMATH_CALUDE_remainder_777_power_777_mod_13_l603_60323

theorem remainder_777_power_777_mod_13 : 777^777 ≡ 12 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_power_777_mod_13_l603_60323


namespace NUMINAMATH_CALUDE_inequality_proof_l603_60372

theorem inequality_proof (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  0 < (x - Real.sin x) / (Real.tan x - Real.sin x) ∧
  (x - Real.sin x) / (Real.tan x - Real.sin x) < 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l603_60372


namespace NUMINAMATH_CALUDE_cubic_sum_equals_265_l603_60310

theorem cubic_sum_equals_265 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_265_l603_60310


namespace NUMINAMATH_CALUDE_number_problem_l603_60324

theorem number_problem : ∃ x : ℝ, x = 40 ∧ 0.8 * x > (4/5) * 25 + 12 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l603_60324


namespace NUMINAMATH_CALUDE_payment_calculation_l603_60331

theorem payment_calculation (rate : ℚ) (rooms : ℚ) : 
  rate = 13/3 → rooms = 8/5 → rate * rooms = 104/15 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l603_60331


namespace NUMINAMATH_CALUDE_sphere_surface_area_l603_60336

theorem sphere_surface_area (r : ℝ) (h : (4 / 3) * π * r^3 = 72 * π) :
  4 * π * r^2 = 36 * π * 2^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l603_60336


namespace NUMINAMATH_CALUDE_problem_solution_l603_60378

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 15 →
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 3970049 / 36000 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l603_60378


namespace NUMINAMATH_CALUDE_complex_absolute_value_l603_60321

theorem complex_absolute_value (z : ℂ) :
  (3 + 4*I) / z = 5*I → Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l603_60321


namespace NUMINAMATH_CALUDE_quadrilateral_side_sum_l603_60351

/-- Represents a quadrilateral with side lengths a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Predicate to check if angles are in arithmetic progression --/
def angles_in_arithmetic_progression (q : Quadrilateral) : Prop :=
  sorry

/-- Predicate to check if the largest side is opposite the largest angle --/
def largest_side_opposite_largest_angle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem --/
theorem quadrilateral_side_sum (q : Quadrilateral) 
  (h1 : angles_in_arithmetic_progression q)
  (h2 : largest_side_opposite_largest_angle q)
  (h3 : q.a = 7)
  (h4 : q.b = 8)
  (h5 : ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a + b + c = 113 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_side_sum_l603_60351


namespace NUMINAMATH_CALUDE_allison_sewing_time_l603_60353

/-- The time it takes Al to sew dresses individually -/
def al_time : ℝ := 12

/-- The time Allison and Al work together -/
def joint_work_time : ℝ := 3

/-- The additional time Allison works alone after Al leaves -/
def allison_extra_time : ℝ := 3.75

/-- The time it takes Allison to sew dresses individually -/
def allison_time : ℝ := 9

theorem allison_sewing_time : 
  (joint_work_time / allison_time + joint_work_time / al_time + allison_extra_time / allison_time) = 1 := by
  sorry

#check allison_sewing_time

end NUMINAMATH_CALUDE_allison_sewing_time_l603_60353


namespace NUMINAMATH_CALUDE_range_of_m_l603_60388

/-- The statement p: The equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- The statement q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∃ m : ℝ, ¬(p m) ∧ q m) →
  (∃ m : ℝ, 1 < m ∧ m ≤ 2) ∧ (∀ m : ℝ, (1 < m ∧ m ≤ 2) → (¬(p m) ∧ q m)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l603_60388


namespace NUMINAMATH_CALUDE_summer_locations_l603_60328

/-- Represents a location with temperature data --/
structure Location where
  temperatures : Finset ℕ
  median : ℕ
  mean : ℕ
  mode : Option ℕ
  variance : Option ℚ

/-- Checks if a location meets the summer criterion --/
def meetsSummerCriterion (loc : Location) : Prop :=
  loc.temperatures.card = 5 ∧ ∀ t ∈ loc.temperatures.toSet, t ≥ 22

/-- Location A --/
def locationA : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 24,
  mean := 0,  -- Not given
  mode := some 22,
  variance := none
}

/-- Location B --/
def locationB : Location := {
  temperatures := {},  -- We don't know the exact temperatures
  median := 27,
  mean := 24,
  mode := none,
  variance := none
}

/-- Location C --/
def locationC : Location := {
  temperatures := {32},  -- We only know one temperature
  median := 0,  -- Not given
  mean := 26,
  mode := none,
  variance := some (108/10)
}

theorem summer_locations :
  meetsSummerCriterion locationA ∧
  meetsSummerCriterion locationC ∧
  ¬ (meetsSummerCriterion locationB) :=
sorry

end NUMINAMATH_CALUDE_summer_locations_l603_60328


namespace NUMINAMATH_CALUDE_complex_equation_solution_l603_60319

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l603_60319


namespace NUMINAMATH_CALUDE_even_z_dominoes_l603_60373

/-- Represents a lattice polygon that can be covered by quad-dominoes -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-quad-domino -/
inductive SQuadDomino

/-- Represents a Z-quad-domino -/
inductive ZQuadDomino

/-- Represents a covering of a lattice polygon with quad-dominoes -/
structure Covering (P : LatticePolygon) where
  s_dominoes : List SQuadDomino
  z_dominoes : List ZQuadDomino
  is_valid : Bool -- Indicates if the covering is valid (no overlap and complete)

/-- Checks if a lattice polygon can be completely covered by S-quad-dominoes -/
def can_cover_with_s (P : LatticePolygon) : Prop :=
  ∃ (c : Covering P), c.z_dominoes.length = 0 ∧ c.is_valid

/-- Main theorem: If a lattice polygon can be covered by S-quad-dominoes,
    then any valid covering with S and Z quad-dominoes uses an even number of Z-quad-dominoes -/
theorem even_z_dominoes (P : LatticePolygon) 
  (h : can_cover_with_s P) : 
  ∀ (c : Covering P), c.is_valid → Even c.z_dominoes.length :=
sorry

end NUMINAMATH_CALUDE_even_z_dominoes_l603_60373


namespace NUMINAMATH_CALUDE_max_value_quadratic_l603_60316

theorem max_value_quadratic :
  ∃ (c : ℝ), c = 3395 / 49 ∧ ∀ (r : ℝ), -7 * r^2 + 50 * r - 20 ≤ c := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l603_60316


namespace NUMINAMATH_CALUDE_simultaneous_equations_imply_quadratic_l603_60369

theorem simultaneous_equations_imply_quadratic (x y : ℝ) :
  (2 * x^2 + 6 * x + 5 * y + 1 = 0) →
  (2 * x + y + 3 = 0) →
  (y^2 + 10 * y - 7 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_imply_quadratic_l603_60369


namespace NUMINAMATH_CALUDE_compare_logarithmic_expressions_l603_60333

open Real

theorem compare_logarithmic_expressions :
  let e := exp 1
  1/e > log (3^(1/3)) ∧ 
  log (3^(1/3)) > log π / π ∧ 
  log π / π > sqrt 15 * log 15 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_compare_logarithmic_expressions_l603_60333


namespace NUMINAMATH_CALUDE_circle_equation_l603_60376

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition for a circle to be tangent to the y-axis
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

-- Define the condition for the center to be on the line 3x - y = 0
def centerOnLine (c : Circle) : Prop :=
  c.center.2 = 3 * c.center.1

-- Define the condition for the circle to pass through point (2,3)
def passesThrough (c : Circle) : Prop :=
  (c.center.1 - 2)^2 + (c.center.2 - 3)^2 = c.radius^2

-- Define the equation of the circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation :
  ∀ c : Circle,
  tangentToYAxis c → centerOnLine c → passesThrough c →
  (∀ x y : ℝ, circleEquation c x y ↔ 
    ((x - 1)^2 + (y - 3)^2 = 1) ∨ 
    ((x - 13/9)^2 + (y - 13/3)^2 = 169/81)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l603_60376


namespace NUMINAMATH_CALUDE_orthogonal_vectors_k_value_l603_60350

/-- Given vector a = (3, -4), a + 2b = (k+1, k-4), and a is orthogonal to b, prove that k = -6 -/
theorem orthogonal_vectors_k_value (k : ℝ) (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (3, -4)
  (a.1 + 2 * b.1 = k + 1 ∧ a.2 + 2 * b.2 = k - 4) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_k_value_l603_60350


namespace NUMINAMATH_CALUDE_twice_gcf_equals_180_l603_60364

def a : ℕ := 180
def b : ℕ := 270
def c : ℕ := 450

theorem twice_gcf_equals_180 : 2 * Nat.gcd a (Nat.gcd b c) = 180 := by
  sorry

end NUMINAMATH_CALUDE_twice_gcf_equals_180_l603_60364


namespace NUMINAMATH_CALUDE_shifted_line_equation_l603_60334

/-- Given a line with equation y = -2x, shifting it one unit upwards
    results in the equation y = -2x + 1 -/
theorem shifted_line_equation (x y : ℝ) :
  (y = -2 * x) → (y + 1 = -2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l603_60334


namespace NUMINAMATH_CALUDE_apple_count_l603_60375

theorem apple_count (apples oranges : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : apples + oranges = 301) : 
  apples = 164 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l603_60375


namespace NUMINAMATH_CALUDE_increment_and_differential_at_point_l603_60317

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point and increment
def x₀ : ℝ := 2
def Δx : ℝ := 0.1

-- Define the increment of the function
def Δy (x : ℝ) (Δx : ℝ) : ℝ := f (x + Δx) - f x

-- Define the differential of the function
def dy (x : ℝ) (Δx : ℝ) : ℝ := 2 * x * Δx

-- Theorem statement
theorem increment_and_differential_at_point :
  (Δy x₀ Δx = 0.41) ∧ (dy x₀ Δx = 0.4) := by
  sorry

end NUMINAMATH_CALUDE_increment_and_differential_at_point_l603_60317


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l603_60349

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of Republicans in the committee -/
def total_republicans : ℕ := 10

/-- The total number of Democrats in the committee -/
def total_democrats : ℕ := 8

/-- The number of Republicans in the subcommittee -/
def subcommittee_republicans : ℕ := 4

/-- The number of Democrats in the subcommittee -/
def subcommittee_democrats : ℕ := 3

/-- The number of ways to form the subcommittee -/
def subcommittee_combinations : ℕ := 
  (binomial total_republicans subcommittee_republicans) * 
  (binomial total_democrats subcommittee_democrats)

theorem subcommittee_formation_count : subcommittee_combinations = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l603_60349


namespace NUMINAMATH_CALUDE_duck_count_relation_l603_60346

theorem duck_count_relation :
  ∀ (muscovy cayuga khaki : ℕ),
    muscovy = 39 →
    muscovy = cayuga + 4 →
    muscovy + cayuga + khaki = 90 →
    muscovy = 2 * cayuga - 31 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_count_relation_l603_60346


namespace NUMINAMATH_CALUDE_order_of_exponentials_l603_60337

theorem order_of_exponentials (a b c : ℝ) : 
  a = 2^(1/5) → b = (2/5)^(1/5) → c = (2/5)^(3/5) → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_exponentials_l603_60337


namespace NUMINAMATH_CALUDE_sector_area_l603_60314

/-- The area of a sector with central angle 2π/3 and radius 3 is 3π. -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = 3) :
  (θ / (2 * Real.pi)) * Real.pi * r^2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l603_60314


namespace NUMINAMATH_CALUDE_unique_number_base_conversion_l603_60374

/-- Represents a digit in a given base -/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to decimal -/
def ToDecimal (tens : ℕ) (ones : ℕ) (base : ℕ) : ℕ := base * tens + ones

theorem unique_number_base_conversion :
  ∃! n : ℕ, n > 0 ∧
    ∃ C D : ℕ,
      IsDigit C 8 ∧
      IsDigit D 8 ∧
      IsDigit C 6 ∧
      IsDigit D 6 ∧
      n = ToDecimal C D 8 ∧
      n = ToDecimal D C 6 ∧
      n = 43 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_base_conversion_l603_60374


namespace NUMINAMATH_CALUDE_square_sum_inequality_l603_60398

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l603_60398


namespace NUMINAMATH_CALUDE_platform_length_l603_60347

/-- Given a train's speed and crossing times, calculate the platform length -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- 72 kmph
  (h2 : platform_crossing_time = 32)  -- 32 seconds
  (h3 : man_crossing_time = 18)  -- 18 seconds
  : ∃ (platform_length : ℝ), platform_length = 280 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l603_60347


namespace NUMINAMATH_CALUDE_model1_best_fit_l603_60308

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.98
def R2_model2 : ℝ := 0.80
def R2_model3 : ℝ := 0.50
def R2_model4 : ℝ := 0.25

-- Define a function to compare R² values
def better_fit (a b : ℝ) : Prop := a > b

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit :
  better_fit R2_model1 R2_model2 ∧
  better_fit R2_model1 R2_model3 ∧
  better_fit R2_model1 R2_model4 :=
by sorry

end NUMINAMATH_CALUDE_model1_best_fit_l603_60308


namespace NUMINAMATH_CALUDE_triple_hash_twenty_l603_60348

/-- The # operation defined on real numbers -/
def hash (N : ℝ) : ℝ := 0.75 * N + 3

/-- Theorem stating that applying the hash operation three times to 20 results in 15.375 -/
theorem triple_hash_twenty : hash (hash (hash 20)) = 15.375 := by sorry

end NUMINAMATH_CALUDE_triple_hash_twenty_l603_60348


namespace NUMINAMATH_CALUDE_smallest_n_for_less_than_one_percent_probability_l603_60393

def double_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => (n+2) * double_factorial n

def townspeople_win_probability (n : ℕ) : ℚ :=
  (n.factorial : ℚ) / (double_factorial (2*n+1) : ℚ)

theorem smallest_n_for_less_than_one_percent_probability :
  ∀ k : ℕ, k < 6 → townspeople_win_probability k ≥ 1/100 ∧
  townspeople_win_probability 6 < 1/100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_less_than_one_percent_probability_l603_60393


namespace NUMINAMATH_CALUDE_M_equals_set_l603_60312

def M : Set ℕ := {m : ℕ | m > 0 ∧ (∃ k : ℤ, 10 = k * (m + 1))}

theorem M_equals_set : M = {1, 4, 9} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_set_l603_60312


namespace NUMINAMATH_CALUDE_lori_beanie_babies_l603_60326

theorem lori_beanie_babies (sydney_beanie_babies : ℕ) 
  (h1 : sydney_beanie_babies + 15 * sydney_beanie_babies = 320) : 
  15 * sydney_beanie_babies = 300 := by
  sorry

end NUMINAMATH_CALUDE_lori_beanie_babies_l603_60326


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l603_60335

theorem contrapositive_equivalence (a b : ℝ) :
  ((a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l603_60335


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l603_60390

theorem polynomial_divisibility (a b : ℕ) (h1 : a ≥ 2 * b) (h2 : b > 1) :
  ∃ P : Polynomial ℕ, (Polynomial.degree P > 0) ∧ 
    (∀ i, Polynomial.coeff P i < b) ∧
    (P.eval a % P.eval b = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l603_60390


namespace NUMINAMATH_CALUDE_erdos_theorem_l603_60382

/-- For any integer k, there exists a graph H with girth greater than k and chromatic number greater than k. -/
theorem erdos_theorem (k : ℕ) : ∃ H : SimpleGraph ℕ, SimpleGraph.girth H > k ∧ SimpleGraph.chromaticNumber H > k := by
  sorry

end NUMINAMATH_CALUDE_erdos_theorem_l603_60382


namespace NUMINAMATH_CALUDE_line_parallelism_l603_60366

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallelism 
  (a b : Line) 
  (α β : Plane) 
  (l : Line) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : intersection α β = l) 
  (h4 : parallel_lines a l) 
  (h5 : subset_line_plane b β) 
  (h6 : parallel_line_plane b α) : 
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallelism_l603_60366


namespace NUMINAMATH_CALUDE_intersection_condition_l603_60304

/-- The function f(x) = (m-3)x^2 - 4x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 - 4*x + 2

/-- The graph of f intersects the x-axis at only one point -/
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! x, f m x = 0

/-- Theorem: The graph of f(x) = (m-3)x^2 - 4x + 2 intersects the x-axis at only one point
    if and only if m = 3 or m = 5 -/
theorem intersection_condition (m : ℝ) :
  intersects_at_one_point m ↔ m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l603_60304


namespace NUMINAMATH_CALUDE_min_value_fraction_l603_60389

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l603_60389


namespace NUMINAMATH_CALUDE_female_managers_count_l603_60379

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * (c.total_employees - c.female_employees)) / 5

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : 
  c.total_managers - c.male_managers = 200 := by
  sorry

end NUMINAMATH_CALUDE_female_managers_count_l603_60379


namespace NUMINAMATH_CALUDE_total_tissues_l603_60338

def group1 : ℕ := 15
def group2 : ℕ := 20
def group3 : ℕ := 18
def group4 : ℕ := 22
def group5 : ℕ := 25
def tissues_per_box : ℕ := 70

theorem total_tissues : 
  (group1 + group2 + group3 + group4 + group5) * tissues_per_box = 7000 := by
  sorry

end NUMINAMATH_CALUDE_total_tissues_l603_60338


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l603_60327

/-- Given two chemical solutions x and y, and their mixture, prove the percentage of chemical b in solution x -/
theorem chemical_mixture_problem (x_a : ℝ) (y_a y_b : ℝ) (mixture_a : ℝ) :
  x_a = 0.3 →
  y_a = 0.4 →
  y_b = 0.6 →
  mixture_a = 0.32 →
  0.8 * x_a + 0.2 * y_a = mixture_a →
  1 - x_a = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l603_60327


namespace NUMINAMATH_CALUDE_mall_sales_growth_rate_l603_60370

theorem mall_sales_growth_rate :
  let initial_sales := 1000000  -- January sales in yuan
  let feb_decrease := 0.1       -- 10% decrease in February
  let april_sales := 1296000    -- April sales in yuan
  let growth_rate := 0.2        -- 20% growth rate to be proven
  (initial_sales * (1 - feb_decrease) * (1 + growth_rate)^2 = april_sales) := by
  sorry

end NUMINAMATH_CALUDE_mall_sales_growth_rate_l603_60370


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l603_60330

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℚ) (decrement : ℚ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n : ℚ) * original_mean - n * decrement = n * 191 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l603_60330


namespace NUMINAMATH_CALUDE_triangle_theorem_l603_60384

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The given condition relating sides and angles -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = (Real.cos (t.A + t.C)) / (Real.cos t.C)

theorem triangle_theorem (t : Triangle) (h : triangle_condition t) :
  t.C = 2 * π / 3 ∧ 1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l603_60384


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l603_60325

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem min_sum_of_primes (m n p : ℕ) :
  isPrime m → isPrime n → isPrime p →
  m ≠ n → n ≠ p → m ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ,
    isPrime m' → isPrime n' → isPrime p' →
    m' ≠ n' → n' ≠ p' → m' ≠ p' →
    m' + n' = p' →
    m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l603_60325


namespace NUMINAMATH_CALUDE_birch_planting_l603_60301

theorem birch_planting (total_students : ℕ) (roses_per_girl : ℕ) (total_plants : ℕ) (total_birches : ℕ)
  (h1 : total_students = 24)
  (h2 : roses_per_girl = 3)
  (h3 : total_plants = 24)
  (h4 : total_birches = 6) :
  (total_students - (total_plants - total_birches) / roses_per_girl) / 3 = total_birches :=
by sorry

end NUMINAMATH_CALUDE_birch_planting_l603_60301


namespace NUMINAMATH_CALUDE_pi_is_irrational_l603_60315

-- Define the property of being an infinite non-repeating decimal
def is_infinite_non_repeating_decimal (x : ℝ) : Prop := sorry

-- Define the property of being an irrational number
def is_irrational (x : ℝ) : Prop := sorry

-- Axiom: All irrational numbers are infinite non-repeating decimals
axiom irrational_are_infinite_non_repeating : 
  ∀ x : ℝ, is_irrational x → is_infinite_non_repeating_decimal x

-- Given: π is an infinite non-repeating decimal
axiom pi_is_infinite_non_repeating : is_infinite_non_repeating_decimal Real.pi

-- Theorem to prove
theorem pi_is_irrational : is_irrational Real.pi := sorry

end NUMINAMATH_CALUDE_pi_is_irrational_l603_60315


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l603_60365

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem: The smallest number of identical cubes needed to fill a box with 
    dimensions 36x45x18 inches is 40 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨36, 45, 18⟩ = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l603_60365


namespace NUMINAMATH_CALUDE_cells_after_three_divisions_l603_60367

/-- The number of cells after n divisions, given that each division doubles the number of cells -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of cells after 3 divisions is 8 -/
theorem cells_after_three_divisions : num_cells 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_three_divisions_l603_60367


namespace NUMINAMATH_CALUDE_cherry_pricing_and_profit_l603_60368

/-- Represents the cost and quantity of cherries --/
structure CherryData where
  yellow_cost : ℝ
  red_cost : ℝ
  yellow_quantity : ℝ
  red_quantity : ℝ

/-- Represents the sales data for red light cherries --/
structure SalesData where
  week1_price : ℝ
  week1_quantity : ℝ
  week2_price_decrease : ℝ
  week2_quantity : ℝ
  week3_discount : ℝ

/-- Theorem stating the cost price of red light cherries and minimum value of m --/
theorem cherry_pricing_and_profit (data : CherryData) (sales : SalesData) :
  data.yellow_cost = 6000 ∧
  data.red_cost = 1000 ∧
  data.yellow_quantity = data.red_quantity + 100 ∧
  data.yellow_cost / data.yellow_quantity = 2 * (data.red_cost / data.red_quantity) ∧
  sales.week1_price = 40 ∧
  sales.week2_quantity = 20 ∧
  sales.week3_discount = 0.3 →
  data.red_cost / data.red_quantity = 20 ∧
  ∃ m : ℝ,
    m ≥ 5 ∧
    sales.week1_quantity = 3 * m ∧
    sales.week2_price_decrease = 0.5 * m ∧
    (40 - 20) * (3 * m) + 20 * (40 - 0.5 * m - 20) + (40 * 0.7 - 20) * (50 - 3 * m - 20) ≥ 770 ∧
    ∀ m' : ℝ,
      m' < 5 →
      (40 - 20) * (3 * m') + 20 * (40 - 0.5 * m' - 20) + (40 * 0.7 - 20) * (50 - 3 * m' - 20) < 770 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pricing_and_profit_l603_60368


namespace NUMINAMATH_CALUDE_no_event_with_prob_1_5_l603_60387

-- Define the probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Axiom: Probability is always between 0 and 1
axiom prob_bounds (E : Set Ω) : 0 ≤ P E ∧ P E ≤ 1

-- Theorem: There does not exist an event with probability 1.5
theorem no_event_with_prob_1_5 : ¬∃ (A : Set Ω), P A = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_no_event_with_prob_1_5_l603_60387


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l603_60399

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  sides : ℕ
  is_irregular : Bool
  is_convex : Bool
  right_angles : ℕ

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that a nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals (P : Polygon 9) 
  (h1 : P.is_irregular = true) 
  (h2 : P.is_convex = true) 
  (h3 : P.right_angles = 2) : 
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l603_60399


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l603_60362

/-- The equation of a line passing through two given points is correct. -/
theorem reflected_ray_equation (x y : ℝ) :
  let p1 : ℝ × ℝ := (-1, -3)  -- Symmetric point of (-1, 3) with respect to x-axis
  let p2 : ℝ × ℝ := (4, 6)    -- Given point that the reflected ray passes through
  9 * x - 5 * y - 6 = 0 ↔ (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l603_60362


namespace NUMINAMATH_CALUDE_xiaofang_final_score_l603_60345

/-- Calculates the final score in a speech contest given the scores and weights for each category -/
def calculate_final_score (speech_content_score : ℝ) (language_expression_score : ℝ) (overall_effect_score : ℝ) 
  (speech_content_weight : ℝ) (language_expression_weight : ℝ) (overall_effect_weight : ℝ) : ℝ :=
  speech_content_score * speech_content_weight + 
  language_expression_score * language_expression_weight + 
  overall_effect_score * overall_effect_weight

/-- Theorem stating that Xiaofang's final score is 90 points -/
theorem xiaofang_final_score : 
  calculate_final_score 85 95 90 0.4 0.4 0.2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_xiaofang_final_score_l603_60345


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_line_l603_60300

/-- The minimum distance sum from a point on the parabola y² = -4x to the y-axis and the line 2x + y - 4 = 0 -/
theorem min_distance_sum_parabola_line : 
  ∃ (min_sum : ℝ), 
    min_sum = (6 * Real.sqrt 5) / 5 - 1 ∧
    ∀ (x y : ℝ),
      y^2 = -4*x →  -- point (x,y) is on the parabola
      ∃ (m n : ℝ),
        m = |x| ∧   -- distance to y-axis
        n = |2*x + y - 4| / Real.sqrt 5 ∧  -- distance to line
        m + n ≥ min_sum :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_parabola_line_l603_60300
