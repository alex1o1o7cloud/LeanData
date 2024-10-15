import Mathlib

namespace NUMINAMATH_CALUDE_complex_sum_problem_l1797_179752

theorem complex_sum_problem (b d e f : ℝ) : 
  b = 2 →
  e = -5 →
  (2 : ℂ) + b * I + (3 : ℂ) + d * I + e + f * I = (1 : ℂ) - 3 * I →
  d + f = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1797_179752


namespace NUMINAMATH_CALUDE_total_cats_is_thirteen_l1797_179770

/-- The number of cats owned by Jamie, Gordon, and Hawkeye --/
def total_cats : ℕ :=
  let jamie_persians : ℕ := 4
  let jamie_maine_coons : ℕ := 2
  let gordon_persians : ℕ := jamie_persians / 2
  let gordon_maine_coons : ℕ := jamie_maine_coons + 1
  let hawkeye_persians : ℕ := 0
  let hawkeye_maine_coons : ℕ := gordon_maine_coons - 1
  jamie_persians + jamie_maine_coons +
  gordon_persians + gordon_maine_coons +
  hawkeye_persians + hawkeye_maine_coons

theorem total_cats_is_thirteen : total_cats = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_thirteen_l1797_179770


namespace NUMINAMATH_CALUDE_simplified_root_sum_l1797_179747

theorem simplified_root_sum (a b : ℕ+) :
  (2^11 * 5^5 : ℝ)^(1/4) = a * b^(1/4) → a + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_simplified_root_sum_l1797_179747


namespace NUMINAMATH_CALUDE_four_stamps_cost_l1797_179720

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of n stamps in dollars -/
def n_stamps_cost (n : ℕ) : ℚ := n * stamp_cost

theorem four_stamps_cost :
  n_stamps_cost 4 = 136/100 :=
by sorry

end NUMINAMATH_CALUDE_four_stamps_cost_l1797_179720


namespace NUMINAMATH_CALUDE_general_term_formula_l1797_179733

-- Define the sequence
def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else if n = 2 then 8/3
  else if n = 3 then 15/4
  else if n = 4 then 24/5
  else if n = 5 then 35/6
  else if n = 6 then 48/7
  else (n^2 + 2*n) / (n + 1)

-- State the theorem
theorem general_term_formula (n : ℕ) (h : n > 0) :
  a n = (n^2 + 2*n) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l1797_179733


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l1797_179764

/-- 
Given an approximate number obtained by rounding, represented as 6.18 × 10^4,
prove that it is accurate to the hundred place.
-/
theorem rounded_number_accuracy : 
  let rounded_number : ℝ := 6.18 * 10^4
  ∃ (exact_number : ℝ), 
    (abs (exact_number - rounded_number) ≤ 50) ∧ 
    (∀ (place : ℕ), place > 2 → 
      ∃ (n : ℤ), rounded_number = (n * 10^place : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l1797_179764


namespace NUMINAMATH_CALUDE_stirring_evenly_key_to_representativeness_l1797_179746

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| Other

/-- Represents actions in the lottery method -/
inductive LotteryAction
| MakeTickets
| StirEvenly
| DrawOneByOne
| DrawWithoutReplacement

/-- Represents the property of being representative -/
def IsRepresentative (sample : Set α) : Prop := sorry

/-- The lottery method -/
def lotteryMethod : SamplingMethod := SamplingMethod.Lottery

/-- Function to determine if an action is key to representativeness -/
def isKeyToRepresentativeness (action : LotteryAction) (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_evenly_key_to_representativeness :
  isKeyToRepresentativeness LotteryAction.StirEvenly lotteryMethod := by sorry

end NUMINAMATH_CALUDE_stirring_evenly_key_to_representativeness_l1797_179746


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1797_179744

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (3 * Real.pi / 5) ^ 2) = -Real.cos (3 * Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1797_179744


namespace NUMINAMATH_CALUDE_urn_gold_coin_percentage_l1797_179734

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ
  bronzeCoinPercentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def goldCoinPercentage (u : UrnComposition) : ℝ :=
  (1 - u.beadPercentage) * u.goldCoinPercentage

/-- The theorem states that given the specified urn composition,
    the percentage of gold coins in the urn is 35% --/
theorem urn_gold_coin_percentage :
  ∀ (u : UrnComposition),
    u.beadPercentage = 0.3 ∧
    u.silverCoinPercentage = 0.25 * (1 - u.beadPercentage) ∧
    u.goldCoinPercentage = 0.5 * (1 - u.beadPercentage) ∧
    u.bronzeCoinPercentage = (1 - u.beadPercentage) * (1 - 0.25 - 0.5) →
    goldCoinPercentage u = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_urn_gold_coin_percentage_l1797_179734


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1797_179779

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |4 - 3*x| - 5 ≤ 0 ↔ -1/3 ≤ x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1797_179779


namespace NUMINAMATH_CALUDE_range_of_x_l1797_179728

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_x (h1 : ∀ x ∈ [-1, 1], Monotone f) 
  (h2 : ∀ x, f (x - 1) < f (1 - 3*x)) :
  ∃ S : Set ℝ, S = {x | 0 ≤ x ∧ x < 1/2} ∧ 
  (∀ x, x ∈ S ↔ (x - 1 ∈ [-1, 1] ∧ 1 - 3*x ∈ [-1, 1] ∧ f (x - 1) < f (1 - 3*x))) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l1797_179728


namespace NUMINAMATH_CALUDE_exists_greater_than_product_l1797_179714

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ k : ℕ+, (∃ (S : Finset (ℕ+ × ℕ+)), S.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ a p.1 p.2 = k))

theorem exists_greater_than_product (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_than_product_l1797_179714


namespace NUMINAMATH_CALUDE_line_canonical_form_l1797_179787

theorem line_canonical_form (x y z : ℝ) :
  (2 * x - 3 * y - 3 * z - 9 = 0 ∧ x - 2 * y + z + 3 = 0) →
  ∃ (t : ℝ), x = 9 * t ∧ y = 5 * t ∧ z = t - 3 :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_form_l1797_179787


namespace NUMINAMATH_CALUDE_candy_distribution_l1797_179774

theorem candy_distribution (total : Nat) (sisters : Nat) (take_away : Nat) : 
  total = 24 →
  sisters = 5 →
  take_away = 4 →
  (total - take_away) % sisters = 0 →
  ∀ x : Nat, x < take_away → (total - x) % sisters ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1797_179774


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l1797_179731

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l1797_179731


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1797_179732

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/2) * x ∨ y = -(3/2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1797_179732


namespace NUMINAMATH_CALUDE_det_transformation_l1797_179705

/-- Given a 2x2 matrix with determinant 6, prove that a specific transformation of this matrix results in a determinant of 24. -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 6) :
  Matrix.det !![p, 8*p + 4*q; r, 8*r + 4*s] = 24 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l1797_179705


namespace NUMINAMATH_CALUDE_inequality_solution_l1797_179700

/-- Given an inequality ax^2 - 3x + 2 < 0 with solution set {x | 1 < x < b}, prove a + b = 3 -/
theorem inequality_solution (a b : ℝ) 
  (h : ∀ x, ax^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < b) : 
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1797_179700


namespace NUMINAMATH_CALUDE_find_extreme_stone_l1797_179777

/-- A stone with a specific weight -/
structure Stone where
  weight : ℝ

/-- A two-tiered balance scale that can compare two pairs of stones -/
def TwoTieredScale (stones : Finset Stone) : Prop :=
  ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    ((a.weight + b.weight) > (c.weight + d.weight)) ∨
    ((a.weight + b.weight) < (c.weight + d.weight))

/-- The theorem stating that we can find either the heaviest or the lightest stone -/
theorem find_extreme_stone
  (stones : Finset Stone)
  (h_count : stones.card = 10)
  (h_distinct_weights : ∀ (a b : Stone), a ∈ stones → b ∈ stones → a ≠ b → a.weight ≠ b.weight)
  (h_distinct_sums : ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    (a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c)) →
    a.weight + b.weight ≠ c.weight + d.weight)
  (h_scale : TwoTieredScale stones) :
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≥ t.weight) ∨
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≤ t.weight) :=
sorry

end NUMINAMATH_CALUDE_find_extreme_stone_l1797_179777


namespace NUMINAMATH_CALUDE_number_pattern_l1797_179767

theorem number_pattern (A : ℕ) : 10 * A + 9 = A * 9 + (A + 9) := by
  sorry

end NUMINAMATH_CALUDE_number_pattern_l1797_179767


namespace NUMINAMATH_CALUDE_angle_C_is_two_pi_third_l1797_179786

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define vectors p and q
def p (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b)
def q (t : Triangle) : ℝ × ℝ := (t.b + t.a, t.c - t.a)

-- Define parallelism of vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem angle_C_is_two_pi_third (t : Triangle) 
  (h : parallel (p t) (q t)) : t.C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_two_pi_third_l1797_179786


namespace NUMINAMATH_CALUDE_largest_non_expressible_l1797_179789

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ (a : ℕ) (b : ℕ), n = 48 * a + b ∧ is_composite b ∧ 0 < b

theorem largest_non_expressible :
  (∀ n > 95, is_expressible n) ∧
  ¬is_expressible 95 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l1797_179789


namespace NUMINAMATH_CALUDE_power_of_three_decomposition_l1797_179762

theorem power_of_three_decomposition : 3^25 = 27^7 * 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_decomposition_l1797_179762


namespace NUMINAMATH_CALUDE_stacy_paper_completion_time_l1797_179772

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 
  63 / 9

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := 63

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that Stacy has 7 days to complete her paper -/
theorem stacy_paper_completion_time : days_to_complete = 7 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_completion_time_l1797_179772


namespace NUMINAMATH_CALUDE_problem_statement_l1797_179798

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + y = 5) 
  (h2 : x + 3 * y = 6) : 
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1797_179798


namespace NUMINAMATH_CALUDE_inequality_proof_l1797_179781

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1797_179781


namespace NUMINAMATH_CALUDE_different_color_probability_l1797_179725

def totalChips : ℕ := 18

def blueChips : ℕ := 4
def greenChips : ℕ := 5
def redChips : ℕ := 6
def yellowChips : ℕ := 3

def probBlue : ℚ := blueChips / totalChips
def probGreen : ℚ := greenChips / totalChips
def probRed : ℚ := redChips / totalChips
def probYellow : ℚ := yellowChips / totalChips

theorem different_color_probability : 
  (probBlue * probGreen * probRed + 
   probBlue * probGreen * probYellow + 
   probBlue * probRed * probYellow + 
   probGreen * probRed * probYellow) * 6 = 141 / 162 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_l1797_179725


namespace NUMINAMATH_CALUDE_football_group_stage_teams_l1797_179709

/-- The number of participating teams in the football group stage -/
def num_teams : ℕ := 16

/-- The number of stadiums used -/
def num_stadiums : ℕ := 6

/-- The number of games scheduled at each stadium per day -/
def games_per_stadium_per_day : ℕ := 4

/-- The number of consecutive days to complete all group stage matches -/
def num_days : ℕ := 10

theorem football_group_stage_teams :
  num_teams * (num_teams - 1) = num_stadiums * games_per_stadium_per_day * num_days :=
by sorry

end NUMINAMATH_CALUDE_football_group_stage_teams_l1797_179709


namespace NUMINAMATH_CALUDE_max_distance_le_150cm_l1797_179729

/-- Represents the extended table with two semicircles and a rectangular section -/
structure ExtendedTable where
  semicircle_diameter : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The maximum distance between any two points on the extended table -/
def max_distance (table : ExtendedTable) : ℝ :=
  sorry

/-- Theorem stating that the maximum distance between any two points on the extended table
    is less than or equal to 150 cm -/
theorem max_distance_le_150cm (table : ExtendedTable)
  (h1 : table.semicircle_diameter = 1)
  (h2 : table.rectangle_length = 1)
  (h3 : table.rectangle_width = 0.5) :
  max_distance table ≤ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_le_150cm_l1797_179729


namespace NUMINAMATH_CALUDE_percentage_problem_l1797_179750

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1797_179750


namespace NUMINAMATH_CALUDE_thabo_book_difference_l1797_179707

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- The conditions of Thabo's book collection -/
def thabosBooks (b : BookCollection) : Prop :=
  b.paperbackFiction + b.paperbackNonfiction + b.hardcoverNonfiction = 200 ∧
  b.paperbackNonfiction > b.hardcoverNonfiction ∧
  b.paperbackFiction = 2 * b.paperbackNonfiction ∧
  b.hardcoverNonfiction = 35

theorem thabo_book_difference (b : BookCollection) 
  (h : thabosBooks b) : 
  b.paperbackNonfiction - b.hardcoverNonfiction = 20 := by
  sorry


end NUMINAMATH_CALUDE_thabo_book_difference_l1797_179707


namespace NUMINAMATH_CALUDE_value_of_A_l1797_179717

/-- Given the value of letters in words, find the value of A -/
theorem value_of_A (H M A T E : ℤ) : 
  H = 12 →
  M + A + T + H = 40 →
  T + E + A + M = 50 →
  M + E + E + T = 44 →
  A = 28 := by
sorry

end NUMINAMATH_CALUDE_value_of_A_l1797_179717


namespace NUMINAMATH_CALUDE_inequality_proof_l1797_179768

theorem inequality_proof (a b u v k : ℝ) 
  (ha : a > 0) (hb : b > 0) (huv : u < v) (hk : k > 0) :
  (a^u + b^u) / (a^v + b^v) ≥ (a^(u+k) + b^(u+k)) / (a^(v+k) + b^(v+k)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1797_179768


namespace NUMINAMATH_CALUDE_veggie_patty_percentage_l1797_179761

/-- Proves that the percentage of a veggie patty that is not made up of spices and additives is 70% -/
theorem veggie_patty_percentage (total_weight spice_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : spice_weight = 45) :
  (total_weight - spice_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_veggie_patty_percentage_l1797_179761


namespace NUMINAMATH_CALUDE_tinas_weekly_income_l1797_179715

/-- Calculates Tina's weekly income based on her work schedule and pay rates. -/
def calculate_weekly_income (hourly_wage : ℚ) (regular_hours : ℚ) (weekday_hours : ℚ) (weekend_hours : ℚ) : ℚ :=
  let overtime_rate := hourly_wage + hourly_wage / 2
  let double_overtime_rate := hourly_wage * 2
  let weekday_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (weekday_hours - regular_hours)
  ) * 5
  let weekend_pay := (
    hourly_wage * regular_hours + 
    overtime_rate * (regular_hours - regular_hours) +
    double_overtime_rate * (weekend_hours - regular_hours - (regular_hours - regular_hours))
  ) * 2
  weekday_pay + weekend_pay

/-- Theorem stating that Tina's weekly income is $1530.00 given her work schedule and pay rates. -/
theorem tinas_weekly_income :
  calculate_weekly_income 18 8 10 12 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tinas_weekly_income_l1797_179715


namespace NUMINAMATH_CALUDE_largest_x_for_equation_l1797_179745

theorem largest_x_for_equation : 
  (∀ x y : ℤ, x > 3 → x^2 - x*y - 2*y^2 ≠ 9) ∧ 
  (∃ y : ℤ, 3^2 - 3*y - 2*y^2 = 9) :=
sorry

end NUMINAMATH_CALUDE_largest_x_for_equation_l1797_179745


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1797_179718

theorem arithmetic_evaluation : 6 + (3 * 6) - 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1797_179718


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1797_179759

theorem algebraic_expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 5) :
  -2*a^2 - 4*a + 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1797_179759


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l1797_179792

/-- The Ferris wheel problem -/
theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l1797_179792


namespace NUMINAMATH_CALUDE_license_plate_difference_l1797_179741

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Alpha state -/
def alpha_plates : ℕ := num_letters^3 * num_digits^4

/-- The number of possible license plates for Beta state -/
def beta_plates : ℕ := num_letters^4 * num_digits^3

/-- The difference in the number of possible license plates between Beta and Alpha -/
def plate_difference : ℕ := beta_plates - alpha_plates

theorem license_plate_difference : plate_difference = 281216000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1797_179741


namespace NUMINAMATH_CALUDE_min_n_with_three_same_color_l1797_179724

/-- A coloring of an n × n grid using three colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- Checks if a coloring satisfies the condition of having at least three squares
    of the same color in a row or column. -/
def satisfiesCondition (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (i : Fin n) (color : Fin 3),
    (∃ (j₁ j₂ j₃ : Fin n), j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧
      c i j₁ = color ∧ c i j₂ = color ∧ c i j₃ = color) ∨
    (∃ (i₁ i₂ i₃ : Fin n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧
      c i₁ i = color ∧ c i₂ i = color ∧ c i₃ i = color)

/-- The main theorem stating that 7 is the smallest n that satisfies the condition. -/
theorem min_n_with_three_same_color :
  (∀ (c : Coloring 7), satisfiesCondition 7 c) ∧
  (∀ (n : ℕ), n < 7 → ∃ (c : Coloring n), ¬satisfiesCondition n c) :=
sorry

end NUMINAMATH_CALUDE_min_n_with_three_same_color_l1797_179724


namespace NUMINAMATH_CALUDE_parabola_and_tangent_circle_l1797_179704

noncomputable section

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the directrix l
def directrix_l (x : ℝ) : Prop := x = -2

-- Define point P on the directrix
def point_P (t : ℝ) : ℝ × ℝ := (-2, 3*t - 1/t)

-- Define point Q on the y-axis
def point_Q (t : ℝ) : ℝ × ℝ := (0, 2*t)

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x-2)^2 + y^2 = 4

-- Define a line through two points
def line_through (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Main theorem
theorem parabola_and_tangent_circle (t : ℝ) (ht : t ≠ 0) :
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧
  (∀ x y, line_through (point_P t) (point_Q t) x y →
    ∃ x0 y0, circle_M x0 y0 ∧
      ((x - x0)^2 + (y - y0)^2 = 4 ∧
       ((x - x0) * (x - x0) + (y - y0) * (y - y0) = 4))) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_tangent_circle_l1797_179704


namespace NUMINAMATH_CALUDE_intersection_distance_l1797_179756

-- Define the line C₁
def C₁ (x y : ℝ) : Prop := y - 2*x + 1 = 0

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1797_179756


namespace NUMINAMATH_CALUDE_expression_value_at_three_l1797_179708

theorem expression_value_at_three :
  ∀ x : ℝ, x ≠ 2 → x = 3 → (x^2 - 5*x + 6) / (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l1797_179708


namespace NUMINAMATH_CALUDE_tangent_ratio_problem_l1797_179713

theorem tangent_ratio_problem (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_problem_l1797_179713


namespace NUMINAMATH_CALUDE_range_of_a_l1797_179763

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1797_179763


namespace NUMINAMATH_CALUDE_square_count_theorem_l1797_179773

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Represents the configuration of two perpendicular families of lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)

/-- Represents the set of intersection points -/
def IntersectionPoints (config : LineConfiguration) : ℕ :=
  config.family1.count * config.family2.count

/-- Counts the number of squares with sides parallel to the coordinate axes -/
def countParallelSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- Counts the number of slanted squares -/
def countSlantedSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem square_count_theorem (config : LineConfiguration) 
  (h1 : config.family1.count = 15)
  (h2 : config.family2.count = 11)
  (h3 : IntersectionPoints config = 165) :
  countParallelSquares config + countSlantedSquares config ≥ 1986 :=
sorry

end NUMINAMATH_CALUDE_square_count_theorem_l1797_179773


namespace NUMINAMATH_CALUDE_total_swordfish_caught_l1797_179769

def fishing_trips : ℕ := 5

def shelly_catch : ℕ := 5 - 2

def sam_catch : ℕ := shelly_catch - 1

theorem total_swordfish_caught : shelly_catch * fishing_trips + sam_catch * fishing_trips = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_caught_l1797_179769


namespace NUMINAMATH_CALUDE_third_year_students_l1797_179738

theorem third_year_students (total_first_year : ℕ) (total_selected : ℕ) (second_year_selected : ℕ) :
  total_first_year = 720 →
  total_selected = 180 →
  second_year_selected = 40 →
  ∃ (first_year_selected third_year_selected : ℕ),
    first_year_selected = (second_year_selected + third_year_selected) / 2 ∧
    first_year_selected + second_year_selected + third_year_selected = total_selected ∧
    (total_first_year * third_year_selected : ℚ) / first_year_selected = 960 :=
by sorry

end NUMINAMATH_CALUDE_third_year_students_l1797_179738


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l1797_179755

/-- The imaginary part of i²(1+i) is -1 -/
theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l1797_179755


namespace NUMINAMATH_CALUDE_specific_window_height_l1797_179760

/-- Represents a rectangular window with glass panes. -/
structure Window where
  num_panes : ℕ
  rows : ℕ
  columns : ℕ
  pane_height_ratio : ℚ
  pane_width_ratio : ℚ
  border_width : ℕ

/-- Calculates the height of a window given its specifications. -/
def window_height (w : Window) : ℕ :=
  let pane_width := 4 * w.border_width
  let pane_height := 3 * w.border_width
  pane_height * w.rows + w.border_width * (w.rows + 1)

/-- The theorem stating the height of the specific window. -/
theorem specific_window_height :
  let w : Window := {
    num_panes := 8,
    rows := 4,
    columns := 2,
    pane_height_ratio := 3/4,
    pane_width_ratio := 4/3,
    border_width := 3
  }
  window_height w = 51 := by sorry

end NUMINAMATH_CALUDE_specific_window_height_l1797_179760


namespace NUMINAMATH_CALUDE_stratified_selection_count_l1797_179785

def female_students : ℕ := 8
def male_students : ℕ := 4
def total_selected : ℕ := 3

theorem stratified_selection_count :
  (Nat.choose female_students 2 * Nat.choose male_students 1) +
  (Nat.choose female_students 1 * Nat.choose male_students 2) = 112 :=
by sorry

end NUMINAMATH_CALUDE_stratified_selection_count_l1797_179785


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1797_179797

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its focal length is 10 and the point (1, 2) lies on its asymptote,
    then its equation is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 25) →
  (b - 2*a = 0) →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/5 - y^2/20 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1797_179797


namespace NUMINAMATH_CALUDE_age_problem_l1797_179737

theorem age_problem :
  ∀ (a b c : ℕ),
  a + b + c = 29 →
  a = b →
  c = 11 →
  a = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l1797_179737


namespace NUMINAMATH_CALUDE_sequence_formula_l1797_179782

/-- For a sequence {a_n} where a_1 = 1 and a_{n+1} = 2^n * a_n for n ≥ 1,
    a_n = 2^(n*(n-1)/2) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2^n * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n*(n-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l1797_179782


namespace NUMINAMATH_CALUDE_range_of_m_l1797_179703

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(¬(q m)) → m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1797_179703


namespace NUMINAMATH_CALUDE_garden_dimensions_possible_longest_side_l1797_179701

/-- Represents a rectangular garden with one side along a wall -/
structure RectangularGarden where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  total_fence : ℕ
  h_fence : side1 + side2 + side3 = total_fence

/-- The total fence length is 140 meters -/
def total_fence : ℕ := 140

theorem garden_dimensions (g : RectangularGarden) 
  (h1 : g.side1 = 40) (h2 : g.side2 = 40) (h_total : g.total_fence = total_fence) : 
  g.side3 = 60 := by
  sorry

theorem possible_longest_side (g : RectangularGarden) (h_total : g.total_fence = total_fence) :
  (∃ (g' : RectangularGarden), g'.side1 = 65 ∨ g'.side2 = 65 ∨ g'.side3 = 65) ∧
  (¬∃ (g' : RectangularGarden), g'.side1 = 85 ∨ g'.side2 = 85 ∨ g'.side3 = 85) := by
  sorry

end NUMINAMATH_CALUDE_garden_dimensions_possible_longest_side_l1797_179701


namespace NUMINAMATH_CALUDE_banana_count_l1797_179711

/-- Represents the contents and costs of a fruit basket -/
structure FruitBasket where
  num_bananas : ℕ
  num_apples : ℕ
  num_strawberries : ℕ
  num_avocados : ℕ
  num_grape_bunches : ℕ
  banana_cost : ℚ
  apple_cost : ℚ
  strawberry_dozen_cost : ℚ
  avocado_cost : ℚ
  half_grape_bunch_cost : ℚ
  total_cost : ℚ

/-- Theorem stating the number of bananas in the fruit basket -/
theorem banana_count (basket : FruitBasket) 
  (h1 : basket.num_apples = 3)
  (h2 : basket.num_strawberries = 24)
  (h3 : basket.num_avocados = 2)
  (h4 : basket.num_grape_bunches = 1)
  (h5 : basket.banana_cost = 1)
  (h6 : basket.apple_cost = 2)
  (h7 : basket.strawberry_dozen_cost = 4)
  (h8 : basket.avocado_cost = 3)
  (h9 : basket.half_grape_bunch_cost = 2)
  (h10 : basket.total_cost = 28) :
  basket.num_bananas = 4 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l1797_179711


namespace NUMINAMATH_CALUDE_comparison_inequality_l1797_179716

theorem comparison_inequality : ∀ x : ℝ, (x - 2) * (x + 3) > x^2 + x - 7 := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequality_l1797_179716


namespace NUMINAMATH_CALUDE_line_intercept_sum_minimum_equality_condition_l1797_179736

theorem line_intercept_sum_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b ≥ 4 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b = 4 ↔ a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_minimum_equality_condition_l1797_179736


namespace NUMINAMATH_CALUDE_C_power_50_l1797_179719

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_power_50 : C^50 = !![(-299), (-100); 800, 251] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1797_179719


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1797_179721

theorem unique_modular_solution : 
  ∀ n : ℤ, (10 ≤ n ∧ n ≤ 20) ∧ (n ≡ 7882 [ZMOD 7]) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1797_179721


namespace NUMINAMATH_CALUDE_spinner_probability_l1797_179727

theorem spinner_probability : ∀ (p_A p_B p_C p_D p_E : ℝ),
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 3/20 :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l1797_179727


namespace NUMINAMATH_CALUDE_triangle_problem_l1797_179765

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  let vec_a : ℝ × ℝ := (Real.cos A, Real.cos B)
  let vec_b : ℝ × ℝ := (a, 2*c - b)
  (∃ k : ℝ, vec_a = k • vec_b) →  -- vectors are parallel
  b = 3 →
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 →  -- area condition
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1797_179765


namespace NUMINAMATH_CALUDE_math_competition_score_l1797_179799

theorem math_competition_score (x : ℕ) : 
  let total_problems := 8 * x + x
  let missed_problems := 2 * x
  let bonus_problems := x
  let standard_points := (total_problems - missed_problems - bonus_problems)
  let bonus_points := 2 * bonus_problems
  let total_available_points := total_problems + bonus_problems
  let scored_points := standard_points + bonus_points
  (scored_points : ℚ) / total_available_points = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_l1797_179799


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1797_179783

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 = 3 →
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1797_179783


namespace NUMINAMATH_CALUDE_fixed_monthly_costs_l1797_179766

/-- A problem about calculating fixed monthly costs for a computer manufacturer. -/
theorem fixed_monthly_costs (production_cost shipping_cost monthly_units lowest_price : ℕ) :
  production_cost = 80 →
  shipping_cost = 2 →
  monthly_units = 150 →
  lowest_price = 190 →
  (production_cost + shipping_cost) * monthly_units + 16200 = lowest_price * monthly_units :=
by sorry

end NUMINAMATH_CALUDE_fixed_monthly_costs_l1797_179766


namespace NUMINAMATH_CALUDE_point_on_positive_x_axis_l1797_179778

theorem point_on_positive_x_axis (m : ℝ) : 
  let x := m^2 + Real.pi
  let y := 0
  x > 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_on_positive_x_axis_l1797_179778


namespace NUMINAMATH_CALUDE_min_stool_height_l1797_179742

/-- The minimum height of the stool for Alice to reach the ceiling fan switch -/
theorem min_stool_height (ceiling_height : ℝ) (switch_below_ceiling : ℝ) 
  (alice_height : ℝ) (alice_reach : ℝ) (books_height : ℝ) 
  (h1 : ceiling_height = 300) 
  (h2 : switch_below_ceiling = 15)
  (h3 : alice_height = 160)
  (h4 : alice_reach = 50)
  (h5 : books_height = 12) : 
  ∃ (s : ℝ), s ≥ 63 ∧ 
  ∀ (x : ℝ), x < 63 → alice_height + alice_reach + books_height + x < ceiling_height - switch_below_ceiling :=
sorry

end NUMINAMATH_CALUDE_min_stool_height_l1797_179742


namespace NUMINAMATH_CALUDE_bryans_precious_stones_l1797_179730

theorem bryans_precious_stones (price_per_stone total_amount : ℕ) 
  (h1 : price_per_stone = 1785)
  (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end NUMINAMATH_CALUDE_bryans_precious_stones_l1797_179730


namespace NUMINAMATH_CALUDE_binomial_sum_of_squares_l1797_179771

theorem binomial_sum_of_squares (a : ℝ) : 
  3 * a^4 + 1 = (a^2 + a)^2 + (a^2 - a)^2 + (a^2 - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_of_squares_l1797_179771


namespace NUMINAMATH_CALUDE_four_dice_same_face_probability_l1797_179794

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being tossed -/
def numDice : ℕ := 4

/-- The probability of a specific outcome on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability of all dice showing the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

theorem four_dice_same_face_probability :
  allSameProbability = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_face_probability_l1797_179794


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_distance_l1797_179743

-- Define the parabola function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Theorem statement
theorem parabola_x_intercepts_distance : 
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_distance_l1797_179743


namespace NUMINAMATH_CALUDE_base_eight_4372_equals_2298_l1797_179753

def base_eight_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_4372_equals_2298 :
  base_eight_to_decimal [2, 7, 3, 4] = 2298 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_4372_equals_2298_l1797_179753


namespace NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_100_l1797_179710

/-- Calculates the profit for a scout troop selling candy bars -/
theorem scout_troop_profit (total_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : ℚ :=
  let cost_per_bar := (2 : ℚ) / 5
  let sell_per_bar := (1 : ℚ) / 2
  let total_cost := total_bars * cost_per_bar
  let total_revenue := total_bars * sell_per_bar
  total_revenue - total_cost

/-- Proves that the scout troop's profit is $100 -/
theorem scout_troop_profit_is_100 :
  scout_troop_profit 1000 ((2 : ℚ) / 5) ((1 : ℚ) / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_scout_troop_profit_is_100_l1797_179710


namespace NUMINAMATH_CALUDE_height_distribution_study_l1797_179723

-- Define the type for students
def Student : Type := Unit

-- Define the school population
def schoolPopulation : Finset Student := sorry

-- Define the sample of measured students
def measuredSample : Finset Student := sorry

-- State the theorem
theorem height_distribution_study :
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation) ∧
  (measuredSample ⊆ schoolPopulation) ∧
  (Finset.card measuredSample = 40) →
  (Finset.card schoolPopulation = 240) ∧
  (∀ s : Student, s ∈ schoolPopulation → s = s) ∧
  (measuredSample = measuredSample) ∧
  (Finset.card measuredSample = 40) := by
  sorry

end NUMINAMATH_CALUDE_height_distribution_study_l1797_179723


namespace NUMINAMATH_CALUDE_special_function_characterization_l1797_179757

/-- A monotonic and invertible function from ℝ to ℝ satisfying f(x) + f⁻¹(x) = 2x for all x ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Function.Bijective f ∧ ∀ x, f x + (Function.invFun f) x = 2 * x

/-- The theorem stating that any function satisfying SpecialFunction is of the form f(x) = x + c -/
theorem special_function_characterization (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_special_function_characterization_l1797_179757


namespace NUMINAMATH_CALUDE_line_representation_l1797_179702

/-- A line in the xy-plane is represented by the equation y = k(x+1) -/
structure Line where
  k : ℝ

/-- The point (-1,0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- A line passes through a point if the point satisfies the line's equation -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- A line is perpendicular to the x-axis if its slope is undefined (i.e., infinite) -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  l.k = 0

/-- Main theorem: The equation y = k(x+1) represents all lines passing through
    the point (-1,0) and not perpendicular to the x-axis -/
theorem line_representation (l : Line) :
  (passes_through l point ∧ ¬perpendicular_to_x_axis l) ↔ 
  ∃ (k : ℝ), l = Line.mk k :=
sorry

end NUMINAMATH_CALUDE_line_representation_l1797_179702


namespace NUMINAMATH_CALUDE_quiz_score_ratio_l1797_179775

/-- Given a quiz taken by three people with specific scoring conditions,
    prove that the ratio of Tatuya's score to Ivanna's score is 2:1 -/
theorem quiz_score_ratio (tatuya_score ivanna_score dorothy_score : ℚ) : 
  dorothy_score = 90 →
  ivanna_score = (3/5) * dorothy_score →
  (tatuya_score + ivanna_score + dorothy_score) / 3 = 84 →
  tatuya_score / ivanna_score = 2 := by
sorry

end NUMINAMATH_CALUDE_quiz_score_ratio_l1797_179775


namespace NUMINAMATH_CALUDE_exists_number_not_in_progressions_l1797_179748

/-- Represents a geometric progression of natural numbers -/
structure GeometricProgression where
  first_term : ℕ
  common_ratio : ℕ
  h_positive : common_ratio > 1

/-- Checks if a natural number is in a geometric progression -/
def isInProgression (n : ℕ) (gp : GeometricProgression) : Prop :=
  ∃ k : ℕ, n = gp.first_term * gp.common_ratio ^ k

/-- The main theorem -/
theorem exists_number_not_in_progressions (progressions : Fin 100 → GeometricProgression) :
  ∃ n : ℕ, ∀ i : Fin 100, ¬ isInProgression n (progressions i) := by
  sorry


end NUMINAMATH_CALUDE_exists_number_not_in_progressions_l1797_179748


namespace NUMINAMATH_CALUDE_secret_room_number_l1797_179790

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def has_digit_8 (n : ℕ) : Prop := (n / 10 = 8) ∨ (n % 10 = 8)

def exactly_three_true (p q r s : Prop) : Prop :=
  (p ∧ q ∧ r ∧ ¬s) ∨ (p ∧ q ∧ ¬r ∧ s) ∨ (p ∧ ¬q ∧ r ∧ s) ∨ (¬p ∧ q ∧ r ∧ s)

theorem secret_room_number (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : exactly_three_true (divisible_by_4 n) (is_odd n) (sum_of_digits n = 12) (has_digit_8 n)) :
  n % 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_secret_room_number_l1797_179790


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_projection_l1797_179780

/-- Represents a right isosceles triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  right_angle : Bool
  isosceles : Bool

/-- Represents the projection of a triangle -/
def project (t : RightIsoscelesTriangle) (parallel : Bool) : RightIsoscelesTriangle :=
  if parallel then t else sorry

theorem right_isosceles_triangle_projection
  (t : RightIsoscelesTriangle)
  (h_side : t.side = 6)
  (h_right : t.right_angle = true)
  (h_isosceles : t.isosceles = true)
  (h_parallel : parallel = true) :
  let projected := project t parallel
  projected.side = 6 ∧
  projected.right_angle = true ∧
  projected.isosceles = true ∧
  Real.sqrt (2 * projected.side ^ 2) = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_projection_l1797_179780


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l1797_179788

/-- The time it takes for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 130)
  (h2 : bridge_length = 150)
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * (5/18)) = 28 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l1797_179788


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_special_right_triangle_l1797_179776

-- Define a structure for a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  sum_less_than_45 : angle1 + angle2 < 45
  angles_positive : 0 < angle1 ∧ 0 < angle2

-- Define a predicate for primality (approximate for real numbers)
def is_prime_approx (x : ℝ) : Prop := sorry

-- Define the theorem
theorem smallest_prime_angle_in_special_right_triangle :
  ∀ (t : RightTriangle),
    is_prime_approx t.angle1 →
    is_prime_approx t.angle2 →
    ∃ (smaller_angle : ℝ),
      smaller_angle = min t.angle1 t.angle2 ∧
      smaller_angle ≥ 2.3 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_angle_in_special_right_triangle_l1797_179776


namespace NUMINAMATH_CALUDE_multiplication_formula_l1797_179754

theorem multiplication_formula (x y z : ℝ) :
  (2*x + y + z) * (2*x - y - z) = 4*x^2 - y^2 - 2*y*z - z^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_l1797_179754


namespace NUMINAMATH_CALUDE_mittens_per_box_l1797_179706

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_items : ℕ) 
  (h1 : num_boxes = 4)
  (h2 : scarves_per_box = 2)
  (h3 : total_items = 32) :
  (total_items - num_boxes * scarves_per_box) / num_boxes = 6 :=
by sorry

end NUMINAMATH_CALUDE_mittens_per_box_l1797_179706


namespace NUMINAMATH_CALUDE_evaluate_expression_l1797_179722

theorem evaluate_expression : (16^24) / (64^8) = 16^8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1797_179722


namespace NUMINAMATH_CALUDE_rachel_piggy_bank_l1797_179791

/-- The amount of money Rachel originally had in her piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel now has in her piggy bank -/
def current_amount : ℕ := 3

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := original_amount - current_amount

theorem rachel_piggy_bank :
  amount_taken = 2 :=
sorry

end NUMINAMATH_CALUDE_rachel_piggy_bank_l1797_179791


namespace NUMINAMATH_CALUDE_auction_bids_per_person_l1797_179739

theorem auction_bids_per_person 
  (starting_price : ℕ) 
  (final_price : ℕ) 
  (price_increase : ℕ) 
  (num_bidders : ℕ) 
  (h1 : starting_price = 15)
  (h2 : final_price = 65)
  (h3 : price_increase = 5)
  (h4 : num_bidders = 2) :
  (final_price - starting_price) / price_increase / num_bidders = 5 :=
by sorry

end NUMINAMATH_CALUDE_auction_bids_per_person_l1797_179739


namespace NUMINAMATH_CALUDE_periodic_function_l1797_179793

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) + f (x - 1) = Real.sqrt 3 * f x

/-- The period of a function -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    HasPeriod f 12 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l1797_179793


namespace NUMINAMATH_CALUDE_shelbys_drive_l1797_179758

/-- Represents the weather conditions during Shelby's drive --/
inductive Weather
  | Sunny
  | Rainy
  | Foggy

/-- Shelby's driving scenario --/
structure DrivingScenario where
  speed : Weather → ℝ
  total_distance : ℝ
  total_time : ℝ
  time_in_weather : Weather → ℝ

/-- The theorem statement for Shelby's driving problem --/
theorem shelbys_drive (scenario : DrivingScenario) : 
  scenario.speed Weather.Sunny = 35 ∧ 
  scenario.speed Weather.Rainy = 25 ∧ 
  scenario.speed Weather.Foggy = 15 ∧ 
  scenario.total_distance = 19.5 ∧ 
  scenario.total_time = 45 ∧ 
  (scenario.time_in_weather Weather.Sunny + 
   scenario.time_in_weather Weather.Rainy + 
   scenario.time_in_weather Weather.Foggy = scenario.total_time) ∧
  (scenario.speed Weather.Sunny * scenario.time_in_weather Weather.Sunny / 60 +
   scenario.speed Weather.Rainy * scenario.time_in_weather Weather.Rainy / 60 +
   scenario.speed Weather.Foggy * scenario.time_in_weather Weather.Foggy / 60 = 
   scenario.total_distance) →
  scenario.time_in_weather Weather.Foggy = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_shelbys_drive_l1797_179758


namespace NUMINAMATH_CALUDE_composite_sum_l1797_179749

theorem composite_sum (a b : ℕ+) (h : 34 * a = 43 * b) : 
  ∃ (k m : ℕ) (hk : k > 1) (hm : m > 1), a + b = k * m := by
sorry

end NUMINAMATH_CALUDE_composite_sum_l1797_179749


namespace NUMINAMATH_CALUDE_win_sector_area_l1797_179751

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * (π * r^2) = 24 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l1797_179751


namespace NUMINAMATH_CALUDE_baseball_earnings_l1797_179795

/-- The total earnings from two baseball games -/
def total_earnings (saturday_earnings wednesday_earnings : ℚ) : ℚ :=
  saturday_earnings + wednesday_earnings

/-- Theorem stating the total earnings from two baseball games -/
theorem baseball_earnings : 
  ∃ (saturday_earnings wednesday_earnings : ℚ),
    saturday_earnings = 2662.50 ∧
    wednesday_earnings = saturday_earnings - 142.50 ∧
    total_earnings saturday_earnings wednesday_earnings = 5182.50 :=
by sorry

end NUMINAMATH_CALUDE_baseball_earnings_l1797_179795


namespace NUMINAMATH_CALUDE_absolute_value_inequality_rational_inequality_l1797_179712

-- Problem 1
theorem absolute_value_inequality (x : ℝ) :
  (|x - 2| + |2*x - 3| < 4) ↔ (1/3 < x ∧ x < 3) := by sorry

-- Problem 2
theorem rational_inequality (x : ℝ) :
  ((x^2 - 3*x) / (x^2 - x - 2) ≤ x) ↔ 
  ((-1 < x ∧ x ≤ 0) ∨ x = 1 ∨ (2 < x)) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_rational_inequality_l1797_179712


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1797_179784

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := sorry

-- Define points A, B, and C
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_foci_distance :
  -- AB is the major axis of ellipse Γ
  (∀ p ∈ Γ, (p.1 - A.1)^2 + (p.2 - A.2)^2 ≤ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  -- Point C is on Γ
  C ∈ Γ →
  -- Angle CBA = π/4
  Real.arccos ((C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = π/4 →
  -- AB = 4
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 →
  -- BC = √2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 2 →
  -- The distance between the two foci is 4√6/3
  ∃ F₁ F₂ : ℝ × ℝ, F₁ ∈ Γ ∧ F₂ ∈ Γ ∧
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 4 * Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1797_179784


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1797_179796

theorem complex_fraction_simplification (x y z : ℚ) 
  (hx : x = 4)
  (hy : y = 5)
  (hz : z = 2) :
  (1 / z / y) / (1 / x) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1797_179796


namespace NUMINAMATH_CALUDE_projection_matrix_values_l1797_179726

/-- A projection matrix Q satisfies Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The given matrix form -/
def projection_matrix (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 12/25],
    ![y, 13/25]]

/-- Theorem stating that the projection matrix has x = 0 and y = 12/25 -/
theorem projection_matrix_values :
  ∀ x y : ℚ, is_projection_matrix (projection_matrix x y) → x = 0 ∧ y = 12/25 := by
  sorry


end NUMINAMATH_CALUDE_projection_matrix_values_l1797_179726


namespace NUMINAMATH_CALUDE_range_of_a_l1797_179740

theorem range_of_a (P : Set ℝ) (M : Set ℝ) (a : ℝ) 
  (h1 : P = {x : ℝ | x^2 ≤ 1})
  (h2 : M = {a})
  (h3 : P ∪ M = P) : 
  -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1797_179740


namespace NUMINAMATH_CALUDE_initial_sets_count_l1797_179735

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
def num_initial_sets : ℕ := num_letters ^ set_length

theorem initial_sets_count : num_initial_sets = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l1797_179735
