import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_existential_l3305_330514

theorem negation_of_existential (a : ℝ) :
  (¬ ∃ x : ℝ, a * x^2 - 2 * a * x + 1 ≤ 0) ↔ (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_l3305_330514


namespace NUMINAMATH_CALUDE_distance_equality_l3305_330506

/-- Given four points in 3D space, prove that a specific point P satisfies the distance conditions --/
theorem distance_equality (A B C D P : ℝ × ℝ × ℝ) : 
  A = (10, 0, 0) →
  B = (0, -6, 0) →
  C = (0, 0, 8) →
  D = (1, 1, 1) →
  P = (3, -2, 5) →
  dist A P = dist B P ∧ 
  dist A P = dist C P ∧ 
  dist A P = dist D P - 3 := by
  sorry

where
  dist : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ
  | (x₁, y₁, z₁), (x₂, y₂, z₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

end NUMINAMATH_CALUDE_distance_equality_l3305_330506


namespace NUMINAMATH_CALUDE_expression_equals_polynomial_l3305_330543

/-- The given expression is equal to the simplified polynomial for all real x -/
theorem expression_equals_polynomial (x : ℝ) :
  (3 * x^3 + 4 * x^2 - 5 * x + 8) * (x - 2) -
  (x - 2) * (2 * x^3 - 7 * x^2 + 10) +
  (7 * x - 15) * (x - 2) * (2 * x + 1) =
  x^4 + 23 * x^3 - 78 * x^2 + 39 * x + 34 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_polynomial_l3305_330543


namespace NUMINAMATH_CALUDE_equation_system_solution_equality_l3305_330562

theorem equation_system_solution_equality (x y r s : ℝ) : 
  3 * x + 2 * y = 16 →
  5 * x + 3 * y = r →
  5 * x + 3 * y = s →
  r - s = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_equality_l3305_330562


namespace NUMINAMATH_CALUDE_booklet_pages_theorem_l3305_330545

theorem booklet_pages_theorem (n : ℕ) (r : ℕ) : 
  (∃ (n : ℕ) (r : ℕ), 2 * n * (2 * n + 1) / 2 - (4 * r - 1) = 963 ∧ 
   1 ≤ r ∧ r ≤ n) → 
  (n = 22 ∧ r = 7) := by
  sorry

end NUMINAMATH_CALUDE_booklet_pages_theorem_l3305_330545


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l3305_330584

/-- F is a quadratic function of x with parameter m -/
def F (x m : ℚ) : ℚ := (6 * x^2 + 16 * x + 3 * m) / 6

/-- A linear function of x -/
def linear (a b x : ℚ) : ℚ := a * x + b

theorem square_of_linear_expression (m : ℚ) :
  (∃ a b : ℚ, ∀ x : ℚ, F x m = (linear a b x)^2) → m = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l3305_330584


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3305_330549

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, |3*y - 4| ≤ 18 → y ≥ x) → |3*x - 4| ≤ 18 → x = -4 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3305_330549


namespace NUMINAMATH_CALUDE_remaining_distance_to_cave_end_l3305_330565

theorem remaining_distance_to_cave_end (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_cave_end_l3305_330565


namespace NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l3305_330590

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  yellow : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.white + bag.yellow)

/-- The initial contents of the bag -/
def initialBag : BagContents := ⟨2, 3⟩

/-- The contents of the bag after adding balls -/
def finalBag : BagContents := ⟨initialBag.white + 4, initialBag.yellow + 3⟩

/-- Theorem stating that the probabilities are equal after adding balls -/
theorem equal_probabilities_after_adding_balls :
  probability finalBag finalBag.white = probability finalBag finalBag.yellow := by
  sorry

end NUMINAMATH_CALUDE_equal_probabilities_after_adding_balls_l3305_330590


namespace NUMINAMATH_CALUDE_monochromatic_triangle_probability_l3305_330556

/-- The number of vertices in the complete graph -/
def n : ℕ := 6

/-- The number of colors used for coloring the edges -/
def num_colors : ℕ := 3

/-- The probability of a specific triangle being non-monochromatic -/
def p_non_monochromatic : ℚ := 24 / 27

/-- The number of triangles in a complete graph with n vertices -/
def num_triangles : ℕ := n.choose 3

/-- The probability of having at least one monochromatic triangle -/
noncomputable def p_at_least_one_monochromatic : ℚ :=
  1 - p_non_monochromatic ^ num_triangles

theorem monochromatic_triangle_probability :
  p_at_least_one_monochromatic = 872 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_probability_l3305_330556


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3305_330525

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℤ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 ∧ departureLA.minutes = 15 →
  arrivalNY.hours = 18 ∧ arrivalNY.minutes = 25 →
  0 < m ∧ m < 60 →
  timeDifferenceInMinutes departureLA { hours := arrivalNY.hours - 3, minutes := arrivalNY.minutes, valid := sorry } = h * 60 + m →
  h + m = 16 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3305_330525


namespace NUMINAMATH_CALUDE_johns_remaining_money_is_135_l3305_330573

/-- Calculates John's remaining money after dog walking and expenses in April --/
def johns_remaining_money : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (total_days : ℕ) (sundays : ℕ) (weekday_rate : ℕ) (weekend_rate : ℕ)
      (mark_help_days : ℕ) (book_cost : ℕ) (book_discount : ℕ)
      (sister_percentage : ℕ) (gift_cost : ℕ) =>
    let working_days := total_days - sundays
    let weekends := sundays
    let weekdays := working_days - weekends
    let weekday_earnings := weekdays * weekday_rate
    let weekend_earnings := weekends * weekend_rate
    let mark_split_earnings := (mark_help_days * weekday_rate) / 2
    let total_earnings := weekday_earnings + weekend_earnings + mark_split_earnings
    let discounted_book_cost := book_cost - (book_cost * book_discount / 100)
    let after_books := total_earnings - discounted_book_cost
    let sister_share := after_books * sister_percentage / 100
    let after_sister := after_books - sister_share
    let after_gift := after_sister - gift_cost
    let food_cost := weekends * 10
    after_gift - food_cost

theorem johns_remaining_money_is_135 :
  johns_remaining_money 30 4 10 15 3 50 10 20 25 = 135 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_is_135_l3305_330573


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3305_330559

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l3305_330559


namespace NUMINAMATH_CALUDE_expression_value_l3305_330569

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) :
  3 * x - 5 * y + 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3305_330569


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3305_330554

theorem radical_conjugate_sum_product (c d : ℝ) : 
  (c + Real.sqrt d) + (c - Real.sqrt d) = 6 → 
  (c + Real.sqrt d) * (c - Real.sqrt d) = 4 → 
  c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3305_330554


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l3305_330500

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (h1 : original_intensity = 0.6) 
  (h2 : added_intensity = 0.3) 
  (h3 : replaced_fraction = 2/3) : 
  (1 - replaced_fraction) * original_intensity + replaced_fraction * added_intensity = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_intensity_l3305_330500


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3305_330526

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ i / (1 + i) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3305_330526


namespace NUMINAMATH_CALUDE_island_not_named_Maya_l3305_330550

-- Define the inhabitants
inductive Inhabitant : Type
| A : Inhabitant
| B : Inhabitant

-- Define the possible states of an inhabitant
inductive State : Type
| TruthTeller : State
| Liar : State

-- Define the name of the island
def IslandName : Type := Bool

-- Define the statements made by A and B
def statement_A (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∧ state_B = State.Liar) ∧ island_name = true

def statement_B (state_A state_B : State) (island_name : IslandName) : Prop :=
  (state_A = State.Liar ∨ state_B = State.Liar) ∧ island_name = false

-- The main theorem
theorem island_not_named_Maya :
  ∀ (state_A state_B : State) (island_name : IslandName),
    (state_A = State.Liar → ¬statement_A state_A state_B island_name) ∧
    (state_A = State.TruthTeller → statement_A state_A state_B island_name) ∧
    (state_B = State.Liar → ¬statement_B state_A state_B island_name) ∧
    (state_B = State.TruthTeller → statement_B state_A state_B island_name) →
    island_name = false :=
by
  sorry


end NUMINAMATH_CALUDE_island_not_named_Maya_l3305_330550


namespace NUMINAMATH_CALUDE_subtracted_value_l3305_330594

theorem subtracted_value (x y : ℝ) : 
  (x - 5) / 7 = 7 → 
  (x - y) / 10 = 2 → 
  y = 34 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_l3305_330594


namespace NUMINAMATH_CALUDE_line_vector_at_4_l3305_330522

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_4 :
  (∃ (a d : ℝ × ℝ × ℝ),
    (∀ t : ℝ, line_vector t = a + t • d) ∧
    line_vector (-2) = (2, 6, 16) ∧
    line_vector 1 = (-1, -4, -8)) →
  line_vector 4 = (-4, -10, -32) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_at_4_l3305_330522


namespace NUMINAMATH_CALUDE_jim_paycheck_amount_l3305_330575

/-- Calculates the final amount on a paycheck after retirement and tax deductions -/
def final_paycheck_amount (gross_pay : ℝ) (retirement_rate : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_rate) - tax_deduction

/-- Theorem stating that given the specific conditions, the final paycheck amount is $740 -/
theorem jim_paycheck_amount :
  final_paycheck_amount 1120 0.25 100 = 740 := by
  sorry

#eval final_paycheck_amount 1120 0.25 100

end NUMINAMATH_CALUDE_jim_paycheck_amount_l3305_330575


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l3305_330546

theorem fence_cost_per_foot
  (area : ℝ)
  (total_cost : ℝ)
  (h_area : area = 289)
  (h_total_cost : total_cost = 3944) :
  total_cost / (4 * Real.sqrt area) = 58 :=
by sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l3305_330546


namespace NUMINAMATH_CALUDE_two_non_congruent_triangles_l3305_330536

/-- A triangle with integer side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle. -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Check if a triangle satisfies the triangle inequality. -/
def is_valid (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Check if two triangles are congruent. -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid triangles with perimeter 11. -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 11 ∧ is_valid t}

/-- The theorem to be proved. -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    ¬is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ valid_triangles →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end NUMINAMATH_CALUDE_two_non_congruent_triangles_l3305_330536


namespace NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3305_330513

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Checks if a point is on the extension of a line segment -/
def isOnExtension (A B H : Point) : Prop := sorry

/-- Checks if two line segments intersect at a point -/
def intersectsAt (P Q R S J : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_intersection_theorem (EFGH : Parallelogram) (H J K : Point) : 
  isOnExtension EFGH.E EFGH.F H →
  intersectsAt EFGH.G H EFGH.E EFGH.F J →
  intersectsAt EFGH.G H EFGH.F EFGH.G K →
  distance J K = 40 →
  distance H K = 30 →
  distance EFGH.G J = 20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_theorem_l3305_330513


namespace NUMINAMATH_CALUDE_find_m_value_l3305_330501

theorem find_m_value (m : ℤ) : 
  (∃ (x : ℤ), x - m / 3 ≥ 0 ∧ 2 * x - 3 ≥ 3 * (x - 2)) ∧ 
  (∃! (a b : ℤ), a ≠ b ∧ 
    (a - m / 3 ≥ 0 ∧ 2 * a - 3 ≥ 3 * (a - 2)) ∧ 
    (b - m / 3 ≥ 0 ∧ 2 * b - 3 ≥ 3 * (b - 2))) ∧
  (∃ (k : ℤ), k > 0 ∧ 4 * (m + 1) = k * (m^2 - 1)) →
  m = 5 :=
sorry

end NUMINAMATH_CALUDE_find_m_value_l3305_330501


namespace NUMINAMATH_CALUDE_rectangle_problem_l3305_330580

theorem rectangle_problem (num_rectangles : ℕ) (area_large : ℝ) 
  (h1 : num_rectangles = 6)
  (h2 : area_large = 6000) :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    (num_rectangles : ℝ) * (2/5 * x) * x = area_large ∧ 
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l3305_330580


namespace NUMINAMATH_CALUDE_b_age_is_twelve_l3305_330512

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 32
  Prove that b is 12 years old -/
theorem b_age_is_twelve (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 32) : 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_twelve_l3305_330512


namespace NUMINAMATH_CALUDE_power_five_mod_thirteen_l3305_330524

theorem power_five_mod_thirteen : 5^2006 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_thirteen_l3305_330524


namespace NUMINAMATH_CALUDE_power_product_equals_two_l3305_330572

theorem power_product_equals_two :
  (1/2)^2016 * (-2)^2017 * (-1)^2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_two_l3305_330572


namespace NUMINAMATH_CALUDE_choir_average_age_l3305_330586

/-- The average age of people in a choir given the number and average age of females and males -/
theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 28) 
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 30.4 := by
sorry


end NUMINAMATH_CALUDE_choir_average_age_l3305_330586


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l3305_330529

/-- Given two digits X and Y in base d > 8, prove that X - Y = -1 in base d
    when XY + XX = 234 in base d. -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : ℕ) : d > 8 →
  X < d → Y < d →
  (X * d + Y) + (X * d + X) = 2 * d * d + 3 * d + 4 →
  X - Y = d - 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l3305_330529


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3305_330507

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 - a^(x + 1)
  f (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3305_330507


namespace NUMINAMATH_CALUDE_school_raffle_earnings_l3305_330595

/-- The amount of money Zoe's school made from selling raffle tickets -/
def total_money_made (cost_per_ticket : ℕ) (num_tickets_sold : ℕ) : ℕ :=
  cost_per_ticket * num_tickets_sold

/-- Theorem stating that Zoe's school made 620 dollars from selling raffle tickets -/
theorem school_raffle_earnings :
  total_money_made 4 155 = 620 := by
  sorry

end NUMINAMATH_CALUDE_school_raffle_earnings_l3305_330595


namespace NUMINAMATH_CALUDE_meryll_question_ratio_l3305_330591

theorem meryll_question_ratio : 
  ∀ (total_mc : ℕ) (total_ps : ℕ) (written_mc_fraction : ℚ) (remaining : ℕ),
    total_mc = 35 →
    total_ps = 15 →
    written_mc_fraction = 2/5 →
    remaining = 31 →
    (total_mc * written_mc_fraction).num.toNat + 
    (total_ps - (remaining - (total_mc - (total_mc * written_mc_fraction).num.toNat))) = 
    total_ps / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_meryll_question_ratio_l3305_330591


namespace NUMINAMATH_CALUDE_last_round_win_ratio_l3305_330547

/-- Represents the number of matches in a kickboxing competition --/
structure KickboxingCompetition where
  firstTwoRoundsMatches : ℕ  -- Total matches in first two rounds
  lastRoundMatches : ℕ      -- Total matches in last round
  totalWins : ℕ             -- Total matches won by Brendan

/-- Theorem stating the ratio of matches won in the last round --/
theorem last_round_win_ratio (comp : KickboxingCompetition)
  (h1 : comp.firstTwoRoundsMatches = 12)
  (h2 : comp.lastRoundMatches = 4)
  (h3 : comp.totalWins = 14) :
  (comp.totalWins - comp.firstTwoRoundsMatches) * 2 = comp.lastRoundMatches := by
  sorry

#check last_round_win_ratio

end NUMINAMATH_CALUDE_last_round_win_ratio_l3305_330547


namespace NUMINAMATH_CALUDE_correct_urea_decomposing_bacteria_culture_l3305_330531

-- Define the types of culture media
inductive CultureMedium
| SelectiveNitrogen
| IdentificationPhenolRed

-- Define the process of bacterial culture
def BacterialCulture := List CultureMedium

-- Define the property of being a correct culture process
def IsCorrectCulture (process : BacterialCulture) : Prop :=
  process = [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed]

-- Theorem: The correct culture process for urea-decomposing bacteria
theorem correct_urea_decomposing_bacteria_culture :
  IsCorrectCulture [CultureMedium.SelectiveNitrogen, CultureMedium.IdentificationPhenolRed] :=
by sorry

end NUMINAMATH_CALUDE_correct_urea_decomposing_bacteria_culture_l3305_330531


namespace NUMINAMATH_CALUDE_angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l3305_330587

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = t.b * Real.tan t.A ∧
  t.B > Real.pi / 2

-- Theorem 1
theorem angle_B_when_A_is_pi_sixth (t : Triangle) 
  (h : is_valid_triangle t) (h_A : t.A = Real.pi / 6) : 
  t.B = 2 * Real.pi / 3 := 
sorry

-- Theorem 2
theorem sin_A_plus_sin_C_range (t : Triangle) 
  (h : is_valid_triangle t) : 
  Real.sqrt 2 / 2 < Real.sin t.A + Real.sin t.C ∧ 
  Real.sin t.A + Real.sin t.C ≤ 9 / 8 := 
sorry

end NUMINAMATH_CALUDE_angle_B_when_A_is_pi_sixth_sin_A_plus_sin_C_range_l3305_330587


namespace NUMINAMATH_CALUDE_function_existence_condition_l3305_330578

theorem function_existence_condition (k a : ℕ) :
  (∃ f : ℕ → ℕ, ∀ n, (f^[k] n = n + a)) ↔ (a ≥ 0 ∧ k ∣ a) :=
by sorry

end NUMINAMATH_CALUDE_function_existence_condition_l3305_330578


namespace NUMINAMATH_CALUDE_frank_work_days_l3305_330534

/-- Calculates the number of days worked given total hours and hours per day -/
def days_worked (total_hours : Float) (hours_per_day : Float) : Float :=
  total_hours / hours_per_day

/-- Theorem: Frank worked 4 days given the conditions -/
theorem frank_work_days :
  let total_hours : Float := 8.0
  let hours_per_day : Float := 2.0
  days_worked total_hours hours_per_day = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_frank_work_days_l3305_330534


namespace NUMINAMATH_CALUDE_circle_center_sum_l3305_330596

theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 10*x - 4*y + 18 → (x - 5)^2 + (y + 2)^2 = 25 ∧ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3305_330596


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3305_330516

theorem simplify_and_evaluate (a : ℤ) 
  (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a ≠ 0) (h4 : a ≠ 1) (h5 : a ≠ -1) :
  (a - a^2 / (a^2 - 1)) / (a^2 / (a^2 - 1)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3305_330516


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3305_330577

/-- Proves that given a shirt with a list price of 150, a final price of 105 after two successive discounts, and a second discount of 12.5%, the first discount percentage is 20%. -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5) :
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧ 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3305_330577


namespace NUMINAMATH_CALUDE_place_value_ratio_l3305_330555

/-- Represents a decimal number with its integer and fractional parts -/
structure DecimalNumber where
  integerPart : ℕ
  fractionalPart : ℕ
  fractionalDigits : ℕ

/-- Returns the place value of a digit at a given position in a decimal number -/
def placeValue (n : DecimalNumber) (position : ℤ) : ℚ :=
  10 ^ position

/-- The decimal number 50467.8912 -/
def number : DecimalNumber :=
  { integerPart := 50467
  , fractionalPart := 8912
  , fractionalDigits := 4 }

/-- The position of digit 8 in the number (counting from right, negative for fractional part) -/
def pos8 : ℤ := -1

/-- The position of digit 7 in the number (counting from right) -/
def pos7 : ℤ := 1

theorem place_value_ratio :
  (placeValue number pos8) / (placeValue number pos7) = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3305_330555


namespace NUMINAMATH_CALUDE_kaleb_final_amount_l3305_330570

def kaleb_lawn_business (spring_earnings summer_earnings supply_costs : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supply_costs

theorem kaleb_final_amount :
  kaleb_lawn_business 4 50 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_final_amount_l3305_330570


namespace NUMINAMATH_CALUDE_cargo_volume_maximized_l3305_330539

/-- Represents the number of round trips as a function of the number of small boats towed -/
def roundTrips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total cargo volume as a function of the number of small boats towed -/
def cargoVolume (x : ℝ) (M : ℝ) : ℝ := M * x * roundTrips x

theorem cargo_volume_maximized :
  ∀ M : ℝ, M > 0 →
  ∀ x : ℝ, x > 0 →
  cargoVolume 6 M ≥ cargoVolume x M ∧
  roundTrips 4 = 16 ∧
  roundTrips 7 = 10 :=
sorry

end NUMINAMATH_CALUDE_cargo_volume_maximized_l3305_330539


namespace NUMINAMATH_CALUDE_continuous_function_characterization_l3305_330527

theorem continuous_function_characterization
  (f : ℝ → ℝ)
  (hf_continuous : Continuous f)
  (hf_zero : f 0 = 0)
  (hf_ineq : ∀ x y : ℝ, f (x^2 - y^2) ≥ x * f x - y * f y) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_continuous_function_characterization_l3305_330527


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3305_330560

theorem cubic_equation_solution :
  ∃ (x : ℝ), x + x^3 = 10 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3305_330560


namespace NUMINAMATH_CALUDE_room_occupancy_correct_answer_l3305_330532

theorem room_occupancy (num_empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * num_empty_chairs
  let seated_people := (2 * total_chairs) / 3
  let total_people := 2 * seated_people
  total_people

theorem correct_answer : room_occupancy 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_room_occupancy_correct_answer_l3305_330532


namespace NUMINAMATH_CALUDE_fraction_simplification_l3305_330574

theorem fraction_simplification (x : ℝ) (hx : x > 0) :
  (x^(3/4) - 25*x^(1/4)) / (x^(1/2) + 5*x^(1/4)) = x^(1/4) - 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3305_330574


namespace NUMINAMATH_CALUDE_train_speed_l3305_330564

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 240 ∧ 
  bridge_length = 150 ∧ 
  crossing_time = 20 →
  (train_length + bridge_length) / crossing_time * 3.6 = 70.2 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3305_330564


namespace NUMINAMATH_CALUDE_girls_percentage_in_class_l3305_330551

theorem girls_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (girls_ratio : ℚ) / (boys_ratio + girls_ratio) * total_students / total_students * 100 = 57.14 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_in_class_l3305_330551


namespace NUMINAMATH_CALUDE_abc_sum_bound_l3305_330566

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ a' * b' + a' * c' + b' * c' > M ∧
    a * b + a * c + b * c ≤ 1/2 ∧
    a * b + a * c + b * c < 1/2 + ε :=
sorry

end NUMINAMATH_CALUDE_abc_sum_bound_l3305_330566


namespace NUMINAMATH_CALUDE_total_cups_in_trays_l3305_330509

theorem total_cups_in_trays (first_tray second_tray : ℕ) 
  (h1 : second_tray = first_tray - 20) 
  (h2 : second_tray = 240) : 
  first_tray + second_tray = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_in_trays_l3305_330509


namespace NUMINAMATH_CALUDE_supreme_sports_package_channels_prove_supreme_sports_package_channels_l3305_330576

theorem supreme_sports_package_channels : ℕ → Prop :=
  fun x =>
    let initial_channels : ℕ := 150
    let removed_channels : ℕ := 20
    let replaced_channels : ℕ := 12
    let reduced_channels : ℕ := 10
    let sports_package_channels : ℕ := 8
    let final_channels : ℕ := 147
    let intermediate_channels : ℕ := 
      initial_channels - removed_channels + replaced_channels - reduced_channels + sports_package_channels
    x = final_channels - intermediate_channels

theorem prove_supreme_sports_package_channels : 
  supreme_sports_package_channels 7 := by sorry

end NUMINAMATH_CALUDE_supreme_sports_package_channels_prove_supreme_sports_package_channels_l3305_330576


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3305_330533

/-- Given two parabolas, prove that their intersection points lie on a circle with radius squared equal to 16 -/
theorem intersection_points_on_circle (x y : ℝ) : 
  y = (x - 2)^2 ∧ x = (y - 5)^2 - 1 → (x - 2)^2 + (y - 5)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3305_330533


namespace NUMINAMATH_CALUDE_kids_savings_l3305_330538

/-- The total amount saved by three kids given their coin collections -/
def total_savings (teagan_pennies rex_nickels toni_dimes : ℕ) : ℚ :=
  (teagan_pennies : ℚ) * (1 / 100) +
  (rex_nickels : ℚ) * (5 / 100) +
  (toni_dimes : ℚ) * (10 / 100)

/-- Theorem stating that the total savings of the three kids is $40 -/
theorem kids_savings : total_savings 200 100 330 = 40 := by
  sorry

end NUMINAMATH_CALUDE_kids_savings_l3305_330538


namespace NUMINAMATH_CALUDE_excursion_min_parents_l3305_330563

/-- The minimum number of parents needed for an excursion -/
def min_parents_needed (num_students : ℕ) (car_capacity : ℕ) : ℕ :=
  Nat.ceil (num_students / (car_capacity - 1))

/-- Theorem: The minimum number of parents needed for 30 students with 5-seat cars is 8 -/
theorem excursion_min_parents :
  min_parents_needed 30 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_excursion_min_parents_l3305_330563


namespace NUMINAMATH_CALUDE_exp_ln_one_equals_one_l3305_330519

theorem exp_ln_one_equals_one : Real.exp (Real.log 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ln_one_equals_one_l3305_330519


namespace NUMINAMATH_CALUDE_gcd_problems_l3305_330582

theorem gcd_problems :
  (Nat.gcd 63 84 = 21) ∧ (Nat.gcd 351 513 = 27) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l3305_330582


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3305_330517

theorem complex_fraction_equals_negative_two
  (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a*b + b^2 = 0) :
  (a^7 + b^7) / (a + b)^7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l3305_330517


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l3305_330508

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : (100 * A.val + 10 * B.val + A.val) * (10 * B.val + C.val) = 
              1000 * B.val + 100 * C.val + 10 * B.val + C.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : 
  p.A.val + p.C.val = 5 := by sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l3305_330508


namespace NUMINAMATH_CALUDE_smallest_n_for_non_integer_expression_l3305_330579

theorem smallest_n_for_non_integer_expression : ∃ n : ℕ, n > 0 ∧ n = 11 ∧
  ∃ k : ℕ, k < n ∧
    (∀ a m : ℕ, a % n = k ∧ m > 0 →
      ¬(∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1))) ∧
    (∀ n' : ℕ, 0 < n' ∧ n' < n →
      ∀ k' : ℕ, k' < n' →
        ∃ a m : ℕ, a % n' = k' ∧ m > 0 ∧
          ∃ z : ℤ, (a^m + 3^m : ℤ) = z * (a^2 - 3*a + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_smallest_n_for_non_integer_expression_l3305_330579


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3305_330515

theorem simplify_and_evaluate : 
  let x : ℚ := -1
  (1 - 1 / (x - 1)) / ((x^2 - 4) / (x^2 - x)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3305_330515


namespace NUMINAMATH_CALUDE_smallest_n_for_equal_candy_costs_l3305_330510

theorem smallest_n_for_equal_candy_costs : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    ¬∃ (p y o : ℕ), p > 0 ∧ y > 0 ∧ o > 0 ∧ 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * m) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_equal_candy_costs_l3305_330510


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3305_330568

theorem tan_alpha_value (α : Real) (h : Real.cos α + 2 * Real.sin α = Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3305_330568


namespace NUMINAMATH_CALUDE_mom_has_one_eye_l3305_330541

/-- Represents the number of eyes for each family member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  kids_eyes : ℕ
  num_kids : ℕ

/-- The total number of eyes in the monster family -/
def total_eyes (f : MonsterFamily) : ℕ :=
  f.mom_eyes + f.dad_eyes + f.kids_eyes * f.num_kids

/-- Theorem stating that the mom has 1 eye given the conditions -/
theorem mom_has_one_eye (f : MonsterFamily) 
  (h1 : f.dad_eyes = 3)
  (h2 : f.kids_eyes = 4)
  (h3 : f.num_kids = 3)
  (h4 : total_eyes f = 16) : 
  f.mom_eyes = 1 := by
  sorry


end NUMINAMATH_CALUDE_mom_has_one_eye_l3305_330541


namespace NUMINAMATH_CALUDE_inverse_g_84_l3305_330535

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l3305_330535


namespace NUMINAMATH_CALUDE_local_max_value_l3305_330521

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem local_max_value (a : ℝ) :
  (∃ (x : ℝ), x = 2 ∧ IsLocalMin (f a) x) →
  (∃ (y : ℝ), IsLocalMax (f a) y ∧ f a y = 16) :=
by sorry

end NUMINAMATH_CALUDE_local_max_value_l3305_330521


namespace NUMINAMATH_CALUDE_f_composition_value_l3305_330583

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem f_composition_value :
  f (f (π / 12)) = (1 / 2) * Real.sin (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3305_330583


namespace NUMINAMATH_CALUDE_board_theorem_l3305_330557

/-- Represents a board with gold and silver cells. -/
structure Board :=
  (size : ℕ)
  (is_gold : ℕ → ℕ → Bool)

/-- Counts the number of gold cells in a given rectangle of the board. -/
def count_gold (b : Board) (x y w h : ℕ) : ℕ :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold (x + i) (y + j) then 1 else 0))

/-- Checks if the board satisfies the conditions for all 3x3 squares and 2x4/4x2 rectangles. -/
def valid_board (b : Board) (A Z : ℕ) : Prop :=
  (∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A) ∧
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

theorem board_theorem :
  ∀ b : Board, b.size = 2016 →
    (∃ A Z, valid_board b A Z) →
    (∃ A Z, valid_board b A Z ∧ ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8))) :=
sorry

end NUMINAMATH_CALUDE_board_theorem_l3305_330557


namespace NUMINAMATH_CALUDE_solve_equation_l3305_330511

theorem solve_equation (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3305_330511


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3305_330520

theorem quadratic_roots_sum_product (a b : ℝ) : 
  a^2 + a - 1 = 0 → b^2 + b - 1 = 0 → a ≠ b → ab + a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3305_330520


namespace NUMINAMATH_CALUDE_volume_between_spheres_l3305_330567

theorem volume_between_spheres (π : ℝ) (h : π > 0) :
  let volume_sphere (r : ℝ) := (4 / 3) * π * r^3
  (volume_sphere 10 - volume_sphere 4) = (3744 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_volume_between_spheres_l3305_330567


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_power_of_three_l3305_330585

theorem smallest_k_divisible_by_power_of_three : ∃ k : ℕ, 
  (∀ m : ℕ, m < k → ¬(3^67 ∣ 2016^m)) ∧ (3^67 ∣ 2016^k) ∧ k = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_power_of_three_l3305_330585


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3305_330552

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is an even function with range (-∞, 4] -/
def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of f is (-∞, 4] -/
def has_range_neg_inf_to_4 (f : ℝ → ℝ) : Prop := 
  (∀ y, y ≤ 4 → ∃ x, f x = y) ∧ (∀ x, f x ≤ 4)

theorem quadratic_function_theorem (a b : ℝ) : 
  is_even_function (f · a b) → has_range_neg_inf_to_4 (f · a b) → 
  ∀ x, f x a b = -2 * x^2 + 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3305_330552


namespace NUMINAMATH_CALUDE_days_to_finish_book_l3305_330540

theorem days_to_finish_book (total_pages book_chapters pages_per_day : ℕ) : 
  total_pages = 193 → book_chapters = 15 → pages_per_day = 44 → 
  (total_pages + pages_per_day - 1) / pages_per_day = 5 := by
sorry

end NUMINAMATH_CALUDE_days_to_finish_book_l3305_330540


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l3305_330523

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def drawn : ℕ := 3

/-- The probability of drawing at least one white ball -/
theorem prob_at_least_one_white :
  (1 - (Nat.choose num_red drawn / Nat.choose total_balls drawn : ℚ)) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l3305_330523


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l3305_330504

theorem unique_solution_cube_equation (x : ℝ) (h : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 ↔ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l3305_330504


namespace NUMINAMATH_CALUDE_parabola_param_valid_l3305_330558

/-- A parameterization of the curve y = x^2 -/
def parabola_param (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The curve y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

theorem parabola_param_valid :
  ∀ (x : ℝ), ∃ (t : ℝ), parabola_param t = (x, parabola x) :=
sorry

end NUMINAMATH_CALUDE_parabola_param_valid_l3305_330558


namespace NUMINAMATH_CALUDE_new_average_age_l3305_330598

def initial_people : ℕ := 6
def initial_average_age : ℚ := 25
def leaving_age : ℕ := 20
def entering_age : ℕ := 30

theorem new_average_age :
  let initial_total_age : ℚ := initial_people * initial_average_age
  let new_total_age : ℚ := initial_total_age - leaving_age + entering_age
  new_total_age / initial_people = 26.67 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_l3305_330598


namespace NUMINAMATH_CALUDE_scientific_notation_3120000_l3305_330561

theorem scientific_notation_3120000 :
  3120000 = 3.12 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3120000_l3305_330561


namespace NUMINAMATH_CALUDE_percentage_of_x_l3305_330530

theorem percentage_of_x (x y z : ℚ) : 
  x / y = 4 → 
  x + y = z → 
  y ≠ 0 → 
  z > 0 → 
  (2 * x - y) / x = 175 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l3305_330530


namespace NUMINAMATH_CALUDE_residue_calculation_l3305_330589

theorem residue_calculation : (240 * 15 - 21 * 9 + 6) % 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l3305_330589


namespace NUMINAMATH_CALUDE_mountain_has_three_sections_l3305_330581

/-- Given a mountain with eagles, calculate the number of sections. -/
def mountain_sections (eagles_per_section : ℕ) (total_eagles : ℕ) : ℕ :=
  total_eagles / eagles_per_section

/-- Theorem: The mountain has 3 sections given the specified conditions. -/
theorem mountain_has_three_sections :
  let eagles_per_section := 6
  let total_eagles := 18
  mountain_sections eagles_per_section total_eagles = 3 := by
  sorry

end NUMINAMATH_CALUDE_mountain_has_three_sections_l3305_330581


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l3305_330597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

theorem fixed_point_power_function 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (α : ℝ) 
  (h3 : ∀ x : ℝ, (x : ℝ)^α = x^α) -- To ensure g is a power function
  (h4 : (2 : ℝ)^α = 1/4) -- g passes through (2, 1/4)
  : (1/2 : ℝ)^α = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l3305_330597


namespace NUMINAMATH_CALUDE_sanoop_tshirts_l3305_330542

/-- The number of t-shirts Sanoop initially bought -/
def initial_tshirts : ℕ := 8

/-- The initial average price of t-shirts in Rs -/
def initial_avg_price : ℚ := 526

/-- The average price of t-shirts after returning one, in Rs -/
def new_avg_price : ℚ := 505

/-- The price of the returned t-shirt in Rs -/
def returned_price : ℚ := 673

theorem sanoop_tshirts :
  initial_tshirts = 8 ∧
  initial_avg_price * initial_tshirts = 
    new_avg_price * (initial_tshirts - 1) + returned_price :=
by sorry

end NUMINAMATH_CALUDE_sanoop_tshirts_l3305_330542


namespace NUMINAMATH_CALUDE_pauls_shopping_bill_l3305_330503

def dress_shirt_price : ℝ := 15.00
def pants_price : ℝ := 40.00
def suit_price : ℝ := 150.00
def sweater_price : ℝ := 30.00

def num_dress_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.20
def coupon_discount : ℝ := 0.10

def total_before_discount : ℝ := 
  dress_shirt_price * num_dress_shirts +
  pants_price * num_pants +
  suit_price * num_suits +
  sweater_price * num_sweaters

def final_price : ℝ := 
  total_before_discount * (1 - store_discount) * (1 - coupon_discount)

theorem pauls_shopping_bill : final_price = 252.00 := by
  sorry

end NUMINAMATH_CALUDE_pauls_shopping_bill_l3305_330503


namespace NUMINAMATH_CALUDE_thief_hiding_speeds_l3305_330599

/-- Configuration of roads, houses, and police movement --/
structure Configuration where
  road_distance : ℝ
  house_size : ℝ
  house_spacing : ℝ
  house_road_distance : ℝ
  police_speed : ℝ
  police_interval : ℝ

/-- Thief's movement relative to police --/
inductive ThiefMovement
  | Opposite
  | Same

/-- Proposition that the thief can stay hidden --/
def can_stay_hidden (config : Configuration) (thief_speed : ℝ) (direction : ThiefMovement) : Prop :=
  match direction with
  | ThiefMovement.Opposite => thief_speed = 2 * config.police_speed
  | ThiefMovement.Same => thief_speed = config.police_speed / 2

/-- Theorem stating the only two viable speeds for the thief --/
theorem thief_hiding_speeds (config : Configuration) 
  (h1 : config.road_distance = 30)
  (h2 : config.house_size = 10)
  (h3 : config.house_spacing = 20)
  (h4 : config.house_road_distance = 10)
  (h5 : config.police_interval = 90)
  (thief_speed : ℝ)
  (direction : ThiefMovement) :
  can_stay_hidden config thief_speed direction ↔ 
    (thief_speed = 2 * config.police_speed ∧ direction = ThiefMovement.Opposite) ∨
    (thief_speed = config.police_speed / 2 ∧ direction = ThiefMovement.Same) :=
  sorry

end NUMINAMATH_CALUDE_thief_hiding_speeds_l3305_330599


namespace NUMINAMATH_CALUDE_sum_of_digits_in_special_number_l3305_330553

theorem sum_of_digits_in_special_number (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  A ≠ B → A ≠ C → A ≠ D → A ≠ E → 
  B ≠ C → B ≠ D → B ≠ E → 
  C ≠ D → C ≠ E → 
  D ≠ E →
  (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E) % 9 = 0 →
  A + B + C + D + E = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_special_number_l3305_330553


namespace NUMINAMATH_CALUDE_spherical_shell_surface_area_l3305_330537

/-- The surface area of a spherical shell formed by two hemispheres -/
theorem spherical_shell_surface_area 
  (r : ℝ) -- radius of the inner hemisphere
  (h1 : r > 0) -- radius is positive
  (h2 : r^2 * π = 200 * π) -- base area of inner hemisphere is 200π
  : 2 * π * ((r + 1)^2 - r^2) = 2 * π + 40 * Real.sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_spherical_shell_surface_area_l3305_330537


namespace NUMINAMATH_CALUDE_eleven_bonnets_per_orphanage_l3305_330505

/-- The number of bonnets Mrs. Young makes in a week and distributes to orphanages -/
def bonnet_distribution (monday : ℕ) : ℕ → ℕ :=
  fun orphanages =>
    let tuesday_wednesday := 2 * monday
    let thursday := monday + 5
    let friday := thursday - 5
    let total := monday + tuesday_wednesday + thursday + friday
    total / orphanages

/-- Theorem stating that given the conditions in the problem, each orphanage receives 11 bonnets -/
theorem eleven_bonnets_per_orphanage :
  bonnet_distribution 10 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_bonnets_per_orphanage_l3305_330505


namespace NUMINAMATH_CALUDE_product_of_17_terms_geometric_sequence_l3305_330571

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the product of the first n terms of a sequence
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * a (i + 1)) 1

-- Theorem statement
theorem product_of_17_terms_geometric_sequence 
  (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_a9 : a 9 = -2) :
  product_of_first_n_terms a 17 = -2^17 := by
  sorry

end NUMINAMATH_CALUDE_product_of_17_terms_geometric_sequence_l3305_330571


namespace NUMINAMATH_CALUDE_expand_expression_l3305_330588

theorem expand_expression (x : ℝ) : (2 * x + 1) * (x - 2) = 2 * x^2 - 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3305_330588


namespace NUMINAMATH_CALUDE_rectangle_area_l3305_330502

/-- Given a rectangle ABCD with the following properties:
  - Sides AB and CD have length 3x
  - Sides AD and BC have length x
  - A circle with radius r is tangent to side AB at its midpoint, AD, and CD
  - 2r = x
  Prove that the area of rectangle ABCD is 12r^2 -/
theorem rectangle_area (x r : ℝ) (h1 : 2 * r = x) : 3 * x * x = 12 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3305_330502


namespace NUMINAMATH_CALUDE_min_value_expression_l3305_330548

theorem min_value_expression (x y : ℝ) : 
  x^2 - 2*x*y + y^2 + 2*y + 1 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 - 2*a*b + b^2 + 2*b + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3305_330548


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3305_330592

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 3
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -1 ∧ f x1 = 0 ∧ f x2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3305_330592


namespace NUMINAMATH_CALUDE_equation_solution_l3305_330518

theorem equation_solution :
  ∃ y : ℚ, 3 * y^(1/4) - 5 * (y / y^(3/4)) = 2 + y^(1/4) ∧ y = 16/81 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3305_330518


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3305_330593

theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 6) (h2 : b = 2) :
  2 * Real.sqrt (a^2 - b^2) = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3305_330593


namespace NUMINAMATH_CALUDE_gcf_of_21_and_12_l3305_330528

theorem gcf_of_21_and_12 (h : Nat.lcm 21 12 = 42) : Nat.gcd 21 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_21_and_12_l3305_330528


namespace NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_for_not_p_true_l3305_330544

theorem either_false_sufficient_not_necessary_for_not_p_true (p q : Prop) :
  (((¬p ∧ ¬q) → ¬p) ∧ ∃ (r : Prop), (¬r ∧ ¬(¬r ∧ ¬q))) := by
  sorry

end NUMINAMATH_CALUDE_either_false_sufficient_not_necessary_for_not_p_true_l3305_330544
