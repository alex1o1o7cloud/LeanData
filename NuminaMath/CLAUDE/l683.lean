import Mathlib

namespace NUMINAMATH_CALUDE_transformed_equation_solutions_l683_68385

theorem transformed_equation_solutions
  (h : ∀ x : ℝ, x^2 + 2*x - 3 = 0 ↔ x = 1 ∨ x = -3) :
  ∀ x : ℝ, (x + 3)^2 + 2*(x + 3) - 3 = 0 ↔ x = -2 ∨ x = -6 := by
sorry

end NUMINAMATH_CALUDE_transformed_equation_solutions_l683_68385


namespace NUMINAMATH_CALUDE_original_fraction_proof_l683_68322

theorem original_fraction_proof (x y : ℚ) : 
  (1.15 * x) / (0.92 * y) = 15 / 16 → x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_proof_l683_68322


namespace NUMINAMATH_CALUDE_section_b_average_weight_l683_68318

/-- Given a class with two sections, prove the average weight of section B --/
theorem section_b_average_weight
  (total_students : ℕ)
  (section_a_students : ℕ)
  (section_b_students : ℕ)
  (section_a_avg_weight : ℝ)
  (total_avg_weight : ℝ)
  (h1 : total_students = section_a_students + section_b_students)
  (h2 : section_a_students = 50)
  (h3 : section_b_students = 50)
  (h4 : section_a_avg_weight = 60)
  (h5 : total_avg_weight = 70)
  : (total_students * total_avg_weight - section_a_students * section_a_avg_weight) / section_b_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l683_68318


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l683_68337

theorem smallest_number_with_remainders : ∃! a : ℕ+, 
  (a : ℤ) % 4 = 1 ∧ 
  (a : ℤ) % 3 = 1 ∧ 
  (a : ℤ) % 5 = 2 ∧ 
  (∀ n : ℕ+, n < a → ((n : ℤ) % 4 ≠ 1 ∨ (n : ℤ) % 3 ≠ 1 ∨ (n : ℤ) % 5 ≠ 2)) ∧
  a = 37 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l683_68337


namespace NUMINAMATH_CALUDE_quotient_of_arctangents_eq_one_l683_68374

theorem quotient_of_arctangents_eq_one :
  (π - Real.arctan (8/15)) / (2 * Real.arctan 4) = 1 := by sorry

end NUMINAMATH_CALUDE_quotient_of_arctangents_eq_one_l683_68374


namespace NUMINAMATH_CALUDE_three_tribes_at_campfire_l683_68341

/-- Represents a native at the campfire -/
structure Native where
  tribe : ℕ

/-- Represents the circle of natives around the campfire -/
def Campfire := Vector Native 7

/-- Check if a native tells the truth to their left neighbor -/
def tellsTruth (c : Campfire) (i : Fin 7) : Prop :=
  (c.get i).tribe = (c.get ((i + 1) % 7)).tribe →
    (∀ j : Fin 7, j ≠ i ∧ j ≠ ((i + 1) % 7) → (c.get j).tribe ≠ (c.get i).tribe)

/-- The main theorem: there are exactly 3 tribes represented at the campfire -/
theorem three_tribes_at_campfire (c : Campfire) 
  (h : ∀ i : Fin 7, tellsTruth c i) :
  ∃! n : ℕ, n = 3 ∧ (∀ t : ℕ, (∃ i : Fin 7, (c.get i).tribe = t) → t ≤ n) :=
sorry

end NUMINAMATH_CALUDE_three_tribes_at_campfire_l683_68341


namespace NUMINAMATH_CALUDE_remainder_11_pow_101_mod_7_l683_68316

theorem remainder_11_pow_101_mod_7 : 11^101 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_101_mod_7_l683_68316


namespace NUMINAMATH_CALUDE_inequality_proof_minimum_value_proof_l683_68377

-- Define the variables and conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Part I: Prove the inequality
theorem inequality_proof : Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) ≤ 4 := by
  sorry

-- Part II: Prove the minimum value
theorem minimum_value_proof : ∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1 / (a + 1) + 1 / (b + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_minimum_value_proof_l683_68377


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_cubed_l683_68376

theorem floor_negative_seven_fourths_cubed : ⌊(-7/4)^3⌋ = -6 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_cubed_l683_68376


namespace NUMINAMATH_CALUDE_work_problem_solution_l683_68366

/-- Proves that given the conditions of the work problem, c worked for 4 days -/
theorem work_problem_solution :
  let a_days : ℕ := 16
  let b_days : ℕ := 9
  let c_wage : ℚ := 71.15384615384615
  let total_earning : ℚ := 1480
  let wage_ratio_a : ℚ := 3
  let wage_ratio_b : ℚ := 4
  let wage_ratio_c : ℚ := 5
  let a_wage : ℚ := (wage_ratio_a / wage_ratio_c) * c_wage
  let b_wage : ℚ := (wage_ratio_b / wage_ratio_c) * c_wage
  ∃ c_days : ℕ,
    c_days * c_wage + a_days * a_wage + b_days * b_wage = total_earning ∧
    c_days = 4 :=
by sorry

end NUMINAMATH_CALUDE_work_problem_solution_l683_68366


namespace NUMINAMATH_CALUDE_math_contest_problem_count_l683_68381

/-- Represents the number of problems solved by each participant -/
structure ParticipantSolutions where
  neznayka : ℕ
  pilyulkin : ℕ
  knopochka : ℕ
  vintik : ℕ
  znayka : ℕ

/-- Defines the conditions of the math contest -/
def MathContest (n : ℕ) (solutions : ParticipantSolutions) : Prop :=
  solutions.neznayka = 6 ∧
  solutions.znayka = 10 ∧
  solutions.pilyulkin > solutions.neznayka ∧
  solutions.pilyulkin < solutions.znayka ∧
  solutions.knopochka > solutions.neznayka ∧
  solutions.knopochka < solutions.znayka ∧
  solutions.vintik > solutions.neznayka ∧
  solutions.vintik < solutions.znayka ∧
  solutions.neznayka + solutions.pilyulkin + solutions.knopochka + solutions.vintik + solutions.znayka = 4 * n

theorem math_contest_problem_count (solutions : ParticipantSolutions) :
  ∃ n : ℕ, MathContest n solutions → n = 10 :=
by sorry

end NUMINAMATH_CALUDE_math_contest_problem_count_l683_68381


namespace NUMINAMATH_CALUDE_usual_work_week_l683_68321

/-- Proves that given the conditions, the employee's usual work week is 40 hours -/
theorem usual_work_week (hourly_rate : ℝ) (weekly_salary : ℝ) (worked_fraction : ℝ) :
  hourly_rate = 15 →
  weekly_salary = 480 →
  worked_fraction = 4 / 5 →
  worked_fraction * (weekly_salary / hourly_rate) = 40 := by
sorry

end NUMINAMATH_CALUDE_usual_work_week_l683_68321


namespace NUMINAMATH_CALUDE_election_win_margin_l683_68303

theorem election_win_margin 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (winner_votes : ℕ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 744)
  (h3 : ↑winner_votes = winner_percentage * ↑total_votes) :
  winner_votes - (total_votes - winner_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_win_margin_l683_68303


namespace NUMINAMATH_CALUDE_students_walking_home_l683_68358

theorem students_walking_home (total : ℚ) (bus auto bike scooter : ℚ) : 
  total = 1 →
  bus = 1/3 →
  auto = 1/5 →
  bike = 1/8 →
  scooter = 1/10 →
  total - (bus + auto + bike + scooter) = 29/120 :=
by sorry

end NUMINAMATH_CALUDE_students_walking_home_l683_68358


namespace NUMINAMATH_CALUDE_joe_monthly_income_correct_l683_68347

/-- Joe's monthly income in dollars -/
def monthly_income : ℝ := 2120

/-- The fraction of Joe's income that goes to taxes -/
def tax_rate : ℝ := 0.4

/-- The amount Joe pays in taxes each month in dollars -/
def monthly_tax : ℝ := 848

/-- Theorem stating that Joe's monthly income is correct given the tax rate and monthly tax amount -/
theorem joe_monthly_income_correct : 
  tax_rate * monthly_income = monthly_tax := by sorry

end NUMINAMATH_CALUDE_joe_monthly_income_correct_l683_68347


namespace NUMINAMATH_CALUDE_rearrangement_divisible_by_seven_l683_68324

/-- A function that checks if a natural number contains the digits 1, 3, 7, and 9 -/
def containsRequiredDigits (n : ℕ) : Prop := sorry

/-- A function that represents all possible rearrangements of digits in a natural number -/
def rearrangeDigits (n : ℕ) : Set ℕ := sorry

/-- Theorem: For any natural number containing the digits 1, 3, 7, and 9,
    there exists a rearrangement of its digits that is divisible by 7 -/
theorem rearrangement_divisible_by_seven (n : ℕ) :
  containsRequiredDigits n →
  ∃ m ∈ rearrangeDigits n, m % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_rearrangement_divisible_by_seven_l683_68324


namespace NUMINAMATH_CALUDE_negation_of_every_constant_is_geometric_l683_68319

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is constant if all its terms are equal. -/
def IsConstant (s : Sequence) : Prop := ∀ n m : ℕ, s n = s m

/-- A sequence is geometric if the ratio between any two consecutive terms is constant and nonzero. -/
def IsGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = r * s n

/-- The statement "Every constant sequence is a geometric sequence" -/
def EveryConstantIsGeometric : Prop :=
  ∀ s : Sequence, IsConstant s → IsGeometric s

/-- The negation of "Every constant sequence is a geometric sequence" -/
theorem negation_of_every_constant_is_geometric :
  ¬EveryConstantIsGeometric ↔ ∃ s : Sequence, IsConstant s ∧ ¬IsGeometric s :=
by
  sorry


end NUMINAMATH_CALUDE_negation_of_every_constant_is_geometric_l683_68319


namespace NUMINAMATH_CALUDE_absolute_sum_nonzero_iff_either_nonzero_l683_68356

theorem absolute_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  |x| + |y| ≠ 0 ↔ x ≠ 0 ∨ y ≠ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_sum_nonzero_iff_either_nonzero_l683_68356


namespace NUMINAMATH_CALUDE_irrational_x_with_rational_expressions_l683_68343

theorem irrational_x_with_rational_expressions (x : ℝ) :
  Irrational x →
  ∃ q₁ q₂ : ℚ, (x^3 - 6*x : ℝ) = (q₁ : ℝ) ∧ (x^4 - 8*x^2 : ℝ) = (q₂ : ℝ) →
  x = Real.sqrt 6 ∨ x = -Real.sqrt 6 ∨
  x = 1 + Real.sqrt 3 ∨ x = -(1 + Real.sqrt 3) ∨
  x = 1 - Real.sqrt 3 ∨ x = -(1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_irrational_x_with_rational_expressions_l683_68343


namespace NUMINAMATH_CALUDE_square_pieces_count_l683_68368

/-- Represents a square sheet of paper -/
structure SquareSheet :=
  (side : ℝ)
  (area : ℝ := side * side)

/-- Represents the state of the paper after folding and cutting -/
structure FoldedCutSheet :=
  (original : SquareSheet)
  (num_folds : ℕ)
  (num_cuts : ℕ)

/-- Counts the number of square pieces after unfolding -/
def count_square_pieces (sheet : FoldedCutSheet) : ℕ :=
  sorry

/-- Theorem stating that folding a square sheet twice and cutting twice results in 5 square pieces -/
theorem square_pieces_count (s : SquareSheet) :
  let folded_cut := FoldedCutSheet.mk s 2 2
  count_square_pieces folded_cut = 5 :=
sorry

end NUMINAMATH_CALUDE_square_pieces_count_l683_68368


namespace NUMINAMATH_CALUDE_hotel_profit_equation_l683_68389

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  totalRooms : ℕ
  basePrice : ℕ
  priceStep : ℕ
  costPerRoom : ℕ

/-- Calculates the number of occupied rooms based on the current price -/
def occupiedRooms (h : Hotel) (price : ℕ) : ℕ :=
  h.totalRooms - (price - h.basePrice) / h.priceStep

/-- Calculates the profit for a given price -/
def profit (h : Hotel) (price : ℕ) : ℕ :=
  (price - h.costPerRoom) * occupiedRooms h price

/-- Theorem stating that the given equation correctly represents the hotel's profit -/
theorem hotel_profit_equation (desiredProfit : ℕ) :
  let h : Hotel := {
    totalRooms := 50,
    basePrice := 180,
    priceStep := 10,
    costPerRoom := 20
  }
  ∀ x : ℕ, profit h x = desiredProfit ↔ (x - 20) * (50 - (x - 180) / 10) = desiredProfit :=
by sorry

end NUMINAMATH_CALUDE_hotel_profit_equation_l683_68389


namespace NUMINAMATH_CALUDE_range_of_f_l683_68333

/-- A monotonically increasing odd function f with f(1) = 2 and f(2) = 3 -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonically increasing -/
axiom f_increasing (x y : ℝ) : x < y → f x < f y

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(1) = 2 -/
axiom f_1 : f 1 = 2

/-- f(2) = 3 -/
axiom f_2 : f 2 = 3

/-- The main theorem -/
theorem range_of_f (x : ℝ) : 
  (-3 < f (x - 3) ∧ f (x - 3) < 2) ↔ (1 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l683_68333


namespace NUMINAMATH_CALUDE_celebration_day_l683_68302

/-- Given a person born on a Friday, their 1200th day of life will fall on a Saturday -/
theorem celebration_day (birth_day : Nat) (birth_weekday : Nat) : 
  birth_weekday = 5 → (birth_day + 1199) % 7 = 6 := by
  sorry

#check celebration_day

end NUMINAMATH_CALUDE_celebration_day_l683_68302


namespace NUMINAMATH_CALUDE_tangent_circles_bisector_l683_68314

-- Define the basic geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def Point := ℝ × ℝ

-- Define the geometric relations
def tangentCircles (c1 c2 : Circle) (p : Point) : Prop := sorry

def tangentLineToCircle (l : Line) (c : Circle) (a : Point) : Prop := sorry

def lineIntersectsCircle (l : Line) (c : Circle) (b c : Point) : Prop := sorry

def angleBisector (l : Line) (a b c : Point) : Prop := sorry

-- State the theorem
theorem tangent_circles_bisector
  (c1 c2 : Circle) (p a b c : Point) (d : Line) :
  tangentCircles c1 c2 p →
  tangentLineToCircle d c1 a →
  lineIntersectsCircle d c2 b c →
  angleBisector (Line.mk p a) p b c := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_bisector_l683_68314


namespace NUMINAMATH_CALUDE_car_service_month_l683_68348

/-- Represents the months of the year -/
inductive Month : Type
| jan | feb | mar | apr | may | jun | jul | aug | sep | oct | nov | dec

/-- Convert a number to a month -/
def num_to_month (n : Nat) : Month :=
  match n % 12 with
  | 1 => Month.jan
  | 2 => Month.feb
  | 3 => Month.mar
  | 4 => Month.apr
  | 5 => Month.may
  | 6 => Month.jun
  | 7 => Month.jul
  | 8 => Month.aug
  | 9 => Month.sep
  | 10 => Month.oct
  | 11 => Month.nov
  | _ => Month.dec

theorem car_service_month (service_interval : Nat) (first_service : Month) (n : Nat) :
  service_interval = 7 →
  first_service = Month.jan →
  n = 30 →
  num_to_month ((n - 1) * service_interval % 12 + 1) = Month.dec :=
by
  sorry

end NUMINAMATH_CALUDE_car_service_month_l683_68348


namespace NUMINAMATH_CALUDE_square_sum_given_difference_l683_68361

theorem square_sum_given_difference (a : ℝ) (h : a - 1/a = 3) : (a + 1/a)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_l683_68361


namespace NUMINAMATH_CALUDE_dog_weight_ratio_l683_68353

/-- Represents the weight of a dog at different ages --/
structure DogWeight where
  week7 : ℝ
  week9 : ℝ
  month3 : ℝ
  month5 : ℝ
  year1 : ℝ

/-- Theorem stating the ratio of a dog's weight at 9 weeks to 7 weeks --/
theorem dog_weight_ratio (w : DogWeight) 
  (h1 : w.week7 = 6)
  (h2 : w.month3 = 2 * w.week9)
  (h3 : w.month5 = 2 * w.month3)
  (h4 : w.year1 = w.month5 + 30)
  (h5 : w.year1 = 78) :
  w.week9 / w.week7 = 2 := by
  sorry

#check dog_weight_ratio

end NUMINAMATH_CALUDE_dog_weight_ratio_l683_68353


namespace NUMINAMATH_CALUDE_reverse_digit_integers_l683_68304

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit positive integer
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit positive integer
  (q.div 10 = r.mod 10 ∧ q.mod 10 = r.div 10) →  -- q and r have the same digits in reverse order
  (q > r → q - r < 20) →  -- positive difference is less than 20
  (r > q → r - q < 20) →  -- positive difference is less than 20
  (∀ a b : ℕ, (a ≥ 10 ∧ a < 100) → (b ≥ 10 ∧ b < 100) → 
    (a.div 10 = b.mod 10 ∧ a.mod 10 = b.div 10) → (a - b ≤ 18)) →  -- greatest possible difference is 18
  (q.div 10 = q.mod 10 + 2) →  -- tens digit is 2 more than units digit for q
  (r.div 10 + 2 = r.mod 10) -- tens digit is 2 more than units digit for r (reverse of q)
  := by sorry

end NUMINAMATH_CALUDE_reverse_digit_integers_l683_68304


namespace NUMINAMATH_CALUDE_total_spending_theorem_l683_68369

/-- Represents the spending of Terry, Maria, and Raj over a week -/
structure WeeklySpending where
  terry : List Float
  maria : List Float
  raj : List Float

/-- Calculates the total spending of all three people over the week -/
def totalSpending (ws : WeeklySpending) : Float :=
  (ws.terry.sum + ws.maria.sum + ws.raj.sum)

/-- Theorem stating that the total spending equals $752.50 -/
theorem total_spending_theorem (ws : WeeklySpending) : 
  ws.terry = [6, 12, 36, 18, 14, 21, 33] ∧ 
  ws.maria = [3, 10, 72, 8, 14, 12, 33] ∧ 
  ws.raj = [7.5, 10, 216, 108, 14, 21, 84] → 
  totalSpending ws = 752.5 := by
  sorry

#eval totalSpending {
  terry := [6, 12, 36, 18, 14, 21, 33],
  maria := [3, 10, 72, 8, 14, 12, 33],
  raj := [7.5, 10, 216, 108, 14, 21, 84]
}

end NUMINAMATH_CALUDE_total_spending_theorem_l683_68369


namespace NUMINAMATH_CALUDE_average_rounds_is_four_l683_68309

/-- Represents the distribution of golfers and rounds played --/
structure GolfData :=
  (rounds : Fin 6 → ℕ)
  (golfers : Fin 6 → ℕ)

/-- Calculates the average number of rounds played, rounded to the nearest whole number --/
def averageRoundsRounded (data : GolfData) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => (data.rounds i.succ) * (data.golfers i.succ))
  let totalGolfers := (Finset.range 6).sum (λ i => data.golfers i.succ)
  (totalRounds + totalGolfers / 2) / totalGolfers

/-- The given golf data --/
def givenData : GolfData :=
  { rounds := λ i => i,
    golfers := λ i => match i with
      | 1 => 6
      | 2 => 3
      | 3 => 2
      | 4 => 4
      | 5 => 6
      | 6 => 4 }

theorem average_rounds_is_four :
  averageRoundsRounded givenData = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rounds_is_four_l683_68309


namespace NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l683_68378

theorem not_prime_n4_plus_n2_plus_1 (n : ℤ) (h : n ≥ 2) :
  ¬(Nat.Prime (n^4 + n^2 + 1).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_plus_n2_plus_1_l683_68378


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_33_l683_68391

theorem inverse_of_5_mod_33 : ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 := by
  use 20
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_33_l683_68391


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l683_68354

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 1 x ≤ 7} = Set.Iic 4 := by sorry

-- Part 2
theorem range_of_a_part2 : 
  {a : ℝ | ∀ x, f a x ≥ 2*a + 1} = Set.Iic (-1) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l683_68354


namespace NUMINAMATH_CALUDE_intersection_count_l683_68315

-- Define the two curves
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6
def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

-- Define a function to count distinct intersection points
def count_distinct_intersections : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem intersection_count :
  count_distinct_intersections = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l683_68315


namespace NUMINAMATH_CALUDE_average_sour_candies_is_correct_l683_68339

/-- The number of people in the group -/
def num_people : ℕ := 4

/-- The number of sour candies Wendy's brother has -/
def brother_sour_candies : ℕ := 4

/-- The number of sour candies Wendy has -/
def wendy_sour_candies : ℕ := 5

/-- The number of sour candies their cousin has -/
def cousin_sour_candies : ℕ := 1

/-- The number of sour candies their uncle has -/
def uncle_sour_candies : ℕ := 3

/-- The total number of sour candies -/
def total_sour_candies : ℕ := brother_sour_candies + wendy_sour_candies + cousin_sour_candies + uncle_sour_candies

/-- The average number of sour candies per person -/
def average_sour_candies : ℚ := total_sour_candies / num_people

theorem average_sour_candies_is_correct : average_sour_candies = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_average_sour_candies_is_correct_l683_68339


namespace NUMINAMATH_CALUDE_book_cd_price_difference_l683_68308

/-- Proves that the difference between book price and CD price is $4 -/
theorem book_cd_price_difference :
  let album_price : ℝ := 20
  let cd_price : ℝ := 0.7 * album_price
  let book_price : ℝ := 18
  book_price - cd_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_book_cd_price_difference_l683_68308


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l683_68388

/-- Given a triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 5,
    the area of the triangle is 18.75. -/
theorem triangle_area_in_circle (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 5 →  -- Circle radius is 5
  b = (3/2) * a →  -- Side length ratio 2:3
  c = 2 * a →  -- Side length ratio 2:4
  c = 2 * r →  -- Diameter of the circle
  (1/2) * a * b = 18.75 :=  -- Area of the triangle
by sorry


end NUMINAMATH_CALUDE_triangle_area_in_circle_l683_68388


namespace NUMINAMATH_CALUDE_younger_person_age_l683_68332

/-- 
Given two persons whose ages differ by 20 years, and 10 years ago the elder was 5 times as old as the younger,
prove that the present age of the younger person is 15 years.
-/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 10 = 5 * (y - 10) → 
  y = 15 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_l683_68332


namespace NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l683_68325

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l683_68325


namespace NUMINAMATH_CALUDE_find_a_l683_68349

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}
def B : Set ℝ := Set.Ioo (-3) 2

-- Define the property that A ∩ B is the solution set of x^2 + ax + b < 0
def is_solution_set (a b : ℝ) : Prop :=
  ∀ x, x ∈ A ∩ B ↔ x^2 + a*x + b < 0

-- State the theorem
theorem find_a :
  ∃ b, is_solution_set (-1) b :=
sorry

end NUMINAMATH_CALUDE_find_a_l683_68349


namespace NUMINAMATH_CALUDE_max_books_borrowed_l683_68365

theorem max_books_borrowed (total_students : Nat) (no_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 32) (h2 : no_books = 2) (h3 : one_book = 12) (h4 : two_books = 10) 
  (h5 : avg_books = 2) : 
  ∃ (max_books : Nat), max_books = 11 ∧ 
  (∀ (student_books : Nat), student_books ≤ max_books) ∧
  (∃ (rest_books : Nat), 
    rest_books * (total_students - no_books - one_book - two_books) + 
    no_books * 0 + one_book * 1 + two_books * 2 + max_books = 
    total_students * avg_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l683_68365


namespace NUMINAMATH_CALUDE_range_of_a_l683_68380

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) → 
  -5 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l683_68380


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l683_68362

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def Binary := List Bool

/-- The binary number 1011₂ -/
def b1 : Binary := [true, false, true, true]

/-- The binary number 101₂ -/
def b2 : Binary := [true, false, true]

/-- The binary number 11001₂ -/
def b3 : Binary := [true, true, false, false, true]

/-- The binary number 1110₂ -/
def b4 : Binary := [true, true, true, false]

/-- The binary number 100101₂ -/
def b5 : Binary := [true, false, false, true, false, true]

/-- The expected sum 1111010₂ -/
def expectedSum : Binary := [true, true, true, true, false, true, false]

/-- Theorem stating that the sum of the given binary numbers equals the expected sum -/
theorem binary_sum_theorem :
  binaryToDecimal b1 + binaryToDecimal b2 + binaryToDecimal b3 + 
  binaryToDecimal b4 + binaryToDecimal b5 = binaryToDecimal expectedSum := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l683_68362


namespace NUMINAMATH_CALUDE_expression_decrease_l683_68396

theorem expression_decrease (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let original := x^2 * y^3 * z
  let new_x := 0.8 * x
  let new_y := 0.75 * y
  let new_z := 0.9 * z
  let new_expression := new_x^2 * new_y^3 * new_z
  new_expression / original = 0.2414 :=
by sorry

end NUMINAMATH_CALUDE_expression_decrease_l683_68396


namespace NUMINAMATH_CALUDE_ratio_problem_l683_68382

theorem ratio_problem (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ a^2 + c^2 = 180 → b = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l683_68382


namespace NUMINAMATH_CALUDE_fifth_odd_integer_in_sequence_l683_68394

theorem fifth_odd_integer_in_sequence (n : ℕ) (sum : ℕ) (h1 : n = 20) (h2 : sum = 400) :
  let seq := fun i => 2 * i - 1
  let first := (sum - n * (n - 1)) / (2 * n)
  seq (first + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_odd_integer_in_sequence_l683_68394


namespace NUMINAMATH_CALUDE_new_students_count_l683_68379

theorem new_students_count (initial_students : Nat) (left_students : Nat) (final_students : Nat) :
  initial_students = 11 →
  left_students = 6 →
  final_students = 47 →
  final_students - (initial_students - left_students) = 42 :=
by sorry

end NUMINAMATH_CALUDE_new_students_count_l683_68379


namespace NUMINAMATH_CALUDE_complex_number_problem_l683_68395

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 2 + 3 * Complex.I) :
  α = 6 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l683_68395


namespace NUMINAMATH_CALUDE_improper_integral_convergence_l683_68326

open Real MeasureTheory

/-- The improper integral ∫[a to b] 1/(x-a)^α dx converges if and only if 0 < α < 1, given α > 0 and b > a -/
theorem improper_integral_convergence 
  (a b : ℝ) (α : ℝ) 
  (h1 : α > 0) 
  (h2 : b > a) : 
  (∃ (I : ℝ), ∫ x in a..b, 1 / (x - a) ^ α = I) ↔ 0 < α ∧ α < 1 :=
sorry

end NUMINAMATH_CALUDE_improper_integral_convergence_l683_68326


namespace NUMINAMATH_CALUDE_favorite_numbers_product_l683_68311

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- Definition of a favorite number -/
def is_favorite (n : ℕ+) : Prop :=
  n * sum_of_digits n = 10 * n

/-- Theorem statement -/
theorem favorite_numbers_product :
  ∃ (a b c : ℕ+),
    a * b * c = 71668 ∧
    is_favorite a ∧
    is_favorite b ∧
    is_favorite c := by sorry

end NUMINAMATH_CALUDE_favorite_numbers_product_l683_68311


namespace NUMINAMATH_CALUDE_increasing_sufficient_not_necessary_l683_68355

/-- A function f: ℝ → ℝ is increasing on [1, +∞) -/
def IncreasingOnIntervalOneInf (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f x < f y

/-- A sequence a_n = f(n) is increasing -/
def IncreasingSequence (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → f n < f (n + 1)

/-- The main theorem stating that IncreasingOnIntervalOneInf is sufficient but not necessary for IncreasingSequence -/
theorem increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (IncreasingOnIntervalOneInf f → IncreasingSequence f) ∧
  ∃ g : ℝ → ℝ, IncreasingSequence g ∧ ¬IncreasingOnIntervalOneInf g :=
sorry

end NUMINAMATH_CALUDE_increasing_sufficient_not_necessary_l683_68355


namespace NUMINAMATH_CALUDE_roots_equation_value_l683_68342

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by sorry

end NUMINAMATH_CALUDE_roots_equation_value_l683_68342


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l683_68392

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, x^3 + 2*x*y - 7 = 0 ↔ 
    (x = -7 ∧ y = -25) ∨ 
    (x = -1 ∧ y = -4) ∨ 
    (x = 1 ∧ y = 3) ∨ 
    (x = 7 ∧ y = -24) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l683_68392


namespace NUMINAMATH_CALUDE_triangle_area_equality_l683_68367

/-- Given a triangle MNH with points U on MN and C on NH, where:
  MU = s, UN = 6, NC = 20, CH = s, HM = 25,
  and the areas of triangle UNC and quadrilateral MUCH are equal,
  prove that s = 4. -/
theorem triangle_area_equality (s : ℝ) : 
  s > 0 ∧ 
  (1/2 : ℝ) * 6 * 20 = (1/2 : ℝ) * (s + 6) * (s + 20) - (1/2 : ℝ) * 6 * 20 → 
  s = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_equality_l683_68367


namespace NUMINAMATH_CALUDE_fruitBaskets_eq_96_l683_68351

/-- The number of ways to choose k items from n identical items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of fruit baskets with at least 3 pieces of fruit,
    given 7 apples and 12 oranges -/
def fruitBaskets : ℕ :=
  let totalBaskets := (choose 7 0 + choose 7 1 + choose 7 2 + choose 7 3 +
                       choose 7 4 + choose 7 5 + choose 7 6 + choose 7 7) *
                      (choose 12 0 + choose 12 1 + choose 12 2 + choose 12 3 +
                       choose 12 4 + choose 12 5 + choose 12 6 + choose 12 7 +
                       choose 12 8 + choose 12 9 + choose 12 10 + choose 12 11 +
                       choose 12 12)
  let invalidBaskets := choose 7 0 * choose 12 0 +
                        choose 7 0 * choose 12 1 +
                        choose 7 0 * choose 12 2 +
                        choose 7 1 * choose 12 0 +
                        choose 7 1 * choose 12 1 +
                        choose 7 2 * choose 12 0
  totalBaskets - invalidBaskets

theorem fruitBaskets_eq_96 : fruitBaskets = 96 := by sorry

end NUMINAMATH_CALUDE_fruitBaskets_eq_96_l683_68351


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l683_68301

theorem negation_of_existence_proposition :
  ¬(∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l683_68301


namespace NUMINAMATH_CALUDE_points_are_collinear_l683_68345

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem points_are_collinear :
  let p1 : Point := ⟨3, 1⟩
  let p2 : Point := ⟨6, 6.4⟩
  let p3 : Point := ⟨8, 10⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_points_are_collinear_l683_68345


namespace NUMINAMATH_CALUDE_monthly_income_p_l683_68352

def average_income (x y : ℕ) := (x + y) / 2

theorem monthly_income_p (p q r : ℕ) 
  (h1 : average_income p q = 5050)
  (h2 : average_income q r = 6250)
  (h3 : average_income p r = 5200) :
  p = 4000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_p_l683_68352


namespace NUMINAMATH_CALUDE_haley_albums_l683_68364

theorem haley_albums (total_pics : ℕ) (first_album_pics : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pics = 65)
  (h2 : first_album_pics = 17)
  (h3 : pics_per_album = 8) :
  (total_pics - first_album_pics) / pics_per_album = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_albums_l683_68364


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l683_68334

theorem smallest_positive_integer_congruence (x : ℕ) : x = 29 ↔ 
  x > 0 ∧
  (5 * x) % 20 = 25 % 20 ∧
  (3 * x + 1) % 7 = 4 % 7 ∧
  (2 * x - 3) % 13 = x % 13 ∧
  ∀ y : ℕ, y > 0 → 
    ((5 * y) % 20 = 25 % 20 ∧
     (3 * y + 1) % 7 = 4 % 7 ∧
     (2 * y - 3) % 13 = y % 13) → 
    x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l683_68334


namespace NUMINAMATH_CALUDE_max_value_product_sum_l683_68329

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l683_68329


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l683_68305

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ a ∧ |x - a| + |2*x + 1| ≤ 2*a + x) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l683_68305


namespace NUMINAMATH_CALUDE_duck_buying_problem_l683_68350

theorem duck_buying_problem (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) % 9 = 0 →
  ephraim ≥ 1 →
  kolton ≥ 1 →
  (adelaide + ephraim + kolton) / 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_duck_buying_problem_l683_68350


namespace NUMINAMATH_CALUDE_intersection_M_N_l683_68393

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l683_68393


namespace NUMINAMATH_CALUDE_triangle_not_unique_l683_68359

/-- A triangle is defined by three side lengths -/
structure Triangle :=
  (a b c : ℝ)

/-- Predicate to check if three real numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- Given one side and the sum of the other two sides, 
    the triangle is not uniquely determined -/
theorem triangle_not_unique (s : ℝ) (sum : ℝ) :
  ∃ (t1 t2 : Triangle),
    t1 ≠ t2 ∧
    t1.a = s ∧
    t2.a = s ∧
    t1.b + t1.c = sum ∧
    t2.b + t2.c = sum ∧
    is_triangle t1.a t1.b t1.c ∧
    is_triangle t2.a t2.b t2.c :=
  sorry


end NUMINAMATH_CALUDE_triangle_not_unique_l683_68359


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l683_68312

/-- Proves that given a cloth length of 200 metres sold for Rs. 12000 with a loss of Rs. 12 per metre, the cost price for one metre of cloth is Rs. 72. -/
theorem cost_price_per_metre (total_length : ℕ) (selling_price : ℕ) (loss_per_metre : ℕ) :
  total_length = 200 →
  selling_price = 12000 →
  loss_per_metre = 12 →
  (selling_price + total_length * loss_per_metre) / total_length = 72 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l683_68312


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l683_68373

-- Define the propositions p and q as functions of x
def p (x : ℝ) : Prop := Real.log (x - 1) < 0
def q (x : ℝ) : Prop := |1 - x| < 2

-- State the theorem
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬(p x)) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l683_68373


namespace NUMINAMATH_CALUDE_three_dice_same_number_l683_68399

/-- A standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of a specific outcome on a standard die -/
def prob_specific_outcome : ℚ := 1 / 6

/-- The probability of all three dice showing the same number -/
def prob_all_same : ℚ := 1 / 36

/-- Theorem: The probability of three standard six-sided dice showing the same number
    when tossed simultaneously is 1/36 -/
theorem three_dice_same_number :
  (1 : ℚ) * prob_specific_outcome * prob_specific_outcome = prob_all_same := by
  sorry

end NUMINAMATH_CALUDE_three_dice_same_number_l683_68399


namespace NUMINAMATH_CALUDE_continuous_fraction_value_l683_68397

theorem continuous_fraction_value : 
  ∃ (x : ℝ), x = 2 + 4 / (1 + 4/x) ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_continuous_fraction_value_l683_68397


namespace NUMINAMATH_CALUDE_factor_calculation_l683_68307

theorem factor_calculation (x : ℝ) (f : ℝ) : 
  x = 22.142857142857142 → 
  ((x + 5) * f / 5) - 5 = 66 / 2 → 
  f = 7 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l683_68307


namespace NUMINAMATH_CALUDE_triangle_lines_l683_68371

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)

-- Define the altitude line l₁
def l₁ (x y : ℝ) : Prop := 4 * x + y - 5 = 0

-- Define the two possible equations for line l₂
def l₂_1 (x y : ℝ) : Prop := x + y - 7 = 0
def l₂_2 (x y : ℝ) : Prop := 2 * x - 3 * y + 6 = 0

-- Theorem statement
theorem triangle_lines :
  -- l₁ is the altitude from A to BC
  (∀ x y : ℝ, l₁ x y ↔ (y - A.2 = -(B.2 - C.2)/(B.1 - C.1) * (x - A.1))) ∧
  -- l₂ passes through C
  ((∀ x y : ℝ, l₂_1 x y → x = C.1 ∧ y = C.2) ∨
   (∀ x y : ℝ, l₂_2 x y → x = C.1 ∧ y = C.2)) ∧
  -- Distances from A and B to l₂ are equal
  ((∀ x y : ℝ, l₂_1 x y → 
    (|x + y - (A.1 + A.2)|/Real.sqrt 2 = |x + y - (B.1 + B.2)|/Real.sqrt 2)) ∨
   (∀ x y : ℝ, l₂_2 x y → 
    (|2*x - 3*y + 6 - (2*A.1 - 3*A.2 + 6)|/Real.sqrt 13 = 
     |2*x - 3*y + 6 - (2*B.1 - 3*B.2 + 6)|/Real.sqrt 13))) :=
by sorry


end NUMINAMATH_CALUDE_triangle_lines_l683_68371


namespace NUMINAMATH_CALUDE_stevens_peaches_l683_68310

-- Define the number of peaches Jake has
def jakes_peaches : ℕ := 7

-- Define the difference in peaches between Steven and Jake
def peach_difference : ℕ := 6

-- Theorem stating that Steven has 13 peaches
theorem stevens_peaches : 
  jakes_peaches + peach_difference = 13 := by sorry

end NUMINAMATH_CALUDE_stevens_peaches_l683_68310


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l683_68384

/-- The quadratic equation x^2 - 2mx + m^2 + m - 3 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0

/-- The range of m for which the equation has real roots -/
def m_range : Set ℝ :=
  {m : ℝ | m ≤ 3}

/-- The product of the roots of the equation -/
def root_product (m : ℝ) : ℝ := m^2 + m - 3

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m ↔ m ∈ m_range) ∧
  (∃ m : ℝ, root_product m = 17 ∧ m = -5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l683_68384


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l683_68360

theorem triangle_angle_measure (a b c : ℝ) (A : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  S > 0 →
  (4 * Real.sqrt 3 / 3) * S = b^2 + c^2 - a^2 →
  S = (1/2) * b * c * Real.sin A →
  0 < A → A < π →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l683_68360


namespace NUMINAMATH_CALUDE_sum_34_27_base5_l683_68320

def base10_to_base5 (n : ℕ) : List ℕ :=
  sorry

theorem sum_34_27_base5 :
  base10_to_base5 (34 + 27) = [2, 2, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_27_base5_l683_68320


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l683_68327

-- Define the arithmetic sequence an
def an (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the sequence bn
def bn (n : ℕ) : ℝ := an n - 2 * n

-- Define the sum of the first n terms of bn
def Tn (n : ℕ) : ℝ := 3^n - 1 - n^2 - n

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → an n = 2 * 3^(n - 1)) ∧
  (an 2 = 6) ∧
  (an 1 + an 2 + an 3 = 26) ∧
  (∀ n : ℕ, n ≥ 1 → Tn n = 3^n - 1 - n^2 - n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l683_68327


namespace NUMINAMATH_CALUDE_base4_calculation_l683_68344

/-- Convert a base 4 number to base 10 -/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun ⟨i, d⟩ acc => acc + d * (4 ^ i)) 0

/-- Convert a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Multiplication in base 4 -/
def multBase4 (a b : List Nat) : List Nat :=
  base10ToBase4 (base4ToBase10 a * base4ToBase10 b)

/-- Division in base 4 -/
def divBase4 (a b : List Nat) : List Nat :=
  base10ToBase4 (base4ToBase10 a / base4ToBase10 b)

theorem base4_calculation :
  divBase4 (multBase4 [1, 3, 2] [4, 2]) [3] = [0, 3, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l683_68344


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l683_68313

theorem least_positive_integer_to_multiple_of_five : 
  ∃! n : ℕ, n > 0 ∧ (525 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (525 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l683_68313


namespace NUMINAMATH_CALUDE_chocolate_division_l683_68346

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 60 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  piles_given * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l683_68346


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l683_68338

theorem quadratic_expression_value : 
  let x : ℤ := -2
  (x^2 + 6*x - 7) = -15 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l683_68338


namespace NUMINAMATH_CALUDE_adam_total_score_l683_68300

/-- Calculates the total points scored in a game given points per round and number of rounds -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that given 71 points per round and 4 rounds, the total points is 284 -/
theorem adam_total_score : totalPoints 71 4 = 284 := by
  sorry

end NUMINAMATH_CALUDE_adam_total_score_l683_68300


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l683_68370

/-- Given a = 15, b = 19, c = 25, and S = a + b + c = 59, prove that the expression
    (a² * (1/b - 1/c) + b² * (1/c - 1/a) + c² * (1/a - 1/b) + 37) /
    (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2)
    equals 77.5 -/
theorem complex_fraction_evaluation (a b c S : ℚ) 
    (ha : a = 15) (hb : b = 19) (hc : c = 25) (hS : S = a + b + c) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + 37) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2) = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l683_68370


namespace NUMINAMATH_CALUDE_erased_number_l683_68387

theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) 
  (h2 : (a - 4) + (a - 3) + (a - 2) + (a - 1) + a + (a + 1) + (a + 2) + (a + 3) + (a + 4) - (a + b) = 1703) : 
  a + b = 214 := by
sorry

end NUMINAMATH_CALUDE_erased_number_l683_68387


namespace NUMINAMATH_CALUDE_unique_hyperdeficient_number_l683_68386

/-- Sum of divisors function -/
def f (n : ℕ) : ℕ := sorry

/-- A number is hyperdeficient if f(f(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := f (f n) = n + 3

theorem unique_hyperdeficient_number : 
  ∃! n : ℕ, n > 0 ∧ is_hyperdeficient n :=
sorry

end NUMINAMATH_CALUDE_unique_hyperdeficient_number_l683_68386


namespace NUMINAMATH_CALUDE_terrier_to_poodle_groom_ratio_l683_68383

def poodle_groom_time : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_groom_time : ℕ := 210

theorem terrier_to_poodle_groom_ratio :
  ∃ (terrier_groom_time : ℕ),
    terrier_groom_time * num_terriers + poodle_groom_time * num_poodles = total_groom_time ∧
    2 * terrier_groom_time = poodle_groom_time :=
by sorry

end NUMINAMATH_CALUDE_terrier_to_poodle_groom_ratio_l683_68383


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l683_68398

/-- The total surface area of a cylinder with height 12 and radius 5 is 170π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 5
  let lateral_area := 2 * Real.pi * r * h
  let base_area := 2 * Real.pi * r^2
  lateral_area + base_area = 170 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l683_68398


namespace NUMINAMATH_CALUDE_smaller_is_999_l683_68390

/-- Two 3-digit positive integers whose average equals their decimal concatenation -/
structure SpecialIntegerPair where
  m : ℕ
  n : ℕ
  m_three_digit : 100 ≤ m ∧ m ≤ 999
  n_three_digit : 100 ≤ n ∧ n ≤ 999
  avg_eq_concat : (m + n) / 2 = m + n / 1000

/-- The smaller of the two integers in a SpecialIntegerPair is 999 -/
theorem smaller_is_999 (pair : SpecialIntegerPair) : min pair.m pair.n = 999 := by
  sorry

end NUMINAMATH_CALUDE_smaller_is_999_l683_68390


namespace NUMINAMATH_CALUDE_gcd_98_63_l683_68372

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l683_68372


namespace NUMINAMATH_CALUDE_lizzies_garbage_collection_l683_68306

theorem lizzies_garbage_collection (x : ℝ) 
  (h1 : x + (x - 39) = 735) : x = 387 := by
  sorry

end NUMINAMATH_CALUDE_lizzies_garbage_collection_l683_68306


namespace NUMINAMATH_CALUDE_fraction_equality_l683_68328

theorem fraction_equality (x y z w k : ℝ) : 
  (9 / (x + y + w) = k / (x + z + w)) ∧ 
  (k / (x + z + w) = 12 / (z - y)) → 
  k = 21 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l683_68328


namespace NUMINAMATH_CALUDE_exam_items_count_l683_68331

theorem exam_items_count :
  ∀ (total_items : ℕ) (liza_correct : ℕ) (rose_correct : ℕ) (rose_incorrect : ℕ),
    liza_correct = (90 * total_items) / 100 →
    rose_correct = liza_correct + 2 →
    rose_incorrect = 4 →
    total_items = rose_correct + rose_incorrect →
    total_items = 60 := by
  sorry

end NUMINAMATH_CALUDE_exam_items_count_l683_68331


namespace NUMINAMATH_CALUDE_mass_is_not_vector_l683_68363

-- Define the properties of a physical quantity
structure PhysicalQuantity where
  has_magnitude : Bool
  has_direction : Bool

-- Define what makes a quantity a vector
def is_vector (q : PhysicalQuantity) : Prop :=
  q.has_magnitude ∧ q.has_direction

-- Define the physical quantities
def mass : PhysicalQuantity :=
  { has_magnitude := true, has_direction := false }

def velocity : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def displacement : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

def force : PhysicalQuantity :=
  { has_magnitude := true, has_direction := true }

-- Theorem to prove
theorem mass_is_not_vector : ¬(is_vector mass) := by
  sorry

end NUMINAMATH_CALUDE_mass_is_not_vector_l683_68363


namespace NUMINAMATH_CALUDE_min_value_sum_ratios_l683_68336

theorem min_value_sum_ratios (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / a + (a + b + c) / b + (a + b + c) / c ≥ 9 ∧
  ((a + b + c) / a + (a + b + c) / b + (a + b + c) / c = 9 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_ratios_l683_68336


namespace NUMINAMATH_CALUDE_shorter_can_radius_l683_68375

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Given two cans with equal volume, one with 4 times the height of the other,
    and the taller can having a radius of 5, prove the radius of the shorter can is 10 -/
theorem shorter_can_radius (can1 can2 : Can) 
  (h_volume : π * can1.radius^2 * can1.height = π * can2.radius^2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_taller_radius : can2.radius = 5) :
  can1.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_can_radius_l683_68375


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l683_68357

theorem sales_tax_percentage 
  (total_bill : ℝ) 
  (tip_percentage : ℝ) 
  (food_price : ℝ) 
  (h1 : total_bill = 211.20)
  (h2 : tip_percentage = 0.20)
  (h3 : food_price = 160) :
  ∃ (sales_tax_percentage : ℝ),
    sales_tax_percentage = 0.10 ∧
    total_bill = food_price * (1 + sales_tax_percentage) * (1 + tip_percentage) := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l683_68357


namespace NUMINAMATH_CALUDE_polynomial_root_coefficients_l683_68340

theorem polynomial_root_coefficients :
  ∀ (a b c : ℝ),
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 + b * (2 - Complex.I : ℂ) ^ 2 - 2 * (2 - Complex.I : ℂ) + c = 0 →
  a = 2 + 2 * Real.sqrt 1.5 ∧
  b = 10 + 2 * Real.sqrt 1.5 ∧
  c = 10 - 8 * Real.sqrt 1.5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_coefficients_l683_68340


namespace NUMINAMATH_CALUDE_intersection_probability_formula_l683_68330

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2023

/-- The probability of selecting six distinct points A, B, C, D, E, F from n evenly spaced points 
    on a circle, such that chord AB intersects chord CD but neither intersects chord EF -/
def intersection_probability : ℚ :=
  2 * (Nat.choose (n / 2) 2) / Nat.choose n 6

/-- Theorem stating the probability calculation -/
theorem intersection_probability_formula : 
  intersection_probability = 2 * (Nat.choose (n / 2) 2) / Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_formula_l683_68330


namespace NUMINAMATH_CALUDE_total_distance_walked_l683_68317

/-- Calculates the total distance walked to various destinations in a school. -/
theorem total_distance_walked (water_fountain_dist : ℕ) (main_office_dist : ℕ) (teacher_lounge_dist : ℕ)
  (water_fountain_trips : ℕ) (main_office_trips : ℕ) (teacher_lounge_trips : ℕ)
  (h1 : water_fountain_dist = 30)
  (h2 : main_office_dist = 50)
  (h3 : teacher_lounge_dist = 35)
  (h4 : water_fountain_trips = 4)
  (h5 : main_office_trips = 2)
  (h6 : teacher_lounge_trips = 3) :
  water_fountain_dist * water_fountain_trips +
  main_office_dist * main_office_trips +
  teacher_lounge_dist * teacher_lounge_trips = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l683_68317


namespace NUMINAMATH_CALUDE_cos_negative_eleven_fourths_pi_l683_68323

theorem cos_negative_eleven_fourths_pi :
  Real.cos (-11/4 * Real.pi) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_eleven_fourths_pi_l683_68323


namespace NUMINAMATH_CALUDE_range_of_a_l683_68335

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x < 3
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the range of a
def a_range (a : ℝ) : Prop := 1 ≤ a ∧ a < 3

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a))) →
  (∀ a : ℝ, a_range a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l683_68335
