import Mathlib

namespace solve_for_m_l4013_401327

theorem solve_for_m : ∃ m : ℚ, (10 * (1/2 : ℚ) + m = 2) ∧ m = -3 := by
  sorry

end solve_for_m_l4013_401327


namespace parabola_m_minus_one_opens_downward_l4013_401306

/-- A parabola y = ax^2 opens downward if and only if a < 0 -/
axiom parabola_opens_downward (a : ℝ) : (∀ x y : ℝ, y = a * x^2) → (∀ x : ℝ, a * x^2 ≤ 0) ↔ a < 0

/-- For the parabola y = (m-1)x^2 to open downward, m must be less than 1 -/
theorem parabola_m_minus_one_opens_downward (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x^2) → (∀ x : ℝ, (m - 1) * x^2 ≤ 0) → m < 1 := by
  sorry

end parabola_m_minus_one_opens_downward_l4013_401306


namespace sparrow_distribution_l4013_401335

theorem sparrow_distribution (total : ℕ) (moved : ℕ) (flew_away : ℕ) :
  total = 25 →
  moved = 5 →
  flew_away = 7 →
  (∃ (initial_first initial_second : ℕ),
    initial_first + initial_second = total ∧
    initial_first - moved = 2 * (initial_second + moved - flew_away) ∧
    initial_first = 17 ∧
    initial_second = 8) :=
by sorry

end sparrow_distribution_l4013_401335


namespace circle_radius_relation_l4013_401396

theorem circle_radius_relation (square_area : ℝ) (small_circle_circumference : ℝ) :
  square_area = 784 →
  small_circle_circumference = 8 →
  ∃ (x : ℝ) (r_s r_l : ℝ),
    r_s = 4 / π ∧
    r_l = 14 ∧
    r_l = x - (1/3) * r_s ∧
    x = 14 + 4 / (3 * π) := by
  sorry

end circle_radius_relation_l4013_401396


namespace workers_total_earning_approx_1480_l4013_401336

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℝ
  workDays : ℕ

/-- Calculates the total earning of a worker -/
def totalEarning (w : Worker) : ℝ :=
  w.dailyWage * w.workDays

/-- Theorem stating that the total earning of the three workers is approximately 1480 -/
theorem workers_total_earning_approx_1480 
  (a b c : Worker)
  (h_a_days : a.workDays = 16)
  (h_b_days : b.workDays = 9)
  (h_c_days : c.workDays = 4)
  (h_c_wage : c.dailyWage = 71.15384615384615)
  (h_wage_ratio : a.dailyWage / c.dailyWage = 3 / 5 ∧ b.dailyWage / c.dailyWage = 4 / 5) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    abs ((totalEarning a + totalEarning b + totalEarning c) - 1480) < ε :=
sorry

end workers_total_earning_approx_1480_l4013_401336


namespace edward_money_theorem_l4013_401301

/-- Calculates Edward's earnings from mowing lawns --/
def lawn_earnings (small medium large : ℕ) : ℕ :=
  8 * small + 12 * medium + 15 * large

/-- Calculates Edward's earnings from cleaning gardens --/
def garden_earnings (gardens : ℕ) : ℕ :=
  if gardens = 0 then 0
  else if gardens = 1 then 10
  else if gardens = 2 then 22
  else 22 + 15 * (gardens - 2)

/-- Calculates Edward's total earnings --/
def total_earnings (small medium large gardens : ℕ) : ℕ :=
  lawn_earnings small medium large + garden_earnings gardens

/-- Calculates Edward's final amount of money --/
def edward_final_money (small medium large gardens savings fuel_cost rental_cost : ℕ) : ℕ :=
  total_earnings small medium large gardens + savings - (fuel_cost + rental_cost)

theorem edward_money_theorem :
  edward_final_money 3 1 1 5 7 10 15 = 100 := by
  sorry

#eval edward_final_money 3 1 1 5 7 10 15

end edward_money_theorem_l4013_401301


namespace fixed_stable_points_equality_l4013_401319

def f (a x : ℝ) : ℝ := a * x^2 - 1

def isFixedPoint (a x : ℝ) : Prop := f a x = x

def isStablePoint (a x : ℝ) : Prop := f a (f a x) = x

def fixedPointSet (a : ℝ) : Set ℝ := {x | isFixedPoint a x}

def stablePointSet (a : ℝ) : Set ℝ := {x | isStablePoint a x}

theorem fixed_stable_points_equality (a : ℝ) :
  (fixedPointSet a = stablePointSet a ∧ (fixedPointSet a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 := by
  sorry

end fixed_stable_points_equality_l4013_401319


namespace sqrt_equation_solution_l4013_401368

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end sqrt_equation_solution_l4013_401368


namespace fruit_display_problem_l4013_401384

theorem fruit_display_problem (bananas oranges apples : ℕ) 
  (h1 : apples = 2 * oranges)
  (h2 : oranges = 2 * bananas)
  (h3 : bananas + oranges + apples = 35) :
  bananas = 5 := by
  sorry

end fruit_display_problem_l4013_401384


namespace abie_chips_bought_l4013_401310

theorem abie_chips_bought (initial_bags : ℕ) (given_away : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : given_away = 4)
  (h3 : final_bags = 22) :
  final_bags - (initial_bags - given_away) = 6 :=
by sorry

end abie_chips_bought_l4013_401310


namespace isosceles_triangle_perimeter_l4013_401393

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 2 ∧ b = 4 ∧ c = 4 →  -- Two sides are 4, one side is 2
  a + b + c = 10 :=        -- The perimeter is 10
by
  sorry


end isosceles_triangle_perimeter_l4013_401393


namespace digit_count_theorem_l4013_401313

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def four_digit_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * (d.card - 1) * (d.card - 2) * (d.card - 3)

def four_digit_even_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 2 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

def four_digit_div5_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 5 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

theorem digit_count_theorem :
  four_digit_no_repeat digits = 720 ∧
  four_digit_even_no_repeat digits = 420 ∧
  four_digit_div5_no_repeat digits = 220 := by
  sorry

end digit_count_theorem_l4013_401313


namespace circles_intersect_iff_l4013_401352

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
theorem circles_intersect_iff (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end circles_intersect_iff_l4013_401352


namespace prob_more_heads_10_coins_l4013_401329

def num_coins : ℕ := 10

-- Probability of getting more heads than tails
def prob_more_heads : ℚ := 193 / 512

theorem prob_more_heads_10_coins : 
  (prob_more_heads : ℚ) = 193 / 512 := by sorry

end prob_more_heads_10_coins_l4013_401329


namespace complex_quadrant_l4013_401338

theorem complex_quadrant (a : ℝ) (z : ℂ) : 
  z = a^2 - 3*a - 4 + (a - 4)*Complex.I →
  z.re = 0 →
  (a - a*Complex.I).re < 0 ∧ (a - a*Complex.I).im > 0 :=
sorry

end complex_quadrant_l4013_401338


namespace sum_of_squares_with_inequality_l4013_401390

theorem sum_of_squares_with_inequality (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (k : ℕ), n = 5 * k) 
  (h3 : ∃ (a b : ℤ), n = a^2 + b^2) : 
  ∃ (x y : ℤ), n = x^2 + y^2 ∧ x^2 ≥ 4 * y^2 := by
  sorry

end sum_of_squares_with_inequality_l4013_401390


namespace index_card_area_l4013_401377

theorem index_card_area (width height : ℝ) (h1 : width = 5) (h2 : height = 8) : 
  ((width - 2) * height = 24 → width * (height - 2) = 30) ∧ 
  ((width * (height - 2) = 24 → (width - 2) * height = 30)) :=
by sorry

end index_card_area_l4013_401377


namespace isaac_sleep_time_l4013_401332

-- Define a simple representation of time
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

def Time.isAM (t : Time) : Bool :=
  t.hour < 12

def Time.toPM (t : Time) : Time :=
  if t.isAM then { hour := t.hour + 12, minute := t.minute }
  else t

def Time.fromPM (t : Time) : Time :=
  if t.isAM then t
  else { hour := t.hour - 12, minute := t.minute }

def subtractHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute
  let newTotalMinutes := totalMinutes - h * 60
  let newHour := newTotalMinutes / 60
  let newMinute := newTotalMinutes % 60
  { hour := newHour, minute := newMinute }

theorem isaac_sleep_time (wakeUpTime sleepTime : Time) (sleepDuration : Nat) :
  wakeUpTime = { hour := 7, minute := 0 } →
  sleepDuration = 8 →
  sleepTime = (subtractHours wakeUpTime sleepDuration).toPM →
  sleepTime = { hour := 23, minute := 0 } :=
by sorry

end isaac_sleep_time_l4013_401332


namespace quadratic_residue_product_l4013_401302

theorem quadratic_residue_product (p a b : ℤ) (hp : Prime p) (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) :
  (∃ x : ℤ, x^2 ≡ a [ZMOD p]) → (∃ y : ℤ, y^2 ≡ b [ZMOD p]) →
  (∃ z : ℤ, z^2 ≡ a * b [ZMOD p]) := by
sorry

end quadratic_residue_product_l4013_401302


namespace geometric_series_first_term_l4013_401305

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) : 
  (a / (1 - r) = 20) → 
  (a^2 / (1 - r^2) = 80) → 
  a = 20/3 := by sorry

end geometric_series_first_term_l4013_401305


namespace pie_slices_today_l4013_401395

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := 7

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- The total number of slices of pie served today -/
def total_slices : ℕ := lunch_slices + dinner_slices

theorem pie_slices_today : total_slices = 12 := by
  sorry

end pie_slices_today_l4013_401395


namespace sqrt_equation_solution_l4013_401397

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 8 → x = -30 := by
  sorry

end sqrt_equation_solution_l4013_401397


namespace liu_hui_author_of_sea_island_arithmetic_l4013_401334

/-- Represents a mathematical work -/
structure MathWork where
  title : String
  author : String
  significance : Bool
  advance_years : ℕ

/-- The Sea Island Arithmetic -/
def sea_island_arithmetic : MathWork :=
  { title := "The Sea Island Arithmetic"
  , author := "Unknown" -- We'll prove this is Liu Hui
  , significance := true
  , advance_years := 1300 }

/-- Theorem: Liu Hui is the author of The Sea Island Arithmetic -/
theorem liu_hui_author_of_sea_island_arithmetic :
  sea_island_arithmetic.author = "Liu Hui" :=
by
  sorry

#check liu_hui_author_of_sea_island_arithmetic

end liu_hui_author_of_sea_island_arithmetic_l4013_401334


namespace scientific_notation_of_55000000_l4013_401351

theorem scientific_notation_of_55000000 :
  55000000 = 5.5 * (10 ^ 7) := by sorry

end scientific_notation_of_55000000_l4013_401351


namespace complex_equation_solution_l4013_401379

theorem complex_equation_solution (z : ℂ) : (Complex.I * (z + 1) = -3 + 2 * Complex.I) → z = 1 + 3 * Complex.I := by
  sorry

end complex_equation_solution_l4013_401379


namespace inequality_proof_l4013_401371

theorem inequality_proof (x : ℝ) (hx : x > 0) : (x + 1) * Real.sqrt (x + 1) ≥ Real.sqrt 2 * (x + Real.sqrt x) := by
  sorry

end inequality_proof_l4013_401371


namespace largest_n_for_equation_l4013_401323

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6)) ∧
  (∀ (m : ℕ), m > 0 →
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6) →
    m ≤ 8) :=
by sorry

end largest_n_for_equation_l4013_401323


namespace defective_units_percentage_l4013_401337

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0024) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.06 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by
  sorry

end defective_units_percentage_l4013_401337


namespace product_of_primes_l4013_401370

theorem product_of_primes (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_range : 15 < p * q ∧ p * q < 36)
  (hp_range : 2 < p ∧ p < 6)
  (hq_range : 8 < q ∧ q < 24) :
  p * q = 33 := by
sorry

end product_of_primes_l4013_401370


namespace x_value_l4013_401326

theorem x_value : ∃ x : ℝ, x = 80 * (1 + 0.12) ∧ x = 89.6 := by
  sorry

end x_value_l4013_401326


namespace interest_difference_proof_l4013_401369

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the difference between principal and interest -/
def principalInterestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal - simpleInterest principal rate time

theorem interest_difference_proof :
  let principal : ℝ := 1100
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principalInterestDifference principal rate time = 572 := by
sorry

end interest_difference_proof_l4013_401369


namespace mushroom_drying_l4013_401333

/-- Given an initial mass of mushrooms and moisture contents before and after drying,
    calculate the mass of mushrooms after drying. -/
theorem mushroom_drying (initial_mass : ℝ) (initial_moisture : ℝ) (final_moisture : ℝ) :
  initial_mass = 100 →
  initial_moisture = 99 / 100 →
  final_moisture = 98 / 100 →
  (1 - initial_moisture) * initial_mass / (1 - final_moisture) = 50 := by
  sorry

#check mushroom_drying

end mushroom_drying_l4013_401333


namespace triangle_area_rational_l4013_401356

/-- A point on the unit circle with rational coordinates -/
structure RationalUnitCirclePoint where
  x : ℚ
  y : ℚ
  on_circle : x^2 + y^2 = 1

/-- The area of a triangle with vertices on the unit circle is rational -/
theorem triangle_area_rational (p₁ p₂ p₃ : RationalUnitCirclePoint) :
  ∃ a : ℚ, a = (1/2) * |p₁.x * (p₂.y - p₃.y) + p₂.x * (p₃.y - p₁.y) + p₃.x * (p₁.y - p₂.y)| :=
sorry

end triangle_area_rational_l4013_401356


namespace train_length_l4013_401307

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 8 seconds,
    and the platform length is 1162.5 meters, prove that the length of the train is 300 meters. -/
theorem train_length (crossing_platform_time : ℝ) (crossing_pole_time : ℝ) (platform_length : ℝ)
  (h1 : crossing_platform_time = 39)
  (h2 : crossing_pole_time = 8)
  (h3 : platform_length = 1162.5) :
  ∃ (train_length : ℝ), train_length = 300 := by
  sorry

end train_length_l4013_401307


namespace tour_group_dish_choices_l4013_401362

/-- Represents the number of people in the tour group -/
def total_people : ℕ := 92

/-- Represents the number of different dish combinations -/
def dish_combinations : ℕ := 9

/-- Represents the minimum number of people who must choose the same combination -/
def min_same_choice : ℕ := total_people / dish_combinations + 1

theorem tour_group_dish_choices :
  ∃ (combination : Fin dish_combinations),
    (Finset.filter (λ person : Fin total_people =>
      person.val % dish_combinations = combination.val) (Finset.univ : Finset (Fin total_people))).card
    ≥ min_same_choice :=
sorry

end tour_group_dish_choices_l4013_401362


namespace coprime_and_indivisible_l4013_401364

theorem coprime_and_indivisible (n : ℕ) (h1 : n > 3) (h2 : Odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ Nat.gcd (a * b * (a + b)) n = 1 ∧ ¬(n ∣ (a - b)) := by
  sorry

end coprime_and_indivisible_l4013_401364


namespace satisfying_function_is_identity_l4013_401317

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  ∀ x : ℝ, f x = x :=
sorry

end satisfying_function_is_identity_l4013_401317


namespace order_of_differences_l4013_401383

theorem order_of_differences (a b c : ℝ) : 
  a = Real.sqrt 3 - Real.sqrt 2 →
  b = Real.sqrt 6 - Real.sqrt 5 →
  c = Real.sqrt 7 - Real.sqrt 6 →
  a > b ∧ b > c :=
by sorry

end order_of_differences_l4013_401383


namespace minutes_to_year_l4013_401373

/-- Proves that 525,600 minutes is equivalent to 365 days (1 year) --/
theorem minutes_to_year (minutes_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : 
  minutes_per_hour = 60 → hours_per_day = 24 → days_per_year = 365 →
  525600 / (minutes_per_hour * hours_per_day) = days_per_year := by
  sorry

end minutes_to_year_l4013_401373


namespace intersection_when_m_is_one_subset_condition_l4013_401394

-- Define set A
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

-- Define set B as a function of m
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 1}

-- Statement 1
theorem intersection_when_m_is_one : 
  A ∩ B 1 = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Statement 2
theorem subset_condition : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end intersection_when_m_is_one_subset_condition_l4013_401394


namespace modulus_of_complex_reciprocal_l4013_401312

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  z = (Complex.I - 1)⁻¹ → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_complex_reciprocal_l4013_401312


namespace count_triples_product_million_l4013_401365

theorem count_triples_product_million : 
  (Finset.filter (fun (triple : ℕ × ℕ × ℕ) => triple.1 * triple.2.1 * triple.2.2 = 10^6) (Finset.product (Finset.range (10^6 + 1)) (Finset.product (Finset.range (10^6 + 1)) (Finset.range (10^6 + 1))))).card = 784 := by
  sorry

end count_triples_product_million_l4013_401365


namespace working_light_bulbs_l4013_401380

theorem working_light_bulbs (total_lamps : ℕ) (bulbs_per_lamp : ℕ) 
  (lamps_with_two_burnt : ℕ) (lamps_with_one_burnt : ℕ) (lamps_with_three_burnt : ℕ) :
  total_lamps = 60 →
  bulbs_per_lamp = 7 →
  lamps_with_two_burnt = total_lamps / 3 →
  lamps_with_one_burnt = total_lamps / 4 →
  lamps_with_three_burnt = total_lamps / 5 →
  (total_lamps - (lamps_with_two_burnt + lamps_with_one_burnt + lamps_with_three_burnt)) * bulbs_per_lamp +
  lamps_with_two_burnt * (bulbs_per_lamp - 2) +
  lamps_with_one_burnt * (bulbs_per_lamp - 1) +
  lamps_with_three_burnt * (bulbs_per_lamp - 3) = 329 :=
by
  sorry


end working_light_bulbs_l4013_401380


namespace expression_simplification_l4013_401330

theorem expression_simplification (a : ℤ) (n : ℕ) (h : n ≠ 1) :
  (a^(3*n) / (a^n - 1) + 1 / (a^n + 1)) - (a^(2*n) / (a^n + 1) + 1 / (a^n - 1)) = a^(2*n) + 2 :=
by sorry

end expression_simplification_l4013_401330


namespace mall_walking_methods_l4013_401386

/-- The number of entrances in the mall -/
def num_entrances : ℕ := 4

/-- The number of different walking methods through the mall -/
def num_walking_methods : ℕ := num_entrances * (num_entrances - 1)

/-- Theorem stating the number of different walking methods -/
theorem mall_walking_methods :
  num_walking_methods = 12 :=
sorry

end mall_walking_methods_l4013_401386


namespace apples_added_l4013_401308

theorem apples_added (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 8)
  (h2 : final_apples = 13) :
  final_apples - initial_apples = 5 := by
  sorry

end apples_added_l4013_401308


namespace imoProblem1995_l4013_401353

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Checks if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

theorem imoProblem1995 (n : ℕ) (h_n : n > 3) :
  (∃ (A : Fin n → Point) (r : Fin n → ℝ),
    (∀ (i j k : Fin n), i < j → j < k → ¬areCollinear (A i) (A j) (A k)) ∧
    (∀ (i j k : Fin n), i < j → j < k → 
      triangleArea (A i) (A j) (A k) = r i + r j + r k)) ↔ 
  n = 4 := by
sorry

end imoProblem1995_l4013_401353


namespace friends_pen_cost_l4013_401347

def robertPens : ℕ := 4
def juliaPens : ℕ := 3 * robertPens
def dorothyPens : ℕ := juliaPens / 2
def penCost : ℚ := 3/2

def totalPens : ℕ := robertPens + juliaPens + dorothyPens
def totalCost : ℚ := (totalPens : ℚ) * penCost

theorem friends_pen_cost : totalCost = 33 := by sorry

end friends_pen_cost_l4013_401347


namespace distribution_ways_l4013_401389

theorem distribution_ways (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  (k : ℕ) ^ n = 243 := by
  sorry

end distribution_ways_l4013_401389


namespace original_price_calculation_l4013_401346

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 20 / 100) * (1 - 10 / 100) = 2) → 
  (initial_price = 100 / 54) := by
sorry

end original_price_calculation_l4013_401346


namespace picture_on_wall_l4013_401385

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 22) 
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end picture_on_wall_l4013_401385


namespace sum_of_numbers_with_lcm_and_ratio_l4013_401314

/-- Given three positive integers in the ratio 2:3:5 with LCM 180, prove their sum is 60 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 3 = b * 2 →
  a * 5 = c * 2 →
  Nat.lcm (Nat.lcm a b) c = 180 →
  a + b + c = 60 := by
sorry

end sum_of_numbers_with_lcm_and_ratio_l4013_401314


namespace difference_of_squares_l4013_401309

theorem difference_of_squares : 55^2 - 45^2 = 1000 := by
  sorry

end difference_of_squares_l4013_401309


namespace unique_solution_system_l4013_401318

theorem unique_solution_system :
  ∃! (a b c d : ℝ),
    (a * b + c + d = 3) ∧
    (b * c + d + a = 5) ∧
    (c * d + a + b = 2) ∧
    (d * a + b + c = 6) ∧
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by sorry

end unique_solution_system_l4013_401318


namespace histogram_approximates_density_curve_l4013_401381

/-- Represents a sample frequency distribution histogram --/
structure SampleHistogram where
  sampleSize : ℕ
  groupInterval : ℝ
  distribution : ℝ → ℝ

/-- Represents a population density curve --/
def PopulationDensityCurve := ℝ → ℝ

/-- Measures the difference between a histogram and a density curve --/
def difference (h : SampleHistogram) (p : PopulationDensityCurve) : ℝ := sorry

theorem histogram_approximates_density_curve
  (h : ℕ → SampleHistogram)
  (p : PopulationDensityCurve)
  (hsize : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).sampleSize > 1 / ε)
  (hinterval : ∀ ε > 0, ∃ N, ∀ n ≥ N, (h n).groupInterval < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, difference (h n) p < ε :=
sorry

end histogram_approximates_density_curve_l4013_401381


namespace circle_radius_theorem_l4013_401331

theorem circle_radius_theorem (r : ℝ) (h : r > 0) : 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end circle_radius_theorem_l4013_401331


namespace sara_balloons_l4013_401361

theorem sara_balloons (tom_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : tom_balloons = 9)
  (h2 : total_balloons = 17) :
  total_balloons - tom_balloons = 8 := by sorry

end sara_balloons_l4013_401361


namespace ferry_tourists_sum_ferry_tourists_sum_proof_l4013_401382

theorem ferry_tourists_sum : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun n a d s =>
    n = 10 ∧ a = 120 ∧ d = 2 →
    s = n * (2 * a - (n - 1) * d) / 2 →
    s = 1110

-- The proof is omitted
theorem ferry_tourists_sum_proof : ferry_tourists_sum 10 120 2 1110 := by sorry

end ferry_tourists_sum_ferry_tourists_sum_proof_l4013_401382


namespace smallest_square_with_rook_l4013_401321

/-- Represents a chessboard with rooks -/
structure ChessBoard (n : ℕ) where
  size : ℕ := 3 * n
  rooks : Set (ℕ × ℕ)
  beats_entire_board : ∀ (x y : ℕ), x ≤ size ∧ y ≤ size → 
    ∃ (rx ry : ℕ), (rx, ry) ∈ rooks ∧ (rx = x ∨ ry = y)
  beats_at_most_one : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∧ r1.2 ≠ r2.2) ∨ (r1.1 ≠ r2.1 ∧ r1.2 = r2.2)

/-- The main theorem to be proved -/
theorem smallest_square_with_rook (n : ℕ) (h : n > 0) (board : ChessBoard n) :
  (∀ (k : ℕ), k > 2 * n → 
    ∀ (x y : ℕ), x ≤ board.size - k + 1 → y ≤ board.size - k + 1 →
      ∃ (rx ry : ℕ), (rx, ry) ∈ board.rooks ∧ rx ≥ x ∧ rx < x + k ∧ ry ≥ y ∧ ry < y + k) ∧
  (∃ (x y : ℕ), x ≤ board.size - 2 * n + 1 ∧ y ≤ board.size - 2 * n + 1 ∧
    ∀ (rx ry : ℕ), (rx, ry) ∈ board.rooks → (rx < x ∨ rx ≥ x + 2 * n ∨ ry < y ∨ ry ≥ y + 2 * n)) :=
by sorry

end smallest_square_with_rook_l4013_401321


namespace notebook_purchase_difference_l4013_401343

theorem notebook_purchase_difference (price : ℚ) (marie_count jake_count : ℕ) : 
  price > (1/4 : ℚ) →
  price * marie_count = (15/4 : ℚ) →
  price * jake_count = 5 →
  jake_count - marie_count = 5 :=
by
  sorry

end notebook_purchase_difference_l4013_401343


namespace erdos_szekeres_theorem_l4013_401315

theorem erdos_szekeres_theorem (m n : ℕ) (seq : Fin (m * n + 1) → ℝ) :
  (∃ (subseq : Fin (m + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≤ seq (subseq j))) ∨
  (∃ (subseq : Fin (n + 1) → Fin (m * n + 1)),
    (∀ i j, i < j → subseq i < subseq j) ∧
    (∀ i j, i < j → seq (subseq i) ≥ seq (subseq j))) :=
by sorry

end erdos_szekeres_theorem_l4013_401315


namespace reciprocal_equation_l4013_401348

theorem reciprocal_equation (x : ℚ) : 
  2 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 1 / 2 := by
  sorry

end reciprocal_equation_l4013_401348


namespace mysoon_ornament_collection_l4013_401342

/-- The number of ornaments in Mysoon's collection -/
def total_ornaments : ℕ := 20

/-- The number of handmade ornaments -/
def handmade_ornaments : ℕ := total_ornaments / 6 + 10

/-- The number of handmade antique ornaments -/
def handmade_antique_ornaments : ℕ := total_ornaments / 3

theorem mysoon_ornament_collection :
  (handmade_ornaments = total_ornaments / 6 + 10) ∧
  (handmade_antique_ornaments = handmade_ornaments / 2) ∧
  (handmade_antique_ornaments = total_ornaments / 3) →
  total_ornaments = 20 := by
sorry

end mysoon_ornament_collection_l4013_401342


namespace min_x_prime_factorization_l4013_401399

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 17^15 * 13^2 ∧
    (∀ x' : ℕ+, 13 * x'^7 = 17 * y^11 → x' ≥ x) ∧
    a + b + c + d = 47 := by
  sorry

end min_x_prime_factorization_l4013_401399


namespace product_five_reciprocal_squares_sum_l4013_401358

theorem product_five_reciprocal_squares_sum (a b : ℕ) (h : a * b = 5) :
  (1 : ℝ) / (a^2 : ℝ) + (1 : ℝ) / (b^2 : ℝ) = 1.04 := by
  sorry

end product_five_reciprocal_squares_sum_l4013_401358


namespace log_identity_l4013_401357

theorem log_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hcb : c > b) 
  (h_pythagorean : a^2 + b^2 = c^2) : 
  Real.log a / Real.log (c + b) + Real.log a / Real.log (c - b) = 
  2 * (Real.log a / Real.log (c + b)) * (Real.log a / Real.log (c - b)) := by
sorry

end log_identity_l4013_401357


namespace binomial_coefficient_ratio_l4013_401359

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃ + a₅) = -(122 / 121) := by
sorry

end binomial_coefficient_ratio_l4013_401359


namespace solve_equation_l4013_401372

theorem solve_equation (x : ℤ) (h : 9773 + x = 13200) : x = 3427 := by
  sorry

end solve_equation_l4013_401372


namespace complement_union_A_B_l4013_401392

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

-- Theorem to prove
theorem complement_union_A_B : (U \ (A ∪ B)) = {0, 3, 5} := by sorry

end complement_union_A_B_l4013_401392


namespace marshmallow_challenge_l4013_401303

/-- The marshmallow challenge theorem -/
theorem marshmallow_challenge (haley michael brandon : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  haley + michael + brandon = 44 := by
  sorry

end marshmallow_challenge_l4013_401303


namespace negative_power_product_l4013_401322

theorem negative_power_product (x : ℝ) : -x^2 * x = -x^3 := by
  sorry

end negative_power_product_l4013_401322


namespace unique_intersection_l4013_401344

/-- The value of k for which the line x = k intersects the parabola x = -y^2 - 4y + 2 at exactly one point -/
def intersection_k : ℝ := 6

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -y^2 - 4*y + 2

theorem unique_intersection :
  ∀ k : ℝ, (∃! y : ℝ, k = parabola y) ↔ k = intersection_k :=
by sorry

end unique_intersection_l4013_401344


namespace eggs_left_is_five_l4013_401325

-- Define the problem parameters
def total_eggs : ℕ := 30
def total_cost : ℕ := 500  -- in cents
def price_per_egg : ℕ := 20  -- in cents

-- Define the function to calculate eggs left after recovering capital
def eggs_left_after_recovery : ℕ :=
  total_eggs - (total_cost / price_per_egg)

-- Theorem statement
theorem eggs_left_is_five : eggs_left_after_recovery = 5 := by
  sorry

end eggs_left_is_five_l4013_401325


namespace line_parameterization_l4013_401350

/-- Given a line y = 2x + 5 parameterized as (x, y) = (s, -2) + t(3, m), prove that s = -7/2 and m = 6 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, x = s + 3*t ∧ y = -2 + m*t → y = 2*x + 5) →
  s = -7/2 ∧ m = 6 := by
sorry

end line_parameterization_l4013_401350


namespace max_knights_between_knights_theorem_l4013_401360

/-- Represents the seating arrangement around a round table -/
structure SeatingArrangement where
  knights : ℕ
  samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights between knights for the given arrangement -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.knights = 40)
  (h2 : arrangement.samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights ⟨40, 10, 7⟩

end max_knights_between_knights_theorem_l4013_401360


namespace log_expression_equality_l4013_401300

theorem log_expression_equality (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = Real.sqrt 8 + 5^(2/3) := by
  sorry

end log_expression_equality_l4013_401300


namespace fixed_point_parabola_l4013_401363

theorem fixed_point_parabola (s : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + s * x - 3 * s
  f 3 = 36 := by
  sorry

end fixed_point_parabola_l4013_401363


namespace podcast_storage_theorem_l4013_401398

def podcast_duration : ℕ := 837
def cd_capacity : ℕ := 75

theorem podcast_storage_theorem :
  let num_cds : ℕ := (podcast_duration + cd_capacity - 1) / cd_capacity
  let audio_per_cd : ℚ := podcast_duration / num_cds
  audio_per_cd = 69.75 := by sorry

end podcast_storage_theorem_l4013_401398


namespace sale_price_calculation_l4013_401378

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  let sellingPrice := costPrice * (1 + profitRate)
  sellingPrice * (1 + taxRate)

/-- Theorem stating that the sale price with tax is approximately 677.61 -/
theorem sale_price_calculation :
  let costPrice := 526.50
  let profitRate := 0.17
  let taxRate := 0.10
  abs (salePriceWithTax costPrice profitRate taxRate - 677.61) < 0.01 := by
  sorry

#eval salePriceWithTax 526.50 0.17 0.10

end sale_price_calculation_l4013_401378


namespace total_bottles_proof_l4013_401354

/-- Represents the total number of bottles -/
def total_bottles : ℕ := 180

/-- Represents the number of bottles containing only cider -/
def cider_bottles : ℕ := 40

/-- Represents the number of bottles containing only beer -/
def beer_bottles : ℕ := 80

/-- Represents the number of bottles given to the first house -/
def first_house_bottles : ℕ := 90

/-- Proves that the total number of bottles is 180 given the problem conditions -/
theorem total_bottles_proof :
  total_bottles = cider_bottles + beer_bottles + (2 * first_house_bottles - cider_bottles - beer_bottles) :=
by sorry

end total_bottles_proof_l4013_401354


namespace bank_transfer_balance_l4013_401367

theorem bank_transfer_balance (initial_balance first_transfer second_transfer service_charge_rate : ℝ) 
  (h1 : initial_balance = 400)
  (h2 : first_transfer = 90)
  (h3 : second_transfer = 60)
  (h4 : service_charge_rate = 0.02)
  : initial_balance - (first_transfer + first_transfer * service_charge_rate + second_transfer * service_charge_rate) = 307 := by
  sorry

end bank_transfer_balance_l4013_401367


namespace arithmetic_mean_greater_than_geometric_mean_l4013_401311

theorem arithmetic_mean_greater_than_geometric_mean (x y : ℝ) (hx : x = 16) (hy : y = 64) :
  (x + y) / 2 > Real.sqrt (x * y) := by
  sorry

end arithmetic_mean_greater_than_geometric_mean_l4013_401311


namespace percent_of_y_l4013_401339

theorem percent_of_y (y : ℝ) (h : y > 0) : ((2 * y) / 5 + (3 * y) / 10) / y * 100 = 70 := by
  sorry

end percent_of_y_l4013_401339


namespace tangent_circles_theorem_l4013_401374

/-- Two concentric circles with radii 1 and 3 -/
def inner_radius : ℝ := 1
def outer_radius : ℝ := 3

/-- Radius of circles tangent to both concentric circles -/
def tangent_circle_radius : ℝ := 1

/-- Maximum number of non-overlapping tangent circles -/
def max_tangent_circles : ℕ := 6

/-- Theorem stating the radius of tangent circles and the maximum number of such circles -/
theorem tangent_circles_theorem :
  (tangent_circle_radius = 1) ∧
  (max_tangent_circles = 6) := by
  sorry

#check tangent_circles_theorem

end tangent_circles_theorem_l4013_401374


namespace world_not_ending_l4013_401328

theorem world_not_ending (n : ℕ) : ¬(∃ k : ℕ, (1 + n) = 11 * k ∧ (3 + 7 * n) = 11 * k) := by
  sorry

end world_not_ending_l4013_401328


namespace inequality_proof_l4013_401340

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_proof_l4013_401340


namespace polynomial_equivalence_l4013_401349

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2*(y^2 + 2*y - 5) = 0 :=
by sorry

end polynomial_equivalence_l4013_401349


namespace oil_percentage_in_dressing_q_l4013_401375

/-- Represents the composition of a salad dressing -/
structure Dressing where
  vinegar : ℝ
  oil : ℝ

/-- Represents the mixture of two dressings -/
structure Mixture where
  dressing_p : Dressing
  dressing_q : Dressing
  p_ratio : ℝ
  q_ratio : ℝ
  vinegar : ℝ

/-- Theorem stating that given the conditions of the problem, 
    the oil percentage in dressing Q is 90% -/
theorem oil_percentage_in_dressing_q 
  (p : Dressing)
  (q : Dressing)
  (mix : Mixture)
  (h1 : p.vinegar = 0.3)
  (h2 : p.oil = 0.7)
  (h3 : q.vinegar = 0.1)
  (h4 : mix.dressing_p = p)
  (h5 : mix.dressing_q = q)
  (h6 : mix.p_ratio = 0.1)
  (h7 : mix.q_ratio = 0.9)
  (h8 : mix.vinegar = 0.12)
  : q.oil = 0.9 := by
  sorry

#check oil_percentage_in_dressing_q

end oil_percentage_in_dressing_q_l4013_401375


namespace union_of_A_and_B_l4013_401366

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | x < 3} := by sorry

end union_of_A_and_B_l4013_401366


namespace purely_imaginary_complex_l4013_401391

theorem purely_imaginary_complex (a : ℝ) :
  let z : ℂ := Complex.mk (a + 1) (1 + a^2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end purely_imaginary_complex_l4013_401391


namespace fair_coin_prob_heads_l4013_401320

/-- A fair coin is a coin where the probability of getting heads is equal to the probability of getting tails -/
def is_fair_coin (coin : Type) (prob_heads : coin → ℝ) : Prop :=
  ∀ c : coin, prob_heads c = 1 - prob_heads c

/-- The probability of an event is independent of previous events if the probability remains constant regardless of previous outcomes -/
def is_independent_event {α : Type} (prob : α → ℝ) : Prop :=
  ∀ (a b : α), prob a = prob b

/-- Theorem: For a fair coin, the probability of getting heads on any single toss is 1/2, regardless of previous tosses -/
theorem fair_coin_prob_heads {coin : Type} (prob_heads : coin → ℝ) 
  (h_fair : is_fair_coin coin prob_heads) 
  (h_indep : is_independent_event prob_heads) :
  ∀ c : coin, prob_heads c = 1/2 :=
by sorry

end fair_coin_prob_heads_l4013_401320


namespace f_9_eq_two_thirds_l4013_401345

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is odd -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(x-2) = f(x+2) for all x -/
axiom f_period : ∀ x, f (x - 2) = f (x + 2)

/-- f(x) = 3^x - 1 for x in [-2,0] -/
axiom f_def : ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3^x - 1

/-- The main theorem: f(9) = 2/3 -/
theorem f_9_eq_two_thirds : f 9 = 2/3 := by sorry

end f_9_eq_two_thirds_l4013_401345


namespace worksheet_problems_l4013_401355

theorem worksheet_problems (total_worksheets : ℕ) (graded_worksheets : ℕ) (problems_left : ℕ) :
  total_worksheets = 17 →
  graded_worksheets = 8 →
  problems_left = 63 →
  (total_worksheets - graded_worksheets) * (problems_left / (total_worksheets - graded_worksheets)) = 7 :=
by sorry

end worksheet_problems_l4013_401355


namespace tenth_term_of_sequence_l4013_401324

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem tenth_term_of_sequence :
  let a₁ : ℤ := 10
  let d : ℤ := -2
  arithmetic_sequence a₁ d 10 = -8 := by sorry

end tenth_term_of_sequence_l4013_401324


namespace greater_than_reciprocal_reciprocal_comparison_l4013_401376

theorem greater_than_reciprocal (x : ℝ) : Prop :=
  x ≠ 0 ∧ x > 1 / x

theorem reciprocal_comparison : 
  ¬ greater_than_reciprocal (-3/2) ∧
  ¬ greater_than_reciprocal (-1) ∧
  ¬ greater_than_reciprocal (1/3) ∧
  greater_than_reciprocal 2 ∧
  greater_than_reciprocal 3 := by
sorry

end greater_than_reciprocal_reciprocal_comparison_l4013_401376


namespace pauls_hourly_wage_l4013_401316

/-- Calculates the hourly wage given the number of hours worked, tax rate, expense rate, and remaining money. -/
def calculate_hourly_wage (hours_worked : ℕ) (tax_rate : ℚ) (expense_rate : ℚ) (remaining_money : ℚ) : ℚ :=
  remaining_money / ((1 - expense_rate) * ((1 - tax_rate) * hours_worked))

/-- Theorem stating that under the given conditions, the hourly wage is $12.50 -/
theorem pauls_hourly_wage :
  let hours_worked : ℕ := 40
  let tax_rate : ℚ := 1/5
  let expense_rate : ℚ := 3/20
  let remaining_money : ℚ := 340
  calculate_hourly_wage hours_worked tax_rate expense_rate remaining_money = 25/2 := by
  sorry


end pauls_hourly_wage_l4013_401316


namespace ninth_term_is_512_l4013_401388

/-- Given a geometric sequence where:
  * The first term is 2
  * The common ratio is 2
  * n is the term number
  This function calculates the nth term of the sequence -/
def geometricSequenceTerm (n : ℕ) : ℕ := 2 * 2^(n - 1)

/-- Theorem stating that the 9th term of the geometric sequence is 512 -/
theorem ninth_term_is_512 : geometricSequenceTerm 9 = 512 := by
  sorry

end ninth_term_is_512_l4013_401388


namespace greatest_divisor_with_remainders_l4013_401387

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (ha : a = 1657) (hb : b = 2037) (hr1 : r1 = 6) (hr2 : r2 = 5) :
  Nat.gcd (a - r1) (b - r2) = 127 :=
sorry

end greatest_divisor_with_remainders_l4013_401387


namespace integral_arctan_fraction_l4013_401304

open Real

theorem integral_arctan_fraction (x : ℝ) :
  deriv (fun x => (1/2) * (4 * (arctan x)^2 - log (1 + x^2))) x
  = (4 * arctan x - x) / (1 + x^2) :=
by sorry

end integral_arctan_fraction_l4013_401304


namespace value_of_x_l4013_401341

theorem value_of_x : ∀ (x y z w u : ℤ),
  x = y + 12 →
  y = z + 15 →
  z = w + 25 →
  w = u + 10 →
  u = 95 →
  x = 157 := by
sorry

end value_of_x_l4013_401341
