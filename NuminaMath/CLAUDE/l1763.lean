import Mathlib

namespace lawyer_percentage_l1763_176311

theorem lawyer_percentage (total_members : ℝ) (h1 : total_members > 0) :
  let women_percentage : ℝ := 0.80
  let woman_lawyer_prob : ℝ := 0.32
  let women_lawyers_percentage : ℝ := woman_lawyer_prob / women_percentage
  women_lawyers_percentage = 0.40 := by
  sorry

end lawyer_percentage_l1763_176311


namespace project_hours_difference_l1763_176326

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 216) 
  (kate_hours : ℕ) (pat_hours : ℕ) (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours * 3 = mark_hours) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 120 := by
sorry

end project_hours_difference_l1763_176326


namespace transistors_2010_l1763_176381

/-- Moore's law: number of transistors doubles every two years -/
def moores_law (years : ℕ) : ℕ → ℕ := fun n => n * 2^(years / 2)

/-- Number of transistors in 1995 -/
def transistors_1995 : ℕ := 2000000

/-- Years between 1995 and 2010 -/
def years_passed : ℕ := 15

theorem transistors_2010 :
  moores_law years_passed transistors_1995 = 256000000 := by
  sorry

end transistors_2010_l1763_176381


namespace min_cost_notebooks_l1763_176344

/-- Represents the unit price of type A notebooks -/
def price_A : ℝ := 11

/-- Represents the unit price of type B notebooks -/
def price_B : ℝ := price_A + 1

/-- Represents the total number of notebooks to be purchased -/
def total_notebooks : ℕ := 100

/-- Represents the constraint on the quantity of type B notebooks -/
def type_B_constraint (a : ℕ) : Prop := total_notebooks - a ≤ 3 * a

/-- Represents the cost function for purchasing notebooks -/
def cost_function (a : ℕ) : ℝ := price_A * a + price_B * (total_notebooks - a)

/-- Theorem stating that the minimum cost for purchasing 100 notebooks is $1100 -/
theorem min_cost_notebooks : 
  ∃ (a : ℕ), a ≤ total_notebooks ∧ 
  type_B_constraint a ∧ 
  (∀ (b : ℕ), b ≤ total_notebooks → type_B_constraint b → cost_function a ≤ cost_function b) ∧
  cost_function a = 1100 := by
  sorry

end min_cost_notebooks_l1763_176344


namespace olivias_cookie_baggies_l1763_176343

def cookies_per_baggie : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

theorem olivias_cookie_baggies :
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie = 6 := by
  sorry

end olivias_cookie_baggies_l1763_176343


namespace log_inequality_l1763_176308

theorem log_inequality (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.log x / Real.log 3) ∧ f a > f 2) → a > 2 := by
  sorry

end log_inequality_l1763_176308


namespace a_closed_form_l1763_176348

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 6) / (a n + 1)

theorem a_closed_form (n : ℕ) :
  a n = (3 * 4^(n+1) + 2 * (-1)^(n+1)) / (4^(n+1) + (-1)^n) := by
  sorry

end a_closed_form_l1763_176348


namespace bahs_equivalent_to_1000_yahs_l1763_176387

/-- The number of bahs equivalent to one rah -/
def bah_per_rah : ℚ := 15 / 24

/-- The number of rahs equivalent to one yah -/
def rah_per_yah : ℚ := 9 / 15

/-- The number of bahs equivalent to 1000 yahs -/
def bahs_per_1000_yahs : ℚ := 1000 * rah_per_yah * bah_per_rah

theorem bahs_equivalent_to_1000_yahs : bahs_per_1000_yahs = 375 := by
  sorry

end bahs_equivalent_to_1000_yahs_l1763_176387


namespace total_shaded_area_l1763_176353

/-- Represents the fraction of area shaded at each level of division -/
def shaded_fraction : ℚ := 1 / 4

/-- Represents the ratio between successive terms in the geometric series -/
def common_ratio : ℚ := 1 / 16

/-- Theorem stating that the total shaded area is 4/15 -/
theorem total_shaded_area :
  (shaded_fraction / (1 - common_ratio) : ℚ) = 4 / 15 := by sorry

end total_shaded_area_l1763_176353


namespace martha_blocks_found_l1763_176371

/-- The number of blocks Martha found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Martha's initial number of blocks -/
def martha_initial : ℕ := 4

/-- Martha's final number of blocks -/
def martha_final : ℕ := 84

theorem martha_blocks_found : blocks_found martha_initial martha_final = 80 := by
  sorry

end martha_blocks_found_l1763_176371


namespace arithmetic_sequence_property_l1763_176331

/-- An arithmetic sequence with index starting from 1 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end arithmetic_sequence_property_l1763_176331


namespace rental_cost_11_days_l1763_176366

/-- Calculates the total cost of a car rental given the rental duration, daily rate, and weekly rate. -/
def rental_cost (days : ℕ) (daily_rate : ℕ) (weekly_rate : ℕ) : ℕ :=
  let weeks := days / 7
  let remaining_days := days % 7
  weeks * weekly_rate + remaining_days * daily_rate

/-- Theorem stating that the rental cost for 11 days is $310 given the specified rates. -/
theorem rental_cost_11_days :
  rental_cost 11 30 190 = 310 := by
  sorry

end rental_cost_11_days_l1763_176366


namespace unique_function_solution_l1763_176374

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - f y) = 1 - x - y

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → ∀ x : ℝ, f x = 1/2 - x :=
by sorry

end unique_function_solution_l1763_176374


namespace lightbulb_most_suitable_l1763_176323

/-- Represents a survey option --/
inductive SurveyOption
  | SecurityCheck
  | ClassmateExercise
  | JobInterview
  | LightbulbLifespan

/-- Defines what makes a survey suitable for sampling --/
def suitableForSampling (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.SecurityCheck => false
  | SurveyOption.ClassmateExercise => false
  | SurveyOption.JobInterview => false
  | SurveyOption.LightbulbLifespan => true

/-- Theorem stating that the lightbulb lifespan survey is most suitable for sampling --/
theorem lightbulb_most_suitable :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.LightbulbLifespan →
    suitableForSampling SurveyOption.LightbulbLifespan ∧
    ¬(suitableForSampling option) :=
by
  sorry

#check lightbulb_most_suitable

end lightbulb_most_suitable_l1763_176323


namespace cubic_root_sum_l1763_176315

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 3 and -2 are roots of the equation, then (b+c)/a = -7 -/
theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 3^3 + b * 3^2 + c * 3 + d = 0)
  (h2 : a * (-2)^3 + b * (-2)^2 + c * (-2) + d = 0) :
  (b + c) / a = -7 := by
  sorry

end cubic_root_sum_l1763_176315


namespace max_value_on_circle_l1763_176317

theorem max_value_on_circle (x y z : ℝ) : 
  x^2 + y^2 = 4 → z = 2*x + y → z ≤ 2 * Real.sqrt 5 := by
  sorry

end max_value_on_circle_l1763_176317


namespace range_of_a_squared_minus_2b_l1763_176302

/-- A quadratic function with two real roots in [0, 1] -/
structure QuadraticWithRootsInUnitInterval where
  a : ℝ
  b : ℝ
  has_two_roots_in_unit_interval : ∃ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- The range of a^2 - 2b for quadratic functions with roots in [0, 1] -/
theorem range_of_a_squared_minus_2b (f : QuadraticWithRootsInUnitInterval) :
  ∃ (z : ℝ), z = f.a^2 - 2*f.b ∧ 0 ≤ z ∧ z ≤ 2 :=
sorry

end range_of_a_squared_minus_2b_l1763_176302


namespace gcd_sum_diff_l1763_176378

theorem gcd_sum_diff (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) :
  (Nat.gcd (a + b) (a - b) = 1) ∨ (Nat.gcd (a + b) (a - b) = 2) :=
sorry

end gcd_sum_diff_l1763_176378


namespace excursion_existence_l1763_176393

theorem excursion_existence (S : Finset Nat) (E : Finset (Finset Nat)) 
  (h1 : S.card = 20) 
  (h2 : ∀ e ∈ E, e.card > 0) 
  (h3 : ∀ e ∈ E, e ⊆ S) :
  ∃ e ∈ E, ∀ s ∈ e, (E.filter (λ f => s ∈ f)).card ≥ E.card / 20 := by
sorry


end excursion_existence_l1763_176393


namespace hearty_beads_count_l1763_176360

/-- The number of packages of blue beads Hearty bought -/
def blue_packages : ℕ := 3

/-- The number of packages of red beads Hearty bought -/
def red_packages : ℕ := 5

/-- The number of beads in each package -/
def beads_per_package : ℕ := 40

/-- The total number of beads Hearty has -/
def total_beads : ℕ := (blue_packages + red_packages) * beads_per_package

theorem hearty_beads_count : total_beads = 320 := by
  sorry

end hearty_beads_count_l1763_176360


namespace circle_M_properties_l1763_176337

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the center of a circle
def is_center (cx cy : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - cx)^2 + (y - cy)^2 = (x - cx)^2 + (y - cy)^2

-- Define a tangent line to a circle
def is_tangent_line (m b : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃! x y, circle x y ∧ y = m*x + b

-- Main theorem
theorem circle_M_properties :
  (is_center (-2) 1 circle_M) ∧
  (∀ m b : ℝ, is_tangent_line m b circle_M ∧ 0 = m*(-3) + b → b = -3) :=
sorry

end circle_M_properties_l1763_176337


namespace enriques_commission_l1763_176328

/-- Represents the commission rate as a real number between 0 and 1 -/
def commission_rate : ℝ := 0.15

/-- Represents the number of suits sold -/
def suits_sold : ℕ := 2

/-- Represents the price of each suit in dollars -/
def suit_price : ℝ := 700.00

/-- Represents the number of shirts sold -/
def shirts_sold : ℕ := 6

/-- Represents the price of each shirt in dollars -/
def shirt_price : ℝ := 50.00

/-- Represents the number of loafers sold -/
def loafers_sold : ℕ := 2

/-- Represents the price of each pair of loafers in dollars -/
def loafer_price : ℝ := 150.00

/-- Calculates the total sales amount -/
def total_sales : ℝ := 
  suits_sold * suit_price + shirts_sold * shirt_price + loafers_sold * loafer_price

/-- Theorem: Enrique's commission is $300.00 -/
theorem enriques_commission : commission_rate * total_sales = 300.00 := by
  sorry

end enriques_commission_l1763_176328


namespace discount_percentage_l1763_176365

theorem discount_percentage (regular_price : ℝ) (num_shirts : ℕ) (total_paid : ℝ) : 
  regular_price = 50 ∧ num_shirts = 2 ∧ total_paid = 60 →
  (regular_price * num_shirts - total_paid) / (regular_price * num_shirts) * 100 = 40 := by
sorry

end discount_percentage_l1763_176365


namespace set_operations_l1763_176362

def M : Set ℝ := {x | 4 * x^2 - 4 * x - 15 > 0}

def N : Set ℝ := {x | (x + 1) / (6 - x) < 0}

theorem set_operations (M N : Set ℝ) :
  (M = {x | 4 * x^2 - 4 * x - 15 > 0}) →
  (N = {x | (x + 1) / (6 - x) < 0}) →
  (M ∪ N = {x | x < -1 ∨ x ≥ 5/2}) ∧
  ((Set.univ \ M) ∩ (Set.univ \ N) = {x | -1 ≤ x ∧ x < 5/2}) := by
  sorry

end set_operations_l1763_176362


namespace father_daughter_speed_problem_l1763_176379

theorem father_daughter_speed_problem 
  (total_distance : ℝ) 
  (speed_ratio : ℝ) 
  (speed_increase : ℝ) 
  (time_difference : ℝ) :
  total_distance = 60 ∧ 
  speed_ratio = 2 ∧ 
  speed_increase = 2 ∧ 
  time_difference = 1/12 →
  ∃ (father_speed daughter_speed : ℝ),
    father_speed = 14 ∧ 
    daughter_speed = 28 ∧
    daughter_speed = speed_ratio * father_speed ∧
    (total_distance / (2 * father_speed + speed_increase) - 
     (total_distance / 2) / (father_speed + speed_increase)) = time_difference :=
by sorry

end father_daughter_speed_problem_l1763_176379


namespace possible_winning_scores_for_A_l1763_176304

/-- Represents the outcome of a single question for a team -/
inductive QuestionOutcome
  | Correct
  | Incorrect
  | NoBuzz

/-- Calculates the score for a single question based on the outcome -/
def scoreQuestion (outcome : QuestionOutcome) : Int :=
  match outcome with
  | QuestionOutcome.Correct => 1
  | QuestionOutcome.Incorrect => -1
  | QuestionOutcome.NoBuzz => 0

/-- Calculates the total score for a team based on their outcomes for three questions -/
def calculateScore (q1 q2 q3 : QuestionOutcome) : Int :=
  scoreQuestion q1 + scoreQuestion q2 + scoreQuestion q3

/-- Defines a winning condition for team A -/
def teamAWins (scoreA scoreB : Int) : Prop :=
  scoreA > scoreB

/-- The main theorem stating the possible winning scores for team A -/
theorem possible_winning_scores_for_A :
  ∀ (q1A q2A q3A q1B q2B q3B : QuestionOutcome),
    let scoreA := calculateScore q1A q2A q3A
    let scoreB := calculateScore q1B q2B q3B
    teamAWins scoreA scoreB →
    (scoreA = -1 ∨ scoreA = 0 ∨ scoreA = 1 ∨ scoreA = 3) :=
  sorry


end possible_winning_scores_for_A_l1763_176304


namespace train_platform_time_l1763_176320

theorem train_platform_time (l t T : ℝ) (v : ℝ) (h1 : v > 0) (h2 : l > 0) (h3 : t > 0) :
  v = l / t →
  v = (l + 2.5 * l) / T →
  T = 3.5 * t := by
sorry

end train_platform_time_l1763_176320


namespace dormitory_students_l1763_176310

theorem dormitory_students (T : ℝ) (h1 : T > 0) : 
  let first_year := T / 2
  let second_year := T / 2
  let first_year_undeclared := (4 / 5) * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := 4 * first_year_declared
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / T = 1 / 10 := by
sorry


end dormitory_students_l1763_176310


namespace max_value_constraint_l1763_176332

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2/2 = 1) :
  a * Real.sqrt (1 + b^2) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end max_value_constraint_l1763_176332


namespace carpet_needed_proof_l1763_176358

/-- Given a room with length and width, and an amount of existing carpet,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width existing_carpet : ℝ) : ℝ :=
  length * width - existing_carpet

/-- Proof that for a room of 4 feet by 20 feet with 18 square feet of existing carpet,
    62 square feet of additional carpet is needed. -/
theorem carpet_needed_proof :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

#eval additional_carpet_needed 4 20 18

end carpet_needed_proof_l1763_176358


namespace max_label_outcomes_l1763_176340

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of times a ball is drawn -/
def num_draws : ℕ := 3

/-- The total number of possible outcomes when drawing num_draws times from num_balls balls -/
def total_outcomes : ℕ := num_balls ^ num_draws

/-- The number of outcomes that don't include the maximum label -/
def outcomes_without_max : ℕ := 8

/-- Theorem: The number of ways to draw a maximum label of 3 when drawing 3 balls 
    (with replacement) from a box containing balls labeled 1, 2, and 3 is equal to 19 -/
theorem max_label_outcomes : 
  total_outcomes - outcomes_without_max = 19 := by sorry

end max_label_outcomes_l1763_176340


namespace average_and_relation_implies_values_l1763_176363

theorem average_and_relation_implies_values :
  ∀ x y : ℝ,
  (15 + 30 + x + y) / 4 = 25 →
  x = y + 10 →
  x = 32.5 ∧ y = 22.5 := by
sorry

end average_and_relation_implies_values_l1763_176363


namespace afternoon_fliers_fraction_l1763_176397

theorem afternoon_fliers_fraction (total : ℕ) (morning_fraction : ℚ) (left_over : ℕ) 
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_left : left_over = 1350) :
  (total - left_over - (morning_fraction * total)) / (total - (morning_fraction * total)) = 1 / 4 :=
by sorry

end afternoon_fliers_fraction_l1763_176397


namespace melanie_picked_seven_plums_l1763_176386

/-- The number of plums Melanie picked from the orchard -/
def plums_picked : ℕ := sorry

/-- The number of plums Sam gave to Melanie -/
def plums_from_sam : ℕ := 3

/-- The total number of plums Melanie has now -/
def total_plums : ℕ := 10

/-- Theorem stating that Melanie picked 7 plums from the orchard -/
theorem melanie_picked_seven_plums :
  plums_picked = 7 ∧ plums_picked + plums_from_sam = total_plums :=
sorry

end melanie_picked_seven_plums_l1763_176386


namespace binomial_expansion_equality_l1763_176336

theorem binomial_expansion_equality (a b : ℝ) (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k < n ∧
    (Nat.choose n 0) * a^n = (Nat.choose n 2) * a^(n-2) * b^2) →
  a^2 = n * (n - 1) * b :=
by sorry

end binomial_expansion_equality_l1763_176336


namespace sum_of_powers_positive_l1763_176364

theorem sum_of_powers_positive 
  (a b c : ℝ) 
  (h1 : a * b * c > 0) 
  (h2 : a + b + c > 0) : 
  ∀ n : ℕ, a^n + b^n + c^n > 0 :=
by sorry

end sum_of_powers_positive_l1763_176364


namespace squirrel_acorns_at_spring_l1763_176334

def calculate_acorns_at_spring (initial_stash : ℕ) 
  (first_month_percent second_month_percent third_month_percent : ℚ)
  (first_month_taken second_month_taken third_month_taken : ℚ)
  (first_month_found second_month_lost third_month_found : ℤ) : ℚ :=
  let first_month := (initial_stash : ℚ) * first_month_percent * (1 - first_month_taken) + first_month_found
  let second_month := (initial_stash : ℚ) * second_month_percent * (1 - second_month_taken) - second_month_lost
  let third_month := (initial_stash : ℚ) * third_month_percent * (1 - third_month_taken) + third_month_found
  first_month + second_month + third_month

theorem squirrel_acorns_at_spring :
  calculate_acorns_at_spring 500 (2/5) (3/10) (3/10) (1/5) (1/4) (3/20) 15 10 20 = 425 := by
  sorry

end squirrel_acorns_at_spring_l1763_176334


namespace updated_mean_after_decrement_l1763_176359

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 47 →
  (n * original_mean - n * decrement) / n = 153 := by
  sorry

end updated_mean_after_decrement_l1763_176359


namespace jars_to_fill_l1763_176377

def stars_per_jar : ℕ := 85
def initial_stars : ℕ := 33
def additional_stars : ℕ := 307

theorem jars_to_fill :
  (initial_stars + additional_stars) / stars_per_jar = 4 :=
by sorry

end jars_to_fill_l1763_176377


namespace odd_as_difference_of_squares_l1763_176357

theorem odd_as_difference_of_squares :
  ∀ n : ℤ, Odd n → ∃ a b : ℤ, n = a^2 - b^2 :=
by sorry

end odd_as_difference_of_squares_l1763_176357


namespace ellipse_condition_necessary_not_sufficient_l1763_176375

/-- The condition for the equation to potentially represent an ellipse -/
def ellipse_condition (m : ℝ) : Prop := 1 < m ∧ m < 3

/-- The equation representing a potential ellipse -/
def ellipse_equation (m x y : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / (3 - m) = 1

/-- Predicate for whether the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse_equation m x y ∧ 
  ¬(∃ c : ℝ, ∀ x y : ℝ, ellipse_equation m x y ↔ x^2 + y^2 = c)

theorem ellipse_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → ellipse_condition m) ∧
  ¬(∀ m : ℝ, ellipse_condition m → is_ellipse m) :=
sorry

end ellipse_condition_necessary_not_sufficient_l1763_176375


namespace quadratic_equation_solution_l1763_176383

theorem quadratic_equation_solution :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := -4
  let x₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + 5*x₁ - 4 = 0 ∧ x₂^2 + 5*x₂ - 4 = 0 ∧ x₁ ≠ x₂ :=
by sorry

#check quadratic_equation_solution

end quadratic_equation_solution_l1763_176383


namespace exactly_one_even_l1763_176388

theorem exactly_one_even (a b c : ℕ) : 
  ¬((a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ 
    (a % 2 = 0 ∧ b % 2 = 0) ∨ 
    (a % 2 = 0 ∧ c % 2 = 0) ∨ 
    (b % 2 = 0 ∧ c % 2 = 0)) → 
  ((a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨
   (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0)) :=
by sorry

end exactly_one_even_l1763_176388


namespace highest_numbered_street_l1763_176333

/-- Represents the length of Apple Street in meters -/
def street_length : ℝ := 3200

/-- Represents the distance between intersecting streets in meters -/
def intersection_distance : ℝ := 200

/-- Represents the number of non-numbered streets (Peach and Cherry) -/
def non_numbered_streets : ℕ := 2

/-- Theorem stating that the highest-numbered street is the 14th street -/
theorem highest_numbered_street :
  ⌊street_length / intersection_distance⌋ - non_numbered_streets = 14 := by
  sorry


end highest_numbered_street_l1763_176333


namespace candy_cost_theorem_l1763_176345

def candy_problem (caramel_price : ℚ) : Prop :=
  let candy_bar_price := 2 * caramel_price
  let cotton_candy_price := 2 * candy_bar_price
  6 * candy_bar_price + 3 * caramel_price + cotton_candy_price = 57

theorem candy_cost_theorem : candy_problem 3 := by
  sorry

end candy_cost_theorem_l1763_176345


namespace roper_lawn_cut_area_l1763_176339

/-- Calculates the average area of grass cut per month for a rectangular lawn --/
def average_area_cut_per_month (length width : ℝ) (cuts_per_month_high cuts_per_month_low : ℕ) (months_high months_low : ℕ) : ℝ :=
  let lawn_area := length * width
  let total_cuts_per_year := cuts_per_month_high * months_high + cuts_per_month_low * months_low
  let average_cuts_per_month := total_cuts_per_year / 12
  lawn_area * average_cuts_per_month

/-- Theorem stating that the average area of grass cut per month for Mr. Roper's lawn is 14175 square meters --/
theorem roper_lawn_cut_area :
  average_area_cut_per_month 45 35 15 3 6 6 = 14175 := by sorry

end roper_lawn_cut_area_l1763_176339


namespace am_gm_inequality_for_two_l1763_176301

theorem am_gm_inequality_for_two (x : ℝ) (hx : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end am_gm_inequality_for_two_l1763_176301


namespace cube_shadow_problem_l1763_176368

/-- The shadow area function calculates the area of the shadow cast by a cube,
    excluding the area beneath the cube. -/
def shadow_area (cube_edge : ℝ) (light_height : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem cube_shadow_problem (y : ℝ) : 
  shadow_area 2 y = 200 → 
  ⌊1000 * y⌋ = 6140 := by sorry

end cube_shadow_problem_l1763_176368


namespace remainder_after_adding_4032_l1763_176347

theorem remainder_after_adding_4032 (m : ℤ) (h : m % 8 = 3) :
  (m + 4032) % 8 = 3 := by
  sorry

end remainder_after_adding_4032_l1763_176347


namespace min_weighings_is_two_l1763_176350

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A strategy for finding the real medal -/
def Strategy := List WeighResult → Nat

/-- The total number of medals -/
def totalMedals : Nat := 9

/-- The number of real medals -/
def realMedals : Nat := 1

/-- A weighing operation that compares two sets of medals -/
def weigh (leftSet rightSet : List Nat) : WeighResult := sorry

/-- Checks if a strategy correctly identifies the real medal -/
def isValidStrategy (s : Strategy) : Prop := sorry

/-- The minimum number of weighings required to find the real medal -/
def minWeighings : Nat := sorry

theorem min_weighings_is_two :
  minWeighings = 2 := by sorry

end min_weighings_is_two_l1763_176350


namespace sum_of_fourth_powers_l1763_176338

theorem sum_of_fourth_powers (a : ℝ) (h : (a + 1/a)^4 = 16) : a^4 + 1/a^4 = 2 := by
  sorry

end sum_of_fourth_powers_l1763_176338


namespace smallest_m_for_integral_solutions_l1763_176325

theorem smallest_m_for_integral_solutions :
  let has_integral_solutions (m : ℤ) := ∃ x y : ℤ, 10 * x^2 - m * x + 780 = 0 ∧ 10 * y^2 - m * y + 780 = 0 ∧ x ≠ y
  ∀ m : ℤ, m > 0 → has_integral_solutions m → m ≥ 190 ∧
  has_integral_solutions 190 :=
by sorry


end smallest_m_for_integral_solutions_l1763_176325


namespace special_triangle_properties_l1763_176322

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  side_relation : a = (1/2) * c + b * Real.cos C
  area : (1/2) * a * c * Real.sin ((1/3) * Real.pi) = Real.sqrt 3
  side_b : b = Real.sqrt 13

/-- Properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : 
  t.B = (1/3) * Real.pi ∧ t.a + t.c = 5 := by
  sorry

end special_triangle_properties_l1763_176322


namespace fraction_simplification_l1763_176356

theorem fraction_simplification 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (hyz : y - z / x ≠ 0) : 
  (x + z / y) / (y + z / x) = x / y :=
sorry

end fraction_simplification_l1763_176356


namespace potato_problem_solution_l1763_176389

/-- Represents the potato problem with given conditions --/
def potato_problem (total_potatoes wedge_potatoes wedges_per_potato chips_per_potato : ℕ) : Prop :=
  let remaining_potatoes := total_potatoes - wedge_potatoes
  let chip_potatoes := remaining_potatoes / 2
  let total_chips := chip_potatoes * chips_per_potato
  let total_wedges := wedge_potatoes * wedges_per_potato
  total_chips - total_wedges = 436

/-- Theorem stating the solution to the potato problem --/
theorem potato_problem_solution :
  potato_problem 67 13 8 20 := by
  sorry

#check potato_problem_solution

end potato_problem_solution_l1763_176389


namespace binomial_sum_one_l1763_176382

theorem binomial_sum_one (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x, (a - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end binomial_sum_one_l1763_176382


namespace johns_primary_colors_l1763_176373

/-- Given that John has 5 liters of paint for each color and 15 liters of paint in total,
    prove that the number of primary colors he is using is 3. -/
theorem johns_primary_colors (paint_per_color : ℝ) (total_paint : ℝ) 
    (h1 : paint_per_color = 5)
    (h2 : total_paint = 15) :
    total_paint / paint_per_color = 3 := by
  sorry

end johns_primary_colors_l1763_176373


namespace unique_triple_l1763_176303

theorem unique_triple : 
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x^2 + y - z = 100 ∧ 
    x + y^2 - z = 124 ∧
    x = 12 ∧ y = 13 ∧ z = 57 := by
  sorry

end unique_triple_l1763_176303


namespace b_payment_correct_l1763_176367

/-- The payment for a job completed by three workers A, B, and C. -/
def total_payment : ℚ := 529

/-- The fraction of work completed by A and C together. -/
def work_ac : ℚ := 19 / 23

/-- Calculate the payment for worker B given the total payment and the fraction of work done by A and C. -/
def payment_b (total : ℚ) (work_ac : ℚ) : ℚ :=
  total * (1 - work_ac)

theorem b_payment_correct : payment_b total_payment work_ac = 92 := by
  sorry

end b_payment_correct_l1763_176367


namespace total_coins_is_660_l1763_176316

/-- The number of coins Jayden received -/
def jayden_coins : ℕ := 300

/-- The additional coins Jason received compared to Jayden -/
def jason_extra_coins : ℕ := 60

/-- The total number of coins given to both boys -/
def total_coins : ℕ := jayden_coins + (jayden_coins + jason_extra_coins)

/-- Theorem stating that the total number of coins given to both boys is 660 -/
theorem total_coins_is_660 : total_coins = 660 := by
  sorry

end total_coins_is_660_l1763_176316


namespace one_nonneg_solution_iff_l1763_176370

/-- The quadratic equation with parameter a -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

/-- The condition for having exactly one non-negative solution -/
def has_one_nonneg_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic a x = 0

/-- The theorem stating the condition on parameter a -/
theorem one_nonneg_solution_iff (a : ℝ) :
  has_one_nonneg_solution a ↔ (-1 ≤ a ∧ a ≤ 1) ∨ a = 3 := by sorry

end one_nonneg_solution_iff_l1763_176370


namespace expression_evaluation_l1763_176394

theorem expression_evaluation (d : ℕ) (h : d = 2) : 
  (d^d + d*(d+1)^d)^d = 484 := by
  sorry

end expression_evaluation_l1763_176394


namespace flooring_per_box_l1763_176390

theorem flooring_per_box 
  (room_length : ℝ) 
  (room_width : ℝ) 
  (flooring_done : ℝ) 
  (boxes_needed : ℕ) 
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : flooring_done = 250)
  (h4 : boxes_needed = 7) :
  (room_length * room_width - flooring_done) / boxes_needed = 10 := by
  sorry

end flooring_per_box_l1763_176390


namespace sequence_properties_l1763_176318

def a (n : ℕ+) : ℚ := (9 * n^2 - 9 * n + 2) / (9 * n^2 - 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (∀ n : ℕ+, a n ≠ 99 / 100) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) ∧
  (∃! n : ℕ+, 1 / 3 < a n ∧ a n < 2 / 3) :=
by sorry

end sequence_properties_l1763_176318


namespace triangle_transformation_l1763_176300

-- Define the initial triangle
def initial_triangle : List (ℝ × ℝ) := [(0, 0), (1, 0), (0, 1)]

-- Define the transformation functions
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def translate_right (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2, p.2)

-- Define the composite transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_right (reflect_x_axis (rotate_180 p))

-- Theorem statement
theorem triangle_transformation :
  List.map transform initial_triangle = [(2, 0), (1, 0), (2, 1)] := by
  sorry

end triangle_transformation_l1763_176300


namespace multiples_2_3_not_5_l1763_176341

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_2_3_not_5 (max : ℕ) (h : max = 200) :
  (count_multiples 2 max + count_multiples 3 max - count_multiples 6 max) -
  (count_multiples 10 max + count_multiples 15 max - count_multiples 30 max) = 107 :=
by sorry

end multiples_2_3_not_5_l1763_176341


namespace number_equation_l1763_176361

theorem number_equation (x : ℝ) : 3 * x - 6 = 2 * x ↔ x = 6 := by
  sorry

end number_equation_l1763_176361


namespace arithmetic_cube_reciprocal_roots_l1763_176355

theorem arithmetic_cube_reciprocal_roots :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x = (abs x) ^ (1/2)) →
  (∀ x : ℝ, x > 0 → (x ^ (1/3)) ^ 3 = x) →
  (∀ x : ℝ, x ≠ 0 → x * (1/x) = 1) →
  (Real.sqrt ((-81)^2) = 9) ∧
  ((1/27) ^ (1/3) = 1/3) ∧
  (1 / Real.sqrt 2 = Real.sqrt 2 / 2) := by
  sorry

end arithmetic_cube_reciprocal_roots_l1763_176355


namespace solution_in_quadrant_II_l1763_176352

theorem solution_in_quadrant_II (k : ℝ) :
  (∃ x y : ℝ, 2 * x + y = 6 ∧ k * x - y = 4 ∧ x < 0 ∧ y > 0) ↔ k < -2 := by
  sorry

end solution_in_quadrant_II_l1763_176352


namespace probability_of_two_positive_roots_l1763_176307

-- Define the interval for a
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 4*a - 3

-- Define the condition for two positive roots
def has_two_positive_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- Define the probability measure on the interval
noncomputable def probability_measure : MeasureTheory.Measure ℝ :=
  sorry

-- State the theorem
theorem probability_of_two_positive_roots :
  probability_measure {a ∈ interval | has_two_positive_roots a} = 3/8 :=
sorry

end probability_of_two_positive_roots_l1763_176307


namespace set_intersection_problem_l1763_176324

theorem set_intersection_problem (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end set_intersection_problem_l1763_176324


namespace davids_biology_marks_l1763_176319

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def average_marks : ℕ := 93
def total_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * total_subjects
  all_subjects_total - known_subjects_total = 95 := by
  sorry

end davids_biology_marks_l1763_176319


namespace sally_balloons_l1763_176305

/-- 
Given that Sally has x orange balloons initially, finds 2 more orange balloons,
and ends up with 11 orange balloons in total, prove that x = 9.
-/
theorem sally_balloons (x : ℝ) : x + 2 = 11 → x = 9 := by
  sorry

end sally_balloons_l1763_176305


namespace f_properties_l1763_176395

noncomputable section

-- Define the function f
def f (p : ℝ) (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (p + x)

-- State the theorem
theorem f_properties (p : ℝ) (a : ℝ) (h_p : p > -1) (h_a : 0 < a ∧ a < 1) :
  -- Part 1: Domain of f
  (∀ x, f p x ≠ 0 ↔ -p < x ∧ x < 1) ∧
  -- Part 2: Minimum value of f when p = 1
  (∃ min_val, ∀ x, -a < x ∧ x ≤ a → f 1 x ≥ min_val) ∧
  (f 1 a = Real.log (1 - a^2)) :=
sorry

end

end f_properties_l1763_176395


namespace boat_distance_along_stream_l1763_176376

def boat_problem (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  let stream_speed := boat_speed - against_stream_distance
  boat_speed + stream_speed

theorem boat_distance_along_stream 
  (boat_speed : ℝ) 
  (against_stream_distance : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : against_stream_distance = 9) : 
  boat_problem boat_speed against_stream_distance = 21 := by
  sorry

end boat_distance_along_stream_l1763_176376


namespace test_time_calculation_l1763_176335

theorem test_time_calculation (total_questions : ℕ) (unanswered : ℕ) (time_per_question : ℕ) : 
  total_questions = 100 →
  unanswered = 40 →
  time_per_question = 2 →
  (total_questions - unanswered) * time_per_question / 60 = 2 := by
  sorry

end test_time_calculation_l1763_176335


namespace absolute_value_simplification_l1763_176314

theorem absolute_value_simplification : |(-4^3 + 5^2 - 6)| = 45 := by
  sorry

end absolute_value_simplification_l1763_176314


namespace linear_functions_product_sign_l1763_176309

theorem linear_functions_product_sign (a b c d : ℝ) :
  b < 0 →
  d < 0 →
  ((a > 0 ∧ c < 0) ∨ (a < 0 ∧ c > 0)) →
  a * b * c * d < 0 := by
sorry

end linear_functions_product_sign_l1763_176309


namespace factor_sum_l1763_176372

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 4) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 15 := by
sorry

end factor_sum_l1763_176372


namespace benny_seashells_l1763_176398

theorem benny_seashells (initial_seashells : Real) (percentage_given : Real) 
  (h1 : initial_seashells = 66.5)
  (h2 : percentage_given = 75) :
  initial_seashells - (percentage_given / 100) * initial_seashells = 16.625 := by
  sorry

end benny_seashells_l1763_176398


namespace system_solution_l1763_176392

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 12 ∧ 9 * x - 12 * y = -24 ∧ x = 2/3 ∧ y = 5/2 := by
  sorry

end system_solution_l1763_176392


namespace solution_set_inequality_l1763_176399

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 3) = {x | (x - 1)^2 * (x + 2) * (x - 3) ≤ 0} := by sorry

end solution_set_inequality_l1763_176399


namespace complex_number_properties_l1763_176391

/-- The complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 4*m) (m^2 - m - 6)

/-- Predicate for a complex number being in the third quadrant -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- Predicate for a complex number being on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Predicate for a complex number being on the line x - y + 3 = 0 -/
def on_line (z : ℂ) : Prop := z.re - z.im + 3 = 0

theorem complex_number_properties (m : ℝ) :
  (in_third_quadrant (z m) ↔ 0 < m ∧ m < 3) ∧
  (on_imaginary_axis (z m) ↔ m = 0 ∨ m = 4) ∧
  (on_line (z m) ↔ m = 3) := by sorry

end complex_number_properties_l1763_176391


namespace triangle_coloring_theorem_l1763_176329

/-- The number of colors available for coloring the triangle vertices -/
def num_colors : ℕ := 4

/-- The number of vertices in a triangle -/
def num_vertices : ℕ := 3

/-- 
Calculates the number of ways to color the vertices of a triangle
such that no two vertices have the same color
-/
def triangle_coloring_ways : ℕ :=
  num_colors * (num_colors - 1) * (num_colors - 2)

/-- 
Theorem: The number of ways to color the vertices of a triangle
with 4 colors, such that no two vertices have the same color, is 24
-/
theorem triangle_coloring_theorem : 
  triangle_coloring_ways = 24 := by
  sorry

end triangle_coloring_theorem_l1763_176329


namespace actual_distance_traveled_l1763_176396

/-- Proves that the actual distance traveled is 20 km given the conditions -/
theorem actual_distance_traveled (initial_speed time_taken : ℝ) 
  (h1 : initial_speed = 5)
  (h2 : initial_speed * time_taken + 20 = 2 * initial_speed * time_taken) :
  initial_speed * time_taken = 20 := by
  sorry

#check actual_distance_traveled

end actual_distance_traveled_l1763_176396


namespace pencils_added_l1763_176321

theorem pencils_added (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 41)
  (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end pencils_added_l1763_176321


namespace inequality_proof_l1763_176313

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 10 / 2 := by
  sorry

end inequality_proof_l1763_176313


namespace circle_C_properties_l1763_176384

/-- Circle C defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of circle C --/
def center : ℝ × ℝ := (1, -2)

/-- The radius of circle C --/
def radius : ℝ := 3

/-- A line with slope 1 --/
def line_with_slope_1 (a b : ℝ) (x y : ℝ) : Prop := y - b = x - a

/-- Theorem stating the properties of circle C and the existence of special lines --/
theorem circle_C_properties :
  (∀ x y : ℝ, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  (∃ a b : ℝ, (line_with_slope_1 a b (0) (0) ∧ 
              (line_with_slope_1 a b (-4) (-4) ∨ line_with_slope_1 a b (1) (1)) ∧
              (∃ x₁ y₁ x₂ y₂ : ℝ, 
                circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                line_with_slope_1 a b x₁ y₁ ∧ line_with_slope_1 a b x₂ y₂ ∧
                (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * ((x₁ + x₂)/2)^2 + 4 * ((y₁ + y₂)/2)^2))) :=
sorry

end circle_C_properties_l1763_176384


namespace shopping_expenditure_l1763_176327

theorem shopping_expenditure (x : ℝ) 
  (emma_spent : x > 0)
  (elsa_spent : ℝ → ℝ)
  (elizabeth_spent : ℝ → ℝ)
  (elsa_condition : elsa_spent x = 2 * x)
  (elizabeth_condition : elizabeth_spent x = 4 * elsa_spent x)
  (total_spent : x + elsa_spent x + elizabeth_spent x = 638) :
  x = 58 := by
sorry

end shopping_expenditure_l1763_176327


namespace factorization_equality_l1763_176385

theorem factorization_equality (a x y : ℝ) : 
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end factorization_equality_l1763_176385


namespace fruit_salad_count_l1763_176354

def total_fruit_salads (alaya_salads : ℕ) (angel_multiplier : ℕ) : ℕ :=
  alaya_salads + angel_multiplier * alaya_salads

theorem fruit_salad_count :
  total_fruit_salads 200 2 = 600 :=
by sorry

end fruit_salad_count_l1763_176354


namespace trivia_team_score_l1763_176306

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 14 →
  absent_members = 7 →
  total_points = 35 →
  (total_points / (total_members - absent_members) : ℚ) = 5 := by
  sorry

end trivia_team_score_l1763_176306


namespace r_th_term_of_sequence_l1763_176369

/-- Given a sequence where the sum of the first n terms is Sn = 3n + 4n^2,
    prove that the r-th term of the sequence is 8r - 1 -/
theorem r_th_term_of_sequence (n r : ℕ) (Sn : ℕ → ℤ) 
  (h : ∀ n, Sn n = 3*n + 4*n^2) :
  Sn r - Sn (r-1) = 8*r - 1 := by
  sorry

end r_th_term_of_sequence_l1763_176369


namespace y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l1763_176342

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

-- Theorem for the maximum and minimum values of y/x
theorem y_over_x_bounds {x y : ℝ} (h : satisfies_equation x y) (hx : x ≠ 0) :
  y / x ≤ Real.sqrt 3 ∧ y / x ≥ -Real.sqrt 3 :=
sorry

-- Theorem for the maximum and minimum values of y - x
theorem y_minus_x_bounds {x y : ℝ} (h : satisfies_equation x y) :
  y - x ≤ -2 + Real.sqrt 6 ∧ y - x ≥ -2 - Real.sqrt 6 :=
sorry

-- Theorem for the maximum and minimum values of x^2 + y^2
theorem x_squared_plus_y_squared_bounds {x y : ℝ} (h : satisfies_equation x y) :
  x^2 + y^2 ≤ 7 + 4 * Real.sqrt 3 ∧ x^2 + y^2 ≥ 7 - 4 * Real.sqrt 3 :=
sorry

end y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l1763_176342


namespace employed_females_percentage_l1763_176351

/-- Given the employment statistics of town X, calculate the percentage of employed females among all employed people. -/
theorem employed_females_percentage (total_employed : ℝ) (employed_males : ℝ) 
  (h1 : total_employed = 60) 
  (h2 : employed_males = 48) : 
  (total_employed - employed_males) / total_employed * 100 = 20 := by
  sorry

end employed_females_percentage_l1763_176351


namespace rectangle_circle_area_ratio_l1763_176330

theorem rectangle_circle_area_ratio (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * Real.pi * r) (h2 : l = 2 * w) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end rectangle_circle_area_ratio_l1763_176330


namespace smallest_possible_a_l1763_176346

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (29 * ↑x)) :
  ∃ a_min : ℝ, a_min = 10 * Real.pi - 29 ∧ 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (29 * ↑x)) → a_min ≤ a') :=
sorry

end smallest_possible_a_l1763_176346


namespace A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1763_176349

/-- Given algebraic expressions A and B -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B = 5xy - 2x + 2y -/
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

/-- Theorem 2: A - 2B = -7 when x = -1 and y = 3 -/
theorem A_minus_2B_specific : A (-1) 3 - 2 * B (-1) 3 = -7 := by sorry

/-- Theorem 3: y = 2/5 when A - 2B is independent of x -/
theorem y_value_when_independent (y : ℝ) :
  (∀ x, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1763_176349


namespace arccos_sqrt3_over_2_l1763_176380

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end arccos_sqrt3_over_2_l1763_176380


namespace beth_book_collection_l1763_176312

theorem beth_book_collection (novels_percent : Real) (graphic_novels : Nat) (comic_books_percent : Real) :
  novels_percent = 0.65 →
  comic_books_percent = 0.2 →
  graphic_novels = 18 →
  ∃ (total_books : Nat), 
    (novels_percent + comic_books_percent + (graphic_novels : Real) / total_books) = 1 ∧
    total_books = 120 := by
  sorry

end beth_book_collection_l1763_176312
