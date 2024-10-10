import Mathlib

namespace sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l2893_289357

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by
  sorry

theorem sqrt_three_times_sqrt_two_equals_sqrt_six : 
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end sqrt_product_sqrt_three_times_sqrt_two_equals_sqrt_six_l2893_289357


namespace puzzle_solution_l2893_289379

-- Define the types of characters
inductive Character
| Human
| Ape

-- Define the types of statements
inductive StatementType
| Truthful
| Lie

-- Define a structure for a person
structure Person where
  species : Character
  statementType : StatementType

-- Define the statements made by A and B
def statement_A (b : Person) (a : Person) : Prop :=
  b.statementType = StatementType.Lie ∧ 
  b.species = Character.Ape ∧ 
  a.species = Character.Human

def statement_B (a : Person) : Prop :=
  a.statementType = StatementType.Truthful

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Person),
    (statement_A b a = (a.statementType = StatementType.Lie)) ∧
    (statement_B a = (b.statementType = StatementType.Lie)) ∧
    a.species = Character.Ape ∧
    a.statementType = StatementType.Lie ∧
    b.species = Character.Human ∧
    b.statementType = StatementType.Lie :=
  sorry

end puzzle_solution_l2893_289379


namespace max_profit_at_seventh_grade_l2893_289337

/-- Represents the profit function for a product with different quality grades. -/
def profit_function (x : ℕ) : ℝ :=
  let profit_per_unit := 6 + 2 * (x - 1)
  let units_produced := 60 - 4 * (x - 1)
  profit_per_unit * units_produced

/-- Represents the maximum grade available. -/
def max_grade : ℕ := 10

/-- Theorem stating that the 7th grade maximizes profit and the maximum profit is 648 yuan. -/
theorem max_profit_at_seventh_grade :
  (∃ (x : ℕ), x ≤ max_grade ∧ ∀ (y : ℕ), y ≤ max_grade → profit_function x ≥ profit_function y) ∧
  (∃ (x : ℕ), x ≤ max_grade ∧ profit_function x = 648) ∧
  profit_function 7 = 648 := by
  sorry

#eval profit_function 7  -- Should output 648

end max_profit_at_seventh_grade_l2893_289337


namespace negation_of_universal_proposition_l2893_289361

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → 2^x > x^2) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x ≤ x^2) :=
by sorry

end negation_of_universal_proposition_l2893_289361


namespace rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l2893_289378

/-- Represents an event that could potentially be surveyed. -/
inductive Event
  | AirplaneSecurity
  | SpacecraftInspection
  | TeacherRecruitment
  | RiceDumplingQuality

/-- Characteristics that make an event suitable for sampling survey. -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_inspection : Bool
  representative_sample_possible : Bool

/-- Defines the characteristics of a sampling survey for each event. -/
def event_characteristics : Event → SamplingSurveyCharacteristics
  | Event.AirplaneSecurity => ⟨false, false, false⟩
  | Event.SpacecraftInspection => ⟨false, false, false⟩
  | Event.TeacherRecruitment => ⟨false, false, false⟩
  | Event.RiceDumplingQuality => ⟨true, true, true⟩

/-- Determines if an event is suitable for a sampling survey based on its characteristics. -/
def is_suitable_for_sampling (e : Event) : Prop :=
  let c := event_characteristics e
  c.large_population ∧ c.impractical_full_inspection ∧ c.representative_sample_possible

/-- Theorem stating that the rice dumpling quality investigation is suitable for a sampling survey. -/
theorem rice_dumpling_suitable_for_sampling :
  is_suitable_for_sampling Event.RiceDumplingQuality :=
by
  sorry

/-- Theorem stating that the rice dumpling quality investigation is the only event suitable for a sampling survey. -/
theorem only_rice_dumpling_suitable_for_sampling :
  ∀ e : Event, is_suitable_for_sampling e ↔ e = Event.RiceDumplingQuality :=
by
  sorry

end rice_dumpling_suitable_for_sampling_only_rice_dumpling_suitable_for_sampling_l2893_289378


namespace mutually_exclusive_not_contradictory_l2893_289385

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The result of drawing two balls from the bag -/
inductive Draw
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Define the events -/
def exactlyOneBlack (d : Draw) : Prop :=
  d = Draw.OneRedOneBlack

def exactlyTwoBlack (d : Draw) : Prop :=
  d = Draw.TwoBlack

/-- The probability of a draw given a bag -/
def prob (b : Bag) (d : Draw) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem mutually_exclusive_not_contradictory (b : Bag) 
  (h1 : b.red = 2) (h2 : b.black = 2) : 
  (∀ d, ¬(exactlyOneBlack d ∧ exactlyTwoBlack d)) ∧ 
  (∃ d, exactlyOneBlack d ∨ exactlyTwoBlack d) :=
sorry

end mutually_exclusive_not_contradictory_l2893_289385


namespace muscle_gain_percentage_l2893_289349

/-- Proves that the percentage of body weight gained in muscle is 20% -/
theorem muscle_gain_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 120)
  (h2 : final_weight = 150)
  (h3 : ∀ (x : ℝ), x * initial_weight + (x / 4) * initial_weight = final_weight - initial_weight) :
  ∃ (muscle_gain_percent : ℝ), muscle_gain_percent = 20 := by
  sorry

end muscle_gain_percentage_l2893_289349


namespace other_number_from_hcf_lcm_and_one_number_l2893_289302

/-- Given two positive integers with known HCF, LCM, and one of the numbers,
    prove that the other number is as calculated. -/
theorem other_number_from_hcf_lcm_and_one_number
  (a b : ℕ+) 
  (hcf : Nat.gcd a b = 16)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 176 := by
  sorry

end other_number_from_hcf_lcm_and_one_number_l2893_289302


namespace fraction_multiplication_l2893_289384

theorem fraction_multiplication : (3 : ℚ) / 4 * 5 / 7 * 11 / 13 = 165 / 364 := by
  sorry

end fraction_multiplication_l2893_289384


namespace smallest_n_with_g_having_8_or_higher_l2893_289366

/-- Sum of digits in base b representation of n -/
def sumDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-five representation of n -/
def f (n : ℕ) : ℕ := sumDigits n 5

/-- g(n) is the sum of digits in base-nine representation of f(n) -/
def g (n : ℕ) : ℕ := sumDigits (f n) 9

/-- A number has a digit '8' or higher in base-nine if it's greater than or equal to 8 -/
def hasDigit8OrHigher (n : ℕ) : Prop := n ≥ 8

theorem smallest_n_with_g_having_8_or_higher :
  (∀ m : ℕ, m < 248 → ¬hasDigit8OrHigher (g m)) ∧ hasDigit8OrHigher (g 248) :=
sorry

end smallest_n_with_g_having_8_or_higher_l2893_289366


namespace triangle_equilateral_l2893_289311

/-- A triangle with sides a, b, c corresponding to angles A, B, C is equilateral if
    a * cos(C) = c * cos(A) and a, b, c are in geometric progression. -/
theorem triangle_equilateral (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  a * Real.cos C = c * Real.cos A →
  ∃ r : ℝ, r > 0 ∧ a = b / r ∧ b = c / r →
  a = b ∧ b = c := by sorry

end triangle_equilateral_l2893_289311


namespace blue_marbles_after_replacement_l2893_289317

/-- Represents the distribution of marbles in a jar -/
structure MarbleDistribution where
  total : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ
  orange : ℕ
  green : ℕ

/-- Calculates the number of blue marbles after replacement -/
def blueMarbleCount (dist : MarbleDistribution) : ℕ :=
  dist.blue + dist.red / 3

theorem blue_marbles_after_replacement (dist : MarbleDistribution) 
  (h1 : dist.total = 160)
  (h2 : dist.red = 40)
  (h3 : dist.yellow = 32)
  (h4 : dist.blue = 16)
  (h5 : dist.purple = 24)
  (h6 : dist.orange = 8)
  (h7 : dist.green = 40)
  (h8 : dist.total = dist.red + dist.yellow + dist.blue + dist.purple + dist.orange + dist.green) :
  blueMarbleCount dist = 29 := by
  sorry

end blue_marbles_after_replacement_l2893_289317


namespace liam_money_left_l2893_289334

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (
  savings_rate : ℕ
) (
  savings_duration_months : ℕ
) (
  bills_cost : ℕ
) : ℕ :=
  savings_rate * savings_duration_months - bills_cost

/-- Proves that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end liam_money_left_l2893_289334


namespace a_greater_than_b_greater_than_one_l2893_289322

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (hn : n ≥ 2)
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a^n = a + 1)
  (eq2 : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end a_greater_than_b_greater_than_one_l2893_289322


namespace smallest_multiple_360_l2893_289350

theorem smallest_multiple_360 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≥ 360 :=
by
  sorry

end smallest_multiple_360_l2893_289350


namespace cubic_polynomial_property_l2893_289301

/-- Represents a cubic polynomial of the form x³ + px² + qx + r -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The sum of zeros of a cubic polynomial -/
def sumOfZeros (poly : CubicPolynomial) : ℝ := -poly.p

/-- The product of zeros of a cubic polynomial -/
def productOfZeros (poly : CubicPolynomial) : ℝ := -poly.r

/-- The sum of coefficients of a cubic polynomial -/
def sumOfCoefficients (poly : CubicPolynomial) : ℝ := 1 + poly.p + poly.q + poly.r

/-- The y-intercept of a cubic polynomial -/
def yIntercept (poly : CubicPolynomial) : ℝ := poly.r

theorem cubic_polynomial_property (poly : CubicPolynomial) :
  sumOfZeros poly = 2 * productOfZeros poly ∧
  sumOfZeros poly = sumOfCoefficients poly ∧
  yIntercept poly = 5 →
  poly.q = -24 := by sorry

end cubic_polynomial_property_l2893_289301


namespace sandwich_cost_l2893_289372

-- Define the given values
def num_sandwiches : ℕ := 3
def num_energy_bars : ℕ := 3
def num_drinks : ℕ := 2
def drink_cost : ℚ := 4
def energy_bar_cost : ℚ := 3
def energy_bar_discount : ℚ := 0.2
def total_spent : ℚ := 40.80

-- Define the theorem
theorem sandwich_cost :
  let drink_total : ℚ := num_drinks * drink_cost
  let energy_bar_total : ℚ := num_energy_bars * energy_bar_cost * (1 - energy_bar_discount)
  let sandwich_total : ℚ := total_spent - drink_total - energy_bar_total
  sandwich_total / num_sandwiches = 8.53 := by sorry

end sandwich_cost_l2893_289372


namespace probability_three_out_of_seven_greater_than_six_l2893_289343

/-- The probability of a single 12-sided die showing a number greater than 6 -/
def p_greater_than_6 : ℚ := 1 / 2

/-- The number of dice rolled -/
def n : ℕ := 7

/-- The number of dice we want to show a number greater than 6 -/
def k : ℕ := 3

/-- The probability of exactly k out of n dice showing a number greater than 6 -/
def probability_k_out_of_n (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_three_out_of_seven_greater_than_six :
  probability_k_out_of_n n k p_greater_than_6 = 35 / 128 := by
  sorry

end probability_three_out_of_seven_greater_than_six_l2893_289343


namespace katy_brownies_l2893_289308

theorem katy_brownies (x : ℕ) : 
  x + 2 * x = 15 → x = 5 := by sorry

end katy_brownies_l2893_289308


namespace max_vertex_sum_l2893_289399

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- Calculates the sum of vertex coordinates for a given parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T - (36 : ℚ) * p.T^2 / (2 * p.T + 2)^2

/-- Theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum :
  ∀ p : Parabola, vertexSum p ≤ (-5 : ℚ) / 4 := by sorry

end max_vertex_sum_l2893_289399


namespace inequality_system_solution_range_l2893_289393

theorem inequality_system_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3 ∧ x - a < 0) ↔ x < a) → 
  a ≤ 2 := by
sorry

end inequality_system_solution_range_l2893_289393


namespace smallest_n_l2893_289376

/-- The smallest three-digit positive integer n such that n + 7 is divisible by 9 and n - 10 is divisible by 6 -/
theorem smallest_n : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 10)) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 10))) → False) ∧
  n = 118 := by
sorry

end smallest_n_l2893_289376


namespace rectangular_plot_difference_l2893_289341

/-- Proves that for a rectangular plot with breadth 8 meters and area 18 times its breadth,
    the difference between the length and the breadth is 10 meters. -/
theorem rectangular_plot_difference (length breadth : ℝ) : 
  breadth = 8 →
  length * breadth = 18 * breadth →
  length - breadth = 10 := by
  sorry

end rectangular_plot_difference_l2893_289341


namespace howard_last_week_money_l2893_289323

/-- Howard's money situation --/
def howard_money (current_money washing_money last_week_money : ℕ) : Prop :=
  current_money = washing_money + last_week_money

/-- Theorem: Howard had 26 dollars last week --/
theorem howard_last_week_money :
  ∃ (last_week_money : ℕ),
    howard_money 52 26 last_week_money ∧ last_week_money = 26 :=
sorry

end howard_last_week_money_l2893_289323


namespace evaluate_expression_l2893_289318

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a^2 - 15 * a + 5) * (3 * a - 4) = 6 := by
  sorry

end evaluate_expression_l2893_289318


namespace square_numbers_between_24_and_150_divisible_by_6_l2893_289369

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem square_numbers_between_24_and_150_divisible_by_6 :
  {x : ℕ | 24 < x ∧ x < 150 ∧ is_square x ∧ x % 6 = 0} = {36, 144} := by
  sorry

end square_numbers_between_24_and_150_divisible_by_6_l2893_289369


namespace max_comic_books_l2893_289381

/-- The cost function for buying comic books -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 1.2 * n else 12 + 1.1 * (n - 10)

/-- Jason's budget -/
def budget : ℚ := 15

/-- Predicate to check if a number of books is affordable -/
def is_affordable (n : ℕ) : Prop :=
  cost n ≤ budget

/-- The maximum number of comic books Jason can buy -/
def max_books : ℕ := 12

theorem max_comic_books : 
  (∀ n : ℕ, is_affordable n → n ≤ max_books) ∧ 
  is_affordable max_books :=
sorry

end max_comic_books_l2893_289381


namespace science_fiction_total_pages_l2893_289358

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by sorry

end science_fiction_total_pages_l2893_289358


namespace system_solution_ratio_l2893_289377

theorem system_solution_ratio (x y z : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  let k : ℝ := 95 / 12
  (x + k * y + 4 * z = 0) →
  (4 * x + k * y - 3 * z = 0) →
  (3 * x + 5 * y - 4 * z = 0) →
  x^2 * z / y^3 = -60 := by
sorry

end system_solution_ratio_l2893_289377


namespace hyperbola_asymptotes_l2893_289333

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola m x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
   -- The foci of the hyperbola and ellipse are the same
   (x₁ = x₂ ∧ y₁ = y₂)) →
  (∀ x y : ℝ, hyperbola m x y → asymptotes x y) :=
sorry

end hyperbola_asymptotes_l2893_289333


namespace triangle_area_l2893_289395

theorem triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (BC = 8 ∧ AB = 10 ∧ AC^2 + BC^2 = AB^2) →
  (1/2 * BC * AC = 24) :=
by
  sorry

end triangle_area_l2893_289395


namespace jill_watching_time_l2893_289332

/-- The length of the first show Jill watched, in minutes. -/
def first_show_length : ℝ := 30

/-- The length of the second show Jill watched, in minutes. -/
def second_show_length : ℝ := 4 * first_show_length

/-- The total time Jill spent watching shows, in minutes. -/
def total_watching_time : ℝ := 150

theorem jill_watching_time :
  first_show_length + second_show_length = total_watching_time :=
by sorry

end jill_watching_time_l2893_289332


namespace expression_value_l2893_289312

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end expression_value_l2893_289312


namespace congruence_solution_l2893_289390

theorem congruence_solution (n : ℕ) : n = 21 → 0 ≤ n ∧ n < 47 ∧ (13 * n) % 47 = 8 := by
  sorry

end congruence_solution_l2893_289390


namespace complex_magnitude_l2893_289325

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = -4 + 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_l2893_289325


namespace parabola_line_intersection_l2893_289346

/-- Given a line and a parabola in a Cartesian coordinate system, 
    this theorem states the conditions for the parabola to intersect 
    the line segment between two points on the line at two distinct points. -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a_neq_zero : a ≠ 0) 
  (h_line : ∀ x y : ℝ, y = (1/2) * x + 1/2 ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) 
  (h_parabola : ∀ x y : ℝ, y = a * x^2 - x + 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧
   ((1/2) * x1 + 1/2 = a * x1^2 - x1 + 1) ∧
   ((1/2) * x2 + 1/2 = a * x2^2 - x2 + 1)) ↔
  (1 ≤ a ∧ a < 9/8) :=
by sorry

end parabola_line_intersection_l2893_289346


namespace coal_burning_duration_l2893_289387

theorem coal_burning_duration (total : ℝ) (burned_fraction : ℝ) (burned_days : ℝ) 
  (h1 : total > 0)
  (h2 : burned_fraction = 2 / 9)
  (h3 : burned_days = 6)
  (h4 : burned_fraction < 1) :
  (total - burned_fraction * total) / (burned_fraction * total / burned_days) = 21 := by
  sorry

end coal_burning_duration_l2893_289387


namespace pizzas_served_today_l2893_289321

theorem pizzas_served_today (lunch_pizzas dinner_pizzas : ℝ) 
  (h1 : lunch_pizzas = 12.5)
  (h2 : dinner_pizzas = 8.25) : 
  lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end pizzas_served_today_l2893_289321


namespace sphere_radius_from_shadows_l2893_289304

/-- Given a sphere and a cone on a horizontal field with parallel sun rays,
    if the sphere's shadow extends 20 meters from its base,
    and a 3-meter-high cone casts a 5-meter-long shadow,
    then the radius of the sphere is 12 meters. -/
theorem sphere_radius_from_shadows (sphere_shadow : ℝ) (cone_height cone_shadow : ℝ)
  (h_sphere_shadow : sphere_shadow = 20)
  (h_cone_height : cone_height = 3)
  (h_cone_shadow : cone_shadow = 5) :
  sphere_shadow * (cone_height / cone_shadow) = 12 :=
by sorry

end sphere_radius_from_shadows_l2893_289304


namespace book_cost_problem_l2893_289329

theorem book_cost_problem : ∃ (s b c : ℕ+), 
  s > 18 ∧ 
  b > 1 ∧ 
  c > b ∧ 
  s * b * c = 3203 ∧ 
  c = 11 := by
  sorry

end book_cost_problem_l2893_289329


namespace min_value_a_plus_b_l2893_289327

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 2 / (1 + b) = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / (x + 1) + 2 / (1 + y) = 1 → a + b ≤ x + y ∧ a + b = 2 * Real.sqrt 2 + 1 :=
sorry

end min_value_a_plus_b_l2893_289327


namespace dice_probability_l2893_289362

/-- The number of possible outcomes for a single die roll -/
def die_outcomes : ℕ := 6

/-- The number of favorable outcomes for a single die roll (not equal to 2) -/
def favorable_outcomes : ℕ := 5

/-- The probability that (a-2)(b-2)(c-2) ≠ 0 when three standard dice are tossed -/
theorem dice_probability : 
  (favorable_outcomes ^ 3 : ℚ) / (die_outcomes ^ 3 : ℚ) = 125 / 216 := by
sorry

end dice_probability_l2893_289362


namespace georges_new_socks_l2893_289347

theorem georges_new_socks (initial_socks : ℝ) (dad_socks : ℝ) (total_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : dad_socks = 4)
  (h3 : total_socks = 68) :
  ↑total_socks - initial_socks - dad_socks = 36 :=
by sorry

end georges_new_socks_l2893_289347


namespace ratio_w_to_y_l2893_289365

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 5 / 3)
  (hz : z / x = 1 / 6) :
  w / y = 9 / 1 := by
sorry

end ratio_w_to_y_l2893_289365


namespace polynomial_divisibility_l2893_289367

/-- A polynomial of degree 4 with coefficients a, b, and c. -/
def polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  x^4 + a*x^2 + b*x + c

/-- The condition for divisibility by (x-1)^3. -/
def isDivisibleByXMinusOneCubed (a b c : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, polynomial a b c x = (x - 1)^3 * q x

/-- Theorem stating the necessary and sufficient conditions for divisibility. -/
theorem polynomial_divisibility (a b c : ℝ) :
  isDivisibleByXMinusOneCubed a b c ↔ a = 0 ∧ b = 2 ∧ c = -1 := by
  sorry

end polynomial_divisibility_l2893_289367


namespace partnership_profit_share_l2893_289386

/-- 
Given three partners A, B, and C in a partnership where:
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

This theorem proves that B's share of the profit is 1760.
-/
theorem partnership_profit_share 
  (investment_A investment_B investment_C : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2/3 * investment_C)
  (h3 : total_profit = 4400) :
  (investment_B / (investment_A + investment_B + investment_C)) * total_profit = 1760 := by
  sorry


end partnership_profit_share_l2893_289386


namespace probability_one_white_one_black_l2893_289315

/-- The probability of drawing one white ball and one black ball from a bag -/
theorem probability_one_white_one_black (white_balls black_balls : ℕ) :
  white_balls = 6 →
  black_balls = 5 →
  (white_balls.choose 1 * black_balls.choose 1 : ℚ) / (white_balls + black_balls).choose 2 = 6/11 :=
by sorry

end probability_one_white_one_black_l2893_289315


namespace tea_milk_problem_l2893_289373

/-- Represents the amount of liquid in a mug -/
structure Mug where
  tea : ℚ
  milk : ℚ

/-- Calculates the fraction of milk in a mug -/
def milkFraction (m : Mug) : ℚ :=
  m.milk / (m.tea + m.milk)

theorem tea_milk_problem : 
  let mug1_initial := Mug.mk 5 0
  let mug2_initial := Mug.mk 0 3
  let mug1_after_first_transfer := Mug.mk (mug1_initial.tea - 2) 0
  let mug2_after_first_transfer := Mug.mk 2 3
  let tea_fraction_in_mug2 := mug2_after_first_transfer.tea / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let milk_fraction_in_mug2 := mug2_after_first_transfer.milk / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let tea_returned := 3 * tea_fraction_in_mug2
  let milk_returned := 3 * milk_fraction_in_mug2
  let mug1_final := Mug.mk (mug1_after_first_transfer.tea + tea_returned) milk_returned
  milkFraction mug1_final = 3/10 := by
sorry

end tea_milk_problem_l2893_289373


namespace function_minimum_l2893_289356

/-- The function f(x) = x³ - 3x² + 4 attains its minimum value at x = 2 -/
theorem function_minimum (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 3*x^2 + 4) :
  ∃ x₀ : ℝ, x₀ = 2 ∧ ∀ x, f x₀ ≤ f x := by sorry

end function_minimum_l2893_289356


namespace sum_of_squares_extremes_l2893_289314

theorem sum_of_squares_extremes (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ b = 9 → a^2 + c^2 = 180 := by
  sorry

end sum_of_squares_extremes_l2893_289314


namespace cafeteria_pie_count_l2893_289342

/-- Given a cafeteria with apples and pie-making scenario, calculate the number of pies that can be made -/
theorem cafeteria_pie_count (total_apples handed_out apples_per_pie : ℕ) 
  (h1 : total_apples = 96)
  (h2 : handed_out = 42)
  (h3 : apples_per_pie = 6) :
  (total_apples - handed_out) / apples_per_pie = 9 :=
by
  sorry

#check cafeteria_pie_count

end cafeteria_pie_count_l2893_289342


namespace sequence_perfect_square_property_l2893_289313

/-- Given two sequences of natural numbers satisfying a specific equation,
    prove that yₙ - 1 is a perfect square for all n. -/
theorem sequence_perfect_square_property
  (x y : ℕ → ℕ)
  (h : ∀ n : ℕ, (x n : ℝ) + Real.sqrt 2 * (y n : ℝ) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) :
  ∀ n : ℕ, ∃ k : ℕ, y n - 1 = k ^ 2 := by
  sorry

end sequence_perfect_square_property_l2893_289313


namespace consecutive_odd_integers_sum_l2893_289326

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (∃ y : ℤ, y = x + 2 ∧ x % 2 = 1 ∧ y % 2 = 1 ∧ y = 3 * x) → 
  x + (x + 2) = 4 :=
by sorry

end consecutive_odd_integers_sum_l2893_289326


namespace product_and_squared_sum_l2893_289307

theorem product_and_squared_sum (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  x * y = 875 ∧ (x + y)^2 = 3600 := by
  sorry

end product_and_squared_sum_l2893_289307


namespace no_three_distinct_reals_l2893_289363

theorem no_three_distinct_reals : ¬∃ (a b c p : ℝ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b * c = p ∧
  b + c * a = p ∧
  c + a * b = p := by
  sorry

end no_three_distinct_reals_l2893_289363


namespace moe_eating_time_l2893_289368

/-- Given that a lizard named Moe eats 40 pieces of cuttlebone in 10 seconds, 
    this theorem proves that it takes 200 seconds for Moe to eat 800 pieces. -/
theorem moe_eating_time : ∀ (rate : ℝ) (pieces : ℕ) (time : ℝ),
  rate = 40 / 10 →
  pieces = 800 →
  time = pieces / rate →
  time = 200 := by sorry

end moe_eating_time_l2893_289368


namespace inverse_of_49_mod_89_l2893_289396

theorem inverse_of_49_mod_89 (h : (7⁻¹ : ZMod 89) = 55) : (49⁻¹ : ZMod 89) = 1 := by
  sorry

end inverse_of_49_mod_89_l2893_289396


namespace cube_function_property_l2893_289344

theorem cube_function_property (a : ℝ) : 
  (fun x : ℝ ↦ x^3 + 1) a = 11 → (fun x : ℝ ↦ x^3 + 1) (-a) = -9 := by
sorry

end cube_function_property_l2893_289344


namespace inequality_solution_l2893_289364

theorem inequality_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9) > 0 ↔ x < -3 ∨ (2 < x ∧ x < 3) ∨ x > 4 :=
by sorry

end inequality_solution_l2893_289364


namespace min_value_3x_minus_2y_l2893_289380

theorem min_value_3x_minus_2y (x y : ℝ) (h : 4 * (x^2 + y^2 + x*y) = 2 * (x + y)) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (a b : ℝ), 4 * (a^2 + b^2 + a*b) = 2 * (a + b) → 3*a - 2*b ≥ m := by
  sorry

end min_value_3x_minus_2y_l2893_289380


namespace water_level_accurate_l2893_289306

/-- Represents the water level function for a reservoir -/
def waterLevel (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  waterLevel x = 6 + 0.3 * x ∧ 
  waterLevel 0 = 6 ∧
  ∀ t₁ t₂, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 5 → (waterLevel t₂ - waterLevel t₁) / (t₂ - t₁) = 0.3 := by
  sorry

end water_level_accurate_l2893_289306


namespace min_value_x_plus_2y_min_value_equals_l2893_289335

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/w = 2 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∃ z w : ℝ, z > 0 ∧ w > 0 ∧ 1/z + 1/w = 2 ∧ z + 2*w = (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end min_value_x_plus_2y_min_value_equals_l2893_289335


namespace quadratic_equation_real_roots_zero_product_property_l2893_289370

-- Proposition 1
theorem quadratic_equation_real_roots (k : ℝ) (h : k > 0) :
  ∃ x : ℝ, x^2 + 2*x - k = 0 :=
sorry

-- Proposition 4
theorem zero_product_property (x y : ℝ) :
  x * y = 0 → x = 0 ∨ y = 0 :=
sorry

end quadratic_equation_real_roots_zero_product_property_l2893_289370


namespace algebraic_expression_value_l2893_289360

theorem algebraic_expression_value (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / ((a-b)*(a-c)*(a-d))) + (b^5 / ((b-a)*(b-c)*(b-d))) + 
  (c^5 / ((c-a)*(c-b)*(c-d))) + (d^5 / ((d-a)*(d-b)*(d-c))) = -9 :=
sorry

end algebraic_expression_value_l2893_289360


namespace math_representative_selection_l2893_289353

theorem math_representative_selection (male_students female_students : ℕ) 
  (h1 : male_students = 26) 
  (h2 : female_students = 24) : 
  (male_students + female_students : ℕ) = 50 := by
  sorry

end math_representative_selection_l2893_289353


namespace average_spring_headcount_equals_10700_l2893_289340

def spring_headcount_02_03 : ℕ := 10900
def spring_headcount_03_04 : ℕ := 10500
def spring_headcount_04_05 : ℕ := 10700

def average_spring_headcount : ℚ :=
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3

theorem average_spring_headcount_equals_10700 :
  round average_spring_headcount = 10700 := by
  sorry

end average_spring_headcount_equals_10700_l2893_289340


namespace pond_filling_time_l2893_289388

/-- Proves the time required to fill a pond under drought conditions -/
theorem pond_filling_time (pond_capacity : ℝ) (normal_rate : ℝ) (drought_factor : ℝ) : 
  pond_capacity = 200 →
  normal_rate = 6 →
  drought_factor = 2/3 →
  (pond_capacity / (normal_rate * drought_factor) = 50) :=
by
  sorry

end pond_filling_time_l2893_289388


namespace function_inequality_implies_positive_a_l2893_289348

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end function_inequality_implies_positive_a_l2893_289348


namespace square_triangle_area_equality_l2893_289392

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 8 := by
  sorry

end square_triangle_area_equality_l2893_289392


namespace quadratic_equation_in_y_l2893_289338

theorem quadratic_equation_in_y : 
  ∀ x y : ℝ, 
  (3 * x^2 - 4 * x + 7 * y + 3 = 0) → 
  (3 * x - 5 * y + 6 = 0) → 
  (25 * y^2 - 39 * y + 69 = 0) :=
by sorry

end quadratic_equation_in_y_l2893_289338


namespace twentieth_sample_number_l2893_289371

/-- Calculates the nth number in a systematic sample. -/
def systematicSample (totalItems : Nat) (sampleSize : Nat) (firstNumber : Nat) (n : Nat) : Nat :=
  let k := totalItems / sampleSize
  firstNumber + (n - 1) * k

/-- Proves that the 20th number in the systematic sample is 395. -/
theorem twentieth_sample_number 
  (totalItems : Nat) 
  (sampleSize : Nat) 
  (firstNumber : Nat) 
  (h1 : totalItems = 1000) 
  (h2 : sampleSize = 50) 
  (h3 : firstNumber = 15) :
  systematicSample totalItems sampleSize firstNumber 20 = 395 := by
  sorry

#eval systematicSample 1000 50 15 20

end twentieth_sample_number_l2893_289371


namespace negative_slope_probability_l2893_289345

def LineSet : Set ℤ := {-3, -1, 0, 2, 7}

def ValidPair (a b : ℤ) : Prop :=
  a ∈ LineSet ∧ b ∈ LineSet ∧ a ≠ b

def NegativeSlope (a b : ℤ) : Prop :=
  ValidPair a b ∧ (a / b < 0)

def TotalPairs : ℕ := 20

def NegativeSlopePairs : ℕ := 4

theorem negative_slope_probability :
  (NegativeSlopePairs : ℚ) / TotalPairs = 1 / 5 :=
sorry

end negative_slope_probability_l2893_289345


namespace highest_power_of_two_dividing_P_l2893_289382

def P : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (3^(i+1) + 1)) 1

theorem highest_power_of_two_dividing_P :
  ∃ (k : ℕ), (2^3030 ∣ P 2020) ∧ ¬(2^(3030 + 1) ∣ P 2020) := by
  sorry

end highest_power_of_two_dividing_P_l2893_289382


namespace composite_odd_number_characterization_l2893_289359

theorem composite_odd_number_characterization (c : ℕ) (h_odd : Odd c) :
  (∃ (a : ℕ), a ≤ c / 3 - 1 ∧ ∃ (k : ℕ), (2 * a - 1)^2 + 8 * c = (2 * k + 1)^2) ↔
  (∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ c = p * q) :=
sorry

end composite_odd_number_characterization_l2893_289359


namespace estimate_black_balls_l2893_289352

theorem estimate_black_balls (total_balls : Nat) (total_draws : Nat) (black_draws : Nat) :
  total_balls = 15 →
  total_draws = 100 →
  black_draws = 60 →
  (black_draws : Real) / total_draws * total_balls = 9 := by
  sorry

end estimate_black_balls_l2893_289352


namespace train_car_count_l2893_289310

theorem train_car_count (total_cars : ℕ) (passenger_cars : ℕ) (cargo_cars : ℕ) : 
  total_cars = 71 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + 2 →
  passenger_cars = 44 := by
sorry

end train_car_count_l2893_289310


namespace max_D_value_l2893_289316

/-- Represents a building block with three binary attributes -/
structure Block :=
  (shape : Bool)
  (color : Bool)
  (city : Bool)

/-- The set of all possible blocks -/
def allBlocks : Finset Block := sorry

/-- The number of blocks -/
def numBlocks : Nat := Finset.card allBlocks

/-- Checks if two blocks share exactly two attributes -/
def sharesTwoAttributes (b1 b2 : Block) : Bool := sorry

/-- The number of ways to select n blocks such that each subsequent block
    shares exactly two attributes with the previously selected block -/
def D (n : Nat) : Nat := sorry

/-- The maximum value of D(n) for 2 ≤ n ≤ 8 -/
def maxD : Nat := sorry

theorem max_D_value :
  numBlocks = 8 →
  (∀ (b1 b2 : Block), b1 ∈ allBlocks ∧ b2 ∈ allBlocks → b1 ≠ b2) →
  maxD = 240 := by sorry

end max_D_value_l2893_289316


namespace intersection_count_l2893_289331

/-- The set A as defined in the problem -/
def A : Set (ℤ × ℤ) := {p | ∃ m : ℤ, m > 0 ∧ p.1 = m ∧ p.2 = -3*m + 2}

/-- The set B as defined in the problem -/
def B (a : ℤ) : Set (ℤ × ℤ) := {p | ∃ n : ℤ, n > 0 ∧ p.1 = n ∧ p.2 = a*(a^2 - n + 1)}

/-- The theorem stating that there are exactly 10 integer values of a for which A ∩ B ≠ ∅ -/
theorem intersection_count :
  ∃! (s : Finset ℤ), s.card = 10 ∧ ∀ a : ℤ, a ∈ s ↔ (A ∩ B a).Nonempty :=
by sorry

end intersection_count_l2893_289331


namespace ratio_c_over_a_l2893_289389

theorem ratio_c_over_a (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_arithmetic_seq : 2 * Real.log (a * c) = Real.log (a * b) + Real.log (b * c))
  (h_relation : 4 * (a + c) = 17 * b) :
  c / a = 16 ∨ c / a = 1 / 16 := by
sorry

end ratio_c_over_a_l2893_289389


namespace election_vote_ratio_l2893_289398

theorem election_vote_ratio :
  let joey_votes : ℕ := 8
  let barry_votes : ℕ := 2 * (joey_votes + 3)
  let marcy_votes : ℕ := 66
  (marcy_votes : ℚ) / barry_votes = 3 / 1 :=
by sorry

end election_vote_ratio_l2893_289398


namespace expression_equivalence_l2893_289351

theorem expression_equivalence (a b c : ℝ) : a - (2*b - 3*c) = a + (-2*b + 3*c) := by
  sorry

end expression_equivalence_l2893_289351


namespace loan_principal_calculation_l2893_289336

theorem loan_principal_calculation (interest_rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ) :
  interest_rate = 12 →
  time = 3 →
  interest = 6480 →
  interest = principal * interest_rate * time / 100 →
  principal = 18000 := by
sorry

end loan_principal_calculation_l2893_289336


namespace min_value_of_4x2_plus_y2_l2893_289330

theorem min_value_of_4x2_plus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 6) :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 6 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2 ∧ 4 * x^2 + y^2 = 18 :=
by sorry

end min_value_of_4x2_plus_y2_l2893_289330


namespace cherry_soda_count_l2893_289309

theorem cherry_soda_count (total : ℕ) (cherry : ℕ) (orange : ℕ) 
  (h1 : total = 24)
  (h2 : orange = 2 * cherry)
  (h3 : total = cherry + orange) : cherry = 8 := by
  sorry

end cherry_soda_count_l2893_289309


namespace negation_of_implication_l2893_289319

theorem negation_of_implication (x y : ℝ) : 
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 ∧ x * y > 1) :=
sorry

end negation_of_implication_l2893_289319


namespace complex_imaginary_part_l2893_289354

theorem complex_imaginary_part (z : ℂ) (h : z * (3 - 4*I) = 1) : z.im = 4/25 := by
  sorry

end complex_imaginary_part_l2893_289354


namespace good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2893_289300

def is_good_number (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*x*y + 2*y^2

theorem good_number_characterization (n : ℤ) :
  is_good_number n ↔ ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

theorem twenty_nine_is_good : is_good_number 29 :=
sorry

def good_numbers_up_to_nine : List ℤ := [1, 2, 4, 5, 8, 9]

theorem good_numbers_up_to_nine_correct :
  ∀ n : ℤ, n ∈ good_numbers_up_to_nine ↔ (1 ≤ n ∧ n ≤ 9 ∧ is_good_number n) :=
sorry

theorem product_of_good_numbers_is_good (m n : ℤ) :
  is_good_number m → is_good_number n → is_good_number (m * n) :=
sorry

end good_number_characterization_twenty_nine_is_good_good_numbers_up_to_nine_correct_product_of_good_numbers_is_good_l2893_289300


namespace sufficient_not_necessary_condition_l2893_289391

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬(∀ a : ℝ, (a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end sufficient_not_necessary_condition_l2893_289391


namespace stone_skipping_ratio_l2893_289324

theorem stone_skipping_ratio (x y : ℕ) : 
  x > 0 → -- First throw has at least one skip
  x + 2 > 0 → -- Second throw has at least one skip
  y > 0 → -- Third throw has at least one skip
  y - 3 > 0 → -- Fourth throw has at least one skip
  y - 2 = 8 → -- Fifth throw skips 8 times
  x + (x + 2) + y + (y - 3) + (y - 2) = 33 → -- Total skips is 33
  y = x + 2 -- Ratio of third to second throw is 1:1
  := by sorry

end stone_skipping_ratio_l2893_289324


namespace daves_coins_l2893_289394

theorem daves_coins (n : ℕ) : n > 0 ∧ 
  n % 7 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 3 = 1 ∧ 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 5 = 3 → m % 3 = 1 → n ≤ m) → 
  n = 58 := by
sorry

end daves_coins_l2893_289394


namespace solve_equation1_solve_equation2_l2893_289374

-- Define the equations
def equation1 (x : ℚ) : Prop := 5 * x - 2 * (x - 1) = 3
def equation2 (x : ℚ) : Prop := (x + 3) / 2 - 1 = (2 * x - 1) / 3

-- Theorem statements
theorem solve_equation1 : ∃ x : ℚ, equation1 x ∧ x = 1/3 := by sorry

theorem solve_equation2 : ∃ x : ℚ, equation2 x ∧ x = 3 := by sorry

end solve_equation1_solve_equation2_l2893_289374


namespace polynomial_division_remainder_l2893_289383

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ,
  X^4 = (X^2 + 3*X + 2) * q + (-18*X - 16) := by sorry

end polynomial_division_remainder_l2893_289383


namespace inequality_proof_l2893_289339

theorem inequality_proof (a b c : ℝ) : 
  ((a^2 + b^2 + a*c)^2 + (a^2 + b^2 + b*c)^2) / (a^2 + b^2) ≥ (a + b + c)^2 := by
  sorry

end inequality_proof_l2893_289339


namespace inequality_proof_l2893_289320

theorem inequality_proof (a b : ℝ) (h1 : -1 < b) (h2 : b < 0) (h3 : a < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_proof_l2893_289320


namespace last_two_digits_product_l2893_289305

/-- Given an integer n, returns the tens digit -/
def tensDigit (n : ℤ) : ℤ := (n / 10) % 10

/-- Given an integer n, returns the units digit -/
def unitsDigit (n : ℤ) : ℤ := n % 10

/-- Theorem: For any integer divisible by 5 with the sum of its last two digits being 12,
    the product of its last two digits is 35 -/
theorem last_two_digits_product (n : ℤ) : 
  n % 5 = 0 → 
  tensDigit n + unitsDigit n = 12 → 
  tensDigit n * unitsDigit n = 35 := by
  sorry

end last_two_digits_product_l2893_289305


namespace sum_of_numbers_l2893_289397

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5)
  (ga : a > 0.1) (gb : b > 0.1) (gc : c > 0.1) : a + b + c = 1.8 := by
  sorry

end sum_of_numbers_l2893_289397


namespace second_discount_percentage_l2893_289328

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 350 →
  first_discount = 20 →
  final_price = 266 →
  (original_price * (1 - first_discount / 100) * (1 - (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100))) = final_price) →
  (original_price * (1 - first_discount / 100) - final_price) / (original_price * (1 - first_discount / 100)) * 100 = 5 :=
by sorry

end second_discount_percentage_l2893_289328


namespace arithmetic_sequence_length_l2893_289355

/-- An arithmetic sequence starting with 2 and ending with 2006 has 502 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a : ℕ → ℕ), 
    a 0 = 2 → 
    (∃ n : ℕ, a n = 2006) → 
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) → 
    (∃ n : ℕ, a n = 2006 ∧ n + 1 = 502) := by
  sorry

end arithmetic_sequence_length_l2893_289355


namespace triangle_perimeter_upper_bound_l2893_289303

theorem triangle_perimeter_upper_bound (a b c : ℝ) : 
  a = 8 → b = 15 → a + b > c → a + c > b → b + c > a → 
  ∃ n : ℕ, n = 46 ∧ ∀ m : ℕ, (m : ℝ) > a + b + c → m ≥ n :=
sorry

end triangle_perimeter_upper_bound_l2893_289303


namespace larger_number_proof_l2893_289375

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 := by
  sorry

end larger_number_proof_l2893_289375
