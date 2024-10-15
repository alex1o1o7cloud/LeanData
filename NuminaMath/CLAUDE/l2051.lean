import Mathlib

namespace NUMINAMATH_CALUDE_meaningful_expression_range_l2051_205110

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / ((x - 3)^2)) ↔ x ≥ -1 ∧ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l2051_205110


namespace NUMINAMATH_CALUDE_abs_diff_lt_abs_one_minus_prod_l2051_205118

theorem abs_diff_lt_abs_one_minus_prod {x y : ℝ} (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_lt_abs_one_minus_prod_l2051_205118


namespace NUMINAMATH_CALUDE_find_b_l2051_205119

-- Define the set A
def A (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 + b}

-- Theorem statement
theorem find_b : ∃ b : ℝ, (1, 5) ∈ A b ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l2051_205119


namespace NUMINAMATH_CALUDE_james_change_l2051_205128

/-- Calculates the change received when buying candy -/
def calculate_change (num_packs : ℕ) (cost_per_pack : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_packs * cost_per_pack)

/-- Proves that James received $11 in change -/
theorem james_change :
  let num_packs : ℕ := 3
  let cost_per_pack : ℕ := 3
  let amount_paid : ℕ := 20
  calculate_change num_packs cost_per_pack amount_paid = 11 := by
  sorry

end NUMINAMATH_CALUDE_james_change_l2051_205128


namespace NUMINAMATH_CALUDE_tangent_line_sum_range_l2051_205187

theorem tangent_line_sum_range (m n : ℝ) :
  (∀ x y : ℝ, m * x + n * y - 2 = 0 → x^2 + y^2 ≠ 1) ∧
  (∃ x y : ℝ, m * x + n * y - 2 = 0 ∧ x^2 + y^2 = 1) →
  -2 * Real.sqrt 2 ≤ m + n ∧ m + n ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_range_l2051_205187


namespace NUMINAMATH_CALUDE_hydrogen_mass_percentage_l2051_205193

/-- Molecular weight of C3H6O in g/mol -/
def mw_C3H6O : ℝ := 58.09

/-- Molecular weight of NH3 in g/mol -/
def mw_NH3 : ℝ := 17.04

/-- Molecular weight of H2SO4 in g/mol -/
def mw_H2SO4 : ℝ := 98.09

/-- Mass of hydrogen in one mole of C3H6O in g -/
def mass_H_in_C3H6O : ℝ := 6.06

/-- Mass of hydrogen in one mole of NH3 in g -/
def mass_H_in_NH3 : ℝ := 3.03

/-- Mass of hydrogen in one mole of H2SO4 in g -/
def mass_H_in_H2SO4 : ℝ := 2.02

/-- Number of moles of C3H6O in the mixture -/
def moles_C3H6O : ℝ := 3

/-- Number of moles of NH3 in the mixture -/
def moles_NH3 : ℝ := 2

/-- Number of moles of H2SO4 in the mixture -/
def moles_H2SO4 : ℝ := 1

/-- Theorem stating that the mass percentage of hydrogen in the given mixture is approximately 8.57% -/
theorem hydrogen_mass_percentage :
  let total_mass_H := moles_C3H6O * mass_H_in_C3H6O + moles_NH3 * mass_H_in_NH3 + moles_H2SO4 * mass_H_in_H2SO4
  let total_mass_mixture := moles_C3H6O * mw_C3H6O + moles_NH3 * mw_NH3 + moles_H2SO4 * mw_H2SO4
  let mass_percentage_H := (total_mass_H / total_mass_mixture) * 100
  ∃ ε > 0, |mass_percentage_H - 8.57| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_hydrogen_mass_percentage_l2051_205193


namespace NUMINAMATH_CALUDE_james_total_points_l2051_205197

/-- Quiz Bowl Scoring System -/
structure QuizBowl where
  correct_points : ℕ := 2
  incorrect_penalty : ℕ := 1
  quick_answer_bonus : ℕ := 1
  rounds : ℕ := 5
  questions_per_round : ℕ := 5

/-- James' Performance -/
structure Performance where
  correct_answers : ℕ
  missed_questions : ℕ
  quick_answers : ℕ

/-- Calculate total points for a given performance in the quiz bowl -/
def calculate_points (qb : QuizBowl) (perf : Performance) : ℕ :=
  qb.correct_points * perf.correct_answers + qb.quick_answer_bonus * perf.quick_answers

/-- Theorem: James' total points in the quiz bowl -/
theorem james_total_points (qb : QuizBowl) (james : Performance) 
  (h1 : james.correct_answers = qb.rounds * qb.questions_per_round - james.missed_questions)
  (h2 : james.missed_questions = 1)
  (h3 : james.quick_answers = 4) :
  calculate_points qb james = 52 := by
  sorry

end NUMINAMATH_CALUDE_james_total_points_l2051_205197


namespace NUMINAMATH_CALUDE_hansel_raise_percentage_l2051_205198

/-- Proves that Hansel's raise percentage is 10% given the problem conditions --/
theorem hansel_raise_percentage : 
  ∀ (hansel_initial gretel_initial hansel_final gretel_final gretel_raise hansel_raise : ℝ),
  hansel_initial = 30000 →
  gretel_initial = 30000 →
  gretel_raise = 0.15 →
  gretel_final = gretel_initial * (1 + gretel_raise) →
  hansel_final = gretel_final - 1500 →
  hansel_raise = (hansel_final - hansel_initial) / hansel_initial →
  hansel_raise = 0.1 := by
sorry


end NUMINAMATH_CALUDE_hansel_raise_percentage_l2051_205198


namespace NUMINAMATH_CALUDE_cantor_bernstein_l2051_205173

theorem cantor_bernstein {α β : Type*} (f : α → β) (g : β → α) 
  (hf : Function.Injective f) (hg : Function.Injective g) : 
  Nonempty (α ≃ β) :=
sorry

end NUMINAMATH_CALUDE_cantor_bernstein_l2051_205173


namespace NUMINAMATH_CALUDE_simplify_radicals_l2051_205184

theorem simplify_radicals : 
  Real.sqrt 18 * Real.sqrt 72 - Real.sqrt 32 = 36 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l2051_205184


namespace NUMINAMATH_CALUDE_cs_majors_consecutive_probability_l2051_205171

def total_people : ℕ := 12
def cs_majors : ℕ := 5
def chem_majors : ℕ := 4
def lit_majors : ℕ := 3

theorem cs_majors_consecutive_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let consecutive_arrangements := Nat.factorial (total_people - cs_majors) * Nat.factorial cs_majors
  (consecutive_arrangements : ℚ) / total_arrangements = 1 / 66 := by
  sorry

end NUMINAMATH_CALUDE_cs_majors_consecutive_probability_l2051_205171


namespace NUMINAMATH_CALUDE_division_problem_l2051_205115

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 190 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2051_205115


namespace NUMINAMATH_CALUDE_min_value_theorem_l2051_205138

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) :
  ∀ z : ℝ, z = (1 / (x + 1)) + (4 / y) → z ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2051_205138


namespace NUMINAMATH_CALUDE_solutions_based_on_discriminant_l2051_205160

/-- Represents the system of equations -/
def SystemOfEquations (a b c : ℝ) (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (a ≠ 0) ∧ 
  (∀ i ∈ Finset.range n, a * (x i)^2 + b * (x i) + c = x ((i + 1) % n))

/-- Theorem stating the number of solutions based on the discriminant -/
theorem solutions_based_on_discriminant (a b c : ℝ) (n : ℕ) :
  (a ≠ 0) ∧ (n > 0) →
  (((b - 1)^2 - 4*a*c < 0 → ¬∃ x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c = 0 → ∃! x, SystemOfEquations a b c n x) ∧
   ((b - 1)^2 - 4*a*c > 0 → ∃ x y, x ≠ y ∧ SystemOfEquations a b c n x ∧ SystemOfEquations a b c n y)) :=
by sorry

end NUMINAMATH_CALUDE_solutions_based_on_discriminant_l2051_205160


namespace NUMINAMATH_CALUDE_mushroom_collectors_l2051_205179

theorem mushroom_collectors (n : ℕ) : 
  (n^2 + 9*n - 2) % (n + 11) = 0 → n < 11 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collectors_l2051_205179


namespace NUMINAMATH_CALUDE_average_price_per_book_l2051_205182

def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1150
def books_shop2 : ℕ := 50
def price_shop2 : ℕ := 920

theorem average_price_per_book :
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_price_per_book_l2051_205182


namespace NUMINAMATH_CALUDE_final_crayons_count_l2051_205143

def initial_crayons : ℝ := 7.5
def mary_took : ℝ := 3.2
def mark_took : ℝ := 0.5
def jane_took : ℝ := 1.3
def mary_returned : ℝ := 0.7
def sarah_added : ℝ := 3.5
def tom_added : ℝ := 2.8
def alice_took : ℝ := 1.5

theorem final_crayons_count :
  initial_crayons - mary_took - mark_took - jane_took + mary_returned + sarah_added + tom_added - alice_took = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_crayons_count_l2051_205143


namespace NUMINAMATH_CALUDE_bret_dinner_coworkers_l2051_205180

theorem bret_dinner_coworkers :
  let main_meal_cost : ℚ := 12
  let appetizer_cost : ℚ := 6
  let num_appetizers : ℕ := 2
  let tip_percentage : ℚ := 0.2
  let rush_order_fee : ℚ := 5
  let total_spent : ℚ := 77

  let total_people (coworkers : ℕ) : ℕ := coworkers + 1
  let main_meals_cost (coworkers : ℕ) : ℚ := main_meal_cost * (total_people coworkers : ℚ)
  let appetizers_total : ℚ := appetizer_cost * (num_appetizers : ℚ)
  let subtotal (coworkers : ℕ) : ℚ := main_meals_cost coworkers + appetizers_total
  let tip (coworkers : ℕ) : ℚ := tip_percentage * subtotal coworkers
  let total_cost (coworkers : ℕ) : ℚ := subtotal coworkers + tip coworkers + rush_order_fee

  ∃ (coworkers : ℕ), total_cost coworkers = total_spent ∧ coworkers = 3 :=
by sorry

end NUMINAMATH_CALUDE_bret_dinner_coworkers_l2051_205180


namespace NUMINAMATH_CALUDE_scavenger_hunt_theorem_l2051_205165

/-- Represents the number of choices for each day of the scavenger hunt --/
def scavenger_hunt_choices : List Nat := [1, 2, 4, 3, 1]

/-- The total number of combinations for the scavenger hunt --/
def total_combinations : Nat := scavenger_hunt_choices.prod

theorem scavenger_hunt_theorem :
  total_combinations = 24 := by
  sorry

end NUMINAMATH_CALUDE_scavenger_hunt_theorem_l2051_205165


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l2051_205196

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l2051_205196


namespace NUMINAMATH_CALUDE_romanov_savings_l2051_205116

/-- Represents the electricity pricing and consumption data for the Romanov family --/
structure ElectricityData where
  multi_tariff_meter_cost : ℝ
  installation_cost : ℝ
  monthly_consumption : ℝ
  night_consumption : ℝ
  day_rate : ℝ
  night_rate : ℝ
  standard_rate : ℝ
  years : ℕ

/-- Calculates the savings from using a multi-tariff meter over the given period --/
def calculate_savings (data : ElectricityData) : ℝ :=
  let standard_cost := data.standard_rate * data.monthly_consumption * 12 * data.years
  let day_consumption := data.monthly_consumption - data.night_consumption
  let multi_tariff_cost := (data.day_rate * day_consumption + data.night_rate * data.night_consumption) * 12 * data.years
  let total_multi_tariff_cost := multi_tariff_cost + data.multi_tariff_meter_cost + data.installation_cost
  standard_cost - total_multi_tariff_cost

/-- Theorem stating the savings for the Romanov family --/
theorem romanov_savings :
  let data : ElectricityData := {
    multi_tariff_meter_cost := 3500,
    installation_cost := 1100,
    monthly_consumption := 300,
    night_consumption := 230,
    day_rate := 5.2,
    night_rate := 3.4,
    standard_rate := 4.6,
    years := 3
  }
  calculate_savings data = 3824 := by
  sorry

end NUMINAMATH_CALUDE_romanov_savings_l2051_205116


namespace NUMINAMATH_CALUDE_frank_pizza_slices_l2051_205177

/-- Proves that Frank ate 3 slices of Hawaiian pizza given the conditions of the problem -/
theorem frank_pizza_slices (total_slices dean_slices sammy_slices leftover_slices : ℕ) :
  total_slices = 2 * 12 →
  dean_slices = 12 / 2 →
  sammy_slices = 12 / 3 →
  leftover_slices = 11 →
  total_slices - leftover_slices - dean_slices - sammy_slices = 3 := by
  sorry

#check frank_pizza_slices

end NUMINAMATH_CALUDE_frank_pizza_slices_l2051_205177


namespace NUMINAMATH_CALUDE_smallRectLengthIsFourTimesWidth_l2051_205186

/-- Represents the arrangement of squares and a rectangle -/
structure SquareArrangement where
  s : ℝ
  largeTotalWidth : ℝ
  largeLength : ℝ
  smallRectWidth : ℝ
  smallRectLength : ℝ

/-- The conditions of the problem -/
def validArrangement (a : SquareArrangement) : Prop :=
  a.largeTotalWidth = 3 * a.s ∧
  a.largeLength = 2 * a.largeTotalWidth ∧
  a.smallRectWidth = a.s

/-- The theorem to prove -/
theorem smallRectLengthIsFourTimesWidth (a : SquareArrangement) 
  (h : validArrangement a) : a.smallRectLength = 4 * a.smallRectWidth :=
sorry

end NUMINAMATH_CALUDE_smallRectLengthIsFourTimesWidth_l2051_205186


namespace NUMINAMATH_CALUDE_dollar_op_five_negative_two_l2051_205101

-- Define the $ operation
def dollar_op (c d : Int) : Int := c * (d + 1) + c * d

-- Theorem statement
theorem dollar_op_five_negative_two :
  dollar_op 5 (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_five_negative_two_l2051_205101


namespace NUMINAMATH_CALUDE_min_three_colors_proof_l2051_205195

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  blue : ℕ
  white : ℕ

/-- The total number of balls in the box -/
def total_balls : ℕ := 111

/-- The number of balls that, when drawn, ensures getting all four colors -/
def all_colors_draw : ℕ := 100

/-- Predicate to check if a BallCounts configuration is valid -/
def valid_configuration (counts : BallCounts) : Prop :=
  counts.red + counts.green + counts.blue + counts.white = total_balls ∧
  ∀ (n : ℕ), n ≥ all_colors_draw →
    n - counts.red < all_colors_draw ∧
    n - counts.green < all_colors_draw ∧
    n - counts.blue < all_colors_draw ∧
    n - counts.white < all_colors_draw

/-- The smallest number of balls to draw to ensure at least three colors -/
def min_three_colors_draw : ℕ := 88

theorem min_three_colors_proof :
  ∀ (counts : BallCounts),
    valid_configuration counts →
    (∀ (n : ℕ), n ≥ min_three_colors_draw →
      ∃ (colors : Finset (Fin 4)),
        colors.card ≥ 3 ∧
        (∀ (i : Fin 4),
          i ∈ colors ↔
            (i = 0 ∧ n > total_balls - counts.red) ∨
            (i = 1 ∧ n > total_balls - counts.green) ∨
            (i = 2 ∧ n > total_balls - counts.blue) ∨
            (i = 3 ∧ n > total_balls - counts.white))) ∧
    (∀ (m : ℕ), m < min_three_colors_draw →
      ∃ (counts' : BallCounts),
        valid_configuration counts' ∧
        ∃ (colors : Finset (Fin 4)),
          colors.card < 3 ∧
          (∀ (i : Fin 4),
            i ∈ colors ↔
              (i = 0 ∧ m > total_balls - counts'.red) ∨
              (i = 1 ∧ m > total_balls - counts'.green) ∨
              (i = 2 ∧ m > total_balls - counts'.blue) ∨
              (i = 3 ∧ m > total_balls - counts'.white))) :=
by sorry

end NUMINAMATH_CALUDE_min_three_colors_proof_l2051_205195


namespace NUMINAMATH_CALUDE_toy_ratio_after_removal_l2051_205199

/-- Proves that given 134 total toys, with 90 initially red, after removing 2 red toys, 
    the ratio of red to white toys is 2:1. -/
theorem toy_ratio_after_removal (total : ℕ) (initial_red : ℕ) (removed : ℕ) : 
  total = 134 → initial_red = 90 → removed = 2 →
  (initial_red - removed) / (total - initial_red) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_toy_ratio_after_removal_l2051_205199


namespace NUMINAMATH_CALUDE_piravena_flight_cost_l2051_205149

/-- Represents a right-angled triangle with sides DE and DF -/
structure RightTriangle where
  de : ℝ
  df : ℝ

/-- Calculates the cost of flying between two points -/
def flyCost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

theorem piravena_flight_cost (triangle : RightTriangle) 
  (h1 : triangle.de = 3750)
  (h2 : triangle.df = 3500) : 
  flyCost triangle.de = 570 := by
  sorry

#eval flyCost 3750

end NUMINAMATH_CALUDE_piravena_flight_cost_l2051_205149


namespace NUMINAMATH_CALUDE_triangle_inequality_l2051_205123

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a^2 - b^2| / c + |b^2 - c^2| / a ≥ |c^2 - a^2| / b := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2051_205123


namespace NUMINAMATH_CALUDE_max_value_fraction_l2051_205152

theorem max_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -1 ∧ 1 ≤ y' ∧ y' ≤ 3 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2051_205152


namespace NUMINAMATH_CALUDE_function_positivity_condition_l2051_205100

theorem function_positivity_condition (m : ℝ) : 
  (∀ x : ℝ, max (2*m*x^2 - 2*(4-m)*x + 1) (m*x) > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end NUMINAMATH_CALUDE_function_positivity_condition_l2051_205100


namespace NUMINAMATH_CALUDE_not_third_PSU_l2051_205170

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the ordering relation for runners
def beats (a b : Runner) : Prop := sorry

-- Define the conditions
axiom P_beats_Q : beats Runner.P Runner.Q
axiom Q_beats_R : beats Runner.Q Runner.R
axiom T_beats_S : beats Runner.T Runner.S
axiom T_beats_U : beats Runner.T Runner.U
axiom U_after_P_before_Q : beats Runner.P Runner.U ∧ beats Runner.U Runner.Q

-- Define what it means to finish third
def finishes_third (r : Runner) : Prop := sorry

-- Theorem statement
theorem not_third_PSU : 
  ¬(finishes_third Runner.P) ∧ 
  ¬(finishes_third Runner.S) ∧ 
  ¬(finishes_third Runner.U) := by sorry

end NUMINAMATH_CALUDE_not_third_PSU_l2051_205170


namespace NUMINAMATH_CALUDE_line_slope_proof_l2051_205127

/-- Given a line passing through points P(-2, m) and Q(m, 4) with slope 1, prove that m = 1 -/
theorem line_slope_proof (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l2051_205127


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l2051_205167

/-- Given a triangle ABC with vertices A(2,8), B(2,2), C(10,2), 
    D is the midpoint of AB, E is the midpoint of BC, 
    and F is the intersection point of AE and CD. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (2, 8) → B = (2, 2) → C = (10, 2) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 13 := by
sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_l2051_205167


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2051_205134

theorem quadratic_root_problem (k : ℝ) :
  (∃ x : ℝ, x^2 + (k - 5) * x + (4 - k) = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 + (k - 5) * y + (4 - k) = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2051_205134


namespace NUMINAMATH_CALUDE_penny_shark_species_l2051_205102

/-- Given the number of species Penny identified in an aquarium, prove the number of shark species. -/
theorem penny_shark_species (total : ℕ) (eels : ℕ) (whales : ℕ) (sharks : ℕ)
  (h1 : total = 55)
  (h2 : eels = 15)
  (h3 : whales = 5)
  (h4 : total = sharks + eels + whales) :
  sharks = 35 := by
  sorry

end NUMINAMATH_CALUDE_penny_shark_species_l2051_205102


namespace NUMINAMATH_CALUDE_euro_op_calculation_l2051_205112

-- Define the € operation
def euro_op (x y z : ℕ) : ℕ := 3 * x * y * z

-- State the theorem
theorem euro_op_calculation : 
  euro_op 3 (euro_op 4 5 6) 1 = 3240 := by
  sorry

end NUMINAMATH_CALUDE_euro_op_calculation_l2051_205112


namespace NUMINAMATH_CALUDE_certain_number_equation_l2051_205151

theorem certain_number_equation (x : ℝ) : 112 * x^4 = 70000 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2051_205151


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2051_205169

/-- Given a geometric sequence {a_n} where a_2 = 8 and a_5 = 64, the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2051_205169


namespace NUMINAMATH_CALUDE_M_equals_set_l2051_205113

def M : Set ℕ := {m | m > 0 ∧ ∃ k : ℤ, (10 : ℤ) = k * (m + 1)}

theorem M_equals_set : M = {1, 4, 9} := by sorry

end NUMINAMATH_CALUDE_M_equals_set_l2051_205113


namespace NUMINAMATH_CALUDE_snowdrift_depth_l2051_205109

theorem snowdrift_depth (initial_depth melted_depth third_day_depth fourth_day_depth final_depth : ℝ) :
  melted_depth = initial_depth / 2 →
  third_day_depth = melted_depth + 6 →
  fourth_day_depth = third_day_depth + 18 →
  final_depth = 34 →
  initial_depth = 20 :=
by sorry

end NUMINAMATH_CALUDE_snowdrift_depth_l2051_205109


namespace NUMINAMATH_CALUDE_power_difference_l2051_205192

theorem power_difference (a m n : ℝ) (hm : a^m = 6) (hn : a^n = 2) : a^(m-n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l2051_205192


namespace NUMINAMATH_CALUDE_alcohol_percentage_P_correct_l2051_205129

/-- The percentage of alcohol in vessel P that results in the given mixture ratio -/
def alcohol_percentage_P : ℝ := 62.5

/-- The percentage of alcohol in vessel Q -/
def alcohol_percentage_Q : ℝ := 87.5

/-- The volume of liquid taken from each vessel -/
def volume_per_vessel : ℝ := 4

/-- The ratio of alcohol to water in the resulting mixture -/
def mixture_ratio : ℝ := 3

/-- The total volume of the mixture -/
def total_volume : ℝ := 2 * volume_per_vessel

theorem alcohol_percentage_P_correct :
  (alcohol_percentage_P / 100 * volume_per_vessel +
   alcohol_percentage_Q / 100 * volume_per_vessel) / total_volume = mixture_ratio / (mixture_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_P_correct_l2051_205129


namespace NUMINAMATH_CALUDE_tank_capacity_l2051_205175

theorem tank_capacity : 
  ∀ (T : ℝ), 
  (T > 0) →
  ((9/10 : ℝ) * T - (3/4 : ℝ) * T = 5) →
  T = 100/3 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l2051_205175


namespace NUMINAMATH_CALUDE_max_sum_four_numbers_l2051_205163

theorem max_sum_four_numbers (a b c d : ℕ) :
  a < b → b < c → c < d →
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 806 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_four_numbers_l2051_205163


namespace NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l2051_205104

/-- Represents a rectangular grid of toothpicks with diagonal lines -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then 2 * grid.height else 0
  horizontal + vertical + diagonal

/-- The theorem stating that a 15x15 grid with diagonals has 510 toothpicks -/
theorem fifteen_by_fifteen_grid_toothpicks :
  total_toothpicks { height := 15, width := 15, has_diagonals := true } = 510 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l2051_205104


namespace NUMINAMATH_CALUDE_condition_analysis_l2051_205126

theorem condition_analysis (a : ℕ) : 
  let A : Set ℕ := {1, a}
  let B : Set ℕ := {1, 2, 3}
  (a = 3 → A ⊆ B) ∧ (∃ x ≠ 3, {1, x} ⊆ B) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l2051_205126


namespace NUMINAMATH_CALUDE_pencil_count_l2051_205140

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) : 
  num_pens = 640 →
  max_students = 40 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 40 * k :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2051_205140


namespace NUMINAMATH_CALUDE_triangle_area_l2051_205164

theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * k, 12 * k, 13 * k) → k > 0) 
  (h_perimeter : a + b + c = 60) : (a * b : ℝ) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2051_205164


namespace NUMINAMATH_CALUDE_roots_imply_composite_sum_of_squares_l2051_205122

theorem roots_imply_composite_sum_of_squares (a b : ℤ) :
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_composite_sum_of_squares_l2051_205122


namespace NUMINAMATH_CALUDE_positive_solution_x_l2051_205176

theorem positive_solution_x (x y z : ℝ) : 
  x * y = 8 - 2 * x - 3 * y →
  y * z = 8 - 4 * y - 2 * z →
  x * z = 40 - 5 * x - 3 * z →
  x > 0 →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l2051_205176


namespace NUMINAMATH_CALUDE_parallel_planes_l2051_205146

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (lies_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem parallel_planes
  (α β : Plane) (a b : Line) (A : Point) :
  lies_in a α →
  lies_in b α →
  intersect a b →
  ¬ line_parallel a β →
  ¬ line_parallel b β →
  parallel α β :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_l2051_205146


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2051_205156

/-- Proves that the weight of the replaced person is 65 kg given the conditions of the problem -/
theorem weight_of_replaced_person
  (n : ℕ)
  (original_average : ℝ)
  (new_average_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : n = 10)
  (h2 : new_average_increase = 7.2)
  (h3 : new_person_weight = 137)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = new_person_weight - n * new_average_increase ∧
    replaced_weight = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2051_205156


namespace NUMINAMATH_CALUDE_waiter_customer_count_l2051_205185

theorem waiter_customer_count :
  let num_tables : ℕ := 9
  let women_per_table : ℕ := 7
  let men_per_table : ℕ := 3
  let total_customers := num_tables * (women_per_table + men_per_table)
  total_customers = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l2051_205185


namespace NUMINAMATH_CALUDE_parabola_equation_l2051_205141

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in its general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Function to calculate the greatest common divisor of six integers -/
def gcd6 (a b c d e f : ℤ) : ℤ := sorry

/-- Theorem stating the equation of the parabola with given focus and directrix -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 2 ∧ focus.y = 4 ∧ 
  directrix.a = 4 ∧ directrix.b = 5 ∧ directrix.c = 20 → 
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -40 ∧ p.c = 16 ∧ p.d = 0 ∧ p.e = 0 ∧ p.f = 0 ∧ 
    p.a > 0 ∧ 
    gcd6 (abs p.a) (abs p.b) (abs p.c) (abs p.d) (abs p.e) (abs p.f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2051_205141


namespace NUMINAMATH_CALUDE_expression_simplification_l2051_205135

theorem expression_simplification (b : ℝ) :
  ((2 * b + 6) - 5 * b) / 2 = -3/2 * b + 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2051_205135


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2051_205121

theorem sum_first_six_primes_mod_seventh_prime : 
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2051_205121


namespace NUMINAMATH_CALUDE_simplify_expression_l2051_205194

theorem simplify_expression (z : ℝ) : (5 - 2 * z^2) - (4 * z^2 - 7) = 12 - 6 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2051_205194


namespace NUMINAMATH_CALUDE_meeting_arrangements_presidency_meeting_arrangements_l2051_205117

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club --/
structure Club :=
  (schools : Finset School)
  (total_members : Nat)

/-- Represents a meeting arrangement --/
structure MeetingArrangement :=
  (host : School)
  (host_representatives : Nat)
  (other_representatives : Nat)

/-- The number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Theorem: Number of possible meeting arrangements --/
theorem meeting_arrangements (club : Club) (arrangement : MeetingArrangement) : Nat :=
  let num_schools := Finset.card club.schools
  let host_choices := choose num_schools 1
  let host_rep_choices := choose arrangement.host.members arrangement.host_representatives
  let other_rep_choices := (choose arrangement.host.members arrangement.other_representatives) ^ (num_schools - 1)
  host_choices * host_rep_choices * other_rep_choices

/-- Main theorem: Prove the number of possible arrangements is 40,000 --/
theorem presidency_meeting_arrangements :
  ∀ (club : Club) (arrangement : MeetingArrangement),
    Finset.card club.schools = 4 →
    (∀ s ∈ club.schools, s.members = 5) →
    club.total_members = 20 →
    arrangement.host_representatives = 3 →
    arrangement.other_representatives = 2 →
    meeting_arrangements club arrangement = 40000 :=
sorry

end NUMINAMATH_CALUDE_meeting_arrangements_presidency_meeting_arrangements_l2051_205117


namespace NUMINAMATH_CALUDE_equation_solution_l2051_205145

theorem equation_solution :
  ∃ x : ℝ, x + 2*x + 12 = 500 - (3*x + 4*x) → x = 48.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2051_205145


namespace NUMINAMATH_CALUDE_probability_point_near_vertex_l2051_205108

/-- The probability of a randomly selected point from a square being within a certain distance from a vertex -/
theorem probability_point_near_vertex (side_length : ℝ) (distance : ℝ) : 
  side_length > 0 → distance > 0 → distance ≤ side_length →
  (π * distance^2) / (4 * side_length^2) = π / 16 ↔ side_length = 4 ∧ distance = 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_point_near_vertex_l2051_205108


namespace NUMINAMATH_CALUDE_summer_pizza_sales_l2051_205130

/-- Given information about pizza sales in different seasons, prove that summer sales are 2 million pizzas. -/
theorem summer_pizza_sales :
  let spring_percent : ℝ := 0.3
  let spring_sales : ℝ := 4.8
  let autumn_sales : ℝ := 7
  let winter_sales : ℝ := 2.2
  let total_sales : ℝ := spring_sales / spring_percent
  let summer_sales : ℝ := total_sales - spring_sales - autumn_sales - winter_sales
  summer_sales = 2 := by
  sorry


end NUMINAMATH_CALUDE_summer_pizza_sales_l2051_205130


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l2051_205142

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumSequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (7 * (n + 1)) * 3 + sumSequence n

theorem last_two_digits_of_sum :
  lastTwoDigits (sumSequence 15) = 20 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_l2051_205142


namespace NUMINAMATH_CALUDE_not_divisible_by_6_and_11_l2051_205168

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  (n - 1) - (n - 1) / a - (n - 1) / b + (n - 1) / (a * b)

theorem not_divisible_by_6_and_11 :
  count_not_divisible 1500 6 11 = 1136 := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_6_and_11_l2051_205168


namespace NUMINAMATH_CALUDE_jungkook_smallest_number_l2051_205157

-- Define the set of students
inductive Student : Type
| Yoongi : Student
| Jungkook : Student
| Yuna : Student
| Yoojung : Student
| Taehyung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → ℕ
| Student.Yoongi => 7
| Student.Jungkook => 6
| Student.Yuna => 9
| Student.Yoojung => 8
| Student.Taehyung => 10

-- Theorem: Jungkook has the smallest number
theorem jungkook_smallest_number :
  ∀ s : Student, studentNumber Student.Jungkook ≤ studentNumber s :=
by sorry

end NUMINAMATH_CALUDE_jungkook_smallest_number_l2051_205157


namespace NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l2051_205133

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3}

theorem complement_of_B_relative_to_A : A \ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_B_relative_to_A_l2051_205133


namespace NUMINAMATH_CALUDE_quadratic_points_ordering_l2051_205137

/-- Quadratic function f(x) = -(x-2)² + h -/
def f (x h : ℝ) : ℝ := -(x - 2)^2 + h

theorem quadratic_points_ordering (h : ℝ) :
  let y₁ := f (-1/2) h
  let y₂ := f 1 h
  let y₃ := f 2 h
  y₁ < y₂ ∧ y₂ < y₃ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_ordering_l2051_205137


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2051_205158

/-- A quadratic function satisfying the given condition -/
noncomputable def f : ℝ → ℝ :=
  fun x => -(1/2) * x^2 - 2*x

/-- Function g defined in terms of f -/
noncomputable def g : ℝ → ℝ :=
  fun x => x * Real.log x + f x

/-- The set of real numbers satisfying the inequality -/
def solution_set : Set ℝ :=
  {x | x ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 0 1}

theorem quadratic_function_theorem :
  (∀ x, f (x + 1) + f x = -x^2 - 5*x - 5/2) ∧
  (f = fun x => -(1/2) * x^2 - 2*x) ∧
  (∀ x, x > 0 → (g (x^2 + x) ≥ g 2 ↔ x ∈ solution_set)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2051_205158


namespace NUMINAMATH_CALUDE_movie_collection_difference_l2051_205144

theorem movie_collection_difference (shared_movies : ℕ) (andrew_total : ℕ) (john_unique : ℕ)
  (h1 : shared_movies = 12)
  (h2 : andrew_total = 23)
  (h3 : john_unique = 8) :
  andrew_total - shared_movies + john_unique = 19 := by
sorry

end NUMINAMATH_CALUDE_movie_collection_difference_l2051_205144


namespace NUMINAMATH_CALUDE_koala_bear_ratio_is_one_half_l2051_205114

/-- Represents the number of tickets spent on different items -/
structure TicketSpending where
  total : ℕ
  earbuds : ℕ
  glowBracelets : ℕ
  koalaBear : ℕ

/-- The ratio of tickets spent on the koala bear to the total number of tickets -/
def koalaBearRatio (ts : TicketSpending) : Rat :=
  ts.koalaBear / ts.total

theorem koala_bear_ratio_is_one_half (ts : TicketSpending) 
  (h_total : ts.total = 50)
  (h_earbuds : ts.earbuds = 10)
  (h_glow : ts.glowBracelets = 15)
  (h_koala : ts.koalaBear = ts.total - ts.earbuds - ts.glowBracelets) :
  koalaBearRatio ts = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_koala_bear_ratio_is_one_half_l2051_205114


namespace NUMINAMATH_CALUDE_rectangular_to_square_formation_l2051_205148

theorem rectangular_to_square_formation :
  ∃ n : ℕ,
    (∃ a : ℕ, 8 * n + 120 = a * a) ∧
    (∃ b : ℕ, 8 * n - 120 = b * b) →
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_square_formation_l2051_205148


namespace NUMINAMATH_CALUDE_circle_equation_range_l2051_205189

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + m*x - 2*y + 3 = 0

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 * Real.sqrt 2 ∨ m > 2 * Real.sqrt 2

-- Theorem statement
theorem circle_equation_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_circle_equation_range_l2051_205189


namespace NUMINAMATH_CALUDE_max_third_side_length_l2051_205132

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 15) :
  ∃ (c : ℝ), c ≤ 21 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧
  ∀ (d : ℝ), (d > 21 ∨ d ≤ 0 ∨ a + b ≤ d ∨ a + d ≤ b ∨ b + d ≤ a) →
  ¬(∃ (e : ℕ), e > 21 ∧ (e : ℝ) = d) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2051_205132


namespace NUMINAMATH_CALUDE_money_division_l2051_205139

/-- The problem of dividing money among three people -/
theorem money_division (total : ℚ) (c_share : ℚ) (b_ratio : ℚ) :
  total = 328 →
  c_share = 64 →
  b_ratio = 65 / 100 →
  ∃ (a_share : ℚ),
    a_share + b_ratio * a_share + c_share = total ∧
    (c_share * 100) / a_share = 40 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2051_205139


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2051_205183

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > 0 and eccentricity is 2,
    prove that a = √6/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  a = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2051_205183


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2051_205174

/-- Calculates the profit percentage given cost price, marked price, and discount percentage. -/
def profit_percentage (cost_price marked_price discount_percent : ℚ) : ℚ :=
  let discount := (discount_percent / 100) * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that for the given conditions, the profit percentage is 25%. -/
theorem profit_percentage_is_25_percent :
  profit_percentage 95 125 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l2051_205174


namespace NUMINAMATH_CALUDE_proportionality_statements_l2051_205166

-- Define the basic concepts
def is_direct_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * g x

def is_inverse_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * g x = k

def is_not_proportional (f g : ℝ → ℝ) : Prop :=
  ¬(is_direct_proportion f g) ∧ ¬(is_inverse_proportion f g)

-- Define the specific relationships
def brick_area (n : ℝ) : ℝ := sorry
def brick_count (n : ℝ) : ℝ := sorry

def walk_speed (t : ℝ) : ℝ := sorry
def walk_time (t : ℝ) : ℝ := sorry

def circle_area (r : ℝ) : ℝ := sorry
def circle_radius (r : ℝ) : ℝ := sorry

-- State the theorem
theorem proportionality_statements :
  (is_direct_proportion brick_area brick_count) ∧
  (is_inverse_proportion walk_speed walk_time) ∧
  (is_not_proportional circle_area circle_radius) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_statements_l2051_205166


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_leq_zero_l2051_205191

theorem negation_of_forall_positive_square_leq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_leq_zero_l2051_205191


namespace NUMINAMATH_CALUDE_combined_weight_of_boxes_l2051_205107

theorem combined_weight_of_boxes (box1 box2 box3 : ℕ) 
  (h1 : box1 = 2) 
  (h2 : box2 = 11) 
  (h3 : box3 = 5) : 
  box1 + box2 + box3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_boxes_l2051_205107


namespace NUMINAMATH_CALUDE_third_term_of_sequence_l2051_205154

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem third_term_of_sequence (a : ℤ) (d : ℤ) :
  arithmetic_sequence a d 20 = 18 →
  arithmetic_sequence a d 21 = 20 →
  arithmetic_sequence a d 3 = -16 :=
by
  sorry

end NUMINAMATH_CALUDE_third_term_of_sequence_l2051_205154


namespace NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_for_profit_l2051_205131

-- Define the given constants
def initial_cost : ℝ := 40
def initial_price : ℝ := 60
def march_sales : ℝ := 192
def may_sales : ℝ := 300
def sales_increase_per_reduction : ℝ := 20  -- 40 pieces per 2 yuan reduction

-- Define the target profit
def target_profit : ℝ := 6080

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate : ∃ (x : ℝ), 
  march_sales * (1 + x)^2 = may_sales ∧ x = 0.25 := by sorry

-- Part 2: Price reduction for target profit
theorem price_reduction_for_profit : ∃ (m : ℝ),
  (initial_price - m - initial_cost) * (may_sales + sales_increase_per_reduction * m) = target_profit ∧
  m = 4 := by sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_for_profit_l2051_205131


namespace NUMINAMATH_CALUDE_special_function_value_l2051_205111

/-- A function satisfying f(xy) = f(x)/y² for positive reals -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / (y ^ 2)

/-- Theorem stating that if f is a special function and f(40) = 50, then f(80) = 12.5 -/
theorem special_function_value
  (f : ℝ → ℝ)
  (h_special : special_function f)
  (h_f40 : f 40 = 50) :
  f 80 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_special_function_value_l2051_205111


namespace NUMINAMATH_CALUDE_orange_marbles_count_l2051_205172

/-- The number of orange marbles in a jar, given the total number of marbles,
    the number of red marbles, and that half of the marbles are blue. -/
def orangeMarbles (total : ℕ) (red : ℕ) (halfAreBlue : Bool) : ℕ :=
  total - (total / 2 + red)

/-- Theorem stating that there are 6 orange marbles in a jar with 24 total marbles,
    6 red marbles, and half of the marbles being blue. -/
theorem orange_marbles_count :
  orangeMarbles 24 6 true = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l2051_205172


namespace NUMINAMATH_CALUDE_slower_speed_percentage_l2051_205120

theorem slower_speed_percentage (D : ℝ) (S : ℝ) (S_slow : ℝ) 
    (h1 : D = S * 16)
    (h2 : D = S_slow * 40) :
  S_slow / S = 0.4 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_percentage_l2051_205120


namespace NUMINAMATH_CALUDE_captain_age_is_your_age_l2051_205188

/-- Represents the age of a person in years -/
def Age : Type := ℕ

/-- Represents a person -/
structure Person where
  age : Age

/-- Represents the captain of the steamboat -/
def Captain : Person := sorry

/-- Represents you -/
def You : Person := sorry

/-- The theorem states that the captain's age is equal to your age -/
theorem captain_age_is_your_age : Captain.age = You.age := by sorry

end NUMINAMATH_CALUDE_captain_age_is_your_age_l2051_205188


namespace NUMINAMATH_CALUDE_coefficient_value_l2051_205155

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 4*x^2 + c*x - 20

-- Theorem statement
theorem coefficient_value (c : ℝ) : 
  (∀ x, P c x = 0 → x = 5) → c = -41 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l2051_205155


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2051_205136

/-- A quadratic function of the form y = ax² - 4x + 2 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * x + 2

/-- The discriminant of the quadratic function -/
def discriminant (a : ℝ) : ℝ := 16 - 8 * a

theorem quadratic_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2051_205136


namespace NUMINAMATH_CALUDE_cone_radius_l2051_205103

/-- Given a cone with surface area 6π and whose lateral surface unfolds into a semicircle,
    the radius of the base of the cone is √2. -/
theorem cone_radius (r : Real) (l : Real) : 
  (π * r * r + π * r * l = 6 * π) →  -- Surface area of cone is 6π
  (2 * π * r = π * l) →              -- Lateral surface unfolds into a semicircle
  r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l2051_205103


namespace NUMINAMATH_CALUDE_not_p_or_q_l2051_205159

-- Define proposition p
def p : Prop := ∀ (A B C : ℝ) (sinA sinB : ℝ),
  (sinA = Real.sin A ∧ sinB = Real.sin B) →
  (A > B → sinA > sinB) ∧ ¬(sinA > sinB → A > B)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + 2*x + 2 ≤ 0

-- Theorem to prove
theorem not_p_or_q : ¬p ∨ q := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_l2051_205159


namespace NUMINAMATH_CALUDE_punch_machine_settings_l2051_205178

/-- Represents a punching pattern for a 9-field ticket -/
def PunchingPattern := Fin 9 → Bool

/-- Checks if a punching pattern is symmetric when reversed -/
def is_symmetric (p : PunchingPattern) : Prop :=
  ∀ i : Fin 9, p i = p (8 - i)

/-- The total number of possible punching patterns -/
def total_patterns : ℕ := 2^9

/-- The number of symmetric punching patterns -/
def symmetric_patterns : ℕ := 2^6

/-- The number of valid punching patterns (different when reversed) -/
def valid_patterns : ℕ := total_patterns - symmetric_patterns

theorem punch_machine_settings :
  valid_patterns = 448 :=
sorry

end NUMINAMATH_CALUDE_punch_machine_settings_l2051_205178


namespace NUMINAMATH_CALUDE_num_boys_is_three_l2051_205181

/-- The number of boys sitting at the table -/
def num_boys : ℕ := sorry

/-- The number of girls sitting at the table -/
def num_girls : ℕ := 5

/-- The total number of buns on the plate -/
def total_buns : ℕ := 30

/-- The number of buns given by girls to boys they know -/
def buns_girls_to_boys : ℕ := num_girls * num_boys

/-- The number of buns given by boys to girls they don't know -/
def buns_boys_to_girls : ℕ := num_boys * num_girls

/-- Theorem stating that the number of boys is 3 -/
theorem num_boys_is_three : num_boys = 3 :=
  by
    have h1 : buns_girls_to_boys + buns_boys_to_girls = total_buns := sorry
    have h2 : num_girls * num_boys + num_boys * num_girls = total_buns := sorry
    have h3 : 2 * (num_girls * num_boys) = total_buns := sorry
    have h4 : 2 * (5 * num_boys) = 30 := sorry
    have h5 : 10 * num_boys = 30 := sorry
    sorry

end NUMINAMATH_CALUDE_num_boys_is_three_l2051_205181


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_product_l2051_205150

theorem two_numbers_sum_and_product : 
  ∃ (x y : ℝ), x + y = 10 ∧ x * y = 24 ∧ ((x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_product_l2051_205150


namespace NUMINAMATH_CALUDE_factorization_implies_m_values_l2051_205124

theorem factorization_implies_m_values (m : ℤ) :
  (∃ (a b : ℤ), ∀ (x : ℤ), x^2 + m*x - 4 = a*x + b) →
  m ∈ ({-3, 0, 3} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_factorization_implies_m_values_l2051_205124


namespace NUMINAMATH_CALUDE_triangle_numbers_exist_l2051_205162

theorem triangle_numbers_exist : 
  ∃ (a b c d e f g : ℕ), 
    (b = c * d) ∧ 
    (e - f = a + c * d - a * c) ∧ 
    (e - f = g) ∧ 
    (g = a + d) ∧ 
    (c > 0) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) := by
  sorry

end NUMINAMATH_CALUDE_triangle_numbers_exist_l2051_205162


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2051_205147

theorem cubic_equation_solution (a w : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * w) : w = 49 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2051_205147


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2051_205105

theorem cube_root_equation_solution (x : ℝ) :
  (x * (x^2)^(1/2))^(1/3) = 2 → x = 2 * (2^(1/2)) ∨ x = -2 * (2^(1/2)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2051_205105


namespace NUMINAMATH_CALUDE_kate_wand_sale_l2051_205190

/-- The amount of money Kate collected after selling magic wands -/
def kateCollected (numBought : ℕ) (numSold : ℕ) (costPerWand : ℕ) (markup : ℕ) : ℕ :=
  numSold * (costPerWand + markup)

/-- Theorem stating how much money Kate collected from selling magic wands -/
theorem kate_wand_sale :
  kateCollected 3 2 60 5 = 130 := by
  sorry

end NUMINAMATH_CALUDE_kate_wand_sale_l2051_205190


namespace NUMINAMATH_CALUDE_prove_weekly_savings_l2051_205161

def employee1_rate : ℝ := 20
def employee2_rate : ℝ := 22
def subsidy_rate : ℝ := 6
def hours_per_week : ℝ := 40

def weekly_savings : ℝ := (employee1_rate * hours_per_week) - ((employee2_rate - subsidy_rate) * hours_per_week)

theorem prove_weekly_savings : weekly_savings = 160 := by
  sorry

end NUMINAMATH_CALUDE_prove_weekly_savings_l2051_205161


namespace NUMINAMATH_CALUDE_fraction_problem_l2051_205106

theorem fraction_problem (p q x y : ℚ) :
  p / q = 4 / 5 →
  x / y + (2 * q - p) / (2 * q + p) = 3 →
  x / y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2051_205106


namespace NUMINAMATH_CALUDE_select_four_with_both_genders_eq_34_l2051_205153

/-- The number of ways to select 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls. -/
def select_four_with_both_genders (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.choose (num_boys + num_girls) 4 - Nat.choose num_boys 4

/-- Theorem stating that selecting 4 individuals from 4 boys and 3 girls,
    such that the selection includes both boys and girls, results in 34 ways. -/
theorem select_four_with_both_genders_eq_34 :
  select_four_with_both_genders 4 3 = 34 := by
  sorry

#eval select_four_with_both_genders 4 3

end NUMINAMATH_CALUDE_select_four_with_both_genders_eq_34_l2051_205153


namespace NUMINAMATH_CALUDE_toby_first_part_distance_l2051_205125

/-- Represents Toby's journey with a loaded and unloaded sled -/
def toby_journey (x : ℝ) : Prop :=
  let loaded_speed : ℝ := 10
  let unloaded_speed : ℝ := 20
  let second_part : ℝ := 120
  let third_part : ℝ := 80
  let fourth_part : ℝ := 140
  let total_time : ℝ := 39
  (x / loaded_speed) + (second_part / unloaded_speed) + 
  (third_part / loaded_speed) + (fourth_part / unloaded_speed) = total_time

/-- Theorem stating that Toby pulled the loaded sled for 180 miles in the first part of the journey -/
theorem toby_first_part_distance : 
  ∃ (x : ℝ), toby_journey x ∧ x = 180 :=
sorry

end NUMINAMATH_CALUDE_toby_first_part_distance_l2051_205125
