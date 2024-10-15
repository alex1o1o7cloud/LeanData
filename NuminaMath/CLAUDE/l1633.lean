import Mathlib

namespace NUMINAMATH_CALUDE_boys_without_calculators_l1633_163325

theorem boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : 
  total_boys = 20 →
  total_with_calculators = 30 →
  girls_with_calculators = 18 →
  total_boys - (total_with_calculators - girls_with_calculators) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l1633_163325


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_64_l1633_163393

/-- The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y -/
def sum_of_coefficients (x y : ℝ) : ℝ := (x + 3)^3

/-- Theorem: The sum of the coefficients of the terms in the expansion of (x+y+3)^3 that do not contain y is 64 -/
theorem sum_of_coefficients_is_64 :
  ∀ x y : ℝ, sum_of_coefficients x y = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_64_l1633_163393


namespace NUMINAMATH_CALUDE_seven_eighths_of_sixteen_thirds_l1633_163314

theorem seven_eighths_of_sixteen_thirds :
  (7 / 8 : ℚ) * (16 / 3 : ℚ) = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_sixteen_thirds_l1633_163314


namespace NUMINAMATH_CALUDE_new_xanadu_license_plates_l1633_163356

/-- The number of possible letters in each letter position of a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in New Xanadu. -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

/-- Theorem stating the total number of valid license plates in New Xanadu. -/
theorem new_xanadu_license_plates : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_new_xanadu_license_plates_l1633_163356


namespace NUMINAMATH_CALUDE_triangle_properties_l1633_163396

noncomputable section

-- Define the triangle
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Theorem statement
theorem triangle_properties {A B C a b c : ℝ} (h : triangle A B C a b c) :
  a = 2 * Real.sqrt 3 ∧ 
  Real.cos (2 * A + π / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l1633_163396


namespace NUMINAMATH_CALUDE_range_of_m_l1633_163341

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x = x^2 - 4*x - 6) →
  (Set.range f = Set.Icc (-10) (-6)) →
  m ∈ Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1633_163341


namespace NUMINAMATH_CALUDE_trace_of_matrix_minus_inverse_zero_l1633_163397

/-- Given a 2x2 matrix A with real entries a, 2, -3, and d,
    if A - A^(-1) is the zero matrix, then the trace of A is a + d. -/
theorem trace_of_matrix_minus_inverse_zero (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  (A - A⁻¹ = 0) → Matrix.trace A = a + d := by
  sorry

end NUMINAMATH_CALUDE_trace_of_matrix_minus_inverse_zero_l1633_163397


namespace NUMINAMATH_CALUDE_blanch_snack_slices_l1633_163315

/-- Calculates the number of pizza slices Blanch took as a snack -/
def snack_slices (initial : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) (left : ℕ) : ℕ :=
  initial - breakfast - lunch - dinner - left

theorem blanch_snack_slices :
  snack_slices 15 4 2 5 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_blanch_snack_slices_l1633_163315


namespace NUMINAMATH_CALUDE_student_pet_difference_l1633_163338

/-- Represents a fourth-grade classroom at Maplewood School -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  guinea_pigs : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A standard fourth-grade classroom at Maplewood School -/
def standard_classroom : Classroom :=
  { students := 24
  , rabbits := 3
  , guinea_pigs := 2 }

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * standard_classroom.students

/-- The total number of pets (rabbits and guinea pigs) in all classrooms -/
def total_pets : ℕ := num_classrooms * (standard_classroom.rabbits + standard_classroom.guinea_pigs)

/-- Theorem: The difference between the total number of students and the total number of pets is 95 -/
theorem student_pet_difference : total_students - total_pets = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l1633_163338


namespace NUMINAMATH_CALUDE_pants_price_l1633_163336

theorem pants_price (total coat pants shoes : ℕ) 
  (h1 : total = 700)
  (h2 : total = coat + pants + shoes)
  (h3 : coat = pants + 340)
  (h4 : coat = shoes + pants + 180) :
  pants = 100 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_l1633_163336


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1633_163365

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there exists one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l1633_163365


namespace NUMINAMATH_CALUDE_discount_and_increase_l1633_163367

theorem discount_and_increase (original_price : ℝ) (h : original_price > 0) :
  let discounted_price := original_price * (1 - 0.2)
  let increased_price := discounted_price * (1 + 0.25)
  increased_price = original_price :=
by sorry

end NUMINAMATH_CALUDE_discount_and_increase_l1633_163367


namespace NUMINAMATH_CALUDE_remainder_theorem_l1633_163309

theorem remainder_theorem : (7 * 9^20 - 2^20) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1633_163309


namespace NUMINAMATH_CALUDE_max_robot_A_l1633_163360

def robot_problem (transport_rate_A transport_rate_B : ℕ) 
                  (price_A price_B total_budget : ℕ) 
                  (total_robots : ℕ) : Prop :=
  (transport_rate_A = transport_rate_B + 30) ∧
  (1500 / transport_rate_A = 1000 / transport_rate_B) ∧
  (price_A = 50000) ∧
  (price_B = 30000) ∧
  (total_robots = 12) ∧
  (total_budget = 450000)

theorem max_robot_A (transport_rate_A transport_rate_B : ℕ) 
                    (price_A price_B total_budget : ℕ) 
                    (total_robots : ℕ) :
  robot_problem transport_rate_A transport_rate_B price_A price_B total_budget total_robots →
  ∀ m : ℕ, m ≤ total_robots ∧ 
           price_A * m + price_B * (total_robots - m) ≤ total_budget →
  m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_robot_A_l1633_163360


namespace NUMINAMATH_CALUDE_equation_solutions_l1633_163347

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ 
    x₁^2 - 2*x₁ - 2 = 0 ∧ x₂^2 - 2*x₂ - 2 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3/2 ∧ y₂ = 7/2 ∧ 
    2*(y₁ - 3)^2 = y₁ - 3 ∧ 2*(y₂ - 3)^2 = y₂ - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1633_163347


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1633_163334

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m+2)*x + m
  -- The equation always has two distinct real roots
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  -- When the sum condition is satisfied, m = 3
  (x₁ + x₂ + 2*x₁*x₂ = 1 → m = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1633_163334


namespace NUMINAMATH_CALUDE_pretzel_ratio_is_three_to_one_l1633_163383

/-- The number of pretzels Barry bought -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels Angie bought -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels Shelly bought -/
def shelly_pretzels : ℕ := barry_pretzels / 2

/-- The ratio of pretzels Angie bought to pretzels Shelly bought -/
def pretzel_ratio : ℚ := angie_pretzels / shelly_pretzels

theorem pretzel_ratio_is_three_to_one :
  pretzel_ratio = 3 := by sorry

end NUMINAMATH_CALUDE_pretzel_ratio_is_three_to_one_l1633_163383


namespace NUMINAMATH_CALUDE_egg_game_probabilities_l1633_163308

/-- Represents the color of an egg -/
inductive EggColor
| Yellow
| Red
| Blue

/-- Represents the game setup -/
structure EggGame where
  total_eggs : Nat
  yellow_eggs : Nat
  red_eggs : Nat
  blue_eggs : Nat
  game_fee : Int
  same_color_reward : Int
  diff_color_reward : Int

/-- Defines the game rules -/
def game : EggGame :=
  { total_eggs := 9
  , yellow_eggs := 3
  , red_eggs := 3
  , blue_eggs := 3
  , game_fee := 10
  , same_color_reward := 100
  , diff_color_reward := 10 }

/-- Event A: Picking a yellow egg on the first draw -/
def eventA (g : EggGame) : Rat :=
  g.yellow_eggs / g.total_eggs

/-- Event B: Winning the maximum reward -/
def eventB (g : EggGame) : Rat :=
  3 / Nat.choose g.total_eggs 3

/-- Probability of both events A and B occurring -/
def eventAB (g : EggGame) : Rat :=
  1 / Nat.choose g.total_eggs 3

/-- Expected profit from playing the game -/
def expectedProfit (g : EggGame) : Rat :=
  (g.same_color_reward - g.game_fee) * eventB g +
  (g.diff_color_reward - g.game_fee) * (9 / 28) +
  (-g.game_fee) * (18 / 28)

theorem egg_game_probabilities :
  eventA game = 1/3 ∧
  eventB game = 1/28 ∧
  eventAB game = eventA game * eventB game ∧
  expectedProfit game < 0 :=
sorry

end NUMINAMATH_CALUDE_egg_game_probabilities_l1633_163308


namespace NUMINAMATH_CALUDE_threes_squared_threes_2009_squared_l1633_163302

/-- Represents a number consisting of n repeated digits -/
def repeated_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | k + 1 => d + 10 * (repeated_digit d k)

/-- The theorem to be proved -/
theorem threes_squared (n : Nat) (h : n > 0) :
  (repeated_digit 3 n) ^ 2 = 
    repeated_digit 1 (n-1) * 10^n + 
    repeated_digit 8 (n-1) * 10 + 9 := by
  sorry

/-- The specific case for 2009 threes -/
theorem threes_2009_squared :
  (repeated_digit 3 2009) ^ 2 = 
    repeated_digit 1 2008 * 10^2009 + 
    repeated_digit 8 2008 * 10 + 9 := by
  sorry

end NUMINAMATH_CALUDE_threes_squared_threes_2009_squared_l1633_163302


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l1633_163379

theorem evaluate_polynomial : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l1633_163379


namespace NUMINAMATH_CALUDE_train_length_l1633_163362

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ)
  (h1 : bridge_length = 150)
  (h2 : bridge_time = 7.5)
  (h3 : lamp_time = 2.5)
  (h4 : bridge_time > 0)
  (h5 : lamp_time > 0) :
  ∃ (train_length : ℝ),
    train_length = 75 ∧
    (train_length + bridge_length) / bridge_time = train_length / lamp_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1633_163362


namespace NUMINAMATH_CALUDE_smaller_number_is_ten_l1633_163335

theorem smaller_number_is_ten (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) (h3 : x ≤ y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_is_ten_l1633_163335


namespace NUMINAMATH_CALUDE_centroid_trajectory_on_hyperbola_l1633_163388

/-- The trajectory of the centroid of a triangle formed by a point on a hyperbola and its foci -/
theorem centroid_trajectory_on_hyperbola (x y m n : ℝ) :
  let f₁ : ℝ × ℝ := (5, 0)
  let f₂ : ℝ × ℝ := (-5, 0)
  let p : ℝ × ℝ := (m, n)
  let g : ℝ × ℝ := (x, y)
  (m^2 / 16 - n^2 / 9 = 1) →  -- P is on the hyperbola
  (x = (m + 5 + (-5)) / 3 ∧ y = n / 3) →  -- G is the centroid of ΔF₁F₂P
  (y ≠ 0) →
  (x^2 / (16/9) - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_centroid_trajectory_on_hyperbola_l1633_163388


namespace NUMINAMATH_CALUDE_difference_between_x_and_y_l1633_163323

theorem difference_between_x_and_y : 
  ∀ x y : ℤ, x = 10 ∧ y = 5 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_x_and_y_l1633_163323


namespace NUMINAMATH_CALUDE_race_finish_orders_l1633_163350

-- Define the number of racers
def num_racers : ℕ := 4

-- Define the function to calculate the number of permutations
def num_permutations (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem race_finish_orders :
  num_permutations num_racers = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l1633_163350


namespace NUMINAMATH_CALUDE_garden_length_difference_l1633_163363

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Properties of the rectangular garden -/
def GardenProperties (garden : RectangularGarden) : Prop :=
  garden.length > 3 * garden.width ∧
  2 * (garden.length + garden.width) = 100 ∧
  garden.length = 38

theorem garden_length_difference (garden : RectangularGarden) 
  (h : GardenProperties garden) : 
  garden.length - 3 * garden.width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_difference_l1633_163363


namespace NUMINAMATH_CALUDE_meeting_equation_correct_l1633_163322

/-- Represents the scenario of two people meeting on a straight road -/
def meeting_equation (x : ℝ) : Prop :=
  x / 6 + (x - 1) / 4 = 1

/-- The time it takes for A to travel the entire distance -/
def time_A : ℝ := 4

/-- The time it takes for B to travel the entire distance -/
def time_B : ℝ := 6

/-- The time difference between A and B starting their journey -/
def time_difference : ℝ := 1

/-- Theorem stating that the meeting equation correctly represents the scenario -/
theorem meeting_equation_correct (x : ℝ) :
  (x ≥ time_difference) →
  (x / time_B + (x - time_difference) / time_A = 1) ↔ meeting_equation x :=
by sorry

end NUMINAMATH_CALUDE_meeting_equation_correct_l1633_163322


namespace NUMINAMATH_CALUDE_sugar_percentage_approx_l1633_163311

-- Define the initial solution volume
def initial_volume : ℝ := 500

-- Define the initial composition percentages
def water_percent : ℝ := 0.60
def cola_percent : ℝ := 0.08
def orange_percent : ℝ := 0.10
def lemon_percent : ℝ := 0.12

-- Define the added components
def added_sugar : ℝ := 4
def added_water : ℝ := 15
def added_cola : ℝ := 9
def added_orange : ℝ := 5
def added_lemon : ℝ := 7
def added_ice : ℝ := 8

-- Calculate the new total volume
def new_volume : ℝ := initial_volume + added_sugar + added_water + added_cola + added_orange + added_lemon + added_ice

-- Define the theorem
theorem sugar_percentage_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |added_sugar / new_volume - 0.0073| < ε :=
sorry

end NUMINAMATH_CALUDE_sugar_percentage_approx_l1633_163311


namespace NUMINAMATH_CALUDE_hcf_problem_l1633_163369

theorem hcf_problem (a b : ℕ) (h1 : a = 280) (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 14) :
  Nat.gcd a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1633_163369


namespace NUMINAMATH_CALUDE_equation_solutions_inequality_system_solution_l1633_163391

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 2*x - 4 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := 4*(x - 1) < x + 2 ∧ (x + 7) / 3 > x

-- Theorem for the equation solutions
theorem equation_solutions : 
  ∃ (x1 x2 : ℝ), x1 = 1 + Real.sqrt 5 ∧ x2 = 1 - Real.sqrt 5 ∧ 
  equation x1 ∧ equation x2 ∧ 
  ∀ (x : ℝ), equation x → x = x1 ∨ x = x2 := by sorry

-- Theorem for the inequality system solution
theorem inequality_system_solution :
  ∀ (x : ℝ), inequality_system x ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_inequality_system_solution_l1633_163391


namespace NUMINAMATH_CALUDE_gcd_of_specific_powers_of_two_l1633_163353

theorem gcd_of_specific_powers_of_two : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 8191 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_powers_of_two_l1633_163353


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l1633_163371

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
  , tuesday_hours := 6
  , wednesday_hours := 8
  , thursday_hours := 6
  , friday_hours := 8
  , weekly_earnings := 432 }

theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 12 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l1633_163371


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1633_163339

/-- The equation that the vertex coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- Theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
    ∀ (x' y' : ℝ), vertex_equation x' y' →
      rectangle_area x ≥ rectangle_area x' ∧
      rectangle_area x = 34.171875 :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1633_163339


namespace NUMINAMATH_CALUDE_mika_birthday_stickers_l1633_163380

/-- The number of stickers Mika received for her birthday -/
def birthday_stickers (initial : ℕ) (bought : ℕ) (given_away : ℕ) (used : ℕ) (left : ℕ) : ℕ :=
  (left + given_away + used) - (initial + bought)

/-- Theorem stating that Mika received 20 stickers for her birthday -/
theorem mika_birthday_stickers : 
  birthday_stickers 20 26 6 58 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mika_birthday_stickers_l1633_163380


namespace NUMINAMATH_CALUDE_derivative_at_zero_implies_k_value_l1633_163389

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem derivative_at_zero_implies_k_value (k : ℝ) :
  (deriv (f k)) 0 = 27 → k = 3 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_zero_implies_k_value_l1633_163389


namespace NUMINAMATH_CALUDE_fraction_calculation_l1633_163327

theorem fraction_calculation : (1/3 + 1/6) * 4/7 * 5/9 = 10/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1633_163327


namespace NUMINAMATH_CALUDE_mia_has_110_dollars_l1633_163392

/-- The amount of money Darwin has -/
def darwins_money : ℕ := 45

/-- The amount of money Mia has -/
def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end NUMINAMATH_CALUDE_mia_has_110_dollars_l1633_163392


namespace NUMINAMATH_CALUDE_complete_square_integer_l1633_163386

theorem complete_square_integer (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_integer_l1633_163386


namespace NUMINAMATH_CALUDE_pie_chart_probability_l1633_163358

theorem pie_chart_probability (pE pF pG pH : ℚ) : 
  pE = 1/3 →
  pF = 1/6 →
  pG = pH →
  pE + pF + pG + pH = 1 →
  pG = 1/4 := by
sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l1633_163358


namespace NUMINAMATH_CALUDE_exponential_characterization_l1633_163305

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

theorem exponential_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :
  is_exponential f :=
sorry

end NUMINAMATH_CALUDE_exponential_characterization_l1633_163305


namespace NUMINAMATH_CALUDE_water_added_to_tank_l1633_163345

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 64) 
  (initial_fraction : ℚ) 
  (h2 : initial_fraction = 3/4) 
  (final_fraction : ℚ) 
  (h3 : final_fraction = 7/8) : 
  (final_fraction - initial_fraction) * tank_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l1633_163345


namespace NUMINAMATH_CALUDE_counterexample_exists_l1633_163330

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ a*b := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1633_163330


namespace NUMINAMATH_CALUDE_female_officers_count_l1633_163390

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) : 
  total_on_duty = 152 →
  female_percentage = 19 / 100 →
  (total_on_duty / 2 : ℚ) = female_percentage * 400 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l1633_163390


namespace NUMINAMATH_CALUDE_vector_on_line_l1633_163354

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Two distinct vectors p and q define a line. The vector (3/5)*p + (2/5)*q lies on that line. -/
theorem vector_on_line (p q : V) (h : p ≠ q) :
  ∃ t : ℝ, (3/5 : ℝ) • p + (2/5 : ℝ) • q = p + t • (q - p) := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_l1633_163354


namespace NUMINAMATH_CALUDE_matches_played_l1633_163359

/-- Represents the number of matches played -/
def n : ℕ := sorry

/-- The current batting average -/
def current_average : ℕ := 50

/-- The runs scored in the next match -/
def next_match_runs : ℕ := 78

/-- The new batting average after the next match -/
def new_average : ℕ := 54

/-- The total runs scored before the next match -/
def total_runs : ℕ := n * current_average

/-- The total runs after the next match -/
def new_total_runs : ℕ := total_runs + next_match_runs

/-- The theorem stating the number of matches played -/
theorem matches_played : n = 6 := by sorry

end NUMINAMATH_CALUDE_matches_played_l1633_163359


namespace NUMINAMATH_CALUDE_line_intersection_with_circle_l1633_163342

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y m : ℝ) : Prop := y = x + m

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define a chord of the circle
def chord (A B : ℝ × ℝ) : Prop :=
  point_on_circle A.1 A.2 ∧ point_on_circle B.1 B.2

-- Define a circle passing through three points
def circle_through_points (A B O : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (O.1 - center.1)^2 + (O.2 - center.2)^2 = radius^2

-- Theorem statement
theorem line_intersection_with_circle :
  ∃ (m : ℝ), m = 1 ∨ m = -4 ∧
  ∀ (x y : ℝ),
    line_with_slope_1 x y m →
    (∃ (A B : ℝ × ℝ),
      chord A B ∧
      line_with_slope_1 A.1 A.2 m ∧
      line_with_slope_1 B.1 B.2 m ∧
      circle_through_points A B origin) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_circle_l1633_163342


namespace NUMINAMATH_CALUDE_sequence_property_l1633_163373

theorem sequence_property (a : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1633_163373


namespace NUMINAMATH_CALUDE_circle_radius_equality_l1633_163357

/-- The radius of a circle whose area is equal to the sum of the areas of four circles with radius 2 cm is 4 cm. -/
theorem circle_radius_equality (r : ℝ) : r > 0 → π * r^2 = 4 * (π * 2^2) → r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_equality_l1633_163357


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1633_163348

/-- Represents the quantity ratio of products A, B, and C -/
def quantity_ratio : Fin 3 → ℕ
  | 0 => 2  -- Product A
  | 1 => 3  -- Product B
  | 2 => 5  -- Product C
  | _ => 0  -- Unreachable case

/-- The sample size of product A -/
def sample_size_A : ℕ := 10

/-- Calculates the total sample size based on the sample size of product A -/
def total_sample_size (sample_A : ℕ) : ℕ :=
  sample_A * (quantity_ratio 0 + quantity_ratio 1 + quantity_ratio 2) / quantity_ratio 0

theorem stratified_sample_size :
  total_sample_size sample_size_A = 50 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1633_163348


namespace NUMINAMATH_CALUDE_second_period_odds_correct_l1633_163316

/-- Represents the types of light bulbs -/
inductive BulbType
  | A
  | B
  | C

/-- Initial odds of burning out for each bulb type -/
def initialOdds (t : BulbType) : ℚ :=
  match t with
  | BulbType.A => 1/3
  | BulbType.B => 1/4
  | BulbType.C => 1/5

/-- Decrease rate for each bulb type -/
def decreaseRate (t : BulbType) : ℚ :=
  match t with
  | BulbType.A => 1/2
  | BulbType.B => 1/3
  | BulbType.C => 1/4

/-- Calculates the odds of burning out for the second 6-month period -/
def secondPeriodOdds (t : BulbType) : ℚ :=
  initialOdds t * decreaseRate t

/-- Theorem stating the odds of burning out for each bulb type in the second 6-month period -/
theorem second_period_odds_correct :
  (secondPeriodOdds BulbType.A = 1/6) ∧
  (secondPeriodOdds BulbType.B = 1/12) ∧
  (secondPeriodOdds BulbType.C = 1/20) := by
  sorry

end NUMINAMATH_CALUDE_second_period_odds_correct_l1633_163316


namespace NUMINAMATH_CALUDE_fraction_calculation_l1633_163352

theorem fraction_calculation : (7 / 9 - 5 / 6 + 5 / 18) * 18 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1633_163352


namespace NUMINAMATH_CALUDE_problem_solution_l1633_163372

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1633_163372


namespace NUMINAMATH_CALUDE_ball_max_height_l1633_163375

/-- The path of a ball thrown on a planet with stronger gravity -/
def ballPath (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

/-- The maximum height reached by the ball -/
def maxHeight : ℝ := 140

theorem ball_max_height :
  ∃ t : ℝ, ballPath t = maxHeight ∧ ∀ s : ℝ, ballPath s ≤ maxHeight := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l1633_163375


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1633_163332

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [17, 29, 41, 53]
  (numbers.sum / numbers.length : ℝ) = 35 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1633_163332


namespace NUMINAMATH_CALUDE_expression_factorization_l1633_163368

theorem expression_factorization (x y : ℝ) : 
  (x + y)^2 + 4*(x - y)^2 - 4*(x^2 - y^2) = (x - 3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1633_163368


namespace NUMINAMATH_CALUDE_cistern_filling_fraction_l1633_163340

/-- Given a pipe that can fill a cistern in 55 minutes, 
    this theorem proves that the fraction of the cistern 
    filled in 5 minutes is 1/11. -/
theorem cistern_filling_fraction 
  (total_time : ℕ) 
  (filling_time : ℕ) 
  (h1 : total_time = 55) 
  (h2 : filling_time = 5) : 
  (filling_time : ℚ) / total_time = 1 / 11 :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_fraction_l1633_163340


namespace NUMINAMATH_CALUDE_greatest_multiple_of_8_remainder_l1633_163321

def is_valid_number (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ 0 ∧ d₂ ≠ 0

theorem greatest_multiple_of_8_remainder (M : ℕ) : 
  (∀ n, n > M → ¬(is_valid_number n ∧ 8 ∣ n)) →
  is_valid_number M →
  8 ∣ M →
  M % 1000 = 984 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_8_remainder_l1633_163321


namespace NUMINAMATH_CALUDE_max_rental_income_l1633_163304

/-- Represents the daily rental income function for a hotel --/
def rental_income (x : ℕ) : ℕ :=
  (100 + 10 * x) * (300 - 10 * x)

/-- Theorem stating the maximum daily rental income and the rent at which it's achieved --/
theorem max_rental_income :
  (∃ x : ℕ, x < 30 ∧ rental_income x = 40000) ∧
  (∀ y : ℕ, y < 30 → rental_income y ≤ 40000) ∧
  (rental_income 10 = 40000) := by
  sorry

#check max_rental_income

end NUMINAMATH_CALUDE_max_rental_income_l1633_163304


namespace NUMINAMATH_CALUDE_probability_at_least_three_successes_in_five_trials_l1633_163317

theorem probability_at_least_three_successes_in_five_trials : 
  let n : ℕ := 5
  let p : ℝ := 1/2
  let binomial_probability (k : ℕ) := (n.choose k) * p^k * (1-p)^(n-k)
  (binomial_probability 3 + binomial_probability 4 + binomial_probability 5) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_successes_in_five_trials_l1633_163317


namespace NUMINAMATH_CALUDE_gross_monthly_salary_l1633_163337

theorem gross_monthly_salary (rent food_expenses mortgage savings taxes gross_salary : ℚ) : 
  rent = 600 →
  food_expenses = (3/5) * rent →
  mortgage = 3 * food_expenses →
  savings = 2000 →
  taxes = (2/5) * savings →
  gross_salary = rent + food_expenses + mortgage + taxes + savings →
  gross_salary = 4840 := by
sorry

end NUMINAMATH_CALUDE_gross_monthly_salary_l1633_163337


namespace NUMINAMATH_CALUDE_investment_duration_l1633_163313

/-- Represents a partner in the investment scenario -/
structure Partner where
  investment : ℚ
  profit : ℚ
  duration : ℚ

/-- The investment scenario with two partners -/
def InvestmentScenario (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.duration = 8

theorem investment_duration (p q : Partner) 
  (h : InvestmentScenario p q) : q.duration = 16 := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_l1633_163313


namespace NUMINAMATH_CALUDE_equilateral_triangles_count_l1633_163333

/-- The number of points evenly spaced on a circle -/
def n : ℕ := 900

/-- The number of points needed to form an equilateral triangle on the circle -/
def equilateral_spacing : ℕ := n / 3

/-- The number of equilateral triangles with all vertices on the circle -/
def all_vertices_on_circle : ℕ := equilateral_spacing

/-- The number of ways to choose 2 points from n points -/
def choose_two : ℕ := n * (n - 1) / 2

/-- The total number of equilateral triangles with at least two vertices from the n points -/
def total_triangles : ℕ := 2 * choose_two - all_vertices_on_circle

theorem equilateral_triangles_count : total_triangles = 808800 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_count_l1633_163333


namespace NUMINAMATH_CALUDE_area_of_region_s_l1633_163382

/-- A square with side length 4 -/
structure Square :=
  (side_length : ℝ)
  (is_four : side_length = 4)

/-- A region S in the square -/
structure Region (sq : Square) :=
  (area : ℝ)
  (in_square : area ≤ sq.side_length^2)
  (closer_to_vertex : area > 0)

/-- Theorem: The area of region S is 2 -/
theorem area_of_region_s (sq : Square) (S : Region sq) : S.area = 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_region_s_l1633_163382


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l1633_163300

def fibonacci_like_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci_like_sequence (n + 1) + fibonacci_like_sequence n

theorem fifth_term_is_five : fibonacci_like_sequence 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l1633_163300


namespace NUMINAMATH_CALUDE_john_quilt_cost_l1633_163384

/-- Calculates the cost of a rectangular quilt given its dimensions and cost per square foot -/
def quiltCost (length width costPerSqFt : ℝ) : ℝ :=
  length * width * costPerSqFt

/-- Proves that a 7ft by 8ft quilt at $40 per square foot costs $2240 -/
theorem john_quilt_cost :
  quiltCost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_john_quilt_cost_l1633_163384


namespace NUMINAMATH_CALUDE_congruence_solution_l1633_163398

theorem congruence_solution (x : ℤ) : 
  (∃ (a m : ℤ), m ≥ 2 ∧ 0 ≤ a ∧ a < m ∧ x ≡ a [ZMOD m]) →
  ((10 * x + 3) ≡ 6 [ZMOD 15] ↔ x ≡ 0 [ZMOD 3]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1633_163398


namespace NUMINAMATH_CALUDE_cookies_given_to_friend_l1633_163378

theorem cookies_given_to_friend (total : ℕ) (given_to_friend : ℕ) (given_to_family : ℕ) (eaten : ℕ) (left : ℕ) :
  total = 19 →
  given_to_family = (total - given_to_friend) / 2 →
  eaten = 2 →
  left = 5 →
  left = total - given_to_friend - given_to_family - eaten →
  given_to_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_given_to_friend_l1633_163378


namespace NUMINAMATH_CALUDE_felix_weight_lifting_l1633_163395

/-- Felix's weight lifting problem -/
theorem felix_weight_lifting (felix_weight : ℝ) (felix_lift : ℝ) (brother_lift : ℝ)
  (h1 : felix_lift = 150)
  (h2 : brother_lift = 600)
  (h3 : brother_lift = 3 * (2 * felix_weight)) :
  felix_lift / felix_weight = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_felix_weight_lifting_l1633_163395


namespace NUMINAMATH_CALUDE_puppy_sale_revenue_l1633_163346

/-- Calculates the total amount received from selling puppies --/
theorem puppy_sale_revenue (num_dogs : ℕ) (puppies_per_dog : ℕ) (sale_fraction : ℚ) (price_per_puppy : ℕ) : 
  num_dogs = 2 → 
  puppies_per_dog = 10 → 
  sale_fraction = 3/4 → 
  price_per_puppy = 200 → 
  (↑num_dogs * ↑puppies_per_dog : ℚ) * sale_fraction * ↑price_per_puppy = 3000 := by
  sorry

#check puppy_sale_revenue

end NUMINAMATH_CALUDE_puppy_sale_revenue_l1633_163346


namespace NUMINAMATH_CALUDE_volume_conversion_l1633_163387

/-- Proves that a volume of 108 cubic feet is equal to 4 cubic yards -/
theorem volume_conversion (box_volume_cubic_feet : ℝ) (cubic_feet_per_cubic_yard : ℝ) :
  box_volume_cubic_feet = 108 →
  cubic_feet_per_cubic_yard = 27 →
  box_volume_cubic_feet / cubic_feet_per_cubic_yard = 4 := by
sorry

end NUMINAMATH_CALUDE_volume_conversion_l1633_163387


namespace NUMINAMATH_CALUDE_apple_difference_l1633_163394

theorem apple_difference (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 46)
  (h2 : remaining_apples = 14) : 
  initial_apples - remaining_apples = 32 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l1633_163394


namespace NUMINAMATH_CALUDE_new_students_admitted_l1633_163318

theorem new_students_admitted (initial_students_per_section : ℕ) 
  (new_sections : ℕ) (final_total_sections : ℕ) (final_students_per_section : ℕ) :
  initial_students_per_section = 24 →
  new_sections = 3 →
  final_total_sections = 16 →
  final_students_per_section = 21 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 24 := by
  sorry

end NUMINAMATH_CALUDE_new_students_admitted_l1633_163318


namespace NUMINAMATH_CALUDE_study_group_selection_probability_l1633_163351

/-- Represents the probability of selecting a member with specific characteristics from a study group -/
def study_group_probability (women_percent : ℝ) (men_percent : ℝ) 
  (women_lawyer_percent : ℝ) (women_doctor_percent : ℝ) (women_engineer_percent : ℝ)
  (men_lawyer_percent : ℝ) (men_doctor_percent : ℝ) (men_engineer_percent : ℝ) : ℝ :=
  let woman_lawyer_prob := women_percent * women_lawyer_percent
  let man_doctor_prob := men_percent * men_doctor_percent
  woman_lawyer_prob + man_doctor_prob

/-- The probability of selecting a woman lawyer or a man doctor from the study group is 0.33 -/
theorem study_group_selection_probability : 
  study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30 = 0.33 := by
  sorry

#eval study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30

end NUMINAMATH_CALUDE_study_group_selection_probability_l1633_163351


namespace NUMINAMATH_CALUDE_function_properties_l1633_163328

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log (x + b)) / x

noncomputable def g (a x : ℝ) : ℝ := x + 2 / x - a - 2

noncomputable def F (a b x : ℝ) : ℝ := f a b x + g a x

theorem function_properties (a b : ℝ) (ha : a ≤ 2) (ha_nonzero : a ≠ 0) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = f a b x) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x - f a b 1 = m * (x - 1) → f a b 3 = 0) →
  (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ F a b x = 0) →
  (b = 2 * a ∧ (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2))) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1633_163328


namespace NUMINAMATH_CALUDE_white_squares_37th_row_l1633_163324

/-- Represents the number of squares in a row of the stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def num_white_squares (n : ℕ) : ℕ := (num_squares n + 1) / 2

theorem white_squares_37th_row :
  num_white_squares 37 = 37 := by sorry

end NUMINAMATH_CALUDE_white_squares_37th_row_l1633_163324


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1633_163374

theorem sin_alpha_value (m : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (m, -3) ∧ P.1 * Real.cos α = P.2 * Real.sin α) →
  Real.tan α = -3/4 →
  Real.sin α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1633_163374


namespace NUMINAMATH_CALUDE_pyramid_stack_height_l1633_163310

/-- Represents a stack of square blocks arranged in a stepped pyramid. -/
structure BlockStack where
  blockSideLength : ℝ
  numLayers : ℕ
  blocksPerLayer : ℕ → ℕ

/-- Calculates the total height of a block stack. -/
def totalHeight (stack : BlockStack) : ℝ :=
  stack.blockSideLength * stack.numLayers

/-- Theorem: The total height of a specific stepped pyramid stack is 30 cm. -/
theorem pyramid_stack_height :
  let stack : BlockStack := {
    blockSideLength := 10,
    numLayers := 3,
    blocksPerLayer := fun n => 3 - n + 1
  }
  totalHeight stack = 30 := by sorry

end NUMINAMATH_CALUDE_pyramid_stack_height_l1633_163310


namespace NUMINAMATH_CALUDE_flight_cost_proof_l1633_163364

theorem flight_cost_proof (initial_cost : ℝ) : 
  (∃ (cost_per_person_4 cost_per_person_5 : ℝ),
    cost_per_person_4 = initial_cost / 4 ∧
    cost_per_person_5 = initial_cost / 5 ∧
    cost_per_person_4 - cost_per_person_5 = 30) →
  initial_cost = 600 := by
sorry

end NUMINAMATH_CALUDE_flight_cost_proof_l1633_163364


namespace NUMINAMATH_CALUDE_johns_share_is_18_l1633_163306

/-- The amount one person pays when splitting the cost of multiple items equally -/
def split_cost (num_items : ℕ) (price_per_item : ℚ) (num_people : ℕ) : ℚ :=
  (num_items : ℚ) * price_per_item / (num_people : ℚ)

/-- Theorem: John's share of the cake cost is $18 -/
theorem johns_share_is_18 :
  split_cost 3 12 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_johns_share_is_18_l1633_163306


namespace NUMINAMATH_CALUDE_base_prime_rep_1170_l1633_163377

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def base_prime_representation (n : ℕ) : List ℕ := sorry

theorem base_prime_rep_1170 :
  base_prime_representation 1170 = [1, 2, 1, 0, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_1170_l1633_163377


namespace NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l1633_163312

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- Function to calculate the minimum number of rings to separate and reattach -/
def minRingsToConnect (chain : ChainCollection) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rings to separate and reattach for the given problem -/
theorem min_rings_to_connect_five_links :
  let chain := ChainCollection.mk (List.replicate 5 (ChainLink.mk 3))
  minRingsToConnect chain = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l1633_163312


namespace NUMINAMATH_CALUDE_newer_train_distance_proof_l1633_163361

/-- The distance traveled by the newer train -/
def newer_train_distance (older_train_distance : ℝ) : ℝ :=
  older_train_distance * 1.5

theorem newer_train_distance_proof (older_train_distance : ℝ) 
  (h : older_train_distance = 300) :
  newer_train_distance older_train_distance = 450 := by
  sorry

end NUMINAMATH_CALUDE_newer_train_distance_proof_l1633_163361


namespace NUMINAMATH_CALUDE_frog_jump_distance_l1633_163355

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_grasshopper_diff : ℕ) 
  (h1 : grasshopper_jump = 36)
  (h2 : frog_grasshopper_diff = 17) :
  grasshopper_jump + frog_grasshopper_diff = 53 :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l1633_163355


namespace NUMINAMATH_CALUDE_units_digit_product_l1633_163303

theorem units_digit_product (n : ℕ) : 
  (4^150 * 9^151 * 16^152) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_product_l1633_163303


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1633_163343

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  (∀ φ, 0 < φ ∧ φ < Real.pi → 
    Real.sin (φ / 2) * (1 + Real.cos φ) ≤ Real.sin (θ / 2) * (1 + Real.cos θ)) ↔ 
  Real.sin (θ / 2) * (1 + Real.cos θ) = 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1633_163343


namespace NUMINAMATH_CALUDE_area_is_zero_l1633_163331

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + 5*y + 50 = 25 + 15*y - y^2

-- Define the line
def line (x : ℝ) : ℝ := x - 4

-- Define the region above the line
def region_above_line (x y : ℝ) : Prop :=
  y > line x

-- Define the area of the region
def area_of_region : ℝ := 0

-- Theorem statement
theorem area_is_zero :
  area_of_region = 0 :=
sorry

end NUMINAMATH_CALUDE_area_is_zero_l1633_163331


namespace NUMINAMATH_CALUDE_nobel_laureates_count_l1633_163349

/-- The number of Nobel Prize laureates at a workshop given specific conditions -/
theorem nobel_laureates_count (total : ℕ) (wolf : ℕ) (wolf_and_nobel : ℕ) :
  total = 50 →
  wolf = 31 →
  wolf_and_nobel = 14 →
  (total - wolf) = 2 * (total - wolf - 3) / 2 →
  ∃ (nobel : ℕ), nobel = 25 ∧ nobel = wolf_and_nobel + (total - wolf - 3) / 2 + 3 := by
  sorry

#check nobel_laureates_count

end NUMINAMATH_CALUDE_nobel_laureates_count_l1633_163349


namespace NUMINAMATH_CALUDE_cube_painting_l1633_163329

/-- Given a cube of side length n constructed from n³ smaller cubes,
    if (n-2)³ = 343 small cubes remain unpainted after some faces are painted,
    then exactly 3 faces of the large cube must have been painted. -/
theorem cube_painting (n : ℕ) (h : (n - 2)^3 = 343) :
  ∃ (painted_faces : ℕ), painted_faces = 3 ∧ painted_faces < 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_l1633_163329


namespace NUMINAMATH_CALUDE_inequality_proof_l1633_163370

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + y^2)^2 ≥ (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ∧
  ((x^2 + y^2)^2 = (x+y+z)*(x-y+z)*(x+y-z)*(y+z-x) ↔ x = y ∧ z = x*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1633_163370


namespace NUMINAMATH_CALUDE_james_total_points_l1633_163301

/-- Represents the quiz bowl game rules and James' performance --/
structure QuizBowl where
  points_per_correct : ℕ := 2
  points_per_incorrect : ℕ := 1
  bonus_points : ℕ := 4
  total_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_correct_answers : ℕ := 24
  james_unanswered : ℕ := 1

/-- Calculates the total points James earned in the quiz bowl --/
def calculate_points (game : QuizBowl) : ℕ :=
  let total_questions := game.total_rounds * game.questions_per_round
  let points_from_correct := game.james_correct_answers * game.points_per_correct
  let full_rounds := (total_questions - game.james_unanswered - game.james_correct_answers) / game.questions_per_round
  let bonus_points := full_rounds * game.bonus_points
  points_from_correct + bonus_points

/-- Theorem stating that James' total points in the quiz bowl are 64 --/
theorem james_total_points (game : QuizBowl) : calculate_points game = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_total_points_l1633_163301


namespace NUMINAMATH_CALUDE_greatest_color_count_l1633_163366

theorem greatest_color_count (α β : ℝ) (h1 : 1 < α) (h2 : α < β) : 
  (∀ (r : ℕ), r > 2 → 
    ∃ (f : ℕ+ → Fin r), ∀ (x y : ℕ+), 
      f x = f y → (α : ℝ) ≤ (x : ℝ) / (y : ℝ) → (x : ℝ) / (y : ℝ) ≤ β → False) ∧
  (∀ (f : ℕ+ → Fin 2), ∃ (x y : ℕ+), 
    f x = f y ∧ (α : ℝ) ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β) :=
by sorry

end NUMINAMATH_CALUDE_greatest_color_count_l1633_163366


namespace NUMINAMATH_CALUDE_rectangle_area_l1633_163385

theorem rectangle_area (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 40 →
  length = 2 * width →
  width * length = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1633_163385


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l1633_163307

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the total number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together for 30 minutes will produce 3000 copies -/
theorem two_machines_total_copies 
  (machine1 : CopyMachine) 
  (machine2 : CopyMachine) 
  (h1 : machine1.rate = 35) 
  (h2 : machine2.rate = 65) : 
  copies_made machine1 30 + copies_made machine2 30 = 3000 := by
  sorry

#check two_machines_total_copies

end NUMINAMATH_CALUDE_two_machines_total_copies_l1633_163307


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1633_163319

/-- Given two quadratic functions f and g, if f has two distinct real roots,
    then g must have at least one real root. -/
theorem quadratic_roots_relation (a b c : ℝ) (h : a * c ≠ 0) :
  let f := fun x : ℝ ↦ a * x^2 + b * x + c
  let g := fun x : ℝ ↦ c * x^2 + b * x + a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ z : ℝ, g z = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1633_163319


namespace NUMINAMATH_CALUDE_sams_cans_sams_final_can_count_l1633_163320

/-- Sam's can collection problem -/
theorem sams_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) 
  (bags_given_away : ℕ) (large_cans_found : ℕ) : ℕ :=
  let total_bags := saturday_bags + sunday_bags
  let total_cans := total_bags * cans_per_bag
  let cans_given_away := bags_given_away * cans_per_bag
  let remaining_cans := total_cans - cans_given_away
  let large_cans_equivalent := large_cans_found * 2
  remaining_cans + large_cans_equivalent

/-- Proof of Sam's final can count -/
theorem sams_final_can_count : sams_cans 3 4 9 2 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sams_cans_sams_final_can_count_l1633_163320


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1633_163326

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997 * (x - 1) = -1)
  (hy : (y - 1)^3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1633_163326


namespace NUMINAMATH_CALUDE_marble_probability_l1633_163344

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 5) (h3 : red = 7) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l1633_163344


namespace NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l1633_163381

theorem only_negative_one_squared_is_negative : 
  ((-1 : ℝ)^0 < 0 ∨ |(-1 : ℝ)| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1 : ℝ)^2 < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |(-1 : ℝ)| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1 : ℝ)^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l1633_163381


namespace NUMINAMATH_CALUDE_direct_proportion_exponent_l1633_163399

theorem direct_proportion_exponent (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -2 * x^(m-2) = k * x) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_exponent_l1633_163399


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l1633_163376

/-- The hyperbola defined by (x^2/9) - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line defined by y = (1/3)(x+1) -/
def line (x y : ℝ) : Prop := y = (1/3)*(x+1)

/-- The number of intersection points between the hyperbola and the line -/
def intersection_count : ℕ := 1

theorem hyperbola_line_intersection :
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l1633_163376
