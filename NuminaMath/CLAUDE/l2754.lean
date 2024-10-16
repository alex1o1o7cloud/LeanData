import Mathlib

namespace NUMINAMATH_CALUDE_age_sum_theorem_l2754_275465

theorem age_sum_theorem (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2050 → 
  a + b + c = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l2754_275465


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_l2754_275463

/-- Given a bag with 5 balls where the probability of drawing 2 white balls out of 2 draws is 3/10,
    prove that the probability of getting at least 1 white ball in 2 draws is 9/10. -/
theorem prob_at_least_one_white (total_balls : ℕ) (prob_two_white : ℚ) :
  total_balls = 5 →
  prob_two_white = 3 / 10 →
  (∃ white_balls : ℕ, white_balls ≤ total_balls ∧
    prob_two_white = (white_balls.choose 2 : ℚ) / (total_balls.choose 2 : ℚ)) →
  (1 : ℚ) - ((total_balls - white_balls).choose 2 : ℚ) / (total_balls.choose 2 : ℚ) = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_l2754_275463


namespace NUMINAMATH_CALUDE_solve_animal_videos_problem_l2754_275407

def animal_videos_problem (cat_video_length : ℕ) : Prop :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  let elephant_video_length := cat_video_length + dog_video_length + gorilla_video_length
  let dolphin_video_length := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length
  let total_time := cat_video_length + dog_video_length + gorilla_video_length + elephant_video_length + dolphin_video_length
  (cat_video_length = 4) → (total_time = 144)

theorem solve_animal_videos_problem :
  animal_videos_problem 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_animal_videos_problem_l2754_275407


namespace NUMINAMATH_CALUDE_days_to_build_floor_l2754_275436

-- Define the daily pay rate for a builder
def builder_daily_pay : ℕ := 100

-- Define the total cost for the project
def total_project_cost : ℕ := 270000

-- Define the number of builders for the project
def project_builders : ℕ := 6

-- Define the number of houses in the project
def project_houses : ℕ := 5

-- Define the number of floors per house in the project
def floors_per_house : ℕ := 6

-- Theorem to prove
theorem days_to_build_floor (builders : ℕ) (days : ℕ) : 
  builders = 3 → days = 30 → 
  (builders * builder_daily_pay * days = 
   total_project_cost * builders / project_builders / 
   (project_houses * floors_per_house)) := by sorry

end NUMINAMATH_CALUDE_days_to_build_floor_l2754_275436


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l2754_275449

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 1 < 0 ∧ |x| < 2}
  S = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l2754_275449


namespace NUMINAMATH_CALUDE_tim_balloon_count_l2754_275467

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim has more balloons than Dan -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

theorem tim_balloon_count : tim_balloons = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l2754_275467


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2754_275420

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2754_275420


namespace NUMINAMATH_CALUDE_t_square_four_equal_parts_l2754_275444

/-- A figure composed of three equal squares -/
structure TSquareFigure where
  square_area : ℝ
  total_area : ℝ
  h_total_area : total_area = 3 * square_area

/-- A division of the figure into four parts -/
structure FourPartDivision (fig : TSquareFigure) where
  part_area : ℝ
  h_part_count : ℕ
  h_part_count_eq : h_part_count = 4
  h_total_area : fig.total_area = h_part_count * part_area

/-- Theorem stating that a T-square figure can be divided into four equal parts -/
theorem t_square_four_equal_parts (fig : TSquareFigure) : 
  ∃ (div : FourPartDivision fig), div.part_area = 3 * fig.square_area / 4 := by
  sorry

end NUMINAMATH_CALUDE_t_square_four_equal_parts_l2754_275444


namespace NUMINAMATH_CALUDE_problem_solution_l2754_275409

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) ↔ (1 ≤ m ∧ m ≤ 5)) ∧
  (∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) → m₀ ≤ m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 1 →
    a^2 + b^2 + c^2 ≥ 1/50 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
      3*a₀ + 4*b₀ + 5*c₀ = 1 ∧ a₀^2 + b₀^2 + c₀^2 = 1/50) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2754_275409


namespace NUMINAMATH_CALUDE_vector_properties_and_projection_l2754_275460

/-- Given vectors in ℝ², prove properties about their relationships and projections -/
theorem vector_properties_and_projection :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (x, 3)
  let c : ℝ × ℝ := (5, y)
  let d : ℝ × ℝ := (8, 6)

  -- b is parallel to d
  (b.2 / b.1 = d.2 / d.1) →
  -- 4a + d is perpendicular to c
  ((4 * a.1 + d.1) * c.1 + (4 * a.2 + d.2) * c.2 = 0) →

  -- Prove that b and c have specific values
  (b = (4, 3) ∧ c = (5, -2)) ∧
  -- Prove that the projection of c onto a is -7√2/2
  (let proj := (a.1 * c.1 + a.2 * c.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = -7 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_and_projection_l2754_275460


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l2754_275464

/-- Circle C₁ in the Cartesian plane -/
def C₁ (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - (4*m+6)*y - 4 = 0

/-- Circle C₂ in the Cartesian plane -/
def C₂ (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = (x + 2)^2 + (y - 3)^2

/-- The theorem stating the value of m for the given conditions -/
theorem circle_intersection_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  C₁ m x₁ y₁ ∧ C₁ m x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
  x₁^2 - x₂^2 = y₂^2 - y₁^2 →
  m = -6 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l2754_275464


namespace NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l2754_275416

/-- Two vectors are collinear if their cross product is zero -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Vector a is defined as (1, m-1) -/
def a (m : ℝ) : ℝ × ℝ := (1, m - 1)

/-- Vector b is defined as (m, 2) -/
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Theorem stating that m = 2 is a sufficient but not necessary condition for collinearity -/
theorem m_eq_2_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → are_collinear (a m) (b m)) ∧
  ¬(∀ m : ℝ, are_collinear (a m) (b m) → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l2754_275416


namespace NUMINAMATH_CALUDE_line_equation_and_sum_l2754_275418

/-- Given a line with slope 5 passing through (-2, 4), prove its equation and m + b value -/
theorem line_equation_and_sum (m b : ℝ) : 
  m = 5 → -- Given slope
  4 = m * (-2) + b → -- Point (-2, 4) lies on the line
  (∀ x y, y = m * x + b ↔ y - 4 = m * (x + 2)) → -- Line equation
  m + b = 19 := by
sorry

end NUMINAMATH_CALUDE_line_equation_and_sum_l2754_275418


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l2754_275425

theorem max_min_x_plus_y :
  ∀ x y : ℝ,
  x + y = Real.sqrt (2 * x - 1) + Real.sqrt (4 * y + 3) →
  (x + y ≤ 3 + Real.sqrt (21 / 2)) ∧
  (x + y ≥ 1 + Real.sqrt (3 / 2)) ∧
  (∃ x₁ y₁ : ℝ, x₁ + y₁ = Real.sqrt (2 * x₁ - 1) + Real.sqrt (4 * y₁ + 3) ∧ x₁ + y₁ = 3 + Real.sqrt (21 / 2)) ∧
  (∃ x₂ y₂ : ℝ, x₂ + y₂ = Real.sqrt (2 * x₂ - 1) + Real.sqrt (4 * y₂ + 3) ∧ x₂ + y₂ = 1 + Real.sqrt (3 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_max_min_x_plus_y_l2754_275425


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2754_275497

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 108 → x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2754_275497


namespace NUMINAMATH_CALUDE_monkey_travel_distance_l2754_275473

/-- Represents the speed and time of movement for a monkey --/
structure MonkeyMovement where
  swingingSpeed : ℝ
  runningSpeed : ℝ
  runningTime : ℝ
  swingingTime : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (m : MonkeyMovement) : ℝ :=
  m.runningSpeed * m.runningTime + m.swingingSpeed * m.swingingTime

/-- Theorem stating the total distance traveled by the monkey --/
theorem monkey_travel_distance :
  ∀ (m : MonkeyMovement),
  m.swingingSpeed = 10 ∧
  m.runningSpeed = 15 ∧
  m.runningTime = 5 ∧
  m.swingingTime = 10 →
  totalDistance m = 175 := by
  sorry

end NUMINAMATH_CALUDE_monkey_travel_distance_l2754_275473


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2754_275442

/-- The value of a repeating decimal 0.abcabcabc... where a, b, c are digits -/
def repeating_decimal (a b c : Nat) : ℚ := (100 * a + 10 * b + c : ℚ) / 999

/-- Theorem stating that 0.246246246... - 0.135135135... - 0.012012012... = 1/9 -/
theorem repeating_decimal_subtraction :
  repeating_decimal 2 4 6 - repeating_decimal 1 3 5 - repeating_decimal 0 1 2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2754_275442


namespace NUMINAMATH_CALUDE_min_value_of_function_l2754_275429

theorem min_value_of_function (x : ℝ) : |3 - x| + |x - 7| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2754_275429


namespace NUMINAMATH_CALUDE_johns_house_wall_planks_l2754_275456

/-- The number of planks needed for a house wall --/
def total_planks (large_planks small_planks : ℕ) : ℕ :=
  large_planks + small_planks

/-- Theorem stating the total number of planks needed for John's house wall --/
theorem johns_house_wall_planks :
  total_planks 37 42 = 79 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_wall_planks_l2754_275456


namespace NUMINAMATH_CALUDE_function_equality_implies_m_value_l2754_275458

-- Define the functions f and g
def f (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + m
def g (m : ℚ) (x : ℚ) : ℚ := x^2 - 3*x + 5*m

-- State the theorem
theorem function_equality_implies_m_value :
  ∀ m : ℚ, 3 * (f m 5) = 2 * (g m 5) → m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_value_l2754_275458


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l2754_275462

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | Real.sin x < a} = {x : ℝ | ∃ n : ℤ, (2 * n - 1) * π - θ < x ∧ x < 2 * n * π + θ} := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l2754_275462


namespace NUMINAMATH_CALUDE_simplify_expression_l2754_275419

theorem simplify_expression : 
  (9 * 10^12) / (3 * 10^4) + (2 * 10^8) / (4 * 10^2) = 300500000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2754_275419


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2754_275433

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → 
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2754_275433


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l2754_275471

/-- Proves that a car can travel approximately 56.01 kilometers on a liter of fuel given specific conditions. -/
theorem car_fuel_efficiency (travel_time : Real) (fuel_used_gallons : Real) (speed_mph : Real)
    (gallons_to_liters : Real) (miles_to_km : Real)
    (h1 : travel_time = 5.7)
    (h2 : fuel_used_gallons = 3.9)
    (h3 : speed_mph = 91)
    (h4 : gallons_to_liters = 3.8)
    (h5 : miles_to_km = 1.6) :
    ∃ km_per_liter : Real, abs (km_per_liter - 56.01) < 0.01 ∧
    km_per_liter = (speed_mph * travel_time * miles_to_km) / (fuel_used_gallons * gallons_to_liters) :=
by
  sorry


end NUMINAMATH_CALUDE_car_fuel_efficiency_l2754_275471


namespace NUMINAMATH_CALUDE_four_integer_problem_l2754_275454

def satisfies_condition (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (b * c * d) % a = 1 ∧
  (a * c * d) % b = 1 ∧
  (a * b * d) % c = 1 ∧
  (a * b * c) % d = 1

theorem four_integer_problem :
  (∀ a b c d : ℕ, satisfies_condition a b c d →
    ((a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13))) ∧
  (satisfies_condition 2 3 7 11) ∧
  (satisfies_condition 2 3 11 13) :=
sorry

end NUMINAMATH_CALUDE_four_integer_problem_l2754_275454


namespace NUMINAMATH_CALUDE_L8_2_7_exponent_is_columns_l2754_275441

/-- Represents an orthogonal array -/
structure OrthogonalArray where
  experiments : ℕ
  levels : ℕ
  columns : ℕ

/-- The specific orthogonal array L₈(2⁷) -/
def L8_2_7 : OrthogonalArray :=
  { experiments := 8
  , levels := 2
  , columns := 7 }

theorem L8_2_7_exponent_is_columns : L8_2_7.columns = 7 := by
  sorry

end NUMINAMATH_CALUDE_L8_2_7_exponent_is_columns_l2754_275441


namespace NUMINAMATH_CALUDE_blake_bought_six_chocolate_packs_l2754_275410

/-- The number of lollipops Blake bought -/
def lollipops : ℕ := 4

/-- The cost of one lollipop in dollars -/
def lollipop_cost : ℕ := 2

/-- The number of $10 bills Blake gave to the cashier -/
def bills_given : ℕ := 6

/-- The amount of change Blake received in dollars -/
def change_received : ℕ := 4

/-- The cost of one pack of chocolate in terms of lollipops -/
def chocolate_pack_cost : ℕ := 4 * lollipop_cost

/-- The total amount Blake spent in dollars -/
def total_spent : ℕ := bills_given * 10 - change_received

/-- Theorem stating that Blake bought 6 packs of chocolate -/
theorem blake_bought_six_chocolate_packs : 
  (total_spent - lollipops * lollipop_cost) / chocolate_pack_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_blake_bought_six_chocolate_packs_l2754_275410


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2754_275450

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -3 →
  (6 * x - 3) / (x^2 - 9*x - 36) = (23/5) / (x - 12) + (7/5) / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2754_275450


namespace NUMINAMATH_CALUDE_min_grid_size_l2754_275486

theorem min_grid_size (k : ℝ) (h : k > 0.9999) :
  (∃ n : ℕ, n ≥ 51 ∧ 4 * n * (n - 1) * k * (1 - k) = k) ∧
  (∀ m : ℕ, m < 51 → 4 * m * (m - 1) * k * (1 - k) ≠ k) := by
  sorry

end NUMINAMATH_CALUDE_min_grid_size_l2754_275486


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2754_275482

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem to be proved -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 5 + a 6 = 4) →
  (a 15 + a 16 = 16) →
  (a 25 + a 26 = 64) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2754_275482


namespace NUMINAMATH_CALUDE_solution_to_system_l2754_275478

theorem solution_to_system (x y : ℝ) : 
  x + y = 3 ∧ x^5 + y^5 = 33 → (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l2754_275478


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2754_275408

theorem complex_modulus_problem (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*a*i)*i = 1 - b*i) : 
  Complex.abs (a + b*i) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2754_275408


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2754_275424

/-- Given a parabola y = 2x^2 - 4x - 5 translated 3 units left and 2 units up to obtain parabola C,
    the equation of the parabola symmetric to C about the y-axis is y = 2x^2 - 8x + 3 -/
theorem parabola_symmetry (x y : ℝ) :
  let original := fun x => 2 * x^2 - 4 * x - 5
  let translated := fun x => original (x + 3) + 2
  let symmetric := fun x => translated (-x)
  symmetric x = 2 * x^2 - 8 * x + 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2754_275424


namespace NUMINAMATH_CALUDE_dot_product_result_l2754_275480

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_result : (2 • a + b) • a = 1 := by sorry

end NUMINAMATH_CALUDE_dot_product_result_l2754_275480


namespace NUMINAMATH_CALUDE_correct_observation_value_l2754_275461

theorem correct_observation_value
  (n : ℕ)
  (original_mean : ℚ)
  (incorrect_value : ℚ)
  (corrected_mean : ℚ)
  (h1 : n = 50)
  (h2 : original_mean = 36)
  (h3 : incorrect_value = 23)
  (h4 : corrected_mean = 36.5) :
  (n : ℚ) * corrected_mean - ((n : ℚ) * original_mean - incorrect_value) = 48 :=
by sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2754_275461


namespace NUMINAMATH_CALUDE_watchman_max_demand_l2754_275423

/-- The amount of the bet made by the trespasser with his friends -/
def bet_amount : ℕ := 100

/-- The trespasser's net loss if he pays the watchman -/
def net_loss_if_pay (amount : ℕ) : ℤ := amount - bet_amount

/-- The trespasser's net loss if he doesn't pay the watchman -/
def net_loss_if_not_pay : ℕ := bet_amount

/-- Predicate to determine if the trespasser will pay for a given amount -/
def will_pay (amount : ℕ) : Prop :=
  net_loss_if_pay amount < net_loss_if_not_pay

/-- The maximum amount the watchman can demand -/
def max_demand : ℕ := 199

theorem watchman_max_demand :
  (∀ n : ℕ, n ≤ max_demand → will_pay n) ∧
  (∀ n : ℕ, n > max_demand → ¬will_pay n) :=
sorry

end NUMINAMATH_CALUDE_watchman_max_demand_l2754_275423


namespace NUMINAMATH_CALUDE_sum_mod_9_equals_5_l2754_275481

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Given a list of natural numbers, compute the sum of their modulo 9 values
    after reducing each number to the sum of its digits -/
def sum_mod_9_of_digit_sums (numbers : List ℕ) : ℕ :=
  (numbers.map (fun n => sum_of_digits n % 9)).sum % 9

/-- The main theorem stating that the sum of modulo 9 values of the given numbers
    after reducing each to the sum of its digits is 5 -/
theorem sum_mod_9_equals_5 :
  sum_mod_9_of_digit_sums [1, 21, 333, 4444, 55555, 666666, 7777777, 88888888, 999999999] = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_9_equals_5_l2754_275481


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2754_275470

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2754_275470


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2754_275437

theorem quadratic_factorization (a : ℝ) : a^2 - 6*a + 9 = (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2754_275437


namespace NUMINAMATH_CALUDE_lottery_winning_numbers_l2754_275499

/-- Calculates the number of winning numbers on each lottery ticket -/
theorem lottery_winning_numbers
  (num_tickets : ℕ)
  (winning_number_value : ℕ)
  (total_amount_won : ℕ)
  (h1 : num_tickets = 3)
  (h2 : winning_number_value = 20)
  (h3 : total_amount_won = 300)
  (h4 : total_amount_won % winning_number_value = 0)
  (h5 : (total_amount_won / winning_number_value) % num_tickets = 0) :
  total_amount_won / winning_number_value / num_tickets = 5 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winning_numbers_l2754_275499


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2754_275494

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2754_275494


namespace NUMINAMATH_CALUDE_fraction_power_calculation_l2754_275415

theorem fraction_power_calculation (x y : ℚ) 
  (hx : x = 2/3) (hy : y = 3/2) : 
  (3/4) * x^8 * y^9 = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_calculation_l2754_275415


namespace NUMINAMATH_CALUDE_birth_year_age_proof_l2754_275498

/-- Given a birth year Y between 1900 and 1987 (inclusive), 
    if the person's age in 1988 equals the product of the last two digits of Y,
    then Y must be 1964 and the person's age must be 24. -/
theorem birth_year_age_proof (Y : ℕ) 
  (h1 : 1900 ≤ Y) (h2 : Y < 1988) :
  (1988 - Y = (Y % 100 / 10) * (Y % 10)) → (Y = 1964 ∧ 1988 - Y = 24) :=
by sorry

end NUMINAMATH_CALUDE_birth_year_age_proof_l2754_275498


namespace NUMINAMATH_CALUDE_sin_cos_product_zero_l2754_275459

theorem sin_cos_product_zero (θ : Real) (h : Real.sin θ + Real.cos θ = -1) : 
  Real.sin θ * Real.cos θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_zero_l2754_275459


namespace NUMINAMATH_CALUDE_vector_scalar_properties_l2754_275488

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_scalar_properties :
  (∀ (m : ℝ) (a b : V), m • (a - b) = m • a - m • b) ∧
  (∀ (m n : ℝ) (a : V), (m - n) • a = m • a - n • a) ∧
  (∃ (m : ℝ) (a b : V), m • a = m • b ∧ a ≠ b) ∧
  (∀ (m n : ℝ) (a : V), a ≠ 0 → m • a = n • a → m = n) :=
by sorry

end NUMINAMATH_CALUDE_vector_scalar_properties_l2754_275488


namespace NUMINAMATH_CALUDE_unique_point_on_circle_l2754_275451

-- Define the points A and B
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (2, 1)

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - 2)^2 = 16

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem unique_point_on_circle (a : ℝ) : 
  (∃! P : ℝ × ℝ, C a P.1 P.2 ∧ distanceSquared P A + 2 * distanceSquared P B = 24) →
  a = -1 ∨ a = 3 := by
sorry


end NUMINAMATH_CALUDE_unique_point_on_circle_l2754_275451


namespace NUMINAMATH_CALUDE_equation_solution_range_l2754_275452

theorem equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (2*x + m)/(x - 2) + (x - 1)/(2 - x) = 3) →
  (m > -7 ∧ m ≠ -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2754_275452


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2754_275468

/-- The sum of the repeating decimals 0.666... and 0.333... is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (10 * x - x = 6 ∧ 10 * y - y = 3) → x + y = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2754_275468


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l2754_275404

/-- Calculates the daily fine for absence given the contract parameters -/
def calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let earned_amount := daily_pay * worked_days
  (earned_amount - total_payment) / absent_days

/-- Proves that the daily fine is 7.5 given the contract parameters -/
theorem contractor_fine_calculation :
  calculate_daily_fine 30 25 490 8 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l2754_275404


namespace NUMINAMATH_CALUDE_oil_distance_theorem_l2754_275477

/-- Represents the relationship between remaining oil and distance traveled --/
def oil_distance_relation (x : ℝ) : ℝ := 62 - 0.12 * x

theorem oil_distance_theorem :
  let initial_oil : ℝ := 62
  let data_points : List (ℝ × ℝ) := [(100, 50), (200, 38), (300, 26), (400, 14)]
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = oil_distance_relation x :=
by sorry

end NUMINAMATH_CALUDE_oil_distance_theorem_l2754_275477


namespace NUMINAMATH_CALUDE_special_function_form_l2754_275430

/-- A bijective, monotonic function from ℝ to ℝ satisfying f(t) + f⁻¹(t) = 2t for all t ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧ 
  Monotone f ∧ 
  ∀ t : ℝ, f t + Function.invFun f t = 2 * t

/-- The theorem stating that any SpecialFunction is of the form f(x) = x + c for some constant c -/
theorem special_function_form (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := by sorry

end NUMINAMATH_CALUDE_special_function_form_l2754_275430


namespace NUMINAMATH_CALUDE_percentage_of_number_l2754_275446

theorem percentage_of_number (percentage : ℝ) (number : ℝ) (result : ℝ) :
  percentage = 110 ∧ number = 500 ∧ result = 550 →
  (percentage / 100) * number = result := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l2754_275446


namespace NUMINAMATH_CALUDE_least_n_mod_1000_l2754_275483

/-- Sum of digits in base 4 representation -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ :=
  sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ :=
  sorry

theorem least_n_mod_1000 : N % 1000 = 151 := by
  sorry

end NUMINAMATH_CALUDE_least_n_mod_1000_l2754_275483


namespace NUMINAMATH_CALUDE_automotive_test_distance_l2754_275403

theorem automotive_test_distance (d : ℝ) (h1 : d / 4 + d / 5 + d / 6 = 37) : 3 * d = 180 := by
  sorry

#check automotive_test_distance

end NUMINAMATH_CALUDE_automotive_test_distance_l2754_275403


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l2754_275484

/-- The number of ways to arrange 5 students in a row with specific constraints -/
def arrangement_count : ℕ := 36

/-- A function that calculates the number of valid arrangements -/
def calculate_arrangements : ℕ :=
  let total_students : ℕ := 5
  let ab_pair_arrangements : ℕ := 3 * 2  -- 3! for AB pair and 2 others, 2! for AB swap
  let c_placement_options : ℕ := 3       -- C always has 3 valid positions
  ab_pair_arrangements * c_placement_options

/-- Theorem stating that the number of valid arrangements is 36 -/
theorem arrangement_count_is_correct : arrangement_count = calculate_arrangements := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l2754_275484


namespace NUMINAMATH_CALUDE_lunch_group_size_l2754_275453

/-- Proves that the number of people in the lunch group is 7 given the conditions of the problem -/
theorem lunch_group_size :
  ∀ (x : ℕ),
  (x > 2) →
  (175 / (x - 2) : ℚ) - (175 / x : ℚ) = 10 →
  x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_group_size_l2754_275453


namespace NUMINAMATH_CALUDE_fraction_problem_l2754_275422

theorem fraction_problem (x : ℚ) :
  (x / (2 * x + 11) = 3 / 4) → x = -33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2754_275422


namespace NUMINAMATH_CALUDE_kims_earrings_l2754_275402

/-- Proves that Kim brings 5 pairs of earrings on the third day to have enough gumballs for 42 days -/
theorem kims_earrings (gumballs_per_pair : ℕ) (day1_pairs : ℕ) (daily_consumption : ℕ) (total_days : ℕ) :
  gumballs_per_pair = 9 →
  day1_pairs = 3 →
  daily_consumption = 3 →
  total_days = 42 →
  let day2_pairs := 2 * day1_pairs
  let day3_pairs := day2_pairs - 1
  let total_gumballs := gumballs_per_pair * (day1_pairs + day2_pairs + day3_pairs)
  total_gumballs = daily_consumption * total_days →
  day3_pairs = 5 := by
sorry


end NUMINAMATH_CALUDE_kims_earrings_l2754_275402


namespace NUMINAMATH_CALUDE_characterize_g_l2754_275427

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_g_l2754_275427


namespace NUMINAMATH_CALUDE_xyz_value_l2754_275496

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2754_275496


namespace NUMINAMATH_CALUDE_closest_fraction_l2754_275421

def actual_fraction : ℚ := 24 / 150

def candidate_fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (closest : ℚ), closest ∈ candidate_fractions ∧
  (∀ (f : ℚ), f ∈ candidate_fractions → |f - actual_fraction| ≥ |closest - actual_fraction|) ∧
  closest = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_l2754_275421


namespace NUMINAMATH_CALUDE_special_sequence_has_repeats_l2754_275400

/-- A sequence of rational numbers satisfying the given property -/
def SpecialSequence := ℕ → ℚ

/-- The property that defines our special sequence -/
def HasSpecialProperty (a : SpecialSequence) : Prop :=
  ∀ m n : ℕ, a m + a n = a (m * n)

/-- The theorem stating that a sequence with the special property has repeated elements -/
theorem special_sequence_has_repeats (a : SpecialSequence) (h : HasSpecialProperty a) :
  ∃ i j : ℕ, i ≠ j ∧ a i = a j := by sorry

end NUMINAMATH_CALUDE_special_sequence_has_repeats_l2754_275400


namespace NUMINAMATH_CALUDE_kola_solution_water_percentage_l2754_275435

/-- Proves that the initial percentage of water in a kola solution is 88% given specific conditions --/
theorem kola_solution_water_percentage :
  ∀ (W S : ℝ),
  -- Initial volume
  let initial_volume : ℝ := 340
  -- Initial percentages sum to 100%
  W + S + 5 = 100 →
  -- New volume after additions
  let new_volume : ℝ := initial_volume + 3.2 + 10 + 6.8
  -- New sugar amount equals 7.5% of new volume
  (S / 100) * initial_volume + 3.2 = 0.075 * new_volume →
  -- Conclusion: Initial water percentage is 88%
  W = 88 := by
sorry

end NUMINAMATH_CALUDE_kola_solution_water_percentage_l2754_275435


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2754_275489

theorem opposite_of_2023 : 
  ∀ (x : ℤ), (x + 2023 = 0) → x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2754_275489


namespace NUMINAMATH_CALUDE_train_speed_on_time_l2754_275440

/-- Proves that the speed at which the train arrives on time is 84 km/h, given the conditions -/
theorem train_speed_on_time (d : ℝ) (t : ℝ) :
  (d = 80 * (t + 24/60)) →
  (d = 90 * (t - 32/60)) →
  (d / t = 84) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_on_time_l2754_275440


namespace NUMINAMATH_CALUDE_basketball_court_length_l2754_275476

theorem basketball_court_length :
  ∀ (width length : ℝ),
  length = width + 14 →
  2 * length + 2 * width = 96 →
  length = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_court_length_l2754_275476


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2754_275485

theorem sum_of_cubes_zero (x y : ℝ) (h1 : x + y = 0) (h2 : x * y = -1) : x^3 + y^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2754_275485


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l2754_275447

theorem fahrenheit_celsius_conversion (C F : ℚ) : 
  C = (4/7) * (F - 32) → C = 35 → F = 93.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l2754_275447


namespace NUMINAMATH_CALUDE_divides_power_minus_constant_l2754_275431

theorem divides_power_minus_constant (n : ℕ) : 13 ∣ 14^n - 27 := by
  sorry

end NUMINAMATH_CALUDE_divides_power_minus_constant_l2754_275431


namespace NUMINAMATH_CALUDE_circle_center_range_l2754_275466

theorem circle_center_range (a : ℝ) : 
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 - (a-2))^2 = 9}
  let M : ℝ × ℝ := (0, 3)
  (3, -2) ∈ C ∧ (0, -5) ∈ C ∧ 
  (∃ N ∈ C, (N.1 - M.1)^2 + (N.2 - M.2)^2 = 4 * ((N.1 - a)^2 + (N.2 - (a-2))^2)) →
  (-3 ≤ a ∧ a ≤ 0) ∨ (1 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_circle_center_range_l2754_275466


namespace NUMINAMATH_CALUDE_layla_babysitting_earnings_l2754_275491

theorem layla_babysitting_earnings :
  let donaldson_rate : ℕ := 15
  let merck_rate : ℕ := 18
  let hille_rate : ℕ := 20
  let johnson_rate : ℕ := 22
  let ramos_rate : ℕ := 25
  let donaldson_hours : ℕ := 7
  let merck_hours : ℕ := 6
  let hille_hours : ℕ := 3
  let johnson_hours : ℕ := 4
  let ramos_hours : ℕ := 2
  donaldson_rate * donaldson_hours +
  merck_rate * merck_hours +
  hille_rate * hille_hours +
  johnson_rate * johnson_hours +
  ramos_rate * ramos_hours = 411 :=
by sorry

end NUMINAMATH_CALUDE_layla_babysitting_earnings_l2754_275491


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2754_275417

/-- Given two lines l₁ and l₂, prove that if they are perpendicular, then a = 0 or a = 5/3 -/
theorem perpendicular_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | a * x + 3 * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + (a^2 - a) * y + 3 = 0}
  let perpendicular := ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (x₂ - x₁) * (a * (x₂ - x₁) + 3 * (y₂ - y₁)) + 
    (y₂ - y₁) * (2 * (x₂ - x₁) + (a^2 - a) * (y₂ - y₁)) = 0
  perpendicular → a = 0 ∨ a = 5/3 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2754_275417


namespace NUMINAMATH_CALUDE_smallest_k_for_four_color_rectangle_l2754_275474

/-- Represents a coloring of an n × n board -/
def Coloring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Predicate that checks if four cells form a rectangle with different colors -/
def hasFourColorRectangle (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r1 r2 c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    c r1 c1 ≠ c r1 c2 ∧ c r1 c1 ≠ c r2 c1 ∧ c r1 c1 ≠ c r2 c2 ∧
    c r1 c2 ≠ c r2 c1 ∧ c r1 c2 ≠ c r2 c2 ∧
    c r2 c1 ≠ c r2 c2

/-- Main theorem stating the smallest k that guarantees a four-color rectangle -/
theorem smallest_k_for_four_color_rectangle (n : ℕ) (h : n ≥ 2) :
  (∀ k : ℕ, k ≥ 2*n → ∀ c : Coloring n k, hasFourColorRectangle n k c) ∧
  (∃ c : Coloring n (2*n - 1), ¬hasFourColorRectangle n (2*n - 1) c) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_four_color_rectangle_l2754_275474


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2754_275472

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (r s t : ℂ) 
  (ha : a ≠ 0) (hd : d ≠ 0)
  (h_cubic : ∀ x : ℂ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) :
  1 / r^2 + 1 / s^2 + 1 / t^2 = (c^2 - 2 * b * d) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l2754_275472


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2754_275432

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2754_275432


namespace NUMINAMATH_CALUDE_scale_length_l2754_275493

-- Define the number of parts in the scale
def num_parts : ℕ := 5

-- Define the length of each part in inches
def part_length : ℕ := 16

-- Theorem stating the total length of the scale
theorem scale_length : num_parts * part_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_l2754_275493


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2754_275469

/-- The repeating decimal 4.252525... -/
def repeating_decimal : ℚ := 4 + 25 / 99

/-- The fraction 421/99 -/
def target_fraction : ℚ := 421 / 99

/-- Theorem stating that the repeating decimal 4.252525... is equal to 421/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2754_275469


namespace NUMINAMATH_CALUDE_sequence_general_term_l2754_275406

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 6) 
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5^n) : 
  ∀ n : ℕ, n ≥ 1 → a n = 5^n + 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2754_275406


namespace NUMINAMATH_CALUDE_sara_pumpkins_l2754_275479

/-- The number of pumpkins Sara originally grew -/
def original_pumpkins : ℕ := sorry

/-- The number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := 23

/-- The number of pumpkins remaining -/
def remaining_pumpkins : ℕ := 20

/-- Theorem stating that Sara originally grew 43 pumpkins -/
theorem sara_pumpkins : original_pumpkins = eaten_pumpkins + remaining_pumpkins :=
by sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l2754_275479


namespace NUMINAMATH_CALUDE_jason_egg_consumption_l2754_275495

/-- The number of eggs Jason uses for one omelet -/
def eggs_per_omelet : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks we're considering -/
def weeks_considered : ℕ := 2

/-- Theorem: Jason consumes 42 eggs in two weeks -/
theorem jason_egg_consumption :
  eggs_per_omelet * days_per_week * weeks_considered = 42 := by
  sorry

end NUMINAMATH_CALUDE_jason_egg_consumption_l2754_275495


namespace NUMINAMATH_CALUDE_collins_savings_l2754_275455

/-- The amount earned per aluminum can in dollars -/
def earnings_per_can : ℚ := 25 / 100

/-- The number of cans found at home -/
def cans_at_home : ℕ := 12

/-- The number of cans found at grandparents' house -/
def cans_at_grandparents : ℕ := 3 * cans_at_home

/-- The number of cans given by the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans brought by dad from the office -/
def cans_from_dad : ℕ := 250

/-- The total number of cans collected -/
def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_dad

/-- The total earnings from recycling all cans -/
def total_earnings : ℚ := earnings_per_can * total_cans

/-- The amount Collin needs to put into savings -/
def savings_amount : ℚ := total_earnings / 2

/-- Theorem stating that the amount Collin needs to put into savings is $43.00 -/
theorem collins_savings : savings_amount = 43 := by sorry

end NUMINAMATH_CALUDE_collins_savings_l2754_275455


namespace NUMINAMATH_CALUDE_students_not_in_biology_l2754_275475

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880)
  (h2 : biology_percentage = 325 / 1000) :
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l2754_275475


namespace NUMINAMATH_CALUDE_mary_seth_age_difference_l2754_275438

/-- Represents the age difference between Mary and Seth -/
def age_difference : ℝ → ℝ → ℝ := λ m s => m - s

/-- Mary's age after one year will be three times Seth's age after one year -/
def future_age_relation (m : ℝ) (s : ℝ) : Prop := m + 1 = 3 * (s + 1)

theorem mary_seth_age_difference :
  ∀ (m s : ℝ),
  m > s →
  future_age_relation m s →
  m + s = 3.5 →
  age_difference m s = 2.75 := by
sorry

end NUMINAMATH_CALUDE_mary_seth_age_difference_l2754_275438


namespace NUMINAMATH_CALUDE_benny_missed_games_l2754_275411

/-- The number of baseball games Benny missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Benny missed 25 games -/
theorem benny_missed_games :
  let total_games : ℕ := 39
  let attended_games : ℕ := 14
  games_missed total_games attended_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_benny_missed_games_l2754_275411


namespace NUMINAMATH_CALUDE_ultratown_block_perimeter_difference_l2754_275487

/-- Represents a rectangular city block with surrounding streets -/
structure CityBlock where
  length : ℝ
  width : ℝ
  street_width : ℝ

/-- Calculates the difference between outer and inner perimeters of a city block -/
def perimeter_difference (block : CityBlock) : ℝ :=
  2 * ((block.length + 2 * block.street_width) + (block.width + 2 * block.street_width)) -
  2 * (block.length + block.width)

/-- Theorem: The difference between outer and inner perimeters of the specified block is 200 feet -/
theorem ultratown_block_perimeter_difference :
  let block : CityBlock := {
    length := 500,
    width := 300,
    street_width := 25
  }
  perimeter_difference block = 200 := by
  sorry

end NUMINAMATH_CALUDE_ultratown_block_perimeter_difference_l2754_275487


namespace NUMINAMATH_CALUDE_smallest_sum_mn_l2754_275443

theorem smallest_sum_mn (m n : ℕ) (hm : m > n) (h_div : (70^2 : ℕ) ∣ (2023^m - 2023^n)) : m + n ≥ 24 ∧ ∃ (m₀ n₀ : ℕ), m₀ + n₀ = 24 ∧ m₀ > n₀ ∧ (70^2 : ℕ) ∣ (2023^m₀ - 2023^n₀) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_mn_l2754_275443


namespace NUMINAMATH_CALUDE_lunks_needed_for_bananas_l2754_275445

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 7

/-- Exchange rate between kunks and bananas -/
def kunks_to_bananas_rate : ℚ := 5 / 3

/-- Number of bananas to purchase -/
def bananas_to_buy : ℕ := 20

/-- Theorem stating the number of lunks needed to buy the specified number of bananas -/
theorem lunks_needed_for_bananas : 
  ⌈(bananas_to_buy : ℚ) / (kunks_to_bananas_rate * lunks_to_kunks_rate)⌉ = 21 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_bananas_l2754_275445


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2754_275426

/-- Proves that given a simple interest of 4016.25, an interest rate of 10% per annum,
    and a time period of 5 years, the principal sum is 8032.5. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4016.25
  let rate : ℝ := 10  -- 10% per annum
  let time : ℝ := 5   -- 5 years
  let principal : ℝ := simple_interest * 100 / (rate * time)
  principal = 8032.5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2754_275426


namespace NUMINAMATH_CALUDE_cube_edge_sum_l2754_275414

theorem cube_edge_sum (surface_area : ℝ) (edge_sum : ℝ) : 
  surface_area = 150 → edge_sum = 12 * (surface_area / 6).sqrt → edge_sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l2754_275414


namespace NUMINAMATH_CALUDE_teds_age_l2754_275405

theorem teds_age (t s : ℝ) 
  (h1 : t = 3 * s - 10)  -- Ted's age is 10 years less than three times Sally's age
  (h2 : t + s = 60)      -- The sum of their ages is 60
  : t = 42.5 :=          -- Ted's age is 42.5
by sorry

end NUMINAMATH_CALUDE_teds_age_l2754_275405


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l2754_275490

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ i j, i ≠ j → digits.get i ≠ digits.get j) ∧
  (0 ∉ digits) ∧
  (digits.sum = 18)

theorem largest_number_with_conditions :
  ∀ n : ℕ, is_valid_number n → n ≤ 6543 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l2754_275490


namespace NUMINAMATH_CALUDE_hospital_baby_probability_l2754_275492

/-- The probability of success for a single trial -/
def p : ℚ := 1/3

/-- The number of trials -/
def n : ℕ := 6

/-- The number of successes we're interested in -/
def k : ℕ := 3

/-- The probability of at least k successes in n trials with probability p -/
def prob_at_least (p : ℚ) (n k : ℕ) : ℚ :=
  1 - (Finset.range k).sum (λ i => Nat.choose n i * p^i * (1-p)^(n-i))

theorem hospital_baby_probability :
  prob_at_least p n k = 233/729 := by sorry

end NUMINAMATH_CALUDE_hospital_baby_probability_l2754_275492


namespace NUMINAMATH_CALUDE_gym_budget_problem_l2754_275412

/-- Proves that given a budget that allows for the purchase of 10 softballs at $9 each after a 20% increase,
    the original budget would allow for the purchase of 15 dodgeballs at $5 each. -/
theorem gym_budget_problem (original_budget : ℝ) 
  (h1 : original_budget * 1.2 = 10 * 9) 
  (h2 : original_budget > 0) : 
  original_budget / 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_gym_budget_problem_l2754_275412


namespace NUMINAMATH_CALUDE_find_y_value_l2754_275434

theorem find_y_value (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (equal_after_changes : x + 10 = y - 10 ∧ y - 10 = 3 * z) : 
  y = 150 * 7/14 + 10 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l2754_275434


namespace NUMINAMATH_CALUDE_concyclic_roots_l2754_275448

theorem concyclic_roots (m : ℝ) : 
  (∀ x : ℂ, (x^2 - 2*x + 2 = 0 ∨ x^2 + 2*m*x + 1 = 0) → 
    (∃ (a b r : ℝ), ∀ y : ℂ, (y^2 - 2*y + 2 = 0 ∨ y^2 + 2*m*y + 1 = 0) → 
      (y.re - a)^2 + (y.im - b)^2 = r^2)) ↔ 
  (-1 < m ∧ m < 1) ∨ m = -3/2 := by
sorry

end NUMINAMATH_CALUDE_concyclic_roots_l2754_275448


namespace NUMINAMATH_CALUDE_undefined_expression_l2754_275457

theorem undefined_expression (x : ℝ) : 
  (x^2 - 18*x + 81 = 0) ↔ (x = 9) := by
  sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l2754_275457


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l2754_275401

theorem no_solutions_to_equation :
  ¬ ∃ x : ℝ, (2 * x^2 - 10 * x) / (x^2 - 5 * x) = x - 3 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l2754_275401


namespace NUMINAMATH_CALUDE_intersection_points_vary_at_least_one_intersection_l2754_275413

/-- The number of intersection points between y = Bx^2 and y^3 + 2 = x^2 + 4y varies with B -/
theorem intersection_points_vary (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y ∧
  ∃ (B₁ B₂ : ℝ) (hB₁ : B₁ > 0) (hB₂ : B₂ > 0),
    (∀ (x₁ y₁ : ℝ), y₁ = B₁ * x₁^2 → y₁^3 + 2 = x₁^2 + 4 * y₁ →
      ∀ (x₂ y₂ : ℝ), y₂ = B₂ * x₂^2 → y₂^3 + 2 = x₂^2 + 4 * y₂ →
        (x₁, y₁) ≠ (x₂, y₂)) :=
by
  sorry

/-- There is at least one intersection point for any positive B -/
theorem at_least_one_intersection (B : ℝ) (hB : B > 0) :
  ∃ (x y : ℝ), y = B * x^2 ∧ y^3 + 2 = x^2 + 4 * y :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_vary_at_least_one_intersection_l2754_275413


namespace NUMINAMATH_CALUDE_jen_jam_consumption_l2754_275428

theorem jen_jam_consumption (total_jam : ℚ) : 
  let lunch_consumption := (1 : ℚ) / 3
  let after_lunch := total_jam - lunch_consumption * total_jam
  let after_dinner := (4 : ℚ) / 7 * total_jam
  let dinner_consumption := (after_lunch - after_dinner) / after_lunch
  dinner_consumption = (1 : ℚ) / 7 := by sorry

end NUMINAMATH_CALUDE_jen_jam_consumption_l2754_275428


namespace NUMINAMATH_CALUDE_abcdef_hex_bits_l2754_275439

def hex_to_decimal (h : String) : ℕ := 
  match h with
  | "A" => 10
  | "B" => 11
  | "C" => 12
  | "D" => 13
  | "E" => 14
  | "F" => 15
  | _ => 0  -- This case should never be reached for valid hex digits

theorem abcdef_hex_bits : 
  let decimal : ℕ := 
    (hex_to_decimal "A") * (16^5) +
    (hex_to_decimal "B") * (16^4) +
    (hex_to_decimal "C") * (16^3) +
    (hex_to_decimal "D") * (16^2) +
    (hex_to_decimal "E") * (16^1) +
    (hex_to_decimal "F")
  ∃ n : ℕ, 2^n ≤ decimal ∧ decimal < 2^(n+1) ∧ n + 1 = 24 :=
by sorry

end NUMINAMATH_CALUDE_abcdef_hex_bits_l2754_275439
