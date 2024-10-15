import Mathlib

namespace NUMINAMATH_CALUDE_probability_two_female_volunteers_l2054_205411

/-- The probability of selecting 2 female volunteers from a group of 3 female and 2 male volunteers (5 in total) is 3/10. -/
theorem probability_two_female_volunteers :
  let total_volunteers : ℕ := 5
  let female_volunteers : ℕ := 3
  let male_volunteers : ℕ := 2
  let selected_volunteers : ℕ := 2
  let total_combinations := Nat.choose total_volunteers selected_volunteers
  let female_combinations := Nat.choose female_volunteers selected_volunteers
  (female_combinations : ℚ) / total_combinations = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_female_volunteers_l2054_205411


namespace NUMINAMATH_CALUDE_installation_solution_l2054_205429

/-- Represents the number of installations of each type -/
structure Installations where
  type1 : ℕ
  type2 : ℕ
  type3 : ℕ

/-- Checks if the given installation numbers satisfy all conditions -/
def satisfiesConditions (i : Installations) : Prop :=
  i.type1 + i.type2 + i.type3 ≥ 100 ∧
  i.type2 = 4 * i.type1 ∧
  ∃ k : ℕ, i.type3 = k * i.type1 ∧
  5 * i.type3 = i.type2 + 22

theorem installation_solution :
  ∃ i : Installations, satisfiesConditions i ∧ i.type1 = 22 ∧ i.type2 = 88 ∧ i.type3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_installation_solution_l2054_205429


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2054_205482

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function y = (x + 1)(x - a) -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - a)

/-- If f(a) is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2054_205482


namespace NUMINAMATH_CALUDE_bus_seats_columns_l2054_205407

/-- The number of rows in each bus -/
def rows : ℕ := 10

/-- The number of buses -/
def buses : ℕ := 6

/-- The total number of students that can be accommodated -/
def total_students : ℕ := 240

/-- The number of columns of seats in each bus -/
def columns : ℕ := 4

theorem bus_seats_columns :
  columns * rows * buses = total_students :=
sorry

end NUMINAMATH_CALUDE_bus_seats_columns_l2054_205407


namespace NUMINAMATH_CALUDE_max_product_price_for_given_conditions_l2054_205442

/-- Represents a company's product line -/
structure ProductLine where
  numProducts : ℕ
  averagePrice : ℝ
  minPrice : ℝ
  numLowPriced : ℕ
  lowPriceThreshold : ℝ

/-- The greatest possible selling price of the most expensive product -/
def maxProductPrice (pl : ProductLine) : ℝ :=
  sorry

/-- Theorem stating the maximum product price for the given conditions -/
theorem max_product_price_for_given_conditions :
  let pl : ProductLine := {
    numProducts := 25,
    averagePrice := 1200,
    minPrice := 400,
    numLowPriced := 10,
    lowPriceThreshold := 1000
  }
  maxProductPrice pl = 12000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_price_for_given_conditions_l2054_205442


namespace NUMINAMATH_CALUDE_quadratic_function_values_l2054_205488

theorem quadratic_function_values (p q : ℝ) : ¬ (∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l2054_205488


namespace NUMINAMATH_CALUDE_proposition_implication_l2054_205460

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l2054_205460


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2054_205456

theorem no_real_solutions_for_equation : 
  ¬ ∃ x : ℝ, (x + 4)^2 = 3*(x - 2) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l2054_205456


namespace NUMINAMATH_CALUDE_tangent_point_on_curve_l2054_205485

theorem tangent_point_on_curve (x y : ℝ) : 
  y = x^4 ∧ (4 : ℝ) * x^3 = 4 → x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_on_curve_l2054_205485


namespace NUMINAMATH_CALUDE_y_divisibility_l2054_205499

def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_divisibility :
  (∃ k : ℕ, y = 3 * k) ∧
  (∃ k : ℕ, y = 9 * k) ∧
  (∃ k : ℕ, y = 27 * k) ∧
  (∃ k : ℕ, y = 81 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l2054_205499


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2054_205465

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) :
  a^2 - 2*a*b + b^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2054_205465


namespace NUMINAMATH_CALUDE_equal_area_point_on_diagonal_l2054_205458

/-- A point inside a rectangle where lines through it parallel to the sides create equal-area subrectangles -/
structure EqualAreaPoint (a b : ℝ) where
  x : ℝ
  y : ℝ
  x_bounds : 0 < x ∧ x < a
  y_bounds : 0 < y ∧ y < b
  equal_areas : x * y = (a - x) * y ∧ x * (b - y) = (a - x) * (b - y)

/-- The diagonals of a rectangle -/
def rectangleDiagonals (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (p = (t * a, t * b) ∨ p = ((1 - t) * a, t * b))}

/-- Theorem: Points satisfying the equal area condition lie on the diagonals of the rectangle -/
theorem equal_area_point_on_diagonal (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (p : EqualAreaPoint a b) : (p.x, p.y) ∈ rectangleDiagonals a b := by
  sorry

end NUMINAMATH_CALUDE_equal_area_point_on_diagonal_l2054_205458


namespace NUMINAMATH_CALUDE_dog_meal_amount_proof_l2054_205417

/-- The amount of food a dog eats at each meal, in pounds -/
def dog_meal_amount : ℝ := 4

/-- The number of puppies -/
def num_puppies : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of times a dog eats per day -/
def dog_meals_per_day : ℕ := 3

/-- The total amount of food eaten by dogs and puppies in a day, in pounds -/
def total_food_per_day : ℝ := 108

theorem dog_meal_amount_proof :
  dog_meal_amount * num_dogs * dog_meals_per_day + 
  (dog_meal_amount / 2) * num_puppies * (3 * dog_meals_per_day) = total_food_per_day :=
by sorry

end NUMINAMATH_CALUDE_dog_meal_amount_proof_l2054_205417


namespace NUMINAMATH_CALUDE_intersection_point_height_l2054_205473

theorem intersection_point_height (x : ℝ) : x ∈ (Set.Ioo 0 (π/2)) →
  6 * Real.cos x = 5 * Real.tan x →
  ∃ P₁ P₂ : ℝ × ℝ,
    P₁.1 = x ∧ P₁.2 = 0 ∧
    P₂.1 = x ∧ P₂.2 = (1/2) * Real.sin x ∧
    |P₂.2 - P₁.2| = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_height_l2054_205473


namespace NUMINAMATH_CALUDE_elderly_selected_in_scenario_l2054_205476

/-- Represents the number of elderly people selected in a stratified sampling -/
def elderly_selected (total_population : ℕ) (elderly_population : ℕ) (sample_size : ℕ) : ℚ :=
  (sample_size : ℚ) * (elderly_population : ℚ) / (total_population : ℚ)

/-- Theorem stating the number of elderly people selected in the given scenario -/
theorem elderly_selected_in_scenario : 
  elderly_selected 100 60 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_elderly_selected_in_scenario_l2054_205476


namespace NUMINAMATH_CALUDE_function_always_positive_l2054_205494

-- Define a function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the given condition
variable (h : ∀ x, 2 * f x + x * f' x > x^2)

-- Theorem to prove
theorem function_always_positive : ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l2054_205494


namespace NUMINAMATH_CALUDE_carpet_length_independent_of_steps_carpet_sufficient_l2054_205498

/-- Represents a staircase with its properties --/
structure Staircase :=
  (steps : ℕ)
  (length : ℝ)
  (height : ℝ)

/-- Calculates the length of carpet required for a given staircase --/
def carpet_length (s : Staircase) : ℝ := s.length + s.height

/-- Theorem stating that carpet length depends only on staircase length and height --/
theorem carpet_length_independent_of_steps (s1 s2 : Staircase) :
  s1.length = s2.length → s1.height = s2.height →
  carpet_length s1 = carpet_length s2 := by
  sorry

/-- Specific instance for the problem --/
def staircase1 : Staircase := ⟨9, 2, 2⟩
def staircase2 : Staircase := ⟨10, 2, 2⟩

/-- Theorem stating that the carpet for staircase1 is enough for staircase2 --/
theorem carpet_sufficient : carpet_length staircase1 = carpet_length staircase2 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_independent_of_steps_carpet_sufficient_l2054_205498


namespace NUMINAMATH_CALUDE_range_of_a_l2054_205462

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  (a > 0) →
  3 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2054_205462


namespace NUMINAMATH_CALUDE_complex_equality_sum_l2054_205495

theorem complex_equality_sum (a b : ℝ) (h : a - 2 * Complex.I = b + a * Complex.I) : a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l2054_205495


namespace NUMINAMATH_CALUDE_student_weight_is_75_l2054_205474

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The total weight of the student and his sister is 110 kilograms -/
axiom total_weight : student_weight + sister_weight = 110

/-- If the student loses 5 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 5 = 2 * sister_weight

/-- The student's present weight is 75 kilograms -/
theorem student_weight_is_75 : student_weight = 75 := by sorry

end NUMINAMATH_CALUDE_student_weight_is_75_l2054_205474


namespace NUMINAMATH_CALUDE_bus_passing_time_l2054_205441

theorem bus_passing_time (distance : ℝ) (time : ℝ) (bus_length : ℝ) : 
  distance = 12 → time = 5 → bus_length = 200 →
  (bus_length / (distance * 1000 / (time * 60))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_passing_time_l2054_205441


namespace NUMINAMATH_CALUDE_dihedral_angle_ge_line_angle_l2054_205448

/-- A dihedral angle with its plane angle -/
structure DihedralAngle where
  φ : Real
  φ_nonneg : 0 ≤ φ
  φ_le_pi : φ ≤ π

/-- A line contained in one plane of a dihedral angle -/
structure ContainedLine (d : DihedralAngle) where
  θ : Real
  θ_nonneg : 0 ≤ θ
  θ_le_pi_div_2 : θ ≤ π / 2

/-- The plane angle of a dihedral angle is always greater than or equal to 
    the angle between any line in one of its planes and the other plane -/
theorem dihedral_angle_ge_line_angle (d : DihedralAngle) (l : ContainedLine d) : 
  d.φ ≥ l.θ := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_ge_line_angle_l2054_205448


namespace NUMINAMATH_CALUDE_josh_initial_money_l2054_205431

-- Define the given conditions
def spent_on_drink : ℝ := 1.75
def spent_additional : ℝ := 1.25
def money_left : ℝ := 6.00

-- Define the theorem
theorem josh_initial_money :
  ∃ (initial_money : ℝ),
    initial_money = spent_on_drink + spent_additional + money_left ∧
    initial_money = 9.00 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l2054_205431


namespace NUMINAMATH_CALUDE_prob_adjacent_20_3_l2054_205470

/-- The number of people sitting at the round table -/
def n : ℕ := 20

/-- The number of specific people we're interested in -/
def k : ℕ := 3

/-- The probability of at least two out of three specific people sitting next to each other
    in a random seating arrangement of n people at a round table -/
def prob_adjacent (n k : ℕ) : ℚ :=
  17/57

/-- Theorem stating the probability for the given problem -/
theorem prob_adjacent_20_3 : prob_adjacent n k = 17/57 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_20_3_l2054_205470


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2054_205493

theorem max_value_of_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 64) + Real.sqrt (20 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 285.72 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2054_205493


namespace NUMINAMATH_CALUDE_cube_edge_length_l2054_205497

theorem cube_edge_length (a : ℕ) (h1 : a > 0) :
  6 * a^2 = 3 * (12 * a) → a + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2054_205497


namespace NUMINAMATH_CALUDE_simplify_expression_l2054_205475

theorem simplify_expression (x : ℝ) : 3*x + 5 - 2*x - 6 + 4*x + 7 - 5*x - 9 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2054_205475


namespace NUMINAMATH_CALUDE_sum_of_extrema_x_l2054_205451

theorem sum_of_extrema_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extrema_x_l2054_205451


namespace NUMINAMATH_CALUDE_perfect_square_base_l2054_205410

theorem perfect_square_base : ∃! (d : ℕ), d > 1 ∧ ∃ (n : ℕ), d^4 + d^3 + d^2 + d + 1 = n^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_base_l2054_205410


namespace NUMINAMATH_CALUDE_parabola_properties_l2054_205414

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 4)^2 - 5

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), y = parabola x ∧ ∀ (x' : ℝ), parabola x' ≥ y) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < 4 ∧ x₂ > 4 → parabola x₁ > parabola 4 ∧ parabola x₂ > parabola 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2054_205414


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l2054_205492

theorem cubic_equation_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l2054_205492


namespace NUMINAMATH_CALUDE_invested_sum_is_700_l2054_205491

/-- Represents the simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the invested sum is $700 given the problem conditions --/
theorem invested_sum_is_700 
  (peter_amount : ℝ) 
  (david_amount : ℝ) 
  (peter_time : ℝ) 
  (david_time : ℝ) 
  (h1 : peter_amount = 815)
  (h2 : david_amount = 850)
  (h3 : peter_time = 3)
  (h4 : david_time = 4)
  : ∃ (principal rate : ℝ),
    simple_interest principal rate peter_time = peter_amount ∧
    simple_interest principal rate david_time = david_amount ∧
    principal = 700 := by
  sorry

end NUMINAMATH_CALUDE_invested_sum_is_700_l2054_205491


namespace NUMINAMATH_CALUDE_intersection_condition_l2054_205427

-- Define the curves
def curve1 (b x y : ℝ) : Prop := x^2 + y^2 = 2 * b^2
def curve2 (b x y : ℝ) : Prop := y = x^2 - b

-- Define the intersection condition
def intersect_at_four_points (b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (curve1 b x1 y1 ∧ curve2 b x1 y1) ∧
    (curve1 b x2 y2 ∧ curve2 b x2 y2) ∧
    (curve1 b x3 y3 ∧ curve2 b x3 y3) ∧
    (curve1 b x4 y4 ∧ curve2 b x4 y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (curve1 b x y ∧ curve2 b x y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))

-- State the theorem
theorem intersection_condition (b : ℝ) :
  intersect_at_four_points b ↔ b > 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2054_205427


namespace NUMINAMATH_CALUDE_heroes_on_front_l2054_205435

theorem heroes_on_front (total : ℕ) (back : ℕ) (front : ℕ) : 
  total = 9 → back = 7 → total = front + back → front = 2 := by
  sorry

end NUMINAMATH_CALUDE_heroes_on_front_l2054_205435


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l2054_205447

/-- The sticker price of a laptop --/
def stickerPrice : ℝ := sorry

/-- The price at store A after discount and rebate --/
def priceA : ℝ := 0.82 * stickerPrice - 100

/-- The price at store B after discount --/
def priceB : ℝ := 0.75 * stickerPrice

/-- Theorem stating that the sticker price is $1300 given the conditions --/
theorem laptop_sticker_price : 
  priceB - priceA = 10 → stickerPrice = 1300 := by sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l2054_205447


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2054_205472

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 6) + 18 + 3*x + 12 + (x + 9) + (3*x - 5)) / 6 = 19 → x = 37/4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2054_205472


namespace NUMINAMATH_CALUDE_shirt_cost_is_9_l2054_205454

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $61 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 61

/-- Theorem: The cost of one shirt is $9 -/
theorem shirt_cost_is_9 : shirt_cost = 9 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_is_9_l2054_205454


namespace NUMINAMATH_CALUDE_modulus_of_z_l2054_205424

-- Define the complex number z
def z : ℂ := 3 - 4 * Complex.I

-- Theorem statement
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2054_205424


namespace NUMINAMATH_CALUDE_carl_josh_wage_ratio_l2054_205445

/-- Represents the hourly wage ratio between Carl and Josh -/
def wage_ratio : ℚ := 1 / 2

theorem carl_josh_wage_ratio : 
  let josh_hours_per_day : ℕ := 8
  let josh_days_per_week : ℕ := 5
  let josh_weeks_per_month : ℕ := 4
  let carl_hours_less_per_day : ℕ := 2
  let josh_hourly_wage : ℚ := 9
  let total_monthly_pay : ℚ := 1980
  
  let josh_monthly_hours : ℕ := josh_hours_per_day * josh_days_per_week * josh_weeks_per_month
  let carl_monthly_hours : ℕ := (josh_hours_per_day - carl_hours_less_per_day) * josh_days_per_week * josh_weeks_per_month
  let josh_monthly_pay : ℚ := josh_hourly_wage * josh_monthly_hours
  let carl_monthly_pay : ℚ := total_monthly_pay - josh_monthly_pay
  let carl_hourly_wage : ℚ := carl_monthly_pay / carl_monthly_hours

  carl_hourly_wage / josh_hourly_wage = wage_ratio :=
by
  sorry

#check carl_josh_wage_ratio

end NUMINAMATH_CALUDE_carl_josh_wage_ratio_l2054_205445


namespace NUMINAMATH_CALUDE_s_iff_m_range_p_or_q_and_not_q_implies_m_range_l2054_205430

-- Define propositions p, q, and s
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), x^2 / (4 - m) + y^2 / m = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def s (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 = 0

-- Theorem 1
theorem s_iff_m_range (m : ℝ) : s m ↔ m < 0 ∨ m ≥ 2 := by sorry

-- Theorem 2
theorem p_or_q_and_not_q_implies_m_range (m : ℝ) : (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_s_iff_m_range_p_or_q_and_not_q_implies_m_range_l2054_205430


namespace NUMINAMATH_CALUDE_unique_zero_of_f_l2054_205444

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else Real.log x / Real.log a

-- Theorem statement
theorem unique_zero_of_f (a : ℝ) (h : a > 0) :
  ∃! x, f a x = a :=
sorry

end NUMINAMATH_CALUDE_unique_zero_of_f_l2054_205444


namespace NUMINAMATH_CALUDE_fraction_sum_division_l2054_205437

theorem fraction_sum_division (a b c d e f g h : ℚ) :
  a = 3/7 →
  b = 5/8 →
  c = 5/12 →
  d = 2/9 →
  e = a + b →
  f = c + d →
  g = e / f →
  g = 531/322 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_division_l2054_205437


namespace NUMINAMATH_CALUDE_range_of_m_l2054_205408

-- Define the sets A and B
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 10}
def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x : ℝ, x ∈ A → x ∈ B m) →
  (∃ x : ℝ, x ∈ B m ∧ x ∉ A) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2054_205408


namespace NUMINAMATH_CALUDE_range_of_m_l2054_205426

-- Define the propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)

def not_q (x m : ℝ) : Prop := ¬(q x m)

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ x, not_q x m → not_p x) ∧ 
  (∃ x, not_p x ∧ ¬(not_q x m))

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2054_205426


namespace NUMINAMATH_CALUDE_min_value_x_plus_4y_lower_bound_achievable_l2054_205443

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 1 / (2 * y) = 1) : 
  x + 4 * y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

theorem lower_bound_achievable : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 1 / x + 1 / (2 * y) = 1 ∧ 
  x + 4 * y = 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_4y_lower_bound_achievable_l2054_205443


namespace NUMINAMATH_CALUDE_expand_product_l2054_205404

theorem expand_product (x : ℝ) : (x + 4) * (x - 7) = x^2 - 3*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2054_205404


namespace NUMINAMATH_CALUDE_shaded_area_is_five_l2054_205403

/-- Given a parallelogram with regions labeled by areas, prove that the shaded region α has area 5 -/
theorem shaded_area_is_five (x y α : ℝ) 
  (h1 : 3 + α + y = 4 + α + x)
  (h2 : 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)) : 
  α = 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_five_l2054_205403


namespace NUMINAMATH_CALUDE_triangle_sine_sum_zero_l2054_205433

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_sine_sum_zero (t : Triangle) : 
  t.a^3 * Real.sin (t.B - t.C) + t.b^3 * Real.sin (t.C - t.A) + t.c^3 * Real.sin (t.A - t.B) = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_zero_l2054_205433


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2054_205481

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let runs_scored := game.first_part_run_rate * game.first_part_overs
  let runs_needed := game.target_runs - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame)
  (h1 : game.total_overs = 50)
  (h2 : game.first_part_overs = 10)
  (h3 : game.first_part_run_rate = 3.8)
  (h4 : game.target_runs = 282) :
  required_run_rate game = 6.1 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.8,
  target_runs := 282
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2054_205481


namespace NUMINAMATH_CALUDE_total_days_2010_to_2014_l2054_205401

def days_in_year (year : ℕ) : ℕ :=
  if year = 2012 then 366 else 365

def years_range : List ℕ := [2010, 2011, 2012, 2013, 2014]

theorem total_days_2010_to_2014 :
  (years_range.map days_in_year).sum = 1826 := by sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2014_l2054_205401


namespace NUMINAMATH_CALUDE_quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l2054_205405

/-- Represents a quadratic function f(x) = ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic function --/
def roots (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

/-- The x-intercepts of a quadratic function --/
def xIntercepts (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.a * x^2 + f.b * x + f.c = 0}

theorem quadratic_roots_are_x_intercepts (f : QuadraticFunction) :
  roots f = xIntercepts f := by sorry

theorem ac_sign_not_guaranteed (f : QuadraticFunction) :
  ¬∀ f : QuadraticFunction, f.a * f.c < 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_are_x_intercepts_ac_sign_not_guaranteed_l2054_205405


namespace NUMINAMATH_CALUDE_quadratic_sum_value_l2054_205432

/-- 
Given two quadratic trinomials that differ by the interchange of the constant term 
and the second coefficient, if their sum has a unique root, then the value of their 
sum at x = 2 is either 8 or 32.
-/
theorem quadratic_sum_value (p q : ℝ) : 
  let f := fun x : ℝ => x^2 + p*x + q
  let g := fun x : ℝ => x^2 + q*x + p
  let sum := fun x : ℝ => f x + g x
  (∃! r : ℝ, sum r = 0) → (sum 2 = 8 ∨ sum 2 = 32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_value_l2054_205432


namespace NUMINAMATH_CALUDE_order_of_abc_l2054_205419

theorem order_of_abc : 
  let a := Real.log 1.01
  let b := 2 / 201
  let c := Real.sqrt 1.02 - 1
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l2054_205419


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2054_205486

/-- The shaded area of a tiled floor with white quarter circles in each tile corner -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size radius : ℝ)
  (h_floor_length : floor_length = 12)
  (h_floor_width : floor_width = 16)
  (h_tile_size : tile_size = 2)
  (h_radius : radius = 1/2)
  (h_positive : floor_length > 0 ∧ floor_width > 0 ∧ tile_size > 0 ∧ radius > 0) :
  let num_tiles : ℝ := (floor_length * floor_width) / (tile_size * tile_size)
  let white_area_per_tile : ℝ := 4 * π * radius^2
  let shaded_area_per_tile : ℝ := tile_size * tile_size - white_area_per_tile
  num_tiles * shaded_area_per_tile = 192 - 48 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2054_205486


namespace NUMINAMATH_CALUDE_interview_probability_l2054_205468

/-- The number of students enrolled in at least one foreign language class -/
def total_students : ℕ := 25

/-- The number of students in the French class -/
def french_students : ℕ := 18

/-- The number of students in the Spanish class -/
def spanish_students : ℕ := 21

/-- The number of students to be chosen -/
def chosen_students : ℕ := 2

/-- The probability of selecting at least one student from French class
    and at least one student from Spanish class -/
def probability_both_classes : ℚ := 91 / 100

theorem interview_probability :
  let students_in_both := french_students + spanish_students - total_students
  let only_french := french_students - students_in_both
  let only_spanish := spanish_students - students_in_both
  probability_both_classes = 1 - (Nat.choose only_french chosen_students + Nat.choose only_spanish chosen_students : ℚ) / Nat.choose total_students chosen_students :=
by sorry

end NUMINAMATH_CALUDE_interview_probability_l2054_205468


namespace NUMINAMATH_CALUDE_alien_rock_count_l2054_205455

/-- Converts a three-digit number in base 7 to base 10 --/
def base7ToBase10 (hundreds tens units : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + units * 7^0

/-- The number of rocks seen by the alien --/
def alienRocks : ℕ := base7ToBase10 3 5 1

theorem alien_rock_count : alienRocks = 183 := by sorry

end NUMINAMATH_CALUDE_alien_rock_count_l2054_205455


namespace NUMINAMATH_CALUDE_inequality_proof_l2054_205479

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2054_205479


namespace NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l2054_205471

theorem sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_minus_sqrt_eight_equals_sqrt_two_l2054_205471


namespace NUMINAMATH_CALUDE_article_choice_correct_l2054_205440

-- Define the possible article choices
inductive Article
  | A
  | The
  | None

-- Define the structure for an article combination
structure ArticleCombination where
  first : Article
  second : Article

-- Define the conditions of the problem
def is_general_reference (a : Article) : Prop :=
  a = Article.A

def is_specific_reference (a : Article) : Prop :=
  a = Article.The

-- Define the correct combination
def correct_combination : ArticleCombination :=
  { first := Article.A, second := Article.The }

-- Theorem to prove
theorem article_choice_correct
  (german_engineer_general : is_general_reference correct_combination.first)
  (car_invention_specific : is_specific_reference correct_combination.second) :
  correct_combination = { first := Article.A, second := Article.The } := by
  sorry

end NUMINAMATH_CALUDE_article_choice_correct_l2054_205440


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2054_205446

/-- The equation (3x+8)(x-6) = -52 + kx has exactly one real solution if and only if k = 4√3 - 10 or k = -4√3 - 10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -52 + k*x) ↔ (k = 4*Real.sqrt 3 - 10 ∨ k = -4*Real.sqrt 3 - 10) := by
sorry


end NUMINAMATH_CALUDE_unique_solution_condition_l2054_205446


namespace NUMINAMATH_CALUDE_tan_value_of_sequences_l2054_205434

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem tan_value_of_sequences (a b : ℕ → ℝ) :
  is_geometric_sequence a →
  is_arithmetic_sequence b →
  a 1 - a 6 - a 11 = -3 * Real.sqrt 3 →
  b 1 + b 6 + b 11 = 7 * Real.pi →
  Real.tan ((b 3 + b 9) / (1 - a 4 - a 3)) = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_value_of_sequences_l2054_205434


namespace NUMINAMATH_CALUDE_pauls_chickens_l2054_205457

theorem pauls_chickens (neighbor_sale quick_sale remaining : ℕ) 
  (h1 : neighbor_sale = 12)
  (h2 : quick_sale = 25)
  (h3 : remaining = 43) :
  neighbor_sale + quick_sale + remaining = 80 :=
by sorry

end NUMINAMATH_CALUDE_pauls_chickens_l2054_205457


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2054_205413

theorem expected_adjacent_red_pairs (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 40) (h2 : red_cards = 20) :
  let prob_red_after_red := (red_cards - 1) / (total_cards - 1)
  let expected_pairs := red_cards * prob_red_after_red
  expected_pairs = 380 / 39 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2054_205413


namespace NUMINAMATH_CALUDE_player_A_best_performance_l2054_205484

structure Player where
  name : String
  average_score : Float
  variance : Float

def players : List Player := [
  ⟨"A", 9.9, 4.2⟩,
  ⟨"B", 9.8, 5.2⟩,
  ⟨"C", 9.9, 5.2⟩,
  ⟨"D", 9.0, 4.2⟩
]

def has_best_performance (p : Player) (ps : List Player) : Prop :=
  ∀ q ∈ ps, p.average_score ≥ q.average_score ∧ 
    (p.average_score > q.average_score ∨ p.variance ≤ q.variance)

theorem player_A_best_performance :
  ∃ p ∈ players, p.name = "A" ∧ has_best_performance p players := by
  sorry

end NUMINAMATH_CALUDE_player_A_best_performance_l2054_205484


namespace NUMINAMATH_CALUDE_line_canonical_to_general_equations_l2054_205439

/-- Given a line in 3D space defined by canonical equations, prove that the general equations are equivalent. -/
theorem line_canonical_to_general_equations :
  ∀ (x y z : ℝ),
  ((x - 2) / 3 = (y + 1) / 5 ∧ (x - 2) / 3 = (z - 3) / (-1)) ↔
  (5 * x - 3 * y = 13 ∧ x + 3 * z = 11) :=
by sorry

end NUMINAMATH_CALUDE_line_canonical_to_general_equations_l2054_205439


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2054_205412

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (m-2)*x + m + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 + (m-2)*y + m + 1 = 0 → y = x) ↔ 
  m = 0 ∨ m = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2054_205412


namespace NUMINAMATH_CALUDE_factor_tree_root_value_l2054_205415

/-- Represents a node in the factor tree -/
inductive FactorNode
  | Prime (n : Nat)
  | Composite (left right : FactorNode)

/-- Computes the value of a FactorNode -/
def nodeValue : FactorNode → Nat
  | FactorNode.Prime n => n
  | FactorNode.Composite left right => nodeValue left * nodeValue right

/-- The factor tree structure as given in the problem -/
def factorTree : FactorNode :=
  FactorNode.Composite
    (FactorNode.Composite
      (FactorNode.Prime 7)
      (FactorNode.Composite (FactorNode.Prime 7) (FactorNode.Prime 3)))
    (FactorNode.Composite
      (FactorNode.Prime 11)
      (FactorNode.Composite (FactorNode.Prime 11) (FactorNode.Prime 3)))

theorem factor_tree_root_value :
  nodeValue factorTree = 53361 := by
  sorry


end NUMINAMATH_CALUDE_factor_tree_root_value_l2054_205415


namespace NUMINAMATH_CALUDE_point_coordinates_l2054_205467

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    isSecondQuadrant p →
    distanceToXAxis p = 2 →
    distanceToYAxis p = 3 →
    p.x = -3 ∧ p.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2054_205467


namespace NUMINAMATH_CALUDE_city_population_growth_l2054_205496

/-- Represents the birth rate and death rate in a city, and proves the birth rate given conditions --/
theorem city_population_growth (death_rate : ℕ) (net_increase : ℕ) (intervals_per_day : ℕ) 
  (h1 : death_rate = 3)
  (h2 : net_increase = 43200)
  (h3 : intervals_per_day = 43200) :
  ∃ (birth_rate : ℕ), 
    birth_rate = 4 ∧ 
    (birth_rate - death_rate) * intervals_per_day = net_increase :=
sorry

end NUMINAMATH_CALUDE_city_population_growth_l2054_205496


namespace NUMINAMATH_CALUDE_two_face_cubes_5x5x5_l2054_205418

/-- The number of unit cubes with exactly two faces on the surface of a 5x5x5 cube -/
def two_face_cubes (n : ℕ) : ℕ := 12 * (n - 2)

/-- Theorem stating that the number of unit cubes with exactly two faces
    on the surface of a 5x5x5 cube is 36 -/
theorem two_face_cubes_5x5x5 :
  two_face_cubes 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_two_face_cubes_5x5x5_l2054_205418


namespace NUMINAMATH_CALUDE_completing_square_min_value_compare_expressions_l2054_205477

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- Theorem 1: Completing the square
theorem completing_square : ∀ x : ℝ, f x = (x - 2)^2 + 2 := by sorry

-- Theorem 2: Minimum value and corresponding x
theorem min_value : 
  (∃ x_min : ℝ, ∀ x : ℝ, f x ≥ f x_min) ∧
  (∃ x_min : ℝ, f x_min = 2) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f x = 2 → x = x_min) ∧
  (∃ x_min : ℝ, x_min = 2) := by sorry

-- Theorem 3: Comparison of two expressions
theorem compare_expressions : ∀ x : ℝ, x^2 - 1 > 2*x - 3 := by sorry

end NUMINAMATH_CALUDE_completing_square_min_value_compare_expressions_l2054_205477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2054_205480

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₅ = 3 and a₉ = 6,
    prove that a₁₃ = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = 3)
    (h_a9 : a 9 = 6) :
  a 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2054_205480


namespace NUMINAMATH_CALUDE_nigels_money_l2054_205402

theorem nigels_money (initial_amount : ℕ) (given_away : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  given_away = 25 →
  final_amount = 2 * initial_amount + 10 →
  final_amount - (initial_amount - given_away) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_nigels_money_l2054_205402


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l2054_205490

theorem restaurant_bill_split (num_people : ℕ) (individual_payment : ℚ) (original_bill : ℚ) : 
  num_people = 8 →
  individual_payment = 314.15 →
  original_bill = num_people * individual_payment →
  original_bill = 2513.20 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_split_l2054_205490


namespace NUMINAMATH_CALUDE_bekah_reading_days_l2054_205420

/-- Given the total pages to read, pages already read, and pages to read per day,
    calculate the number of days left to finish reading. -/
def days_left_to_read (total_pages pages_read pages_per_day : ℕ) : ℕ :=
  (total_pages - pages_read) / pages_per_day

/-- Theorem: Given 408 total pages, 113 pages read, and 59 pages per day,
    the number of days left to finish reading is 5. -/
theorem bekah_reading_days : days_left_to_read 408 113 59 = 5 := by
  sorry

#eval days_left_to_read 408 113 59

end NUMINAMATH_CALUDE_bekah_reading_days_l2054_205420


namespace NUMINAMATH_CALUDE_non_negative_inequality_l2054_205483

theorem non_negative_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_inequality_l2054_205483


namespace NUMINAMATH_CALUDE_turtle_ratio_l2054_205436

theorem turtle_ratio : 
  ∀ (trey kris kristen : ℕ),
  kristen = 12 →
  kris = kristen / 4 →
  trey = kristen + 9 →
  trey / kris = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l2054_205436


namespace NUMINAMATH_CALUDE_walking_time_equals_time_saved_l2054_205425

/-- Represents the scenario of a man walking and his wife driving to meet him -/
structure CommuteScenario where
  usual_drive_time : ℝ
  actual_drive_time : ℝ
  time_saved : ℝ
  walking_time : ℝ

/-- Theorem stating that the walking time equals the time saved -/
theorem walking_time_equals_time_saved (scenario : CommuteScenario) 
  (h1 : scenario.usual_drive_time > 0)
  (h2 : scenario.actual_drive_time > 0)
  (h3 : scenario.time_saved > 0)
  (h4 : scenario.walking_time > 0)
  (h5 : scenario.usual_drive_time = scenario.actual_drive_time + scenario.time_saved)
  (h6 : scenario.walking_time = scenario.time_saved) : 
  scenario.walking_time = scenario.time_saved :=
by sorry

end NUMINAMATH_CALUDE_walking_time_equals_time_saved_l2054_205425


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2054_205452

/-- The number of red roses at Greene Nursery -/
def red_roses : ℕ := 1491

/-- The number of yellow carnations at Greene Nursery -/
def yellow_carnations : ℕ := 3025

/-- The number of white roses at Greene Nursery -/
def white_roses : ℕ := 1768

/-- The number of purple tulips at Greene Nursery -/
def purple_tulips : ℕ := 2150

/-- The number of pink daisies at Greene Nursery -/
def pink_daisies : ℕ := 3500

/-- The number of blue irises at Greene Nursery -/
def blue_irises : ℕ := 2973

/-- The number of orange marigolds at Greene Nursery -/
def orange_marigolds : ℕ := 4234

/-- The total number of flowers at Greene Nursery -/
def total_flowers : ℕ := red_roses + yellow_carnations + white_roses + purple_tulips + 
                          pink_daisies + blue_irises + orange_marigolds

theorem greene_nursery_flower_count : total_flowers = 19141 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2054_205452


namespace NUMINAMATH_CALUDE_bodyguard_hours_theorem_l2054_205459

/-- The number of hours per day Tim hires bodyguards -/
def hours_per_day (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ) : ℕ :=
  weekly_payment / (num_bodyguards * hourly_rate * days_per_week)

/-- Theorem stating that Tim hires bodyguards for 8 hours per day -/
theorem bodyguard_hours_theorem (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ)
  (h1 : num_bodyguards = 2)
  (h2 : hourly_rate = 20)
  (h3 : weekly_payment = 2240)
  (h4 : days_per_week = 7) :
  hours_per_day num_bodyguards hourly_rate weekly_payment days_per_week = 8 := by
  sorry

end NUMINAMATH_CALUDE_bodyguard_hours_theorem_l2054_205459


namespace NUMINAMATH_CALUDE_tan_sin_identity_l2054_205450

theorem tan_sin_identity : 2 * Real.tan (10 * π / 180) + 3 * Real.sin (10 * π / 180) = 5 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_identity_l2054_205450


namespace NUMINAMATH_CALUDE_tarantulas_in_egg_sac_l2054_205438

/-- The number of legs a tarantula has -/
def tarantula_legs : ℕ := 8

/-- The total number of baby tarantula legs in the egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the baby tarantulas -/
def num_egg_sacs : ℕ := 4

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := total_legs / (tarantula_legs * num_egg_sacs)

theorem tarantulas_in_egg_sac : tarantulas_per_sac = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tarantulas_in_egg_sac_l2054_205438


namespace NUMINAMATH_CALUDE_job_completion_time_l2054_205487

/-- Proves that the initial estimated time to finish the job is 8 days given the problem conditions. -/
theorem job_completion_time : 
  ∀ (initial_workers : ℕ) 
    (additional_workers : ℕ) 
    (days_before_joining : ℕ) 
    (days_after_joining : ℕ),
  initial_workers = 6 →
  additional_workers = 4 →
  days_before_joining = 3 →
  days_after_joining = 3 →
  ∃ (initial_estimate : ℕ),
    initial_estimate * initial_workers = 
      (initial_workers * days_before_joining + 
       (initial_workers + additional_workers) * days_after_joining) ∧
    initial_estimate = 8 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l2054_205487


namespace NUMINAMATH_CALUDE_aaron_position_100_l2054_205422

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Defines Aaron's movement rules -/
def nextPosition (current : Position) (dir : Direction) (visited : List Position) : Position × Direction :=
  sorry

/-- Calculates Aaron's position after n moves -/
def aaronPosition (n : Nat) : Position :=
  sorry

/-- Theorem stating Aaron's position after 100 moves -/
theorem aaron_position_100 : aaronPosition 100 = Position.mk 22 (-6) := by
  sorry

end NUMINAMATH_CALUDE_aaron_position_100_l2054_205422


namespace NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l2054_205449

def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

theorem apple_percentage_after_adding_oranges :
  (initial_apples : ℚ) / total_fruits * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_apple_percentage_after_adding_oranges_l2054_205449


namespace NUMINAMATH_CALUDE_actual_weekly_earnings_increase_l2054_205461

/-- Calculates the actual increase in weekly earnings given a raise, work hours, and housing benefit reduction. -/
theorem actual_weekly_earnings_increase
  (hourly_raise : ℝ)
  (weekly_hours : ℝ)
  (monthly_benefit_reduction : ℝ)
  (h1 : hourly_raise = 0.50)
  (h2 : weekly_hours = 40)
  (h3 : monthly_benefit_reduction = 60)
  : ∃ (actual_increase : ℝ), abs (actual_increase - 6.14) < 0.01 := by
  sorry

#check actual_weekly_earnings_increase

end NUMINAMATH_CALUDE_actual_weekly_earnings_increase_l2054_205461


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l2054_205466

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 1/2  -- probability of bottling chocolate milk each day
  (n.choose k) * p^k * (1-p)^(n-k) = 21/128 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l2054_205466


namespace NUMINAMATH_CALUDE_intersection_point_l2054_205400

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 9*x + 15

theorem intersection_point (a b : ℝ) :
  (∀ x ≠ a, f x ≠ f a) ∧ 
  f a = b ∧ 
  f b = a ∧ 
  (∀ x y, f x = y ∧ f y = x → x = a ∧ y = b) →
  a = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l2054_205400


namespace NUMINAMATH_CALUDE_line_equation_l2054_205469

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := x + 3*y + 5 = 0

-- Define the intersection point
def intersection_point : ℝ × ℝ :=
  (0, 2)

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Theorem statement
theorem line_equation : 
  ∃ (A B C : ℝ), 
    (A ≠ 0 ∨ B ≠ 0) ∧ 
    (∀ x y : ℝ, A*x + B*y + C = 0 ↔ 
      (line1 x y ∧ line2 x y) ∨
      (x = intersection_point.1 ∧ y = intersection_point.2) ∨
      (∃ m : ℝ, perpendicular m (-1/3) ∧ y - intersection_point.2 = m * (x - intersection_point.1))) ∧
    A = 3 ∧ B = -1 ∧ C = 2 :=
  sorry

end NUMINAMATH_CALUDE_line_equation_l2054_205469


namespace NUMINAMATH_CALUDE_certain_number_problem_l2054_205421

theorem certain_number_problem : ∃ x : ℤ, (5 + (x + 3) = 19) ∧ (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2054_205421


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l2054_205409

theorem orange_juice_fraction : 
  let pitcher1_capacity : ℚ := 800
  let pitcher2_capacity : ℚ := 700
  let pitcher1_juice_fraction : ℚ := 1/4
  let pitcher2_juice_fraction : ℚ := 3/7
  let total_juice := pitcher1_capacity * pitcher1_juice_fraction + pitcher2_capacity * pitcher2_juice_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_juice / total_volume = 1/3 := by sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l2054_205409


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l2054_205464

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 61 ∧ n % 6 = 5 ∧ ∀ m : ℕ, m < 61 ∧ m % 6 = 5 → m ≤ n → n = 59 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l2054_205464


namespace NUMINAMATH_CALUDE_box_min_height_l2054_205406

/-- Represents a rectangular box with square bases -/
structure Box where
  base_side : ℝ
  height : ℝ

/-- Calculates the surface area of a box -/
def surface_area (b : Box) : ℝ :=
  2 * b.base_side^2 + 4 * b.base_side * b.height

/-- The minimum height of a box satisfying the given conditions -/
def min_height : ℝ := 6

theorem box_min_height :
  ∀ (b : Box),
    b.height = b.base_side + 4 →
    surface_area b ≥ 120 →
    b.height ≥ min_height :=
by
  sorry

end NUMINAMATH_CALUDE_box_min_height_l2054_205406


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2054_205428

/-- Given a geometric sequence a, prove that if a₃ = -9 and a₇ = -1, then a₅ = -3 -/
theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_3 : a 3 = -9) 
  (h_7 : a 7 = -1) : 
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2054_205428


namespace NUMINAMATH_CALUDE_parabola_vertex_l2054_205453

/-- A parabola defined by the equation y^2 + 6y + 2x + 5 = 0 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 6*p.2 + 2*p.1 + 5 = 0}

/-- The vertex of a parabola -/
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem stating that the vertex of the given parabola is (2, -3) -/
theorem parabola_vertex : vertex Parabola = (2, -3) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2054_205453


namespace NUMINAMATH_CALUDE_number_difference_l2054_205478

theorem number_difference (L S : ℕ) (hL : L = 1600) (hDiv : L = S * 16 + 15) : L - S = 1501 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2054_205478


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l2054_205416

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_when_x_in_open_interval :
  {a : ℝ | ∀ x ∈ Set.Ioo 0 1, f a x > x} = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_when_x_in_open_interval_l2054_205416


namespace NUMINAMATH_CALUDE_inequality_range_proof_l2054_205463

theorem inequality_range_proof : 
  {x : ℝ | ∀ t : ℝ, |t - 3| + |2*t + 1| ≥ |2*x - 1| + |x + 2|} = 
  {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/6} := by sorry

end NUMINAMATH_CALUDE_inequality_range_proof_l2054_205463


namespace NUMINAMATH_CALUDE_milk_tea_sales_ratio_l2054_205423

theorem milk_tea_sales_ratio (total_sales : ℕ) (okinawa_ratio : ℚ) (chocolate_sales : ℕ) : 
  total_sales = 50 →
  okinawa_ratio = 3 / 10 →
  chocolate_sales = 15 →
  (total_sales - (okinawa_ratio * total_sales).num - chocolate_sales) * 5 = total_sales * 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_tea_sales_ratio_l2054_205423


namespace NUMINAMATH_CALUDE_inequality_always_true_l2054_205489

theorem inequality_always_true : ∀ x : ℝ, (x + 1) * (2 - x) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l2054_205489
