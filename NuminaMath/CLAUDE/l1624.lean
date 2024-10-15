import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1624_162401

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 9 * x + a < 0 ∧ (x^2 - 4 * x + 3 < 0 ∨ x^2 - 6 * x + 8 < 0)) ↔
  a < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1624_162401


namespace NUMINAMATH_CALUDE_min_framing_for_enlarged_picture_l1624_162473

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Round up to nearest foot
  perimeter_feet

/-- The theorem states that for a 4-inch by 6-inch picture enlarged by quadrupling its dimensions
    and adding a 3-inch border on each side, the minimum number of linear feet of framing needed is 9. -/
theorem min_framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_enlarged_picture_l1624_162473


namespace NUMINAMATH_CALUDE_min_sum_squares_l1624_162402

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

#check min_sum_squares

end NUMINAMATH_CALUDE_min_sum_squares_l1624_162402


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1624_162479

theorem perfect_square_polynomial (g : ℕ) : 
  (∃ k : ℕ, g^4 + g^3 + g^2 + g + 1 = k^2) → g = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1624_162479


namespace NUMINAMATH_CALUDE_reflection_composition_maps_points_l1624_162469

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define reflection operations
def reflectY (p : Point) : Point :=
  (-p.1, p.2)

def reflectX (p : Point) : Point :=
  (p.1, -p.2)

-- Define the composition of reflections
def reflectYX (p : Point) : Point :=
  reflectX (reflectY p)

-- Theorem statement
theorem reflection_composition_maps_points :
  let C : Point := (3, -2)
  let D : Point := (4, -5)
  let C' : Point := (-3, 2)
  let D' : Point := (-4, 5)
  reflectYX C = C' ∧ reflectYX D = D' := by sorry

end NUMINAMATH_CALUDE_reflection_composition_maps_points_l1624_162469


namespace NUMINAMATH_CALUDE_total_weight_moved_proof_l1624_162426

/-- Calculates the total weight moved during three sets of back squat, front squat, and deadlift exercises --/
def total_weight_moved (initial_back_squat : ℝ) (back_squat_increase : ℝ) : ℝ :=
  let updated_back_squat := initial_back_squat + back_squat_increase
  let front_squat_ratio := 0.8
  let deadlift_ratio := 1.2
  let back_squat_increase_ratio := 1.05
  let front_squat_increase_ratio := 1.04
  let deadlift_increase_ratio := 1.03
  let back_squat_performance_ratio := 1.0
  let front_squat_performance_ratio := 0.9
  let deadlift_performance_ratio := 0.85
  let back_squat_reps := 3
  let front_squat_reps := 3
  let deadlift_reps := 2

  let back_squat_set1 := updated_back_squat * back_squat_performance_ratio * back_squat_reps
  let back_squat_set2 := updated_back_squat * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps
  let back_squat_set3 := updated_back_squat * back_squat_increase_ratio * back_squat_increase_ratio * back_squat_performance_ratio * back_squat_reps

  let front_squat_base := updated_back_squat * front_squat_ratio
  let front_squat_set1 := front_squat_base * front_squat_performance_ratio * front_squat_reps
  let front_squat_set2 := front_squat_base * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps
  let front_squat_set3 := front_squat_base * front_squat_increase_ratio * front_squat_increase_ratio * front_squat_performance_ratio * front_squat_reps

  let deadlift_base := updated_back_squat * deadlift_ratio
  let deadlift_set1 := deadlift_base * deadlift_performance_ratio * deadlift_reps
  let deadlift_set2 := deadlift_base * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps
  let deadlift_set3 := deadlift_base * deadlift_increase_ratio * deadlift_increase_ratio * deadlift_performance_ratio * deadlift_reps

  back_squat_set1 + back_squat_set2 + back_squat_set3 +
  front_squat_set1 + front_squat_set2 + front_squat_set3 +
  deadlift_set1 + deadlift_set2 + deadlift_set3

theorem total_weight_moved_proof (initial_back_squat : ℝ) (back_squat_increase : ℝ) :
  initial_back_squat = 200 → back_squat_increase = 50 →
  total_weight_moved initial_back_squat back_squat_increase = 5626.398 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_moved_proof_l1624_162426


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l1624_162477

/-- The area of a square with vertices at (2,1), (4,3), (2,5), and (0,3) divided by the area of a 5x5 square -/
theorem shaded_square_area_fraction : 
  let vertices : List (ℤ × ℤ) := [(2,1), (4,3), (2,5), (0,3)]
  let side_length := Real.sqrt ((4 - 2)^2 + (3 - 1)^2)
  let shaded_area := side_length ^ 2
  let grid_area := 5^2
  shaded_area / grid_area = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l1624_162477


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l1624_162472

theorem max_triangle_side_length (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l1624_162472


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1624_162454

theorem no_solutions_for_equation : 
  ¬ ∃ (n : ℕ+), (n + 900) / 60 = ⌊Real.sqrt n⌋ := by
sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1624_162454


namespace NUMINAMATH_CALUDE_bill_score_l1624_162430

theorem bill_score (john sue ella bill : ℕ) 
  (h1 : bill = john + 20)
  (h2 : bill * 2 = sue)
  (h3 : ella = bill + john - 10)
  (h4 : bill + john + sue + ella = 250) : 
  bill = 50 := by
sorry

end NUMINAMATH_CALUDE_bill_score_l1624_162430


namespace NUMINAMATH_CALUDE_count_pairs_eq_15_l1624_162437

def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem count_pairs_eq_15 : count_pairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_15_l1624_162437


namespace NUMINAMATH_CALUDE_initial_average_price_l1624_162489

/-- The price of an apple in cents -/
def apple_price : ℕ := 40

/-- The price of an orange in cents -/
def orange_price : ℕ := 60

/-- The total number of fruits Mary initially selects -/
def total_fruits : ℕ := 10

/-- The number of oranges Mary puts back -/
def oranges_removed : ℕ := 5

/-- The average price of remaining fruits after removing oranges, in cents -/
def remaining_avg_price : ℕ := 48

theorem initial_average_price (a o : ℕ) 
  (h1 : a + o = total_fruits)
  (h2 : (apple_price * a + orange_price * o) / total_fruits = 54)
  (h3 : (apple_price * a + orange_price * (o - oranges_removed)) / (total_fruits - oranges_removed) = remaining_avg_price) :
  (apple_price * a + orange_price * o) / total_fruits = 54 :=
sorry

end NUMINAMATH_CALUDE_initial_average_price_l1624_162489


namespace NUMINAMATH_CALUDE_tire_swap_optimal_l1624_162474

/-- Represents the wear rate of a tire in km^(-1) -/
def WearRate := ℝ

/-- Calculates the remaining life of a tire after driving a certain distance -/
def remaining_life (total_life : ℝ) (distance_driven : ℝ) : ℝ :=
  total_life - distance_driven

/-- Theorem: Swapping tires at 9375 km results in simultaneous wear-out -/
theorem tire_swap_optimal (front_life rear_life swap_distance : ℝ)
  (h_front : front_life = 25000)
  (h_rear : rear_life = 15000)
  (h_swap : swap_distance = 9375) :
  remaining_life front_life swap_distance / rear_life =
  remaining_life rear_life swap_distance / front_life := by
  sorry

#check tire_swap_optimal

end NUMINAMATH_CALUDE_tire_swap_optimal_l1624_162474


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1624_162429

/-- Given a quadratic expression px^2 + qx + r that can be expressed as 5(x + 3)^2 - 15,
    prove that when 4px^2 + 4qx + 4r is written in the form m(x - h)^2 + k, then h = -3 -/
theorem quadratic_transformation (p q r : ℝ) 
  (h : ∀ x, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) :
  ∃ (m k : ℝ), ∀ x, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - (-3))^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1624_162429


namespace NUMINAMATH_CALUDE_cupcake_business_net_profit_l1624_162482

/-- Calculates the net profit from a cupcake business given the following conditions:
  * Each cupcake costs $0.75 to make
  * First 2 dozen cupcakes burnt and were thrown out
  * Next 2 dozen came out perfectly
  * 5 cupcakes were eaten right away
  * Later made 2 more dozen cupcakes
  * 4 more cupcakes were eaten
  * Remaining cupcakes are sold at $2.00 each
-/
theorem cupcake_business_net_profit :
  let cost_per_cupcake : ℚ := 75 / 100
  let sell_price : ℚ := 2
  let dozen : ℕ := 12
  let burnt_cupcakes : ℕ := 2 * dozen
  let eaten_cupcakes : ℕ := 5 + 4
  let total_cupcakes : ℕ := 6 * dozen
  let remaining_cupcakes : ℕ := total_cupcakes - burnt_cupcakes - eaten_cupcakes
  let revenue : ℚ := remaining_cupcakes * sell_price
  let total_cost : ℚ := total_cupcakes * cost_per_cupcake
  let net_profit : ℚ := revenue - total_cost
  net_profit = 24 := by sorry

end NUMINAMATH_CALUDE_cupcake_business_net_profit_l1624_162482


namespace NUMINAMATH_CALUDE_expression_behavior_l1624_162456

/-- Given a > b > c, this theorem characterizes the behavior of the expression (a-x)(b-x)/(c-x) for different values of x. -/
theorem expression_behavior (a b c x : ℝ) (h : a > b ∧ b > c) :
  let f := fun (x : ℝ) => (a - x) * (b - x) / (c - x)
  (x < c ∨ (b < x ∧ x < a) → f x > 0) ∧
  ((c < x ∧ x < b) ∨ x > a → f x < 0) ∧
  (x = a ∨ x = b → f x = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ y, 0 < |y - c| ∧ |y - c| < δ → |f y| > 1/ε) :=
by sorry

end NUMINAMATH_CALUDE_expression_behavior_l1624_162456


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1624_162460

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1624_162460


namespace NUMINAMATH_CALUDE_binomial_12_11_squared_l1624_162432

theorem binomial_12_11_squared : (Nat.choose 12 11)^2 = 144 := by sorry

end NUMINAMATH_CALUDE_binomial_12_11_squared_l1624_162432


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1624_162476

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 1) : 
    ∃ d, d = 2 ∧ ∀ n, seq.a (n + 1) - seq.a n = d :=
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1624_162476


namespace NUMINAMATH_CALUDE_solve_for_x_l1624_162436

theorem solve_for_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 3) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1624_162436


namespace NUMINAMATH_CALUDE_b_worked_nine_days_l1624_162444

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c in dollars -/
def wage_c : ℕ := 100

/-- The total earnings of all three persons in dollars -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- The number of days worked by person b -/
def days_b : ℕ := 9

theorem b_worked_nine_days :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_c * wage_ratio 0 / wage_ratio 2 ∧
    wage_b = wage_c * wage_ratio 1 / wage_ratio 2 ∧
    days_a * wage_a + days_b * wage_b + days_c * wage_c = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_b_worked_nine_days_l1624_162444


namespace NUMINAMATH_CALUDE_value_of_A_l1624_162438

/-- Given the values of words and letters, prove the value of A -/
theorem value_of_A (L LEAD DEAL DELL : ℤ) (h1 : L = 15) (h2 : LEAD = 50) (h3 : DEAL = 55) (h4 : DELL = 60) : ∃ A : ℤ, A = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_A_l1624_162438


namespace NUMINAMATH_CALUDE_deposit_percentage_l1624_162434

def deposit : ℝ := 3800
def monthly_income : ℝ := 11875

theorem deposit_percentage : (deposit / monthly_income) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l1624_162434


namespace NUMINAMATH_CALUDE_identical_differences_exist_l1624_162427

theorem identical_differences_exist (a : Fin 20 → ℕ) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bounded : ∀ i, a i ≤ 70) : 
  ∃ (i₁ j₁ i₂ j₂ i₃ j₃ i₄ j₄ : Fin 20), 
    i₁ < j₁ ∧ i₂ < j₂ ∧ i₃ < j₃ ∧ i₄ < j₄ ∧ 
    (i₁ ≠ i₂ ∨ j₁ ≠ j₂) ∧ (i₁ ≠ i₃ ∨ j₁ ≠ j₃) ∧ (i₁ ≠ i₄ ∨ j₁ ≠ j₄) ∧
    (i₂ ≠ i₃ ∨ j₂ ≠ j₃) ∧ (i₂ ≠ i₄ ∨ j₂ ≠ j₄) ∧ (i₃ ≠ i₄ ∨ j₃ ≠ j₄) ∧
    a j₁ - a i₁ = a j₂ - a i₂ ∧ 
    a j₁ - a i₁ = a j₃ - a i₃ ∧ 
    a j₁ - a i₁ = a j₄ - a i₄ :=
by sorry

end NUMINAMATH_CALUDE_identical_differences_exist_l1624_162427


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1624_162442

theorem fraction_equivalence : 
  ∀ (n : ℕ), (4 + n : ℚ) / (7 + n) = 7 / 8 → n = 17 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1624_162442


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l1624_162493

/-- A polynomial satisfying the given condition -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating the form of polynomials satisfying the condition -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) 
  (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x, p x = a₂ * x^2 + a₁ * x :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l1624_162493


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_implies_m_negative_l1624_162405

/-- A curve represented by the equation x²/m + y²/(1-m) = 1 -/
def is_curve (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/m + y^2/(1-m) = 1

/-- The curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  is_curve m ∧ ∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, y^2/(1-m) - x^2/(-m) = 1

/-- The theorem stating that if the curve is a hyperbola with foci on the y-axis, then m < 0 -/
theorem hyperbola_y_axis_implies_m_negative (m : ℝ) :
  is_hyperbola_y_axis m → m < 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_implies_m_negative_l1624_162405


namespace NUMINAMATH_CALUDE_work_completion_time_l1624_162458

theorem work_completion_time (b_time : ℕ) (b_worked : ℕ) (a_remaining : ℕ) : 
  b_time = 15 → b_worked = 10 → a_remaining = 3 → 
  ∃ (a_time : ℕ), a_time = 9 ∧ 
    (b_worked : ℚ) / b_time + a_remaining / a_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1624_162458


namespace NUMINAMATH_CALUDE_ramanujan_number_l1624_162491

def hardy_number : ℂ := 4 + 2 * Complex.I
def product : ℂ := 18 - 34 * Complex.I

theorem ramanujan_number : 
  ∃ r : ℂ, r * hardy_number = product ∧ r = 0.2 - 8.6 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l1624_162491


namespace NUMINAMATH_CALUDE_birthday_money_theorem_l1624_162494

def birthday_money_problem (initial_amount : ℚ) (video_game_fraction : ℚ) (goggles_fraction : ℚ) : ℚ :=
  let remaining_after_game := initial_amount * (1 - video_game_fraction)
  remaining_after_game * (1 - goggles_fraction)

theorem birthday_money_theorem :
  birthday_money_problem 100 (1/4) (1/5) = 60 := by
  sorry

end NUMINAMATH_CALUDE_birthday_money_theorem_l1624_162494


namespace NUMINAMATH_CALUDE_sum_of_cubes_constraint_l1624_162416

theorem sum_of_cubes_constraint (a b : ℝ) :
  a^3 + b^3 = 1 - 3*a*b → (a + b = 1 ∨ a + b = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_constraint_l1624_162416


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1624_162496

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * (z - 1) = 1 + Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1624_162496


namespace NUMINAMATH_CALUDE_sqrt_four_squared_five_cubed_divided_by_five_l1624_162445

theorem sqrt_four_squared_five_cubed_divided_by_five (x : ℝ) :
  x = (Real.sqrt (4^2 * 5^3)) / 5 → x = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_five_cubed_divided_by_five_l1624_162445


namespace NUMINAMATH_CALUDE_school_sampling_l1624_162449

theorem school_sampling (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), 
    boys + girls = sample_size ∧ 
    boys = girls + 10 ∧ 
    (boys : ℚ) / total_students = (boys : ℚ) / sample_size) :
  ∃ (school_boys : ℕ), school_boys = 630 ∧ 
    (school_boys : ℚ) / total_students = 
    ((sample_size / 2 + 5) : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_school_sampling_l1624_162449


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1624_162418

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1624_162418


namespace NUMINAMATH_CALUDE_comics_bought_l1624_162481

theorem comics_bought (initial_amount remaining_amount cost_per_comic : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : remaining_amount = 55)
  (h3 : cost_per_comic = 4) :
  (initial_amount - remaining_amount) / cost_per_comic = 8 := by
  sorry

end NUMINAMATH_CALUDE_comics_bought_l1624_162481


namespace NUMINAMATH_CALUDE_yellow_tint_percentage_l1624_162467

/-- Calculates the percentage of yellow tint in a new mixture after adding more yellow tint -/
theorem yellow_tint_percentage 
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow : ℝ) :
  initial_volume = 50 →
  initial_yellow_percentage = 25 →
  added_yellow = 10 →
  let initial_yellow := initial_volume * (initial_yellow_percentage / 100)
  let new_yellow := initial_yellow + added_yellow
  let new_volume := initial_volume + added_yellow
  (new_yellow / new_volume) * 100 = 37.5 := by
sorry


end NUMINAMATH_CALUDE_yellow_tint_percentage_l1624_162467


namespace NUMINAMATH_CALUDE_inequality_proof_l1624_162412

theorem inequality_proof (a : ℝ) (n : ℕ) (h1 : a > -1) (h2 : a ≠ 0) (h3 : n ≥ 2) :
  (1 + a)^n > 1 + n * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1624_162412


namespace NUMINAMATH_CALUDE_find_divisor_l1624_162421

theorem find_divisor (x d : ℚ) : 
  x = 55 → 
  x / d + 10 = 21 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_find_divisor_l1624_162421


namespace NUMINAMATH_CALUDE_matrix_not_invertible_l1624_162455

theorem matrix_not_invertible : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2 + 16/19, 9; 4 - 16/19, 10]
  ¬(IsUnit (Matrix.det A)) := by
sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_l1624_162455


namespace NUMINAMATH_CALUDE_largest_c_for_two_in_range_l1624_162409

theorem largest_c_for_two_in_range : 
  let f (x c : ℝ) := 3 * x^2 - 6 * x + c
  ∃ (c_max : ℝ), c_max = 5 ∧ 
    (∀ c : ℝ, (∃ x : ℝ, f x c = 2) → c ≤ c_max) ∧
    (∃ x : ℝ, f x c_max = 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_two_in_range_l1624_162409


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1624_162497

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

-- Define the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-1) 2) :
  a < 0 ∧
  a + b + c > 0 ∧
  solution_set b c (3*a) = Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1624_162497


namespace NUMINAMATH_CALUDE_max_value_problem_l1624_162490

theorem max_value_problem (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 5 * b ≤ 11) : 
  2 * a + b ≤ 48 / 11 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l1624_162490


namespace NUMINAMATH_CALUDE_floor_distinctness_iff_range_l1624_162485

variable (N M : ℕ)
variable (a : ℝ)

-- Define the property that floor values of ka are distinct
def distinctFloorMultiples : Prop :=
  ∀ k l, k ≠ l → k ≤ N → l ≤ N → ⌊k * a⌋ ≠ ⌊l * a⌋

-- Define the property that floor values of k/a are distinct
def distinctFloorDivisions : Prop :=
  ∀ k l, k ≠ l → k ≤ M → l ≤ M → ⌊k / a⌋ ≠ ⌊l / a⌋

theorem floor_distinctness_iff_range (hN : N > 1) (hM : M > 1) :
  (distinctFloorMultiples N a ∧ distinctFloorDivisions M a) ↔
  ((N - 1 : ℝ) / N ≤ a ∧ a ≤ M / (M - 1 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_floor_distinctness_iff_range_l1624_162485


namespace NUMINAMATH_CALUDE_equation_solution_l1624_162452

theorem equation_solution : 
  let f (x : ℝ) := 1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 10) + 1 / (x^2 - 11*x - 10)
  ∀ x : ℝ, f x = 0 ↔ 
    x = (-15 + Real.sqrt 265) / 2 ∨ 
    x = (-15 - Real.sqrt 265) / 2 ∨ 
    x = (6 + Real.sqrt 76) / 2 ∨ 
    x = (6 - Real.sqrt 76) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1624_162452


namespace NUMINAMATH_CALUDE_second_party_amount_2000_4_16_l1624_162466

/-- Calculates the amount received by the second party in a two-party division given a total amount and a ratio --/
def calculate_second_party_amount (total : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) : ℕ :=
  let total_parts := ratio1 + ratio2
  let part_value := total / total_parts
  ratio2 * part_value

theorem second_party_amount_2000_4_16 :
  calculate_second_party_amount 2000 4 16 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_second_party_amount_2000_4_16_l1624_162466


namespace NUMINAMATH_CALUDE_leah_coin_value_l1624_162464

/-- Represents the types of coins Leah has --/
inductive Coin
| Penny
| Nickel
| Dime

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

/-- Leah's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 15
  dime_nickel_relation : dimes - 1 = nickels

theorem leah_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + 
  c.nickels * coinValue Coin.Nickel + 
  c.dimes * coinValue Coin.Dime = 89 := by
  sorry

#check leah_coin_value

end NUMINAMATH_CALUDE_leah_coin_value_l1624_162464


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1624_162428

-- Part 1: System of equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 13 ∧ 2 * x + 3 * y = -8 ∧ x = 11 ∧ y = -10 := by sorry

-- Part 2: System of inequalities
theorem solve_system_inequalities :
  ∀ y : ℝ, ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2 ∧ 2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1624_162428


namespace NUMINAMATH_CALUDE_eight_operations_proof_l1624_162443

theorem eight_operations_proof :
  (((8 : ℚ) / 8) * ((8 : ℚ) / 8) = 1) ∧
  (((8 : ℚ) / 8) + ((8 : ℚ) / 8) = 2) := by
  sorry

end NUMINAMATH_CALUDE_eight_operations_proof_l1624_162443


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1624_162400

theorem cubic_roots_problem (p q r : ℂ) (u v w : ℂ) : 
  (p^3 + 5*p^2 + 6*p - 8 = 0) →
  (q^3 + 5*q^2 + 6*q - 8 = 0) →
  (r^3 + 5*r^2 + 6*r - 8 = 0) →
  ((p+q)^3 + u*(p+q)^2 + v*(p+q) + w = 0) →
  ((q+r)^3 + u*(q+r)^2 + v*(q+r) + w = 0) →
  ((r+p)^3 + u*(r+p)^2 + v*(r+p) + w = 0) →
  w = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1624_162400


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l1624_162450

theorem terminal_side_in_second_quadrant (α : Real) 
  (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  ∃ x y : Real, x < 0 ∧ y > 0 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ 
  Real.sin α = y / Real.sqrt (x^2 + y^2) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l1624_162450


namespace NUMINAMATH_CALUDE_set_operations_l1624_162407

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) ∧
  (A ∩ (U \ B) = {x | 2 ≤ x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1624_162407


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1624_162423

theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →  -- exactly one solution
  (a + c = 35) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1624_162423


namespace NUMINAMATH_CALUDE_group_purchase_equation_l1624_162435

/-- Represents a group purchase scenario -/
structure GroupPurchase where
  price : ℝ  -- Price of the item
  contribution1 : ℝ  -- First contribution amount per person
  excess : ℝ  -- Excess amount for first contribution
  contribution2 : ℝ  -- Second contribution amount per person
  shortage : ℝ  -- Shortage amount for second contribution

/-- Theorem stating the equation for the group purchase scenario -/
theorem group_purchase_equation (gp : GroupPurchase) 
  (h1 : gp.contribution1 = 8) 
  (h2 : gp.excess = 3) 
  (h3 : gp.contribution2 = 7) 
  (h4 : gp.shortage = 4) :
  (gp.price + gp.excess) / gp.contribution1 = (gp.price - gp.shortage) / gp.contribution2 := by
  sorry

end NUMINAMATH_CALUDE_group_purchase_equation_l1624_162435


namespace NUMINAMATH_CALUDE_polynomial_coefficient_b_l1624_162457

-- Define the polynomial Q(x)
def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

-- State the theorem
theorem polynomial_coefficient_b (d b e : ℝ) :
  -- Conditions
  (∀ x, Q x d b e = 0 → -d/3 = x) ∧  -- Mean of zeros
  (∀ x y z, Q x d b e = 0 ∧ Q y d b e = 0 ∧ Q z d b e = 0 → x*y*z = -e) ∧  -- Product of zeros
  (-d/3 = 1 + d + b + e) ∧  -- Sum of coefficients equals mean of zeros
  (e = 6)  -- y-intercept is 6
  →
  b = -31 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_b_l1624_162457


namespace NUMINAMATH_CALUDE_medians_intersect_l1624_162475

/-- Definition of a triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of a point being the midpoint of a line segment -/
def isMidpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

/-- Definition of a median -/
def isMedian (P Q R S : ℝ × ℝ) : Prop :=
  isMidpoint S Q R

/-- Theorem: The medians of a triangle intersect at a single point -/
theorem medians_intersect (t : Triangle) 
  (A' : ℝ × ℝ) (B' : ℝ × ℝ) (C' : ℝ × ℝ)
  (h1 : isMidpoint A' t.B t.C)
  (h2 : isMidpoint B' t.C t.A)
  (h3 : isMidpoint C' t.A t.B)
  (h4 : isMedian t.A t.B t.C A')
  (h5 : isMedian t.B t.C t.A B')
  (h6 : isMedian t.C t.A t.B C') :
  ∃ G : ℝ × ℝ, (∃ k₁ k₂ k₃ : ℝ, 
    G = k₁ • t.A + (1 - k₁) • A' ∧
    G = k₂ • t.B + (1 - k₂) • B' ∧
    G = k₃ • t.C + (1 - k₃) • C') :=
  sorry


end NUMINAMATH_CALUDE_medians_intersect_l1624_162475


namespace NUMINAMATH_CALUDE_ellipse_equation_l1624_162419

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a > b
  h4 : a^2 = b^2 + c^2
  h5 : (4 : ℝ) / a^2 + 3 / b^2 = 1

/-- The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic progression. -/
def is_arithmetic_progression (e : Ellipse) : Prop :=
  ∃ (d : ℝ), 2 * e.a = 4 * e.c ∧ e.c > 0

theorem ellipse_equation (e : Ellipse) (h : is_arithmetic_progression e) :
  e.a = 2 * Real.sqrt 2 ∧ e.b = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1624_162419


namespace NUMINAMATH_CALUDE_total_guests_proof_l1624_162403

def number_of_guests (adults children seniors teenagers toddlers vip : ℕ) : ℕ :=
  adults + children + seniors + teenagers + toddlers + vip

theorem total_guests_proof :
  ∃ (adults children seniors teenagers toddlers vip : ℕ),
    adults = 58 ∧
    children = adults - 35 ∧
    seniors = 2 * children ∧
    teenagers = seniors - 15 ∧
    toddlers = teenagers / 2 ∧
    vip = teenagers - 20 ∧
    ∃ (n : ℕ), vip = n^2 ∧
    number_of_guests adults children seniors teenagers toddlers vip = 198 :=
by
  sorry

end NUMINAMATH_CALUDE_total_guests_proof_l1624_162403


namespace NUMINAMATH_CALUDE_intersection_when_m_zero_m_range_for_necessary_condition_l1624_162411

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define predicates p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part (I)
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (II)
theorem m_range_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_zero_m_range_for_necessary_condition_l1624_162411


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1624_162448

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1624_162448


namespace NUMINAMATH_CALUDE_average_change_after_removal_l1624_162484

def average_after_removal (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) : ℚ :=
  ((n : ℚ) * initial_avg - removed1 - removed2) / ((n - 2) : ℚ)

theorem average_change_after_removal :
  average_after_removal 50 38 45 55 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_average_change_after_removal_l1624_162484


namespace NUMINAMATH_CALUDE_seventeen_pairs_sold_l1624_162439

/-- Represents the sales data for an optometrist's contact lens business --/
structure ContactLensSales where
  soft_price : ℝ
  hard_price : ℝ
  soft_hard_difference : ℕ
  discount_rate : ℝ
  total_sales : ℝ

/-- Calculates the total number of contact lens pairs sold given the sales data --/
def total_pairs_sold (sales : ContactLensSales) : ℕ :=
  sorry

/-- Theorem stating that given the specific sales data, 17 pairs of lenses were sold --/
theorem seventeen_pairs_sold :
  let sales := ContactLensSales.mk 175 95 7 0.1 2469
  total_pairs_sold sales = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_pairs_sold_l1624_162439


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1624_162431

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 4) (hh : h = 5) :
  π * r * h = 20 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1624_162431


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1624_162415

theorem point_in_fourth_quadrant (a : ℤ) : 
  (2*a + 6 > 0) ∧ (3*a + 3 < 0) → (2*a + 6 = 2 ∧ 3*a + 3 = -3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1624_162415


namespace NUMINAMATH_CALUDE_dice_product_probability_dice_product_probability_proof_l1624_162483

/-- The probability of obtaining a product of 2 when tossing four standard dice -/
theorem dice_product_probability : ℝ :=
  let n_dice : ℕ := 4
  let dice_sides : ℕ := 6
  let target_product : ℕ := 2
  1 / 324

/-- Proof of the dice product probability theorem -/
theorem dice_product_probability_proof :
  dice_product_probability = 1 / 324 := by
  sorry

end NUMINAMATH_CALUDE_dice_product_probability_dice_product_probability_proof_l1624_162483


namespace NUMINAMATH_CALUDE_counterexample_condition_counterexample_existence_l1624_162440

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k ∧ n ≠ 0

theorem counterexample_condition (n : ℕ) : Prop :=
  n > 5 ∧
  ¬(n % 3 = 0) ∧
  ¬(∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q)

theorem counterexample_existence : 
  (∃ n : ℕ, counterexample_condition n) →
  ¬(∀ n : ℕ, n > 5 → ¬(n % 3 = 0) → 
    ∃ (p q : ℕ), IsPrime p ∧ IsPowerOfTwo q ∧ n = p + q) :=
by
  sorry

end NUMINAMATH_CALUDE_counterexample_condition_counterexample_existence_l1624_162440


namespace NUMINAMATH_CALUDE_ball_return_to_start_l1624_162410

theorem ball_return_to_start (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) : 
  ∃ m : ℕ, m > 0 ∧ (m * k) % n = 0 ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_ball_return_to_start_l1624_162410


namespace NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l1624_162413

def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∀ x y : ℝ, 3 * x^2 + 9 * y^2 - 12 * x + 18 * y = k → k > -21

theorem non_degenerate_ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -21 := by sorry

end NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l1624_162413


namespace NUMINAMATH_CALUDE_randy_spends_two_dollars_per_trip_l1624_162422

/-- Calculates the amount spent per store trip -/
def amount_per_trip (initial_amount final_amount trips_per_month months : ℕ) : ℚ :=
  (initial_amount - final_amount : ℚ) / (trips_per_month * months : ℚ)

/-- Theorem: Randy spends $2 per store trip -/
theorem randy_spends_two_dollars_per_trip :
  amount_per_trip 200 104 4 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_spends_two_dollars_per_trip_l1624_162422


namespace NUMINAMATH_CALUDE_connectivity_determination_bound_l1624_162480

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  adj : Fin n → Fin n → Bool

/-- Distance between two vertices in a graph -/
def distance (G : Graph n) (u v : Fin n) : ℕ := sorry

/-- Whether a graph is connected -/
def is_connected (G : Graph n) : Prop := sorry

/-- A query about the distance between two vertices -/
structure Query (n : ℕ) where
  u : Fin n
  v : Fin n

/-- Result of a query -/
inductive QueryResult
  | LessThan
  | EqualTo
  | GreaterThan

/-- Function to determine if a graph is connected using queries -/
def determine_connectivity (n k : ℕ) (h : k ≤ n) (G : Graph n) : 
  ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := sorry

/-- Main theorem -/
theorem connectivity_determination_bound (n k : ℕ) (h : k ≤ n) :
  ∀ G : Graph n, ∃ (queries : List (Query n)), 
    queries.length ≤ 2 * n^2 / k ∧ 
    (∀ q : Query n, q ∈ queries → ∃ r : QueryResult, r = sorry) → 
    ∃ b : Bool, b = is_connected G := by
  sorry

end NUMINAMATH_CALUDE_connectivity_determination_bound_l1624_162480


namespace NUMINAMATH_CALUDE_breakfast_expectation_l1624_162495

/-- Represents the possible outcomes of rolling a fair six-sided die, excluding 1 (which leads to a reroll) -/
inductive DieOutcome
| two
| three
| four
| five
| six

/-- The probability of rolling an even number (2, 4, or 6) after accounting for rerolls on 1 -/
def prob_even : ℚ := 3/5

/-- The probability of rolling an odd number (3 or 5) after accounting for rerolls on 1 -/
def prob_odd : ℚ := 2/5

/-- The number of days in a non-leap year -/
def days_in_year : ℕ := 365

/-- The expected difference between days eating pancakes and days eating oatmeal -/
def expected_difference : ℚ := prob_even * days_in_year - prob_odd * days_in_year

theorem breakfast_expectation :
  expected_difference = 73 := by sorry

end NUMINAMATH_CALUDE_breakfast_expectation_l1624_162495


namespace NUMINAMATH_CALUDE_ratio_equals_three_tenths_l1624_162465

-- Define the system of equations
def system (k x y z w : ℝ) : Prop :=
  x + 2*k*y + 4*z - w = 0 ∧
  4*x + k*y + 2*z + w = 0 ∧
  3*x + 5*y - 3*z + 2*w = 0 ∧
  2*x + 3*y + z - 4*w = 0

-- Theorem statement
theorem ratio_equals_three_tenths :
  ∃ (k x y z w : ℝ), 
    system k x y z w ∧ 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
    x * y / (z * w) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_three_tenths_l1624_162465


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1624_162478

theorem decimal_to_fraction :
  (3.375 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1624_162478


namespace NUMINAMATH_CALUDE_currency_notes_total_l1624_162433

/-- Proves that given the specified conditions, the total amount of currency notes is 5000 rupees. -/
theorem currency_notes_total (total_notes : ℕ) (amount_50 : ℕ) : 
  total_notes = 85 →
  amount_50 = 3500 →
  ∃ (notes_100 notes_50 : ℕ),
    notes_100 + notes_50 = total_notes ∧
    50 * notes_50 = amount_50 ∧
    100 * notes_100 + 50 * notes_50 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_currency_notes_total_l1624_162433


namespace NUMINAMATH_CALUDE_omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l1624_162462

-- Define the complex number ω as a function of m
def ω (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 - m - 12)

-- Theorem 1: ω is real iff m = 4 or m = -3
theorem omega_real_iff_m_eq_4_or_neg_3 (m : ℝ) :
  ω m ∈ Set.range Complex.ofReal ↔ m = 4 ∨ m = -3 :=
sorry

-- Theorem 2: ω is in the fourth quadrant iff 3 < m < 4
theorem omega_in_fourth_quadrant_iff_3_lt_m_lt_4 (m : ℝ) :
  (Complex.re (ω m) > 0 ∧ Complex.im (ω m) < 0) ↔ 3 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_omega_real_iff_m_eq_4_or_neg_3_omega_in_fourth_quadrant_iff_3_lt_m_lt_4_l1624_162462


namespace NUMINAMATH_CALUDE_cows_eating_grass_l1624_162459

-- Define the amount of hectares a cow eats per week
def cow_eat_rate : ℚ := 1/2

-- Define the amount of hectares of grass that grows per week
def grass_growth_rate : ℚ := 1/2

-- Define the function that calculates the amount of grass eaten
def grass_eaten (cows : ℕ) (weeks : ℕ) : ℚ :=
  (cows : ℚ) * cow_eat_rate * (weeks : ℚ)

-- Define the function that calculates the amount of grass regrown
def grass_regrown (hectares : ℕ) (weeks : ℕ) : ℚ :=
  (hectares : ℚ) * grass_growth_rate * (weeks : ℚ)

-- Theorem statement
theorem cows_eating_grass (cows : ℕ) : 
  (grass_eaten 3 2 - grass_regrown 2 2 = 2) →
  (grass_eaten 2 4 - grass_regrown 2 4 = 2) →
  (grass_eaten cows 6 - grass_regrown 6 6 = 6) →
  cows = 3 := by
  sorry

end NUMINAMATH_CALUDE_cows_eating_grass_l1624_162459


namespace NUMINAMATH_CALUDE_odd_squares_sum_representation_l1624_162471

theorem odd_squares_sum_representation (k n : ℕ) (h : k ≠ n) :
  ((2 * k + 1)^2 + (2 * n + 1)^2) / 2 = (k + n + 1)^2 + (k - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_sum_representation_l1624_162471


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l1624_162417

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_N_complement_M : N ∩ (Mᶜ) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l1624_162417


namespace NUMINAMATH_CALUDE_subset_ratio_eight_elements_l1624_162499

theorem subset_ratio_eight_elements :
  let n : ℕ := 8
  let S : ℕ := 2^n
  let T : ℕ := n.choose 3
  (T : ℚ) / S = 7 / 32 := by
sorry

end NUMINAMATH_CALUDE_subset_ratio_eight_elements_l1624_162499


namespace NUMINAMATH_CALUDE_julie_leftover_money_l1624_162470

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def lawns_to_mow : ℕ := 20
def lawn_pay : ℕ := 20
def newspapers_to_deliver : ℕ := 600
def newspaper_pay : ℚ := 40/100
def dogs_to_walk : ℕ := 24
def dog_walk_pay : ℕ := 15

theorem julie_leftover_money :
  let total_earnings := lawns_to_mow * lawn_pay + 
                        (newspapers_to_deliver : ℚ) * newspaper_pay + 
                        dogs_to_walk * dog_walk_pay
  let total_money := (initial_savings : ℚ) + total_earnings
  let leftover := total_money - bike_cost
  leftover = 155 := by sorry

end NUMINAMATH_CALUDE_julie_leftover_money_l1624_162470


namespace NUMINAMATH_CALUDE_savings_calculation_l1624_162498

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given an income of 18000 and an income-to-expenditure ratio of 5:4, the savings are 3600 --/
theorem savings_calculation :
  calculate_savings 18000 5 4 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l1624_162498


namespace NUMINAMATH_CALUDE_trajectory_and_tangent_lines_l1624_162461

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the projection line
def projection_line (x : ℝ) : Prop := x = 3

-- Define the point P
def point_P (M N : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 = M.1 + N.1 - 0 ∧ P.2 = M.2 + N.2 - 0

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (1, 4)

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem trajectory_and_tangent_lines :
  ∀ (M N P : ℝ × ℝ),
    ellipse M.1 M.2 →
    projection_line N.1 →
    point_P M N P →
    (∀ (x y : ℝ), P = (x, y) → trajectory_E x y) ∧
    (∃ (x y : ℝ), (x, y) = point_A ∧ 
      (tangent_line_1 x ∨ tangent_line_2 x y) ∧
      (∀ (t : ℝ), trajectory_E (x + t) (y + t) → t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_tangent_lines_l1624_162461


namespace NUMINAMATH_CALUDE_inequality_proof_l1624_162487

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 + 1 / b^2 ≥ 8) ∧ (1 / a + 1 / b + 1 / (a * b) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1624_162487


namespace NUMINAMATH_CALUDE_james_run_duration_l1624_162486

/-- Calculates the duration of James' run in minutes -/
def run_duration (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) (cal_per_min : ℕ) (excess_cal : ℕ) : ℕ :=
  let total_oz := bags * oz_per_bag
  let total_cal := total_oz * cal_per_oz
  let cal_to_burn := total_cal - excess_cal
  cal_to_burn / cal_per_min

/-- Proves that James' run duration is 40 minutes given the problem conditions -/
theorem james_run_duration :
  run_duration 3 2 150 12 420 = 40 := by
  sorry

end NUMINAMATH_CALUDE_james_run_duration_l1624_162486


namespace NUMINAMATH_CALUDE_series_sum_equals_one_third_l1624_162406

/-- The sum of the infinite series ∑(k=1 to ∞) [2^k / (8^k - 1)] is equal to 1/3 -/
theorem series_sum_equals_one_third :
  ∑' k, (2 : ℝ)^k / ((8 : ℝ)^k - 1) = 1/3 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_third_l1624_162406


namespace NUMINAMATH_CALUDE_coefficient_x4_is_correct_l1624_162408

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ :=
  4 * (x^4 - 2*x^5) + 3 * (x^3 - 3*x^4 + 2*x^6) - (5*x^5 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -3

theorem coefficient_x4_is_correct :
  (deriv (deriv (deriv (deriv expression)))) 0 / 24 = coefficient_x4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_is_correct_l1624_162408


namespace NUMINAMATH_CALUDE_money_distribution_l1624_162414

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : B + C = 340)
  (h3 : C = 40) :
  A + C = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1624_162414


namespace NUMINAMATH_CALUDE_apple_distribution_l1624_162420

/-- The number of apples to be distributed -/
def total_apples : ℕ := 30

/-- The number of people receiving apples -/
def num_people : ℕ := 3

/-- The minimum number of apples each person must receive -/
def min_apples : ℕ := 3

/-- The number of ways to distribute the apples -/
def distribution_ways : ℕ := (total_apples - num_people * min_apples + num_people - 1).choose (num_people - 1)

theorem apple_distribution :
  distribution_ways = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1624_162420


namespace NUMINAMATH_CALUDE_overlap_area_63_l1624_162446

/-- Represents the geometric shapes and their movement --/
structure GeometricSetup where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  initial_distance : ℝ
  relative_speed : ℝ

/-- Calculates the overlapping area at a given time --/
def overlapping_area (setup : GeometricSetup) (t : ℝ) : ℝ :=
  sorry

/-- The main theorem stating when the overlapping area is 63 square centimeters --/
theorem overlap_area_63 (setup : GeometricSetup) 
  (h1 : setup.square_side = 12)
  (h2 : setup.triangle_hypotenuse = 18)
  (h3 : setup.initial_distance = 13)
  (h4 : setup.relative_speed = 5) :
  (∃ t : ℝ, t = 5 ∨ t = 6.2) ∧ (overlapping_area setup t = 63) :=
sorry

end NUMINAMATH_CALUDE_overlap_area_63_l1624_162446


namespace NUMINAMATH_CALUDE_base_for_four_digit_256_l1624_162492

theorem base_for_four_digit_256 : ∃! b : ℕ, 
  b > 1 ∧ b^3 ≤ 256 ∧ 256 < b^4 :=
by sorry

end NUMINAMATH_CALUDE_base_for_four_digit_256_l1624_162492


namespace NUMINAMATH_CALUDE_range_of_f_l1624_162488

def f (x : ℤ) : ℤ := x^2 + 2*x

def domain : Set ℤ := {x | -2 ≤ x ∧ x ≤ 1}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1624_162488


namespace NUMINAMATH_CALUDE_larger_number_proof_l1624_162424

/-- Given two positive integers with specific HCF and LCM, prove the larger one is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1624_162424


namespace NUMINAMATH_CALUDE_f_2_neg3_neg1_eq_half_l1624_162463

def f (a b c : ℚ) : ℚ := (c + a) / (c - b)

theorem f_2_neg3_neg1_eq_half : f 2 (-3) (-1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_2_neg3_neg1_eq_half_l1624_162463


namespace NUMINAMATH_CALUDE_range_of_a_l1624_162451

/-- The function f(x) = x³ + x + 1 -/
def f (x : ℝ) : ℝ := x^3 + x + 1

/-- Theorem stating that if f(x² + a) + f(ax) > 2 for all x, then 0 < a < 4 -/
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a*x) > 2) → 0 < a ∧ a < 4 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1624_162451


namespace NUMINAMATH_CALUDE_average_of_numbers_l1624_162404

def numbers : List ℤ := [-5, -2, 0, 4, 8]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1624_162404


namespace NUMINAMATH_CALUDE_product_equality_l1624_162453

theorem product_equality (h : 213 * 16 = 3408) : 16 * 21.3 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1624_162453


namespace NUMINAMATH_CALUDE_log_expressibility_l1624_162425

-- Define the given logarithm values
noncomputable def log10_5 : ℝ := 0.6990
noncomputable def log10_7 : ℝ := 0.8451

-- Define a function to represent expressibility using given logarithms
def expressible (x : ℝ) : Prop :=
  ∃ (a b c : ℚ), x = a * log10_5 + b * log10_7 + c

-- Theorem statement
theorem log_expressibility :
  (¬ expressible (Real.log 27 / Real.log 10)) ∧
  (¬ expressible (Real.log 21 / Real.log 10)) ∧
  (expressible (Real.log (Real.sqrt 35) / Real.log 10)) ∧
  (expressible (Real.log 1000 / Real.log 10)) ∧
  (expressible (Real.log 0.2 / Real.log 10)) :=
sorry

end NUMINAMATH_CALUDE_log_expressibility_l1624_162425


namespace NUMINAMATH_CALUDE_linear_function_k_value_l1624_162447

/-- Given a linear function y = kx - 2 that passes through the point (-1, 3), prove that k = -5 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x - 2) → -- The function is y = kx - 2
  (3 : ℝ) = k * (-1 : ℝ) - 2 → -- The function passes through the point (-1, 3)
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l1624_162447


namespace NUMINAMATH_CALUDE_expected_successes_bernoulli_l1624_162441

/-- The expected number of successes in 2N Bernoulli trials with p = 0.5 is N -/
theorem expected_successes_bernoulli (N : ℕ) : 
  let n := 2 * N
  let p := (1 : ℝ) / 2
  n * p = N := by sorry

end NUMINAMATH_CALUDE_expected_successes_bernoulli_l1624_162441


namespace NUMINAMATH_CALUDE_product_586645_9999_l1624_162468

theorem product_586645_9999 : 586645 * 9999 = 5865885355 := by
  sorry

end NUMINAMATH_CALUDE_product_586645_9999_l1624_162468
