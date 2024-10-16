import Mathlib

namespace NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l195_19580

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_fifth_power_l195_19580


namespace NUMINAMATH_CALUDE_intersection_of_sets_l195_19525

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {1, 2, 4, 6}
  A ∩ B = {1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l195_19525


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l195_19534

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), (6 * s^2 = 294) → (s^3 = 343) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l195_19534


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_exceeds_1000_l195_19547

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_exceeds_1000 : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_exceeds_1000_l195_19547


namespace NUMINAMATH_CALUDE_profit_in_toys_l195_19575

/-- 
Given:
- A man sold 18 toys for Rs. 18900
- The cost price of a toy is Rs. 900
Prove that the number of toys' cost price gained as profit is 3
-/
theorem profit_in_toys (total_toys : ℕ) (selling_price : ℕ) (cost_per_toy : ℕ) :
  total_toys = 18 →
  selling_price = 18900 →
  cost_per_toy = 900 →
  (selling_price - total_toys * cost_per_toy) / cost_per_toy = 3 :=
by sorry

end NUMINAMATH_CALUDE_profit_in_toys_l195_19575


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_second_or_third_quadrant_l195_19536

/-- A linear function f(x) = kx + b is monotonically decreasing on ℝ -/
def is_monotonically_decreasing (k b : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → k * x + b > k * y + b

/-- The point (k, b) is in the second or third quadrant -/
def is_in_second_or_third_quadrant (k b : ℝ) : Prop :=
  k < 0 ∧ (b > 0 ∨ b < 0)

/-- If a linear function y = kx + b is monotonically decreasing on ℝ,
    then the point (k, b) is in the second or third quadrant -/
theorem monotonically_decreasing_implies_second_or_third_quadrant (k b : ℝ) :
  is_monotonically_decreasing k b → is_in_second_or_third_quadrant k b :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_second_or_third_quadrant_l195_19536


namespace NUMINAMATH_CALUDE_f_max_value_l195_19582

open Real

-- Define the function f
def f (x : ℝ) := (3 + 2*x)^3 * (4 - x)^4

-- Define the interval
def I : Set ℝ := {x | -3/2 < x ∧ x < 4}

-- State the theorem
theorem f_max_value :
  ∃ (x_max : ℝ), x_max ∈ I ∧
  f x_max = 432 * (11/7)^7 ∧
  x_max = 6/7 ∧
  ∀ (x : ℝ), x ∈ I → f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l195_19582


namespace NUMINAMATH_CALUDE_beverage_mix_ratio_l195_19558

theorem beverage_mix_ratio : 
  ∀ (x y : ℝ), 
  x > 0 → y > 0 →
  (5 * x + 4 * y = 5.5 * x + 3.6 * y) →
  (x / y = 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_beverage_mix_ratio_l195_19558


namespace NUMINAMATH_CALUDE_problem_statement_l195_19537

theorem problem_statement :
  (∀ x : ℝ, x < 0 → (2 : ℝ)^x > (3 : ℝ)^x) ∧
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x > x^3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l195_19537


namespace NUMINAMATH_CALUDE_division_expression_equality_l195_19578

theorem division_expression_equality : 
  (1 : ℚ) / 12 / ((1 : ℚ) / 3 - (1 : ℚ) / 4 - (5 : ℚ) / 12) = -(1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_division_expression_equality_l195_19578


namespace NUMINAMATH_CALUDE_game_probabilities_l195_19541

/-- Represents the number of balls of each color in the bag -/
def num_balls_per_color : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 3 * num_balls_per_color

/-- Represents the number of balls drawn in each game -/
def balls_drawn : ℕ := 3

/-- Represents the number of people participating in the game -/
def num_participants : ℕ := 3

/-- Calculates the probability of winning for one person -/
def prob_win : ℚ := 2 / 5

/-- Calculates the probability that exactly 1 person wins out of 3 -/
def prob_one_winner : ℚ := 54 / 125

theorem game_probabilities :
  (prob_win = 2 / 5) ∧
  (prob_one_winner = 54 / 125) := by
  sorry

end NUMINAMATH_CALUDE_game_probabilities_l195_19541


namespace NUMINAMATH_CALUDE_negation_of_implication_l195_19594

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y ≤ 0 → x ≤ 0 ∨ y ≤ 0) ↔ (x + y > 0 → x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l195_19594


namespace NUMINAMATH_CALUDE_soccer_team_math_enrollment_l195_19510

theorem soccer_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 15 →
  both_subjects = 6 →
  ∃ (math_players : ℕ), math_players = 16 ∧ 
    total_players = physics_players + math_players - both_subjects :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_math_enrollment_l195_19510


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l195_19538

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original GDP value in trillions of dollars -/
def originalGDP : ℝ := 1.337

/-- The number of significant figures to use -/
def sigFigs : ℕ := 3

theorem gdp_scientific_notation :
  toScientificNotation (originalGDP * 1000000000000) sigFigs =
    ScientificNotation.mk 1.34 12 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l195_19538


namespace NUMINAMATH_CALUDE_sum_of_integers_l195_19502

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 10)
  (eq2 : y - z + w = 15)
  (eq3 : z - w + x = 9)
  (eq4 : w - x + y = 4) :
  x + y + z + w = 38 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l195_19502


namespace NUMINAMATH_CALUDE_birdseed_solution_l195_19569

/-- The number of boxes of birdseed Leah already had in the pantry -/
def birdseed_problem (new_boxes : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) 
  (box_content : ℕ) (weeks : ℕ) : ℕ :=
  let total_consumption := parrot_consumption + cockatiel_consumption
  let total_needed := total_consumption * weeks
  let total_boxes := (total_needed + box_content - 1) / box_content
  total_boxes - new_boxes

/-- Theorem stating the solution to the birdseed problem -/
theorem birdseed_solution : 
  birdseed_problem 3 100 50 225 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_birdseed_solution_l195_19569


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l195_19560

/-- Given a hostel with students and mess expenses, calculate the original expenditure --/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (student_increase : ℕ) 
  (expense_increase : ℕ) 
  (avg_expense_decrease : ℕ) 
  (h1 : initial_students = 35)
  (h2 : student_increase = 7)
  (h3 : expense_increase = 42)
  (h4 : avg_expense_decrease = 1) :
  ∃ (original_expenditure : ℕ), 
    original_expenditure = initial_students * 
      ((initial_students + student_increase) * 
        (original_expenditure / initial_students - avg_expense_decrease) - 
      original_expenditure) / student_increase ∧
    original_expenditure = 420 :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l195_19560


namespace NUMINAMATH_CALUDE_time_difference_walk_vs_bicycle_l195_19564

/-- Represents the number of blocks from Henrikh's home to his office -/
def distance : ℕ := 12

/-- Represents the time in minutes to walk one block -/
def walkingTimePerBlock : ℚ := 1

/-- Represents the time in minutes to ride a bicycle for one block -/
def bicycleTimePerBlock : ℚ := 20 / 60

/-- Calculates the total time to travel the distance by walking -/
def walkingTime : ℚ := distance * walkingTimePerBlock

/-- Calculates the total time to travel the distance by bicycle -/
def bicycleTime : ℚ := distance * bicycleTimePerBlock

theorem time_difference_walk_vs_bicycle :
  walkingTime - bicycleTime = 8 := by sorry

end NUMINAMATH_CALUDE_time_difference_walk_vs_bicycle_l195_19564


namespace NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l195_19563

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬(∃ (d : ℕ), d > 1 ∧ d ∣ (k - 20) ∧ d ∣ (7 * k + 2))) ∧
  (∃ (d : ℕ), d > 1 ∧ d ∣ (n - 20) ∧ d ∣ (7 * n + 2)) ∧
  n = 22 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l195_19563


namespace NUMINAMATH_CALUDE_mango_count_l195_19515

/-- The number of mangoes in all boxes -/
def total_mangoes (boxes : ℕ) (dozen_per_box : ℕ) : ℕ :=
  boxes * dozen_per_box * 12

/-- Proof that there are 4320 mangoes in 36 boxes with 10 dozen mangoes each -/
theorem mango_count : total_mangoes 36 10 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_mango_count_l195_19515


namespace NUMINAMATH_CALUDE_christel_gave_five_dolls_l195_19551

/-- The number of dolls Christel gave to Andrena -/
def dolls_given_by_christel : ℕ := sorry

theorem christel_gave_five_dolls :
  let debelyn_initial := 20
  let debelyn_gave := 2
  let christel_initial := 24
  let andrena_more_than_christel := 2
  let andrena_more_than_debelyn := 3
  dolls_given_by_christel = 5 := by sorry

end NUMINAMATH_CALUDE_christel_gave_five_dolls_l195_19551


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l195_19544

theorem complex_number_quadrant (m : ℝ) (z : ℂ) 
  (h1 : 2/3 < m) (h2 : m < 1) (h3 : z = Complex.mk (3*m - 2) (m - 1)) : 
  0 < z.re ∧ z.re < 1 ∧ -1/3 < z.im ∧ z.im < 0 :=
by sorry

#check complex_number_quadrant

end NUMINAMATH_CALUDE_complex_number_quadrant_l195_19544


namespace NUMINAMATH_CALUDE_train_length_l195_19546

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 60 → time = 18 → 
  ∃ length : ℝ, abs (length - 300.06) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l195_19546


namespace NUMINAMATH_CALUDE_download_speed_calculation_l195_19566

theorem download_speed_calculation (file_size : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  file_size = 600 ∧ speed_ratio = 15 ∧ time_diff = 140 →
  ∃ (speed_4g : ℝ) (speed_5g : ℝ),
    speed_5g = speed_ratio * speed_4g ∧
    file_size / speed_4g - file_size / speed_5g = time_diff ∧
    speed_4g = 4 ∧ speed_5g = 60 := by
  sorry

end NUMINAMATH_CALUDE_download_speed_calculation_l195_19566


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_function_l195_19568

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the property of f being even when shifted by 2
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x + 2) = f (x + 2)

-- Define the symmetry axis of a function
def symmetry_axis (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- Theorem statement
theorem symmetry_of_shifted_function (f : ℝ → ℝ) 
  (h : is_even_shifted f) : 
  symmetry_axis (fun x ↦ f (x - 1) + 2) 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_function_l195_19568


namespace NUMINAMATH_CALUDE_cafe_chairs_count_l195_19557

/-- Calculates the total number of chairs in a cafe given the number of indoor and outdoor tables
    and the number of chairs per table type. -/
def total_chairs (indoor_tables : ℕ) (outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Theorem stating that the total number of chairs in the cafe is 123. -/
theorem cafe_chairs_count :
  total_chairs 9 11 10 3 = 123 := by
  sorry

#eval total_chairs 9 11 10 3

end NUMINAMATH_CALUDE_cafe_chairs_count_l195_19557


namespace NUMINAMATH_CALUDE_constant_function_no_monotonicity_l195_19549

open Function Set

theorem constant_function_no_monotonicity 
  {f : ℝ → ℝ} {I : Set ℝ} (hI : Interval I) :
  (∀ x ∈ I, HasDerivAt f (0 : ℝ) x) → 
  ∃ c, ∀ x ∈ I, f x = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_no_monotonicity_l195_19549


namespace NUMINAMATH_CALUDE_chocolate_eggs_duration_l195_19561

/-- Proves that given 40 chocolate eggs, eating 2 eggs per day for 5 days a week will result in the eggs lasting for 4 weeks. -/
theorem chocolate_eggs_duration (total_eggs : ℕ) (eggs_per_day : ℕ) (school_days_per_week : ℕ) : 
  total_eggs = 40 → 
  eggs_per_day = 2 → 
  school_days_per_week = 5 → 
  (total_eggs / (eggs_per_day * school_days_per_week) : ℚ) = 4 := by
sorry


end NUMINAMATH_CALUDE_chocolate_eggs_duration_l195_19561


namespace NUMINAMATH_CALUDE_no_real_roots_for_distinct_abc_l195_19526

theorem no_real_roots_for_distinct_abc (a b c : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) : 
  let discriminant := 4 * (a + b + c)^2 - 12 * (a^2 + b^2 + c^2)
  discriminant < 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_distinct_abc_l195_19526


namespace NUMINAMATH_CALUDE_charlottes_distance_l195_19508

/-- The distance between Charlotte's home and school -/
def distance : ℝ := 60

/-- The time taken for Charlotte's one-way journey in hours -/
def journey_time : ℝ := 6

/-- Charlotte's average speed in miles per hour -/
def average_speed : ℝ := 10

/-- Theorem stating that the distance is equal to the product of average speed and journey time -/
theorem charlottes_distance : distance = average_speed * journey_time := by
  sorry

end NUMINAMATH_CALUDE_charlottes_distance_l195_19508


namespace NUMINAMATH_CALUDE_triangle_area_product_l195_19545

/-- Given positive real numbers a and b, and a triangle in the first quadrant
    bounded by the coordinate axes and the line ax + by = 6 with area 6,
    prove that ab = 3. -/
theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ a * x + b * y = 6) →
  ((1 / 2) * (6 / a) * (6 / b) = 6) →
  a * b = 3 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_product_l195_19545


namespace NUMINAMATH_CALUDE_multiply_powers_l195_19507

theorem multiply_powers (a : ℝ) : 6 * a^2 * (1/2 * a^3) = 3 * a^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_l195_19507


namespace NUMINAMATH_CALUDE_parabola_translation_l195_19527

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_translation (p : Parabola) :
  p.a = -1/3 ∧ p.h = 5 ∧ p.k = 3 →
  let p' := translate p 5 3
  p'.a = -1/3 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l195_19527


namespace NUMINAMATH_CALUDE_max_area_difference_l195_19535

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- One focus of the ellipse -/
def Focus : ℝ × ℝ := (-1, 0)

/-- S₁ is the area of triangle ABD -/
noncomputable def S₁ (A B D : ℝ × ℝ) : ℝ := sorry

/-- S₂ is the area of triangle ABC -/
noncomputable def S₂ (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum difference between S₁ and S₂ -/
theorem max_area_difference :
  ∃ (A B C D : ℝ × ℝ),
    Ellipse A.1 A.2 ∧ Ellipse B.1 B.2 ∧ Ellipse C.1 C.2 ∧ Ellipse D.1 D.2 ∧
    (∀ (E : ℝ × ℝ), Ellipse E.1 E.2 → |S₁ A B D - S₂ A B C| ≤ Real.sqrt 3) ∧
    (∃ (F G : ℝ × ℝ), Ellipse F.1 F.2 ∧ Ellipse G.1 G.2 ∧ |S₁ A B F - S₂ A B G| = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l195_19535


namespace NUMINAMATH_CALUDE_bret_frog_count_l195_19576

theorem bret_frog_count :
  ∀ (alster_frogs quinn_frogs bret_frogs : ℕ),
    alster_frogs = 2 →
    quinn_frogs = 2 * alster_frogs →
    bret_frogs = 3 * quinn_frogs →
    bret_frogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_bret_frog_count_l195_19576


namespace NUMINAMATH_CALUDE_johns_total_spending_johns_spending_proof_l195_19501

/-- Calculate John's total spending on a phone and accessories, including sales tax -/
theorem johns_total_spending (online_price : ℝ) (price_increase_rate : ℝ) 
  (accessory_discount_rate : ℝ) (case_price : ℝ) (protector_price : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let store_phone_price := online_price * (1 + price_increase_rate)
  let accessories_regular_price := case_price + protector_price
  let accessories_discounted_price := accessories_regular_price * (1 - accessory_discount_rate)
  let subtotal := store_phone_price + accessories_discounted_price
  let total_with_tax := subtotal * (1 + sales_tax_rate)
  total_with_tax

/-- Proof that John's total spending is $2212.75 -/
theorem johns_spending_proof : 
  johns_total_spending 2000 0.02 0.05 35 15 0.06 = 2212.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_spending_johns_spending_proof_l195_19501


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l195_19593

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a (k + 1) = 3 * a k) →  -- Common ratio is 3
  (∀ k, a k > 0) →  -- Positive terms
  a m * a n = 9 * a 2 ^ 2 →  -- Given condition
  (∀ p q : ℕ, a p * a q = 9 * a 2 ^ 2 → 2 / m + 1 / (2 * n) ≤ 2 / p + 1 / (2 * q)) →
  2 / m + 1 / (2 * n) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l195_19593


namespace NUMINAMATH_CALUDE_optimal_selling_price_l195_19572

/-- Represents the annual profit function for a clothing distributor -/
def annual_profit (x : ℝ) : ℝ := -x^2 + 1000*x - 200000

/-- Represents the annual sales volume function -/
def sales_volume (x : ℝ) : ℝ := 800 - x

theorem optimal_selling_price :
  ∃ (x : ℝ),
    x = 400 ∧
    annual_profit x = 40000 ∧
    ∀ y, y ≠ x → annual_profit y = 40000 → sales_volume x > sales_volume y :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l195_19572


namespace NUMINAMATH_CALUDE_sequence_properties_l195_19579

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := 33 * n - n^2

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℤ := 34 - 2 * n

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (a 1 = 32) ∧
  (∀ n : ℕ, a (n+1) - a n = -2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l195_19579


namespace NUMINAMATH_CALUDE_contrapositive_zero_product_l195_19590

theorem contrapositive_zero_product (a b : ℝ) : a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_zero_product_l195_19590


namespace NUMINAMATH_CALUDE_technicians_avg_salary_is_900_l195_19585

/-- Represents the workshop scenario with workers and salaries -/
structure Workshop where
  total_workers : ℕ
  avg_salary_all : ℕ
  num_technicians : ℕ
  avg_salary_non_tech : ℕ

/-- Calculates the average salary of technicians given workshop data -/
def avg_salary_technicians (w : Workshop) : ℕ :=
  let total_salary := w.total_workers * w.avg_salary_all
  let non_tech_workers := w.total_workers - w.num_technicians
  let non_tech_salary := non_tech_workers * w.avg_salary_non_tech
  let tech_salary := total_salary - non_tech_salary
  tech_salary / w.num_technicians

/-- Theorem stating that the average salary of technicians is 900 given the workshop conditions -/
theorem technicians_avg_salary_is_900 (w : Workshop) 
  (h1 : w.total_workers = 20)
  (h2 : w.avg_salary_all = 750)
  (h3 : w.num_technicians = 5)
  (h4 : w.avg_salary_non_tech = 700) :
  avg_salary_technicians w = 900 := by
  sorry

#eval avg_salary_technicians ⟨20, 750, 5, 700⟩

end NUMINAMATH_CALUDE_technicians_avg_salary_is_900_l195_19585


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l195_19539

/-- If the line x + y = b is the perpendicular bisector of the line segment from (2,4) to (6,10), then b = 11 -/
theorem perpendicular_bisector_value (b : ℝ) : 
  (∀ (x y : ℝ), x + y = b ↔ 
    ((x - 4)^2 + (y - 7)^2 = (2 - 4)^2 + (4 - 7)^2 ∧ 
     (x - 4)^2 + (y - 7)^2 = (6 - 4)^2 + (10 - 7)^2)) → 
  b = 11 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l195_19539


namespace NUMINAMATH_CALUDE_twentieth_fisherman_catch_l195_19509

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_nineteen : ℕ) 
  (h1 : total_fishermen = 20)
  (h2 : total_fish = 10000)
  (h3 : fish_per_nineteen = 400) :
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_fisherman_catch_l195_19509


namespace NUMINAMATH_CALUDE_inequality_proof_l195_19506

theorem inequality_proof (a b c : ℝ) : (1/4) * a^2 + b^2 + c^2 ≥ a*b - a*c + 2*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l195_19506


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l195_19553

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) is 2,
    given that one of its asymptotes is tangent to the circle (x - √3)² + (y - 1)² = 1. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ∧
    ((x - Real.sqrt 3)^2 + (y - 1)^2 = 1 ∨
     (x + Real.sqrt 3)^2 + (y - 1)^2 = 1)) →
  Real.sqrt ((a^2 + b^2) / a^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l195_19553


namespace NUMINAMATH_CALUDE_cubic_sum_equals_one_l195_19595

theorem cubic_sum_equals_one (a b : ℝ) (h : a + b = 1) : a^3 + 3*a*b + b^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_one_l195_19595


namespace NUMINAMATH_CALUDE_range_of_x_l195_19581

theorem range_of_x (x : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 9/a + 1/b = 2 ∧ 
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' ≥ x^2 + 2*x)) → 
  -4 ≤ x ∧ x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l195_19581


namespace NUMINAMATH_CALUDE_largest_expression_l195_19583

theorem largest_expression : 
  let a := 15847
  let b := 3174
  let expr1 := a + 1 / b
  let expr2 := a - 1 / b
  let expr3 := a * (1 / b)
  let expr4 := a / (1 / b)
  let expr5 := a ^ 1.03174
  (expr4 > expr1) ∧ 
  (expr4 > expr2) ∧ 
  (expr4 > expr3) ∧ 
  (expr4 > expr5) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l195_19583


namespace NUMINAMATH_CALUDE_inverse_of_A_l195_19511

-- Define matrix A
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

-- Define the proposed inverse of A
def A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1/2, 0; -1/16, 1/8]

-- Theorem statement
theorem inverse_of_A : A⁻¹ = A_inv := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l195_19511


namespace NUMINAMATH_CALUDE_road_building_divisibility_l195_19567

/-- Represents the number of ways to build roads between n cities with the given constraints -/
def T (n : ℕ) : ℕ :=
  sorry  -- Definition of T_n based on the problem constraints

/-- The main theorem to be proved -/
theorem road_building_divisibility (n : ℕ) (h : n > 1) :
  (n % 2 = 1 → n ∣ T n) ∧ (n % 2 = 0 → (n / 2) ∣ T n) :=
by sorry

end NUMINAMATH_CALUDE_road_building_divisibility_l195_19567


namespace NUMINAMATH_CALUDE_kendall_change_total_l195_19598

/-- Represents the value of coins in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℕ :=
  (coin_value coin) * count

/-- Theorem stating the total amount of money Kendall has in change -/
theorem kendall_change_total : 
  coin_total "quarter" 10 + coin_total "dime" 12 + coin_total "nickel" 6 = 400 := by
  sorry

end NUMINAMATH_CALUDE_kendall_change_total_l195_19598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l195_19597

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 1 + a 2 + a 3 = 21
  product_property : a 1 * a 2 * a 3 = 231

/-- Theorem about the second term and general formula of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 2 = 7) ∧
  ((∀ n, seq.a n = -4 * n + 15) ∨ (∀ n, seq.a n = 4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l195_19597


namespace NUMINAMATH_CALUDE_total_surveys_per_week_l195_19587

/-- Proves that the total number of surveys completed per week is 50 given the problem conditions -/
theorem total_surveys_per_week 
  (regular_rate : ℝ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (total_earnings : ℝ)
  (h1 : regular_rate = 30)
  (h2 : cellphone_rate_increase = 0.2)
  (h3 : cellphone_surveys = 50)
  (h4 : total_earnings = 3300)
  (h5 : total_earnings = cellphone_surveys * (regular_rate * (1 + cellphone_rate_increase))) :
  cellphone_surveys = 50 := by
  sorry

#check total_surveys_per_week

end NUMINAMATH_CALUDE_total_surveys_per_week_l195_19587


namespace NUMINAMATH_CALUDE_trajectory_equation_l195_19522

/-- The trajectory of a point M that satisfies |MF₁| + |MF₂| = 10, where F₁ = (-3, 0) and F₂ = (3, 0) -/
theorem trajectory_equation (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  F₁ = (-3, 0) →
  F₂ = (3, 0) →
  ‖M - F₁‖ + ‖M - F₂‖ = 10 →
  (M.1^2 / 25 + M.2^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l195_19522


namespace NUMINAMATH_CALUDE_parabola_equation_l195_19516

/-- Given a parabola with focus F(0, p/2) where p > 0, if its directrix intersects 
    the hyperbola x^2 - y^2 = 6 at points M and N such that triangle MNF is 
    a right-angled triangle, then the equation of the parabola is x^2 = 4√2y -/
theorem parabola_equation (p : ℝ) (M N : ℝ × ℝ) 
  (h_p : p > 0)
  (h_hyperbola : M.1^2 - M.2^2 = 6 ∧ N.1^2 - N.2^2 = 6)
  (h_right_triangle : (M.1 - 0)^2 + (M.2 - p/2)^2 = p^2 ∧ 
                      (N.1 - 0)^2 + (N.2 - p/2)^2 = p^2) :
  ∃ (x y : ℝ), x^2 = 4 * Real.sqrt 2 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l195_19516


namespace NUMINAMATH_CALUDE_fraction_evaluation_l195_19573

theorem fraction_evaluation : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / 
  (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l195_19573


namespace NUMINAMATH_CALUDE_asterisk_replacement_l195_19550

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 84) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l195_19550


namespace NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l195_19519

/-- Calculate the yearly cost to feed Harry's pets -/
theorem yearly_pet_feeding_cost :
  let num_geckos : ℕ := 3
  let num_iguanas : ℕ := 2
  let num_snakes : ℕ := 4
  let gecko_cost_per_month : ℕ := 15
  let iguana_cost_per_month : ℕ := 5
  let snake_cost_per_month : ℕ := 10
  let months_per_year : ℕ := 12
  
  (num_geckos * gecko_cost_per_month + 
   num_iguanas * iguana_cost_per_month + 
   num_snakes * snake_cost_per_month) * months_per_year = 1140 := by
  sorry

end NUMINAMATH_CALUDE_yearly_pet_feeding_cost_l195_19519


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l195_19523

open Set

def U : Set ℕ := {1,2,3,4,5}
def A : Set ℕ := {1,2,3}
def B : Set ℕ := {2,3,4}

theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {1,4,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l195_19523


namespace NUMINAMATH_CALUDE_system_solution_l195_19517

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x^2 - 9*y^2 = 0 ∧ x^2 + y^2 = 9}
  S = {(9/Real.sqrt 10, 3/Real.sqrt 10), (-9/Real.sqrt 10, 3/Real.sqrt 10),
       (9/Real.sqrt 10, -3/Real.sqrt 10), (-9/Real.sqrt 10, -3/Real.sqrt 10)} :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l195_19517


namespace NUMINAMATH_CALUDE_common_point_theorem_l195_19589

/-- Represents a line with equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the coefficients of a line form an arithmetic progression with common difference a/2 -/
def Line.isArithmeticProgression (l : Line) : Prop :=
  l.b = l.a + l.a/2 ∧ l.c = l.a + 2*(l.a/2)

/-- Checks if a point (x, y) lies on a given line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

theorem common_point_theorem :
  ∀ l : Line, l.isArithmeticProgression → l.containsPoint 0 (4/3) :=
sorry

end NUMINAMATH_CALUDE_common_point_theorem_l195_19589


namespace NUMINAMATH_CALUDE_barrel_capacity_l195_19556

/-- Prove that given the conditions, each barrel stores 2 gallons less than twice the capacity of a cask. -/
theorem barrel_capacity (num_barrels : ℕ) (cask_capacity : ℕ) (total_capacity : ℕ) :
  num_barrels = 4 →
  cask_capacity = 20 →
  total_capacity = 172 →
  (total_capacity - cask_capacity) / num_barrels = 2 * cask_capacity - 2 := by
  sorry


end NUMINAMATH_CALUDE_barrel_capacity_l195_19556


namespace NUMINAMATH_CALUDE_max_students_with_different_options_l195_19592

/-- Represents an answer sheet for a test with 6 questions, each with 3 options -/
def AnswerSheet := Fin 6 → Fin 3

/-- Checks if three answer sheets have at least one question where all options are different -/
def hasDifferentOptions (s1 s2 s3 : AnswerSheet) : Prop :=
  ∃ q : Fin 6, s1 q ≠ s2 q ∧ s1 q ≠ s3 q ∧ s2 q ≠ s3 q

/-- The main theorem stating the maximum number of students -/
theorem max_students_with_different_options :
  ∀ n : ℕ,
  (∀ sheets : Fin n → AnswerSheet,
    ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      hasDifferentOptions (sheets i) (sheets j) (sheets k)) →
  n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_students_with_different_options_l195_19592


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l195_19531

theorem repeating_decimal_sum : 
  (4 : ℚ) / 33 + 34 / 999 + 567 / 99999 = 134255 / 32929667 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l195_19531


namespace NUMINAMATH_CALUDE_function_properties_l195_19512

def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem function_properties (a b : ℝ) :
  (f a b 1 = 3) →
  ((3 * a + 2 * b) = 0) →
  (a = -6 ∧ b = 9) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 15) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -12) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 15) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -12) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l195_19512


namespace NUMINAMATH_CALUDE_rackets_sold_l195_19505

def total_sales : ℝ := 588
def average_price : ℝ := 9.8

theorem rackets_sold (pairs : ℝ) : pairs = total_sales / average_price → pairs = 60 := by
  sorry

end NUMINAMATH_CALUDE_rackets_sold_l195_19505


namespace NUMINAMATH_CALUDE_game_night_group_division_l195_19543

theorem game_night_group_division (n : ℕ) (h : n = 6) :
  Nat.choose n (n / 2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_game_night_group_division_l195_19543


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l195_19518

def repeating_decimal : ℚ := 7 + 17 / 99

theorem repeating_decimal_fraction :
  repeating_decimal = 710 / 99 ∧
  (Nat.gcd 710 99 = 1) ∧
  (710 + 99 = 809) := by
  sorry

#eval repeating_decimal

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l195_19518


namespace NUMINAMATH_CALUDE_previous_salary_calculation_l195_19570

/-- Represents the salary and commission structure of Tom's new job -/
structure NewJob where
  base_salary : ℝ
  commission_rate : ℝ
  sale_value : ℝ

/-- Calculates the total earnings from the new job given a number of sales -/
def earnings_new_job (job : NewJob) (num_sales : ℝ) : ℝ :=
  job.base_salary + job.commission_rate * job.sale_value * num_sales

/-- Theorem stating that if Tom needs to make at least 266.67 sales to not lose money,
    then his previous job salary was $75,000 -/
theorem previous_salary_calculation (job : NewJob) 
    (h1 : job.base_salary = 45000)
    (h2 : job.commission_rate = 0.15)
    (h3 : job.sale_value = 750)
    (h4 : earnings_new_job job 266.67 ≥ earnings_new_job job 266.66) :
    earnings_new_job job 266.67 = 75000 := by
  sorry

#check previous_salary_calculation

end NUMINAMATH_CALUDE_previous_salary_calculation_l195_19570


namespace NUMINAMATH_CALUDE_second_number_value_l195_19552

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 330 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l195_19552


namespace NUMINAMATH_CALUDE_suv_max_distance_l195_19542

/-- Calculates the maximum distance an SUV can travel given its fuel efficiencies and available fuel -/
theorem suv_max_distance (highway_mpg city_mpg mountain_mpg : ℝ) (fuel : ℝ) : 
  highway_mpg = 12.2 →
  city_mpg = 7.6 →
  mountain_mpg = 9.4 →
  fuel = 22 →
  (highway_mpg + city_mpg + mountain_mpg) * fuel = 642.4 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l195_19542


namespace NUMINAMATH_CALUDE_ian_money_left_l195_19571

/-- Calculates the amount of money left after spending half of earnings from surveys -/
def money_left (hours_worked : ℕ) (hourly_rate : ℚ) : ℚ :=
  (hours_worked : ℚ) * hourly_rate / 2

/-- Theorem: Given the conditions, prove that Ian has $72 left -/
theorem ian_money_left : money_left 8 18 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l195_19571


namespace NUMINAMATH_CALUDE_five_gate_park_choices_l195_19524

/-- The number of ways to choose an entry and exit gate in a park with 5 gates -/
def park_gate_choices (num_gates : ℕ) : ℕ :=
  num_gates * (num_gates - 1)

/-- Theorem: In a park with 5 gates, there are 20 ways to choose an entry and different exit gate -/
theorem five_gate_park_choices :
  park_gate_choices 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_five_gate_park_choices_l195_19524


namespace NUMINAMATH_CALUDE_star_value_of_a_l195_19514

-- Define the star operation
def star (a b : ℝ) : ℝ := 3 * a - b^2

-- State the theorem
theorem star_value_of_a : ∃ a : ℝ, star a 4 = 14 ∧ a = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_value_of_a_l195_19514


namespace NUMINAMATH_CALUDE_max_value_theorem_l195_19586

theorem max_value_theorem (x : ℝ) :
  x^4 / (x^8 + 2*x^6 + 4*x^4 + 4*x^2 + 16) ≤ 1/16 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 + 4*y^4 + 4*y^2 + 16) = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l195_19586


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l195_19562

/-- Represents a quadrilateral EFGH with given side lengths -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (HE : ℝ)
  (EG : ℕ)

/-- The specific quadrilateral from the problem -/
def problem_quadrilateral : Quadrilateral :=
  { EF := 7
  , FG := 21
  , GH := 7
  , HE := 13
  , EG := 21 }

/-- Triangle inequality theorem -/
axiom triangle_inequality {a b c : ℝ} : a + b > c

theorem quadrilateral_diagonal_length : 
  ∀ q : Quadrilateral, 
  q.EF = 7 → q.FG = 21 → q.GH = 7 → q.HE = 13 → 
  q.EG = problem_quadrilateral.EG :=
by
  sorry

#check quadrilateral_diagonal_length

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l195_19562


namespace NUMINAMATH_CALUDE_friday_night_revenue_l195_19596

/-- Represents the revenue calculation for a movie theater --/
def theater_revenue (matinee_price evening_price opening_price popcorn_price : ℕ)
                    (matinee_customers evening_customers opening_customers : ℕ) : ℕ :=
  let total_customers := matinee_customers + evening_customers + opening_customers
  let popcorn_sales := total_customers / 2
  (matinee_price * matinee_customers) +
  (evening_price * evening_customers) +
  (opening_price * opening_customers) +
  (popcorn_price * popcorn_sales)

/-- Theorem stating the total revenue of the theater on Friday night --/
theorem friday_night_revenue :
  theater_revenue 5 7 10 10 32 40 58 = 1670 := by
  sorry

end NUMINAMATH_CALUDE_friday_night_revenue_l195_19596


namespace NUMINAMATH_CALUDE_max_trig_combination_l195_19554

open Real

theorem max_trig_combination (a b c : ℝ) :
  (∀ θ : ℝ, c * (cos θ)^2 ≠ -a) →
  (∃ M : ℝ, ∀ θ : ℝ, a * cos θ + b * sin θ + c * tan θ ≤ M ∧
   ∃ θ : ℝ, a * cos θ + b * sin θ + c * tan θ = M) ∧
  (∀ M : ℝ, (∀ θ : ℝ, a * cos θ + b * sin θ + c * tan θ ≤ M) →
   M ≥ Real.sqrt (a^2 + b^2 + c^2)) :=
sorry

end NUMINAMATH_CALUDE_max_trig_combination_l195_19554


namespace NUMINAMATH_CALUDE_frog_eggs_theorem_l195_19503

/-- The number of eggs laid by a frog in a year -/
def eggs_laid : ℕ := 593

/-- The fraction of eggs that don't dry up -/
def not_dried_up : ℚ := 9/10

/-- The fraction of remaining eggs that are not eaten -/
def not_eaten : ℚ := 3/10

/-- The fraction of remaining eggs that hatch -/
def hatch_rate : ℚ := 1/4

/-- The number of frogs that hatch -/
def frogs_hatched : ℕ := 40

theorem frog_eggs_theorem :
  ↑frogs_hatched = ⌈(↑eggs_laid * not_dried_up * not_eaten * hatch_rate)⌉ := by sorry

end NUMINAMATH_CALUDE_frog_eggs_theorem_l195_19503


namespace NUMINAMATH_CALUDE_trapezoid_AD_length_l195_19532

/-- Represents a trapezoid ABCD with point M on CD and perpendicular AH from A to BM -/
structure Trapezoid :=
  (A B C D M H : ℝ × ℝ)
  (is_parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1))
  (M_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • C + t • D)
  (AH_perpendicular : (H.2 - A.2) * (M.1 - B.1) + (H.1 - A.1) * (M.2 - B.2) = 0)
  (AD_eq_HD : dist A D = dist H D)
  (BC_length : dist B C = 16)
  (CM_length : dist C M = 8)
  (MD_length : dist M D = 9)

/-- The length of AD in the given trapezoid is 18 -/
theorem trapezoid_AD_length (t : Trapezoid) : dist t.A t.D = 18 :=
  sorry

end NUMINAMATH_CALUDE_trapezoid_AD_length_l195_19532


namespace NUMINAMATH_CALUDE_product_from_gcd_lcm_l195_19540

theorem product_from_gcd_lcm (a b : ℕ+) :
  Nat.gcd a b = 5 → Nat.lcm a b = 60 → a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_gcd_lcm_l195_19540


namespace NUMINAMATH_CALUDE_min_value_at_one_min_value_is_constant_l195_19521

/-- A quadratic function f(x) = x^2 + (a+2)x + b symmetric about x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a+2)*x + b

/-- The property of f being symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop :=
  ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The minimum value of f -/
def min_value (a b : ℝ) : ℝ := f a b 1

theorem min_value_at_one (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∀ x, f a b x ≥ min_value a b :=
sorry

theorem min_value_is_constant (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∃ c, min_value a b = c :=
sorry

end NUMINAMATH_CALUDE_min_value_at_one_min_value_is_constant_l195_19521


namespace NUMINAMATH_CALUDE_total_beads_needed_l195_19520

/-- The number of green beads in one pattern repeat -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern repeat -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern repeat -/
def red_beads : ℕ := 2 * green_beads

/-- The total number of beads in one pattern repeat -/
def beads_per_repeat : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repeats in one bracelet -/
def repeats_per_bracelet : ℕ := 3

/-- The number of pattern repeats in one necklace -/
def repeats_per_necklace : ℕ := 5

/-- The number of bracelets to make -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to make -/
def num_necklaces : ℕ := 10

theorem total_beads_needed : 
  beads_per_repeat * (repeats_per_bracelet * num_bracelets + 
  repeats_per_necklace * num_necklaces) = 742 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_needed_l195_19520


namespace NUMINAMATH_CALUDE_stating_external_diagonals_inequality_invalid_external_diagonals_l195_19584

/-- Represents the lengths of external diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- 
Theorem stating that for a valid set of external diagonal lengths of a right regular prism,
the sum of squares of the two smaller lengths is greater than or equal to 
the square of the largest length.
-/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a ^ 2 + d.b ^ 2 ≥ d.c ^ 2 := by
  sorry

/-- 
Proves that {6, 8, 11} cannot be the lengths of external diagonals of a right regular prism
-/
theorem invalid_external_diagonals : 
  ¬∃ (d : ExternalDiagonals), d.a = 6 ∧ d.b = 8 ∧ d.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_stating_external_diagonals_inequality_invalid_external_diagonals_l195_19584


namespace NUMINAMATH_CALUDE_curve_properties_l195_19529

-- Define the curve C
def C (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Define the condition k ≠ -1
def k_not_neg_one (k : ℝ) : Prop := k ≠ -1

theorem curve_properties (k : ℝ) (h : k_not_neg_one k) :
  -- 1. C is always a circle
  (∃ (center_x center_y radius : ℝ), ∀ (x y : ℝ),
    C k x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
  -- The centers of the circles lie on the line y = 2x - 5
  (∃ (center_x center_y : ℝ), 
    (∀ (x y : ℝ), C k x y → (x - center_x)^2 + (y - center_y)^2 = (5*(k+1)^2)) ∧
    center_y = 2*center_x - 5) ∧
  -- 2. C passes through the fixed point (1, -3)
  C k 1 (-3) ∧
  -- 3. When C is tangent to the x-axis, k = 5 ± 3√5
  (∃ (x : ℝ), C k x 0 ∧ 
    (∀ (y : ℝ), y ≠ 0 → ¬(C k x y)) →
    (k = 5 + 3*Real.sqrt 5 ∨ k = 5 - 3*Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l195_19529


namespace NUMINAMATH_CALUDE_claire_pets_male_hamster_fraction_l195_19577

theorem claire_pets_male_hamster_fraction :
  ∀ (total_pets gerbils hamsters male_pets male_gerbils male_hamsters : ℕ),
    total_pets = 90 →
    gerbils = 66 →
    total_pets = gerbils + hamsters →
    male_pets = 25 →
    male_gerbils = 16 →
    male_pets = male_gerbils + male_hamsters →
    (male_hamsters : ℚ) / (hamsters : ℚ) = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_claire_pets_male_hamster_fraction_l195_19577


namespace NUMINAMATH_CALUDE_man_speed_against_current_is_10_l195_19513

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def man_speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed against the current is 10 km/hr. -/
theorem man_speed_against_current_is_10 :
  man_speed_against_current 15 2.5 = 10 := by
  sorry

#eval man_speed_against_current 15 2.5

end NUMINAMATH_CALUDE_man_speed_against_current_is_10_l195_19513


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l195_19500

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Prove that the opposite of -2023 is 2023. -/
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l195_19500


namespace NUMINAMATH_CALUDE_unique_m_value_l195_19528

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem unique_m_value : ∃! m : ℝ, 2 ∈ A m ∧ (∀ x ∈ A m, ∀ y ∈ A m, x = y → x = 0 ∨ x = m ∨ x = m^2 - 3*m + 2) :=
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l195_19528


namespace NUMINAMATH_CALUDE_greatest_n_for_inequality_l195_19530

theorem greatest_n_for_inequality (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m = 5 ∧ 101 * m^2 ≤ 3600 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_inequality_l195_19530


namespace NUMINAMATH_CALUDE_units_digit_problem_l195_19533

theorem units_digit_problem : ∃ n : ℕ, (7 * 27 * 1977 + 9) - 7^3 ≡ 9 [ZMOD 10] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l195_19533


namespace NUMINAMATH_CALUDE_total_game_time_calculation_l195_19555

def game_preparation_time (download_time install_time update_time account_time internet_time discussion_time tutorial_time : ℕ) : ℕ :=
  download_time + install_time + update_time + account_time + internet_time + discussion_time + tutorial_time

def in_game_tutorial_time (prep_time : ℕ) : ℕ :=
  3 * prep_time

theorem total_game_time_calculation (download_time : ℕ) : 
  let install_time := download_time / 2
  let update_time := 2 * download_time
  let account_time := 5
  let internet_time := 15
  let discussion_time := 20
  let tutorial_time := 8
  let prep_time := game_preparation_time download_time install_time update_time account_time internet_time discussion_time tutorial_time
  let tutorial_time := in_game_tutorial_time prep_time
  prep_time + tutorial_time = 332 :=
by
  sorry

#check total_game_time_calculation 10

end NUMINAMATH_CALUDE_total_game_time_calculation_l195_19555


namespace NUMINAMATH_CALUDE_negation_of_even_multiple_of_two_l195_19599

theorem negation_of_even_multiple_of_two :
  ¬(∀ n : ℕ, Even n → (∃ k : ℕ, n = 2 * k)) ↔ 
  (∃ n : ℕ, Even n ∧ ¬(∃ k : ℕ, n = 2 * k)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_even_multiple_of_two_l195_19599


namespace NUMINAMATH_CALUDE_flight_time_around_earth_l195_19504

def earth_radius : ℝ := 6000
def jet_speed : ℝ := 600

theorem flight_time_around_earth :
  let circumference := 2 * Real.pi * earth_radius
  let flight_time := circumference / jet_speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (flight_time - 63) < ε := by
sorry

end NUMINAMATH_CALUDE_flight_time_around_earth_l195_19504


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l195_19591

theorem opposite_of_negative_three : -(- 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l195_19591


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l195_19548

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l195_19548


namespace NUMINAMATH_CALUDE_blueberry_carton_size_l195_19559

/-- The number of ounces in a carton of blueberries -/
def blueberry_carton_ounces : ℝ := 6

/-- The cost of a carton of blueberries in dollars -/
def blueberry_carton_cost : ℝ := 5

/-- The cost of a carton of raspberries in dollars -/
def raspberry_carton_cost : ℝ := 3

/-- The number of ounces in a carton of raspberries -/
def raspberry_carton_ounces : ℝ := 8

/-- The number of batches of muffins being made -/
def num_batches : ℝ := 4

/-- The number of ounces of fruit required per batch -/
def ounces_per_batch : ℝ := 12

/-- The amount saved by using raspberries instead of blueberries -/
def amount_saved : ℝ := 22

theorem blueberry_carton_size :
  blueberry_carton_ounces = 6 :=
sorry

end NUMINAMATH_CALUDE_blueberry_carton_size_l195_19559


namespace NUMINAMATH_CALUDE_school_absence_percentage_l195_19588

theorem school_absence_percentage (total_students boys girls : ℕ) 
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys : ℕ := boys / 5)
  (absent_girls : ℕ := girls / 4)
  (total_absent : ℕ := absent_boys + absent_girls) :
  (total_absent : ℚ) / total_students * 100 = 40 / 180 * 100 := by
sorry

end NUMINAMATH_CALUDE_school_absence_percentage_l195_19588


namespace NUMINAMATH_CALUDE_caroline_score_l195_19565

structure Player where
  name : String
  score : ℕ

def winning_score : ℕ := 21

theorem caroline_score (caroline anthony leo : Player)
  (h1 : anthony.score = 19)
  (h2 : leo.score = 28)
  (h3 : ∃ p : Player, p ∈ [caroline, anthony, leo] ∧ p.score = winning_score) :
  caroline.score = winning_score :=
sorry

end NUMINAMATH_CALUDE_caroline_score_l195_19565


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l195_19574

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 7.5 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l195_19574
