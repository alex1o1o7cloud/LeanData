import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1685_168584

theorem algebraic_expression_value (x : ℝ) : 
  3 * x^2 - 2 * x - 1 = 2 → -9 * x^2 + 6 * x - 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1685_168584


namespace NUMINAMATH_CALUDE_toms_initial_books_l1685_168529

/-- Given that Tom sold 4 books, bought 38 new books, and now has 39 books,
    prove that he initially had 5 books. -/
theorem toms_initial_books :
  ∀ (initial_books : ℕ),
    initial_books - 4 + 38 = 39 →
    initial_books = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_initial_books_l1685_168529


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l1685_168548

/-- The cost of three dozen apples in dollars -/
def cost_three_dozen : ℚ := 25.20

/-- The number of dozens we want to calculate the cost for -/
def target_dozens : ℕ := 4

/-- The cost of the target number of dozens of apples -/
def cost_target_dozens : ℚ := 33.60

/-- Theorem stating that the cost of the target number of dozens of apples is correct -/
theorem apple_cost_calculation : 
  (cost_three_dozen / 3) * target_dozens = cost_target_dozens := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l1685_168548


namespace NUMINAMATH_CALUDE_M_mod_51_l1685_168590

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 34 := by sorry

end NUMINAMATH_CALUDE_M_mod_51_l1685_168590


namespace NUMINAMATH_CALUDE_equation_solution_verify_solution_l1685_168524

/-- The solution to the equation √((3x-1)/(x+4)) + 3 - 4√((x+4)/(3x-1)) = 0 -/
theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
           x = 5 / 2 := by
  sorry

/-- Verification that 5/2 is indeed the solution -/
theorem verify_solution :
  let x : ℝ := 5 / 2
  Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_verify_solution_l1685_168524


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1685_168554

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                     -- sum condition
  (a > c) →                          -- inequality condition
  (a = 7 + 2 * Real.sqrt 10 ∧ c = 7 - 2 * Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1685_168554


namespace NUMINAMATH_CALUDE_marks_age_in_five_years_l1685_168585

theorem marks_age_in_five_years :
  ∀ (amy_age mark_age : ℕ),
    amy_age = 15 →
    mark_age = amy_age + 7 →
    mark_age + 5 = 27 :=
by sorry

end NUMINAMATH_CALUDE_marks_age_in_five_years_l1685_168585


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1685_168521

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1685_168521


namespace NUMINAMATH_CALUDE_power_sum_and_division_equals_82_l1685_168577

theorem power_sum_and_division_equals_82 : 2^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_division_equals_82_l1685_168577


namespace NUMINAMATH_CALUDE_project_time_calculation_l1685_168573

theorem project_time_calculation (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  y = (3 * x) / 2 →
  z = 2 * x →
  z = x + 20 →
  x + y + z = 90 :=
by sorry

end NUMINAMATH_CALUDE_project_time_calculation_l1685_168573


namespace NUMINAMATH_CALUDE_two_row_arrangement_count_l1685_168504

/-- The number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.descFactorial n k

theorem two_row_arrangement_count
  (n k k₁ k₂ : ℕ)
  (h₁ : k₁ + k₂ = k)
  (h₂ : 1 ≤ k)
  (h₃ : k ≤ n) :
  (permutations n k₁) * (permutations (n - k₁) k₂) = permutations n k :=
sorry

end NUMINAMATH_CALUDE_two_row_arrangement_count_l1685_168504


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1685_168528

theorem quadratic_equation_properties (k : ℝ) :
  (∃ x y : ℝ, x^2 - k*x + k - 1 = 0 ∧ y^2 - k*y + k - 1 = 0 ∧ (x = y ∨ x ≠ y)) ∧
  (∃ x : ℝ, x^2 - k*x + k - 1 = 0 ∧ x < 0) → k < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1685_168528


namespace NUMINAMATH_CALUDE_elena_marco_sum_ratio_l1685_168579

def sum_odd_integers (n : ℕ) : ℕ := n * n

def sum_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odd_integers 250) / (sum_integers 250) = 2 := by
  sorry

end NUMINAMATH_CALUDE_elena_marco_sum_ratio_l1685_168579


namespace NUMINAMATH_CALUDE_tan_810_degrees_undefined_l1685_168574

theorem tan_810_degrees_undefined : 
  ¬∃ (x : ℝ), Real.tan (810 * π / 180) = x :=
by
  sorry

end NUMINAMATH_CALUDE_tan_810_degrees_undefined_l1685_168574


namespace NUMINAMATH_CALUDE_function_inequality_l1685_168596

theorem function_inequality (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y ≥ Real.sin x + Real.cos y) :
  ∃ p q : ℝ → ℝ, 
    (∀ x : ℝ, f x = (Real.sin x + Real.cos x + p x - q x) / 2) ∧
    (∀ x : ℝ, g x = (Real.sin x - Real.cos x + p x + q x) / 2) ∧
    (∀ x y : ℝ, p x ≥ q y) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1685_168596


namespace NUMINAMATH_CALUDE_system_of_equations_solution_fractional_equation_solution_l1685_168582

-- Problem 1: System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 :=
sorry

-- Problem 2: Fractional equation
theorem fractional_equation_solution :
  ∃! y : ℝ, y ≠ 1 ∧ 3 / (1 - y) = y / (y - 1) - 5 :=
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_fractional_equation_solution_l1685_168582


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l1685_168586

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℚ) (shorter_length : ℝ) : 
  total_length = 60 →
  ratio = 2 / 4 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 40 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l1685_168586


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l1685_168511

theorem missing_digit_divisible_by_three (x : Nat) :
  x < 10 →
  (1357 * 10 + x) * 10 + 2 % 3 = 0 →
  x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_three_l1685_168511


namespace NUMINAMATH_CALUDE_average_of_pqrs_l1685_168517

theorem average_of_pqrs (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_pqrs_l1685_168517


namespace NUMINAMATH_CALUDE_geraldo_tea_consumption_l1685_168503

/-- Proves that given 20 gallons of tea poured into 80 containers, if Geraldo drinks 3.5 containers, he consumes 7 pints of tea. -/
theorem geraldo_tea_consumption 
  (total_gallons : ℝ) 
  (num_containers : ℝ) 
  (geraldo_containers : ℝ) 
  (gallons_to_pints : ℝ → ℝ) :
  total_gallons = 20 ∧ 
  num_containers = 80 ∧ 
  geraldo_containers = 3.5 ∧ 
  (∀ x, gallons_to_pints x = 8 * x) →
  geraldo_containers * (gallons_to_pints total_gallons / num_containers) = 7 :=
by sorry

end NUMINAMATH_CALUDE_geraldo_tea_consumption_l1685_168503


namespace NUMINAMATH_CALUDE_special_number_property_l1685_168562

theorem special_number_property (X : ℕ) : 
  (3 + X % 26 = X / 26) ∧ (X % 29 = X / 29) → X = 270 ∨ X = 540 := by
  sorry

end NUMINAMATH_CALUDE_special_number_property_l1685_168562


namespace NUMINAMATH_CALUDE_solve_for_y_l1685_168581

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1685_168581


namespace NUMINAMATH_CALUDE_x_0_value_l1685_168567

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x_0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x_0_value_l1685_168567


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1685_168510

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1685_168510


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l1685_168505

/-- Calculates the number of hours each trainer spends training dolphins -/
def trainer_hours (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (num_trainers : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / num_trainers

theorem dolphin_training_hours :
  trainer_hours 4 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l1685_168505


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l1685_168523

/-- Base prime representation of a natural number -/
def BasePrimeRepr : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrimeRepr (l : List ℕ) : Prop := sorry

theorem base_prime_repr_360 :
  let repr := BasePrimeRepr 360
  IsValidBasePrimeRepr repr ∧ repr = [3, 2, 1] := by sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l1685_168523


namespace NUMINAMATH_CALUDE_quadrilateral_to_parallelogram_l1685_168565

-- Define the points
variable (A B C D E F O : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

def segments_intersect (P Q R S : ℝ × ℝ) (I : ℝ × ℝ) : Prop := sorry

def divides_into_three_equal_parts (P Q R S : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_to_parallelogram 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_midpoint_E : is_midpoint E A D)
  (h_midpoint_F : is_midpoint F B C)
  (h_intersect : segments_intersect C E D F O)
  (h_divide_AO : divides_into_three_equal_parts A O C D)
  (h_divide_BO : divides_into_three_equal_parts B O C D) :
  is_parallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_to_parallelogram_l1685_168565


namespace NUMINAMATH_CALUDE_inequality_range_l1685_168518

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ 
  (m > -10 ∧ m ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l1685_168518


namespace NUMINAMATH_CALUDE_minimum_c_value_l1685_168509

-- Define the curve
def on_curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the inequality condition
def inequality_holds (c : ℝ) : Prop := ∀ x y : ℝ, on_curve x y → x + y + c ≥ 0

-- State the theorem
theorem minimum_c_value : 
  (∃ c_min : ℝ, (∀ c : ℝ, c ≥ c_min ↔ inequality_holds c) ∧ c_min = Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_minimum_c_value_l1685_168509


namespace NUMINAMATH_CALUDE_square_tiles_l1685_168549

theorem square_tiles (n : ℕ) (h : n * n = 81) :
  n * n - n = 72 :=
sorry

end NUMINAMATH_CALUDE_square_tiles_l1685_168549


namespace NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l1685_168552

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- The main theorem stating that a convex polyhedron with given properties has 310 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 15 ∧
    Q.pentagon_faces = 5 ∧
    space_diagonals Q = 310 :=
  sorry

end NUMINAMATH_CALUDE_space_diagonals_of_specific_polyhedron_l1685_168552


namespace NUMINAMATH_CALUDE_only_one_solution_l1685_168522

def sum_of_squares (K : ℕ) : ℕ := K * (K + 1) * (2 * K + 1) / 6

theorem only_one_solution (K : ℕ) (M : ℕ) :
  sum_of_squares K = M^3 →
  M < 50 →
  K = 1 :=
by sorry

end NUMINAMATH_CALUDE_only_one_solution_l1685_168522


namespace NUMINAMATH_CALUDE_cotangent_sum_equality_l1685_168542

/-- Given a triangle ABC, A'B'C' is the triangle formed by its medians -/
def MedianTriangle (A B C : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- The sum of cotangents of angles in a triangle -/
def SumOfCotangents (A B C : ℝ × ℝ) : ℝ := sorry

theorem cotangent_sum_equality (A B C : ℝ × ℝ) :
  let (A', B', C') := MedianTriangle A B C
  SumOfCotangents A B C = SumOfCotangents A' B' C' := by sorry

end NUMINAMATH_CALUDE_cotangent_sum_equality_l1685_168542


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l1685_168598

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) : 
  (a > 0 ∧ a ≤ 2) ↔ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l1685_168598


namespace NUMINAMATH_CALUDE_catch_up_equation_correct_l1685_168535

/-- Represents the problem of two horses racing, where one starts earlier than the other. -/
structure HorseRace where
  fast_speed : ℕ  -- Speed of the faster horse in miles per day
  slow_speed : ℕ  -- Speed of the slower horse in miles per day
  head_start : ℕ  -- Number of days the slower horse starts earlier

/-- The equation representing when the faster horse catches up to the slower horse -/
def catch_up_equation (race : HorseRace) (x : ℝ) : Prop :=
  (race.fast_speed : ℝ) * x = (race.slow_speed : ℝ) * (x + race.head_start)

/-- The specific race described in the problem -/
def zhu_shijie_race : HorseRace :=
  { fast_speed := 240
  , slow_speed := 150
  , head_start := 12 }

/-- Theorem stating that the given equation correctly represents the race situation -/
theorem catch_up_equation_correct :
  catch_up_equation zhu_shijie_race = fun x => 240 * x = 150 * (x + 12) :=
by sorry


end NUMINAMATH_CALUDE_catch_up_equation_correct_l1685_168535


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l1685_168515

theorem restaurant_bill_division (total_bill : ℕ) (individual_payment : ℕ) (num_friends : ℕ) :
  total_bill = 135 →
  individual_payment = 45 →
  total_bill = individual_payment * num_friends →
  num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l1685_168515


namespace NUMINAMATH_CALUDE_existence_of_x_y_satisfying_conditions_l1685_168564

theorem existence_of_x_y_satisfying_conditions : ∃ (x y : ℝ), x = y + 1 ∧ x^4 = y^4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_y_satisfying_conditions_l1685_168564


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l1685_168519

theorem consecutive_even_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2 ∧ d = n + 4) →
  (a + c = 92) →
  d = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l1685_168519


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1685_168512

theorem fraction_evaluation : 
  (1 / 5 - 1 / 7) / (3 / 8 + 2 / 9) = 144 / 1505 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1685_168512


namespace NUMINAMATH_CALUDE_marias_coffee_order_l1685_168531

/-- Maria's daily coffee order calculation -/
theorem marias_coffee_order (visits_per_day : ℕ) (cups_per_visit : ℕ)
  (h1 : visits_per_day = 2)
  (h2 : cups_per_visit = 3) :
  visits_per_day * cups_per_visit = 6 := by
  sorry

end NUMINAMATH_CALUDE_marias_coffee_order_l1685_168531


namespace NUMINAMATH_CALUDE_sarah_hardback_count_l1685_168527

/-- The number of hardback books Sarah bought -/
def sarah_hardback : ℕ := sorry

/-- The number of paperback books Sarah bought -/
def sarah_paperback : ℕ := 6

/-- The number of paperback books Sarah's brother bought -/
def brother_paperback : ℕ := sarah_paperback / 3

/-- The number of hardback books Sarah's brother bought -/
def brother_hardback : ℕ := 2 * sarah_hardback

/-- The total number of books Sarah's brother bought -/
def brother_total : ℕ := 10

theorem sarah_hardback_count : sarah_hardback = 4 := by
  sorry

end NUMINAMATH_CALUDE_sarah_hardback_count_l1685_168527


namespace NUMINAMATH_CALUDE_brians_trip_distance_l1685_168599

/-- Calculates the distance traveled given car efficiency and fuel consumed -/
def distance_traveled (efficiency : ℝ) (fuel_consumed : ℝ) : ℝ :=
  efficiency * fuel_consumed

/-- Proves that given a car efficiency of 20 miles per gallon and a fuel consumption of 3 gallons, the distance traveled is 60 miles -/
theorem brians_trip_distance :
  distance_traveled 20 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_brians_trip_distance_l1685_168599


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l1685_168583

/-- Proves that the profit from selling a cricket bat is approximately $215.29 --/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) (h1 : selling_price = 850) (h2 : profit_percentage = 33.85826771653544) :
  ∃ (profit : ℝ), abs (profit - 215.29) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l1685_168583


namespace NUMINAMATH_CALUDE_cab_journey_time_l1685_168526

/-- Given a cab traveling at 5/6 of its usual speed is 8 minutes late, 
    prove that its usual time to cover the journey is 48 minutes. -/
theorem cab_journey_time (usual_time : ℝ) : 
  (5 / 6 : ℝ) * usual_time + 8 = usual_time → usual_time = 48 :=
by sorry

end NUMINAMATH_CALUDE_cab_journey_time_l1685_168526


namespace NUMINAMATH_CALUDE_ground_beef_cost_l1685_168555

/-- The price of ground beef per kilogram in dollars -/
def price_per_kg : ℝ := 5.00

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60.00 := by
  sorry

end NUMINAMATH_CALUDE_ground_beef_cost_l1685_168555


namespace NUMINAMATH_CALUDE_piecewise_function_sum_l1685_168501

theorem piecewise_function_sum (f : ℝ → ℝ) (a b c : ℤ) : 
  (∀ x > 0, f x = a * x + b) →
  (∀ x < 0, f x = b * x + c) →
  (f 0 = a * b) →
  (f 2 = 7) →
  (f 0 = 1) →
  (f (-2) = -8) →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_piecewise_function_sum_l1685_168501


namespace NUMINAMATH_CALUDE_money_split_l1685_168558

theorem money_split (total : ℝ) (moses_percent : ℝ) (moses_esther_diff : ℝ) : 
  total = 50 ∧ moses_percent = 0.4 ∧ moses_esther_diff = 5 →
  ∃ (tony esther moses : ℝ),
    moses = total * moses_percent ∧
    tony + esther = total - moses ∧
    moses = esther + moses_esther_diff ∧
    tony = 15 ∧ esther = 15 :=
by sorry

end NUMINAMATH_CALUDE_money_split_l1685_168558


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1685_168593

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x * y > 0) :
  x - y = 2 ∨ x - y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1685_168593


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1685_168561

theorem min_value_of_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z →
    3*x^2 + 2*y^2 + 4*z^2 - x*y - 3*y*z - 5*z*x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1685_168561


namespace NUMINAMATH_CALUDE_abc_value_l1685_168540

theorem abc_value :
  let a := -(2017 * 2017 - 2017) / (2016 * 2016 + 2016)
  let b := -(2018 * 2018 - 2018) / (2017 * 2017 + 2017)
  let c := -(2019 * 2019 - 2019) / (2018 * 2018 + 2018)
  a * b * c = -1 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l1685_168540


namespace NUMINAMATH_CALUDE_spelling_bee_probability_l1685_168538

/-- The probability of selecting all girls in a spelling bee competition -/
theorem spelling_bee_probability (total : ℕ) (girls : ℕ) (selected : ℕ) 
  (h_total : total = 8) 
  (h_girls : girls = 5)
  (h_selected : selected = 3) :
  (Nat.choose girls selected : ℚ) / (Nat.choose total selected) = 5 / 28 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_probability_l1685_168538


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1685_168556

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 20 = 100) :
  ∃ m : ℝ, m = 20 ∧ ∀ x : ℝ, (a 7 + a 14 ≥ x ∧ (∃ y : ℝ, a 7 = y ∧ a 14 = y → a 7 + a 14 = x)) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1685_168556


namespace NUMINAMATH_CALUDE_range_of_a_l1685_168566

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) →
  -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1685_168566


namespace NUMINAMATH_CALUDE_stadium_perimeter_stadium_breadth_l1685_168514

/-- Represents a rectangular stadium -/
structure Stadium where
  perimeter : ℝ
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle is twice the sum of its length and breadth -/
theorem stadium_perimeter (s : Stadium) : s.perimeter = 2 * (s.length + s.breadth) := by sorry

/-- Given a stadium with perimeter 800 and length 100, its breadth is 300 -/
theorem stadium_breadth : 
  ∀ (s : Stadium), s.perimeter = 800 ∧ s.length = 100 → s.breadth = 300 := by sorry

end NUMINAMATH_CALUDE_stadium_perimeter_stadium_breadth_l1685_168514


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1685_168506

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1685_168506


namespace NUMINAMATH_CALUDE_absolute_value_zero_implies_negative_three_l1685_168533

theorem absolute_value_zero_implies_negative_three (a : ℝ) :
  |a + 3| = 0 → a = -3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_zero_implies_negative_three_l1685_168533


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l1685_168578

theorem opposite_of_one_half : -(1/2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l1685_168578


namespace NUMINAMATH_CALUDE_line_chart_appropriate_for_temperature_over_week_l1685_168500

-- Define the types of charts
inductive ChartType
| Bar
| Line
| Pie

-- Define the characteristics of the data
structure TemperatureData :=
  (measurements : List Float)
  (timePoints : List String)
  (duration : Nat)

-- Define the requirements for the chart
structure ChartRequirements :=
  (showQuantity : Bool)
  (showChangeOverTime : Bool)

-- Define a function to determine the appropriate chart type
def appropriateChartType (data : TemperatureData) (req : ChartRequirements) : ChartType :=
  sorry

-- Theorem statement
theorem line_chart_appropriate_for_temperature_over_week :
  ∀ (data : TemperatureData) (req : ChartRequirements),
    data.duration = 7 →
    req.showQuantity = true →
    req.showChangeOverTime = true →
    appropriateChartType data req = ChartType.Line :=
  sorry

end NUMINAMATH_CALUDE_line_chart_appropriate_for_temperature_over_week_l1685_168500


namespace NUMINAMATH_CALUDE_f_tangent_perpendicular_range_l1685_168568

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m + exp (-x))

theorem f_tangent_perpendicular_range :
  ∃ (a b : Set ℝ), a = Set.Ioo 0 (exp (-2)) ∧
  (∀ m : ℝ, m ∈ a ↔ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      (deriv (f m)) x₁ = 0 ∧ 
      (deriv (f m)) x₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_f_tangent_perpendicular_range_l1685_168568


namespace NUMINAMATH_CALUDE_meaningful_expression_l1685_168539

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 1)) ↔ x > -1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1685_168539


namespace NUMINAMATH_CALUDE_unattainable_value_l1685_168572

theorem unattainable_value (x : ℝ) (y : ℝ) (h : x ≠ -4/3) : 
  y = (1 - x) / (3 * x + 4) → y ≠ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_unattainable_value_l1685_168572


namespace NUMINAMATH_CALUDE_sum_digits_base7_of_777_l1685_168534

-- Define a function to convert a number from base 10 to base 7
def toBase7 (n : ℕ) : List ℕ := sorry

-- Define a function to sum the digits of a number represented as a list
def sumDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_base7_of_777 : sumDigits (toBase7 777) = 9 := by sorry

end NUMINAMATH_CALUDE_sum_digits_base7_of_777_l1685_168534


namespace NUMINAMATH_CALUDE_prime_product_l1685_168547

theorem prime_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 3 * p + 7 * q = 41 → (p + 1) * (q - 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_prime_product_l1685_168547


namespace NUMINAMATH_CALUDE_simplify_radical_product_l1685_168571

theorem simplify_radical_product (y z : ℝ) :
  Real.sqrt (50 * y) * Real.sqrt (18 * z) * Real.sqrt (32 * y) = 40 * y * Real.sqrt (2 * z) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l1685_168571


namespace NUMINAMATH_CALUDE_max_sum_of_shorter_l1685_168597

/-- Represents the configuration of houses -/
structure HouseConfig where
  one_story : ℕ
  two_story : ℕ

/-- The total number of floors in the city -/
def total_floors : ℕ := 30

/-- Calculates the sum of shorter houses seen from each roof -/
def sum_of_shorter (config : HouseConfig) : ℕ :=
  config.one_story * config.two_story

/-- Checks if a configuration is valid (i.e., totals 30 floors) -/
def is_valid_config (config : HouseConfig) : Prop :=
  config.one_story + 2 * config.two_story = total_floors

/-- The theorem to be proved -/
theorem max_sum_of_shorter :
  ∃ (config1 config2 : HouseConfig),
    is_valid_config config1 ∧
    is_valid_config config2 ∧
    sum_of_shorter config1 = 112 ∧
    sum_of_shorter config2 = 112 ∧
    (∀ (config : HouseConfig), is_valid_config config → sum_of_shorter config ≤ 112) ∧
    ((config1.one_story = 16 ∧ config1.two_story = 7) ∨
     (config1.one_story = 14 ∧ config1.two_story = 8)) ∧
    ((config2.one_story = 16 ∧ config2.two_story = 7) ∨
     (config2.one_story = 14 ∧ config2.two_story = 8)) ∧
    config1 ≠ config2 :=
  sorry

end NUMINAMATH_CALUDE_max_sum_of_shorter_l1685_168597


namespace NUMINAMATH_CALUDE_point_transformation_to_polar_coordinates_l1685_168508

theorem point_transformation_to_polar_coordinates :
  ∀ (x y : ℝ),
    2 * x = 6 ∧ Real.sqrt 3 * y = -3 →
    ∃ (ρ θ : ℝ),
      ρ = 2 * Real.sqrt 3 ∧
      θ = 11 * π / 6 ∧
      ρ > 0 ∧
      0 ≤ θ ∧ θ < 2 * π ∧
      x = ρ * Real.cos θ ∧
      y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_point_transformation_to_polar_coordinates_l1685_168508


namespace NUMINAMATH_CALUDE_largest_special_square_l1685_168532

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem largest_special_square : 
  (is_perfect_square 1681) ∧ 
  (1681 % 10 ≠ 0) ∧ 
  (is_perfect_square (remove_last_two_digits 1681)) ∧ 
  (∀ m : ℕ, m > 1681 → 
    ¬(is_perfect_square m ∧ 
      m % 10 ≠ 0 ∧ 
      is_perfect_square (remove_last_two_digits m))) :=
by sorry

end NUMINAMATH_CALUDE_largest_special_square_l1685_168532


namespace NUMINAMATH_CALUDE_cube_sum_l1685_168580

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l1685_168580


namespace NUMINAMATH_CALUDE_min_value_on_common_chord_l1685_168525

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem min_value_on_common_chord :
  ∀ a b : ℝ, a > 0 → b > 0 → common_chord a b →
  (∀ x y : ℝ, x > 0 → y > 0 → common_chord x y → 1/a + 9/b ≤ 1/x + 9/y) →
  1/a + 9/b = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_common_chord_l1685_168525


namespace NUMINAMATH_CALUDE_trio_ball_theorem_l1685_168543

/-- The number of minutes each child plays in the trio-ball game -/
def trio_ball_play_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) : ℕ :=
  (total_time * players_per_game) / num_children

/-- Theorem stating that each child plays for 60 minutes in the given scenario -/
theorem trio_ball_theorem :
  trio_ball_play_time 120 6 3 = 60 := by
  sorry

#eval trio_ball_play_time 120 6 3

end NUMINAMATH_CALUDE_trio_ball_theorem_l1685_168543


namespace NUMINAMATH_CALUDE_max_value_inequality_l1685_168569

theorem max_value_inequality (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) : 
  (a * b / (a + b + 1)) + (b * c / (b + c + 1)) + (c * a / (c + a + 1)) ≤ 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1685_168569


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1685_168553

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (5,0) with radius 2.
    The sum of all possible radii of the circle with center C is 14. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l1685_168553


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l1685_168570

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 64) (h2 : v2 = 512) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l1685_168570


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1685_168520

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the given expression -/
def expression : ℕ := 8 * 19 * 1981 - 8^3

theorem units_digit_of_expression :
  unitsDigit expression = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1685_168520


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l1685_168502

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + 1) on this line,
    prove that the x-coordinate of the first point, m, is equal to 2n + 3. -/
theorem x_coordinate_of_first_point
  (m n : ℝ)
  (h1 : m = 2 * n + 3)
  (h2 : m + 2 = 2 * (n + 1) + 3) :
  m = 2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l1685_168502


namespace NUMINAMATH_CALUDE_cistern_water_depth_l1685_168576

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 meters -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 4)
  (h_wet_area : total_wet_area = 88)
  : ∃ (depth : ℝ), depth = 1.25 ∧ total_wet_area = length * width + 2 * depth * (length + width) :=
by sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l1685_168576


namespace NUMINAMATH_CALUDE_fraction_of_number_l1685_168560

theorem fraction_of_number (N : ℝ) : (0.4 * N = 204) → ((1/4) * (1/3) * (2/5) * N = 17) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l1685_168560


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l1685_168589

theorem fifth_power_sum_equality : ∃! (n : ℕ), n > 0 ∧ 120^5 + 105^5 + 78^5 + 33^5 = n^5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l1685_168589


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1685_168575

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1685_168575


namespace NUMINAMATH_CALUDE_work_done_by_combined_forces_l1685_168546

/-- Work done by combined forces -/
theorem work_done_by_combined_forces
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)
  (S : ℝ × ℝ)
  (h₁ : F₁ = (Real.log 2, Real.log 2))
  (h₂ : F₂ = (Real.log 5, Real.log 2))
  (h₃ : S = (2 * Real.log 5, 1)) :
  (F₁.1 + F₂.1) * S.1 + (F₁.2 + F₂.2) * S.2 = 2 := by
  sorry

#check work_done_by_combined_forces

end NUMINAMATH_CALUDE_work_done_by_combined_forces_l1685_168546


namespace NUMINAMATH_CALUDE_shifted_direct_proportion_l1685_168591

/-- Given a direct proportion function y = -3x that is shifted down by 5 units,
    prove that the resulting function is y = -3x - 5 -/
theorem shifted_direct_proportion (x y : ℝ) :
  (y = -3 * x) → (y - 5 = -3 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_shifted_direct_proportion_l1685_168591


namespace NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l1685_168536

/-- The analysis method for proving inequalities -/
def analysis_method : Type := Unit

/-- A condition that makes an inequality hold -/
def condition : Type := Unit

/-- Predicate indicating if a condition is sufficient -/
def is_sufficient (c : condition) : Prop := sorry

/-- The condition sought by the analysis method -/
def sought_condition (m : analysis_method) : condition := sorry

/-- Theorem stating that the analysis method seeks a sufficient condition -/
theorem analysis_method_seeks_sufficient_condition :
  ∀ (m : analysis_method), is_sufficient (sought_condition m) := by
  sorry

end NUMINAMATH_CALUDE_analysis_method_seeks_sufficient_condition_l1685_168536


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l1685_168592

theorem larger_tv_diagonal (area_diff : ℝ) : 
  area_diff = 40 → 
  let small_tv_diagonal : ℝ := 19
  let small_tv_area : ℝ := (small_tv_diagonal / Real.sqrt 2) ^ 2
  let large_tv_area : ℝ := small_tv_area + area_diff
  let large_tv_diagonal : ℝ := Real.sqrt (2 * large_tv_area)
  large_tv_diagonal = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l1685_168592


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_value_l1685_168550

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x-5 → ¬∃ (a b : ℝ), b > 0 ∧ b ≠ 1 ∧ x-5 = a^2 * b

theorem simplest_quadratic_radical_value :
  ∀ x : ℝ, x ∈ ({11, 13, 21, 29} : Set ℝ) →
    (is_simplest_quadratic_radical x ↔ x = 11) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_value_l1685_168550


namespace NUMINAMATH_CALUDE_lamp_post_ratio_l1685_168541

theorem lamp_post_ratio (k m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 9 * x = k ∧ 99 * x = m) → m / k = 11 := by
  sorry

end NUMINAMATH_CALUDE_lamp_post_ratio_l1685_168541


namespace NUMINAMATH_CALUDE_quadratic_function_sign_l1685_168587

/-- Given a quadratic function f(x) = x^2 - x + a, where f(-m) < 0,
    prove that f(m+1) is negative. -/
theorem quadratic_function_sign (a m : ℝ) : 
  let f := λ x : ℝ => x^2 - x + a
  f (-m) < 0 → f (m + 1) < 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_sign_l1685_168587


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_solution_l1685_168507

theorem sum_of_squares_quadratic_solution : 
  ∀ (s₁ s₂ : ℝ), s₁^2 - 10*s₁ + 7 = 0 → s₂^2 - 10*s₂ + 7 = 0 → s₁^2 + s₂^2 = 86 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_solution_l1685_168507


namespace NUMINAMATH_CALUDE_cubic_harmonic_mean_root_condition_l1685_168557

/-- 
Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and d ≠ 0,
if one of its roots is equal to the harmonic mean of the other two roots,
then the coefficients satisfy the equation 27ad² - 9bcd + 2c³ = 0.
-/
theorem cubic_harmonic_mean_root_condition (a b c d : ℝ) 
  (ha : a ≠ 0) (hd : d ≠ 0) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (a * x₁^3 + b * x₁^2 + c * x₁ + d = 0) ∧ 
    (a * x₂^3 + b * x₂^2 + c * x₂ + d = 0) ∧ 
    (a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) ∧ 
    (x₂ = 2 * x₁ * x₃ / (x₁ + x₃))) →
  27 * a * d^2 - 9 * b * c * d + 2 * c^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_harmonic_mean_root_condition_l1685_168557


namespace NUMINAMATH_CALUDE_plumber_distribution_theorem_l1685_168563

/-- The number of ways to distribute n plumbers to k houses -/
def distribute_plumbers (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else Nat.choose n 2 * (Nat.factorial k)

theorem plumber_distribution_theorem :
  distribute_plumbers 4 3 = Nat.choose 4 2 * (Nat.factorial 3) :=
sorry

end NUMINAMATH_CALUDE_plumber_distribution_theorem_l1685_168563


namespace NUMINAMATH_CALUDE_box_comparison_l1685_168595

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def box_lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.x ∧ a.y ≤ b.z ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.x ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.z ∧ a.z ≤ b.x) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.x ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.y ∧ a.z ≤ b.x)

def box_gt (a b : Box) : Prop := box_lt b a

theorem box_comparison :
  let A : Box := ⟨5, 6, 3⟩
  let B : Box := ⟨1, 5, 4⟩
  let C : Box := ⟨2, 2, 3⟩
  (box_gt A B) ∧ (box_lt C A) := by sorry

end NUMINAMATH_CALUDE_box_comparison_l1685_168595


namespace NUMINAMATH_CALUDE_negation_of_odd_function_implication_l1685_168537

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (IsOdd f → IsOdd (fun x ↦ f (-x)))) ↔ (¬ IsOdd f → ¬ IsOdd (fun x ↦ f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_odd_function_implication_l1685_168537


namespace NUMINAMATH_CALUDE_kitten_growth_ratio_l1685_168544

/-- Given the initial length and final length of a kitten, and knowing that the final length is twice the intermediate length, prove that the ratio of intermediate length to initial length is 2. -/
theorem kitten_growth_ratio (L₀ L₂ L₄ : ℝ) (h₀ : L₀ = 4) (h₄ : L₄ = 16) (h_double : L₄ = 2 * L₂) : L₂ / L₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_ratio_l1685_168544


namespace NUMINAMATH_CALUDE_combined_transformation_correct_l1685_168588

def dilation_matrix (scale : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![scale, 0; 0, scale]

def reflection_x_axis : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![5, 0; 0, -5]

theorem combined_transformation_correct :
  combined_transformation = reflection_x_axis * dilation_matrix 5 := by
  sorry

end NUMINAMATH_CALUDE_combined_transformation_correct_l1685_168588


namespace NUMINAMATH_CALUDE_percentage_of_70_to_125_l1685_168551

theorem percentage_of_70_to_125 : ∃ p : ℚ, p = 70 / 125 * 100 ∧ p = 56 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_70_to_125_l1685_168551


namespace NUMINAMATH_CALUDE_invariant_parity_and_final_digit_l1685_168594

/-- Represents the count of each digit -/
structure DigitCounts where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents the possible operations on the board -/
inductive Operation
  | replaceZeroOne
  | replaceOneTwo
  | replaceZeroTwo

/-- Applies an operation to the digit counts -/
def applyOperation (counts : DigitCounts) (op : Operation) : DigitCounts :=
  match op with
  | Operation.replaceZeroOne => ⟨counts.zeros - 1, counts.ones - 1, counts.twos + 1⟩
  | Operation.replaceOneTwo => ⟨counts.zeros + 1, counts.ones - 1, counts.twos - 1⟩
  | Operation.replaceZeroTwo => ⟨counts.zeros - 1, counts.ones + 1, counts.twos - 1⟩

/-- The parity of the sum of digit counts -/
def sumParity (counts : DigitCounts) : ℕ :=
  (counts.zeros + counts.ones + counts.twos) % 2

/-- The final remaining digit -/
def finalDigit (initialCounts : DigitCounts) : ℕ :=
  if initialCounts.zeros % 2 ≠ initialCounts.ones % 2 ∧ initialCounts.zeros % 2 ≠ initialCounts.twos % 2 then 0
  else if initialCounts.ones % 2 ≠ initialCounts.zeros % 2 ∧ initialCounts.ones % 2 ≠ initialCounts.twos % 2 then 1
  else 2

theorem invariant_parity_and_final_digit (initialCounts : DigitCounts) (ops : List Operation) :
  (sumParity initialCounts = sumParity (ops.foldl applyOperation initialCounts)) ∧
  (finalDigit initialCounts = finalDigit (ops.foldl applyOperation initialCounts)) :=
sorry

end NUMINAMATH_CALUDE_invariant_parity_and_final_digit_l1685_168594


namespace NUMINAMATH_CALUDE_birds_on_fence_l1685_168513

theorem birds_on_fence (initial_birds additional_birds : ℕ) : 
  initial_birds = 2 → additional_birds = 4 → initial_birds + additional_birds = 6 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1685_168513


namespace NUMINAMATH_CALUDE_factorization_validity_l1685_168559

theorem factorization_validity (x y : ℝ) :
  x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_validity_l1685_168559


namespace NUMINAMATH_CALUDE_original_paint_intensity_l1685_168516

/-- Proves that the intensity of the original red paint was 45% given the specified conditions. -/
theorem original_paint_intensity
  (replace_fraction : Real)
  (replacement_solution_intensity : Real)
  (final_intensity : Real)
  (h1 : replace_fraction = 0.25)
  (h2 : replacement_solution_intensity = 0.25)
  (h3 : final_intensity = 0.40) :
  ∃ (original_intensity : Real),
    original_intensity = 0.45 ∧
    (1 - replace_fraction) * original_intensity +
    replace_fraction * replacement_solution_intensity = final_intensity :=
by
  sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l1685_168516


namespace NUMINAMATH_CALUDE_first_day_income_l1685_168530

/-- Given a sequence where each term is double the previous term,
    and the 10th term is 18, prove that the first term is 0.03515625 -/
theorem first_day_income (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) (h2 : a 10 = 18) :
  a 1 = 0.03515625 := by
  sorry

end NUMINAMATH_CALUDE_first_day_income_l1685_168530


namespace NUMINAMATH_CALUDE_max_earnings_theorem_l1685_168545

/-- Represents the exchange rates for a given day -/
structure ExchangeRates where
  gbp_to_usd : ℝ
  jpy_to_usd : ℝ
  eur_to_usd : ℝ

/-- Calculates the maximum total earnings in USD -/
def max_total_earnings (usd_hours : ℝ) (gbp_hours : ℝ) (jpy_hours : ℝ) (eur_hours : ℝ)
  (usd_rate : ℝ) (gbp_rate : ℝ) (jpy_rate : ℝ) (eur_rate : ℝ)
  (day1 : ExchangeRates) (day2 : ExchangeRates) (day3 : ExchangeRates) : ℝ :=
  sorry

/-- Theorem stating that the maximum total earnings is $32.61 -/
theorem max_earnings_theorem :
  let day1 : ExchangeRates := { gbp_to_usd := 1.35, jpy_to_usd := 0.009, eur_to_usd := 1.18 }
  let day2 : ExchangeRates := { gbp_to_usd := 1.38, jpy_to_usd := 0.0085, eur_to_usd := 1.20 }
  let day3 : ExchangeRates := { gbp_to_usd := 1.33, jpy_to_usd := 0.0095, eur_to_usd := 1.21 }
  max_total_earnings 4 0.5 1.5 1 5 3 400 4 day1 day2 day3 = 32.61 := by
  sorry

end NUMINAMATH_CALUDE_max_earnings_theorem_l1685_168545
