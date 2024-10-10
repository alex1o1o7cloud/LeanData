import Mathlib

namespace largest_of_seven_consecutive_integers_l2215_221564

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h1 : n > 0)
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2222) : 
  n + 6 = 320 :=
sorry

end largest_of_seven_consecutive_integers_l2215_221564


namespace book_arrangement_l2215_221585

theorem book_arrangement (n m : ℕ) (h : n + m = 11) :
  Nat.choose (n + m) n = 462 :=
sorry

end book_arrangement_l2215_221585


namespace movie_duration_l2215_221568

theorem movie_duration (screens : ℕ) (open_hours : ℕ) (total_movies : ℕ) 
  (h1 : screens = 6) 
  (h2 : open_hours = 8) 
  (h3 : total_movies = 24) : 
  (screens * open_hours) / total_movies = 2 := by
  sorry

end movie_duration_l2215_221568


namespace cube_roots_less_than_12_l2215_221549

theorem cube_roots_less_than_12 : 
  (Finset.range 1728).card = 1727 :=
by sorry

end cube_roots_less_than_12_l2215_221549


namespace exponential_fraction_simplification_l2215_221589

theorem exponential_fraction_simplification :
  (3^1008 + 3^1006) / (3^1008 - 3^1006) = 5/4 := by
  sorry

end exponential_fraction_simplification_l2215_221589


namespace train_crossing_time_l2215_221533

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 110 →
  train_speed_kmh = 144 →
  time_to_cross = train_length / (train_speed_kmh * (1000 / 3600)) →
  time_to_cross = 2.75 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2215_221533


namespace solve_scooter_problem_l2215_221553

def scooter_problem (purchase_price repair_cost gain_percentage : ℝ) : Prop :=
  let total_cost := purchase_price + repair_cost
  let gain := total_cost * (gain_percentage / 100)
  let selling_price := total_cost + gain
  (purchase_price = 4700) ∧ 
  (repair_cost = 1000) ∧ 
  (gain_percentage = 1.7543859649122806) →
  selling_price = 5800

theorem solve_scooter_problem :
  scooter_problem 4700 1000 1.7543859649122806 :=
by
  sorry

end solve_scooter_problem_l2215_221553


namespace elevator_exit_theorem_l2215_221571

/-- The number of ways 9 passengers can exit an elevator in groups of 2, 3, and 4 at any of 10 floors -/
def elevator_exit_ways : ℕ :=
  Nat.factorial 10 / Nat.factorial 4

/-- Theorem stating that the number of ways 9 passengers can exit an elevator
    in groups of 2, 3, and 4 at any of 10 floors is equal to 10! / 4! -/
theorem elevator_exit_theorem :
  elevator_exit_ways = Nat.factorial 10 / Nat.factorial 4 := by
  sorry

end elevator_exit_theorem_l2215_221571


namespace one_zero_point_condition_l2215_221543

/-- A quadratic function with only one zero point -/
def has_one_zero_point (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 - x - 1 = 0

/-- The theorem stating the condition for a quadratic function to have only one zero point -/
theorem one_zero_point_condition (a : ℝ) :
  has_one_zero_point a ↔ a = 0 ∨ a = -1/4 :=
sorry

end one_zero_point_condition_l2215_221543


namespace f_not_in_first_quadrant_l2215_221559

/-- A linear function defined by y = -3x + 2 -/
def f (x : ℝ) : ℝ := -3 * x + 2

/-- Definition of the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Theorem stating that the function f does not pass through the first quadrant -/
theorem f_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
sorry

end f_not_in_first_quadrant_l2215_221559


namespace floor_sqrt_equation_solutions_l2215_221558

theorem floor_sqrt_equation_solutions : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 1000) / 70 = ⌊Real.sqrt n⌋) ∧ 
    Finset.card S = 6 := by sorry

end floor_sqrt_equation_solutions_l2215_221558


namespace min_value_a_plus_2b_l2215_221598

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 := by
  sorry

end min_value_a_plus_2b_l2215_221598


namespace change_distance_scientific_notation_l2215_221537

/-- Definition of scientific notation -/
def is_scientific_notation (n : ℝ) (a : ℝ) (p : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ p ∧ 1 ≤ |a| ∧ |a| < 10

/-- The distance of Chang'e 1 from Earth in kilometers -/
def change_distance : ℝ := 380000

/-- Theorem stating that 380,000 is equal to 3.8 × 10^5 in scientific notation -/
theorem change_distance_scientific_notation :
  is_scientific_notation change_distance 3.8 5 :=
sorry

end change_distance_scientific_notation_l2215_221537


namespace art_gallery_theorem_l2215_221516

theorem art_gallery_theorem (T : ℕ) 
  (h1 : T / 3 = T - (2 * T / 3))  -- 1/3 of pieces are displayed
  (h2 : (T / 3) / 6 = T / 18)  -- 1/6 of displayed pieces are sculptures
  (h3 : (2 * T / 3) / 3 = 2 * T / 9)  -- 1/3 of non-displayed pieces are paintings
  (h4 : T / 18 + 400 = T / 18 + (T - (T / 3)) / 3)  -- 400 sculptures not on display
  (h5 : 3 * (T / 18) = T / 6)  -- 3 photographs for each displayed sculpture
  (h6 : 2 * (T / 18) = T / 9)  -- 2 installations for each displayed sculpture
  : T = 7200 := by
sorry

end art_gallery_theorem_l2215_221516


namespace simplify_expression_l2215_221563

theorem simplify_expression (x : ℝ) (h : x > 3) :
  3 * |3 - x| - |x^2 - 6*x + 10| + |x^2 - 2*x + 1| = 7*x - 18 := by
  sorry

end simplify_expression_l2215_221563


namespace hospital_staff_count_l2215_221541

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h_total : total = 280)
  (h_ratio : doctor_ratio = 5 ∧ nurse_ratio = 9) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 180 :=
by sorry

end hospital_staff_count_l2215_221541


namespace line_l_properties_l2215_221566

/-- Given a line l: (m-1)x + 2my + 2 = 0, where m is a real number -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (m - 1) * x + 2 * m * y + 2 = 0

theorem line_l_properties (m : ℝ) :
  /- 1. Line l passes through the point (2, -1) -/
  line_l m 2 (-1) ∧
  /- 2. If the slope of l is non-positive and y-intercept is non-negative, then m ≤ 0 -/
  ((∀ x y : ℝ, line_l m x y → (1 - m) / (2 * m) ≤ 0 ∧ -1 / m ≥ 0) → m ≤ 0) ∧
  /- 3. When the x-intercept equals the y-intercept, m = -1 -/
  (∃ a : ℝ, a ≠ 0 ∧ line_l m a 0 ∧ line_l m 0 (-a)) → m = -1 :=
by sorry

end line_l_properties_l2215_221566


namespace tank_capacity_l2215_221536

/-- The capacity of a tank given outlet and inlet pipe rates -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (combined_time : ℝ)
  (h1 : outlet_time = 8)
  (h2 : inlet_rate = 8)
  (h3 : combined_time = 12) :
  ∃ (capacity : ℝ), capacity = 11520 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / combined_time :=
sorry

end tank_capacity_l2215_221536


namespace blue_eyed_percentage_is_correct_l2215_221595

def cat_kittens : List (ℕ × ℕ) := [(5, 7), (6, 8), (4, 6), (7, 9), (3, 5)]

def total_blue_eyed : ℕ := (cat_kittens.map Prod.fst).sum

def total_kittens : ℕ := (cat_kittens.map (λ p => p.fst + p.snd)).sum

def blue_eyed_percentage : ℚ := (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100

theorem blue_eyed_percentage_is_correct : 
  blue_eyed_percentage = 125/3 := by sorry

end blue_eyed_percentage_is_correct_l2215_221595


namespace coconut_crab_goat_trade_l2215_221540

theorem coconut_crab_goat_trade (coconuts_per_crab : ℕ) (total_coconuts : ℕ) (final_goats : ℕ) :
  coconuts_per_crab = 3 →
  total_coconuts = 342 →
  final_goats = 19 →
  (total_coconuts / coconuts_per_crab) / final_goats = 6 :=
by sorry

end coconut_crab_goat_trade_l2215_221540


namespace stewart_farm_sheep_horse_ratio_l2215_221597

/-- Represents the Stewart farm with sheep and horses -/
structure StewartFarm where
  sheep : ℕ
  horses : ℕ
  horseFoodPerHorse : ℕ
  totalHorseFood : ℕ

/-- The ratio between two natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the simplified ratio between two natural numbers -/
def simplifiedRatio (a b : ℕ) : Ratio :=
  let gcd := Nat.gcd a b
  { numerator := a / gcd, denominator := b / gcd }

/-- Theorem: The ratio of sheep to horses on the Stewart farm is 5:7 -/
theorem stewart_farm_sheep_horse_ratio (farm : StewartFarm)
    (h1 : farm.sheep = 40)
    (h2 : farm.horseFoodPerHorse = 230)
    (h3 : farm.totalHorseFood = 12880)
    (h4 : farm.horses * farm.horseFoodPerHorse = farm.totalHorseFood) :
    simplifiedRatio farm.sheep farm.horses = { numerator := 5, denominator := 7 } := by
  sorry

end stewart_farm_sheep_horse_ratio_l2215_221597


namespace group_ratio_l2215_221555

theorem group_ratio (x : ℝ) (h1 : x > 0) (h2 : 1 - x > 0) : 
  15 * x + 21 * (1 - x) = 20 → x / (1 - x) = 1 / 5 := by
  sorry

end group_ratio_l2215_221555


namespace matrix_multiplication_result_l2215_221574

/-- Given real numbers d, e, and f, prove that the matrix multiplication of 
    A = [[0, d, -e], [-d, 0, f], [e, -f, 0]] and 
    B = [[f^2, fd, fe], [fd, d^2, de], [fe, de, e^2]] 
    results in 
    C = [[d^2 - e^2, 2fd, 0], [0, f^2 - d^2, de-fe], [0, e^2 - d^2, fe-df]] -/
theorem matrix_multiplication_result (d e f : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![f^2, f*d, f*e; f*d, d^2, d*e; f*e, d*e, e^2]
  let C : Matrix (Fin 3) (Fin 3) ℝ := !![d^2 - e^2, 2*f*d, 0; 0, f^2 - d^2, d*e - f*e; 0, e^2 - d^2, f*e - d*f]
  A * B = C := by sorry

end matrix_multiplication_result_l2215_221574


namespace function_properties_l2215_221567

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

/-- The main theorem stating the properties of the function -/
theorem function_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x := by
  sorry

end function_properties_l2215_221567


namespace calculator_game_sum_l2215_221579

/-- Represents the operations performed on the calculators. -/
def calculatorOperations (n : ℕ) (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (a^3, b^2, c + 1)

/-- Applies the operations n times to the initial values. -/
def applyNTimes (n : ℕ) : ℕ × ℕ × ℕ :=
  match n with
  | 0 => (2, 1, 0)
  | m + 1 => calculatorOperations m (applyNTimes m).1 (applyNTimes m).2.1 (applyNTimes m).2.2

/-- The main theorem to be proved. -/
theorem calculator_game_sum :
  let (a, b, c) := applyNTimes 50
  a + b + c = 307 := by sorry

end calculator_game_sum_l2215_221579


namespace f_minimum_at_two_l2215_221545

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_minimum_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 :=
sorry

end f_minimum_at_two_l2215_221545


namespace men_finished_race_l2215_221593

/-- The number of men who finished the race given the specified conditions -/
def men_who_finished (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped * 3 / 8
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 9
  let dehydrated_not_finished := dehydrated * 11 / 14
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 17 / 100
  let lost_finished := lost * 5 / 11
  let remaining_after_lost := remaining_after_dehydration - lost
  let obstacle := remaining_after_lost * 5 / 12
  let obstacle_finished := obstacle * 7 / 15
  let remaining_after_obstacle := remaining_after_lost - obstacle
  let cramps := remaining_after_obstacle * 3 / 7
  let cramps_finished := cramps * 4 / 5
  tripped_finished + lost_finished + obstacle_finished + cramps_finished

/-- Theorem stating that 25 men finished the race given the specified conditions -/
theorem men_finished_race : men_who_finished 80 = 25 := by
  sorry

end men_finished_race_l2215_221593


namespace right_triangle_legs_sum_l2215_221521

theorem right_triangle_legs_sum (a b c : ℕ) : 
  a + 1 = b →                   -- legs are consecutive integers
  a^2 + b^2 = 41^2 →            -- Pythagorean theorem with hypotenuse 41
  a + b = 57 := by              -- sum of legs is 57
sorry

end right_triangle_legs_sum_l2215_221521


namespace cubic_equation_properties_l2215_221581

theorem cubic_equation_properties (k : ℝ) :
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                (x ≠ y ∨ y ≠ z ∨ x ≠ z)) ∧
  (∀ x y z : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 ∧
                k * y^3 + 2 * k * y^2 + 6 * k * y + 2 = 0 ∧
                k * z^3 + 2 * k * z^2 + 6 * k * z + 2 = 0 →
                x + y + z = -2) :=
by sorry

end cubic_equation_properties_l2215_221581


namespace square_area_and_perimeter_comparison_l2215_221525

theorem square_area_and_perimeter_comparison (a b : ℝ) :
  let square_I_diagonal := 2 * (a + b)
  let square_II_area := 4 * (square_I_diagonal^2 / 4)
  let square_II_perimeter := 4 * Real.sqrt square_II_area
  let rectangle_perimeter := 2 * (4 * (a + b) + (a + b))
  square_II_area = 8 * (a + b)^2 ∧ square_II_perimeter > rectangle_perimeter :=
by sorry

end square_area_and_perimeter_comparison_l2215_221525


namespace books_returned_percentage_l2215_221561

/-- Calculates the percentage of loaned books that were returned -/
def percentage_returned (initial_books : ℕ) (loaned_books : ℕ) (final_books : ℕ) : ℚ :=
  ((final_books - (initial_books - loaned_books)) : ℚ) / (loaned_books : ℚ) * 100

/-- Theorem stating that the percentage of returned books is 65% -/
theorem books_returned_percentage 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 20)
  (h3 : final_books = 68) : 
  percentage_returned initial_books loaned_books final_books = 65 := by
  sorry

#eval percentage_returned 75 20 68

end books_returned_percentage_l2215_221561


namespace sequence_product_l2215_221596

theorem sequence_product (n a : ℕ) :
  ∃ u v : ℕ, n / (n + a) = (u / (u + a)) * (v / (v + a)) :=
by sorry

end sequence_product_l2215_221596


namespace center_is_one_l2215_221514

/-- A 3x3 table of positive real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)
  (all_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)

/-- The conditions for the table -/
def TableConditions (t : Table) : Prop :=
  t.a * t.b * t.c = 1 ∧
  t.d * t.e * t.f = 1 ∧
  t.g * t.h * t.i = 1 ∧
  t.a * t.d * t.g = 1 ∧
  t.b * t.e * t.h = 1 ∧
  t.c * t.f * t.i = 1 ∧
  t.a * t.b * t.d * t.e = 2 ∧
  t.b * t.c * t.e * t.f = 2 ∧
  t.d * t.e * t.g * t.h = 2 ∧
  t.e * t.f * t.h * t.i = 2

/-- The theorem stating that the center cell must be 1 -/
theorem center_is_one (t : Table) (h : TableConditions t) : t.e = 1 := by
  sorry


end center_is_one_l2215_221514


namespace mona_grouped_before_l2215_221551

/-- Represents the game groups Mona joined -/
structure GameGroups where
  totalGroups : ℕ
  playersPerGroup : ℕ
  uniquePlayers : ℕ
  knownPlayersInOneGroup : ℕ

/-- Calculates the number of players Mona had grouped with before in a specific group -/
def playersGroupedBefore (g : GameGroups) : ℕ :=
  g.totalGroups * g.playersPerGroup - g.uniquePlayers - g.knownPlayersInOneGroup

/-- Theorem stating the number of players Mona had grouped with before in a specific group -/
theorem mona_grouped_before (g : GameGroups) 
  (h1 : g.totalGroups = 9)
  (h2 : g.playersPerGroup = 4)
  (h3 : g.uniquePlayers = 33)
  (h4 : g.knownPlayersInOneGroup = 1) :
  playersGroupedBefore g = 2 := by
    sorry

end mona_grouped_before_l2215_221551


namespace f_value_l2215_221556

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (x + 1) = -f (-x + 1)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, -1 ≤ x → x ≤ 0 → f x = -2 * x * (x + 1)

-- State the theorem to be proved
theorem f_value : f (-3/2) = -1/2 := by sorry

end f_value_l2215_221556


namespace integer_root_quadratic_count_l2215_221588

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), (∀ a ∈ S, ∃ r s : ℤ, ∀ x : ℝ, x^2 + a*x + 9*a = 0 ↔ x = r ∨ x = s) ∧ Finset.card S = 10 :=
sorry

end integer_root_quadratic_count_l2215_221588


namespace large_pepperoni_has_14_slices_l2215_221507

/-- The number of slices in a large pepperoni pizza -/
def large_pepperoni_slices (total_eaten : ℕ) (total_left : ℕ) (small_cheese_slices : ℕ) : ℕ :=
  total_eaten + total_left - small_cheese_slices

/-- Theorem stating that the large pepperoni pizza has 14 slices -/
theorem large_pepperoni_has_14_slices : 
  large_pepperoni_slices 18 4 8 = 14 := by
  sorry

end large_pepperoni_has_14_slices_l2215_221507


namespace simplify_expression_l2215_221576

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : 
  x/y + y/x - 3/(x*y) = 1 := by
  sorry

end simplify_expression_l2215_221576


namespace average_book_price_l2215_221587

/-- The average price of books bought from two shops -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 27 →
  books2 = 20 →
  price1 = 581 →
  price2 = 594 →
  (price1 + price2) / (books1 + books2 : ℚ) = 25 := by
  sorry

end average_book_price_l2215_221587


namespace quadratic_form_k_value_l2215_221539

theorem quadratic_form_k_value : ∃ (a h : ℝ), ∀ x : ℝ, 
  x^2 - 6*x = a*(x - h)^2 + (-9) := by
  sorry

end quadratic_form_k_value_l2215_221539


namespace median_equation_altitude_equation_l2215_221569

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the equation of a line
def is_line_equation (a b c : ℝ) (p : ℝ × ℝ) : Prop :=
  a * p.1 + b * p.2 = c

-- Theorem for the median equation
theorem median_equation : 
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 5*x + y = 20) ∧
    is_line_equation a b c A ∧
    is_line_equation a b c ((B.1 + C.1) / 2, (B.2 + C.2) / 2) :=
sorry

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (a b c : ℝ),
    (∀ (x y : ℝ), is_line_equation a b c (x, y) ↔ 3*x + 2*y = 12) ∧
    is_line_equation a b c A ∧
    (∀ (p : ℝ × ℝ), is_line_equation (B.2 - C.2) (C.1 - B.1) 0 p → 
      (p.2 - A.2) * (p.1 - A.1) = -(a * (p.1 - A.1) + b * (p.2 - A.2))^2 / (a^2 + b^2)) :=
sorry

end median_equation_altitude_equation_l2215_221569


namespace students_taking_statistics_l2215_221544

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 59 students are taking history or statistics or both
  * 27 students are taking history but not statistics
  Prove that 32 students are taking statistics -/
theorem students_taking_statistics
  (total : ℕ)
  (history : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 27) :
  history_or_statistics - (history - history_not_statistics) = 32 := by
  sorry

#check students_taking_statistics

end students_taking_statistics_l2215_221544


namespace perpendicular_foot_coordinates_l2215_221526

/-- Given a point P(1, √2, √3) in a 3-D Cartesian coordinate system and a perpendicular line PQ 
    drawn from P to the plane xOy with Q as the foot of the perpendicular, 
    prove that the coordinates of point Q are (1, √2, 0). -/
theorem perpendicular_foot_coordinates :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  ∃ Q : ℝ × ℝ × ℝ, Q ∈ xOy_plane ∧ 
    (Q.1 = P.1 ∧ Q.2.1 = P.2.1 ∧ Q.2.2 = 0) ∧
    (∀ R ∈ xOy_plane, (P.1 - R.1)^2 + (P.2.1 - R.2.1)^2 + (P.2.2 - R.2.2)^2 ≥
                      (P.1 - Q.1)^2 + (P.2.1 - Q.2.1)^2 + (P.2.2 - Q.2.2)^2) :=
by sorry

end perpendicular_foot_coordinates_l2215_221526


namespace simplify_fraction_l2215_221531

theorem simplify_fraction (a : ℝ) (h : a ≠ -3) :
  (a^2 / (a + 3)) - (9 / (a + 3)) = a - 3 := by
  sorry

end simplify_fraction_l2215_221531


namespace sqrt_meaningful_range_l2215_221517

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end sqrt_meaningful_range_l2215_221517


namespace triangle_area_with_perpendicular_medians_l2215_221504

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define a median of a triangle
def Median (A B C M : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def Perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the altitude of a triangle
def Altitude (A B C H : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_with_perpendicular_medians 
  (X Y Z U V : ℝ × ℝ) 
  (h1 : Triangle X Y Z)
  (h2 : Median X Y Z U)
  (h3 : Median Y Z X V)
  (h4 : Perpendicular X U Y V)
  (h5 : Length X U = 10)
  (h6 : Length Y V = 24)
  (h7 : ∃ H, Altitude Z X Y H ∧ Length Z H = 16) :
  TriangleArea X Y Z = 160 := by
  sorry

end triangle_area_with_perpendicular_medians_l2215_221504


namespace complement_intersection_theorem_l2215_221583

def U : Set ℤ := {x | -4 < x ∧ x < 4}
def A : Set ℤ := {-1, 0, 2, 3}
def B : Set ℤ := {-2, 0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {-3} := by sorry

end complement_intersection_theorem_l2215_221583


namespace total_chair_cost_l2215_221523

/-- Represents the cost calculation for chairs in a room -/
structure RoomChairs where
  count : Nat
  price : Nat

/-- Calculates the total cost for a set of room chairs -/
def totalCost (rc : RoomChairs) : Nat :=
  rc.count * rc.price

/-- Theorem: The total cost of chairs for the entire house is $2045 -/
theorem total_chair_cost (livingRoom kitchen diningRoom patio : RoomChairs)
    (h1 : livingRoom = ⟨3, 75⟩)
    (h2 : kitchen = ⟨6, 50⟩)
    (h3 : diningRoom = ⟨8, 100⟩)
    (h4 : patio = ⟨12, 60⟩) :
    totalCost livingRoom + totalCost kitchen + totalCost diningRoom + totalCost patio = 2045 := by
  sorry

#eval totalCost ⟨3, 75⟩ + totalCost ⟨6, 50⟩ + totalCost ⟨8, 100⟩ + totalCost ⟨12, 60⟩

end total_chair_cost_l2215_221523


namespace cube_monotone_l2215_221500

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_monotone_l2215_221500


namespace factorial_sum_equality_l2215_221554

theorem factorial_sum_equality : 
  ∃! (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  Nat.factorial w = Nat.factorial x + Nat.factorial y + Nat.factorial z ∧
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end factorial_sum_equality_l2215_221554


namespace probability_two_nondefective_pens_l2215_221502

/-- Given a box of 12 pens with 3 defective pens, prove that the probability
    of selecting 2 non-defective pens at random without replacement is 6/11. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat)
    (h1 : total_pens = 12)
    (h2 : defective_pens = 3) :
    (total_pens - defective_pens : ℚ) / total_pens *
    ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

end probability_two_nondefective_pens_l2215_221502


namespace snowfall_probability_l2215_221573

theorem snowfall_probability (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.3) :
  (1 - p_A) * (1 - p_B) = 0.42 := by
  sorry

end snowfall_probability_l2215_221573


namespace correct_subtraction_l2215_221542

-- Define the polynomials
def original_poly (x : ℝ) := 2*x^2 - x + 3
def mistaken_poly (x : ℝ) := x^2 + 14*x - 6

-- Theorem statement
theorem correct_subtraction :
  ∀ x : ℝ, original_poly x - mistaken_poly x = -29*x + 15 := by
  sorry

end correct_subtraction_l2215_221542


namespace fraction_equals_one_l2215_221565

theorem fraction_equals_one (x : ℝ) : x ≠ 3 → ((2 * x - 7) / (x - 3) = 1 ↔ x = 4) := by
  sorry

end fraction_equals_one_l2215_221565


namespace problem_statement_l2215_221508

theorem problem_statement : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end problem_statement_l2215_221508


namespace airport_distance_solution_l2215_221509

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistanceProblem (initial_speed : ℝ) (speed_increase : ℝ) (stop_time : ℝ) (early_arrival : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- The first portion is driven at the initial speed
    initial_speed * initial_time = 40 ∧
    -- The total time includes the initial drive, stop time, and the rest of the journey
    (distance - 40) / (initial_speed + speed_increase) + initial_time + stop_time = 
      (distance / initial_speed) - early_arrival ∧
    -- The total distance is 190 miles
    distance = 190

/-- Theorem stating that the solution to the problem is 190 miles --/
theorem airport_distance_solution :
  AirportDistanceProblem 40 20 0.25 0.25 := by
  sorry

#check airport_distance_solution

end airport_distance_solution_l2215_221509


namespace line_through_point_l2215_221535

/-- Given a line equation -1/2 - 2kx = 5y that passes through the point (1/4, -6),
    prove that k = 59 is the unique solution. -/
theorem line_through_point (k : ℝ) : 
  (-1/2 : ℝ) - 2 * k * (1/4 : ℝ) = 5 * (-6 : ℝ) ↔ k = 59 := by
sorry

end line_through_point_l2215_221535


namespace henrys_room_books_l2215_221506

/-- The number of books Henry had in the room to donate -/
def books_in_room (initial_books : ℕ) (bookshelf_boxes : ℕ) (books_per_box : ℕ)
  (coffee_table_books : ℕ) (kitchen_books : ℕ) (free_books_taken : ℕ) (final_books : ℕ) : ℕ :=
  initial_books - (bookshelf_boxes * books_per_box + coffee_table_books + kitchen_books - free_books_taken)

/-- Theorem stating the number of books Henry had in the room to donate -/
theorem henrys_room_books :
  books_in_room 99 3 15 4 18 12 23 = 44 := by
  sorry

end henrys_room_books_l2215_221506


namespace series_sum_equals_one_l2215_221546

theorem series_sum_equals_one : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 1)) = 1 := by sorry

end series_sum_equals_one_l2215_221546


namespace michelle_taxi_cost_l2215_221518

/-- Calculates the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxi_cost (initial_fee : ℝ) (distance : ℝ) (per_mile_charge : ℝ) : ℝ :=
  initial_fee + distance * per_mile_charge

/-- Theorem stating that for the given conditions, the total cost is $12. -/
theorem michelle_taxi_cost : taxi_cost 2 4 2.5 = 12 := by sorry

end michelle_taxi_cost_l2215_221518


namespace sin_five_pi_thirds_l2215_221562

theorem sin_five_pi_thirds : Real.sin (5 * π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_five_pi_thirds_l2215_221562


namespace class_average_score_l2215_221527

theorem class_average_score (total_students : Nat) (group1_students : Nat) (group1_average : ℚ)
  (score1 score2 score3 score4 : ℚ) :
  total_students = 30 →
  group1_students = 26 →
  group1_average = 82 →
  score1 = 90 →
  score2 = 85 →
  score3 = 88 →
  score4 = 80 →
  let group1_total := group1_students * group1_average
  let group2_total := score1 + score2 + score3 + score4
  let class_total := group1_total + group2_total
  class_total / total_students = 82.5 := by
sorry

end class_average_score_l2215_221527


namespace infinite_series_sum_l2215_221520

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to -5/3 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = -5/3 := by
  sorry

end infinite_series_sum_l2215_221520


namespace dessert_preference_l2215_221528

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 50 →
  apple = 22 →
  chocolate = 20 →
  neither = 17 →
  ∃ (both : ℕ), both = apple + chocolate - (total - neither) :=
by sorry

end dessert_preference_l2215_221528


namespace intersection_range_value_range_on_curve_l2215_221560

-- Define the line l
def line_l (α : Real) : Set (Real × Real) :=
  {(x, y) | ∃ t, x = -2 + t * Real.cos α ∧ y = t * Real.sin α}

-- Define the curve C
def curve_C : Set (Real × Real) :=
  {(x, y) | (x - 2)^2 + y^2 = 4}

-- Theorem for part (I)
theorem intersection_range (α : Real) :
  (∃ p, p ∈ line_l α ∧ p ∈ curve_C) ↔ 
  (0 ≤ α ∧ α ≤ Real.pi/6) ∨ (5*Real.pi/6 ≤ α ∧ α ≤ Real.pi) :=
sorry

-- Theorem for part (II)
theorem value_range_on_curve :
  ∀ (x y : Real), (x, y) ∈ curve_C → -2 ≤ x + Real.sqrt 3 * y ∧ x + Real.sqrt 3 * y ≤ 6 :=
sorry

end intersection_range_value_range_on_curve_l2215_221560


namespace sugar_concentration_mixture_l2215_221513

/-- Given two solutions with different sugar concentrations, calculate the sugar concentration of the resulting mixture --/
theorem sugar_concentration_mixture (original_concentration : ℝ) (replacement_concentration : ℝ)
  (replacement_fraction : ℝ) (h1 : original_concentration = 0.12)
  (h2 : replacement_concentration = 0.28000000000000004) (h3 : replacement_fraction = 0.25) :
  (1 - replacement_fraction) * original_concentration + replacement_fraction * replacement_concentration = 0.16 := by
  sorry

end sugar_concentration_mixture_l2215_221513


namespace max_odd_sequence_length_l2215_221578

/-- The type of sequences where each term is obtained by adding the largest digit of the previous term --/
def DigitAddSequence := ℕ → ℕ

/-- The largest digit of a natural number --/
def largest_digit (n : ℕ) : ℕ := sorry

/-- The property that a sequence follows the digit addition rule --/
def is_digit_add_sequence (s : DigitAddSequence) : Prop :=
  ∀ n, s (n + 1) = s n + largest_digit (s n)

/-- The property that a number is odd --/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- The length of a sequence of successive odd terms starting from a given index --/
def odd_sequence_length (s : DigitAddSequence) (start : ℕ) : ℕ := sorry

/-- The theorem stating that the maximal number of successive odd terms is 5 --/
theorem max_odd_sequence_length (s : DigitAddSequence) (h : is_digit_add_sequence s) :
  ∀ start, odd_sequence_length s start ≤ 5 :=
sorry

end max_odd_sequence_length_l2215_221578


namespace studio_audience_size_l2215_221582

theorem studio_audience_size :
  ∀ (total : ℕ) (envelope_ratio winner_ratio : ℚ) (winners : ℕ),
    envelope_ratio = 2/5 →
    winner_ratio = 1/5 →
    winners = 8 →
    (envelope_ratio * winner_ratio * total : ℚ) = winners →
    total = 100 := by
  sorry

end studio_audience_size_l2215_221582


namespace binomial_square_condition_l2215_221532

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end binomial_square_condition_l2215_221532


namespace total_points_is_94_bonus_points_is_7_l2215_221524

/-- Represents the points system and creature counts in the video game --/
structure GameState where
  goblin_points : ℕ := 3
  troll_points : ℕ := 5
  dragon_points : ℕ := 10
  combo_bonus : ℕ := 7
  total_goblins : ℕ := 14
  total_trolls : ℕ := 15
  total_dragons : ℕ := 4
  defeated_goblins : ℕ := 9  -- 70% of 14 rounded down
  defeated_trolls : ℕ := 10  -- 2/3 of 15
  defeated_dragons : ℕ := 1

/-- Calculates the total points earned in the game --/
def calculate_points (state : GameState) : ℕ :=
  state.goblin_points * state.defeated_goblins +
  state.troll_points * state.defeated_trolls +
  state.dragon_points * state.defeated_dragons +
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons))

/-- Theorem stating that the total points earned is 94 --/
theorem total_points_is_94 (state : GameState) : calculate_points state = 94 := by
  sorry

/-- Theorem stating that the bonus points earned is 7 --/
theorem bonus_points_is_7 (state : GameState) : 
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons)) = 7 := by
  sorry

end total_points_is_94_bonus_points_is_7_l2215_221524


namespace certain_number_proof_l2215_221557

theorem certain_number_proof (m : ℤ) (x : ℝ) (h1 : m = 6) (h2 : x^(2*m) = 2^(18 - m)) : x = 2 := by
  sorry

end certain_number_proof_l2215_221557


namespace escalator_time_l2215_221550

/-- Time taken for a person to cover the length of a moving escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 180) :
  escalator_length / (escalator_speed + person_speed) = 9 := by
  sorry

end escalator_time_l2215_221550


namespace polynomial_division_quotient_l2215_221584

theorem polynomial_division_quotient :
  let dividend : Polynomial ℤ := X^6 + 3*X^4 - 2*X^3 + X + 12
  let divisor : Polynomial ℤ := X - 2
  let quotient : Polynomial ℤ := X^5 + 2*X^4 + 6*X^3 + 10*X^2 + 18*X + 34
  dividend = divisor * quotient + 80 := by
  sorry

end polynomial_division_quotient_l2215_221584


namespace wire_length_l2215_221505

/-- The length of a wire stretched between two poles -/
theorem wire_length (d h₁ h₂ : ℝ) (hd : d = 20) (hh₁ : h₁ = 8) (hh₂ : h₂ = 9) :
  Real.sqrt ((d ^ 2) + ((h₂ - h₁) ^ 2)) = Real.sqrt 401 := by
  sorry

end wire_length_l2215_221505


namespace work_completion_theorem_l2215_221510

/-- The number of boys in the first group that satisfies the work conditions -/
def num_boys_in_first_group : ℕ := 16

theorem work_completion_theorem (x : ℕ) 
  (h1 : 5 * (12 * 2 + x) = 4 * (13 * 2 + 24)) : 
  x = num_boys_in_first_group := by
  sorry

#check work_completion_theorem

end work_completion_theorem_l2215_221510


namespace polynomial_degree_of_product_l2215_221519

/-- The degree of the polynomial resulting from multiplying 
    x^5, x + 1/x, and 1 + 2/x + 3/x^2 -/
theorem polynomial_degree_of_product : ℕ := by
  sorry

end polynomial_degree_of_product_l2215_221519


namespace students_not_participating_l2215_221580

/-- Given a class with the following properties:
  * There are 15 students in total
  * 7 students participate in mathematical modeling
  * 9 students participate in computer programming
  * 3 students participate in both activities
  This theorem proves that 2 students do not participate in either activity. -/
theorem students_not_participating (total : ℕ) (modeling : ℕ) (programming : ℕ) (both : ℕ) :
  total = 15 →
  modeling = 7 →
  programming = 9 →
  both = 3 →
  total - (modeling + programming - both) = 2 := by
  sorry

end students_not_participating_l2215_221580


namespace probability_of_selecting_letter_from_word_l2215_221515

/-- The number of characters in the extended alphabet -/
def alphabet_size : ℕ := 30

/-- The word from which we're checking letters -/
def word : String := "MATHEMATICS"

/-- The number of unique letters in the word -/
def unique_letters : ℕ := (word.toList.eraseDups).length

/-- The probability of selecting a letter from the word -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_selecting_letter_from_word :
  probability = 4 / 15 := by sorry

end probability_of_selecting_letter_from_word_l2215_221515


namespace exists_tastrophic_function_l2215_221529

/-- A function is k-tastrophic if its k-th iteration raises its input to the k-th power. -/
def IsTastrophic (k : ℕ) (f : ℕ → ℕ) : Prop :=
  k > 1 ∧ ∀ n : ℕ, n > 0 → (f^[k] n = n^k)

/-- For every integer k > 1, there exists a k-tastrophic function. -/
theorem exists_tastrophic_function :
  ∀ k : ℕ, k > 1 → ∃ f : ℕ → ℕ, IsTastrophic k f :=
sorry

end exists_tastrophic_function_l2215_221529


namespace locus_is_parabolic_arc_l2215_221594

-- Define the semicircle
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a semicircle
def is_tangent_to_semicircle (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ, 
    (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define tangency between a circle and a line (diameter)
def is_tangent_to_diameter (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ,
    p.2 = s.center.2 - s.radius ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ  -- y-coordinate of the directrix

-- Define a point being on a parabola
def on_parabola (p : ℝ × ℝ) (para : Parabola) : Prop :=
  (p.1 - para.focus.1)^2 + (p.2 - para.focus.2)^2 = (p.2 - para.directrix)^2

-- Main theorem
theorem locus_is_parabolic_arc (s : Semicircle) :
  ∀ c : Circle, 
    is_tangent_to_semicircle c s → 
    is_tangent_to_diameter c s → 
    ∃ para : Parabola, 
      para.focus = s.center ∧ 
      para.directrix = s.center.2 - 2 * s.radius ∧
      on_parabola c.center para ∧
      (c.center.1 - s.center.1)^2 + (c.center.2 - s.center.2)^2 < s.radius^2 :=
sorry

end locus_is_parabolic_arc_l2215_221594


namespace tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l2215_221590

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem tangent_points_on_curve :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

theorem unique_tangent_points :
  ∀ x : ℝ, f' x = 4 → (x = 1 ∨ x = -1) :=
sorry

end tangent_points_parallel_to_line_tangent_points_on_curve_unique_tangent_points_l2215_221590


namespace rectangle_equal_angles_l2215_221530

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ  -- width
  b : ℝ  -- height
  pos_a : 0 < a
  pos_b : 0 < b

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Theorem: The set of points P between parallel lines AB and CD of a rectangle
    such that ∠APB = ∠CPD is the line y = b/2 -/
theorem rectangle_equal_angles (rect : Rectangle) :
  ∀ P : Point,
    0 ≤ P.y ∧ P.y ≤ rect.b →
    (angle ⟨0, 0⟩ P ⟨rect.a, 0⟩ = angle ⟨rect.a, rect.b⟩ P ⟨0, rect.b⟩) ↔
    P.y = rect.b / 2 :=
  sorry

end rectangle_equal_angles_l2215_221530


namespace total_balloons_count_l2215_221591

/-- The number of yellow balloons Tom has -/
def tom_balloons : ℕ := 9

/-- The number of yellow balloons Sara has -/
def sara_balloons : ℕ := 8

/-- The total number of yellow balloons Tom and Sara have -/
def total_balloons : ℕ := tom_balloons + sara_balloons

theorem total_balloons_count : total_balloons = 17 := by
  sorry

end total_balloons_count_l2215_221591


namespace intersection_of_A_and_B_l2215_221575

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end intersection_of_A_and_B_l2215_221575


namespace ball_placement_theorem_l2215_221501

/-- The number of ways to place 4 different balls into 4 numbered boxes with exactly one empty box -/
def ball_placement_count : ℕ := 144

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 4

theorem ball_placement_theorem :
  (num_balls = 4) →
  (num_boxes = 4) →
  (ball_placement_count = 144) :=
by sorry

end ball_placement_theorem_l2215_221501


namespace sum_of_integers_l2215_221592

theorem sum_of_integers (x y z : ℕ+) (h : 27 * x.val + 28 * y.val + 29 * z.val = 363) :
  10 * (x.val + y.val + z.val) = 130 := by
  sorry

end sum_of_integers_l2215_221592


namespace candidate_fail_marks_l2215_221503

theorem candidate_fail_marks (max_marks : ℝ) (passing_percentage : ℝ) (candidate_score : ℝ) :
  max_marks = 153.84615384615384 →
  passing_percentage = 52 →
  candidate_score = 45 →
  ⌈passing_percentage / 100 * max_marks⌉ - candidate_score = 35 :=
by
  sorry

end candidate_fail_marks_l2215_221503


namespace sum_of_odd_numbers_l2215_221599

theorem sum_of_odd_numbers (n : ℕ) (sum_of_first_n_odds : ℕ → ℕ) 
  (h1 : ∀ k, sum_of_first_n_odds k = k^2)
  (h2 : sum_of_first_n_odds 100 = 10000)
  (h3 : sum_of_first_n_odds 50 = 2500) :
  sum_of_first_n_odds 100 - sum_of_first_n_odds 50 = 7500 := by
  sorry

#check sum_of_odd_numbers

end sum_of_odd_numbers_l2215_221599


namespace set_difference_proof_l2215_221534

def A : Set Int := {-1, 1, 3, 5, 7, 9}
def B : Set Int := {-1, 5, 7}

theorem set_difference_proof : A \ B = {1, 3, 9} := by sorry

end set_difference_proof_l2215_221534


namespace tangent_line_parallel_point_l2215_221572

/-- The function f(x) = x^4 - x --/
def f (x : ℝ) : ℝ := x^4 - x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_point (P : ℝ × ℝ) :
  P.1 = 1 ∧ P.2 = 0 ↔
    f P.1 = P.2 ∧ f' P.1 = 3 :=
sorry

end tangent_line_parallel_point_l2215_221572


namespace rooms_already_painted_l2215_221548

/-- Given a painting job with the following parameters:
  * total_rooms: The total number of rooms to be painted
  * hours_per_room: The number of hours it takes to paint one room
  * remaining_hours: The number of hours left to complete the job
  This theorem proves that the number of rooms already painted is equal to
  the total number of rooms minus the number of rooms that can be painted
  in the remaining time. -/
theorem rooms_already_painted
  (total_rooms : ℕ)
  (hours_per_room : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_rooms = 10)
  (h2 : hours_per_room = 8)
  (h3 : remaining_hours = 16) :
  total_rooms - (remaining_hours / hours_per_room) = 8 := by
  sorry

end rooms_already_painted_l2215_221548


namespace mary_marbles_l2215_221586

def dan_marbles : ℕ := 8
def mary_times_more : ℕ := 4

theorem mary_marbles : dan_marbles * mary_times_more = 32 := by
  sorry

end mary_marbles_l2215_221586


namespace unique_solution_quadratic_l2215_221552

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + k * x + 16 = 0) ↔ k = 8 * Real.sqrt 3 :=
by sorry

end unique_solution_quadratic_l2215_221552


namespace inequality_proof_l2215_221577

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a)) ≥ 27 / (a + b + c)^2 := by
  sorry

end inequality_proof_l2215_221577


namespace exist_three_numbers_not_exist_four_numbers_l2215_221570

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating the existence of three different natural numbers satisfying the condition -/
theorem exist_three_numbers :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (b * c + 10) :=
sorry

/-- Theorem stating the non-existence of four different natural numbers satisfying the condition -/
theorem not_exist_four_numbers :
  ¬∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (a * d + 10) ∧
    is_perfect_square (b * c + 10) ∧
    is_perfect_square (b * d + 10) ∧
    is_perfect_square (c * d + 10) :=
sorry

end exist_three_numbers_not_exist_four_numbers_l2215_221570


namespace specific_tetrahedron_volume_l2215_221538

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  PS : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is approximately 10.54 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 5.5,
    PR := 3.5,
    QR := 4,
    PS := 4.2,
    QS := 3.7,
    RS := 2.6
  }
  abs (tetrahedronVolume t - 10.54) < 0.01 := by
  sorry

end specific_tetrahedron_volume_l2215_221538


namespace marbles_lost_calculation_specific_marbles_lost_l2215_221522

/-- Given an initial number of marbles and the current number of marbles in a bag,
    calculate the number of lost marbles. -/
def lost_marbles (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that the number of lost marbles is equal to
    the difference between the initial and current number of marbles. -/
theorem marbles_lost_calculation (initial current : ℕ) (h : current ≤ initial) :
  lost_marbles initial current = initial - current :=
by
  sorry

/-- The specific problem instance -/
def initial_marbles : ℕ := 8
def current_marbles : ℕ := 6

/-- Theorem for the specific problem instance -/
theorem specific_marbles_lost :
  lost_marbles initial_marbles current_marbles = 2 :=
by
  sorry

end marbles_lost_calculation_specific_marbles_lost_l2215_221522


namespace hyperbola_asymptote_ratio_l2215_221511

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a ≠ b, 
    if the angle between its asymptotes is 90°, then a/b = 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧ m₁ = a/b ∧ m₂ = -a/b) →
  a / b = 1 := by
  sorry

end hyperbola_asymptote_ratio_l2215_221511


namespace min_value_reciprocal_sum_l2215_221512

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  ∀ z : ℝ, (1 / x + 1 / y) ≥ z → z ≤ 3 / 2 + Real.sqrt 2 :=
by sorry

end min_value_reciprocal_sum_l2215_221512


namespace modulo_nine_equivalence_l2215_221547

theorem modulo_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2022 ≡ n [ZMOD 9] ∧ n = 3 := by
  sorry

end modulo_nine_equivalence_l2215_221547
