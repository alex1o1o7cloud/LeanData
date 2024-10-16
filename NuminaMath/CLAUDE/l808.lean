import Mathlib

namespace NUMINAMATH_CALUDE_parabola_line_intersection_l808_80865

/-- Given a parabola and a line passing through its focus, 
    prove the value of p when the triangle area is 4 -/
theorem parabola_line_intersection (p : ℝ) : 
  let parabola := fun (x y : ℝ) => x^2 = 2*p*y
  let focus := (0, p/2)
  let line := fun (x y : ℝ) => y = Real.sqrt 3 * x + p/2
  let origin := (0, 0)
  let triangle_area (A B : ℝ × ℝ) := 
    abs ((A.1 - origin.1) * (B.2 - origin.2) - (B.1 - origin.1) * (A.2 - origin.2)) / 2
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    triangle_area A B = 4 →
    p = 2 * Real.sqrt 2 ∨ p = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l808_80865


namespace NUMINAMATH_CALUDE_remainder_problem_l808_80867

theorem remainder_problem : (7 * 10^24 + 2^24) % 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l808_80867


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l808_80876

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let decrease := original_price - sale_price
  let percentage_decrease := (decrease / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l808_80876


namespace NUMINAMATH_CALUDE_middle_box_label_l808_80893

/-- Represents the possible labels on a box. -/
inductive BoxLabel
  | NoPrize : BoxLabel
  | PrizeInNeighbor : BoxLabel

/-- Represents a row of boxes. -/
structure BoxRow :=
  (size : Nat)
  (labels : Fin size → BoxLabel)
  (prizeLocation : Fin size)

/-- The condition that exactly one statement is true. -/
def exactlyOneTrue (row : BoxRow) : Prop :=
  ∃! i : Fin row.size, 
    (row.labels i = BoxLabel.NoPrize ∧ i ≠ row.prizeLocation) ∨
    (row.labels i = BoxLabel.PrizeInNeighbor ∧ 
      (i.val + 1 = row.prizeLocation.val ∨ i.val = row.prizeLocation.val + 1))

/-- The theorem stating the label on the middle box. -/
theorem middle_box_label (row : BoxRow) 
  (h_size : row.size = 23)
  (h_one_true : exactlyOneTrue row) :
  row.labels ⟨11, by {rw [h_size]; simp}⟩ = BoxLabel.PrizeInNeighbor :=
sorry

end NUMINAMATH_CALUDE_middle_box_label_l808_80893


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l808_80801

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 7) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l808_80801


namespace NUMINAMATH_CALUDE_number_puzzle_l808_80806

theorem number_puzzle : ∃ x : ℝ, (x / 7 - x / 11 = 100) ∧ (x = 1925) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l808_80806


namespace NUMINAMATH_CALUDE_vacation_pictures_l808_80855

def remaining_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) : ℕ :=
  zoo_pics + museum_pics - deleted_pics

theorem vacation_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ) 
  (h1 : zoo_pics = 50)
  (h2 : museum_pics = 8)
  (h3 : deleted_pics = 38) :
  remaining_pictures zoo_pics museum_pics deleted_pics = 20 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l808_80855


namespace NUMINAMATH_CALUDE_nested_sqrt_simplification_l808_80827

theorem nested_sqrt_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 9) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_simplification_l808_80827


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l808_80883

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) ≥ -498998 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (x^2 + 1/y^2 + 1) * (x^2 + 1/y^2 - 1000) + (y^2 + 1/x^2 + 1) * (y^2 + 1/x^2 - 1000) = -498998 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l808_80883


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l808_80814

theorem triangle_inequality_range (x : ℝ) : 
  (3 : ℝ) > 0 ∧ (1 + 2*x) > 0 ∧ 8 > 0 ∧
  3 + (1 + 2*x) > 8 ∧
  3 + 8 > (1 + 2*x) ∧
  (1 + 2*x) + 8 > 3 ↔
  2 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l808_80814


namespace NUMINAMATH_CALUDE_car_speed_calculation_car_speed_is_21_l808_80830

/-- Calculates the speed of a car given the walking speed of a person and the number of steps taken --/
theorem car_speed_calculation (walking_speed : ℝ) (steps_while_car_visible : ℕ) (steps_after_car_disappeared : ℕ) : ℝ :=
  let total_steps := steps_while_car_visible + steps_after_car_disappeared
  let speed_ratio := total_steps / steps_while_car_visible
  speed_ratio * walking_speed

/-- Proves that the car's speed is 21 km/h given the specific conditions --/
theorem car_speed_is_21 : 
  car_speed_calculation 3.5 27 135 = 21 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_car_speed_is_21_l808_80830


namespace NUMINAMATH_CALUDE_butterfat_percentage_of_added_milk_l808_80802

/-- Prove that the percentage of butterfat in the added milk is 10% -/
theorem butterfat_percentage_of_added_milk
  (initial_volume : ℝ)
  (initial_butterfat_percentage : ℝ)
  (added_volume : ℝ)
  (final_butterfat_percentage : ℝ)
  (h_initial_volume : initial_volume = 8)
  (h_initial_butterfat : initial_butterfat_percentage = 35)
  (h_added_volume : added_volume = 12)
  (h_final_butterfat : final_butterfat_percentage = 20)
  (h_total_volume : initial_volume + added_volume = 20) :
  let added_butterfat_percentage :=
    (final_butterfat_percentage * (initial_volume + added_volume) -
     initial_butterfat_percentage * initial_volume) / added_volume
  added_butterfat_percentage = 10 :=
by sorry

end NUMINAMATH_CALUDE_butterfat_percentage_of_added_milk_l808_80802


namespace NUMINAMATH_CALUDE_cubic_equation_game_strategy_l808_80861

theorem cubic_equation_game_strategy (second_player_choice : ℤ) : 
  ∃ (a b c : ℤ), ∃ (x y z : ℤ),
    (x^3 + a*x^2 + b*x + c = 0) ∧
    (y^3 + a*y^2 + b*y + c = 0) ∧
    (z^3 + a*z^2 + b*z + c = 0) ∧
    (a = second_player_choice ∨ b = second_player_choice) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_game_strategy_l808_80861


namespace NUMINAMATH_CALUDE_candy_problem_l808_80898

def candy_remaining (initial : ℕ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial
  | 1 => initial / 2
  | 2 => initial / 2 * (1 / 3)
  | 3 => initial / 2 * (1 / 3) * (1 / 4)
  | 4 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5)
  | 5 => initial / 2 * (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6)
  | _ => 0

theorem candy_problem (initial : ℕ) :
  candy_remaining initial 5 = 1 ↔ initial = 720 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l808_80898


namespace NUMINAMATH_CALUDE_quadratic_nature_l808_80816

/-- Given a quadratic function f(x) = ax^2 + bx + b^2 / (3a), prove that:
    1. If a > 0, the graph of y = f(x) has a minimum
    2. If a < 0, the graph of y = f(x) has a maximum -/
theorem quadratic_nature (a b : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + b^2 / (3 * a)
  (a > 0 → ∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (a < 0 → ∃ x₀, ∀ x, f x ≤ f x₀) :=
sorry

end NUMINAMATH_CALUDE_quadratic_nature_l808_80816


namespace NUMINAMATH_CALUDE_eighth_term_value_l808_80844

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, n ≥ 1 → a (n + 2) = a n + a (n + 1))

theorem eighth_term_value 
  (a : ℕ → ℕ) 
  (h_seq : RecurrenceSequence a) 
  (h_seventh : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_value_l808_80844


namespace NUMINAMATH_CALUDE_expression_evaluation_l808_80889

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7) = -1393 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l808_80889


namespace NUMINAMATH_CALUDE_circle_center_center_coordinates_l808_80877

theorem circle_center (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ↔ 
  ((x - 1)^2 + (y + 2)^2 = 0) :=
sorry

theorem center_coordinates : 
  ∃ (x y : ℝ), (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) ∧ 
  (x = 1 ∧ y = -2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_center_coordinates_l808_80877


namespace NUMINAMATH_CALUDE_percentage_difference_l808_80833

theorem percentage_difference (x y : ℝ) (h : y = x * (1 + 1/3)) :
  x = y * (1 - 1/4) :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l808_80833


namespace NUMINAMATH_CALUDE_divisibility_by_37_l808_80899

def N (x y : ℕ) : ℕ := 300070003 + 1000000 * x + 100 * y

theorem divisibility_by_37 :
  ∀ x y : ℕ, x ≤ 9 ∧ y ≤ 9 →
  (37 ∣ N x y ↔ (x = 8 ∧ y = 1) ∨ (x = 4 ∧ y = 4) ∨ (x = 0 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l808_80899


namespace NUMINAMATH_CALUDE_max_triangle_area_l808_80809

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def isosceles_trapezoid (m n f2 f1 : Point) : Prop :=
  ∃ (height area : ℝ), height = Real.sqrt 3 ∧ area = 3 * Real.sqrt 3

def line_through_point (p : Point) : Set Point :=
  {q : Point | ∃ (k : ℝ), q.x = k * q.y + p.x}

def intersect_ellipse_line (e : Ellipse) (l : Set Point) : Set Point :=
  {p : Point | p ∈ l ∧ e.equation p}

def triangle_area (t : Triangle) : ℝ :=
  sorry

theorem max_triangle_area (e : Ellipse) (m n f2 f1 : Point) :
  isosceles_trapezoid m n f2 f1 →
  m = Point.mk (-e.a) e.b →
  n = Point.mk e.a e.b →
  (∀ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∀ (a b : Point), a ∈ intersection → b ∈ intersection →
      triangle_area (Triangle.mk f2 a b) ≤ 3) ∧
  (∃ (l : Set Point), f1 ∈ l →
    let intersection := intersect_ellipse_line e l
    ∃ (a b : Point), a ∈ intersection ∧ b ∈ intersection ∧
      triangle_area (Triangle.mk f2 a b) = 3) :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l808_80809


namespace NUMINAMATH_CALUDE_other_communities_students_l808_80879

theorem other_communities_students (total : ℕ) (muslim_percent hindu_percent sikh_percent christian_percent buddhist_percent : ℚ) :
  total = 1500 →
  muslim_percent = 38/100 →
  hindu_percent = 26/100 →
  sikh_percent = 12/100 →
  christian_percent = 6/100 →
  buddhist_percent = 4/100 →
  ↑(total * (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent))) = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_other_communities_students_l808_80879


namespace NUMINAMATH_CALUDE_inequality_properties_l808_80856

theorem inequality_properties (a b c d : ℝ) :
  (∀ (a b c : ℝ), c ≠ 0 → a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b c d : ℝ), a > b → c > d → a + c > b + d) ∧
  (∃ (a b c d : ℝ), a > b ∧ c > d ∧ ¬(a * c > b * d)) ∧
  (∃ (a b : ℝ), a > b ∧ ¬(1 / a > 1 / b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l808_80856


namespace NUMINAMATH_CALUDE_birthday_theorem_l808_80895

def birthday_money (age : ℕ) : ℕ := age * 5

theorem birthday_theorem : 
  ∀ (age : ℕ), age = 3 + 3 * 3 → birthday_money age = 60 := by
  sorry

end NUMINAMATH_CALUDE_birthday_theorem_l808_80895


namespace NUMINAMATH_CALUDE_first_month_sale_is_6435_l808_80894

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  average_target : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the sale in the first month given the sales data -/
def first_month_sale (s : GrocerySales) : ℕ :=
  6 * s.average_target - (s.month2 + s.month3 + s.month4 + s.month5 + s.month6)

/-- Theorem stating that the first month's sale is 6435 given the specific sales data -/
theorem first_month_sale_is_6435 :
  let s : GrocerySales := {
    average_target := 6500,
    month2 := 6927,
    month3 := 6855,
    month4 := 7230,
    month5 := 6562,
    month6 := 4991
  }
  first_month_sale s = 6435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_6435_l808_80894


namespace NUMINAMATH_CALUDE_function_composition_l808_80854

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ y, f y = y^2 + 2*y) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l808_80854


namespace NUMINAMATH_CALUDE_sum_first_100_triangular_numbers_l808_80885

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular_numbers (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => triangular_number (i + 1))

/-- Theorem: The sum of the first 100 triangular numbers is 171700 -/
theorem sum_first_100_triangular_numbers : 
  sum_triangular_numbers 100 = 171700 := by
  sorry

#eval sum_triangular_numbers 100

end NUMINAMATH_CALUDE_sum_first_100_triangular_numbers_l808_80885


namespace NUMINAMATH_CALUDE_six_students_like_no_option_l808_80824

/-- Represents the food preferences in a class --/
structure FoodPreferences where
  total_students : ℕ
  french_fries : ℕ
  burgers : ℕ
  pizza : ℕ
  tacos : ℕ
  fries_burgers : ℕ
  fries_pizza : ℕ
  fries_tacos : ℕ
  burgers_pizza : ℕ
  burgers_tacos : ℕ
  pizza_tacos : ℕ
  fries_burgers_pizza : ℕ
  fries_burgers_tacos : ℕ
  fries_pizza_tacos : ℕ
  burgers_pizza_tacos : ℕ
  all_four : ℕ

/-- Calculates the number of students who don't like any food option --/
def studentsLikingNoOption (prefs : FoodPreferences) : ℕ :=
  prefs.total_students -
  (prefs.french_fries + prefs.burgers + prefs.pizza + prefs.tacos -
   prefs.fries_burgers - prefs.fries_pizza - prefs.fries_tacos -
   prefs.burgers_pizza - prefs.burgers_tacos - prefs.pizza_tacos +
   prefs.fries_burgers_pizza + prefs.fries_burgers_tacos +
   prefs.fries_pizza_tacos + prefs.burgers_pizza_tacos -
   prefs.all_four)

/-- Theorem: Given the food preferences, 6 students don't like any option --/
theorem six_students_like_no_option (prefs : FoodPreferences)
  (h1 : prefs.total_students = 35)
  (h2 : prefs.french_fries = 20)
  (h3 : prefs.burgers = 15)
  (h4 : prefs.pizza = 18)
  (h5 : prefs.tacos = 12)
  (h6 : prefs.fries_burgers = 10)
  (h7 : prefs.fries_pizza = 8)
  (h8 : prefs.fries_tacos = 6)
  (h9 : prefs.burgers_pizza = 7)
  (h10 : prefs.burgers_tacos = 5)
  (h11 : prefs.pizza_tacos = 9)
  (h12 : prefs.fries_burgers_pizza = 4)
  (h13 : prefs.fries_burgers_tacos = 3)
  (h14 : prefs.fries_pizza_tacos = 2)
  (h15 : prefs.burgers_pizza_tacos = 1)
  (h16 : prefs.all_four = 1) :
  studentsLikingNoOption prefs = 6 := by
  sorry


end NUMINAMATH_CALUDE_six_students_like_no_option_l808_80824


namespace NUMINAMATH_CALUDE_binomial_square_simplification_l808_80864

theorem binomial_square_simplification (m n p : ℝ) :
  ¬(∃ a b, (-m - n) * (m + n) = a^2 - b^2) ∧
  (∃ a b, (-m - n) * (-m + n) = a^2 - b^2) ∧
  (∃ a b, (m * n + p) * (m * n - p) = a^2 - b^2) ∧
  (∃ a b, (0.3 * m - n) * (-n - 0.3 * m) = a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_binomial_square_simplification_l808_80864


namespace NUMINAMATH_CALUDE_circle_center_correct_l808_80886

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x + 9 * y^2 + 36 * y + 44 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (1, -2)

/-- Theorem stating that CircleCenter is the center of the circle defined by CircleEquation -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l808_80886


namespace NUMINAMATH_CALUDE_complex_point_coordinates_l808_80890

theorem complex_point_coordinates (Z : ℂ) : Z = Complex.I * (1 + Complex.I) → Z.re = -1 ∧ Z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_coordinates_l808_80890


namespace NUMINAMATH_CALUDE_group_average_age_problem_l808_80880

theorem group_average_age_problem (n : ℕ) : 
  (n * 14 + 32 = 16 * (n + 1)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_average_age_problem_l808_80880


namespace NUMINAMATH_CALUDE_unique_solution_for_squared_geometric_sum_l808_80862

theorem unique_solution_for_squared_geometric_sum : 
  ∃! (n m : ℕ), n > 0 ∧ m > 0 ∧ n^2 = (m^5 - 1) / (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_squared_geometric_sum_l808_80862


namespace NUMINAMATH_CALUDE_simplify_expression_l808_80884

theorem simplify_expression : (81 * (10 ^ 12)) / (9 * (10 ^ 4)) = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l808_80884


namespace NUMINAMATH_CALUDE_count_valid_numbers_l808_80835

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 0

def divisible_by_eleven_and_twenty (n : ℕ) : Prop :=
  n % 11 = 0 ∧ n % 20 = 0

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_number n ∧ divisible_by_eleven_and_twenty n) ∧
    (∀ n, is_valid_number n ∧ divisible_by_eleven_and_twenty n → n ∈ S) ∧
    Finset.card S = 4 :=
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l808_80835


namespace NUMINAMATH_CALUDE_shell_collection_ratio_l808_80858

theorem shell_collection_ratio :
  ∀ (laurie_shells ben_shells alan_shells : ℕ),
    laurie_shells = 36 →
    ben_shells = laurie_shells / 3 →
    alan_shells = 48 →
    alan_shells / ben_shells = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shell_collection_ratio_l808_80858


namespace NUMINAMATH_CALUDE_sum_reciprocals_of_roots_l808_80803

theorem sum_reciprocals_of_roots (x : ℝ) : 
  x^2 - 7*x + 2 = 0 → 
  ∃ a b : ℝ, (x = a ∨ x = b) ∧ (1/a + 1/b = 7/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_of_roots_l808_80803


namespace NUMINAMATH_CALUDE_inequality_proof_l808_80815

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) ≥ (a + b) / Real.sqrt (a * b + 1) ∧
  (a / Real.sqrt (b^2 + 1) + b / Real.sqrt (a^2 + 1) = (a + b) / Real.sqrt (a * b + 1) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l808_80815


namespace NUMINAMATH_CALUDE_exists_43_move_strategy_l808_80860

/-- The number of boxes and chosen numbers -/
def n : ℕ := 2017

/-- A strategy for distributing stones -/
structure Strategy where
  numbers : Fin n → ℕ

/-- The state of the game after some moves -/
def GameState := Fin n → ℕ

/-- Apply a strategy for one move -/
def applyStrategy (s : Strategy) (state : GameState) : GameState :=
  fun i => state i + s.numbers i

/-- Apply a strategy for k moves -/
def applyStrategyKTimes (s : Strategy) (k : ℕ) : GameState :=
  fun i => k * (s.numbers i)

/-- Check if all boxes have the same number of stones -/
def allEqual (state : GameState) : Prop :=
  ∀ i j, state i = state j

/-- The main theorem -/
theorem exists_43_move_strategy :
  ∃ (s : Strategy),
    (allEqual (applyStrategyKTimes s 43)) ∧
    (∀ k, 0 < k → k < 43 → ¬(allEqual (applyStrategyKTimes s k))) := by
  sorry

end NUMINAMATH_CALUDE_exists_43_move_strategy_l808_80860


namespace NUMINAMATH_CALUDE_students_not_visiting_any_exhibit_l808_80839

def total_students : ℕ := 52
def botanical_visitors : ℕ := 12
def animal_visitors : ℕ := 26
def technology_visitors : ℕ := 23
def botanical_and_animal : ℕ := 5
def botanical_and_technology : ℕ := 2
def animal_and_technology : ℕ := 4
def all_three : ℕ := 1

theorem students_not_visiting_any_exhibit : 
  total_students - (botanical_visitors + animal_visitors + technology_visitors
                    - botanical_and_animal - botanical_and_technology - animal_and_technology
                    + all_three) = 1 := by sorry

end NUMINAMATH_CALUDE_students_not_visiting_any_exhibit_l808_80839


namespace NUMINAMATH_CALUDE_cookie_ratio_proof_l808_80843

def cookie_problem (initial cookies_to_friend cookies_eaten cookies_left : ℕ) : Prop :=
  let cookies_after_friend := initial - cookies_to_friend
  let cookies_to_family := cookies_after_friend - cookies_eaten - cookies_left
  (2 * cookies_to_family = cookies_after_friend)

theorem cookie_ratio_proof :
  cookie_problem 19 5 2 5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_proof_l808_80843


namespace NUMINAMATH_CALUDE_impossible_all_defective_l808_80829

theorem impossible_all_defective (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 25)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  Nat.choose defective selected / Nat.choose total selected = 0 :=
by sorry

end NUMINAMATH_CALUDE_impossible_all_defective_l808_80829


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l808_80800

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 24 →
  (1/2) * c * d = 600 →
  a^2 + b^2 = 100 →
  (c / a)^2 = 25 →
  (d / b)^2 = 25 →
  c + d = 70 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l808_80800


namespace NUMINAMATH_CALUDE_same_solution_implies_m_value_l808_80838

theorem same_solution_implies_m_value :
  ∀ (m : ℝ) (x : ℝ),
    (-5 * x - 6 = 3 * x + 10) ∧
    (-2 * m - 3 * x = 10) →
    m = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_value_l808_80838


namespace NUMINAMATH_CALUDE_f_of_5_equals_105_l808_80823

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x

-- State the theorem
theorem f_of_5_equals_105 : f 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_105_l808_80823


namespace NUMINAMATH_CALUDE_functional_equation_implies_linear_l808_80834

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

/-- The main theorem stating that any function satisfying the functional equation is linear -/
theorem functional_equation_implies_linear (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linear_l808_80834


namespace NUMINAMATH_CALUDE_polynomial_factorization_l808_80813

theorem polynomial_factorization (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l808_80813


namespace NUMINAMATH_CALUDE_function_inequality_l808_80818

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l808_80818


namespace NUMINAMATH_CALUDE_max_students_on_playground_l808_80871

def total_pencils : ℕ := 170
def total_notebooks : ℕ := 268
def total_erasers : ℕ := 120
def leftover_pencils : ℕ := 8
def shortage_notebooks : ℕ := 2
def leftover_erasers : ℕ := 12

theorem max_students_on_playground :
  let distributed_pencils := total_pencils - leftover_pencils
  let distributed_notebooks := total_notebooks + shortage_notebooks
  let distributed_erasers := total_erasers - leftover_erasers
  let max_students := Nat.gcd distributed_pencils (Nat.gcd distributed_notebooks distributed_erasers)
  max_students = 54 ∧
  (∃ (p n e : ℕ), 
    distributed_pencils = max_students * p ∧
    distributed_notebooks = max_students * n ∧
    distributed_erasers = max_students * e) ∧
  (∀ s : ℕ, s > max_students →
    ¬(∃ (p n e : ℕ),
      distributed_pencils = s * p ∧
      distributed_notebooks = s * n ∧
      distributed_erasers = s * e)) :=
by sorry

end NUMINAMATH_CALUDE_max_students_on_playground_l808_80871


namespace NUMINAMATH_CALUDE_barbells_bought_l808_80837

theorem barbells_bought (amount_given : ℕ) (change_received : ℕ) (cost_per_barbell : ℕ) : 
  amount_given = 850 → change_received = 40 → cost_per_barbell = 270 → 
  (amount_given - change_received) / cost_per_barbell = 3 :=
by sorry

end NUMINAMATH_CALUDE_barbells_bought_l808_80837


namespace NUMINAMATH_CALUDE_rectangle_division_l808_80888

/-- Given a rectangle with length 3y and width y, divided into a smaller rectangle
    of length x and width y-x surrounded by four congruent right-angled triangles,
    this theorem proves the perimeter of one triangle and the area of the smaller rectangle. -/
theorem rectangle_division (x y : ℝ) : 
  let triangle_perimeter := 3 * y + Real.sqrt (2 * x^2 - 6 * y * x + 9 * y^2)
  let smaller_rectangle_area := x * y - x^2
  ∀ (triangle_side_a triangle_side_b : ℝ),
    triangle_side_a = x ∧ 
    triangle_side_b = 3 * y - x →
    triangle_perimeter = triangle_side_a + triangle_side_b + 
      Real.sqrt (triangle_side_a^2 + triangle_side_b^2) ∧
    smaller_rectangle_area = x * (y - x) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l808_80888


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l808_80887

theorem max_value_of_fraction (x : ℝ) (h : x ≠ 0) :
  x^2 / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) ≤ 1/8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l808_80887


namespace NUMINAMATH_CALUDE_water_reservoir_ratio_l808_80852

/-- The ratio of the amount of water in the reservoir at the end of the month to the normal level -/
theorem water_reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_level : ℝ),
  end_month_level = 30 →
  end_month_level = 0.75 * total_capacity →
  normal_level = total_capacity - 20 →
  end_month_level / normal_level = 1.5 := by
sorry

end NUMINAMATH_CALUDE_water_reservoir_ratio_l808_80852


namespace NUMINAMATH_CALUDE_A_equals_B_l808_80810

/-- Set A defined as {a | a = 12m + 8n + 4l, m, n, l ∈ ℤ} -/
def A : Set ℤ := {a | ∃ m n l : ℤ, a = 12*m + 8*n + 4*l}

/-- Set B defined as {b | b = 20p + 16q + 12r, p, q, r ∈ ℤ} -/
def B : Set ℤ := {b | ∃ p q r : ℤ, b = 20*p + 16*q + 12*r}

/-- Theorem stating that A = B -/
theorem A_equals_B : A = B := by
  sorry


end NUMINAMATH_CALUDE_A_equals_B_l808_80810


namespace NUMINAMATH_CALUDE_company_stores_l808_80857

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) :
  (total_uniforms / uniforms_per_store : ℕ) = 30 :=
by sorry

end NUMINAMATH_CALUDE_company_stores_l808_80857


namespace NUMINAMATH_CALUDE_shopkeeper_gain_l808_80850

/-- Calculates the percentage gain for a shopkeeper given markup and discount percentages -/
theorem shopkeeper_gain (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 35 →
  discount_percent = 20 →
  let marked_price := cost_price * (1 + markup_percent / 100)
  let selling_price := marked_price * (1 - discount_percent / 100)
  let gain := selling_price - cost_price
  gain / cost_price * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_l808_80850


namespace NUMINAMATH_CALUDE_tile_difference_8_9_and_9_10_l808_80875

/-- Represents the number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2

/-- The difference in tiles between two consecutive squares -/
def tile_difference (n : ℕ) : ℕ := tiles (n + 1) - tiles n

theorem tile_difference_8_9_and_9_10 :
  (tile_difference 8 = 17) ∧ (tile_difference 9 = 19) := by
  sorry

end NUMINAMATH_CALUDE_tile_difference_8_9_and_9_10_l808_80875


namespace NUMINAMATH_CALUDE_line_equation_equivalence_slope_intercept_parameters_l808_80805

/-- Given a line equation in vector form, prove its slope-intercept form and parameters -/
theorem line_equation_equivalence :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 11 :=
by sorry

/-- Prove the slope and y-intercept of the line -/
theorem slope_intercept_parameters :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = m * x + b) ∧ m = 2 ∧ b = -11 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_slope_intercept_parameters_l808_80805


namespace NUMINAMATH_CALUDE_chess_tournament_games_l808_80853

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) :
  n = 12 →
  total_games = 132 →
  ∃ (games_per_pair : ℕ),
    total_games = games_per_pair * (n * (n - 1) / 2) ∧
    games_per_pair = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l808_80853


namespace NUMINAMATH_CALUDE_orange_theorem_l808_80804

def orange_problem (betty_oranges bill_oranges frank_multiplier seeds_per_orange oranges_per_tree : ℕ) : ℕ :=
  let total_betty_bill := betty_oranges + bill_oranges
  let frank_oranges := frank_multiplier * total_betty_bill
  let total_seeds := frank_oranges * seeds_per_orange
  let total_oranges := total_seeds * oranges_per_tree
  total_oranges

theorem orange_theorem : 
  orange_problem 15 12 3 2 5 = 810 := by
  sorry

end NUMINAMATH_CALUDE_orange_theorem_l808_80804


namespace NUMINAMATH_CALUDE_perfect_square_product_divisible_by_12_l808_80868

theorem perfect_square_product_divisible_by_12 (n : ℤ) : 
  12 ∣ (n^2 * (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_product_divisible_by_12_l808_80868


namespace NUMINAMATH_CALUDE_investment_dividend_theorem_l808_80863

/-- Calculates the dividend received from an investment in shares with premium and dividend rate --/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem: Given the specified investment conditions, the dividend received is 600 --/
theorem investment_dividend_theorem (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 1/5)
  (h4 : dividend_rate = 1/20) :
  calculate_dividend investment share_value premium_rate dividend_rate = 600 := by
  sorry

#eval calculate_dividend 14400 100 (1/5) (1/20)

end NUMINAMATH_CALUDE_investment_dividend_theorem_l808_80863


namespace NUMINAMATH_CALUDE_unique_sequence_l808_80847

def sequence_condition (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) ∧
  (∀ n : ℕ, n > 0 → a (2 * n) = a n + n) ∧
  (∀ n : ℕ, n > 0 → Prime (a n) → Prime n)

theorem unique_sequence :
  ∀ a : ℕ → ℕ, sequence_condition a → ∀ n : ℕ, n > 0 → a n = n :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_l808_80847


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l808_80832

theorem subtraction_of_fractions : (1 : ℚ) / 2 - (1 : ℚ) / 8 = (3 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l808_80832


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l808_80819

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Given S(n) = 1365, S(n+1) = 1360 -/
theorem sum_of_digits_theorem (n : ℕ) (h : S n = 1365) : S (n + 1) = 1360 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l808_80819


namespace NUMINAMATH_CALUDE_dot_product_AD_AC_l808_80836

-- Define the vectors
def AB : ℝ × ℝ := (1, -2)
def AD : ℝ × ℝ := (2, 1)

-- Define AC as the sum of AB and AD
def AC : ℝ × ℝ := (AB.1 + AD.1, AB.2 + AD.2)

-- Theorem: The dot product of AD and AC is 5
theorem dot_product_AD_AC : AD.1 * AC.1 + AD.2 * AC.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_AD_AC_l808_80836


namespace NUMINAMATH_CALUDE_timothy_chickens_l808_80849

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cows_cost : ℕ := 20 * 1000
def solar_panel_cost : ℕ := 6 * 100 + 6000
def chicken_price : ℕ := 5
def total_cost : ℕ := 147700

theorem timothy_chickens :
  ∃ (num_chickens : ℕ),
    land_cost + house_cost + cows_cost + solar_panel_cost + num_chickens * chicken_price = total_cost ∧
    num_chickens = 100 :=
by sorry

end NUMINAMATH_CALUDE_timothy_chickens_l808_80849


namespace NUMINAMATH_CALUDE_triangle_area_in_square_grid_l808_80821

theorem triangle_area_in_square_grid :
  let square_side : ℝ := 4
  let square_area : ℝ := square_side ^ 2
  let triangle1_area : ℝ := 4
  let triangle2_area : ℝ := 2
  let triangle3_area : ℝ := 3
  let total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area
  let triangle_abc_area : ℝ := square_area - total_triangles_area
  triangle_abc_area = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_area_in_square_grid_l808_80821


namespace NUMINAMATH_CALUDE_negation_of_proposition_l808_80826

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → x^2 - 5*x + 6 ≠ 0)) ↔
  (∀ x : ℝ, (x = 3 ∨ x = 2) → x^2 - 5*x + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l808_80826


namespace NUMINAMATH_CALUDE_missing_number_is_33_l808_80882

def known_numbers : List ℝ := [1, 22, 24, 25, 26, 27, 2]

theorem missing_number_is_33 :
  ∃ x : ℝ, (known_numbers.sum + x) / 8 = 20 ∧ x = 33 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_33_l808_80882


namespace NUMINAMATH_CALUDE_factor_expression_l808_80811

theorem factor_expression (a : ℝ) : 49 * a^3 + 245 * a^2 + 588 * a = 49 * a * (a^2 + 5 * a + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l808_80811


namespace NUMINAMATH_CALUDE_triangle_inequality_l808_80841

theorem triangle_inequality (α β γ a b c : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : α + β + γ = π) : 
  a * (1/β + 1/γ) + b * (1/γ + 1/α) + c * (1/α + 1/β) ≥ 2 * (a/α + b/β + c/γ) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l808_80841


namespace NUMINAMATH_CALUDE_pairs_sold_proof_l808_80870

def total_amount : ℝ := 588
def average_price : ℝ := 9.8

theorem pairs_sold_proof :
  total_amount / average_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_pairs_sold_proof_l808_80870


namespace NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l808_80851

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a rectangular hyperbola -/
structure RectangularHyperbola where
  k : ℝ

/-- Represents the locus of points -/
def locus (h : RectangularHyperbola) : Set PolarPoint :=
  {p : PolarPoint | p.r^2 = 2 * h.k^2 * Real.sin (2 * p.θ)}

/-- The main theorem stating that the locus of the foot of the perpendicular
    from the center of a rectangular hyperbola to a tangent is given by
    the polar equation r^2 = 2k^2 sin 2θ -/
theorem locus_of_perpendicular_foot (h : RectangularHyperbola) :
  ∀ p : PolarPoint, p ∈ locus h ↔
    ∃ (t : ℝ), -- t represents the parameter of a point on the hyperbola
      let tangent_point := (t, h.k^2 / t)
      let tangent_slope := -h.k^2 / t^2
      let perpendicular_slope := -1 / tangent_slope
      p.r * (Real.cos p.θ) = 2 * h.k^4 / (t * (t^4 + h.k^4)) ∧
      p.r * (Real.sin p.θ) = 2 * t * h.k^2 / (t^4 + h.k^4) :=
by
  sorry

end NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l808_80851


namespace NUMINAMATH_CALUDE_unique_solution_l808_80897

-- Define the color type
inductive Color
| Red
| Blue

-- Define the clothing type
structure Clothing where
  tshirt : Color
  shorts : Color

-- Define the children
structure Children where
  alyna : Clothing
  bohdan : Clothing
  vika : Clothing
  grysha : Clothing

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  c.alyna.tshirt = Color.Red ∧
  c.bohdan.tshirt = Color.Red ∧
  c.alyna.shorts ≠ c.bohdan.shorts ∧
  c.vika.tshirt ≠ c.grysha.tshirt ∧
  c.vika.shorts = Color.Blue ∧
  c.grysha.shorts = Color.Blue ∧
  c.alyna.tshirt ≠ c.vika.tshirt ∧
  c.alyna.shorts ≠ c.vika.shorts

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red }
  , bohdan := { tshirt := Color.Red, shorts := Color.Blue }
  , vika := { tshirt := Color.Blue, shorts := Color.Blue }
  , grysha := { tshirt := Color.Red, shorts := Color.Blue }
  }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l808_80897


namespace NUMINAMATH_CALUDE_weight_of_seven_moles_l808_80891

/-- Given a compound with a molecular weight of 1176, prove that the weight of 7 moles of this compound is 8232. -/
theorem weight_of_seven_moles (molecular_weight : ℕ) (h : molecular_weight = 1176) :
  7 * molecular_weight = 8232 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_seven_moles_l808_80891


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l808_80817

/-- The cost price of a bicycle given a series of sales with specified profit margins -/
theorem bicycle_cost_price
  (profit_A_to_B : Real)
  (profit_B_to_C : Real)
  (profit_C_to_D : Real)
  (final_price : Real)
  (h1 : profit_A_to_B = 0.50)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.15)
  (h4 : final_price = 320.75) :
  ∃ (cost_price : Real),
    cost_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l808_80817


namespace NUMINAMATH_CALUDE_coefficient_of_a_half_power_l808_80831

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (a - 1/√a)^5
def expansion (a : ℝ) : ℝ → ℝ := sorry

-- Theorem statement
theorem coefficient_of_a_half_power (a : ℝ) :
  ∃ (c : ℝ), c = -10 ∧ 
  (∀ (k : ℕ), k ≠ 3 → (binomial 5 k) * (-1)^k * a^(5 - k - k/2) ≠ c * a^(1/2)) ∧
  (binomial 5 3) * (-1)^3 * a^(5 - 3 - 3/2) = c * a^(1/2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_a_half_power_l808_80831


namespace NUMINAMATH_CALUDE_bike_owners_without_scooters_l808_80825

theorem bike_owners_without_scooters (total_population : ℕ) 
  (bike_owners : ℕ) (scooter_owners : ℕ) 
  (h1 : total_population = 420)
  (h2 : bike_owners = 380)
  (h3 : scooter_owners = 82)
  (h4 : ∀ p, p ∈ Set.range (Fin.val : Fin total_population → ℕ) → 
    (p ∈ Set.range (Fin.val : Fin bike_owners → ℕ) ∨ 
     p ∈ Set.range (Fin.val : Fin scooter_owners → ℕ))) :
  bike_owners - (bike_owners + scooter_owners - total_population) = 338 :=
sorry

end NUMINAMATH_CALUDE_bike_owners_without_scooters_l808_80825


namespace NUMINAMATH_CALUDE_continued_proportionate_reduction_eq_euclidean_gcd_l808_80820

/-- The Method of Continued Proportionate Reduction as used in ancient Chinese mathematics -/
def continued_proportionate_reduction (a b : ℕ) : ℕ :=
  sorry

/-- The Euclidean algorithm for finding the greatest common divisor -/
def euclidean_gcd (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating the equivalence of the two methods -/
theorem continued_proportionate_reduction_eq_euclidean_gcd :
  ∀ a b : ℕ, continued_proportionate_reduction a b = euclidean_gcd a b :=
sorry

end NUMINAMATH_CALUDE_continued_proportionate_reduction_eq_euclidean_gcd_l808_80820


namespace NUMINAMATH_CALUDE_octagon_area_division_l808_80807

theorem octagon_area_division (CO OM MP PU UT TE : ℝ) (D : ℝ) :
  CO = 1 ∧ OM = 1 ∧ MP = 1 ∧ PU = 1 ∧ UT = 1 ∧ TE = 1 →
  (∃ (COMPUTER_area COMPUTED_area CDR_area : ℝ),
    COMPUTER_area = 6 ∧
    COMPUTED_area = 3 ∧
    CDR_area = 3 ∧
    COMPUTED_area = CDR_area) →
  (∃ (CD DR : ℝ),
    CD = 3 ∧
    CDR_area = 1/2 * CD * DR) →
  DR = 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_division_l808_80807


namespace NUMINAMATH_CALUDE_oranges_left_l808_80869

theorem oranges_left (total : ℕ) (percentage : ℚ) (remaining : ℕ) : 
  total = 96 → 
  percentage = 48/100 →
  remaining = total - Int.floor (percentage * total) →
  remaining = 50 := by
sorry

end NUMINAMATH_CALUDE_oranges_left_l808_80869


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l808_80842

theorem geometric_sequence_value (x : ℝ) : 
  (∃ r : ℝ, x / 12 = r ∧ 3 / x = r) → x = 6 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l808_80842


namespace NUMINAMATH_CALUDE_squat_lift_loss_percentage_l808_80845

/-- Calculates the percentage of squat lift lost given the original lifts and new total lift -/
theorem squat_lift_loss_percentage
  (orig_squat : ℝ)
  (orig_bench : ℝ)
  (orig_deadlift : ℝ)
  (deadlift_loss : ℝ)
  (new_total : ℝ)
  (h1 : orig_squat = 700)
  (h2 : orig_bench = 400)
  (h3 : orig_deadlift = 800)
  (h4 : deadlift_loss = 200)
  (h5 : new_total = 1490) :
  (orig_squat - (new_total - (orig_bench + (orig_deadlift - deadlift_loss)))) / orig_squat * 100 = 30 :=
by sorry

end NUMINAMATH_CALUDE_squat_lift_loss_percentage_l808_80845


namespace NUMINAMATH_CALUDE_sophies_shopping_l808_80896

theorem sophies_shopping (total_budget : ℚ) (trouser_cost : ℚ) (additional_items : ℕ) (additional_item_cost : ℚ) (num_shirts : ℕ) :
  total_budget = 260 →
  trouser_cost = 63 →
  additional_items = 4 →
  additional_item_cost = 40 →
  num_shirts = 2 →
  ∃ (shirt_cost : ℚ), 
    shirt_cost * num_shirts + trouser_cost + (additional_items : ℚ) * additional_item_cost = total_budget ∧
    shirt_cost = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sophies_shopping_l808_80896


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_360_factorization_360_l808_80872

/-- The number of perfect square factors of 360 -/
def perfect_square_factors_360 : ℕ :=
  4

theorem count_perfect_square_factors_360 :
  perfect_square_factors_360 = 4 := by
  sorry

/-- Prime factorization of 360 -/
theorem factorization_360 : 360 = 2^3 * 3^2 * 5 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_360_factorization_360_l808_80872


namespace NUMINAMATH_CALUDE_salad_dressing_calories_l808_80808

/-- Calculates the calories in the salad dressing given the total calories consumed and the calories from other ingredients. -/
theorem salad_dressing_calories :
  let lettuce_calories : ℝ := 50
  let carrot_calories : ℝ := 2 * lettuce_calories
  let pizza_crust_calories : ℝ := 600
  let pepperoni_calories : ℝ := (1 / 3) * pizza_crust_calories
  let cheese_calories : ℝ := 400
  let salad_portion : ℝ := 1 / 4
  let pizza_portion : ℝ := 1 / 5
  let total_calories_consumed : ℝ := 330

  let salad_calories_without_dressing : ℝ := (lettuce_calories + carrot_calories) * salad_portion
  let pizza_calories : ℝ := (pizza_crust_calories + pepperoni_calories + cheese_calories) * pizza_portion
  let calories_without_dressing : ℝ := salad_calories_without_dressing + pizza_calories
  let dressing_calories : ℝ := total_calories_consumed - calories_without_dressing

  dressing_calories = 52.5 := by sorry

end NUMINAMATH_CALUDE_salad_dressing_calories_l808_80808


namespace NUMINAMATH_CALUDE_positive_reals_inequalities_l808_80840

theorem positive_reals_inequalities (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 1) : 
  x + y - 4*x*y ≥ 0 ∧ 1/x + 4/(1+y) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequalities_l808_80840


namespace NUMINAMATH_CALUDE_ruth_math_class_hours_l808_80812

/-- Calculates the number of hours spent in math class per week for a student with given school schedule and math class percentage. -/
def math_class_hours_per_week (hours_per_day : ℕ) (days_per_week : ℕ) (math_class_percentage : ℚ) : ℚ :=
  (hours_per_day * days_per_week : ℚ) * math_class_percentage

/-- Theorem stating that a student who attends school for 8 hours a day, 5 days a week, and spends 25% of their school time in math class, spends 10 hours per week in math class. -/
theorem ruth_math_class_hours :
  math_class_hours_per_week 8 5 (1/4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_hours_l808_80812


namespace NUMINAMATH_CALUDE_g_g_two_roots_l808_80846

/-- The function g(x) defined as x^2 + 2x + c^2 -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c^2

/-- The theorem stating that g(g(x)) has exactly two distinct real roots iff c = ±1 -/
theorem g_g_two_roots (c : ℝ) :
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g c (g c x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ c = 1 ∨ c = -1 :=
sorry

end NUMINAMATH_CALUDE_g_g_two_roots_l808_80846


namespace NUMINAMATH_CALUDE_apple_difference_l808_80874

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_count : adam_apples = 9) 
  (jackie_count : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l808_80874


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l808_80859

theorem cube_sum_inequality (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l808_80859


namespace NUMINAMATH_CALUDE_square_side_length_l808_80822

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 18) : ∃ square_side : ℝ,
  square_side = 12 ∧ 4 * square_side = 2 * (rectangle_width + rectangle_length) := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l808_80822


namespace NUMINAMATH_CALUDE_probability_red_or_green_l808_80848

/-- The probability of drawing a red or green marble from a bag with specified marble counts. -/
theorem probability_red_or_green (red green blue yellow : ℕ) : 
  let total := red + green + blue + yellow
  (red + green : ℚ) / total = 9 / 14 :=
by
  sorry

#check probability_red_or_green 5 4 2 3

end NUMINAMATH_CALUDE_probability_red_or_green_l808_80848


namespace NUMINAMATH_CALUDE_samosa_price_is_two_l808_80873

/-- Represents the cost of a meal at Delicious Delhi restaurant --/
structure MealCost where
  samosa_price : ℝ
  samosa_quantity : ℕ
  pakora_price : ℝ
  pakora_quantity : ℕ
  lassi_price : ℝ
  tip_percentage : ℝ
  total_with_tax : ℝ

/-- Theorem stating that the samosa price is $2 given the conditions of Hilary's meal --/
theorem samosa_price_is_two (meal : MealCost) : meal.samosa_price = 2 :=
  by
  have h1 : meal.samosa_quantity = 3 := by sorry
  have h2 : meal.pakora_price = 3 := by sorry
  have h3 : meal.pakora_quantity = 4 := by sorry
  have h4 : meal.lassi_price = 2 := by sorry
  have h5 : meal.tip_percentage = 0.25 := by sorry
  have h6 : meal.total_with_tax = 25 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_samosa_price_is_two_l808_80873


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l808_80866

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : SumCondition a) :
  ∃ d : ℝ, a 7 - a 8 = -d ∧ ArithmeticSequence a ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l808_80866


namespace NUMINAMATH_CALUDE_number_difference_l808_80828

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23405)
  (b_div_5 : ∃ k : ℕ, b = 5 * k)
  (b_div_10_eq_5a : b / 10 = 5 * a) :
  b - a = 21600 :=
by sorry

end NUMINAMATH_CALUDE_number_difference_l808_80828


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l808_80881

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the parallel condition
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  are_parallel (a.1 + 2 * (b m).1, a.2 + 2 * (b m).2) (2 * a.1 - (b m).1, 2 * a.2 - (b m).2) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l808_80881


namespace NUMINAMATH_CALUDE_girls_ran_27_miles_l808_80878

/-- Calculates the total distance run by the girls in miles -/
def girls_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (mile_per_lap : ℚ) : ℚ :=
  (boys_laps + girls_extra_laps) * mile_per_lap

/-- Proves that the girls ran 27 miles given the specified conditions -/
theorem girls_ran_27_miles :
  let boys_laps : ℕ := 27
  let girls_extra_laps : ℕ := 9
  let mile_per_lap : ℚ := 3/4
  girls_distance boys_laps girls_extra_laps mile_per_lap = 27 := by
  sorry

#eval girls_distance 27 9 (3/4)

end NUMINAMATH_CALUDE_girls_ran_27_miles_l808_80878


namespace NUMINAMATH_CALUDE_max_count_is_1296_l808_80892

/-- Represents a configuration of pluses and minuses in a 30x30 table --/
structure TableConfig where
  pluses : Fin 30 → Fin 30 → Bool
  minuses : Fin 30 → Fin 30 → Bool

/-- The count for a given configuration --/
def count (config : TableConfig) : ℕ :=
  sorry

/-- Constraint: Total number of pluses is 162 --/
def total_pluses (config : TableConfig) : Prop :=
  (Finset.sum Finset.univ fun i => 
   (Finset.sum Finset.univ fun j => if config.pluses i j then 1 else 0)) = 162

/-- Constraint: Total number of minuses is 144 --/
def total_minuses (config : TableConfig) : Prop :=
  (Finset.sum Finset.univ fun i => 
   (Finset.sum Finset.univ fun j => if config.minuses i j then 1 else 0)) = 144

/-- Constraint: Each row contains at most 17 signs --/
def row_constraint (config : TableConfig) : Prop :=
  ∀ i, (Finset.sum Finset.univ fun j => 
       (if config.pluses i j then 1 else 0) + (if config.minuses i j then 1 else 0)) ≤ 17

/-- Constraint: Each column contains at most 17 signs --/
def col_constraint (config : TableConfig) : Prop :=
  ∀ j, (Finset.sum Finset.univ fun i => 
       (if config.pluses i j then 1 else 0) + (if config.minuses i j then 1 else 0)) ≤ 17

/-- Main theorem: The maximum count is 1296 --/
theorem max_count_is_1296 : 
  ∃ (config : TableConfig), 
    total_pluses config ∧ 
    total_minuses config ∧ 
    row_constraint config ∧ 
    col_constraint config ∧ 
    count config = 1296 ∧
    (∀ (other_config : TableConfig), 
      total_pluses other_config → 
      total_minuses other_config → 
      row_constraint other_config → 
      col_constraint other_config → 
      count other_config ≤ 1296) :=
  sorry

end NUMINAMATH_CALUDE_max_count_is_1296_l808_80892
