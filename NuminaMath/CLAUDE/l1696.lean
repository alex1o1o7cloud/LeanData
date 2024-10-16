import Mathlib

namespace NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l1696_169640

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l1696_169640


namespace NUMINAMATH_CALUDE_lucas_sticker_redistribution_l1696_169676

theorem lucas_sticker_redistribution
  (n : ℚ)  -- Noah's initial number of stickers
  (h1 : n > 0)  -- Ensure n is positive
  (emma : ℚ)  -- Emma's initial number of stickers
  (h2 : emma = 3 * n)  -- Emma has 3 times as many stickers as Noah
  (lucas : ℚ)  -- Lucas's initial number of stickers
  (h3 : lucas = 4 * emma)  -- Lucas has 4 times as many stickers as Emma
  : (lucas - (lucas + emma + n) / 3) / lucas = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_lucas_sticker_redistribution_l1696_169676


namespace NUMINAMATH_CALUDE_bike_distance_l1696_169652

/-- Theorem: A bike moving at a constant speed of 4 m/s for 8 seconds travels 32 meters. -/
theorem bike_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 8 → distance = speed * time → distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_l1696_169652


namespace NUMINAMATH_CALUDE_population_growth_rate_l1696_169673

/-- Given that a population increases by 90 persons in 30 minutes,
    prove that it takes 20 seconds for one person to be added. -/
theorem population_growth_rate (increase : ℕ) (time_minutes : ℕ) (time_seconds : ℕ) :
  increase = 90 →
  time_minutes = 30 →
  time_seconds = time_minutes * 60 →
  time_seconds / increase = 20 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l1696_169673


namespace NUMINAMATH_CALUDE_sequence_sum_l1696_169643

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  b - a = c - b ∧
  c * c = b * d ∧
  d - a = 20 →
  a + b + c + d = 46 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1696_169643


namespace NUMINAMATH_CALUDE_equal_earnings_l1696_169664

theorem equal_earnings (t : ℝ) : 
  (t - 4) * (3 * t - 7) = (3 * t - 12) * (t - 3) → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_earnings_l1696_169664


namespace NUMINAMATH_CALUDE_range_of_M_l1696_169617

theorem range_of_M (x y z : ℝ) 
  (h1 : x + y + z = 30)
  (h2 : 3 * x + y - z = 50)
  (h3 : x ≥ 0)
  (h4 : y ≥ 0)
  (h5 : z ≥ 0) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_l1696_169617


namespace NUMINAMATH_CALUDE_pokemon_cards_total_l1696_169649

theorem pokemon_cards_total (jenny : ℕ) (orlando : ℕ) (richard : ℕ) : 
  jenny = 6 →
  orlando = jenny + 2 →
  richard = 3 * orlando →
  jenny + orlando + richard = 38 := by
sorry

end NUMINAMATH_CALUDE_pokemon_cards_total_l1696_169649


namespace NUMINAMATH_CALUDE_inverse_function_point_l1696_169601

-- Define a monotonic function f
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)

-- Define the condition that f(x+1) passes through (-2, 1)
variable (h_point : f (-1) = 1)

-- State the theorem
theorem inverse_function_point :
  (Function.invFun f) 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l1696_169601


namespace NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l1696_169646

/-- Represents a seating arrangement in a row of seats. -/
structure SeatingArrangement where
  total_seats : ℕ
  min_gap : ℕ
  occupied_seats : ℕ

/-- Checks if a seating arrangement is valid according to the rules. -/
def is_valid_arrangement (sa : SeatingArrangement) : Prop :=
  sa.total_seats = 150 ∧ sa.min_gap = 2 ∧ sa.occupied_seats ≤ sa.total_seats

/-- Checks if the next person must sit next to someone in the given arrangement. -/
def forces_adjacent_seating (sa : SeatingArrangement) : Prop :=
  ∀ (new_seat : ℕ), new_seat ≤ sa.total_seats →
    (∃ (occupied : ℕ), occupied ≤ sa.total_seats ∧
      (new_seat = occupied + 1 ∨ new_seat = occupied - 1))

/-- The main theorem stating the minimum number of occupied seats. -/
theorem min_seats_for_adjacent_seating :
  ∃ (sa : SeatingArrangement),
    is_valid_arrangement sa ∧
    forces_adjacent_seating sa ∧
    sa.occupied_seats = 74 ∧
    (∀ (sa' : SeatingArrangement),
      is_valid_arrangement sa' ∧
      forces_adjacent_seating sa' →
      sa'.occupied_seats ≥ 74) :=
  sorry

end NUMINAMATH_CALUDE_min_seats_for_adjacent_seating_l1696_169646


namespace NUMINAMATH_CALUDE_paper_cutting_theorem_l1696_169636

/-- Represents a polygon --/
structure Polygon where
  vertices : ℕ

/-- Represents the state of the paper after cuts --/
structure PaperState where
  polygons : List Polygon
  totalVertices : ℕ

/-- Initial state of the rectangular paper --/
def initialState : PaperState :=
  { polygons := [{ vertices := 4 }], totalVertices := 4 }

/-- Perform a single cut on the paper state --/
def performCut (state : PaperState) : PaperState :=
  { polygons := state.polygons ++ [{ vertices := 3 }],
    totalVertices := state.totalVertices + 2 }

/-- Perform n cuts on the paper state --/
def performCuts (n : ℕ) (state : PaperState) : PaperState :=
  match n with
  | 0 => state
  | n + 1 => performCuts n (performCut state)

/-- The main theorem to prove --/
theorem paper_cutting_theorem :
  (performCuts 100 initialState).totalVertices ≠ 302 :=
by sorry

end NUMINAMATH_CALUDE_paper_cutting_theorem_l1696_169636


namespace NUMINAMATH_CALUDE_farm_field_problem_l1696_169607

/-- Represents the problem of calculating the farm field area and initial work plan --/
theorem farm_field_problem (planned_daily_rate : ℕ) (actual_daily_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) 
  (h1 : planned_daily_rate = 90)
  (h2 : actual_daily_rate = 85)
  (h3 : extra_days = 2)
  (h4 : area_left = 40) :
  ∃ (total_area : ℕ) (initial_days : ℕ),
    total_area = 3780 ∧ 
    initial_days = 42 ∧
    planned_daily_rate * initial_days = total_area ∧
    actual_daily_rate * (initial_days + extra_days) + area_left = total_area :=
by
  sorry

end NUMINAMATH_CALUDE_farm_field_problem_l1696_169607


namespace NUMINAMATH_CALUDE_absolute_value_equation_a_l1696_169656

theorem absolute_value_equation_a (x : ℝ) : |x - 5| = 2 ↔ x = 3 ∨ x = 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_a_l1696_169656


namespace NUMINAMATH_CALUDE_james_printing_sheets_l1696_169624

theorem james_printing_sheets (num_books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) :
  num_books = 2 →
  pages_per_book = 600 →
  pages_per_side = 4 →
  (num_books * pages_per_book) / (2 * pages_per_side) = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_printing_sheets_l1696_169624


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1696_169685

theorem trigonometric_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < π / 2) :
  π / 2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z >
  Real.sin (2 * x) + Real.sin (2 * y) + Real.sin (2 * z) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1696_169685


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l1696_169695

/-- Given two lines intersecting at point P(2,5) with slopes 3 and -1 respectively,
    and forming a triangle PQR with the x-axis, prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P : ℝ × ℝ) (m₁ m₂ : ℝ) : 
  P = (2, 5) →
  m₁ = 3 →
  m₂ = -1 →
  let Q := (P.1 - P.2 / m₁, 0)
  let R := (P.1 + P.2 / m₂, 0)
  (1/2 : ℝ) * |R.1 - Q.1| * P.2 = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l1696_169695


namespace NUMINAMATH_CALUDE_area_of_similar_pentagons_l1696_169650

/-- Theorem: Area of similar pentagons
  Given two similar pentagons with perimeters K₁ and K₂, and areas L₁ and L₂,
  if K₁ = 18, K₂ = 24, and L₁ = 8 7/16, then L₂ = 15.
-/
theorem area_of_similar_pentagons (K₁ K₂ L₁ L₂ : ℝ) : 
  K₁ = 18 → K₂ = 24 → L₁ = 8 + 7/16 → 
  (K₁ / K₂)^2 = L₁ / L₂ → 
  L₂ = 15 := by
  sorry


end NUMINAMATH_CALUDE_area_of_similar_pentagons_l1696_169650


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1696_169692

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x + (5/8) * a - (3/2)

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f a x = 2) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1696_169692


namespace NUMINAMATH_CALUDE_potato_bag_weight_l1696_169672

theorem potato_bag_weight (current_weight : ℝ) (h : current_weight = 12) :
  ∃ (original_weight : ℝ), original_weight / 2 = current_weight ∧ original_weight = 24 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l1696_169672


namespace NUMINAMATH_CALUDE_system_equiv_line_l1696_169682

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system of equations is equivalent to the solution line -/
theorem system_equiv_line : 
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end NUMINAMATH_CALUDE_system_equiv_line_l1696_169682


namespace NUMINAMATH_CALUDE_max_y_coord_sin_3theta_l1696_169645

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 1 -/
theorem max_y_coord_sin_3theta :
  let r : ℝ → ℝ := λ θ ↦ Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ : ℝ), y θ = 1 ∧ ∀ (φ : ℝ), y φ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coord_sin_3theta_l1696_169645


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1696_169603

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p+1)*x + p = 0 → (∃! r, x = r)) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1696_169603


namespace NUMINAMATH_CALUDE_initial_number_of_men_l1696_169693

/-- Given a group of men where replacing two men (aged 20 and 22) with two women (average age 29)
    increases the average age by 2 years, prove that the initial number of men is 8. -/
theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * 29 - (20 + 22) : ℝ) = 2 * M → M = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_men_l1696_169693


namespace NUMINAMATH_CALUDE_trees_survived_vs_died_l1696_169658

theorem trees_survived_vs_died (initial_trees : ℕ) (trees_died : ℕ) : 
  initial_trees = 13 → trees_died = 6 → (initial_trees - trees_died) - trees_died = 1 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_vs_died_l1696_169658


namespace NUMINAMATH_CALUDE_race_probability_l1696_169689

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℚ) : 
  total_cars = 16 →
  prob_Y = 1/12 →
  prob_Z = 1/7 →
  prob_XYZ = 47619047619047616/100000000000000000 →
  ∃ (prob_X : ℚ), 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧
    prob_X = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l1696_169689


namespace NUMINAMATH_CALUDE_three_true_propositions_l1696_169625

theorem three_true_propositions : 
  (¬∀ (a b c : ℝ), a > b → a * c < b * c) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  (∀ (a b : ℝ), a > b ∧ 1 / a > 1 / b → a > 0 ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l1696_169625


namespace NUMINAMATH_CALUDE_number_equation_solution_l1696_169611

theorem number_equation_solution :
  ∃ x : ℚ, (35 + 3 * x = 51) ∧ (x = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1696_169611


namespace NUMINAMATH_CALUDE_xy_equals_twelve_l1696_169605

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_twelve_l1696_169605


namespace NUMINAMATH_CALUDE_license_plate_difference_l1696_169671

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The number of possible Sunshine license plates -/
def sunshine_plates : ℕ := num_letters^3 * num_digits^3

/-- The number of possible Prairie license plates -/
def prairie_plates : ℕ := num_letters^2 * num_digits^4

/-- The difference in the number of possible license plates between Sunshine and Prairie -/
def plate_difference : ℕ := sunshine_plates - prairie_plates

theorem license_plate_difference :
  plate_difference = 10816000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1696_169671


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l1696_169627

theorem opposite_of_negative_three_fourths :
  ∀ x : ℚ, x + (-3/4) = 0 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l1696_169627


namespace NUMINAMATH_CALUDE_wendy_walked_distance_l1696_169621

/-- The number of miles Wendy ran -/
def miles_ran : ℝ := 19.83

/-- The difference between miles ran and walked -/
def difference : ℝ := 10.67

/-- The number of miles Wendy walked -/
def miles_walked : ℝ := miles_ran - difference

theorem wendy_walked_distance : miles_walked = 9.16 := by
  sorry

end NUMINAMATH_CALUDE_wendy_walked_distance_l1696_169621


namespace NUMINAMATH_CALUDE_seller_loss_is_30_l1696_169644

/-- Represents the transaction between a seller and a buyer -/
structure Transaction where
  goods_value : ℕ
  payment : ℕ
  counterfeit : Bool

/-- Calculates the seller's loss given a transaction -/
def seller_loss (t : Transaction) : ℕ :=
  if t.counterfeit then
    t.payment + (t.payment - t.goods_value)
  else
    0

/-- Theorem stating that the seller's loss is 30 rubles given the specific transaction -/
theorem seller_loss_is_30 (t : Transaction) 
  (h1 : t.goods_value = 10)
  (h2 : t.payment = 25)
  (h3 : t.counterfeit = true) : 
  seller_loss t = 30 := by
  sorry

#eval seller_loss { goods_value := 10, payment := 25, counterfeit := true }

end NUMINAMATH_CALUDE_seller_loss_is_30_l1696_169644


namespace NUMINAMATH_CALUDE_product_expansion_l1696_169659

theorem product_expansion (x : ℝ) : (2 + 3 * x) * (-2 + 3 * x) = 9 * x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1696_169659


namespace NUMINAMATH_CALUDE_drainage_pipes_count_l1696_169677

/-- The number of initial drainage pipes -/
def n : ℕ := 5

/-- The time (in days) it takes n pipes to drain the pool -/
def initial_time : ℕ := 12

/-- The time (in days) it takes (n + 10) pipes to drain the pool -/
def faster_time : ℕ := 4

/-- The number of additional pipes -/
def additional_pipes : ℕ := 10

theorem drainage_pipes_count :
  (n : ℚ) * faster_time = (n + additional_pipes) * initial_time :=
sorry

end NUMINAMATH_CALUDE_drainage_pipes_count_l1696_169677


namespace NUMINAMATH_CALUDE_circles_intersection_l1696_169657

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ → ℝ × ℝ := λ m ↦ (m, -1)

-- Define the line equation
def line_equation (x y c : ℝ) : Prop := x - y + c = 0

-- Theorem statement
theorem circles_intersection (m c : ℝ) 
  (h1 : ∃ (center1 center2 : ℝ × ℝ), 
    line_equation center1.1 center1.2 c ∧ 
    line_equation center2.1 center2.2 c) : 
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_l1696_169657


namespace NUMINAMATH_CALUDE_class_average_marks_l1696_169610

theorem class_average_marks (students1 students2 : ℕ) (avg2 combined_avg : ℚ) :
  students1 = 12 →
  students2 = 28 →
  avg2 = 60 →
  combined_avg = 54 →
  (students1 : ℚ) * (40 : ℚ) + (students2 : ℚ) * avg2 = (students1 + students2 : ℚ) * combined_avg :=
by sorry

end NUMINAMATH_CALUDE_class_average_marks_l1696_169610


namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l1696_169690

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define external tangency condition
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y m ∧
  ∀ (x' y' : ℝ), C₁ x' y' → C₂ x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem circles_tangent_m_value :
  ∀ m : ℝ, externally_tangent m → m = 9 :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l1696_169690


namespace NUMINAMATH_CALUDE_income_minus_expenses_tax_lower_l1696_169698

/-- Represents the tax options available --/
inductive TaxOption
  | IncomeTax
  | IncomeMinusExpensesTax

/-- Calculates the tax payable for a given option --/
def calculateTax (option : TaxOption) (totalIncome expenses insuranceContributions : ℕ) : ℕ :=
  match option with
  | TaxOption.IncomeTax =>
      let incomeTax := totalIncome * 6 / 100
      let maxDeduction := min (incomeTax / 2) insuranceContributions
      incomeTax - maxDeduction
  | TaxOption.IncomeMinusExpensesTax =>
      let taxBase := totalIncome - expenses
      let regularTax := taxBase * 15 / 100
      let minimumTax := totalIncome * 1 / 100
      max regularTax minimumTax

/-- Theorem stating that the Income minus expenses tax option results in lower tax --/
theorem income_minus_expenses_tax_lower
  (totalIncome expenses insuranceContributions : ℕ)
  (h1 : totalIncome = 150000000)
  (h2 : expenses = 141480000)
  (h3 : insuranceContributions = 16560000) :
  calculateTax TaxOption.IncomeMinusExpensesTax totalIncome expenses insuranceContributions <
  calculateTax TaxOption.IncomeTax totalIncome expenses insuranceContributions :=
by
  sorry


end NUMINAMATH_CALUDE_income_minus_expenses_tax_lower_l1696_169698


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1696_169641

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 2*x

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1696_169641


namespace NUMINAMATH_CALUDE_divisors_of_factorial_eight_l1696_169687

theorem divisors_of_factorial_eight (n : ℕ) : n = 8 → (Nat.divisors (Nat.factorial n)).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_factorial_eight_l1696_169687


namespace NUMINAMATH_CALUDE_original_people_count_l1696_169683

theorem original_people_count (x : ℕ) : 
  (x / 2 : ℕ) = 18 → 
  x = 36 :=
by
  sorry

#check original_people_count

end NUMINAMATH_CALUDE_original_people_count_l1696_169683


namespace NUMINAMATH_CALUDE_factorial_square_root_problem_l1696_169697

theorem factorial_square_root_problem : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_problem_l1696_169697


namespace NUMINAMATH_CALUDE_equation_solution_l1696_169669

theorem equation_solution : 
  ∃ x : ℝ, (x - 6)^4 = (1/16)⁻¹ ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1696_169669


namespace NUMINAMATH_CALUDE_unique_number_with_divisor_properties_l1696_169638

theorem unique_number_with_divisor_properties :
  ∀ (N p q r : ℕ) (α β γ : ℕ),
    (∃ (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q) (h_prime_r : Nat.Prime r),
      N = p^α * q^β * r^γ ∧
      p * q - r = 3 ∧
      p * r - q = 9 ∧
      (Nat.divisors (N / p)).card = (Nat.divisors N).card - 20 ∧
      (Nat.divisors (N / q)).card = (Nat.divisors N).card - 12 ∧
      (Nat.divisors (N / r)).card = (Nat.divisors N).card - 15) →
    N = 857500 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_divisor_properties_l1696_169638


namespace NUMINAMATH_CALUDE_pick_two_different_suits_standard_deck_l1696_169662

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The number of ways to pick two cards from different suits -/
def pick_two_different_suits (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - d.cards_per_suit)

theorem pick_two_different_suits_standard_deck :
  pick_two_different_suits standard_deck = 2028 := by
  sorry

#eval pick_two_different_suits standard_deck

end NUMINAMATH_CALUDE_pick_two_different_suits_standard_deck_l1696_169662


namespace NUMINAMATH_CALUDE_unique_square_pattern_l1696_169634

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Checks if a number satisfies the squaring pattern -/
def satisfiesSquarePattern (n : ThreeDigitNumber) : Prop :=
  let square := n.toNat * n.toNat
  -- Add conditions here that check if the square follows the pattern
  -- This is a placeholder and should be replaced with actual conditions
  true

/-- The main theorem stating that 748 is the only number satisfying the conditions -/
theorem unique_square_pattern : 
  ∃! n : ThreeDigitNumber, satisfiesSquarePattern n ∧ n.toNat = 748 := by
  sorry

#check unique_square_pattern

end NUMINAMATH_CALUDE_unique_square_pattern_l1696_169634


namespace NUMINAMATH_CALUDE_pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l1696_169648

/-- The number of pufferfish in an aquarium exhibit -/
def num_pufferfish : ℕ := 15

/-- The number of swordfish in the aquarium exhibit -/
def num_swordfish : ℕ := 5 * num_pufferfish

/-- The total number of fish in the aquarium exhibit -/
def total_fish : ℕ := 90

/-- Theorem stating that the number of pufferfish is 15 -/
theorem pufferfish_count : num_pufferfish = 15 := by sorry

/-- Theorem stating that the number of swordfish is five times the number of pufferfish -/
theorem swordfish_to_pufferfish_ratio : num_swordfish = 5 * num_pufferfish := by sorry

/-- Theorem stating that the total number of fish is 90 -/
theorem total_fish_count : total_fish = num_swordfish + num_pufferfish := by sorry

end NUMINAMATH_CALUDE_pufferfish_count_swordfish_to_pufferfish_ratio_total_fish_count_l1696_169648


namespace NUMINAMATH_CALUDE_phone_number_A_value_l1696_169618

def phone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  D % 2 = 0 ∧ E % 2 = 0 ∧ F % 2 = 0 ∧
  D = E + 2 ∧ E = F + 2 ∧
  G % 2 = 1 ∧ H % 2 = 1 ∧ I % 2 = 1 ∧ J % 2 = 1 ∧
  G = H + 2 ∧ H = I + 2 ∧ I = J + 4 ∧
  J = 1 ∧
  A + B + C = 11 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem phone_number_A_value :
  ∀ A B C D E F G H I J : ℕ,
  phone_number A B C D E F G H I J →
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_phone_number_A_value_l1696_169618


namespace NUMINAMATH_CALUDE_angle_between_diagonals_is_133_l1696_169626

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the angles of the quadrilateral
def angle_ABC (q : Quadrilateral) : ℝ := 116
def angle_ADC (q : Quadrilateral) : ℝ := 64
def angle_CAB (q : Quadrilateral) : ℝ := 35
def angle_CAD (q : Quadrilateral) : ℝ := 52

-- Define the angle between diagonals subtended by side AB
def angle_between_diagonals (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem angle_between_diagonals_is_133 (q : Quadrilateral) : 
  angle_between_diagonals q = 133 := by sorry

end NUMINAMATH_CALUDE_angle_between_diagonals_is_133_l1696_169626


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_unknown_blanket_rate_eq_175_l1696_169681

/-- The unknown rate of two blankets given the following conditions:
    - 1 blanket purchased at Rs. 100
    - 5 blankets purchased at Rs. 150 each
    - 2 blankets purchased at an unknown rate
    - The average price of all blankets is Rs. 150
-/
theorem unknown_blanket_rate : ℕ :=
  let num_blankets_1 : ℕ := 1
  let price_1 : ℕ := 100
  let num_blankets_2 : ℕ := 5
  let price_2 : ℕ := 150
  let num_blankets_3 : ℕ := 2
  let total_blankets : ℕ := num_blankets_1 + num_blankets_2 + num_blankets_3
  let average_price : ℕ := 150
  let total_cost : ℕ := average_price * total_blankets
  let known_cost : ℕ := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_rate : ℕ := (total_cost - known_cost) / num_blankets_3
  unknown_rate

theorem unknown_blanket_rate_eq_175 : unknown_blanket_rate = 175 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_unknown_blanket_rate_eq_175_l1696_169681


namespace NUMINAMATH_CALUDE_three_digit_permutations_l1696_169680

def digits : Finset Nat := {1, 5, 8}

theorem three_digit_permutations (d : Finset Nat) (h : d = digits) :
  (d.toList.permutations.filter (fun l => l.length = 3)).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutations_l1696_169680


namespace NUMINAMATH_CALUDE_midpoint_parallelogram_l1696_169653

/-- A quadrilateral in 2D plane represented by its vertices -/
structure Quadrilateral where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Points that divide the sides of a quadrilateral in ratio r -/
def divisionPoints (q : Quadrilateral) (r : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let (x1, y1) := q.v1
  let (x2, y2) := q.v2
  let (x3, y3) := q.v3
  let (x4, y4) := q.v4
  ( ((x2 - x1) * r + x1, (y2 - y1) * r + y1),
    ((x3 - x2) * r + x2, (y3 - y2) * r + y2),
    ((x4 - x3) * r + x3, (y4 - y3) * r + y3),
    ((x1 - x4) * r + x4, (y1 - y4) * r + y4) )

/-- Check if the quadrilateral formed by the division points is a parallelogram -/
def isParallelogram (points : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) := points
  (x3 - x1 = x4 - x2) ∧ (y3 - y1 = y4 - y2)

/-- The main theorem: only midpoints (r = 1/2) form a parallelogram for all quadrilaterals -/
theorem midpoint_parallelogram (q : Quadrilateral) :
    ∀ r : ℝ, (∀ q' : Quadrilateral, isParallelogram (divisionPoints q' r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_parallelogram_l1696_169653


namespace NUMINAMATH_CALUDE_fewest_tiles_required_l1696_169630

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ :=
  d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ :=
  feet * 12

/-- The dimensions of a tile in inches -/
def tileDimensions : Dimensions :=
  { length := 2, width := 3 }

/-- The dimensions of the region in feet -/
def regionDimensionsFeet : Dimensions :=
  { length := 4, width := 6 }

/-- The dimensions of the region in inches -/
def regionDimensionsInches : Dimensions :=
  { length := feetToInches regionDimensionsFeet.length,
    width := feetToInches regionDimensionsFeet.width }

/-- Theorem: The fewest number of tiles required to cover the region is 576 -/
theorem fewest_tiles_required :
  (area regionDimensionsInches) / (area tileDimensions) = 576 := by
  sorry

end NUMINAMATH_CALUDE_fewest_tiles_required_l1696_169630


namespace NUMINAMATH_CALUDE_jack_piggy_bank_total_l1696_169620

/-- Calculates the final amount in Jack's piggy bank after a given number of weeks -/
def piggy_bank_total (initial_amount : ℝ) (weekly_allowance : ℝ) (savings_rate : ℝ) (weeks : ℕ) : ℝ :=
  initial_amount + (weekly_allowance * savings_rate * weeks)

/-- Proves that Jack will have $83 in his piggy bank after 8 weeks -/
theorem jack_piggy_bank_total :
  piggy_bank_total 43 10 0.5 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_jack_piggy_bank_total_l1696_169620


namespace NUMINAMATH_CALUDE_digit_sum_congruence_l1696_169661

/-- The digit sum of n in base r -/
noncomputable def digit_sum (r n : ℕ) : ℕ := sorry

theorem digit_sum_congruence :
  (∀ r > 2, ∃ p : ℕ, Nat.Prime p ∧ ∀ n > 0, digit_sum r n ≡ n [MOD p]) ∧
  (∀ r > 1, ∀ p : ℕ, Nat.Prime p → ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, digit_sum r n ≡ n [MOD p]) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_congruence_l1696_169661


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_trajectory_theorem_l1696_169623

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line passing through (1,0)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the circle condition
def circle_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Define the vector equation
def vector_equation (x y x₁ y₁ x₂ y₂ : ℝ) : Prop := 
  x = x₁ + x₂ - 1/4 ∧ y = y₁ + y₂

-- Theorem 1
theorem parabola_circle_theorem (p : ℝ) :
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂) →
  p = 1/2 :=
sorry

-- Theorem 2
theorem trajectory_theorem (p : ℝ) (x y : ℝ) :
  p = 1/2 →
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂ ∧
    vector_equation x y x₁ y₁ x₂ y₂) →
  y^2 = x - 7/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_trajectory_theorem_l1696_169623


namespace NUMINAMATH_CALUDE_cosine_power_sum_l1696_169665

theorem cosine_power_sum (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) : 
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_sum_l1696_169665


namespace NUMINAMATH_CALUDE_prob_at_least_one_defective_l1696_169606

/-- The probability of selecting at least one defective bulb when randomly choosing two bulbs from a box containing 22 bulbs, of which 4 are defective. -/
theorem prob_at_least_one_defective (total : Nat) (defective : Nat) (h1 : total = 22) (h2 : defective = 4) :
  (1 : ℚ) - (total - defective) * (total - defective - 1) / (total * (total - 1)) = 26 / 77 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_defective_l1696_169606


namespace NUMINAMATH_CALUDE_cubic_symmetry_l1696_169686

/-- A cubic function of the form ax^3 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 6

/-- Theorem: For a cubic function f(x) = ax^3 + bx + 6, if f(5) = 7, then f(-5) = 5 -/
theorem cubic_symmetry (a b : ℝ) : f a b 5 = 7 → f a b (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_symmetry_l1696_169686


namespace NUMINAMATH_CALUDE_divisibility_by_ten_l1696_169699

theorem divisibility_by_ten (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b * c) % 10 = 0 ∧
  (a * b * d) % 10 = 0 ∧
  (a * b * e) % 10 = 0 ∧
  (a * c * d) % 10 = 0 ∧
  (a * c * e) % 10 = 0 ∧
  (a * d * e) % 10 = 0 ∧
  (b * c * d) % 10 = 0 ∧
  (b * c * e) % 10 = 0 ∧
  (b * d * e) % 10 = 0 ∧
  (c * d * e) % 10 = 0 →
  a % 10 = 0 ∨ b % 10 = 0 ∨ c % 10 = 0 ∨ d % 10 = 0 ∨ e % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_ten_l1696_169699


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1696_169600

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  · -- Prove that x = 1 satisfies the equation
    sorry
  · -- Prove that x = 1 is the only positive solution
    sorry

#check unique_positive_solution

end NUMINAMATH_CALUDE_unique_positive_solution_l1696_169600


namespace NUMINAMATH_CALUDE_divisor_problem_l1696_169608

theorem divisor_problem (d : ℕ+) : 
  (∃ n : ℕ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1696_169608


namespace NUMINAMATH_CALUDE_no_charming_numbers_l1696_169691

def is_charming (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a + b^3

theorem no_charming_numbers : ¬∃ (n : ℕ), is_charming n :=
sorry

end NUMINAMATH_CALUDE_no_charming_numbers_l1696_169691


namespace NUMINAMATH_CALUDE_olivers_earnings_theorem_l1696_169663

/-- Calculates the earnings of Oliver's laundry shop over three days -/
def olivers_earnings (price_per_kilo : ℝ) (day1_kilos : ℝ) (day2_increase : ℝ) : ℝ :=
  let day2_kilos := day1_kilos + day2_increase
  let day3_kilos := 2 * day2_kilos
  let total_kilos := day1_kilos + day2_kilos + day3_kilos
  price_per_kilo * total_kilos

/-- Theorem stating that Oliver's earnings for three days equal $70 -/
theorem olivers_earnings_theorem :
  olivers_earnings 2 5 5 = 70 := by
  sorry

#eval olivers_earnings 2 5 5

end NUMINAMATH_CALUDE_olivers_earnings_theorem_l1696_169663


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_implies_M_64_l1696_169622

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2 / 16 - y^2 / 25 = 1
def hyperbola2 (x y M : ℝ) : Prop := y^2 / 100 - x^2 / M = 1

-- Define the asymptotes of the hyperbolas
def asymptote1 (x y : ℝ) : Prop := y = (5/4) * x ∨ y = -(5/4) * x
def asymptote2 (x y M : ℝ) : Prop := y = (10/Real.sqrt M) * x ∨ y = -(10/Real.sqrt M) * x

-- Theorem statement
theorem hyperbolas_same_asymptotes_implies_M_64 :
  ∀ M : ℝ, (∀ x y : ℝ, asymptote1 x y ↔ asymptote2 x y M) → M = 64 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_implies_M_64_l1696_169622


namespace NUMINAMATH_CALUDE_excircle_radii_theorem_l1696_169668

/-- Given a triangle ABC with side lengths a, b, c and excircle radii r_a, r_b, r_c -/
theorem excircle_radii_theorem (a b c r_a r_b r_c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ra : r_a > 0) (h_pos_rb : r_b > 0) (h_pos_rc : r_c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_excircle_a : r_a = (a + b + c) * (a + b - c) / (4 * b * c))
  (h_excircle_b : r_b = (a + b + c) * (b + c - a) / (4 * a * c))
  (h_excircle_c : r_c = (a + b + c) * (c + a - b) / (4 * a * b)) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
sorry

end NUMINAMATH_CALUDE_excircle_radii_theorem_l1696_169668


namespace NUMINAMATH_CALUDE_prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l1696_169615

/-- Represents the total number of intermediate stations -/
def total_stations : ℕ := 7

/-- Represents the number of stations in Wuhan's jurisdiction -/
def wuhan_stations : ℕ := 4

/-- Represents the number of stations in Xiaogan's jurisdiction -/
def xiaogan_stations : ℕ := 3

/-- Represents the number of stations to be selected for research -/
def selected_stations : ℕ := 2

/-- Theorem for the probability of selecting one station from each jurisdiction -/
theorem prob_one_from_each_jurisdiction :
  (total_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  (wuhan_stations * xiaogan_stations : ℚ) / (total_stations.choose selected_stations) := by sorry

/-- Theorem for the probability of selecting at least one station within Xiaogan's jurisdiction -/
theorem prob_at_least_one_from_xiaogan :
  1 - (wuhan_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  5 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l1696_169615


namespace NUMINAMATH_CALUDE_divisor_problem_l1696_169674

theorem divisor_problem (dividend : ℤ) (quotient : ℤ) (remainder : ℤ) (divisor : ℤ) : 
  dividend = 151 ∧ quotient = 11 ∧ remainder = -4 →
  divisor = 14 ∧ dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1696_169674


namespace NUMINAMATH_CALUDE_difference_of_squares_75_25_l1696_169616

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_25_l1696_169616


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1696_169694

def p (x y : ℝ) : Prop := (x - 2) * (y - 5) ≠ 0

def q (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 5

theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1696_169694


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l1696_169679

theorem sqrt_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l1696_169679


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_on_1_5_l1696_169654

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_f_on_1_5 :
  (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_on_1_5_l1696_169654


namespace NUMINAMATH_CALUDE_donut_selection_problem_l1696_169614

theorem donut_selection_problem :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 56 := by
sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l1696_169614


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1696_169609

-- Define the isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

-- Define the conditions of the problem
def triangle : IsoscelesTriangle :=
  { side1 := 6,
    side2 := 8,
    area := 12 }

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) 
  (h1 : t = triangle) : 
  (2 * t.side1 + t.side2 = 20) ∨ (2 * t.side2 + t.side1 = 20) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1696_169609


namespace NUMINAMATH_CALUDE_unique_f_three_l1696_169651

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_f_three (f : RealFunction) 
  (h : ∀ x y : ℝ, f x * f y - f (x + y) = x - y) : 
  f 3 = -3 := by sorry

end NUMINAMATH_CALUDE_unique_f_three_l1696_169651


namespace NUMINAMATH_CALUDE_cube_sum_difference_l1696_169639

/-- Represents a face of a cube --/
inductive Face
| One
| Two
| Three
| Four
| Five
| Six

/-- A single small cube with numbered faces --/
structure SmallCube where
  faces : List Face
  face_count : faces.length = 6
  opposite_faces : 
    (Face.One ∈ faces ↔ Face.Two ∈ faces) ∧
    (Face.Three ∈ faces ↔ Face.Five ∈ faces) ∧
    (Face.Four ∈ faces ↔ Face.Six ∈ faces)

/-- The large 2×2×2 cube composed of small cubes --/
structure LargeCube where
  small_cubes : List SmallCube
  cube_count : small_cubes.length = 8

/-- The sum of numbers on the outer surface of the large cube --/
def outer_surface_sum (lc : LargeCube) : ℕ := sorry

/-- The maximum possible sum of numbers on the outer surface --/
def max_sum (lc : LargeCube) : ℕ := sorry

/-- The minimum possible sum of numbers on the outer surface --/
def min_sum (lc : LargeCube) : ℕ := sorry

/-- The main theorem to prove --/
theorem cube_sum_difference (lc : LargeCube) : 
  max_sum lc - min_sum lc = 24 := by sorry

end NUMINAMATH_CALUDE_cube_sum_difference_l1696_169639


namespace NUMINAMATH_CALUDE_four_digit_number_property_l1696_169670

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_valid_n (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 10 % 10 ≠ 0)

def split_n (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

theorem four_digit_number_property (n : ℕ) 
  (h1 : is_valid_n n) 
  (h2 : let (A, B) := split_n n; is_two_digit A ∧ is_two_digit B)
  (h3 : let (A, B) := split_n n; n % (A * B) = 0) :
  n = 1734 ∨ n = 1352 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l1696_169670


namespace NUMINAMATH_CALUDE_root_distance_range_l1696_169675

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem root_distance_range (ha : a ≠ 0) 
  (hsum : a + b + c = 0) 
  (hf : f 0 * f 1 > 0) 
  (hx₁ : f x₁ = 0) 
  (hx₂ : f x₂ = 0) 
  (hx_distinct : x₁ ≠ x₂) :
  |x₁ - x₂| ∈ Set.Icc (Real.sqrt 3 / 3) (2 / 3) :=
sorry

end NUMINAMATH_CALUDE_root_distance_range_l1696_169675


namespace NUMINAMATH_CALUDE_playground_insects_l1696_169613

def remaining_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem playground_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_playground_insects_l1696_169613


namespace NUMINAMATH_CALUDE_work_completion_time_l1696_169684

/-- Given that:
  - A can do a work in 9 days
  - A and B together can do the work in 6 days
  Prove that B can do the work alone in 18 days -/
theorem work_completion_time (a b : ℝ) (ha : a = 9) (hab : 1 / a + 1 / b = 1 / 6) : b = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1696_169684


namespace NUMINAMATH_CALUDE_room_width_calculation_l1696_169666

/-- Given a room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l1696_169666


namespace NUMINAMATH_CALUDE_officials_selection_count_l1696_169633

/-- Represents the number of ways to choose officials from a club --/
def choose_officials (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  girls * boys * (boys - 1)

/-- Theorem: The number of ways to choose officials under given conditions is 1716 --/
theorem officials_selection_count :
  choose_officials 25 12 13 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_officials_selection_count_l1696_169633


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l1696_169642

theorem ancient_chinese_math_problem (x y : ℕ) : 
  (8 * x = y + 3) → (7 * x = y - 4) → ((y + 3) / 8 : ℚ) = ((y - 4) / 7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l1696_169642


namespace NUMINAMATH_CALUDE_vanessa_score_l1696_169655

/-- Calculates Vanessa's score in a basketball game -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) : 
  total_score = 72 → other_players = 7 → avg_score = 6 →
  total_score - (other_players * avg_score) = 30 := by
sorry

end NUMINAMATH_CALUDE_vanessa_score_l1696_169655


namespace NUMINAMATH_CALUDE_p_less_than_q_l1696_169619

theorem p_less_than_q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
sorry

end NUMINAMATH_CALUDE_p_less_than_q_l1696_169619


namespace NUMINAMATH_CALUDE_overtime_threshold_is_40_l1696_169628

/-- Represents Janet's work and financial situation -/
structure JanetWorkSituation where
  regularRate : ℝ  -- Regular hourly rate
  weeklyHours : ℝ  -- Total weekly work hours
  overtimeMultiplier : ℝ  -- Overtime pay multiplier
  carCost : ℝ  -- Cost of the car
  weeksToSave : ℝ  -- Number of weeks to save for the car

/-- Calculates the weekly earnings given a threshold for overtime hours -/
def weeklyEarnings (j : JanetWorkSituation) (threshold : ℝ) : ℝ :=
  threshold * j.regularRate + (j.weeklyHours - threshold) * j.regularRate * j.overtimeMultiplier

/-- Theorem stating that the overtime threshold is 40 hours -/
theorem overtime_threshold_is_40 (j : JanetWorkSituation) 
    (h1 : j.regularRate = 20)
    (h2 : j.weeklyHours = 52)
    (h3 : j.overtimeMultiplier = 1.5)
    (h4 : j.carCost = 4640)
    (h5 : j.weeksToSave = 4)
    : ∃ (threshold : ℝ), 
      threshold = 40 ∧ 
      weeklyEarnings j threshold ≥ j.carCost / j.weeksToSave ∧
      ∀ t, t > threshold → weeklyEarnings j t < j.carCost / j.weeksToSave :=
by
  sorry

end NUMINAMATH_CALUDE_overtime_threshold_is_40_l1696_169628


namespace NUMINAMATH_CALUDE_sum_of_distances_l1696_169604

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is 2√5 + √130 -/
theorem sum_of_distances (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 5) → D = (4, 3) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2 * Real.sqrt 5 + Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l1696_169604


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1696_169631

/-- Given a quadratic equation x^2 - (m+1)x + 2m = 0 where 3 is a root,
    and an isosceles triangle ABC where two sides have lengths equal to the roots of the equation,
    prove that the perimeter of the triangle is either 10 or 11. -/
theorem isosceles_triangle_perimeter (m : ℝ) :
  (3^2 - (m+1)*3 + 2*m = 0) →
  ∃ (a b : ℝ), (a^2 - (m+1)*a + 2*m = 0) ∧ (b^2 - (m+1)*b + 2*m = 0) ∧ 
  ((a + a + b = 10) ∨ (a + a + b = 11) ∨ (b + b + a = 10) ∨ (b + b + a = 11)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1696_169631


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1696_169647

-- Define the conditions
def p (m : ℝ) : Prop := -2 < m ∧ m < -1

def q (m : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
  2 + m > 0 ∧ m + 1 < 0

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ 
  (∃ m : ℝ, p m ∧ ¬q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1696_169647


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l1696_169688

theorem least_four_digit_multiple : ∀ n : ℕ,
  (1000 ≤ n) →
  (n % 3 = 0) →
  (n % 4 = 0) →
  (n % 9 = 0) →
  1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l1696_169688


namespace NUMINAMATH_CALUDE_inequality_proof_l1696_169678

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) > Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1696_169678


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1696_169632

theorem inverse_variation_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k) 
  (h4 : 2^3 * 8 = x^3 * 512) : x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1696_169632


namespace NUMINAMATH_CALUDE_smallest_consecutive_odd_divisibility_l1696_169629

theorem smallest_consecutive_odd_divisibility (n : ℕ+) :
  ∃ (u_n : ℕ+),
    (∀ (d : ℕ+) (a : ℕ),
      (∀ k : Fin u_n, ∃ m : ℕ, a + 2 * k.val = d * m) →
      (∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    (∀ (v : ℕ+),
      v < u_n →
      ∃ (d : ℕ+) (a : ℕ),
        (∀ k : Fin v, ∃ m : ℕ, a + 2 * k.val = d * m) ∧
        ¬(∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    u_n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_odd_divisibility_l1696_169629


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l1696_169667

theorem pet_store_siamese_cats 
  (total_cats : ℕ) 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (sold_cats : ℕ) 
  (remaining_cats : ℕ) :
  house_cats = 49 →
  sold_cats = 19 →
  remaining_cats = 45 →
  total_cats = siamese_cats + house_cats →
  total_cats = remaining_cats + sold_cats →
  siamese_cats = 15 := by
sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l1696_169667


namespace NUMINAMATH_CALUDE_remainder_sum_mod_nine_l1696_169635

theorem remainder_sum_mod_nine (a b c : ℕ) : 
  0 < a ∧ a < 10 ∧
  0 < b ∧ b < 10 ∧
  0 < c ∧ c < 10 ∧
  (a * b * c) % 9 = 1 ∧
  (4 * c) % 9 = 5 ∧
  (7 * b) % 9 = (4 + b) % 9 →
  (a + b + c) % 9 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_nine_l1696_169635


namespace NUMINAMATH_CALUDE_log_50_between_integers_l1696_169660

theorem log_50_between_integers : ∃ c d : ℤ, (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < (d : ℝ) ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_50_between_integers_l1696_169660


namespace NUMINAMATH_CALUDE_triangle_altitude_after_base_extension_l1696_169602

theorem triangle_altitude_after_base_extension (area : ℝ) (new_base : ℝ) (h : area = 800) (h_base : new_base = 50) :
  let new_altitude := 2 * area / new_base
  new_altitude = 32 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_after_base_extension_l1696_169602


namespace NUMINAMATH_CALUDE_power_twenty_equals_R_S_l1696_169696

theorem power_twenty_equals_R_S (a b : ℤ) (R S : ℝ) 
  (hR : R = (4 : ℝ) ^ a) 
  (hS : S = (5 : ℝ) ^ b) : 
  (20 : ℝ) ^ (a * b) = R ^ b * S ^ a := by sorry

end NUMINAMATH_CALUDE_power_twenty_equals_R_S_l1696_169696


namespace NUMINAMATH_CALUDE_circle_center_l1696_169637

/-- The center of a circle with equation x^2 - 8x + y^2 - 4y = 16 is (4, 2) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 8*x + y^2 - 4*y = 16) → 
  (∃ r : ℝ, (x - 4)^2 + (y - 2)^2 = r^2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l1696_169637


namespace NUMINAMATH_CALUDE_sin_450_degrees_l1696_169612

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l1696_169612
