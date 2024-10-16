import Mathlib

namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1107_110743

/-- If (3sin(α) + 2cos(α)) / (2sin(α) - cos(α)) = 8/3, then tan(α + π/4) = -3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/3) : 
  Real.tan (α + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1107_110743


namespace NUMINAMATH_CALUDE_angle_C_measure_l1107_110717

/-- Given a triangle ABC where sin²A - sin²C = (sin A - sin B) sin B, prove that the measure of angle C is π/3 -/
theorem angle_C_measure (A B C : ℝ) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 - Real.sin C ^ 2 = (Real.sin A - Real.sin B) * Real.sin B) : 
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1107_110717


namespace NUMINAMATH_CALUDE_inequality_proof_l1107_110714

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_condition : a + b + c < 2) : 
  Real.sqrt (a^2 + b*c) + Real.sqrt (b^2 + c*a) + Real.sqrt (c^2 + a*b) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1107_110714


namespace NUMINAMATH_CALUDE_horner_method_v3_l1107_110745

def horner_polynomial (x : ℤ) : ℤ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 ∧
  horner_polynomial (-4) = ((((horner_v3 (-4) * (-4) - 8) * (-4) + 25) * (-4)) + 10) :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1107_110745


namespace NUMINAMATH_CALUDE_investment_difference_l1107_110706

def emma_investment : ℝ := 300
def briana_investment : ℝ := 500
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def time_period : ℝ := 2

theorem investment_difference :
  briana_investment * briana_yield_rate * time_period - 
  emma_investment * emma_yield_rate * time_period = 10 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l1107_110706


namespace NUMINAMATH_CALUDE_x_range_l1107_110727

def p (x : ℝ) : Prop := x^2 - 4*x + 3 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem x_range (x : ℝ) :
  (∀ y : ℝ, ¬(p y ∧ q y)) ∧ (∃ y : ℝ, p y ∨ q y) →
  ((1 < x ∧ x ≤ 2) ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l1107_110727


namespace NUMINAMATH_CALUDE_fraction_problem_l1107_110788

theorem fraction_problem (p q : ℚ) : 
  p = 4 → 
  (1 : ℚ)/7 + (2*q - p)/(2*q + p) = 0.5714285714285714 → 
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1107_110788


namespace NUMINAMATH_CALUDE_student_turtle_difference_is_85_l1107_110769

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 20

/-- The number of pet turtles in each fourth-grade classroom -/
def turtles_per_classroom : ℕ := 3

/-- The difference between the total number of students and the total number of turtles -/
def student_turtle_difference : ℕ :=
  num_classrooms * students_per_classroom - num_classrooms * turtles_per_classroom

theorem student_turtle_difference_is_85 : student_turtle_difference = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_turtle_difference_is_85_l1107_110769


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1107_110738

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 3 ∧ x₂ = 3 - Real.sqrt 3 ∧
    x₁^2 - 6*x₁ + 6 = 0 ∧ x₂^2 - 6*x₂ + 6 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 5 ∧
    (x₁ - 1) * (x₁ - 3) = 8 ∧ (x₂ - 1) * (x₂ - 3) = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1107_110738


namespace NUMINAMATH_CALUDE_equation_solution_l1107_110736

theorem equation_solution :
  ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1107_110736


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1107_110722

theorem least_addition_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x ≤ 3 ∧ (1202 + x) % 4 = 0 ∧ ∀ (y : ℕ), y < x → (1202 + y) % 4 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1107_110722


namespace NUMINAMATH_CALUDE_T_bounds_l1107_110750

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 1 ∧ y = (3*x + 5) / (x + 3)}

theorem T_bounds :
  ∃ (q P : ℝ),
    (∀ y ∈ T, q ≤ y) ∧
    (∀ y ∈ T, y ≤ P) ∧
    q ∈ T ∧
    P ∉ T :=
sorry

end NUMINAMATH_CALUDE_T_bounds_l1107_110750


namespace NUMINAMATH_CALUDE_pages_used_l1107_110792

def cards_per_page : ℕ := 3
def new_cards : ℕ := 3
def old_cards : ℕ := 9

theorem pages_used :
  (new_cards + old_cards) / cards_per_page = 4 :=
by sorry

end NUMINAMATH_CALUDE_pages_used_l1107_110792


namespace NUMINAMATH_CALUDE_smallest_valid_number_proof_l1107_110799

/-- Checks if a natural number contains all digits from 0 to 9 --/
def containsAllDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

/-- The smallest 12-digit number divisible by 36 containing all digits --/
def smallestValidNumber : ℕ := 100023457896

theorem smallest_valid_number_proof :
  (smallestValidNumber ≥ 10^11) ∧ 
  (smallestValidNumber < 10^12) ∧
  (smallestValidNumber % 36 = 0) ∧
  containsAllDigits smallestValidNumber ∧
  ∀ m : ℕ, m ≥ 10^11 ∧ m < 10^12 ∧ m % 36 = 0 ∧ containsAllDigits m → m ≥ smallestValidNumber :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_proof_l1107_110799


namespace NUMINAMATH_CALUDE_min_garden_cost_l1107_110754

/-- Represents the dimensions of a rectangular region -/
structure Region where
  length : ℝ
  width : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The main theorem stating the minimum cost of the garden -/
theorem min_garden_cost (regions : List Region) (flowers : List Flower) : 
  regions.length = 5 →
  flowers.length = 5 →
  regions = [
    ⟨5, 2⟩, 
    ⟨7, 3⟩, 
    ⟨5, 5⟩, 
    ⟨2, 4⟩, 
    ⟨5, 4⟩
  ] →
  flowers = [
    ⟨"Marigold", 1⟩,
    ⟨"Sunflower", 1.75⟩,
    ⟨"Tulip", 1.25⟩,
    ⟨"Orchid", 2.75⟩,
    ⟨"Iris", 3.25⟩
  ] →
  ∃ (assignment : List (Flower × Region)), 
    assignment.length = 5 ∧ 
    (∀ f r, (f, r) ∈ assignment → f ∈ flowers ∧ r ∈ regions) ∧
    (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ assignment) ∧
    (∀ r, r ∈ regions → ∃! f, (f, r) ∈ assignment) ∧
    (assignment.map (λ (f, r) => plantingCost f r)).sum = 140.75 ∧
    ∀ (other_assignment : List (Flower × Region)),
      other_assignment.length = 5 →
      (∀ f r, (f, r) ∈ other_assignment → f ∈ flowers ∧ r ∈ regions) →
      (∀ f, f ∈ flowers → ∃! r, (f, r) ∈ other_assignment) →
      (∀ r, r ∈ regions → ∃! f, (f, r) ∈ other_assignment) →
      (other_assignment.map (λ (f, r) => plantingCost f r)).sum ≥ 140.75 :=
by sorry

end NUMINAMATH_CALUDE_min_garden_cost_l1107_110754


namespace NUMINAMATH_CALUDE_bracelet_price_is_4_l1107_110762

/-- The price of a bracelet in dollars -/
def bracelet_price : ℝ := sorry

/-- The price of a keychain in dollars -/
def keychain_price : ℝ := 5

/-- The price of a coloring book in dollars -/
def coloring_book_price : ℝ := 3

/-- The total cost of the purchases -/
def total_cost : ℝ := 20

theorem bracelet_price_is_4 :
  2 * bracelet_price + keychain_price + bracelet_price + coloring_book_price = total_cost →
  bracelet_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_price_is_4_l1107_110762


namespace NUMINAMATH_CALUDE_select_defective_products_l1107_110729

def total_products : ℕ := 100
def defective_products : ℕ := 6
def products_to_select : ℕ := 3

theorem select_defective_products :
  Nat.choose total_products products_to_select -
  Nat.choose (total_products - defective_products) products_to_select =
  Nat.choose total_products products_to_select -
  Nat.choose 94 products_to_select :=
by sorry

end NUMINAMATH_CALUDE_select_defective_products_l1107_110729


namespace NUMINAMATH_CALUDE_tangent_line_parabola_l1107_110795

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_parabola : 
  ∃ d : ℝ, (∀ x y : ℝ, y = 3*x + d ∧ y^2 = 12*x → 
    ∃! x₀ : ℝ, 3*x₀ + d = (12*x₀).sqrt ∧ 
    ∀ x : ℝ, x ≠ x₀ → 3*x + d ≠ (12*x).sqrt) → 
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_l1107_110795


namespace NUMINAMATH_CALUDE_smallest_value_satisfying_equation_l1107_110708

theorem smallest_value_satisfying_equation :
  ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), (⌊y⌋ = 3 + 50 * (y - ⌊y⌋)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_satisfying_equation_l1107_110708


namespace NUMINAMATH_CALUDE_square_difference_263_257_l1107_110780

theorem square_difference_263_257 : 263^2 - 257^2 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_263_257_l1107_110780


namespace NUMINAMATH_CALUDE_binomial_coefficient_21_15_l1107_110793

theorem binomial_coefficient_21_15 
  (h1 : Nat.choose 20 13 = 77520)
  (h2 : Nat.choose 20 14 = 38760)
  (h3 : Nat.choose 22 15 = 203490) :
  Nat.choose 21 15 = 87210 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_21_15_l1107_110793


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l1107_110781

theorem square_perimeter_from_area (area : ℝ) (perimeter : ℝ) :
  area = 225 → perimeter = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l1107_110781


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l1107_110782

/-- Given two points A(-3,m) and B(m,5), and a line parallel to 3x+y-1=0, prove m = -7 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-3, m)
  let B : ℝ × ℝ := (m, 5)
  let parallel_line_slope : ℝ := -3
  (B.2 - A.2) / (B.1 - A.1) = parallel_line_slope →
  m = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l1107_110782


namespace NUMINAMATH_CALUDE_range_of_a_l1107_110734

theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x > a}
  let B := {x : ℝ | x > 6}
  A ⊆ B ↔ a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1107_110734


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l1107_110783

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l1107_110783


namespace NUMINAMATH_CALUDE_new_group_average_age_l1107_110718

theorem new_group_average_age 
  (initial_count : ℕ) 
  (initial_avg : ℚ) 
  (new_count : ℕ) 
  (final_avg : ℚ) :
  initial_count = 20 →
  initial_avg = 16 →
  new_count = 20 →
  final_avg = 15.5 →
  (initial_count * initial_avg + new_count * (initial_count * final_avg - initial_count * initial_avg) / new_count) / (initial_count + new_count) = 15 :=
by sorry

end NUMINAMATH_CALUDE_new_group_average_age_l1107_110718


namespace NUMINAMATH_CALUDE_books_in_box_l1107_110768

def box_weight : ℕ := 42
def book_weight : ℕ := 3

theorem books_in_box : 
  box_weight / book_weight = 14 := by sorry

end NUMINAMATH_CALUDE_books_in_box_l1107_110768


namespace NUMINAMATH_CALUDE_race_start_theorem_l1107_110779

/-- Represents the start distance one runner can give another in a kilometer race -/
def start_distance (runner1 runner2 : ℕ) : ℝ := sorry

/-- The race distance in meters -/
def race_distance : ℝ := 1000

theorem race_start_theorem (A B C : ℕ) :
  start_distance A B = 50 →
  start_distance B C = 52.63157894736844 →
  start_distance A C = 100 := by
  sorry

end NUMINAMATH_CALUDE_race_start_theorem_l1107_110779


namespace NUMINAMATH_CALUDE_test_failure_rate_l1107_110786

/-- The percentage of students who failed a test, given the number of boys and girls
    and their respective pass rates. -/
def percentageFailed (numBoys numGirls : ℕ) (boyPassRate girlPassRate : ℚ) : ℚ :=
  let totalStudents := numBoys + numGirls
  let failedStudents := numBoys * (1 - boyPassRate) + numGirls * (1 - girlPassRate)
  failedStudents / totalStudents

/-- Theorem stating that given 50 boys and 100 girls, with 50% of boys passing
    and 40% of girls passing, the percentage of total students who failed is 56.67%. -/
theorem test_failure_rate : 
  percentageFailed 50 100 (1/2) (2/5) = 8500/15000 := by
  sorry

end NUMINAMATH_CALUDE_test_failure_rate_l1107_110786


namespace NUMINAMATH_CALUDE_defective_units_shipped_l1107_110758

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ)
  (h1 : defective_rate = 0.09)
  (h2 : shipped_rate = 0.04) :
  (defective_rate * shipped_rate * 100) = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l1107_110758


namespace NUMINAMATH_CALUDE_zoo_layout_problem_l1107_110753

/-- The number of tiger enclosures in a zoo -/
def tigerEnclosures : ℕ := sorry

/-- The number of zebra enclosures in the zoo -/
def zebraEnclosures : ℕ := 2 * tigerEnclosures

/-- The number of giraffe enclosures in the zoo -/
def giraffeEnclosures : ℕ := 3 * zebraEnclosures

/-- The number of tigers per tiger enclosure -/
def tigersPerEnclosure : ℕ := 4

/-- The number of zebras per zebra enclosure -/
def zebrasPerEnclosure : ℕ := 10

/-- The number of giraffes per giraffe enclosure -/
def giraffesPerEnclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def totalAnimals : ℕ := 144

theorem zoo_layout_problem :
  tigerEnclosures * tigersPerEnclosure +
  zebraEnclosures * zebrasPerEnclosure +
  giraffeEnclosures * giraffesPerEnclosure = totalAnimals ∧
  tigerEnclosures = 4 := by sorry

end NUMINAMATH_CALUDE_zoo_layout_problem_l1107_110753


namespace NUMINAMATH_CALUDE_existence_of_m_and_n_l1107_110724

theorem existence_of_m_and_n :
  ∃ (m n : ℕ) (a b : ℝ), (-2 * a^n * b^n)^m + (3 * a^m * b^m)^n = a^6 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_and_n_l1107_110724


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l1107_110709

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 8062 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l1107_110709


namespace NUMINAMATH_CALUDE_temperature_range_l1107_110774

/-- Given the highest and lowest temperatures on a certain day, 
    prove that the range of temperature change is between these two values, inclusive. -/
theorem temperature_range (highest lowest t : ℝ) 
  (h_highest : highest = 26) 
  (h_lowest : lowest = 12) 
  (h_range : lowest ≤ t ∧ t ≤ highest) : 
  12 ≤ t ∧ t ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_temperature_range_l1107_110774


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l1107_110776

/-- Represents the price in cents for a pack of oranges -/
structure PackPrice :=
  (quantity : ℕ)
  (price : ℕ)

/-- Calculates the total cost for a given number of packs -/
def totalCost (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.price * numPacks

/-- Calculates the total number of oranges for a given number of packs -/
def totalOranges (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.quantity * numPacks

theorem orange_pricing_theorem (pack1 pack2 : PackPrice) 
    (h1 : pack1 = ⟨4, 15⟩)
    (h2 : pack2 = ⟨6, 25⟩)
    (h3 : totalOranges pack1 5 + totalOranges pack2 5 = 20) :
  (totalCost pack1 5 + totalCost pack2 5) / 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l1107_110776


namespace NUMINAMATH_CALUDE_ab_sum_problem_l1107_110747

theorem ab_sum_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_upper : a < 15) (hb_upper : b < 15) 
  (h_eq : a + b + a * b = 119) : a + b = 18 ∨ a + b = 19 := by
  sorry

end NUMINAMATH_CALUDE_ab_sum_problem_l1107_110747


namespace NUMINAMATH_CALUDE_isabelle_concert_savings_l1107_110741

/-- Calculates the number of weeks Isabelle must work to afford concert tickets for herself and her brothers. -/
theorem isabelle_concert_savings (isabelle_ticket : ℕ) (brother_ticket : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_earnings : ℕ) : 
  isabelle_ticket = 20 →
  brother_ticket = 10 →
  isabelle_savings = 5 →
  brothers_savings = 5 →
  weekly_earnings = 3 →
  (isabelle_ticket + 2 * brother_ticket - isabelle_savings - brothers_savings) / weekly_earnings = 10 := by
sorry

end NUMINAMATH_CALUDE_isabelle_concert_savings_l1107_110741


namespace NUMINAMATH_CALUDE_kevin_cards_l1107_110789

/-- The number of cards Kevin has at the end of the day -/
def final_cards (initial : ℕ) (found : ℕ) (lost1 : ℕ) (lost2 : ℕ) (won : ℕ) : ℕ :=
  initial + found - lost1 - lost2 + won

/-- Theorem stating that Kevin ends up with 63 cards given the problem conditions -/
theorem kevin_cards : final_cards 20 47 7 12 15 = 63 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l1107_110789


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l1107_110771

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter : ℕ → (ℤ → ℤ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

theorem exists_m_divisible_by_1997 : ∃ m : ℕ+, (1997 : ℤ) ∣ f_iter 99 m.val := by
  sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1997_l1107_110771


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l1107_110711

theorem multiplication_mistake_difference : 
  let correct_multiplication := 137 * 43
  let mistaken_multiplication := 137 * 34
  correct_multiplication - mistaken_multiplication = 1233 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l1107_110711


namespace NUMINAMATH_CALUDE_fourth_grade_students_left_l1107_110737

/-- The number of students who left during the year -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

theorem fourth_grade_students_left : students_left 11 42 47 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_left_l1107_110737


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l1107_110733

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 6) :
  Complex.abs w ^ 2 = 3.375 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l1107_110733


namespace NUMINAMATH_CALUDE_share_difference_l1107_110790

/-- Given four shares in the ratio 3:3:7:4, where the second share is 1500
    and the fourth share is 2000, the difference between the largest share
    and the second-largest share is 1500. -/
theorem share_difference (shares : Fin 4 → ℕ) : 
  (∃ x : ℕ, shares 0 = 3*x ∧ shares 1 = 3*x ∧ shares 2 = 7*x ∧ shares 3 = 4*x) →
  shares 1 = 1500 →
  shares 3 = 2000 →
  (shares 2 - (max (shares 0) (shares 3))) = 1500 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l1107_110790


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l1107_110797

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h_neg2 : f (-2) = 0)
  (h_1 : f 1 = -2)
  (h_3 : f 3 = -6)
  (h_5 : f 5 = -10) :
  f 0 = 29 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l1107_110797


namespace NUMINAMATH_CALUDE_product_of_numbers_l1107_110712

theorem product_of_numbers (x y : ℝ) : x + y = 30 → x^2 + y^2 = 840 → x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1107_110712


namespace NUMINAMATH_CALUDE_count_words_to_1000_l1107_110798

def word_count_1_to_99 : Nat := 171

def word_count_100_to_999 : Nat := 486 + 1944

def word_count_1000 : Nat := 37

theorem count_words_to_1000 :
  word_count_1_to_99 + word_count_100_to_999 + word_count_1000 = 2611 :=
by sorry

end NUMINAMATH_CALUDE_count_words_to_1000_l1107_110798


namespace NUMINAMATH_CALUDE_alex_earnings_l1107_110705

/-- Alex's work hours and earnings problem -/
theorem alex_earnings (hours_week3 : ℕ) (hours_difference : ℕ) (earnings_difference : ℕ) :
  hours_week3 = 28 →
  hours_difference = 10 →
  earnings_difference = 80 →
  (hours_week3 - hours_difference) * (earnings_difference / hours_difference) +
  hours_week3 * (earnings_difference / hours_difference) = 368 := by
  sorry

end NUMINAMATH_CALUDE_alex_earnings_l1107_110705


namespace NUMINAMATH_CALUDE_odd_even_function_sum_l1107_110730

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f g : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even g)
  (h_diff : ∀ x, f x - g x = 2 * x^3 + x^2 + 3) :
  f 2 + g 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_odd_even_function_sum_l1107_110730


namespace NUMINAMATH_CALUDE_order_of_expressions_l1107_110713

theorem order_of_expressions : 
  let a : ℝ := (4 : ℝ) ^ (1/10)
  let b : ℝ := Real.log 0.1 / Real.log 4
  let c : ℝ := (0.4 : ℝ) ^ (1/5)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l1107_110713


namespace NUMINAMATH_CALUDE_medium_stores_selected_is_ten_l1107_110728

/-- Represents the number of stores to be selected in a stratified sampling -/
def total_sample : ℕ := 30

/-- Represents the total number of stores -/
def total_stores : ℕ := 1500

/-- Represents the ratio of large stores -/
def large_ratio : ℕ := 1

/-- Represents the ratio of medium stores -/
def medium_ratio : ℕ := 5

/-- Represents the ratio of small stores -/
def small_ratio : ℕ := 9

/-- Calculates the number of medium-sized stores to be selected in the stratified sampling -/
def medium_stores_selected : ℕ := 
  (total_sample * medium_ratio) / (large_ratio + medium_ratio + small_ratio)

/-- Theorem stating that the number of medium-sized stores to be selected is 10 -/
theorem medium_stores_selected_is_ten : medium_stores_selected = 10 := by
  sorry


end NUMINAMATH_CALUDE_medium_stores_selected_is_ten_l1107_110728


namespace NUMINAMATH_CALUDE_intersection_M_N_l1107_110794

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 2}
def N : Set ℝ := {x : ℝ | x^2 - 25 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Icc 2 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1107_110794


namespace NUMINAMATH_CALUDE_subtraction_to_sum_equality_l1107_110775

theorem subtraction_to_sum_equality : 3 - 10 - 7 = 3 + (-10) + (-7) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_to_sum_equality_l1107_110775


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l1107_110726

theorem triangle_cosine_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (Real.cos (B-C) * Real.cos A + Real.cos (2*A) = 1 + Real.cos A * Real.cos (B+C)) →
  ((B = C → Real.cos A = 2/3) ∧ (b^2 + c^2) / a^2 = 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l1107_110726


namespace NUMINAMATH_CALUDE_apples_needed_for_pies_l1107_110740

theorem apples_needed_for_pies (pies_to_bake : ℕ) (apples_per_pie : ℕ) (apples_on_hand : ℕ) : 
  pies_to_bake * apples_per_pie - apples_on_hand = 110 :=
by
  sorry

#check apples_needed_for_pies 15 10 40

end NUMINAMATH_CALUDE_apples_needed_for_pies_l1107_110740


namespace NUMINAMATH_CALUDE_pigeon_increase_l1107_110748

theorem pigeon_increase (total : ℕ) (initial : ℕ) (h1 : total = 21) (h2 : initial = 15) :
  total - initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_increase_l1107_110748


namespace NUMINAMATH_CALUDE_chord_length_squared_l1107_110702

/-- Two circles with given properties and a line through their intersection point --/
structure TwoCirclesWithLine where
  /-- Radius of the first circle --/
  r₁ : ℝ
  /-- Radius of the second circle --/
  r₂ : ℝ
  /-- Distance between the centers of the circles --/
  d : ℝ
  /-- Length of the chord QP (equal to PR) --/
  x : ℝ

/-- Theorem stating the square of the chord length in the given configuration --/
theorem chord_length_squared (c : TwoCirclesWithLine)
  (h₁ : c.r₁ = 5)
  (h₂ : c.r₂ = 10)
  (h₃ : c.d = 16)
  (h₄ : c.x > 0) :
  c.x^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l1107_110702


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1107_110710

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 + 10*y = -75

-- Define the center and radius of the circle
def is_center_radius (a b r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center_radius a b r ∧ a + b + r = 3 + Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1107_110710


namespace NUMINAMATH_CALUDE_decimal_73_is_four_digits_in_base_4_l1107_110756

/-- Converts a decimal number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem stating that 73 in decimal is a four-digit number in base 4 -/
theorem decimal_73_is_four_digits_in_base_4 :
  (toBase4 73).length = 4 :=
sorry

end NUMINAMATH_CALUDE_decimal_73_is_four_digits_in_base_4_l1107_110756


namespace NUMINAMATH_CALUDE_cookie_circle_radius_l1107_110725

theorem cookie_circle_radius (x y : ℝ) :
  x^2 + y^2 + 36 = 6*x + 12*y →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 3^2 := by
sorry

end NUMINAMATH_CALUDE_cookie_circle_radius_l1107_110725


namespace NUMINAMATH_CALUDE_ticket_sales_income_l1107_110749

/-- Calculates the total income from ticket sales given the number of student and adult tickets sold and their respective prices. -/
def total_income (student_tickets : ℕ) (adult_tickets : ℕ) (student_price : ℚ) (adult_price : ℚ) : ℚ :=
  student_tickets * student_price + adult_tickets * adult_price

/-- Proves that the total income from selling 20 tickets, where 12 are student tickets at $2.00 each and 8 are adult tickets at $4.50 each, is equal to $60.00. -/
theorem ticket_sales_income :
  let student_tickets : ℕ := 12
  let adult_tickets : ℕ := 8
  let student_price : ℚ := 2
  let adult_price : ℚ := 9/2
  total_income student_tickets adult_tickets student_price adult_price = 60 := by
  sorry


end NUMINAMATH_CALUDE_ticket_sales_income_l1107_110749


namespace NUMINAMATH_CALUDE_f_at_negative_one_l1107_110785

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 16

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 5*x^3 + b*x^2 + 150*x + c

theorem f_at_negative_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c (-1) = -1347 := by sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l1107_110785


namespace NUMINAMATH_CALUDE_ways_A_to_C_via_B_l1107_110723

/-- The number of ways to get from point A to point B -/
def ways_AB : ℕ := 3

/-- The number of ways to get from point B to point C -/
def ways_BC : ℕ := 4

/-- The total number of ways to get from point A to point C via point B -/
def total_ways : ℕ := ways_AB * ways_BC

theorem ways_A_to_C_via_B : total_ways = 12 := by
  sorry

end NUMINAMATH_CALUDE_ways_A_to_C_via_B_l1107_110723


namespace NUMINAMATH_CALUDE_stating_max_books_borrowed_is_eight_l1107_110751

/-- Represents the maximum number of books borrowed by a single student -/
def max_books_borrowed (total_students : ℕ) 
                       (zero_book_students : ℕ) 
                       (one_book_students : ℕ) 
                       (two_book_students : ℕ) 
                       (avg_books_per_student : ℕ) : ℕ :=
  let total_books := total_students * avg_books_per_student
  let remaining_students := total_students - (zero_book_students + one_book_students + two_book_students)
  let accounted_books := one_book_students + 2 * two_book_students
  let remaining_books := total_books - accounted_books
  remaining_books - (3 * (remaining_students - 1))

/-- 
Theorem stating that given the conditions in the problem, 
the maximum number of books borrowed by a single student is 8.
-/
theorem max_books_borrowed_is_eight :
  max_books_borrowed 35 2 12 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_books_borrowed_is_eight_l1107_110751


namespace NUMINAMATH_CALUDE_share_ratio_l1107_110761

theorem share_ratio (total money : ℕ) (a_share : ℕ) (x : ℚ) :
  total = 600 →
  a_share = 240 →
  a_share = x * (total - a_share) →
  (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))))) = total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share - (2/3 : ℚ) * (a_share + (total - a_share)))))) →
  (a_share : ℚ) / (total - a_share) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l1107_110761


namespace NUMINAMATH_CALUDE_walter_at_zoo_l1107_110720

theorem walter_at_zoo (seal_time penguin_time elephant_time total_time : ℕ) 
  (h1 : penguin_time = 8 * seal_time)
  (h2 : elephant_time = 13)
  (h3 : total_time = 130)
  (h4 : seal_time + penguin_time + elephant_time = total_time) :
  seal_time = 13 := by
  sorry

end NUMINAMATH_CALUDE_walter_at_zoo_l1107_110720


namespace NUMINAMATH_CALUDE_solution_sets_correct_l1107_110744

-- Define the solution sets for each inequality
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 3/2}
def solution_set2 : Set ℝ := {x | x < 2 ∨ x ≥ 5}

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -2 * x^2 + x < -3
def inequality2 (x : ℝ) : Prop := (x + 1) / (x - 2) ≤ 2

-- Theorem stating that the solution sets are correct
theorem solution_sets_correct :
  (∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_correct_l1107_110744


namespace NUMINAMATH_CALUDE_value_k_std_dev_below_mean_value_two_std_dev_below_mean_l1107_110770

/-- For a normal distribution with mean μ and standard deviation σ,
    the value that is exactly k standard deviations less than the mean is μ - k * σ -/
theorem value_k_std_dev_below_mean (μ σ k : ℝ) :
  let value := μ - k * σ
  value = μ - k * σ := by sorry

/-- For a normal distribution with mean 15.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12.5 -/
theorem value_two_std_dev_below_mean :
  let μ : ℝ := 15.5
  let σ : ℝ := 1.5
  let k : ℝ := 2
  let value := μ - k * σ
  value = 12.5 := by sorry

end NUMINAMATH_CALUDE_value_k_std_dev_below_mean_value_two_std_dev_below_mean_l1107_110770


namespace NUMINAMATH_CALUDE_ukis_bakery_profit_l1107_110746

/-- Uki's Bakery Profit Calculation -/
theorem ukis_bakery_profit : 
  let cupcake_price : ℚ := 3/2
  let cookie_price : ℚ := 2
  let biscuit_price : ℚ := 1
  let daily_cupcakes : ℕ := 20
  let daily_cookies : ℕ := 10
  let daily_biscuits : ℕ := 20
  let cupcake_cost : ℚ := 3/4
  let cookie_cost : ℚ := 1
  let biscuit_cost : ℚ := 1/2
  let days : ℕ := 5

  let daily_earnings := 
    daily_cupcakes * cupcake_price + 
    daily_cookies * cookie_price + 
    daily_biscuits * biscuit_price

  let daily_expenses := 
    daily_cupcakes * cupcake_cost + 
    daily_cookies * cookie_cost + 
    daily_biscuits * biscuit_cost

  let daily_profit := daily_earnings - daily_expenses
  let total_profit := daily_profit * days

  total_profit = 175 := by sorry

end NUMINAMATH_CALUDE_ukis_bakery_profit_l1107_110746


namespace NUMINAMATH_CALUDE_investment_inconsistency_l1107_110760

theorem investment_inconsistency :
  ¬ ∃ (r x y : ℝ), 
    x + y = 10000 ∧ 
    x > y ∧ 
    y > 0 ∧ 
    0.05 * y = 6000 ∧ 
    r * x = 0.05 * y + 160 ∧ 
    r > 0 := by
  sorry

end NUMINAMATH_CALUDE_investment_inconsistency_l1107_110760


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l1107_110719

/-- Two points in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in ℝ² defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Check if a line is vertical -/
def isVertical (l : Line) : Prop :=
  l.p1.x = l.p2.x

/-- Two lines are parallel if they are both vertical or have the same slope -/
def areParallel (l1 l2 : Line) : Prop :=
  (isVertical l1 ∧ isVertical l2) ∨
  (¬isVertical l1 ∧ ¬isVertical l2 ∧
    (l1.p2.y - l1.p1.y) / (l1.p2.x - l1.p1.x) = (l2.p2.y - l2.p1.y) / (l2.p2.x - l2.p1.x))

theorem parallel_lines_x_value (x : ℝ) :
  let l1 : Line := { p1 := { x := -1, y := -2 }, p2 := { x := -1, y := 4 } }
  let l2 : Line := { p1 := { x := 2, y := 1 }, p2 := { x := x, y := 6 } }
  areParallel l1 l2 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l1107_110719


namespace NUMINAMATH_CALUDE_slope_zero_sufficient_not_necessary_l1107_110773

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a line passing through (-1, 1)
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 - 1 = m * (p.1 + 1)}

-- Define tangency
def IsTangent (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∩ Circle ∧ ∀ q : ℝ × ℝ, q ∈ l ∩ Circle → q = p

-- Theorem statement
theorem slope_zero_sufficient_not_necessary :
  (∃ l : Set (ℝ × ℝ), l = Line 0 ∧ IsTangent l) ∧
  (∃ l : Set (ℝ × ℝ), IsTangent l ∧ l ≠ Line 0) :=
sorry

end NUMINAMATH_CALUDE_slope_zero_sufficient_not_necessary_l1107_110773


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l1107_110715

theorem linear_function_decreasing (a b y₁ y₂ : ℝ) :
  a < 0 →
  y₁ = 2 * a * (-1) - b →
  y₂ = 2 * a * 2 - b →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l1107_110715


namespace NUMINAMATH_CALUDE_smallest_fraction_above_three_fifths_l1107_110757

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction where
  numerator : ℕ
  denominator : ℕ
  num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99
  den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99

/-- The fraction 3/5 -/
def three_fifths : ℚ := 3 / 5

/-- The theorem stating that 59/98 is the smallest fraction greater than 3/5
    with two-digit numerator and denominator -/
theorem smallest_fraction_above_three_fifths :
  ∀ f : TwoDigitFraction, (f.numerator : ℚ) / f.denominator > three_fifths →
    (59 : ℚ) / 98 ≤ (f.numerator : ℚ) / f.denominator :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_above_three_fifths_l1107_110757


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l1107_110704

/-- 
Given Stanley's and Carl's hourly lemonade sales rates and a fixed time period,
prove the difference in their total sales.
-/
theorem lemonade_sales_difference 
  (stanley_rate : ℕ) 
  (carl_rate : ℕ) 
  (time_period : ℕ) 
  (h1 : stanley_rate = 4)
  (h2 : carl_rate = 7)
  (h3 : time_period = 3) :
  carl_rate * time_period - stanley_rate * time_period = 9 := by
  sorry

#check lemonade_sales_difference

end NUMINAMATH_CALUDE_lemonade_sales_difference_l1107_110704


namespace NUMINAMATH_CALUDE_yankees_to_mets_ratio_l1107_110721

/-- Represents the number of fans for each baseball team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 390

/-- The theorem stating the ratio of NY Yankees fans to NY Mets fans -/
theorem yankees_to_mets_ratio (fc : FanCounts) : 
  fc.yankees = 156 ∧ fc.mets = 104 ∧ fc.red_sox = 130 →
  fc.yankees + fc.mets + fc.red_sox = total_fans →
  fc.mets * 5 = fc.red_sox * 4 →
  fc.yankees * 2 = fc.mets * 3 := by
  sorry

#check yankees_to_mets_ratio

end NUMINAMATH_CALUDE_yankees_to_mets_ratio_l1107_110721


namespace NUMINAMATH_CALUDE_olympic_torch_relay_schemes_l1107_110716

/-- The number of segments in the Olympic torch relay -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The number of choices for the first torchbearer -/
def first_choices : ℕ := 3

/-- The number of choices for the last torchbearer -/
def last_choices : ℕ := 2

/-- The number of choices for each middle segment -/
def middle_choices : ℕ := num_torchbearers

/-- The number of middle segments -/
def num_middle_segments : ℕ := num_segments - 2

/-- The total number of different relay schemes -/
def total_schemes : ℕ := first_choices * (middle_choices ^ num_middle_segments) * last_choices

theorem olympic_torch_relay_schemes :
  total_schemes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_olympic_torch_relay_schemes_l1107_110716


namespace NUMINAMATH_CALUDE_abc_sum_product_l1107_110777

theorem abc_sum_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^4 + b^4 + c^4 = 128) :
  a*b + b*c + c*a = -8 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_product_l1107_110777


namespace NUMINAMATH_CALUDE_initial_cookies_equality_l1107_110787

/-- The number of cookies Paco initially had -/
def initial_cookies : ℕ := sorry

/-- The number of cookies Paco gave to his friend -/
def cookies_given : ℕ := 14

/-- The number of cookies Paco ate -/
def cookies_eaten : ℕ := 10

/-- The number of cookies Paco had left after giving away and eating -/
def cookies_left : ℕ := 12

/-- Theorem stating that the initial number of cookies is equal to the sum of
    cookies given away, cookies eaten, and cookies left -/
theorem initial_cookies_equality :
  initial_cookies = cookies_given + cookies_eaten + cookies_left :=
sorry

end NUMINAMATH_CALUDE_initial_cookies_equality_l1107_110787


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1107_110752

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^10 = 9000) 
  (h2 : P * (1 + r)^11 = 9990) : 
  r = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1107_110752


namespace NUMINAMATH_CALUDE_abs_z_squared_l1107_110765

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z + |z| = 3 + 12i
def condition (z : ℂ) : Prop := z + Complex.abs z = 3 + 12 * Complex.I

-- Theorem statement
theorem abs_z_squared (h : condition z) : Complex.abs z ^ 2 = 650.25 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_l1107_110765


namespace NUMINAMATH_CALUDE_goose_egg_count_l1107_110784

/-- The number of goose eggs laid at a pond -/
def num_eggs : ℕ := 650

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 2/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 130

theorem goose_egg_count :
  (↑num_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction) : ℚ) = survived_year :=
sorry

end NUMINAMATH_CALUDE_goose_egg_count_l1107_110784


namespace NUMINAMATH_CALUDE_no_k_for_all_positive_quadratic_l1107_110759

theorem no_k_for_all_positive_quadratic : ¬∃ k : ℝ, ∀ x : ℝ, x^2 - (k - 4)*x - (k + 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_k_for_all_positive_quadratic_l1107_110759


namespace NUMINAMATH_CALUDE_probability_not_black_ball_l1107_110732

theorem probability_not_black_ball (white black red : ℕ) 
  (h_white : white = 8) 
  (h_black : black = 9) 
  (h_red : red = 3) : 
  (white + red) / (white + black + red : ℚ) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_black_ball_l1107_110732


namespace NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1107_110796

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d k : ℕ → ℕ) : Prop :=
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ n, a (n + 1) = a n + c) ∧
  (∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ ∀ n, a (k (n + 1)) = a (k n) * q) ∧
  k 1 = 1 ∧ k 2 = 2 ∧ k 3 = 6

theorem arithmetic_geometric_k4 (a : ℕ → ℝ) (d k : ℕ → ℕ) :
  arithmetic_geometric_sequence a d k → k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_k4_l1107_110796


namespace NUMINAMATH_CALUDE_correct_street_loss_percentage_l1107_110764

/-- The percentage of marbles lost into the street -/
def street_loss_percentage : ℝ := 60

/-- The initial number of marbles -/
def initial_marbles : ℕ := 100

/-- The final number of marbles after losses -/
def final_marbles : ℕ := 20

/-- Theorem stating the correct percentage of marbles lost into the street -/
theorem correct_street_loss_percentage :
  street_loss_percentage = 60 ∧
  final_marbles = (initial_marbles - initial_marbles * street_loss_percentage / 100) / 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_street_loss_percentage_l1107_110764


namespace NUMINAMATH_CALUDE_rectangle_area_l1107_110701

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = y^2 → 3 * w^2 = (3 * y^2) / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1107_110701


namespace NUMINAMATH_CALUDE_evaluate_expression_l1107_110778

theorem evaluate_expression : 
  Real.sqrt ((16^10 + 4^15) / (16^7 + 4^16 - 4^8)) = 2 * Real.sqrt 1025 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1107_110778


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l1107_110707

/-- A line with negative slope intersecting a circle is separate from another circle -/
theorem line_separate_from_circle (k : ℝ) (h_k : k < 0) : 
  ∃ (x y : ℝ), y = k * x ∧ (x + 3)^2 + (y + 2)^2 = 9 →
  ∀ (x y : ℝ), y = k * x → x^2 + (y - 2)^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l1107_110707


namespace NUMINAMATH_CALUDE_travel_time_ratio_l1107_110735

theorem travel_time_ratio : 
  let distance : ℝ := 252
  let original_time : ℝ := 6
  let new_speed : ℝ := 28
  let new_time : ℝ := distance / new_speed
  let original_speed : ℝ := distance / original_time
  new_time / original_time = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l1107_110735


namespace NUMINAMATH_CALUDE_consecutive_squares_divisors_l1107_110763

theorem consecutive_squares_divisors :
  ∃ (n : ℕ), 
    (∃ (a : ℕ), a > 1 ∧ a * a ∣ n) ∧
    (∃ (b : ℕ), b > 1 ∧ b * b ∣ (n + 1)) ∧
    (∃ (c : ℕ), c > 1 ∧ c * c ∣ (n + 2)) ∧
    (∃ (d : ℕ), d > 1 ∧ d * d ∣ (n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_divisors_l1107_110763


namespace NUMINAMATH_CALUDE_courtyard_breadth_l1107_110767

/-- Proves that the breadth of a rectangular courtyard is 6 meters -/
theorem courtyard_breadth : 
  ∀ (length width stone_length stone_width stone_count : ℝ),
  length = 15 →
  stone_count = 15 →
  stone_length = 3 →
  stone_width = 2 →
  length * width = stone_count * stone_length * stone_width →
  width = 6 := by
sorry

end NUMINAMATH_CALUDE_courtyard_breadth_l1107_110767


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1107_110742

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 12 and S_20 = 17, prove S_30 = 15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence)
    (h1 : a.S 10 = 12)
    (h2 : a.S 20 = 17) :
  a.S 30 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1107_110742


namespace NUMINAMATH_CALUDE_unique_plane_through_line_and_point_l1107_110739

-- Define the 3D space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable [Fact (finrank ℝ V = 3)]

-- Define a line in 3D space
def Line (p q : V) : Set V :=
  {x | ∃ t : ℝ, x = p + t • (q - p)}

-- Define a plane in 3D space
def Plane (n : V) (c : ℝ) : Set V :=
  {x | inner n x = c}

-- State the theorem
theorem unique_plane_through_line_and_point 
  (l : Set V) (A : V) (p q : V) (h_line : l = Line p q) (h_not_on : A ∉ l) :
  ∃! P : Set V, ∃ n : V, ∃ c : ℝ, 
    P = Plane n c ∧ l ⊆ P ∧ A ∈ P :=
sorry

end NUMINAMATH_CALUDE_unique_plane_through_line_and_point_l1107_110739


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l1107_110791

theorem angle_inequality_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0) ↔
  (0 < θ ∧ θ < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l1107_110791


namespace NUMINAMATH_CALUDE_mother_daughter_age_relation_l1107_110700

theorem mother_daughter_age_relation :
  ∀ (mother_age daughter_age years_ago : ℕ),
  mother_age = 43 →
  daughter_age = 11 →
  mother_age - years_ago = 5 * (daughter_age - years_ago) →
  years_ago = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_relation_l1107_110700


namespace NUMINAMATH_CALUDE_oil_tank_capacity_l1107_110703

theorem oil_tank_capacity (C : ℝ) (h1 : C > 0) :
  (C / 6 : ℝ) / C = 1 / 6 ∧ (C / 6 + 4) / C = 1 / 3 → C = 24 := by
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_l1107_110703


namespace NUMINAMATH_CALUDE_initial_number_of_men_l1107_110731

theorem initial_number_of_men (M : ℕ) (A : ℝ) : 
  (2 * M = 46 - (20 + 10)) →  -- Condition 1 and 2
  (M = 8) :=                  -- Conclusion
by
  sorry  -- Skip the proof

end NUMINAMATH_CALUDE_initial_number_of_men_l1107_110731


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1107_110755

theorem arithmetic_sequence_common_difference 
  (a₁ : ℚ) 
  (aₙ : ℚ) 
  (sum : ℚ) 
  (h₁ : a₁ = 3) 
  (h₂ : aₙ = 50) 
  (h₃ : sum = 318) : 
  ∃ (n : ℕ) (d : ℚ), 
    n > 1 ∧ 
    aₙ = a₁ + (n - 1) * d ∧ 
    sum = (n / 2) * (a₁ + aₙ) ∧ 
    d = 47 / 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1107_110755


namespace NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l1107_110772

theorem complex_product_one_plus_i_one_minus_i : 
  (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l1107_110772


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_sum_27_l1107_110766

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ sum_of_digits n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ sum_of_digits m = 27 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_sum_27_l1107_110766
