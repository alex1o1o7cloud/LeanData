import Mathlib

namespace NUMINAMATH_CALUDE_root_condition_implies_a_range_l2269_226911

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem root_condition_implies_a_range :
  ∀ a : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1) →
  -2 < a ∧ a < 1 :=
by
  sorry


end NUMINAMATH_CALUDE_root_condition_implies_a_range_l2269_226911


namespace NUMINAMATH_CALUDE_library_books_count_l2269_226941

theorem library_books_count :
  ∃ (n : ℕ), 
    500 < n ∧ n < 650 ∧ 
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 ∧
    n = 595 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l2269_226941


namespace NUMINAMATH_CALUDE_floor_equation_iff_solution_set_l2269_226927

def floor_equation (x : ℝ) : Prop :=
  ⌊x^2 - 2*x⌋ + 2*⌊x⌋ = ⌊x⌋^2

def solution_set (x : ℝ) : Prop :=
  (∃ (n : ℤ), n < 0 ∧ x = n) ∨
  x = 0 ∨
  (∃ (n : ℕ), n ≥ 1 ∧ n ≤ x ∧ x < Real.sqrt (n^2 - 2*n + 2) + 1)

theorem floor_equation_iff_solution_set :
  ∀ x : ℝ, floor_equation x ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_floor_equation_iff_solution_set_l2269_226927


namespace NUMINAMATH_CALUDE_married_couple_survival_probability_l2269_226992

/-- The probability problem for a married couple's survival over 10 years -/
theorem married_couple_survival_probability 
  (p_man : ℝ) 
  (p_neither : ℝ) 
  (h_man : p_man = 1/4) 
  (h_neither : p_neither = 1/2) : 
  ∃ p_wife : ℝ, 
    p_wife = 1/3 ∧ 
    p_neither = 1 - (p_man + p_wife - p_man * p_wife) := by
  sorry

end NUMINAMATH_CALUDE_married_couple_survival_probability_l2269_226992


namespace NUMINAMATH_CALUDE_total_waiting_after_changes_l2269_226951

/-- Represents the number of people waiting at each entrance of SFL -/
structure EntranceCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Calculates the total number of people waiting at all entrances -/
def total_waiting (count : EntranceCount) : ℕ :=
  count.a + count.b + count.c + count.d + count.e

/-- Initial count of people waiting at each entrance -/
def initial_count : EntranceCount :=
  { a := 283, b := 356, c := 412, d := 179, e := 389 }

/-- Final count of people waiting at each entrance after changes -/
def final_count : EntranceCount :=
  { a := initial_count.a - 15,
    b := initial_count.b,
    c := initial_count.c + 10,
    d := initial_count.d,
    e := initial_count.e - 20 }

/-- Theorem stating that the total number of people waiting after changes is 1594 -/
theorem total_waiting_after_changes :
  total_waiting final_count = 1594 := by sorry

end NUMINAMATH_CALUDE_total_waiting_after_changes_l2269_226951


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l2269_226928

/-- Given two quadratic equations with a common root, this theorem proves
    properties about the sum and product of the other two roots. -/
theorem quadratic_equations_common_root
  (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a ≠ b)
  (h_common : ∃ x₀ : ℝ, x₀^2 + a*x₀ + b = 0 ∧ x₀^2 + b*x₀ + a = 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : x₁^2 + a*x₁ + b = 0)
  (h_x₂ : x₂^2 + b*x₂ + a = 0) :
  (x₁ + x₂ = -1) ∧ (x₁ * x₂ ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l2269_226928


namespace NUMINAMATH_CALUDE_expected_twos_l2269_226942

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 2's when rolling three standard dice -/
theorem expected_twos : 
  num_dice * prob_two = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_expected_twos_l2269_226942


namespace NUMINAMATH_CALUDE_jean_trips_l2269_226939

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end NUMINAMATH_CALUDE_jean_trips_l2269_226939


namespace NUMINAMATH_CALUDE_solve_equation_l2269_226957

theorem solve_equation : (45 : ℚ) / (9 - 3/7) = 21/4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2269_226957


namespace NUMINAMATH_CALUDE_is_quadratic_equation_l2269_226900

theorem is_quadratic_equation (x : ℝ) : ∃ (a b c : ℝ), a ≠ 0 ∧ 3*(x+1)^2 = 2*(x+1) ↔ a*x^2 + b*x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_is_quadratic_equation_l2269_226900


namespace NUMINAMATH_CALUDE_matrix_power_four_l2269_226937

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; -1, 0]

theorem matrix_power_four : A^4 = !![(-1 : ℤ), (-1 : ℤ); (1 : ℤ), (0 : ℤ)] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2269_226937


namespace NUMINAMATH_CALUDE_equation_is_hyperbola_l2269_226905

/-- A conic section type -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for a given quadratic equation -/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 4y^2 - 2x + 8y - 8 = 0 represents a hyperbola -/
theorem equation_is_hyperbola :
  determineConicSection 1 (-4) 0 (-2) 8 (-8) = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_is_hyperbola_l2269_226905


namespace NUMINAMATH_CALUDE_paper_recycling_trees_saved_l2269_226973

theorem paper_recycling_trees_saved 
  (trees_per_tonne : ℕ) 
  (schools : ℕ) 
  (paper_per_school : ℚ) 
  (h1 : trees_per_tonne = 24)
  (h2 : schools = 4)
  (h3 : paper_per_school = 3/4) : 
  ↑schools * paper_per_school * trees_per_tonne = 72 := by
  sorry

end NUMINAMATH_CALUDE_paper_recycling_trees_saved_l2269_226973


namespace NUMINAMATH_CALUDE_unique_solution_equation_l2269_226912

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l2269_226912


namespace NUMINAMATH_CALUDE_service_period_problem_l2269_226907

/-- Represents the problem of determining the agreed-upon period of service --/
theorem service_period_problem (total_pay : ℕ) (uniform_price : ℕ) (partial_service : ℕ) (partial_pay : ℕ) :
  let full_compensation := total_pay + uniform_price
  let partial_compensation := partial_pay + uniform_price
  (partial_service : ℚ) / (12 : ℚ) = partial_compensation / full_compensation →
  12 = (partial_service * full_compensation) / partial_compensation :=
by
  sorry

#check service_period_problem 900 100 9 650

end NUMINAMATH_CALUDE_service_period_problem_l2269_226907


namespace NUMINAMATH_CALUDE_student_age_problem_l2269_226964

theorem student_age_problem (total_students : Nat) 
  (avg_age_all : Nat) (num_group1 : Nat) (avg_age_group1 : Nat) 
  (num_group2 : Nat) (avg_age_group2 : Nat) :
  total_students = 17 →
  avg_age_all = 17 →
  num_group1 = 5 →
  avg_age_group1 = 14 →
  num_group2 = 9 →
  avg_age_group2 = 16 →
  (total_students * avg_age_all) - (num_group1 * avg_age_group1) - (num_group2 * avg_age_group2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_age_problem_l2269_226964


namespace NUMINAMATH_CALUDE_fraction_equality_l2269_226960

theorem fraction_equality (a b y : ℝ) 
  (h1 : y = (a + 2*b) / a) 
  (h2 : a ≠ -2*b) 
  (h3 : a ≠ 0) : 
  (2*a + 2*b) / (a - 2*b) = (y + 1) / (3 - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2269_226960


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2269_226904

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-2 * a > -2 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2269_226904


namespace NUMINAMATH_CALUDE_inverse_equal_original_unique_solution_l2269_226910

def g (x : ℝ) : ℝ := 4 * x - 5

theorem inverse_equal_original_unique_solution :
  ∃! x : ℝ, g x = (Function.invFun g) x :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_equal_original_unique_solution_l2269_226910


namespace NUMINAMATH_CALUDE_flavoring_corn_syrup_ratio_comparison_l2269_226923

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_formulation : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : DrinkRatio :=
  { flavoring := 1.25, corn_syrup := 5, water := 75 }

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
axiom sport_water_ratio : 
  sport_formulation.flavoring / sport_formulation.water = 
  (standard_formulation.flavoring / standard_formulation.water) / 2

/-- The theorem to be proved -/
theorem flavoring_corn_syrup_ratio_comparison : 
  (sport_formulation.flavoring / sport_formulation.corn_syrup) / 
  (standard_formulation.flavoring / standard_formulation.corn_syrup) = 3 := by
  sorry

end NUMINAMATH_CALUDE_flavoring_corn_syrup_ratio_comparison_l2269_226923


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l2269_226956

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l2269_226956


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l2269_226933

theorem binomial_expansion_example : 50^4 + 4*(50^3) + 6*(50^2) + 4*50 + 1 = 6765201 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l2269_226933


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2269_226901

theorem x_plus_y_value (x y : ℝ) (h1 : x - y = 4) (h2 : |x| + |y| = 7) :
  x + y = 7 ∨ x + y = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2269_226901


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l2269_226929

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (3 + 4*I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (3 + 4*I)) = 1 ∧ ∀ (u : ℂ), Complex.abs (u - (3 + 4*I)) = 1 → Complex.abs u ≤ Complex.abs w ∧ Complex.abs w = 6 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l2269_226929


namespace NUMINAMATH_CALUDE_checkers_placement_divisibility_l2269_226945

/-- Given a prime p ≥ 5, r(p) is the number of ways to place p identical checkers 
    on a p × p checkerboard such that not all checkers are in the same row. -/
def r (p : ℕ) : ℕ := sorry

theorem checkers_placement_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  p^5 ∣ r p := by
  sorry

end NUMINAMATH_CALUDE_checkers_placement_divisibility_l2269_226945


namespace NUMINAMATH_CALUDE_total_pizza_pieces_l2269_226961

/-- Given 10 children, each buying 20 pizzas, and each pizza containing 6 pieces,
    the total number of pizza pieces is 1200. -/
theorem total_pizza_pieces :
  let num_children : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pieces_per_pizza : ℕ := 6
  num_children * pizzas_per_child * pieces_per_pizza = 1200 :=
by
  sorry


end NUMINAMATH_CALUDE_total_pizza_pieces_l2269_226961


namespace NUMINAMATH_CALUDE_rally_ticket_cost_l2269_226931

/-- The cost of tickets bought at the door at a rally --/
def ticket_cost_at_door (total_attendance : ℕ) (pre_rally_ticket_cost : ℚ) 
  (total_receipts : ℚ) (pre_rally_tickets : ℕ) : ℚ :=
  (total_receipts - pre_rally_ticket_cost * pre_rally_tickets) / (total_attendance - pre_rally_tickets)

/-- Theorem stating the cost of tickets bought at the door --/
theorem rally_ticket_cost : 
  ticket_cost_at_door 750 2 (1706.25) 475 = (2.75 : ℚ) := by sorry

end NUMINAMATH_CALUDE_rally_ticket_cost_l2269_226931


namespace NUMINAMATH_CALUDE_multiple_properties_l2269_226995

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
def x_multiple_of_4 : Prop := ∃ k : ℤ, x = 4 * k
def y_multiple_of_9 : Prop := ∃ m : ℤ, y = 9 * m

-- Theorem to prove
theorem multiple_properties
  (hx : x_multiple_of_4 x)
  (hy : y_multiple_of_9 y) :
  (∃ n : ℤ, y = 3 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l2269_226995


namespace NUMINAMATH_CALUDE_m_greater_than_n_l2269_226954

theorem m_greater_than_n (x y : ℝ) : x^2 + y^2 + 1 > 2*(x + y - 1) := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l2269_226954


namespace NUMINAMATH_CALUDE_correct_calculation_l2269_226917

theorem correct_calculation : (1/3) + (-1/2) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2269_226917


namespace NUMINAMATH_CALUDE_remainder_problem_l2269_226968

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2269_226968


namespace NUMINAMATH_CALUDE_m_minus_n_values_l2269_226921

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 4)
  (hn : |n| = 6)
  (hmn : |m + n| = m + n) :
  m - n = -2 ∨ m - n = -10 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l2269_226921


namespace NUMINAMATH_CALUDE_equal_selection_probability_all_students_equal_probability_l2269_226934

/-- Represents the probability of a student being selected -/
def selection_probability (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) : ℚ :=
  selected / (total_students - eliminated)

/-- The selection method results in equal probability for all students -/
theorem equal_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2004) 
  (h2 : eliminated = 4) 
  (h3 : selected = 50) :
  selection_probability total_students eliminated selected = 1 / 40 :=
sorry

/-- The probability of selection is the same for all students -/
theorem all_students_equal_probability 
  (student1 student2 : ℕ) 
  (h_student1 : student1 ≤ 2004) 
  (h_student2 : student2 ≤ 2004) :
  selection_probability 2004 4 50 = selection_probability 2004 4 50 :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_all_students_equal_probability_l2269_226934


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l2269_226909

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l2269_226909


namespace NUMINAMATH_CALUDE_edward_mowed_five_lawns_l2269_226997

/-- Represents the number of lawns Edward mowed -/
def lawns_mowed : ℕ := sorry

/-- Edward's earnings per lawn in dollars -/
def earnings_per_lawn : ℕ := 8

/-- Edward's initial savings in dollars -/
def initial_savings : ℕ := 7

/-- Edward's total money after mowing lawns in dollars -/
def total_money : ℕ := 47

/-- Theorem stating that Edward mowed 5 lawns -/
theorem edward_mowed_five_lawns :
  lawns_mowed = 5 ∧
  earnings_per_lawn * lawns_mowed + initial_savings = total_money :=
sorry

end NUMINAMATH_CALUDE_edward_mowed_five_lawns_l2269_226997


namespace NUMINAMATH_CALUDE_monogram_count_is_300_l2269_226952

/-- The number of letters in the alphabet before 'A' --/
def n : ℕ := 25

/-- The number of initials to choose (first and middle) --/
def k : ℕ := 2

/-- The number of ways to choose k distinct letters from n letters in alphabetical order --/
def monogram_count : ℕ := Nat.choose n k

/-- Theorem stating that the number of possible monograms is 300 --/
theorem monogram_count_is_300 : monogram_count = 300 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_is_300_l2269_226952


namespace NUMINAMATH_CALUDE_negative_less_than_positive_l2269_226976

theorem negative_less_than_positive : 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → x < y) →
  -897 < 0.01 := by sorry

end NUMINAMATH_CALUDE_negative_less_than_positive_l2269_226976


namespace NUMINAMATH_CALUDE_dollar_equality_l2269_226953

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_equality (x y z : ℝ) : dollar ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_equality_l2269_226953


namespace NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l2269_226930

/-- The fixed point on the graph of y = 9x^2 + mx + 3m for all real m -/
theorem fixed_point_on_quadratic_graph :
  ∀ (m : ℝ), 9 * (-3)^2 + m * (-3) + 3 * m = 81 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_quadratic_graph_l2269_226930


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2269_226926

def A : Set ℝ := {x | -5 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x ≤ 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2269_226926


namespace NUMINAMATH_CALUDE_expression_value_l2269_226972

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2269_226972


namespace NUMINAMATH_CALUDE_inequalities_proof_l2269_226984

theorem inequalities_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a * b > b * c) ∧ (2022^(a - c) + a > 2022^(b - c) + b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2269_226984


namespace NUMINAMATH_CALUDE_congruence_solution_l2269_226986

theorem congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2187 [ZMOD 10] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2269_226986


namespace NUMINAMATH_CALUDE_scientific_notation_of_3185800_l2269_226902

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of decimal places -/
def roundToDecimalPlaces (sn : ScientificNotation) (places : ℕ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_3185800 :
  let original := 3185800
  let scientificForm := toScientificNotation original
  let rounded := roundToDecimalPlaces scientificForm 1
  rounded.coefficient = 3.2 ∧ rounded.exponent = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_3185800_l2269_226902


namespace NUMINAMATH_CALUDE_total_units_is_34_l2269_226989

/-- The number of apartment units in two identical buildings with specific floor configurations -/
def total_apartment_units : ℕ := by
  -- Define the number of buildings
  let num_buildings : ℕ := 2

  -- Define the number of floors in each building
  let num_floors : ℕ := 4

  -- Define the number of units on the first floor
  let units_first_floor : ℕ := 2

  -- Define the number of units on each of the other floors
  let units_other_floors : ℕ := 5

  -- Calculate the total number of units in one building
  let units_per_building : ℕ := units_first_floor + (num_floors - 1) * units_other_floors

  -- Calculate the total number of units in all buildings
  exact num_buildings * units_per_building

/-- Theorem stating that the total number of apartment units is 34 -/
theorem total_units_is_34 : total_apartment_units = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_units_is_34_l2269_226989


namespace NUMINAMATH_CALUDE_gcd_n_pow_13_minus_n_l2269_226958

theorem gcd_n_pow_13_minus_n : ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℤ), (d : ℤ) ∣ (n^13 - n)) ∧ 
  (∀ (k : ℕ), k > 0 → (∀ (n : ℤ), (k : ℤ) ∣ (n^13 - n)) → k ∣ d) ∧
  d = 2730 := by
sorry

end NUMINAMATH_CALUDE_gcd_n_pow_13_minus_n_l2269_226958


namespace NUMINAMATH_CALUDE_cylinder_sphere_min_volume_l2269_226993

/-- Given a cylinder with lateral surface area 4π and an external tangent sphere,
    prove that the total surface area of the cylinder is 6π when the volume of the sphere is minimum -/
theorem cylinder_sphere_min_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 4 * Real.pi →
  (∀ R : ℝ, R > 0 → R^2 ≥ r^2 + (h/2)^2) →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_sphere_min_volume_l2269_226993


namespace NUMINAMATH_CALUDE_jerry_age_l2269_226962

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 24 → 
  mickey_age = 4 * jerry_age - 8 → 
  jerry_age = 8 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l2269_226962


namespace NUMINAMATH_CALUDE_greatest_integer_x_l2269_226994

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def f (x : ℤ) : ℚ := (x^2 + 4*x + 13) / (x - 4)

theorem greatest_integer_x : 
  (∀ x : ℤ, x > 49 → ¬ is_integer (f x)) ∧ 
  is_integer (f 49) := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_l2269_226994


namespace NUMINAMATH_CALUDE_share_price_increase_l2269_226971

theorem share_price_increase (P : ℝ) (X : ℝ) : 
  X > 0 →
  (P * (1 + X / 100)) * (1 + 1 / 3) = P * 1.6 →
  X = 20 := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l2269_226971


namespace NUMINAMATH_CALUDE_probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l2269_226919

/-- The probability of getting exactly 7 heads when flipping 10 fair coins -/
theorem probability_seven_heads_ten_coins : ℚ :=
  15 / 128

/-- Proof that the probability of getting exactly 7 heads when flipping 10 fair coins is 15/128 -/
theorem prove_probability_seven_heads_ten_coins :
  probability_seven_heads_ten_coins = 15 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_heads_ten_coins_prove_probability_seven_heads_ten_coins_l2269_226919


namespace NUMINAMATH_CALUDE_initial_amount_proof_l2269_226913

/-- 
Given an initial amount that increases by 1/8th of itself each year, 
this theorem proves that if the amount after two years is 82265.625, 
then the initial amount was 65000.
-/
theorem initial_amount_proof (P : ℚ) : 
  ((9/8 : ℚ)^2 * P = 82265.625) → P = 65000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l2269_226913


namespace NUMINAMATH_CALUDE_function_property_l2269_226983

theorem function_property (f : ℝ → ℝ) (h : ∀ x, f (Real.sin x) = Real.sin (2011 * x)) :
  ∀ x, f (Real.cos x) = Real.cos (2011 * x) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2269_226983


namespace NUMINAMATH_CALUDE_checkerboard_diagonal_squares_l2269_226943

theorem checkerboard_diagonal_squares (m n : ℕ) (hm : m = 91) (hn : n = 28) :
  m + n - Nat.gcd m n = 112 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_diagonal_squares_l2269_226943


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2269_226944

/-- The system of equations has no integer solutions -/
theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + y^2 - z^2 = 17) ∧ 
  (-x^2 + 3*y*z + 3*z^2 = 27) ∧ 
  (x^2 - x*y + 5*z^2 = 50) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2269_226944


namespace NUMINAMATH_CALUDE_base_4_addition_l2269_226940

/-- Convert a base 10 number to base 4 --/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a base 4 number (represented as a list of digits) to base 10 --/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

/-- Add two base 4 numbers (represented as lists of digits) --/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem base_4_addition :
  addBase4 (toBase4 45) (toBase4 28) = [1, 0, 2, 1] ∧ fromBase4 [1, 0, 2, 1] = 45 + 28 := by
  sorry

end NUMINAMATH_CALUDE_base_4_addition_l2269_226940


namespace NUMINAMATH_CALUDE_family_total_weight_l2269_226975

/-- Represents the weights of a family consisting of a mother, daughter, and grandchild. -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family members. -/
def FamilyWeights.total (fw : FamilyWeights) : ℝ :=
  fw.mother + fw.daughter + fw.grandchild

/-- The conditions given in the problem. -/
def satisfies_conditions (fw : FamilyWeights) : Prop :=
  fw.daughter + fw.grandchild = 60 ∧
  fw.grandchild = (1 / 5) * fw.mother ∧
  fw.daughter = 42

/-- Theorem stating that given the conditions, the total weight is 150 kg. -/
theorem family_total_weight (fw : FamilyWeights) 
  (h : satisfies_conditions fw) : fw.total = 150 := by
  sorry

end NUMINAMATH_CALUDE_family_total_weight_l2269_226975


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l2269_226963

/-- A function that checks if a natural number can be expressed as a non-negative linear combination of 5 and 6 -/
def isExpressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The theorem stating that 19 is the largest natural number not exceeding 50 that cannot be expressed as a non-negative linear combination of 5 and 6 -/
theorem largest_inexpressible_number :
  (∀ (k : ℕ), k > 19 ∧ k ≤ 50 → isExpressible k) ∧
  ¬isExpressible 19 :=
sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l2269_226963


namespace NUMINAMATH_CALUDE_older_friend_age_l2269_226908

theorem older_friend_age (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) :
  A = 104.25 := by
  sorry

end NUMINAMATH_CALUDE_older_friend_age_l2269_226908


namespace NUMINAMATH_CALUDE_greatest_integer_with_prime_absolute_value_l2269_226949

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_integer_with_prime_absolute_value :
  ∀ x : ℤ, (is_prime (Int.natAbs (8 * x^2 - 66 * x + 21))) →
    x ≤ 2 ∧ is_prime (Int.natAbs (8 * 2^2 - 66 * 2 + 21)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_prime_absolute_value_l2269_226949


namespace NUMINAMATH_CALUDE_custom_op_result_l2269_226916

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b - 1)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 7 5) 2 = 14 / 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l2269_226916


namespace NUMINAMATH_CALUDE_specific_box_volume_l2269_226977

/-- The volume of an open box constructed from a rectangular sheet --/
def box_volume (sheet_length sheet_width y : ℝ) : ℝ :=
  (sheet_length - 2 * y) * (sheet_width - 2 * y) * y

/-- Theorem stating the volume of the specific box described in the problem --/
theorem specific_box_volume (y : ℝ) :
  box_volume 15 12 y = 180 * y - 54 * y^2 + 4 * y^3 :=
by sorry

end NUMINAMATH_CALUDE_specific_box_volume_l2269_226977


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2269_226955

theorem quadratic_roots_property (a b : ℝ) : 
  a ≠ b ∧ 
  a^2 + 3*a - 5 = 0 ∧ 
  b^2 + 3*b - 5 = 0 → 
  a^2 + 3*a*b + a - 2*b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2269_226955


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2269_226924

theorem quadratic_equation_solution (x : ℝ) : 16 * x^2 = 81 ↔ x = 2.25 ∨ x = -2.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2269_226924


namespace NUMINAMATH_CALUDE_parallel_line_plane_not_implies_parallel_all_lines_l2269_226938

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (contains : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_implies_parallel_all_lines 
  (α : Plane) (a b : Line) : 
  ¬(∀ (p : Plane) (l m : Line), 
    parallel_line_plane l p → 
    contains p m → 
    parallel_lines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_not_implies_parallel_all_lines_l2269_226938


namespace NUMINAMATH_CALUDE_ara_height_ara_current_height_ara_height_is_59_l2269_226998

theorem ara_height (shea_initial : ℝ) (shea_final : ℝ) (ara_initial : ℝ) : ℝ :=
  let shea_growth := shea_final - shea_initial
  let ara_growth := shea_growth / 3
  ara_initial + ara_growth

theorem ara_current_height : ℝ :=
  let shea_final := 64
  let shea_initial := shea_final / 1.25
  let ara_initial := shea_initial + 4
  ara_height shea_initial shea_final ara_initial

theorem ara_height_is_59 : ⌊ara_current_height⌋ = 59 := by
  sorry

end NUMINAMATH_CALUDE_ara_height_ara_current_height_ara_height_is_59_l2269_226998


namespace NUMINAMATH_CALUDE_thirty_first_never_sunday_l2269_226922

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- The number of days in each month -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | Month.February => if isLeapYear then 29 else 28
  | Month.April | Month.June | Month.September | Month.November => 30
  | _ => 31

/-- The theorem stating that 31 is the only date that can never be a Sunday -/
theorem thirty_first_never_sunday :
  ∃! (date : Nat), date > 0 ∧ date ≤ 31 ∧
  ∀ (year : Nat) (m : Month),
    daysInMonth m (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ≥ date →
    ∃ (dow : DayOfWeek), dow ≠ DayOfWeek.Sunday :=
by
  sorry

end NUMINAMATH_CALUDE_thirty_first_never_sunday_l2269_226922


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2269_226950

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2269_226950


namespace NUMINAMATH_CALUDE_clothing_purchase_properties_l2269_226967

/-- Represents the clothing purchase problem for a recitation competition. -/
structure ClothingPurchase where
  total_students : Nat
  combined_cost : Nat
  cost_ratio : Nat → Nat → Prop
  boy_girl_ratio : Nat → Nat → Prop
  max_total_cost : Nat

/-- Calculates the unit prices of men's and women's clothing. -/
def calculate_unit_prices (cp : ClothingPurchase) : Nat × Nat :=
  sorry

/-- Counts the number of valid purchasing plans. -/
def count_valid_plans (cp : ClothingPurchase) : Nat :=
  sorry

/-- Calculates the minimum cost of clothing purchase. -/
def minimum_cost (cp : ClothingPurchase) : Nat :=
  sorry

/-- Main theorem proving the properties of the clothing purchase problem. -/
theorem clothing_purchase_properties (cp : ClothingPurchase) 
  (h1 : cp.total_students = 150)
  (h2 : cp.combined_cost = 220)
  (h3 : cp.cost_ratio = λ m w => 6 * m = 5 * w)
  (h4 : cp.boy_girl_ratio = λ b g => b ≤ 2 * g / 3)
  (h5 : cp.max_total_cost = 17000) :
  let (men_price, women_price) := calculate_unit_prices cp
  men_price = 100 ∧ 
  women_price = 120 ∧ 
  count_valid_plans cp = 11 ∧ 
  minimum_cost cp = 16800 :=
sorry

end NUMINAMATH_CALUDE_clothing_purchase_properties_l2269_226967


namespace NUMINAMATH_CALUDE_range_of_f_l2269_226948

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y ≤ 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2269_226948


namespace NUMINAMATH_CALUDE_rectangle_area_is_200_l2269_226925

/-- A rectangular region with three fenced sides and one wall -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  wall_side : ℝ := long_side
  fenced_sides : ℝ := 2 * short_side + long_side
  area : ℝ := short_side * long_side

/-- The fenced rectangular region satisfying the problem conditions -/
def problem_rectangle : FencedRectangle where
  short_side := 10
  long_side := 20
  fence_length := 40

theorem rectangle_area_is_200 (r : FencedRectangle) :
  r.long_side = 2 * r.short_side →
  r.fence_length = 40 →
  r.area = 200 := by
  sorry

#check rectangle_area_is_200 problem_rectangle

end NUMINAMATH_CALUDE_rectangle_area_is_200_l2269_226925


namespace NUMINAMATH_CALUDE_franks_age_l2269_226974

/-- Represents the ages of Dave, Ella, and Frank -/
structure Ages where
  dave : ℕ
  ella : ℕ
  frank : ℕ

/-- The conditions from the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 10
  (ages.dave + ages.ella + ages.frank) / 3 = 10 ∧
  -- Five years ago, Frank was the same age as Dave is now
  ages.frank - 5 = ages.dave ∧
  -- In 2 years, Ella's age will be 3/4 of Dave's age at that time
  ages.ella + 2 = (3 * (ages.dave + 2)) / 4

/-- The theorem to prove -/
theorem franks_age (ages : Ages) (h : satisfies_conditions ages) : ages.frank = 14 := by
  sorry


end NUMINAMATH_CALUDE_franks_age_l2269_226974


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2269_226988

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2269_226988


namespace NUMINAMATH_CALUDE_find_other_number_l2269_226903

theorem find_other_number (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (3 * a + 4 * b = 131) → 
  (a = 21 ∨ b = 21) :=
by sorry

end NUMINAMATH_CALUDE_find_other_number_l2269_226903


namespace NUMINAMATH_CALUDE_power_of_product_of_ten_l2269_226918

theorem power_of_product_of_ten : (2 * 10^3)^3 = 8 * 10^9 := by sorry

end NUMINAMATH_CALUDE_power_of_product_of_ten_l2269_226918


namespace NUMINAMATH_CALUDE_soccer_lineup_theorem_l2269_226985

/-- The number of ways to choose a soccer lineup -/
def soccer_lineup_count : ℕ := 18 * (Nat.choose 17 4) * (Nat.choose 13 3) * (Nat.choose 10 3)

/-- Theorem stating the number of possible soccer lineups -/
theorem soccer_lineup_theorem : soccer_lineup_count = 147497760 := by
  sorry

end NUMINAMATH_CALUDE_soccer_lineup_theorem_l2269_226985


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2269_226932

theorem smallest_integer_satisfying_conditions : 
  ∃ n : ℤ, (∀ m : ℤ, (m + 15 ≥ 16 ∧ -5 * m < -10) → n ≤ m) ∧ 
           (n + 15 ≥ 16 ∧ -5 * n < -10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l2269_226932


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2269_226947

-- Define the condition
def condition (A : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, A = (k * Real.pi, 0)

-- Define the statement
def statement (A : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, Real.tan (A.1 + x) = -Real.tan (A.1 - x)

-- Theorem stating the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ A : ℝ × ℝ, condition A → statement A) ∧
  ¬(∀ A : ℝ × ℝ, statement A → condition A) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2269_226947


namespace NUMINAMATH_CALUDE_journey_distance_journey_distance_proof_l2269_226999

theorem journey_distance : ℝ → Prop :=
  fun d : ℝ =>
    let t := d / 40
    t + 1/4 = d / 35 →
    d = 70

-- The proof is omitted
theorem journey_distance_proof : journey_distance 70 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_journey_distance_proof_l2269_226999


namespace NUMINAMATH_CALUDE_emily_beads_count_l2269_226946

/-- The number of necklaces Emily made -/
def num_necklaces : ℕ := 11

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 28

/-- The total number of beads Emily had -/
def total_beads : ℕ := num_necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 308 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l2269_226946


namespace NUMINAMATH_CALUDE_g_values_l2269_226991

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom sum_one : ∀ x, g x + f x = 1
axiom g_odd : ∀ x, g (x + 1) = -g (-x + 1)
axiom f_odd : ∀ x, f (2 - x) = -f (2 + x)

-- Define the theorem
theorem g_values : g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_values_l2269_226991


namespace NUMINAMATH_CALUDE_evaluate_expression_l2269_226936

theorem evaluate_expression : 
  3999^3 - 2 * 3998 * 3999^2 - 2 * 3998^2 * 3999 + 3997^3 = 95806315 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2269_226936


namespace NUMINAMATH_CALUDE_remainder_problem_l2269_226935

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2269_226935


namespace NUMINAMATH_CALUDE_root_product_equality_l2269_226914

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equality_l2269_226914


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_implies_parallel_l2269_226966

-- Define a structure for a line in a plane
structure Line where
  -- You can add more properties if needed
  mk :: (id : Nat)

-- Define perpendicularity between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition of perpendicularity

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallelism

-- Theorem statement
theorem perpendicular_to_same_line_implies_parallel 
  (l1 l2 l3 : Line) : 
  perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2 :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_perpendicular_to_same_line_implies_parallel_l2269_226966


namespace NUMINAMATH_CALUDE_bernard_white_notebooks_l2269_226920

/-- The number of white notebooks Bernard had -/
def white_notebooks : ℕ := sorry

/-- The number of red notebooks Bernard had -/
def red_notebooks : ℕ := 15

/-- The number of blue notebooks Bernard had -/
def blue_notebooks : ℕ := 17

/-- The number of notebooks Bernard gave to Tom -/
def notebooks_given : ℕ := 46

/-- The number of notebooks Bernard had left -/
def notebooks_left : ℕ := 5

/-- The total number of notebooks Bernard originally had -/
def total_notebooks : ℕ := notebooks_given + notebooks_left

theorem bernard_white_notebooks : 
  white_notebooks = total_notebooks - (red_notebooks + blue_notebooks) ∧ 
  white_notebooks = 19 := by sorry

end NUMINAMATH_CALUDE_bernard_white_notebooks_l2269_226920


namespace NUMINAMATH_CALUDE_smallest_value_of_roots_sum_l2269_226990

/-- 
Given a quadratic equation x^2 - t*x + q with roots α and β,
where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
the smallest possible value of 1/α^2011 + 1/β^2011 is 2.
-/
theorem smallest_value_of_roots_sum (t q α β : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → α^n + β^n = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ x : ℝ, x^2 - t*x + q = 0 → x = α ∨ x = β) →
  (1 / α^2011 + 1 / β^2011) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_roots_sum_l2269_226990


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2269_226970

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ k > -2*Real.sqrt 3 ∧ k < 2*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2269_226970


namespace NUMINAMATH_CALUDE_geometric_solid_height_l2269_226906

-- Define the geometric solid
structure GeometricSolid where
  radius1 : ℝ
  radius2 : ℝ
  water_height1 : ℝ
  water_height2 : ℝ

-- Define the theorem
theorem geometric_solid_height (s : GeometricSolid) 
  (h1 : s.radius1 = 1)
  (h2 : s.radius2 = 3)
  (h3 : s.water_height1 = 20)
  (h4 : s.water_height2 = 28) :
  ∃ (total_height : ℝ), total_height = 29 := by
  sorry

end NUMINAMATH_CALUDE_geometric_solid_height_l2269_226906


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2269_226959

/-- Given three terms of a geometric progression, prove that the common ratio is 52/25 -/
theorem geometric_progression_ratio (x : ℝ) (h_x : x ≠ 0) :
  let a₁ : ℝ := x / 2
  let a₂ : ℝ := 2 * x - 3
  let a₃ : ℝ := 18 / x + 1
  (a₁ * a₃ = a₂^2) → (a₂ / a₁ = 52 / 25) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2269_226959


namespace NUMINAMATH_CALUDE_construct_octagon_from_square_l2269_226978

/-- A square sheet of paper --/
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

/-- A regular octagon --/
structure RegularOctagon :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Represents the ability to fold paper --/
def can_fold : Prop := True

/-- Represents the ability to cut along creases --/
def can_cut_along_creases : Prop := True

/-- Represents the prohibition of using a compass --/
def no_compass : Prop := True

/-- Represents the prohibition of using a ruler --/
def no_ruler : Prop := True

/-- Theorem stating that a regular octagon can be constructed from a square sheet of paper --/
theorem construct_octagon_from_square 
  (s : Square) 
  (fold : can_fold) 
  (cut : can_cut_along_creases) 
  (no_compass : no_compass) 
  (no_ruler : no_ruler) : 
  ∃ (o : RegularOctagon), True :=
sorry

end NUMINAMATH_CALUDE_construct_octagon_from_square_l2269_226978


namespace NUMINAMATH_CALUDE_badge_exchange_l2269_226996

theorem badge_exchange (x : ℕ) : 
  (x + 5 - (24 * (x + 5)) / 100 + (20 * x) / 100 = x - (20 * x) / 100 + (24 * (x + 5)) / 100 - 1) → 
  (x = 45 ∧ x + 5 = 50) :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l2269_226996


namespace NUMINAMATH_CALUDE_income_percentage_difference_l2269_226981

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_annual B_monthly C_monthly : ℝ),
  A_annual = 436800.0000000001 →
  C_monthly = 13000 →
  A_annual / 12 / B_monthly = 5 / 2 →
  (B_monthly - C_monthly) / C_monthly = 0.12 :=
by
  sorry

end NUMINAMATH_CALUDE_income_percentage_difference_l2269_226981


namespace NUMINAMATH_CALUDE_necklace_labeling_theorem_l2269_226987

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

structure Necklace :=
  (beads : ℕ)

def valid_labeling (n : ℕ) (A B : Necklace) (labeling : ℕ → ℕ) : Prop :=
  (∀ i, i < A.beads + B.beads → n ≤ labeling i ∧ labeling i ≤ n + 32) ∧
  (∀ i j, i ≠ j → i < A.beads + B.beads → j < A.beads + B.beads → labeling i ≠ labeling j) ∧
  (∀ i, i < A.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling 0) (labeling (A.beads - 1))) ∧
  (∀ i, A.beads ≤ i ∧ i < A.beads + B.beads - 1 → is_coprime (labeling i) (labeling (i + 1))) ∧
  (is_coprime (labeling A.beads) (labeling (A.beads + B.beads - 1)))

theorem necklace_labeling_theorem (n : ℕ) (A B : Necklace) 
  (h_n_odd : is_odd n) (h_n_ge_1 : n ≥ 1) (h_A : A.beads = 14) (h_B : B.beads = 19) :
  ∃ labeling : ℕ → ℕ, valid_labeling n A B labeling :=
sorry

end NUMINAMATH_CALUDE_necklace_labeling_theorem_l2269_226987


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l2269_226915

theorem largest_n_with_conditions : ∃ n : ℕ, n = 289 ∧ 
  (∃ m : ℤ, n^2 = (m+1)^3 - m^3) ∧
  (∃ k : ℕ, 2*n + 99 = k^2) ∧
  (∀ n' : ℕ, n' > n → 
    (¬∃ m : ℤ, n'^2 = (m+1)^3 - m^3) ∨
    (¬∃ k : ℕ, 2*n' + 99 = k^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l2269_226915


namespace NUMINAMATH_CALUDE_alice_study_time_l2269_226980

/-- Represents the inverse relationship between study time and test score -/
def inverse_relation (study_time : ℝ) (score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ study_time * score = k

theorem alice_study_time
  (first_study : ℝ)
  (first_score : ℝ)
  (average_score : ℝ)
  (h_inverse : inverse_relation first_study first_score)
  (h_first_study : first_study = 2)
  (h_first_score : first_score = 60)
  (h_average : average_score = 75) :
  ∃ second_study : ℝ,
    inverse_relation second_study ((2 * average_score - first_score)) ∧
    second_study = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_alice_study_time_l2269_226980


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l2269_226982

theorem positive_solution_of_equation : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6) ∧ 
  x = 30 + 2 * Real.sqrt 231 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l2269_226982


namespace NUMINAMATH_CALUDE_center_top_second_row_value_l2269_226969

/-- Represents a 4x4 grid of real numbers -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Checks if a sequence of 4 real numbers is arithmetic -/
def IsArithmeticSequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- The property that each row and column of the grid is an arithmetic sequence -/
def GridProperty (g : Grid) : Prop :=
  (∀ i : Fin 4, IsArithmeticSequence (λ j ↦ g i j)) ∧
  (∀ j : Fin 4, IsArithmeticSequence (λ i ↦ g i j))

theorem center_top_second_row_value
  (g : Grid)
  (h_grid : GridProperty g)
  (h_first_row : g 0 0 = 4 ∧ g 0 3 = 16)
  (h_last_row : g 3 0 = 10 ∧ g 3 3 = 40) :
  g 1 1 = 12 := by
  sorry

end NUMINAMATH_CALUDE_center_top_second_row_value_l2269_226969


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2269_226979

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 + 3 / (a - 2)) / ((a^2 + 2*a + 1) / (a - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2269_226979


namespace NUMINAMATH_CALUDE_faster_train_length_l2269_226965

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 180)
  (h2 : slower_speed = 90)
  (h3 : crossing_time = 15)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := (faster_speed - slower_speed) * (5/18)
  (relative_speed * crossing_time) = 375 := by
sorry

end NUMINAMATH_CALUDE_faster_train_length_l2269_226965
