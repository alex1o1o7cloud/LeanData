import Mathlib

namespace NUMINAMATH_CALUDE_profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l1716_171609

/-- Represents the product pricing model -/
structure PricingModel where
  currentPrice : ℝ
  weeklySales : ℝ
  salesDecrease : ℝ
  costPrice : ℝ

/-- Calculates the profit for a given price increase -/
def profit (model : PricingModel) (priceIncrease : ℝ) : ℝ :=
  (model.currentPrice + priceIncrease - model.costPrice) *
  (model.weeklySales - model.salesDecrease * priceIncrease)

/-- The pricing model for the given problem -/
def givenModel : PricingModel :=
  { currentPrice := 60
    weeklySales := 300
    salesDecrease := 10
    costPrice := 40 }

/-- Theorem: A price increase of 10 yuan results in a weekly profit of 6000 yuan -/
theorem profit_at_10_yuan_increase (ε : ℝ) :
  |profit givenModel 10 - 6000| < ε := by sorry

/-- Theorem: A price increase of 5 yuan maximizes the weekly profit -/
theorem profit_maximum_at_5_yuan_increase :
  ∀ x, profit givenModel 5 ≥ profit givenModel x := by sorry

end NUMINAMATH_CALUDE_profit_at_10_yuan_increase_profit_maximum_at_5_yuan_increase_l1716_171609


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1716_171619

/-- Given a geometric sequence of positive numbers where the fifth term is 32 and the eleventh term is 2, the seventh term is 8. -/
theorem geometric_sequence_seventh_term (a : ℝ) (r : ℝ) (h1 : a * r^4 = 32) (h2 : a * r^10 = 2) :
  a * r^6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1716_171619


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l1716_171616

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 230) (h2 : 0.65 * x = 0.20 * A) : A = 747.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l1716_171616


namespace NUMINAMATH_CALUDE_exists_counterexample_l1716_171651

/-- A binary operation on a set S satisfying a * (b * a) = b for all a, b in S -/
class SpecialOperation (S : Type) where
  op : S → S → S
  property : ∀ (a b : S), op a (op b a) = b

/-- Theorem stating that there exist elements a and b in S such that (a*b)*a ≠ a -/
theorem exists_counterexample {S : Type} [SpecialOperation S] [Inhabited S] [Nontrivial S] :
  ∃ (a b : S), (SpecialOperation.op (SpecialOperation.op a b) a) ≠ a := by sorry

end NUMINAMATH_CALUDE_exists_counterexample_l1716_171651


namespace NUMINAMATH_CALUDE_annual_growth_rate_l1716_171622

theorem annual_growth_rate (initial : ℝ) (final : ℝ) (years : ℕ) (x : ℝ) 
  (h1 : initial = 1000000)
  (h2 : final = 1690000)
  (h3 : years = 2)
  (h4 : x > 0)
  (h5 : (1 + x)^years = final / initial) :
  x = 0.3 := by
sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l1716_171622


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l1716_171666

/-- Given Carina's coffee packages, prove the total amount of coffee. -/
theorem carina_coffee_amount :
  ∀ (five_oz ten_oz : ℕ),
  five_oz = ten_oz + 2 →
  ten_oz = 3 →
  five_oz * 5 + ten_oz * 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l1716_171666


namespace NUMINAMATH_CALUDE_function_property_l1716_171617

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem to be proved -/
theorem function_property (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1716_171617


namespace NUMINAMATH_CALUDE_greatest_common_length_of_cords_l1716_171621

theorem greatest_common_length_of_cords :
  let cord_lengths : List ℝ := [Real.sqrt 20, Real.pi, Real.exp 1, Real.sqrt 98]
  ∀ x : ℝ, (∀ l ∈ cord_lengths, ∃ n : ℕ, l = x * n) → x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_length_of_cords_l1716_171621


namespace NUMINAMATH_CALUDE_crayon_boxes_l1716_171650

theorem crayon_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (boxes_needed : ℕ) : 
  total_crayons = 80 → 
  crayons_per_box = 8 → 
  boxes_needed = total_crayons / crayons_per_box →
  boxes_needed = 10 := by
sorry

end NUMINAMATH_CALUDE_crayon_boxes_l1716_171650


namespace NUMINAMATH_CALUDE_range_of_a_l1716_171607

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 2 → a ≤ x + 1 / (x - 2)) → 
  ∃ s : ℝ, s = 4 ∧ ∀ y : ℝ, (∀ x : ℝ, x > 2 → y ≤ x + 1 / (x - 2)) → y ≤ s :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1716_171607


namespace NUMINAMATH_CALUDE_original_average_l1716_171614

theorem original_average (n : ℕ) (a : ℝ) (h1 : n = 10) (h2 : (n * a + n * 4) / n = 27) : a = 23 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l1716_171614


namespace NUMINAMATH_CALUDE_two_times_three_plus_two_times_three_l1716_171687

theorem two_times_three_plus_two_times_three : 2 * 3 + 2 * 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_times_three_plus_two_times_three_l1716_171687


namespace NUMINAMATH_CALUDE_three_W_five_l1716_171671

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- Theorem statement
theorem three_W_five : W 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_three_W_five_l1716_171671


namespace NUMINAMATH_CALUDE_blue_ball_probability_l1716_171667

noncomputable def bag_probabilities (p_red p_yellow p_blue : ℝ) : Prop :=
  p_red + p_yellow + p_blue = 1 ∧ 0 ≤ p_red ∧ 0 ≤ p_yellow ∧ 0 ≤ p_blue

theorem blue_ball_probability :
  ∀ (p_red p_yellow p_blue : ℝ),
    bag_probabilities p_red p_yellow p_blue →
    p_red = 0.48 →
    p_yellow = 0.35 →
    p_blue = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l1716_171667


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1716_171627

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9375) (h4 : y / x = 15) : 
  x + y = 400 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1716_171627


namespace NUMINAMATH_CALUDE_chord_length_l1716_171648

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, 0), radius := 6 }
def C2 : Circle := { center := (18, 0), radius := 12 }
def C3 : Circle := { center := (38, 0), radius := 38 }
def C4 : Circle := { center := (58, 0), radius := 20 }

-- Define the properties of the circles
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c2.radius - c1.radius)^2

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem chord_length :
  externally_tangent C1 C2 ∧
  internally_tangent C1 C3 ∧
  internally_tangent C2 C3 ∧
  externally_tangent C3 C4 ∧
  collinear C1.center C2.center C3.center →
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l1716_171648


namespace NUMINAMATH_CALUDE_captain_age_is_27_l1716_171684

/-- Represents the age of the cricket team captain -/
def captain_age : ℕ := sorry

/-- Represents the age of the wicket keeper -/
def wicket_keeper_age : ℕ := sorry

/-- The number of players in the cricket team -/
def team_size : ℕ := 11

/-- The average age of the whole team -/
def team_average_age : ℕ := 24

theorem captain_age_is_27 :
  captain_age = 27 ∧
  wicket_keeper_age = captain_age + 3 ∧
  team_size * team_average_age = captain_age + wicket_keeper_age + (team_size - 2) * (team_average_age - 1) :=
by sorry

end NUMINAMATH_CALUDE_captain_age_is_27_l1716_171684


namespace NUMINAMATH_CALUDE_equation_solution_l1716_171646

theorem equation_solution :
  ∃ (x : ℝ), x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5) ∧ x = -9 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1716_171646


namespace NUMINAMATH_CALUDE_africa_passenger_fraction_l1716_171692

theorem africa_passenger_fraction :
  let total_passengers : ℕ := 108
  let north_america_fraction : ℚ := 1 / 12
  let europe_fraction : ℚ := 1 / 4
  let asia_fraction : ℚ := 1 / 6
  let other_continents : ℕ := 42
  let africa_fraction : ℚ := 1 - north_america_fraction - europe_fraction - asia_fraction - (other_continents : ℚ) / total_passengers
  africa_fraction = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_africa_passenger_fraction_l1716_171692


namespace NUMINAMATH_CALUDE_limit_special_function_l1716_171658

/-- The limit of (2 - e^(x^2))^(1 / (1 - cos(π * x))) as x approaches 0 is e^(-2 / π^2) -/
theorem limit_special_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((2 - Real.exp (x^2))^(1 / (1 - Real.cos (π * x)))) - Real.exp (-2 / π^2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_special_function_l1716_171658


namespace NUMINAMATH_CALUDE_probability_one_from_each_group_l1716_171686

theorem probability_one_from_each_group :
  ∀ (total : ℕ) (group1 : ℕ) (group2 : ℕ),
    total = group1 + group2 →
    group1 > 0 →
    group2 > 0 →
    (group1 : ℚ) / total * group2 / (total - 1) +
    (group2 : ℚ) / total * group1 / (total - 1) = 5 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_one_from_each_group_l1716_171686


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l1716_171661

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (10 * x + 10) = 1920 * k) ∧
  (∀ (n : ℤ), n > 1920 → ∃ (y : ℤ), Odd y ∧ ¬(∃ (m : ℤ), (15 * y + 3) * (15 * y + 9) * (10 * y + 10) = n * m)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l1716_171661


namespace NUMINAMATH_CALUDE_dannys_collection_l1716_171605

theorem dannys_collection (initial_wrappers initial_caps found_wrappers found_caps : ℕ) 
  (h1 : initial_wrappers = 67)
  (h2 : initial_caps = 35)
  (h3 : found_wrappers = 18)
  (h4 : found_caps = 15) :
  (initial_wrappers + found_wrappers) - (initial_caps + found_caps) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dannys_collection_l1716_171605


namespace NUMINAMATH_CALUDE_speed_increase_time_reduction_l1716_171642

/-- Represents Vanya's speed to school -/
def speed : ℝ := by sorry

/-- Theorem stating the relationship between speed increase and time reduction -/
theorem speed_increase_time_reduction :
  (speed + 2) / speed = 2.5 →
  (speed + 4) / speed = 4 := by sorry

end NUMINAMATH_CALUDE_speed_increase_time_reduction_l1716_171642


namespace NUMINAMATH_CALUDE_log_23_between_consecutive_integers_l1716_171631

theorem log_23_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < Real.log 23 / Real.log 10) ∧ (Real.log 23 / Real.log 10 < b) ∧ (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_log_23_between_consecutive_integers_l1716_171631


namespace NUMINAMATH_CALUDE_abc_value_for_factored_polynomial_l1716_171634

/-- If a polynomial ax^2 + bx + c can be factored as (x-1)(x-2), then abc = -6 -/
theorem abc_value_for_factored_polynomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) →
  a * b * c = -6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_for_factored_polynomial_l1716_171634


namespace NUMINAMATH_CALUDE_polynomial_alternating_sum_l1716_171647

theorem polynomial_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_alternating_sum_l1716_171647


namespace NUMINAMATH_CALUDE_root_in_interval_l1716_171635

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1716_171635


namespace NUMINAMATH_CALUDE_sheets_in_stack_l1716_171693

/-- Given that 400 sheets of paper are 4 centimeters thick, 
    prove that a 14-inch high stack contains 3556 sheets. -/
theorem sheets_in_stack (sheets_in_4cm : ℕ) (thickness_4cm : ℝ) 
  (stack_height_inches : ℝ) (cm_per_inch : ℝ) :
  sheets_in_4cm = 400 →
  thickness_4cm = 4 →
  stack_height_inches = 14 →
  cm_per_inch = 2.54 →
  (stack_height_inches * cm_per_inch) / (thickness_4cm / sheets_in_4cm) = 3556 := by
  sorry

end NUMINAMATH_CALUDE_sheets_in_stack_l1716_171693


namespace NUMINAMATH_CALUDE_john_writing_years_l1716_171664

/-- Represents the number of months in a year -/
def months_per_year : ℕ := 12

/-- Represents the number of months it takes John to write a book -/
def months_per_book : ℕ := 2

/-- Represents the average earnings per book in dollars -/
def earnings_per_book : ℕ := 30000

/-- Represents the total earnings from writing in dollars -/
def total_earnings : ℕ := 3600000

/-- Calculates the number of years John has been writing -/
def years_writing : ℚ :=
  (total_earnings / earnings_per_book) / (months_per_year / months_per_book)

theorem john_writing_years :
  years_writing = 20 := by sorry

end NUMINAMATH_CALUDE_john_writing_years_l1716_171664


namespace NUMINAMATH_CALUDE_max_triangle_area_max_triangle_area_is_156_l1716_171602

/-- The maximum area of the triangle formed by the intersections of three lines in a coordinate plane. -/
theorem max_triangle_area : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (15, 0)
  let ℓ_A := {(x, y) : ℝ × ℝ | y = 2 * x}
  let ℓ_B := {(x, y) : ℝ × ℝ | x = 8}
  let ℓ_C := {(x, y) : ℝ × ℝ | y = -2 * (x - 15)}
  156

/-- The maximum area of the triangle is 156. -/
theorem max_triangle_area_is_156 : max_triangle_area = 156 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_max_triangle_area_is_156_l1716_171602


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l1716_171676

theorem difference_of_cubes_divisible_by_nine (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 9*k :=
sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_nine_l1716_171676


namespace NUMINAMATH_CALUDE_z_max_plus_z_min_l1716_171623

theorem z_max_plus_z_min (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2*y - 2*z = 4) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z' : ℝ, (x^2 + y^2 + z'^2 = 3 ∧ x + 2*y - 2*z' = 4) → z' ≤ z_max ∧ z' ≥ z_min) ∧
    z_max + z_min = -4 :=
sorry

end NUMINAMATH_CALUDE_z_max_plus_z_min_l1716_171623


namespace NUMINAMATH_CALUDE_task_probability_l1716_171608

/-- The probability that task 1 is completed on time -/
def prob_task1 : ℚ := 5/8

/-- The probability that task 2 is completed on time -/
def prob_task2 : ℚ := 3/5

/-- The probability that task 1 is completed on time but task 2 is not -/
def prob_task1_not_task2 : ℚ := prob_task1 * (1 - prob_task2)

theorem task_probability : prob_task1_not_task2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_task_probability_l1716_171608


namespace NUMINAMATH_CALUDE_unique_prime_generating_x_l1716_171672

theorem unique_prime_generating_x (x : ℕ+) 
  (h : Nat.Prime (x^5 + x + 1)) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_generating_x_l1716_171672


namespace NUMINAMATH_CALUDE_sum_g_32_neg_32_l1716_171655

/-- A function g defined as a polynomial of even degree -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

/-- Theorem stating that the sum of g(32) and g(-32) equals 6 -/
theorem sum_g_32_neg_32 (a b c : ℝ) (h : g a b c 32 = 3) :
  g a b c 32 + g a b c (-32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_32_neg_32_l1716_171655


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1716_171669

theorem fraction_zero_implies_x_equals_one (x : ℝ) : (x - 1) / (x + 2) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1716_171669


namespace NUMINAMATH_CALUDE_parallelogram_construction_l1716_171697

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the predicates
variable (lies_on : Point → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (is_center : Point → Point → Point → Point → Point → Prop)

-- State the theorem
theorem parallelogram_construction
  (l₁ l₂ l₃ l₄ : Line)
  (O : Point)
  (not_parallel : ¬ parallel l₁ l₂ ∧ ¬ parallel l₁ l₃ ∧ ¬ parallel l₁ l₄ ∧
                  ¬ parallel l₂ l₃ ∧ ¬ parallel l₂ l₄ ∧ ¬ parallel l₃ l₄)
  (O_not_on_lines : ¬ lies_on O l₁ ∧ ¬ lies_on O l₂ ∧ ¬ lies_on O l₃ ∧ ¬ lies_on O l₄) :
  ∃ (A B C D : Point),
    lies_on A l₁ ∧ lies_on B l₂ ∧ lies_on C l₃ ∧ lies_on D l₄ ∧
    is_center O A B C D :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_construction_l1716_171697


namespace NUMINAMATH_CALUDE_meeting_probability_approx_point_one_l1716_171652

/-- Object movement in a 2D plane -/
structure Object where
  x : ℤ
  y : ℤ

/-- Probability of movement in each direction -/
structure MoveProb where
  right : ℝ
  up : ℝ
  left : ℝ
  down : ℝ

/-- Calculate the probability of two objects meeting after n steps -/
def meetingProbability (a : Object) (c : Object) (aProb : MoveProb) (cProb : MoveProb) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of A and C meeting after 7 steps is approximately 0.10 -/
theorem meeting_probability_approx_point_one :
  let a := Object.mk 0 0
  let c := Object.mk 6 8
  let aProb := MoveProb.mk 0.5 0.5 0 0
  let cProb := MoveProb.mk 0.1 0.1 0.4 0.4
  abs (meetingProbability a c aProb cProb 7 - 0.1) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_meeting_probability_approx_point_one_l1716_171652


namespace NUMINAMATH_CALUDE_average_speed_two_walks_l1716_171680

theorem average_speed_two_walks 
  (v₁ v₂ t₁ t₂ : ℝ) 
  (h₁ : t₁ > 0) 
  (h₂ : t₂ > 0) :
  let d₁ := v₁ * t₁
  let d₂ := v₂ * t₂
  let total_distance := d₁ + d₂
  let total_time := t₁ + t₂
  (total_distance / total_time) = (v₁ * t₁ + v₂ * t₂) / (t₁ + t₂) := by
sorry

end NUMINAMATH_CALUDE_average_speed_two_walks_l1716_171680


namespace NUMINAMATH_CALUDE_orange_shelves_l1716_171630

/-- The number of oranges on the nth shelf -/
def oranges_on_shelf (n : ℕ) : ℕ := 3 + 5 * (n - 1)

/-- The total number of oranges on n shelves -/
def total_oranges (n : ℕ) : ℕ := n * (oranges_on_shelf 1 + oranges_on_shelf n) / 2

theorem orange_shelves :
  ∃ n : ℕ, n > 0 ∧ total_oranges n = 325 :=
sorry

end NUMINAMATH_CALUDE_orange_shelves_l1716_171630


namespace NUMINAMATH_CALUDE_smallest_prime_with_composite_odd_digit_sum_l1716_171691

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for composite numbers -/
def is_composite (n : ℕ) : Prop := sorry

theorem smallest_prime_with_composite_odd_digit_sum :
  (is_prime 997) ∧ 
  (is_composite (sum_of_digits 997)) ∧ 
  (sum_of_digits 997 % 2 = 1) ∧
  (∀ p < 997, is_prime p → ¬(is_composite (sum_of_digits p) ∧ sum_of_digits p % 2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_composite_odd_digit_sum_l1716_171691


namespace NUMINAMATH_CALUDE_no_integer_roots_l1716_171679

/-- Polynomial P(x) = x^2019 + 2x^2018 + 3x^2017 + ... + 2019x + 2020 -/
def P (x : ℤ) : ℤ := 
  (Finset.range 2020).sum (fun i => (i + 1) * x^(2019 - i))

theorem no_integer_roots : ∀ x : ℤ, P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1716_171679


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_is_eight_l1716_171670

def sixDigitNumber (x y : ℕ) : ℕ := 123400 + 10 * x + y

theorem sum_of_last_two_digits_is_eight 
  (x y : ℕ) 
  (h1 : x < 10 ∧ y < 10) 
  (h2 : sixDigitNumber x y % 8 = 0) 
  (h3 : sixDigitNumber x y % 9 = 0) :
  x + y = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_is_eight_l1716_171670


namespace NUMINAMATH_CALUDE_existence_of_x_l1716_171600

/-- A sequence of nonnegative integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The theorem to be proved -/
theorem existence_of_x (a : ℕ → ℕ) (h : ValidSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_l1716_171600


namespace NUMINAMATH_CALUDE_exam_candidates_l1716_171613

/-- Given an examination where the average marks obtained is 40 and the total marks are 2000,
    prove that the number of candidates who took the examination is 50. -/
theorem exam_candidates (average_marks : ℕ) (total_marks : ℕ) (h1 : average_marks = 40) (h2 : total_marks = 2000) :
  total_marks / average_marks = 50 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_l1716_171613


namespace NUMINAMATH_CALUDE_roots_transformation_l1716_171611

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 9*(3*r₁) + 162 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 9*(3*r₂) + 162 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 9*(3*r₃) + 162 = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_transformation_l1716_171611


namespace NUMINAMATH_CALUDE_f_properties_l1716_171665

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * (Real.cos x) ^ 2

theorem f_properties (a : ℝ) (h : f a (π / 4) = 0) :
  -- The smallest positive period of f(x) is π
  (∃ (T : ℝ), T > 0 ∧ T = π ∧ ∀ (x : ℝ), f a (x + T) = f a x) ∧
  -- The maximum value of f(x) on [π/24, 11π/24] is √2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≤ Real.sqrt 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = Real.sqrt 2 - 1) ∧
  -- The minimum value of f(x) on [π/24, 11π/24] is -√2/2 - 1
  (∀ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) → f a x ≥ -Real.sqrt 2 / 2 - 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (π / 24) (11 * π / 24) ∧ f a x = -Real.sqrt 2 / 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1716_171665


namespace NUMINAMATH_CALUDE_equal_sums_l1716_171699

-- Define the range of numbers
def N : ℕ := 999999

-- Function to determine if a number's nearest perfect square is odd
def nearest_square_odd (n : ℕ) : Prop := sorry

-- Function to determine if a number's nearest perfect square is even
def nearest_square_even (n : ℕ) : Prop := sorry

-- Sum of numbers with odd nearest perfect square
def sum_odd_group : ℕ := sorry

-- Sum of numbers with even nearest perfect square
def sum_even_group : ℕ := sorry

-- Theorem stating that the sums are equal
theorem equal_sums : sum_odd_group = sum_even_group := by sorry

end NUMINAMATH_CALUDE_equal_sums_l1716_171699


namespace NUMINAMATH_CALUDE_best_meeting_days_l1716_171689

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define the team members
inductive Member
| Alice
| Bob
| Cindy
| Dave
| Eve

-- Define the availability function
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cindy, Day.Wednesday => false
  | Member.Cindy, Day.Saturday => false
  | Member.Dave, Day.Monday => false
  | Member.Dave, Day.Tuesday => false
  | Member.Dave, Day.Thursday => false
  | Member.Eve, Day.Thursday => false
  | Member.Eve, Day.Friday => false
  | Member.Eve, Day.Saturday => false
  | _, _ => true

-- Define the function to count available members on a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cindy, Member.Dave, Member.Eve]).length

-- Theorem statement
theorem best_meeting_days :
  (∀ d : Day, availableCount d ≤ 3) ∧
  (availableCount Day.Monday = 3) ∧
  (availableCount Day.Tuesday = 3) ∧
  (availableCount Day.Wednesday = 3) ∧
  (availableCount Day.Friday = 3) ∧
  (∀ d : Day, availableCount d = 3 → d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Friday) :=
by sorry


end NUMINAMATH_CALUDE_best_meeting_days_l1716_171689


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l1716_171639

open Real

theorem sin_shift_equivalence (x : ℝ) :
  sin (2 * (x + π / 6)) = sin (2 * x + π / 3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l1716_171639


namespace NUMINAMATH_CALUDE_fraction_value_l1716_171645

theorem fraction_value (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1716_171645


namespace NUMINAMATH_CALUDE_envelope_stuffing_l1716_171659

/-- The total number of envelopes Rachel needs to stuff -/
def total_envelopes : ℕ := 1500

/-- The total time Rachel has to complete the task -/
def total_time : ℕ := 8

/-- The number of envelopes Rachel stuffs in the first hour -/
def first_hour : ℕ := 135

/-- The number of envelopes Rachel stuffs in the second hour -/
def second_hour : ℕ := 141

/-- The number of envelopes Rachel needs to stuff per hour to finish the job -/
def required_rate : ℕ := 204

theorem envelope_stuffing :
  total_envelopes = first_hour + second_hour + required_rate * (total_time - 2) := by
  sorry

end NUMINAMATH_CALUDE_envelope_stuffing_l1716_171659


namespace NUMINAMATH_CALUDE_solution_to_equation_l1716_171633

theorem solution_to_equation : ∃ x : ℝ, 12*x + 13*x + 16*x + 11 = 134 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1716_171633


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l1716_171626

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_units : Nat
  unpainted_center_size : Nat

/-- Calculates the number of unpainted unit cubes in the given PaintedCube -/
def count_unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem stating that a 6x6x6 cube with 2x2 unpainted centers has 72 unpainted unit cubes -/
theorem unpainted_cubes_count (cube : PaintedCube) 
  (h1 : cube.size = 6)
  (h2 : cube.total_units = 216)
  (h3 : cube.unpainted_center_size = 2) : 
  count_unpainted_cubes cube = 72 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l1716_171626


namespace NUMINAMATH_CALUDE_jays_family_percentage_l1716_171688

theorem jays_family_percentage (total_guests : ℕ) (female_percentage : ℚ) (jays_family_females : ℕ) : 
  total_guests = 240 → 
  female_percentage = 60 / 100 → 
  jays_family_females = 72 → 
  (jays_family_females : ℚ) / (female_percentage * total_guests) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jays_family_percentage_l1716_171688


namespace NUMINAMATH_CALUDE_friends_meeting_problem_l1716_171681

/-- Two friends walk in opposite directions and then run towards each other -/
theorem friends_meeting_problem 
  (misha_initial_speed : ℝ) 
  (vasya_initial_speed : ℝ) 
  (initial_walk_time : ℝ) 
  (speed_increase_factor : ℝ) :
  misha_initial_speed = 8 →
  vasya_initial_speed = misha_initial_speed / 2 →
  initial_walk_time = 3/4 →
  speed_increase_factor = 3/2 →
  ∃ (meeting_time total_distance : ℝ),
    meeting_time = 1/2 ∧ 
    total_distance = 18 :=
by sorry

end NUMINAMATH_CALUDE_friends_meeting_problem_l1716_171681


namespace NUMINAMATH_CALUDE_number_of_envelopes_l1716_171673

-- Define the weight of a single envelope in grams
def envelope_weight : ℝ := 8.5

-- Define the total weight in kilograms
def total_weight_kg : ℝ := 6.8

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem to prove
theorem number_of_envelopes : 
  (total_weight_kg * kg_to_g) / envelope_weight = 800 := by
  sorry

end NUMINAMATH_CALUDE_number_of_envelopes_l1716_171673


namespace NUMINAMATH_CALUDE_photo_collection_l1716_171677

theorem photo_collection (total photos : ℕ) (tim paul tom : ℕ) : 
  total = 152 →
  tim = total - 100 →
  paul = tim + 10 →
  total = tim + paul + tom →
  tom = 38 := by
sorry

end NUMINAMATH_CALUDE_photo_collection_l1716_171677


namespace NUMINAMATH_CALUDE_love_logic_l1716_171675

-- Define the propositions
variable (B : Prop) -- "I love Betty"
variable (J : Prop) -- "I love Jane"

-- State the theorem
theorem love_logic (h1 : B ∨ J) (h2 : B → J) : J ∧ ¬(B ↔ True) :=
  sorry


end NUMINAMATH_CALUDE_love_logic_l1716_171675


namespace NUMINAMATH_CALUDE_line_intersection_l1716_171603

theorem line_intersection : ∃! p : ℚ × ℚ, 
  5 * p.1 - 3 * p.2 = 7 ∧ 
  8 * p.1 + 2 * p.2 = 22 :=
by
  -- The point (40/17, 27/17) satisfies both equations
  have h1 : 5 * (40/17) - 3 * (27/17) = 7 := by sorry
  have h2 : 8 * (40/17) + 2 * (27/17) = 22 := by sorry

  -- Prove uniqueness
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1716_171603


namespace NUMINAMATH_CALUDE_ellipse_equation_circle_diameter_property_l1716_171606

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
structure EllipseConditions (a b c : ℝ) :=
  (a_gt_b : a > b)
  (b_gt_zero : b > 0)
  (perimeter : 2*c + 2*a = 6)
  (focal_distance : 2*c*b = a*b)
  (pythagoras : a^2 = b^2 + c^2)

-- Theorem for part 1
theorem ellipse_equation (a b c : ℝ) (h : EllipseConditions a b c) :
  a = 2 ∧ b = Real.sqrt 3 ∧ c = 1 :=
sorry

-- Theorem for part 2
theorem circle_diameter_property (m : ℝ) :
  let a := 2
  let b := Real.sqrt 3
  ∀ x₀ y₀ : ℝ, 
    ellipse a b x₀ y₀ → 
    x₀ ≠ 2 → 
    x₀ ≠ -2 → 
    (m - 2) * (x₀ - 2) + (y₀^2 / (x₀ + 2)) * (m + 2) = 0 →
    m = 14 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_circle_diameter_property_l1716_171606


namespace NUMINAMATH_CALUDE_ages_solution_l1716_171637

def mother_daughter_ages (daughter_age : ℕ) (mother_age : ℕ) : Prop :=
  (mother_age = daughter_age + 45) ∧
  (mother_age - 5 = 6 * (daughter_age - 5))

theorem ages_solution : ∃ (daughter_age : ℕ) (mother_age : ℕ),
  mother_daughter_ages daughter_age mother_age ∧
  daughter_age = 14 ∧ mother_age = 59 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l1716_171637


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1716_171632

theorem no_positive_integer_solution : 
  ¬∃ (n m : ℕ+), n^4 - m^4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1716_171632


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l1716_171612

/-- Given 2.5 bugs eating 4.5 flowers in total, the number of flowers consumed per bug is 1.8 -/
theorem bugs_eating_flowers (num_bugs : ℝ) (total_flowers : ℝ) 
    (h1 : num_bugs = 2.5) 
    (h2 : total_flowers = 4.5) : 
  total_flowers / num_bugs = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l1716_171612


namespace NUMINAMATH_CALUDE_log_inequality_implies_upper_bound_l1716_171620

theorem log_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, a < Real.log (|x - 3| + |x + 7|)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_upper_bound_l1716_171620


namespace NUMINAMATH_CALUDE_sector_central_angle_l1716_171618

theorem sector_central_angle (area : Real) (perimeter : Real) (r : Real) (θ : Real) :
  area = 1 ∧ perimeter = 4 ∧ area = (1/2) * r^2 * θ ∧ perimeter = 2*r + r*θ → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1716_171618


namespace NUMINAMATH_CALUDE_total_outfits_is_168_l1716_171695

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 7

/-- The number of hats available -/
def num_hats : ℕ := 2

/-- The number of hat options (including not wearing a hat) -/
def hat_options : ℕ := num_hats + 1

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_ties * hat_options

/-- Theorem stating that the total number of outfits is 168 -/
theorem total_outfits_is_168 : total_outfits = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_outfits_is_168_l1716_171695


namespace NUMINAMATH_CALUDE_souvenirs_for_45_colleagues_l1716_171629

def souvenir_pattern : Nat → Nat
| 0 => 1
| 1 => 3
| 2 => 5
| 3 => 7
| n + 4 => souvenir_pattern n

def total_souvenirs (n : Nat) : Nat :=
  (List.range n).map souvenir_pattern |>.sum

theorem souvenirs_for_45_colleagues :
  total_souvenirs 45 = 177 := by
  sorry

end NUMINAMATH_CALUDE_souvenirs_for_45_colleagues_l1716_171629


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1716_171640

/-- Given an arithmetic sequence with common difference 2 and where a₁, a₃, and a₄ form a geometric sequence, prove that a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  (∃ r, a 3 = r * a 1 ∧ a 4 = r * a 3) →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1716_171640


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l1716_171636

/-- Prove that the height of a larger cuboid is 2 meters given specific conditions -/
theorem larger_cuboid_height (small_length small_width small_height : ℝ)
  (large_length large_width : ℝ) (num_small_cuboids : ℝ) :
  small_length = 6 →
  small_width = 4 →
  small_height = 3 →
  large_length = 18 →
  large_width = 15 →
  num_small_cuboids = 7.5 →
  ∃ (large_height : ℝ),
    num_small_cuboids * (small_length * small_width * small_height) =
      large_length * large_width * large_height ∧
    large_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l1716_171636


namespace NUMINAMATH_CALUDE_final_coin_count_l1716_171694

/-- Represents the number of coins in the jar at each hour -/
def coin_count : Fin 11 → ℕ
| 0 => 0  -- Initial state
| 1 => 20
| 2 => coin_count 1 + 30
| 3 => coin_count 2 + 30
| 4 => coin_count 3 + 40
| 5 => coin_count 4 - (coin_count 4 * 20 / 100)
| 6 => coin_count 5 + 50
| 7 => coin_count 6 + 60
| 8 => coin_count 7 - (coin_count 7 / 5)
| 9 => coin_count 8 + 70
| 10 => coin_count 9 - (coin_count 9 * 15 / 100)

theorem final_coin_count : coin_count 10 = 200 := by
  sorry

end NUMINAMATH_CALUDE_final_coin_count_l1716_171694


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l1716_171610

/-- The minimum distance between a point on the line y = (5/12)x - 11 and a point on the parabola y = x² is 6311/624 -/
theorem min_distance_line_parabola :
  let line := λ x : ℝ => (5/12) * x - 11
  let parabola := λ x : ℝ => x^2
  ∃ (d : ℝ), d = 6311/624 ∧
    ∀ (x₁ x₂ : ℝ),
      d ≤ Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l1716_171610


namespace NUMINAMATH_CALUDE_cube_face_sum_l1716_171624

/-- Represents the six face values of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- Calculates the sum of vertex labels for a given set of cube faces -/
def vertexLabelSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- Theorem stating the sum of face values given the conditions -/
theorem cube_face_sum (faces : CubeFaces)
  (h1 : vertexLabelSum faces = 2002)
  (h2 : faces.a + faces.d = 22) :
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f = 42 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l1716_171624


namespace NUMINAMATH_CALUDE_earrings_price_decrease_l1716_171690

/-- Given a pair of earrings with the following properties:
  - Purchase price: $240
  - Original markup: 25% of the selling price
  - Gross profit after price decrease: $16
  Prove that the percentage decrease in the selling price is 5% -/
theorem earrings_price_decrease (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 240 →
  markup_percentage = 0.25 →
  gross_profit = 16 →
  let original_selling_price := purchase_price / (1 - markup_percentage)
  let new_selling_price := original_selling_price - gross_profit
  let price_decrease := original_selling_price - new_selling_price
  let percentage_decrease := price_decrease / original_selling_price * 100
  percentage_decrease = 5 := by
  sorry

end NUMINAMATH_CALUDE_earrings_price_decrease_l1716_171690


namespace NUMINAMATH_CALUDE_cousins_age_sum_l1716_171604

theorem cousins_age_sum : ∀ (a b c : ℕ),
  a < 10 ∧ b < 10 ∧ c < 10 →  -- single-digit positive integers
  a ≠ b ∧ b ≠ c ∧ a ≠ c →     -- distinct
  (a < c ∧ b < c) →           -- one cousin is older than the other two
  a * b = 18 →                -- product of younger two
  c * min a b = 28 →          -- product of oldest and youngest
  a + b + c = 18 :=           -- sum of all three
by sorry

end NUMINAMATH_CALUDE_cousins_age_sum_l1716_171604


namespace NUMINAMATH_CALUDE_bobs_final_score_l1716_171625

/-- Bob's math knowledge competition score calculation -/
theorem bobs_final_score :
  let points_per_correct : ℕ := 5
  let points_per_incorrect : ℕ := 2
  let correct_answers : ℕ := 18
  let incorrect_answers : ℕ := 2
  let total_score := points_per_correct * correct_answers - points_per_incorrect * incorrect_answers
  total_score = 86 := by
  sorry

end NUMINAMATH_CALUDE_bobs_final_score_l1716_171625


namespace NUMINAMATH_CALUDE_little_john_money_l1716_171685

/-- Calculates the remaining money after spending on sweets and giving to friends -/
def remaining_money (initial : ℚ) (spent_on_sweets : ℚ) (given_to_each_friend : ℚ) (num_friends : ℕ) : ℚ :=
  initial - spent_on_sweets - (given_to_each_friend * num_friends)

/-- Theorem stating that given the specific amounts, the remaining money is $2.05 -/
theorem little_john_money : 
  remaining_money 5.10 1.05 1.00 2 = 2.05 := by
  sorry

#eval remaining_money 5.10 1.05 1.00 2

end NUMINAMATH_CALUDE_little_john_money_l1716_171685


namespace NUMINAMATH_CALUDE_parallel_lines_from_perpendicular_to_parallel_planes_l1716_171668

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularLinePlane : Line → Plane → Prop)

-- Define the property of two lines being non-coincident
variable (nonCoincident : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_from_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line) :
  parallelPlanes α β →
  nonCoincident a b →
  perpendicularLinePlane a α →
  perpendicularLinePlane b β →
  parallelLines a b :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_from_perpendicular_to_parallel_planes_l1716_171668


namespace NUMINAMATH_CALUDE_area_FYG_is_86_4_l1716_171696

/-- A trapezoid with the given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  area : ℝ

/-- The area of triangle FYG in the given trapezoid -/
def area_FYG (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that the area of triangle FYG is 86.4 square units -/
theorem area_FYG_is_86_4 (t : Trapezoid) 
  (h1 : t.EF = 24)
  (h2 : t.GH = 36)
  (h3 : t.area = 360) :
  area_FYG t = 86.4 := by sorry

end NUMINAMATH_CALUDE_area_FYG_is_86_4_l1716_171696


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l1716_171678

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) :
  p_blue = 0.25 →
  p_green = 0.4 →
  p_blue + p_green + p_purple = 1 →
  p_purple = 0.35 := by
sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l1716_171678


namespace NUMINAMATH_CALUDE_reciprocal_product_l1716_171615

theorem reciprocal_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) :
  (1 / x) * (1 / y) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_product_l1716_171615


namespace NUMINAMATH_CALUDE_cubic_inequality_l1716_171663

theorem cubic_inequality (x : ℝ) : x^3 - 9*x^2 + 36*x > -16*x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1716_171663


namespace NUMINAMATH_CALUDE_train_crossing_time_l1716_171657

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length > 0 → train_speed_kmh > 0 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 
  (train_length / (train_speed_kmh * (5 / 18))) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1716_171657


namespace NUMINAMATH_CALUDE_bottle_volume_is_one_and_half_quarts_l1716_171644

/-- Represents the daily water consumption of Tim -/
structure DailyWaterConsumption where
  bottles : ℕ := 2
  additional_ounces : ℕ := 20

/-- Represents the weekly water consumption in ounces -/
def weekly_ounces : ℕ := 812

/-- Conversion factor from ounces to quarts -/
def ounces_per_quart : ℕ := 32

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that each bottle contains 1.5 quarts of water -/
theorem bottle_volume_is_one_and_half_quarts 
  (daily : DailyWaterConsumption) 
  (h1 : daily.bottles = 2) 
  (h2 : daily.additional_ounces = 20) :
  (weekly_ounces : ℚ) / (ounces_per_quart * days_per_week * daily.bottles : ℚ) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_bottle_volume_is_one_and_half_quarts_l1716_171644


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l1716_171654

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l1716_171654


namespace NUMINAMATH_CALUDE_hill_height_correct_l1716_171656

/-- The height of the hill in feet -/
def hill_height : ℝ := 900

/-- The uphill speed in feet per second -/
def uphill_speed : ℝ := 9

/-- The downhill speed in feet per second -/
def downhill_speed : ℝ := 12

/-- The total time to run up and down the hill in seconds -/
def total_time : ℝ := 175

/-- Theorem stating that the given hill height satisfies the conditions -/
theorem hill_height_correct : 
  hill_height / uphill_speed + hill_height / downhill_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_hill_height_correct_l1716_171656


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l1716_171660

/-- Parabola structure -/
structure Parabola where
  f : ℝ → ℝ
  eq : ∀ x, f x ^ 2 = 8 * x

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y ^ 2 = 8 * x

/-- Theorem about two points on a parabola -/
theorem parabola_points_theorem (p : Parabola) 
    (A B : PointOnParabola p) (F : ℝ × ℝ) :
  A.y + B.y = 8 →
  F = (2, 0) →
  (B.y - A.y) / (B.x - A.x) = 1 ∧
  ((A.x - F.1) ^ 2 + (A.y - F.2) ^ 2) ^ (1/2 : ℝ) +
  ((B.x - F.1) ^ 2 + (B.y - F.2) ^ 2) ^ (1/2 : ℝ) = 16 :=
by sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l1716_171660


namespace NUMINAMATH_CALUDE_unique_permutations_count_l1716_171641

/-- The number of elements in our multiset -/
def n : ℕ := 5

/-- The number of occurrences of the digit 3 -/
def k₁ : ℕ := 3

/-- The number of occurrences of the digit 7 -/
def k₂ : ℕ := 2

/-- The theorem stating that the number of unique permutations of our multiset is 10 -/
theorem unique_permutations_count : (n.factorial) / (k₁.factorial * k₂.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_count_l1716_171641


namespace NUMINAMATH_CALUDE_concert_duration_is_80_minutes_l1716_171649

/-- Calculates the total duration of a concert given the number of songs, 
    duration of regular songs, duration of the special song, and intermission time. -/
def concertDuration (numSongs : ℕ) (regularSongDuration : ℕ) (specialSongDuration : ℕ) (intermissionTime : ℕ) : ℕ :=
  (numSongs - 1) * regularSongDuration + specialSongDuration + intermissionTime

/-- Proves that the concert duration is 80 minutes given the specified conditions. -/
theorem concert_duration_is_80_minutes :
  concertDuration 13 5 10 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_is_80_minutes_l1716_171649


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l1716_171662

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- Original quadrilateral
  area : ℝ
  -- Lengths of sides
  wz : ℝ
  zx : ℝ
  xy : ℝ
  yw : ℝ
  -- Conditions for extended sides
  wz_extended : ℝ
  zx_extended : ℝ
  xy_extended : ℝ
  yw_extended : ℝ
  -- Conditions for double length
  wz_double : wz_extended = 2 * wz
  zx_double : zx_extended = 2 * zx
  xy_double : xy_extended = 2 * xy
  yw_double : yw_extended = 2 * yw

/-- Theorem stating the relationship between areas of original and extended quadrilaterals -/
theorem extended_quadrilateral_area 
  (q : ExtendedQuadrilateral) : 
  ∃ (extended_area : ℝ), extended_area = 9 * q.area := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l1716_171662


namespace NUMINAMATH_CALUDE_triangle_inequality_with_sum_zero_l1716_171683

theorem triangle_inequality_with_sum_zero (a b c p q r : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (a + b > c) → (b + c > a) → (c + a > b) → 
  (p + q + r = 0) → 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_sum_zero_l1716_171683


namespace NUMINAMATH_CALUDE_sequence_properties_l1716_171653

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of the first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of the arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of T_n as the sum of the first n terms of a_n * b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  (∀ k, S k + a k = 1) ∧
  (b 1 + b 2 = b 3) ∧
  (b 3 = 3) →
  (S n = 1 - (1/2)^n) ∧
  (T n = 2 - (n + 2) * (1/2)^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1716_171653


namespace NUMINAMATH_CALUDE_vowel_classification_l1716_171682

-- Define the set of all English letters
def EnglishLetters : Type := Fin 26

-- Define the categories
inductive Category
| one
| two
| three
| four
| five

-- Define the classification function
def classify : EnglishLetters → Category := sorry

-- Define the vowels
def vowels : Fin 5 → EnglishLetters := sorry

-- Theorem statement
theorem vowel_classification :
  (classify (vowels 0) = Category.four) ∧
  (classify (vowels 1) = Category.three) ∧
  (classify (vowels 2) = Category.one) ∧
  (classify (vowels 3) = Category.one) ∧
  (classify (vowels 4) = Category.four) := by
  sorry

end NUMINAMATH_CALUDE_vowel_classification_l1716_171682


namespace NUMINAMATH_CALUDE_inequality_proof_l1716_171601

theorem inequality_proof (x : ℝ) (h : x ≥ 0) :
  1 + x^2006 ≥ (2*x)^2005 / (1+x)^2004 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1716_171601


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1716_171638

theorem arithmetic_sequence_count :
  let a : ℤ := -5  -- First term
  let l : ℤ := 85  -- Last term
  let d : ℤ := 5   -- Common difference
  (l - a) / d + 1 = 19
  :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1716_171638


namespace NUMINAMATH_CALUDE_competition_results_l1716_171643

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def xiao_li_score : ℕ := 7
def xiao_zhang_score : ℕ := 7

def mode (l : List ℕ) : ℕ := sorry
def average (l : List ℕ) : ℚ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 7.1) ∧
  (median seventh_grade_scores = 6) ∧
  (median eighth_grade_scores = 7.5) ∧
  (xiao_li_score > median seventh_grade_scores) ∧
  (xiao_zhang_score < median eighth_grade_scores) := by
  sorry

#check competition_results

end NUMINAMATH_CALUDE_competition_results_l1716_171643


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1716_171698

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

-- Define the properties of the hyperbola
def has_focus (h : Hyperbola) (fx fy : ℝ) : Prop :=
  h.a^2 + h.b^2 = fx^2 + fy^2

def passes_through (h : Hyperbola) (px py : ℝ) : Prop :=
  h.eq px py

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  has_focus h (Real.sqrt 6) 0 →
  passes_through h (-5) 2 →
  h.a^2 = 5 ∧ h.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1716_171698


namespace NUMINAMATH_CALUDE_equal_sum_and_product_sets_l1716_171628

theorem equal_sum_and_product_sets : ∃ (S₁ S₂ : Finset ℕ),
  S₁ ≠ S₂ ∧
  S₁.card = 8 ∧
  S₂.card = 8 ∧
  (S₁.sum id = S₁.prod id) ∧
  (S₂.sum id = S₂.prod id) :=
by
  sorry

end NUMINAMATH_CALUDE_equal_sum_and_product_sets_l1716_171628


namespace NUMINAMATH_CALUDE_seniority_ranking_l1716_171674

-- Define the colleagues
inductive Colleague
| Julia
| Kevin
| Lana

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the most senior and least senior
def most_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior c other

def least_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior other c

-- Define the statements
def statement_I : Prop := most_senior Colleague.Kevin
def statement_II : Prop := least_senior Colleague.Lana
def statement_III : Prop := ¬(least_senior Colleague.Julia)

-- Main theorem
theorem seniority_ranking :
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) →
  (more_senior Colleague.Kevin Colleague.Lana ∧
   more_senior Colleague.Lana Colleague.Julia) :=
by sorry

end NUMINAMATH_CALUDE_seniority_ranking_l1716_171674
