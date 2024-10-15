import Mathlib

namespace NUMINAMATH_CALUDE_dispatch_plans_count_l1198_119831

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students --/
def total_students : ℕ := 6

/-- The number of students participating in the campaign --/
def participating_students : ℕ := 4

/-- The number of students participating on Sunday --/
def sunday_students : ℕ := 2

/-- The number of students participating on Friday --/
def friday_students : ℕ := 1

/-- The number of students participating on Saturday --/
def saturday_students : ℕ := 1

theorem dispatch_plans_count :
  (choose total_students sunday_students) *
  (choose (total_students - sunday_students) friday_students) *
  (choose (total_students - sunday_students - friday_students) saturday_students) = 180 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l1198_119831


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1198_119805

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1198_119805


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1198_119823

/-- Given two lines in the 2D plane represented by their equations:
    ax + by + c = 0 and dx + ey + f = 0,
    this function returns true if the lines are perpendicular. -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- Given a line ax + by + c = 0 and a point (x₀, y₀),
    this function returns true if the point lies on the line. -/
def point_on_line (a b c x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 4 (-3) 2 3 4 1 ∧
  point_on_line 4 (-3) 2 1 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1198_119823


namespace NUMINAMATH_CALUDE_expression_equality_l1198_119857

theorem expression_equality : ∀ x : ℤ, x = 3 ∨ x = -3 →
  6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1198_119857


namespace NUMINAMATH_CALUDE_cory_needs_78_dollars_l1198_119845

/-- The amount of additional money Cory needs to buy two packs of candies -/
def additional_money_needed (initial_money : ℚ) (candy_pack_cost : ℚ) (num_packs : ℕ) : ℚ :=
  candy_pack_cost * num_packs - initial_money

/-- Theorem stating that Cory needs $78 more to buy two packs of candies -/
theorem cory_needs_78_dollars :
  additional_money_needed 20 49 2 = 78 := by
  sorry

end NUMINAMATH_CALUDE_cory_needs_78_dollars_l1198_119845


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1198_119894

theorem second_polygon_sides (perimeter : ℝ) (side_length_second : ℝ) : 
  perimeter > 0 → side_length_second > 0 →
  perimeter = 50 * (3 * side_length_second) →
  perimeter = 150 * side_length_second := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1198_119894


namespace NUMINAMATH_CALUDE_allison_wins_prob_l1198_119854

/-- Represents a die with a fixed number of faces -/
structure Die where
  faces : List Nat

/-- Allison's die always shows 5 -/
def allison_die : Die := ⟨[5, 5, 5, 5, 5, 5]⟩

/-- Brian's die has faces numbered 1, 2, 3, 4, 4, 5, 5, and 6 -/
def brian_die : Die := ⟨[1, 2, 3, 4, 4, 5, 5, 6]⟩

/-- Noah's die has faces numbered 2, 2, 6, 6, 3, 3, 7, and 7 -/
def noah_die : Die := ⟨[2, 2, 6, 6, 3, 3, 7, 7]⟩

/-- Calculate the probability of rolling less than a given number on a die -/
def prob_less_than (d : Die) (n : Nat) : Rat :=
  (d.faces.filter (· < n)).length / d.faces.length

/-- The probability that Allison's roll is greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_less_than brian_die 5 * prob_less_than noah_die 5 = 5 / 16 := by
  sorry


end NUMINAMATH_CALUDE_allison_wins_prob_l1198_119854


namespace NUMINAMATH_CALUDE_oliver_stickers_l1198_119862

theorem oliver_stickers (initial_stickers : ℕ) (h1 : initial_stickers = 135) :
  let remaining_after_use := initial_stickers - (initial_stickers / 3)
  let given_away := remaining_after_use * 2 / 5
  let kept := remaining_after_use - given_away
  kept = 54 := by sorry

end NUMINAMATH_CALUDE_oliver_stickers_l1198_119862


namespace NUMINAMATH_CALUDE_unique_solution_l1198_119892

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1

/-- The theorem stating that the function g(x) = 2x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, FunctionalEquation g → (∀ x : ℝ, g x = 2 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1198_119892


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1198_119821

/-- A polynomial of the form ax^2 + bx + c is a perfect square trinomial if and only if
    there exist real numbers p and q such that ax^2 + bx + c = (px + q)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- The main theorem stating the condition for the given polynomial to be a perfect square trinomial -/
theorem perfect_square_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (-(k-1)) 9 ↔ k = 13 ∨ k = -11 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_condition_l1198_119821


namespace NUMINAMATH_CALUDE_final_breath_holding_time_l1198_119895

def breath_holding_progress (initial_time : ℝ) : ℝ :=
  let week1 := initial_time * 2
  let week2 := week1 * 2
  let week3 := week2 * 1.5
  week3

theorem final_breath_holding_time :
  breath_holding_progress 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_final_breath_holding_time_l1198_119895


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l1198_119885

theorem opposite_numbers_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  (a + b - 1) * (a / b + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l1198_119885


namespace NUMINAMATH_CALUDE_escalator_speed_increase_l1198_119848

theorem escalator_speed_increase (total_steps : ℕ) (first_climb : ℕ) (second_climb : ℕ)
  (h_total : total_steps = 125)
  (h_first : first_climb = 45)
  (h_second : second_climb = 55)
  (h_first_valid : first_climb < total_steps)
  (h_second_valid : second_climb < total_steps) :
  (second_climb : ℚ) / first_climb * (total_steps - first_climb : ℚ) / (total_steps - second_climb) = 88 / 63 :=
by sorry

end NUMINAMATH_CALUDE_escalator_speed_increase_l1198_119848


namespace NUMINAMATH_CALUDE_combined_future_age_l1198_119850

-- Define the current age of Hurley
def hurley_current_age : ℕ := 14

-- Define the age difference between Richard and Hurley
def age_difference : ℕ := 20

-- Define the number of years into the future
def years_future : ℕ := 40

-- Theorem to prove
theorem combined_future_age :
  (hurley_current_age + years_future) + (hurley_current_age + age_difference + years_future) = 128 := by
  sorry

end NUMINAMATH_CALUDE_combined_future_age_l1198_119850


namespace NUMINAMATH_CALUDE_triangle_properties_l1198_119824

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

theorem triangle_properties (t : Triangle) 
  (m : Vector2D) (n : Vector2D) (angle_mn : ℝ) (area : ℝ) :
  m.x = Real.cos (t.C / 2) ∧ 
  m.y = Real.sin (t.C / 2) ∧
  n.x = Real.cos (t.C / 2) ∧ 
  n.y = -Real.sin (t.C / 2) ∧
  angle_mn = π / 3 ∧
  t.c = 7 / 2 ∧
  area = 3 * Real.sqrt 3 / 2 →
  t.C = π / 3 ∧ t.a + t.b = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1198_119824


namespace NUMINAMATH_CALUDE_percentage_change_xyz_l1198_119886

theorem percentage_change_xyz (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0) :
  let x' := 0.8 * x
  let y' := 0.8 * y
  let z' := 1.1 * z
  (x' * y' * z' - x * y * z) / (x * y * z) = -0.296 :=
by sorry

end NUMINAMATH_CALUDE_percentage_change_xyz_l1198_119886


namespace NUMINAMATH_CALUDE_max_subway_riders_l1198_119842

theorem max_subway_riders (total : ℕ) (part_time full_time : ℕ → ℕ) : 
  total = 251 →
  (∀ p f, part_time p + full_time f = total) →
  (∀ p, part_time p ≤ total) →
  (∀ f, full_time f ≤ total) →
  (∀ p, (part_time p) % 11 = 0) →
  (∀ f, (full_time f) % 13 = 0) →
  (∃ max : ℕ, ∀ p f, 
    part_time p + full_time f = total → 
    (part_time p) / 11 + (full_time f) / 13 ≤ max ∧
    (∃ p' f', part_time p' + full_time f' = total ∧ 
              (part_time p') / 11 + (full_time f') / 13 = max)) →
  (∃ p f, part_time p + full_time f = total ∧ 
          (part_time p) / 11 + (full_time f) / 13 = 22) :=
sorry

end NUMINAMATH_CALUDE_max_subway_riders_l1198_119842


namespace NUMINAMATH_CALUDE_days_without_class_total_course_days_course_duration_proof_l1198_119877

/- Define the parameters of the problem -/
def total_hours : ℕ := 30
def class_duration : ℕ := 1
def afternoons_without_class : ℕ := 20
def mornings_without_class : ℕ := 18

/- Define the theorems to be proved -/
theorem days_without_class : ℕ := by sorry

theorem total_course_days : ℕ := by sorry

/- Main theorem combining both results -/
theorem course_duration_proof :
  (days_without_class = 4) ∧ (total_course_days = 34) := by
  sorry

end NUMINAMATH_CALUDE_days_without_class_total_course_days_course_duration_proof_l1198_119877


namespace NUMINAMATH_CALUDE_hair_extension_ratio_l1198_119855

theorem hair_extension_ratio : 
  let initial_length : ℕ := 18
  let final_length : ℕ := 36
  (final_length : ℚ) / (initial_length : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hair_extension_ratio_l1198_119855


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l1198_119843

/-- The number of different pots of flowers --/
def total_pots : ℕ := 7

/-- The number of pots to be selected --/
def selected_pots : ℕ := 5

/-- The number of pots not allowed in the center --/
def restricted_pots : ℕ := 2

/-- The function to calculate the number of arrangements --/
def flower_arrangements (n m k : ℕ) : ℕ := sorry

theorem flower_arrangement_count :
  flower_arrangements total_pots selected_pots restricted_pots = 1800 := by sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l1198_119843


namespace NUMINAMATH_CALUDE_total_chase_time_distance_equality_at_capture_l1198_119881

/-- Represents the chase scenario between Black Cat Detective and One-Ear --/
structure ChaseScenario where
  v : ℝ  -- One-Ear's speed
  initial_time : ℝ  -- Time before chase begins
  chase_time : ℝ  -- Time of chase

/-- Conditions of the chase scenario --/
def chase_conditions (s : ChaseScenario) : Prop :=
  s.initial_time = 13 ∧
  s.chase_time = 1 ∧
  s.v > 0

/-- The theorem stating the total time of the chase --/
theorem total_chase_time (s : ChaseScenario) 
  (h : chase_conditions s) : s.initial_time + s.chase_time = 14 := by
  sorry

/-- The theorem proving the distance equality at the point of capture --/
theorem distance_equality_at_capture (s : ChaseScenario) 
  (h : chase_conditions s) : 
  (5 * s.v + s.v) * s.initial_time = (7.5 * s.v - s.v) * s.chase_time := by
  sorry

end NUMINAMATH_CALUDE_total_chase_time_distance_equality_at_capture_l1198_119881


namespace NUMINAMATH_CALUDE_gcd_sequence_is_one_l1198_119859

theorem gcd_sequence_is_one (n : ℕ) : 
  Nat.gcd ((7^n - 1) / 6) ((7^(n+1) - 1) / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_sequence_is_one_l1198_119859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l1198_119879

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2023rd_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 6 2 = p + 6)
  (h2 : arithmetic_sequence p 6 3 = 4*p - q)
  (h3 : arithmetic_sequence p 6 4 = 4*p + q) :
  arithmetic_sequence p 6 2023 = 12137 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023rd_term_l1198_119879


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l1198_119810

theorem sqrt_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l1198_119810


namespace NUMINAMATH_CALUDE_set_size_from_averages_l1198_119878

theorem set_size_from_averages (S : Finset ℝ) (sum : ℝ) (n : ℕ) :
  sum = S.sum (λ x => x) →
  n = S.card →
  sum / n = 6.2 →
  (sum + 7) / n = 6.9 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_set_size_from_averages_l1198_119878


namespace NUMINAMATH_CALUDE_infiniteSeries_eq_three_halves_l1198_119844

/-- The sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem stating that the sum of the infinite series Σ(k/(3^k)) for k from 1 to ∞ is equal to 3/2 -/
theorem infiniteSeries_eq_three_halves : infiniteSeries = 3/2 := by sorry

end NUMINAMATH_CALUDE_infiniteSeries_eq_three_halves_l1198_119844


namespace NUMINAMATH_CALUDE_ingrid_income_calculation_l1198_119896

-- Define the given constants
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def john_income : ℝ := 58000
def combined_tax_rate : ℝ := 0.3554

-- Define Ingrid's income as a variable
def ingrid_income : ℝ := 72000

-- Theorem statement
theorem ingrid_income_calculation :
  ingrid_income = 72000 ∧
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_ingrid_income_calculation_l1198_119896


namespace NUMINAMATH_CALUDE_negation_equivalence_l1198_119893

variable (a : ℝ)

def original_proposition : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem negation_equivalence :
  (¬ original_proposition a) ↔ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1198_119893


namespace NUMINAMATH_CALUDE_expansion_coefficient_equals_negative_eighty_l1198_119813

/-- The coefficient of the term containing x in the expansion of (2√x - 1/x)^n -/
def coefficient (n : ℕ) : ℤ :=
  (-1)^((n-2)/3) * 2^((2*n+2)/3) * (n.choose ((n-2)/3))

theorem expansion_coefficient_equals_negative_eighty (n : ℕ) :
  coefficient n = -80 → n = 5 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equals_negative_eighty_l1198_119813


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1198_119874

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The original number we want to express in scientific notation -/
def original_number : ℝ := 0.00000164

/-- The scientific notation representation we want to prove is correct -/
def scientific_rep : ScientificNotation := {
  coefficient := 1.64
  exponent := -6
  coefficient_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1198_119874


namespace NUMINAMATH_CALUDE_six_digit_number_theorem_l1198_119838

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (a b c d e f : ℕ),
    n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧
    a ≠ 0 ∧ 
    10000 * b + 1000 * c + 100 * d + 10 * e + f + 100000 * a = 3 * n

theorem six_digit_number_theorem :
  ∀ n : ℕ, is_valid_number n → (n = 142857 ∨ n = 285714) :=
sorry

end NUMINAMATH_CALUDE_six_digit_number_theorem_l1198_119838


namespace NUMINAMATH_CALUDE_zero_in_interval_l1198_119807

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log (1/2) + x - a

theorem zero_in_interval (a : ℝ) :
  a ∈ Set.Ioo 1 3 →
  ∃ x ∈ Set.Ioo 2 8, f a x = 0 ∧
  ¬(∀ a : ℝ, (∃ x ∈ Set.Ioo 2 8, f a x = 0) → a ∈ Set.Ioo 1 3) :=
by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1198_119807


namespace NUMINAMATH_CALUDE_sandbag_weight_sandbag_problem_l1198_119864

theorem sandbag_weight (capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : ℝ :=
  let sand_weight := capacity * fill_percentage
  let extra_weight := sand_weight * weight_increase
  sand_weight + extra_weight

theorem sandbag_problem :
  sandbag_weight 250 0.8 0.4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_sandbag_problem_l1198_119864


namespace NUMINAMATH_CALUDE_equation_solution_range_l1198_119825

theorem equation_solution_range (x k : ℝ) : 2 * x + 3 * k = 1 → x < 0 → k > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1198_119825


namespace NUMINAMATH_CALUDE_gabrielle_blue_jays_eq_three_l1198_119846

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ :=
  let gabrielle_robins : ℕ := 5
  let gabrielle_cardinals : ℕ := 4
  let chase_robins : ℕ := 2
  let chase_blue_jays : ℕ := 3
  let chase_cardinals : ℕ := 5
  let chase_total : ℕ := chase_robins + chase_blue_jays + chase_cardinals
  let gabrielle_total : ℕ := chase_total + chase_total / 5
  gabrielle_total - gabrielle_robins - gabrielle_cardinals

theorem gabrielle_blue_jays_eq_three : gabrielle_blue_jays = 3 := by
  sorry

end NUMINAMATH_CALUDE_gabrielle_blue_jays_eq_three_l1198_119846


namespace NUMINAMATH_CALUDE_line_direction_vector_l1198_119820

/-- The direction vector of a line y = (2x - 6)/5 parameterized as [x, y] = [4, 0] + t * d,
    where t is the distance between [x, y] and [4, 0] for x ≥ 4. -/
theorem line_direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ (x y t : ℝ), x ≥ 4 →
      y = (2 * x - 6) / 5 →
      (x, y) = (4, 0) + t • d →
      t = Real.sqrt ((x - 4)^2 + y^2)) →
    d = (5 / Real.sqrt 29, 10 / Real.sqrt 29) := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1198_119820


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l1198_119868

theorem linear_equation_equivalence (x y : ℝ) :
  (3 * x - y + 5 = 0) ↔ (y = 3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l1198_119868


namespace NUMINAMATH_CALUDE_product_sum_fractions_l1198_119835

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1/6) = 57 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l1198_119835


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l1198_119899

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) -/
theorem circle_equation_through_points :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 4*x - 6*y = 0) ↔
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l1198_119899


namespace NUMINAMATH_CALUDE_element_in_set_l1198_119897

theorem element_in_set (a b : Type) : a ∈ ({a, b} : Set Type) := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1198_119897


namespace NUMINAMATH_CALUDE_todd_spending_proof_l1198_119829

/-- Calculates the total amount Todd spent given the prices of items, discount rate, and tax rate -/
def todd_spending (candy_price cookies_price soda_price : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let discounted_candy := candy_price * (1 - discount_rate)
  let subtotal := discounted_candy + cookies_price + soda_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Proves that Todd's total spending is $5.53 given the problem conditions -/
theorem todd_spending_proof :
  todd_spending 1.14 2.39 1.75 0.1 0.07 = 5.53 := by
  sorry

end NUMINAMATH_CALUDE_todd_spending_proof_l1198_119829


namespace NUMINAMATH_CALUDE_combined_mass_of_individuals_l1198_119809

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- The length of the boat in meters -/
def boat_length : ℝ := 4

/-- The breadth of the boat in meters -/
def boat_breadth : ℝ := 3

/-- The depth the boat sinks when the first person gets on, in meters -/
def first_person_depth : ℝ := 0.01

/-- The additional depth the boat sinks when the second person gets on, in meters -/
def second_person_depth : ℝ := 0.02

/-- Calculates the mass of water displaced by the boat sinking to a given depth -/
def water_mass (depth : ℝ) : ℝ :=
  boat_length * boat_breadth * depth * water_density

theorem combined_mass_of_individuals :
  water_mass (first_person_depth + second_person_depth) = 360 := by
  sorry

end NUMINAMATH_CALUDE_combined_mass_of_individuals_l1198_119809


namespace NUMINAMATH_CALUDE_max_songs_is_56_l1198_119869

/-- Calculates the maximum number of songs that can be played given the specified conditions -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let time_for_short := min (short_songs * short_duration) total_time
  let remaining_time := total_time - time_for_short
  let short_count := time_for_short / short_duration
  let long_count := remaining_time / long_duration
  short_count + long_count

/-- Theorem stating that the maximum number of songs that can be played is 56 -/
theorem max_songs_is_56 : 
  max_songs_played 50 50 3 5 (3 * 60) = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_songs_is_56_l1198_119869


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l1198_119841

theorem triangle_parallelogram_altitude (b h_t h_p : ℝ) : 
  b > 0 →  -- Ensure base is positive
  h_t = 200 →  -- Given altitude of triangle
  (1 / 2) * b * h_t = b * h_p →  -- Equal areas
  h_p = 100 := by
sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_l1198_119841


namespace NUMINAMATH_CALUDE_inner_circle_distance_l1198_119889

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_lengths : a = 9 ∧ b = 12 ∧ c = 15

/-- The path of the center of a circle rolling inside the triangle -/
def inner_circle_path (t : RightTriangle) (r : ℝ) : ℝ := 
  (t.a - 2*r) + (t.b - 2*r) + (t.c - 2*r)

/-- The theorem to be proved -/
theorem inner_circle_distance (t : RightTriangle) : 
  inner_circle_path t 2 = 24 := by sorry

end NUMINAMATH_CALUDE_inner_circle_distance_l1198_119889


namespace NUMINAMATH_CALUDE_toby_speed_proof_l1198_119849

/-- Represents the speed of Toby when pulling the unloaded sled -/
def unloaded_speed : ℝ := 20

/-- Represents the speed of Toby when pulling the loaded sled -/
def loaded_speed : ℝ := 10

/-- Represents the total journey time in hours -/
def total_time : ℝ := 39

/-- Represents the distance of the first part of the journey (loaded sled) -/
def distance1 : ℝ := 180

/-- Represents the distance of the second part of the journey (unloaded sled) -/
def distance2 : ℝ := 120

/-- Represents the distance of the third part of the journey (loaded sled) -/
def distance3 : ℝ := 80

/-- Represents the distance of the fourth part of the journey (unloaded sled) -/
def distance4 : ℝ := 140

theorem toby_speed_proof :
  (distance1 / loaded_speed) + (distance2 / unloaded_speed) +
  (distance3 / loaded_speed) + (distance4 / unloaded_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_toby_speed_proof_l1198_119849


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1198_119839

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of identical cubes needed to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

#eval smallestNumberOfCubes ⟨27, 15, 6⟩

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1198_119839


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1198_119871

/-- A geometric sequence with given conditions -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 6) 
  (h_a5 : a 5 = 162) : 
  ∀ n : ℕ, a n = 2 * 3^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1198_119871


namespace NUMINAMATH_CALUDE_line_graph_most_suitable_l1198_119884

/-- Represents types of graphs --/
inductive GraphType
  | Bar
  | Pie
  | Line

/-- Represents a geographical direction --/
inductive Direction
  | West
  | East

/-- Represents the characteristics of terrain elevation --/
structure TerrainElevation where
  higher : Direction
  lower : Direction

/-- Represents the requirement for visual representation --/
structure VisualRepresentation where
  showChanges : Bool
  alongLatitude : Bool

/-- Determines the most suitable graph type for representing elevation changes --/
def mostSuitableGraphType (terrain : TerrainElevation) (requirement : VisualRepresentation) : GraphType :=
  sorry

/-- Theorem stating that a line graph is the most suitable for the given conditions --/
theorem line_graph_most_suitable 
  (terrain : TerrainElevation)
  (requirement : VisualRepresentation)
  (h1 : terrain.higher = Direction.West)
  (h2 : terrain.lower = Direction.East)
  (h3 : requirement.showChanges = true)
  (h4 : requirement.alongLatitude = true) :
  mostSuitableGraphType terrain requirement = GraphType.Line :=
  sorry

end NUMINAMATH_CALUDE_line_graph_most_suitable_l1198_119884


namespace NUMINAMATH_CALUDE_race_distance_l1198_119890

/-- The race problem -/
theorem race_distance (t_A t_B : ℕ) (lead : ℕ) (h1 : t_A = 36) (h2 : t_B = 45) (h3 : lead = 24) :
  ∃ D : ℕ, D = 24 ∧ (D : ℚ) / t_A * t_B = D + lead :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l1198_119890


namespace NUMINAMATH_CALUDE_decimal_to_binary_38_l1198_119888

theorem decimal_to_binary_38 : 
  (38 : ℕ).digits 2 = [0, 1, 1, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_38_l1198_119888


namespace NUMINAMATH_CALUDE_perpendicular_line_inclination_angle_l1198_119822

/-- The inclination angle of a line perpendicular to x + √3y - 1 = 0 is π/3 -/
theorem perpendicular_line_inclination_angle : 
  let original_line : Real → Real → Prop := λ x y => x + Real.sqrt 3 * y - 1 = 0
  let perpendicular_slope : Real := Real.sqrt 3
  let inclination_angle : Real := Real.pi / 3
  ∀ x y, original_line x y → 
    ∃ m : Real, m * perpendicular_slope = -1 ∧ 
    Real.tan inclination_angle = perpendicular_slope :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_inclination_angle_l1198_119822


namespace NUMINAMATH_CALUDE_fir_tree_count_l1198_119834

/-- Represents the four children in the problem -/
inductive Child
| Anya
| Borya
| Vera
| Gena

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- Returns the gender of a child -/
def childGender (c : Child) : Gender :=
  match c with
  | Child.Anya => Gender.Girl
  | Child.Borya => Gender.Boy
  | Child.Vera => Gender.Girl
  | Child.Gena => Gender.Boy

/-- Represents a statement made by a child -/
def Statement := ℕ → Prop

/-- Returns the statement made by each child -/
def childStatement (c : Child) : Statement :=
  match c with
  | Child.Anya => λ n => n = 15
  | Child.Borya => λ n => n % 11 = 0
  | Child.Vera => λ n => n < 25
  | Child.Gena => λ n => n % 22 = 0

theorem fir_tree_count :
  ∃ (n : ℕ) (truthTellers : Finset Child),
    n = 11 ∧
    truthTellers.card = 2 ∧
    (∃ (boy girl : Child), boy ∈ truthTellers ∧ girl ∈ truthTellers ∧
      childGender boy = Gender.Boy ∧ childGender girl = Gender.Girl) ∧
    (∀ c ∈ truthTellers, childStatement c n) ∧
    (∀ c ∉ truthTellers, ¬(childStatement c n)) :=
  sorry

end NUMINAMATH_CALUDE_fir_tree_count_l1198_119834


namespace NUMINAMATH_CALUDE_two_cars_problem_l1198_119898

/-- Two cars problem -/
theorem two_cars_problem 
  (distance_between_villages : ℝ) 
  (speed_car_A speed_car_B : ℝ) 
  (target_distance : ℝ) :
  distance_between_villages = 18 →
  speed_car_A = 54 →
  speed_car_B = 36 →
  target_distance = 45 →
  -- Case 1: Cars driving towards each other
  (distance_between_villages + target_distance) / (speed_car_A + speed_car_B) = 0.7 ∧
  -- Case 2a: Cars driving in same direction, faster car behind
  (target_distance + distance_between_villages) / (speed_car_A - speed_car_B) = 3.5 ∧
  -- Case 2b: Cars driving in same direction, faster car ahead
  (target_distance - distance_between_villages) / (speed_car_A - speed_car_B) = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_two_cars_problem_l1198_119898


namespace NUMINAMATH_CALUDE_max_servings_is_fifty_l1198_119866

/-- Represents the number of chunks per serving for each fruit type -/
structure FruitRatio :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Represents the available chunks of each fruit type -/
structure AvailableFruit :=
  (cantaloupe : ℕ)
  (honeydew : ℕ)
  (pineapple : ℕ)
  (watermelon : ℕ)

/-- Calculates the maximum number of servings possible given a ratio and available fruit -/
def maxServings (ratio : FruitRatio) (available : AvailableFruit) : ℕ :=
  min
    (available.cantaloupe / ratio.cantaloupe)
    (min
      (available.honeydew / ratio.honeydew)
      (min
        (available.pineapple / ratio.pineapple)
        (available.watermelon / ratio.watermelon)))

theorem max_servings_is_fifty :
  let ratio : FruitRatio := ⟨3, 2, 1, 4⟩
  let available : AvailableFruit := ⟨150, 135, 60, 220⟩
  let minServings : ℕ := 50
  maxServings ratio available = 50 ∧ maxServings ratio available ≥ minServings :=
by sorry

end NUMINAMATH_CALUDE_max_servings_is_fifty_l1198_119866


namespace NUMINAMATH_CALUDE_two_zeros_sum_less_than_neg_two_l1198_119880

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x * Real.exp x
def g (x : ℝ) : ℝ := (x + 1)^2

-- Define the function G
def G (a : ℝ) (x : ℝ) : ℝ := a * f x + g x

-- Theorem statement
theorem two_zeros_sum_less_than_neg_two (a : ℝ) (x₁ x₂ : ℝ) :
  a > 0 →
  G a x₁ = 0 →
  G a x₂ = 0 →
  x₁ ≠ x₂ →
  x₁ + x₂ + 2 < 0 :=
by sorry

end

end NUMINAMATH_CALUDE_two_zeros_sum_less_than_neg_two_l1198_119880


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_four_l1198_119875

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -4 is 4 -/
theorem opposite_of_neg_four : opposite (-4 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_four_l1198_119875


namespace NUMINAMATH_CALUDE_website_earnings_theorem_l1198_119832

/-- Calculates the earnings for a website over a week given the following conditions:
  - The website gets a fixed number of visitors per day for the first 6 days
  - On the 7th day, it gets twice as many visitors as the previous 6 days combined
  - There is a fixed earning per visit -/
def websiteEarnings (dailyVisitors : ℕ) (earningsPerVisit : ℚ) : ℚ :=
  let firstSixDaysVisits : ℕ := 6 * dailyVisitors
  let seventhDayVisits : ℕ := 2 * firstSixDaysVisits
  let totalVisits : ℕ := firstSixDaysVisits + seventhDayVisits
  (totalVisits : ℚ) * earningsPerVisit

/-- Theorem stating that under the given conditions, the website earnings for the week are $18 -/
theorem website_earnings_theorem :
  websiteEarnings 100 (1 / 100) = 18 := by
  sorry


end NUMINAMATH_CALUDE_website_earnings_theorem_l1198_119832


namespace NUMINAMATH_CALUDE_y_percent_of_x_l1198_119812

theorem y_percent_of_x (x y : ℝ) (h : 0.6 * (x - y) = 0.3 * (x + y)) : y / x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_y_percent_of_x_l1198_119812


namespace NUMINAMATH_CALUDE_circumference_difference_concentric_circles_l1198_119876

/-- Given two concentric circles where the outer circle's radius is 12 feet greater than the inner circle's radius, the difference in their circumferences is 24π feet. -/
theorem circumference_difference_concentric_circles (r : ℝ) : 
  2 * π * (r + 12) - 2 * π * r = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_concentric_circles_l1198_119876


namespace NUMINAMATH_CALUDE_cylinder_height_l1198_119827

/-- The height of a cylinder given its base perimeter and side surface diagonal --/
theorem cylinder_height (base_perimeter : ℝ) (diagonal : ℝ) (h : base_perimeter = 6 ∧ diagonal = 10) :
  ∃ (height : ℝ), height = 8 ∧ height ^ 2 + base_perimeter ^ 2 = diagonal ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l1198_119827


namespace NUMINAMATH_CALUDE_miju_handshakes_l1198_119882

/-- Calculate the total number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def totalHandshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The problem statement -/
theorem miju_handshakes :
  totalHandshakes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_miju_handshakes_l1198_119882


namespace NUMINAMATH_CALUDE_more_students_than_pets_l1198_119853

theorem more_students_than_pets : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 20
  let rabbits_per_classroom : ℕ := 2
  let goldfish_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_pets : ℕ := num_classrooms * (rabbits_per_classroom + goldfish_per_classroom)
  total_students - total_pets = 75 := by
sorry

end NUMINAMATH_CALUDE_more_students_than_pets_l1198_119853


namespace NUMINAMATH_CALUDE_alien_minerals_count_l1198_119828

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- The number of minerals collected by the alien --/
def alienMinerals : ℕ := base7ToBase10 3 2 1

theorem alien_minerals_count :
  alienMinerals = 162 := by sorry

end NUMINAMATH_CALUDE_alien_minerals_count_l1198_119828


namespace NUMINAMATH_CALUDE_quadratic_solution_l1198_119861

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 45 = 0) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1198_119861


namespace NUMINAMATH_CALUDE_equation_solution_l1198_119865

theorem equation_solution : 
  ∃ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1198_119865


namespace NUMINAMATH_CALUDE_snail_reaches_top_in_ten_days_l1198_119860

/-- Represents the snail's climbing problem -/
structure SnailClimb where
  treeHeight : ℕ
  climbUp : ℕ
  slideDown : ℕ

/-- Calculates the number of days needed for the snail to reach the top of the tree -/
def daysToReachTop (s : SnailClimb) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the snail reaches the top in 10 days -/
theorem snail_reaches_top_in_ten_days :
  let s : SnailClimb := ⟨24, 6, 4⟩
  daysToReachTop s = 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_reaches_top_in_ten_days_l1198_119860


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_eq_17965_l1198_119856

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ :=
  d * ((10^n - 1) / 9)

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

theorem sum_of_digits_9ab_eq_17965 :
  let a := repeatedDigit 4 1995
  let b := repeatedDigit 7 1995
  sumOfDigits (9 * a * b) = 17965 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_eq_17965_l1198_119856


namespace NUMINAMATH_CALUDE_valid_triples_l1198_119867

def is_valid_triple (p x y : ℕ) : Prop :=
  Nat.Prime p ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  ∃ a : ℕ, x^(p-1) + y = p^a ∧ 
  ∃ b : ℕ, x + y^(p-1) = p^b

def is_valid_triple_for_two (n i : ℕ) : Prop :=
  n > 0 ∧ n < 2^i

theorem valid_triples :
  ∀ p x y : ℕ, is_valid_triple p x y →
    ((p = 3 ∧ ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2))) ∨
     (p = 2 ∧ ∃ n i : ℕ, is_valid_triple_for_two n i ∧ x = n ∧ y = 2^i - n)) :=
sorry

end NUMINAMATH_CALUDE_valid_triples_l1198_119867


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1198_119816

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 40 →
  area = (perimeter / 4) ^ 2 →
  area = 100 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1198_119816


namespace NUMINAMATH_CALUDE_carl_pink_hats_solution_l1198_119870

/-- The number of pink hard hats Carl took away from the truck -/
def carl_pink_hats : ℕ := sorry

theorem carl_pink_hats_solution : carl_pink_hats = 4 := by
  have initial_pink : ℕ := 26
  have initial_green : ℕ := 15
  have initial_yellow : ℕ := 24
  have john_pink : ℕ := 6
  have john_green : ℕ := 2 * john_pink
  have remaining_hats : ℕ := 43

  sorry

end NUMINAMATH_CALUDE_carl_pink_hats_solution_l1198_119870


namespace NUMINAMATH_CALUDE_double_acute_angle_range_l1198_119814

/-- If θ is an acute angle, then 2θ is a positive angle less than 180°. -/
theorem double_acute_angle_range (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by sorry

end NUMINAMATH_CALUDE_double_acute_angle_range_l1198_119814


namespace NUMINAMATH_CALUDE_max_square_side_length_56_24_l1198_119830

/-- The maximum side length of squares that can be cut from a rectangular paper -/
def max_square_side_length (length width : ℕ) : ℕ := Nat.gcd length width

theorem max_square_side_length_56_24 :
  max_square_side_length 56 24 = 8 := by sorry

end NUMINAMATH_CALUDE_max_square_side_length_56_24_l1198_119830


namespace NUMINAMATH_CALUDE_math_team_selection_l1198_119883

theorem math_team_selection (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 →
  k = 3 →
  total = 10 →
  (Nat.choose (total - 1) k) - (Nat.choose (total - 3) k) = 49 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l1198_119883


namespace NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_l1198_119808

theorem remainder_thirteen_power_thirteen_plus_thirteen (n : ℕ) :
  (13^13 + 13) % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_power_thirteen_plus_thirteen_l1198_119808


namespace NUMINAMATH_CALUDE_factory_production_l1198_119815

/-- Represents a machine in the factory -/
structure Machine where
  rate : Nat  -- shirts produced per minute
  time_yesterday : Nat  -- minutes worked yesterday
  time_today : Nat  -- minutes worked today

/-- Calculates the total number of shirts produced by all machines -/
def total_shirts (machines : List Machine) : Nat :=
  machines.foldl (fun acc m => acc + m.rate * (m.time_yesterday + m.time_today)) 0

/-- Theorem: Given the specified machines, the total number of shirts produced is 432 -/
theorem factory_production : 
  let machines : List Machine := [
    { rate := 6, time_yesterday := 12, time_today := 10 },  -- Machine A
    { rate := 8, time_yesterday := 10, time_today := 15 },  -- Machine B
    { rate := 5, time_yesterday := 20, time_today := 0 }    -- Machine C
  ]
  total_shirts machines = 432 := by
  sorry


end NUMINAMATH_CALUDE_factory_production_l1198_119815


namespace NUMINAMATH_CALUDE_scientific_notation_508_billion_yuan_l1198_119819

theorem scientific_notation_508_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧
    508 * (10 ^ 9) = a * (10 ^ n) ∧
    a = 5.08 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_508_billion_yuan_l1198_119819


namespace NUMINAMATH_CALUDE_percentage_calculation_l1198_119818

theorem percentage_calculation : 
  (0.2 * (0.75 * 800)) / 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1198_119818


namespace NUMINAMATH_CALUDE_factorization_example_l1198_119804

/-- Represents a factorization from left to right -/
def is_factorization (f : ℝ → ℝ → ℝ) (g : ℝ → ℝ → ℝ) (h : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, f a b = g a b * h a b

theorem factorization_example :
  is_factorization (λ a b => a^2*b + a*b^3) (λ a b => a*b) (λ a b => a + b^2) ∧
  ¬is_factorization (λ x _ => x^2 - 1) (λ x _ => x) (λ x _ => x - 1) ∧
  ¬is_factorization (λ x y => x^2 + 2*y + 1) (λ x y => x) (λ x y => x + 2*y) ∧
  ¬is_factorization (λ x y => x*(x+y)) (λ x _ => x^2) (λ _ y => y) :=
by sorry

end NUMINAMATH_CALUDE_factorization_example_l1198_119804


namespace NUMINAMATH_CALUDE_problem_solution_l1198_119817

-- Define the variables
def x : ℝ := 12 * (1 + 0.2)
def y : ℝ := 0.75 * x^2
def z : ℝ := 3 * y + 16
def w : ℝ := 2 * z - y
def v : ℝ := z^3 - 0.5 * y

-- State the theorem
theorem problem_solution :
  v = 112394885.1456 ∧ w = 809.6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1198_119817


namespace NUMINAMATH_CALUDE_intersection_points_count_l1198_119836

/-- A line in a 2D plane, represented by coefficients a, b, and c in the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Determine if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

/-- The three lines given in the problem -/
def line1 : Line := ⟨2, -3, 4⟩
def line2 : Line := ⟨3, 4, 6⟩
def line3 : Line := ⟨6, -9, 8⟩

/-- The theorem to be proved -/
theorem intersection_points_count :
  (intersect line1 line2 ∧ intersect line2 line3 ∧ parallel line1 line3) ∧
  (∃! p : ℝ × ℝ, p.1 * line1.a + p.2 * line1.b = line1.c ∧ p.1 * line2.a + p.2 * line2.b = line2.c) ∧
  (∃! p : ℝ × ℝ, p.1 * line2.a + p.2 * line2.b = line2.c ∧ p.1 * line3.a + p.2 * line3.b = line3.c) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1198_119836


namespace NUMINAMATH_CALUDE_max_value_f_l1198_119891

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem max_value_f (a : ℝ) (h : a ≤ 0) :
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a x ≥ f a y) ∧
  (∃ (max_val : ℝ), 
    (a = 0 → max_val = 1) ∧
    (-2 < a ∧ a < 0 → max_val = Real.exp a) ∧
    (a ≤ -2 → max_val = 4 / (a^2 * Real.exp 2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l1198_119891


namespace NUMINAMATH_CALUDE_strawberry_weight_sum_l1198_119887

theorem strawberry_weight_sum (marco_weight dad_weight : ℕ) 
  (h1 : marco_weight = 15) 
  (h2 : dad_weight = 22) : 
  marco_weight + dad_weight = 37 := by
sorry

end NUMINAMATH_CALUDE_strawberry_weight_sum_l1198_119887


namespace NUMINAMATH_CALUDE_sin_negative_150_degrees_l1198_119802

theorem sin_negative_150_degrees :
  Real.sin (-(150 * π / 180)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_150_degrees_l1198_119802


namespace NUMINAMATH_CALUDE_baseball_card_purchase_l1198_119852

/-- The cost of the rare baseball card -/
def card_cost : ℕ := 100

/-- Patricia's money -/
def patricia_money : ℕ := 6

/-- Lisa's money in terms of Patricia's -/
def lisa_money : ℕ := 5 * patricia_money

/-- Charlotte's money in terms of Lisa's -/
def charlotte_money : ℕ := lisa_money / 2

/-- The total money they have -/
def total_money : ℕ := patricia_money + lisa_money + charlotte_money

/-- The additional amount needed to buy the card -/
def additional_money_needed : ℕ := card_cost - total_money

theorem baseball_card_purchase :
  additional_money_needed = 49 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_purchase_l1198_119852


namespace NUMINAMATH_CALUDE_frog_climb_days_l1198_119803

/-- The number of days required for a frog to climb out of a well -/
def days_to_climb (well_depth : ℕ) (climb_distance : ℕ) (slide_distance : ℕ) : ℕ :=
  (well_depth + climb_distance - slide_distance - 1) / (climb_distance - slide_distance) + 1

/-- Theorem: A frog in a 50-meter well, climbing 5 meters up and sliding 2 meters down daily, 
    takes at least 16 days to reach the top -/
theorem frog_climb_days :
  days_to_climb 50 5 2 ≥ 16 := by
  sorry

#eval days_to_climb 50 5 2

end NUMINAMATH_CALUDE_frog_climb_days_l1198_119803


namespace NUMINAMATH_CALUDE_max_students_above_median_l1198_119872

theorem max_students_above_median (n : ℕ) (h : n = 81) :
  (n + 1) / 2 = (n + 1) / 2 ∧ (n - (n + 1) / 2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_students_above_median_l1198_119872


namespace NUMINAMATH_CALUDE_average_rst_l1198_119873

theorem average_rst (r s t : ℝ) (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_rst_l1198_119873


namespace NUMINAMATH_CALUDE_two_color_theorem_l1198_119826

/-- A type representing a plane divided by lines -/
structure DividedPlane where
  n : ℕ  -- number of lines
  regions : Set (Set ℝ × ℝ)  -- regions as sets of points
  adjacent : regions → regions → Prop  -- adjacency relation

/-- A coloring of the plane -/
def Coloring (p : DividedPlane) := p.regions → Bool

/-- A valid two-coloring of the plane -/
def ValidColoring (p : DividedPlane) (c : Coloring p) : Prop :=
  ∀ r1 r2 : p.regions, p.adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem: any divided plane has a valid two-coloring -/
theorem two_color_theorem (p : DividedPlane) : ∃ c : Coloring p, ValidColoring p c := by
  sorry

end NUMINAMATH_CALUDE_two_color_theorem_l1198_119826


namespace NUMINAMATH_CALUDE_floor_painting_possibilities_l1198_119847

theorem floor_painting_possibilities :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.2 > p.1 ∧ 
       (p.1 - 4) * (p.2 - 4) = 2 * p.1 * p.2 / 3 ∧
       p.1 > 0 ∧ p.2 > 0)) ∧
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_possibilities_l1198_119847


namespace NUMINAMATH_CALUDE_johns_house_nails_l1198_119840

/-- Calculates the total number of nails needed for a house wall -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Proves that John needs 229 nails for the house wall -/
theorem johns_house_nails :
  total_nails 13 17 8 = 229 := by
  sorry

end NUMINAMATH_CALUDE_johns_house_nails_l1198_119840


namespace NUMINAMATH_CALUDE_dan_minimum_speed_l1198_119851

/-- Proves the minimum speed Dan must exceed to arrive before Cara -/
theorem dan_minimum_speed (distance : ℝ) (cara_speed : ℝ) (dan_delay : ℝ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (min_speed : ℝ), min_speed > 36 ∧
    ∀ (dan_speed : ℝ), dan_speed > min_speed →
      distance / dan_speed < distance / cara_speed - dan_delay := by
  sorry

#check dan_minimum_speed

end NUMINAMATH_CALUDE_dan_minimum_speed_l1198_119851


namespace NUMINAMATH_CALUDE_evaluate_expression_l1198_119858

theorem evaluate_expression (b : ℕ) (h : b = 4) : b^3 * b^4 * b^2 = 262144 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1198_119858


namespace NUMINAMATH_CALUDE_factor_z6_minus_64_l1198_119801

theorem factor_z6_minus_64 (z : ℂ) : 
  z^6 - 64 = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := by
  sorry

#check factor_z6_minus_64

end NUMINAMATH_CALUDE_factor_z6_minus_64_l1198_119801


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1198_119833

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first 30 rows of Pascal's Triangle -/
def total_elements : ℕ := sum_first_n 30

theorem pascal_triangle_30_rows :
  total_elements = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l1198_119833


namespace NUMINAMATH_CALUDE_bipartite_ramsey_theorem_l1198_119800

/-- A bipartite graph -/
structure BipartiteGraph where
  X : Type
  Y : Type
  E : X → Y → Prop

/-- An edge coloring of a bipartite graph -/
def EdgeColoring (G : BipartiteGraph) := G.X → G.Y → Bool

/-- A homomorphism between bipartite graphs -/
structure BipartiteHomomorphism (G H : BipartiteGraph) where
  φX : G.X → H.X
  φY : G.Y → H.Y
  preserves_edges : ∀ x y, G.E x y → H.E (φX x) (φY y)

/-- The main theorem -/
theorem bipartite_ramsey_theorem :
  ∀ P : BipartiteGraph, ∃ P' : BipartiteGraph,
    ∀ c : EdgeColoring P',
      ∃ φ : BipartiteHomomorphism P P',
        ∃ color : Bool,
          ∀ x y, P.E x y → c (φ.φX x) (φ.φY y) = color :=
sorry

end NUMINAMATH_CALUDE_bipartite_ramsey_theorem_l1198_119800


namespace NUMINAMATH_CALUDE_both_glasses_and_hair_tied_l1198_119863

def students : Finset ℕ := Finset.range 30

def glasses : Finset ℕ := {1, 3, 7, 10, 23, 27}

def hairTied : Finset ℕ := {1, 9, 11, 20, 23}

theorem both_glasses_and_hair_tied :
  (glasses ∩ hairTied).card = 2 := by sorry

end NUMINAMATH_CALUDE_both_glasses_and_hair_tied_l1198_119863


namespace NUMINAMATH_CALUDE_total_rainfall_2005_l1198_119837

def rainfall_2005 (initial_rainfall : ℝ) (yearly_increase : ℝ) : ℝ :=
  12 * (initial_rainfall + 2 * yearly_increase)

theorem total_rainfall_2005 (initial_rainfall yearly_increase : ℝ) 
  (h1 : initial_rainfall = 30)
  (h2 : yearly_increase = 3) :
  rainfall_2005 initial_rainfall yearly_increase = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2005_l1198_119837


namespace NUMINAMATH_CALUDE_product_remainder_mod_seven_l1198_119806

theorem product_remainder_mod_seven : ((-1234 * 1984 * -1460 * 2008) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_seven_l1198_119806


namespace NUMINAMATH_CALUDE_sequence_properties_l1198_119811

/-- Definition of an arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Definition of a geometric sequence -/
def geometric_seq (g : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, g (n + 1) = g n * q

/-- Main theorem -/
theorem sequence_properties
  (a g : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hg : geometric_seq g)
  (h1 : a 1 = 1)
  (h2 : g 1 = 1)
  (h3 : a 2 = g 2)
  (h4 : a 2 ≠ 1)
  (h5 : ∃ m : ℕ, m > 3 ∧ a m = g 3) :
  (∃ m : ℕ, m > 3 ∧
    (∃ d q : ℝ, d = m - 3 ∧ q = m - 2 ∧
      (∀ n : ℕ, a (n + 1) = a n + d) ∧
      (∀ n : ℕ, g (n + 1) = g n * q))) ∧
  (∃ k : ℕ, a k = g 4) ∧
  (∀ j : ℕ, ∃ k : ℕ, g (j + 1) = a k) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1198_119811
