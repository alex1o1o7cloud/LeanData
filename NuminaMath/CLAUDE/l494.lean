import Mathlib

namespace NUMINAMATH_CALUDE_clarinet_cost_is_125_l494_49405

def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def total_books_sold : ℕ := 25

def clarinet_cost : ℕ := total_books_sold * price_per_book

theorem clarinet_cost_is_125 : clarinet_cost = 125 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_cost_is_125_l494_49405


namespace NUMINAMATH_CALUDE_angle_of_inclination_range_l494_49410

theorem angle_of_inclination_range (θ : ℝ) :
  let α := Real.arctan (1 / (Real.sin θ))
  (∃ x y, x - y * Real.sin θ + 1 = 0) →
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_angle_of_inclination_range_l494_49410


namespace NUMINAMATH_CALUDE_hyperbola_equation_l494_49451

/-- Given a hyperbola with one focus at (5,0) and asymptotes y = ± 4/3 x, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (F : ℝ × ℝ) (slope : ℝ) :
  F = (5, 0) →
  slope = 4/3 →
  ∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1) ↔ 
    (∃ (a b c : ℝ), 
      a^2 + b^2 = c^2 ∧
      c = 5 ∧
      b / a = slope ∧
      x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l494_49451


namespace NUMINAMATH_CALUDE_article_cost_price_l494_49438

theorem article_cost_price (C S : ℝ) : 
  (S = 1.05 * C) →                    -- Condition 1
  (S - 5 = 1.1 * (0.95 * C)) →        -- Condition 2
  C = 1000 :=                         -- Conclusion
by sorry

end NUMINAMATH_CALUDE_article_cost_price_l494_49438


namespace NUMINAMATH_CALUDE_average_speed_approx_202_l494_49411

/-- Calculates the average speed given initial and final odometer readings and total driving time -/
def average_speed (initial_reading final_reading : ℕ) (total_time : ℚ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

theorem average_speed_approx_202 (initial_reading final_reading : ℕ) (total_time : ℚ) :
  initial_reading = 12321 →
  final_reading = 14741 →
  total_time = 12 →
  ∃ ε > 0, |average_speed initial_reading final_reading total_time - 202| < ε :=
by
  sorry

#eval average_speed 12321 14741 12

end NUMINAMATH_CALUDE_average_speed_approx_202_l494_49411


namespace NUMINAMATH_CALUDE_min_value_of_expression_l494_49478

theorem min_value_of_expression (x y : ℝ) : 
  (x*y - 2)^2 + (x - 1 + y)^2 ≥ 2 ∧ 
  ∃ (a b : ℝ), (a*b - 2)^2 + (a - 1 + b)^2 = 2 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_expression_l494_49478


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l494_49458

-- Define the curve E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line passing through two points
def Line (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define a line with a given slope passing through a point
def LineWithSlope (x0 y0 m : ℝ) (x y : ℝ) : Prop :=
  y - y0 = m * (x - x0)

theorem fixed_point_theorem (xA yA xB yB xC yC : ℝ) :
  E xA yA →
  E xB yB →
  E xC yC →
  Line (-3) 2 xA yA xB yB →
  LineWithSlope xA yA 1 xC yC →
  Line xB yB xC yC 5 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l494_49458


namespace NUMINAMATH_CALUDE_f_composition_value_l494_49418

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 0
  else if x = 0 then Real.pi
  else Real.pi^2 + 1

theorem f_composition_value : f (f (f 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l494_49418


namespace NUMINAMATH_CALUDE_f_root_condition_and_inequality_l494_49413

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

theorem f_root_condition_and_inequality (a : ℝ) (b : ℝ) :
  (a > 0 ∧ (∃ x > 0, f a x = 0) ↔ 0 < a ∧ a ≤ 1 / Real.exp 1) ∧
  (a ≥ 2 / Real.exp 1 ∧ b > 1 → f a (Real.log b) > 1 / b) := by
  sorry

end NUMINAMATH_CALUDE_f_root_condition_and_inequality_l494_49413


namespace NUMINAMATH_CALUDE_nabla_calculation_l494_49449

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := a + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 3 2) 2 = 2059 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l494_49449


namespace NUMINAMATH_CALUDE_german_enrollment_l494_49464

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 32)
  (h2 : both_subjects = 12)
  (h3 : only_english = 10)
  (h4 : total_students = both_subjects + only_english + (total_students - (both_subjects + only_english))) :
  total_students - (both_subjects + only_english) + both_subjects = 22 := by
  sorry

#check german_enrollment

end NUMINAMATH_CALUDE_german_enrollment_l494_49464


namespace NUMINAMATH_CALUDE_sum_greater_than_four_necessity_not_sufficiency_l494_49425

theorem sum_greater_than_four_necessity_not_sufficiency (a b : ℝ) :
  (((a > 2) ∧ (b > 2)) → (a + b > 4)) ∧
  (∃ a b : ℝ, (a + b > 4) ∧ ¬((a > 2) ∧ (b > 2))) :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_necessity_not_sufficiency_l494_49425


namespace NUMINAMATH_CALUDE_find_other_number_l494_49436

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 14) (h3 : Nat.lcm A B = 312) :
  B = 182 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l494_49436


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l494_49473

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l494_49473


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l494_49406

theorem subset_implies_m_equals_one (m : ℝ) : 
  let A : Set ℝ := {-1, 2, 2*m-1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l494_49406


namespace NUMINAMATH_CALUDE_average_book_width_l494_49441

theorem average_book_width :
  let book_widths : List ℝ := [3, 0.5, 1.5, 4, 2, 5, 8]
  let sum_widths : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := sum_widths / num_books
  average_width = 3.43 := by sorry

end NUMINAMATH_CALUDE_average_book_width_l494_49441


namespace NUMINAMATH_CALUDE_min_value_of_f_l494_49421

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2 - 5

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l494_49421


namespace NUMINAMATH_CALUDE_coefficient_x3_is_correct_l494_49490

/-- The coefficient of x^3 in the expansion of (x^2 - 2x)(1 + x)^6 -/
def coefficient_x3 : ℤ := -24

/-- The expansion of (x^2 - 2x)(1 + x)^6 -/
def expansion (x : ℚ) : ℚ := (x^2 - 2*x) * (1 + x)^6

theorem coefficient_x3_is_correct :
  (∃ f : ℚ → ℚ, ∀ x, expansion x = f x + coefficient_x3 * x^3 + x^4 * f x) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_correct_l494_49490


namespace NUMINAMATH_CALUDE_max_product_with_851_l494_49498

def digits : Finset Nat := {1, 5, 6, 8, 9}

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit_number (d e : Nat) : Nat := 10 * d + e

theorem max_product_with_851 :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    three_digit_number a b c * two_digit_number d e ≤ three_digit_number 8 5 1 * two_digit_number 9 6 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_851_l494_49498


namespace NUMINAMATH_CALUDE_pauline_garden_capacity_l494_49408

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - (g.tomatoes + g.cucumbers + g.potatoes)

/-- Theorem stating the number of additional vegetables Pauline can plant -/
theorem pauline_garden_capacity :
  ∀ (g : Garden),
    g.rows = 10 ∧
    g.spaces_per_row = 15 ∧
    g.tomatoes = 15 ∧
    g.cucumbers = 20 ∧
    g.potatoes = 30 →
    additional_vegetables g = 85 := by
  sorry


end NUMINAMATH_CALUDE_pauline_garden_capacity_l494_49408


namespace NUMINAMATH_CALUDE_calculation_proof_l494_49456

theorem calculation_proof :
  (1 / 6 + 2 / 3) * (-24) = -20 ∧
  (-3)^2 * (2 - (-6)) + 30 / (-5) = 66 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l494_49456


namespace NUMINAMATH_CALUDE_square_sum_product_equality_l494_49486

theorem square_sum_product_equality : 
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equality_l494_49486


namespace NUMINAMATH_CALUDE_four_distinct_roots_condition_l494_49412

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - 4| = a * x + 6

-- Define the condition for four distinct roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation x₁ a ∧ equation x₂ a ∧ equation x₃ a ∧ equation x₄ a

-- Theorem statement
theorem four_distinct_roots_condition (a : ℝ) :
  has_four_distinct_roots a ↔ (-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_condition_l494_49412


namespace NUMINAMATH_CALUDE_circular_well_volume_l494_49482

/-- The volume of a circular cylinder with diameter 2 metres and height 14 metres is 14π cubic metres. -/
theorem circular_well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 14
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = 14 * π := by
  sorry

end NUMINAMATH_CALUDE_circular_well_volume_l494_49482


namespace NUMINAMATH_CALUDE_expression_values_l494_49497

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l494_49497


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l494_49488

/-- Given a point P with coordinates (3, -4), this theorem proves that its symmetric point
    with respect to the origin has coordinates (-3, 4). -/
theorem symmetric_point_wrt_origin :
  let P : ℝ × ℝ := (3, -4)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (-3, 4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l494_49488


namespace NUMINAMATH_CALUDE_laundry_day_lcm_l494_49474

theorem laundry_day_lcm : Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 12 15)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_laundry_day_lcm_l494_49474


namespace NUMINAMATH_CALUDE_john_house_nails_l494_49420

/-- The number of nails needed for John's house walls -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Theorem: John needs 987 nails for his house walls -/
theorem john_house_nails :
  total_nails 27 36 15 = 987 := by
  sorry

end NUMINAMATH_CALUDE_john_house_nails_l494_49420


namespace NUMINAMATH_CALUDE_baseball_team_wins_l494_49417

theorem baseball_team_wins (total_games : ℕ) (wins losses : ℕ) : 
  total_games = 130 →
  wins + losses = total_games →
  wins = 3 * losses + 14 →
  wins = 101 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l494_49417


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_is_five_sixths_l494_49462

/-- The number of white balls in the pouch -/
def num_white_balls : ℕ := 2

/-- The number of black balls in the pouch -/
def num_black_balls : ℕ := 2

/-- The total number of balls in the pouch -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- The number of balls drawn from the pouch -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 5/6

theorem prob_at_least_one_black_is_five_sixths :
  prob_at_least_one_black = 1 - (num_white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_is_five_sixths_l494_49462


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l494_49491

theorem incorrect_average_calculation (n : Nat) (incorrect_num correct_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 75 ∧ 
  correct_avg = 51 →
  ∃ (S : ℚ), 
    (S + correct_num) / n = correct_avg ∧
    (S + incorrect_num) / n = 46 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l494_49491


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l494_49440

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 1993 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_23_l494_49440


namespace NUMINAMATH_CALUDE_james_balloon_count_l494_49452

/-- The number of balloons Amy has -/
def amy_balloons : ℕ := 101

/-- The number of additional balloons James has compared to Amy -/
def james_additional_balloons : ℕ := 131

/-- The total number of balloons James has -/
def james_balloons : ℕ := amy_balloons + james_additional_balloons

theorem james_balloon_count : james_balloons = 232 := by
  sorry

end NUMINAMATH_CALUDE_james_balloon_count_l494_49452


namespace NUMINAMATH_CALUDE_gvidon_descendants_l494_49444

/-- Represents the genealogy of King Gvidon's descendants -/
structure GvidonGenealogy where
  sons : Nat
  descendants_with_sons : Nat
  sons_per_descendant : Nat

/-- Calculates the total number of descendants in Gvidon's genealogy -/
def total_descendants (g : GvidonGenealogy) : Nat :=
  g.sons + g.descendants_with_sons * g.sons_per_descendant

/-- Theorem stating that King Gvidon's total descendants is 305 -/
theorem gvidon_descendants (g : GvidonGenealogy)
  (h1 : g.sons = 5)
  (h2 : g.descendants_with_sons = 100)
  (h3 : g.sons_per_descendant = 3) :
  total_descendants g = 305 := by
  sorry

#check gvidon_descendants

end NUMINAMATH_CALUDE_gvidon_descendants_l494_49444


namespace NUMINAMATH_CALUDE_centroid_property_l494_49460

/-- The centroid of a triangle divides each median in the ratio 2:1 and creates three equal-area subtriangles. -/
structure Centroid (xA yA xB yB xC yC : ℚ) where
  x : ℚ
  y : ℚ
  is_centroid : x = (xA + xB + xC) / 3 ∧ y = (yA + yB + yC) / 3

/-- Given a triangle ABC with vertices A(5,8), B(3,-2), and C(6,1),
    if D(m,n) is the centroid of the triangle, then 10m + n = 49. -/
theorem centroid_property :
  let d : Centroid 5 8 3 (-2) 6 1 := ⟨(14/3), (7/3), by sorry⟩
  10 * d.x + d.y = 49 := by sorry

end NUMINAMATH_CALUDE_centroid_property_l494_49460


namespace NUMINAMATH_CALUDE_prime_triple_uniqueness_l494_49484

theorem prime_triple_uniqueness : 
  ∀ p : ℕ, p > 0 → Prime p → Prime (p + 2) → Prime (p + 4) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_uniqueness_l494_49484


namespace NUMINAMATH_CALUDE_line_intercepts_equal_l494_49499

theorem line_intercepts_equal (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0) →
  (∃ k : ℝ, k ≠ 0 ∧ k = a - 2 ∧ k = (a - 2) / (a + 1)) →
  (a = 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_equal_l494_49499


namespace NUMINAMATH_CALUDE_federal_guideline_requirement_l494_49455

/-- The daily minimum requirement of vegetables in cups according to federal guidelines. -/
def daily_requirement : ℕ := 3

/-- The number of days Sarah has been eating vegetables. -/
def days_counted : ℕ := 5

/-- The total amount of vegetables Sarah has eaten in cups. -/
def vegetables_eaten : ℕ := 8

/-- Sarah's daily consumption needed to meet the minimum requirement. -/
def sarah_daily_need : ℕ := 3

theorem federal_guideline_requirement :
  daily_requirement = sarah_daily_need :=
by sorry

end NUMINAMATH_CALUDE_federal_guideline_requirement_l494_49455


namespace NUMINAMATH_CALUDE_sum_of_page_numbers_constant_l494_49415

/-- Represents a magazine with nested double sheets. -/
structure Magazine where
  num_double_sheets : ℕ
  pages_per_double_sheet : ℕ

/-- Calculates the sum of page numbers on a double sheet. -/
def sum_of_page_numbers (m : Magazine) (sheet_number : ℕ) : ℕ :=
  sorry

/-- Theorem: The sum of page numbers on each double sheet is always 130. -/
theorem sum_of_page_numbers_constant (m : Magazine) (sheet_number : ℕ) :
  m.num_double_sheets = 16 →
  m.pages_per_double_sheet = 4 →
  sheet_number ≤ m.num_double_sheets →
  sum_of_page_numbers m sheet_number = 130 :=
sorry

end NUMINAMATH_CALUDE_sum_of_page_numbers_constant_l494_49415


namespace NUMINAMATH_CALUDE_max_xy_value_l494_49483

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3*x + 8*y = 48) :
  x*y ≤ 18 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3*x₀ + 8*y₀ = 48 ∧ x₀*y₀ = 18 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l494_49483


namespace NUMINAMATH_CALUDE_num_valid_teams_eq_930_l494_49467

/-- Represents a debater in the team -/
inductive Debater
| Boy : Fin 4 → Debater
| Girl : Fin 4 → Debater

/-- Represents a debate team -/
def DebateTeam := Fin 4 → Debater

/-- Check if Boy A is in the team -/
def has_boy_A (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Boy 0

/-- Check if Girl B is in the team -/
def has_girl_B (team : DebateTeam) : Prop :=
  ∃ i, team i = Debater.Girl 1

/-- Check if Boy A is not the first debater -/
def boy_A_not_first (team : DebateTeam) : Prop :=
  team 0 ≠ Debater.Boy 0

/-- Check if Girl B is not the fourth debater -/
def girl_B_not_fourth (team : DebateTeam) : Prop :=
  team 3 ≠ Debater.Girl 1

/-- Check if the team satisfies all constraints -/
def valid_team (team : DebateTeam) : Prop :=
  boy_A_not_first team ∧
  girl_B_not_fourth team ∧
  (has_boy_A team → has_girl_B team)

/-- The number of valid debate teams -/
def num_valid_teams : ℕ := sorry

theorem num_valid_teams_eq_930 : num_valid_teams = 930 := by sorry

end NUMINAMATH_CALUDE_num_valid_teams_eq_930_l494_49467


namespace NUMINAMATH_CALUDE_f_properties_l494_49475

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2 ∧ ∀ y, 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x) ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 6 ∧ f θ = 4 / 3 → Real.cos (2 * θ) = (Real.sqrt 15 + 2) / 6) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l494_49475


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l494_49409

theorem negation_of_all_squares_positive :
  (¬ ∀ n : ℕ, n^2 > 0) ↔ (∃ n : ℕ, ¬(n^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l494_49409


namespace NUMINAMATH_CALUDE_function_composition_l494_49434

theorem function_composition (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 - 2*x + 5) :
  ∀ x : ℝ, f (x - 1) = x^2 - 4*x + 8 := by
sorry

end NUMINAMATH_CALUDE_function_composition_l494_49434


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l494_49479

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 28 →
  d = 3 →
  n * d = aₙ - a₁ + d →
  (n * (a₁ + aₙ)) / 2 = 145 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l494_49479


namespace NUMINAMATH_CALUDE_function_inequality_l494_49496

open Real

theorem function_inequality (f : ℝ → ℝ) (h : ∀ x > 0, Real.sqrt x * (deriv f x) < (1 / 2)) :
  f 9 - 1 < f 4 ∧ f 4 < f 1 + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l494_49496


namespace NUMINAMATH_CALUDE_expression_value_l494_49459

theorem expression_value : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 := by sorry

end NUMINAMATH_CALUDE_expression_value_l494_49459


namespace NUMINAMATH_CALUDE_inequality_proof_l494_49443

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : d > 0) :
  d / c < (d + 4) / (c + 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l494_49443


namespace NUMINAMATH_CALUDE_banana_arrangements_l494_49404

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l494_49404


namespace NUMINAMATH_CALUDE_burrito_cost_burrito_cost_is_six_l494_49435

/-- Calculates the cost of burritos given the following conditions:
  * There are 10 burritos with 120 calories each
  * 5 burgers with 400 calories each cost $8
  * Burgers provide 50 more calories per dollar than burritos
-/
theorem burrito_cost : ℝ → Prop :=
  fun cost : ℝ =>
    let burrito_count : ℕ := 10
    let burrito_calories : ℕ := 120
    let burger_count : ℕ := 5
    let burger_calories : ℕ := 400
    let burger_cost : ℝ := 8
    let calorie_difference : ℝ := 50

    let total_burrito_calories : ℕ := burrito_count * burrito_calories
    let total_burger_calories : ℕ := burger_count * burger_calories
    let burger_calories_per_dollar : ℝ := total_burger_calories / burger_cost
    let burrito_calories_per_dollar : ℝ := burger_calories_per_dollar - calorie_difference

    cost = total_burrito_calories / burrito_calories_per_dollar ∧
    cost = 6

theorem burrito_cost_is_six : burrito_cost 6 := by
  sorry

end NUMINAMATH_CALUDE_burrito_cost_burrito_cost_is_six_l494_49435


namespace NUMINAMATH_CALUDE_fish_pond_area_increase_l494_49463

/-- Proves that the increase in area of a rectangular fish pond is (20x-4) square meters
    when both length and width are increased by 2 meters. -/
theorem fish_pond_area_increase (x : ℝ) :
  let original_length : ℝ := 5 * x
  let original_width : ℝ := 5 * x - 4
  let new_length : ℝ := original_length + 2
  let new_width : ℝ := original_width + 2
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 20 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_fish_pond_area_increase_l494_49463


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l494_49402

/-- Proves that given 48 pieces of junk mail and 8 houses, each house will receive 6 pieces of junk mail. -/
theorem junk_mail_distribution (total_mail : ℕ) (num_houses : ℕ) (h1 : total_mail = 48) (h2 : num_houses = 8) :
  total_mail / num_houses = 6 := by
sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l494_49402


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l494_49466

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0 → a < b) ∧
   ∃ a b : ℝ, a < b ∧ (a - b) * a^2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l494_49466


namespace NUMINAMATH_CALUDE_coord_sum_of_point_B_l494_49442

/-- Given two points A(0, 0) and B(x, 3) where the slope of AB is 3/4,
    prove that the sum of B's coordinates is 7. -/
theorem coord_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_coord_sum_of_point_B_l494_49442


namespace NUMINAMATH_CALUDE_spinner_probability_l494_49424

/-- A spinner with six sections numbered 1, 3, 5, 7, 8, and 9 -/
def Spinner : Finset ℕ := {1, 3, 5, 7, 8, 9}

/-- The set of numbers on the spinner that are less than 4 -/
def LessThan4 : Finset ℕ := Spinner.filter (· < 4)

/-- The probability of spinning a number less than 4 -/
def probability : ℚ := (LessThan4.card : ℚ) / (Spinner.card : ℚ)

theorem spinner_probability : probability = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l494_49424


namespace NUMINAMATH_CALUDE_g_of_5_equals_18_l494_49403

/-- Given a function g where g(x) = 4x - 2 for all x, prove that g(5) = 18 -/
theorem g_of_5_equals_18 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x - 2) : g 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_18_l494_49403


namespace NUMINAMATH_CALUDE_initial_birds_in_tree_l494_49489

theorem initial_birds_in_tree (additional_birds : ℕ) (total_birds : ℕ) 
  (h1 : additional_birds = 38) 
  (h2 : total_birds = 217) : 
  total_birds - additional_birds = 179 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_in_tree_l494_49489


namespace NUMINAMATH_CALUDE_set_A_equals_list_l494_49433

def A : Set ℚ := {x : ℚ | (x + 1) * (x - 2/3) * (x^2 - 2) = 0}

theorem set_A_equals_list : A = {-1, 2/3} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_list_l494_49433


namespace NUMINAMATH_CALUDE_change_in_math_preference_l494_49423

theorem change_in_math_preference (initial_yes initial_no final_yes final_no absentee_rate : ℝ) :
  initial_yes = 0.4 →
  initial_no = 0.6 →
  final_yes = 0.8 →
  final_no = 0.2 →
  absentee_rate = 0.1 →
  ∃ (min_change max_change : ℝ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    max_change - min_change = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_change_in_math_preference_l494_49423


namespace NUMINAMATH_CALUDE_other_root_of_complex_equation_l494_49448

theorem other_root_of_complex_equation (z : ℂ) :
  z^2 = -75 + 100*I ∧ z = 5 + 10*I → (-z)^2 = -75 + 100*I :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_equation_l494_49448


namespace NUMINAMATH_CALUDE_ball_exchange_game_theorem_l494_49487

/-- Represents a game played by n girls exchanging balls. -/
def BallExchangeGame (n : ℕ) := Unit

/-- A game is nice if at the end nobody has her own ball. -/
def is_nice (game : BallExchangeGame n) : Prop := sorry

/-- A game is tiresome if at the end everybody has her initial ball. -/
def is_tiresome (game : BallExchangeGame n) : Prop := sorry

/-- There exists a nice game for n players. -/
def exists_nice_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_nice game

/-- There exists a tiresome game for n players. -/
def exists_tiresome_game (n : ℕ) : Prop :=
  ∃ (game : BallExchangeGame n), is_tiresome game

theorem ball_exchange_game_theorem (n : ℕ) (h : n ≥ 2) :
  (exists_nice_game n ↔ n ≠ 3) ∧
  (exists_tiresome_game n ↔ n % 4 = 0 ∨ n % 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ball_exchange_game_theorem_l494_49487


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_unity_l494_49468

theorem cos_sum_seventh_roots_unity : 
  Real.cos (2 * π / 7) + Real.cos (4 * π / 7) + Real.cos (6 * π / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_unity_l494_49468


namespace NUMINAMATH_CALUDE_inequality_holds_l494_49493

theorem inequality_holds (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l494_49493


namespace NUMINAMATH_CALUDE_visitor_increase_percentage_l494_49453

/-- Proves that given an entry fee reduction of 25% and an increase in sale of 20%,
    the percentage increase in the number of visitors is 20%. -/
theorem visitor_increase_percentage
  (original_fee : ℝ)
  (fee_reduction_percentage : ℝ)
  (sale_increase_percentage : ℝ)
  (h1 : original_fee = 1)
  (h2 : fee_reduction_percentage = 25)
  (h3 : sale_increase_percentage = 20) :
  sale_increase_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_visitor_increase_percentage_l494_49453


namespace NUMINAMATH_CALUDE_parallelogram_altitude_theorem_l494_49416

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the properties of the parallelogram
def Parallelogram.DC (p : Parallelogram) : ℝ := sorry
def Parallelogram.EB (p : Parallelogram) : ℝ := sorry
def Parallelogram.DE (p : Parallelogram) : ℝ := sorry
def Parallelogram.DF (p : Parallelogram) : ℝ := sorry

-- State the theorem
theorem parallelogram_altitude_theorem (p : Parallelogram) 
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 9) :
  p.DF = 9 := by sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_theorem_l494_49416


namespace NUMINAMATH_CALUDE_certain_number_proof_l494_49430

theorem certain_number_proof (y : ℕ) : (2^14) - (2^12) = 3 * (2^y) → y = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l494_49430


namespace NUMINAMATH_CALUDE_school_supplies_cost_l494_49400

/-- The total cost of school supplies given the number of cartons, boxes per carton, and cost per unit -/
def total_cost (pencil_cartons : ℕ) (pencil_boxes_per_carton : ℕ) (pencil_cost_per_box : ℕ)
                (marker_cartons : ℕ) (marker_boxes_per_carton : ℕ) (marker_cost_per_carton : ℕ) : ℕ :=
  (pencil_cartons * pencil_boxes_per_carton * pencil_cost_per_box) +
  (marker_cartons * marker_cost_per_carton)

/-- Theorem stating that the total cost for the school's purchase is $440 -/
theorem school_supplies_cost :
  total_cost 20 10 2 10 5 4 = 440 := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l494_49400


namespace NUMINAMATH_CALUDE_apple_weeks_theorem_l494_49439

/-- The number of weeks Henry and his brother can spend eating apples -/
def appleWeeks (applesPerBox : ℕ) (numBoxes : ℕ) (applesPerPersonPerDay : ℕ) (numPeople : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (applesPerBox * numBoxes) / (applesPerPersonPerDay * numPeople * daysPerWeek)

/-- Theorem stating that Henry and his brother can spend 3 weeks eating the apples -/
theorem apple_weeks_theorem : appleWeeks 14 3 1 2 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_weeks_theorem_l494_49439


namespace NUMINAMATH_CALUDE_marks_deposit_is_88_l494_49461

-- Define Mark's deposit
def mark_deposit : ℕ := 88

-- Define Bryan's deposit in terms of Mark's
def bryan_deposit : ℕ := 5 * mark_deposit - 40

-- Theorem to prove
theorem marks_deposit_is_88 : mark_deposit = 88 := by
  sorry

end NUMINAMATH_CALUDE_marks_deposit_is_88_l494_49461


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l494_49476

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l494_49476


namespace NUMINAMATH_CALUDE_min_value_m_plus_n_l494_49469

theorem min_value_m_plus_n (a b m n : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : (a + b) / 2 = 1 / 2) (hm : m = a + 1 / a) (hn : n = b + 1 / b) :
  ∀ x y, x > 0 → y > 0 → (x + y) / 2 = 1 / 2 → 
  (x + 1 / x) + (y + 1 / y) ≥ m + n ∧ m + n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_m_plus_n_l494_49469


namespace NUMINAMATH_CALUDE_min_boat_speed_l494_49485

/-- The minimum speed required for a boat to complete a round trip on a river with a given flow speed, distance, and time constraint. -/
theorem min_boat_speed (S v : ℝ) (h_S : S > 0) (h_v : v ≥ 0) :
  let min_speed := (3 * S + Real.sqrt (9 * S^2 + 4 * v^2)) / 2
  ∀ x : ℝ, x ≥ min_speed →
    S / (x - v) + S / (x + v) + 1/12 ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_min_boat_speed_l494_49485


namespace NUMINAMATH_CALUDE_prime_dates_in_leap_year_l494_49419

def isPrimeMonth (m : Nat) : Bool :=
  m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 7 ∨ m = 11

def isPrimeDay (d : Nat) : Bool :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 17 ∨ d = 19 ∨ d = 23 ∨ d = 29 ∨ d = 31

def daysInMonth (m : Nat) : Nat :=
  if m = 2 then 29
  else if m = 4 ∨ m = 11 then 30
  else 31

def countPrimeDates : Nat :=
  (List.range 12).filter isPrimeMonth
    |>.map (fun m => (List.range (daysInMonth m)).filter isPrimeDay |>.length)
    |>.sum

theorem prime_dates_in_leap_year :
  countPrimeDates = 63 := by
  sorry

end NUMINAMATH_CALUDE_prime_dates_in_leap_year_l494_49419


namespace NUMINAMATH_CALUDE_min_b_value_l494_49450

/-- The function f(x) = x^2 + 2bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x

/-- The function g(x) = |x-1| -/
def g (x : ℝ) : ℝ := |x - 1|

/-- The theorem stating the minimum value of b -/
theorem min_b_value :
  ∀ b : ℝ,
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 →
    f b x₁ - f b x₂ < g x₁ - g x₂) →
  b ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_b_value_l494_49450


namespace NUMINAMATH_CALUDE_power_multiplication_l494_49495

theorem power_multiplication (x : ℝ) : x^5 * x^6 = x^11 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l494_49495


namespace NUMINAMATH_CALUDE_max_integer_k_l494_49429

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_l494_49429


namespace NUMINAMATH_CALUDE_inequality_proof_l494_49465

/-- An odd function f with the given property -/
def f (x : ℝ) : ℝ := sorry

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- The property x * f'(x) - f(x) < 0 for x ≠ 0 -/
axiom f_property (x : ℝ) (h : x ≠ 0) : x * f' x - f x < 0

/-- The main theorem -/
theorem inequality_proof :
  f (-3) / (-3) < f (Real.exp 1) / (Real.exp 1) ∧
  f (Real.exp 1) / (Real.exp 1) < f (Real.log 2) / (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l494_49465


namespace NUMINAMATH_CALUDE_parabola_equation_l494_49426

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the left vertex of the hyperbola
def left_vertex : ℝ × ℝ := (-3, 0)

-- Define the point P
def point_P : ℝ × ℝ := (2, -4)

-- Define the parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = 8 * x
def parabola2 (x y : ℝ) : Prop := x^2 = -y

-- Theorem statement
theorem parabola_equation :
  ∀ (f : ℝ × ℝ) (p : (ℝ × ℝ) → Prop),
    (f = left_vertex) →  -- The focus of the parabola is the left vertex of the hyperbola
    (p point_P) →  -- The parabola passes through point P
    (∀ (x y : ℝ), p (x, y) ↔ (parabola1 x y ∨ parabola2 x y)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l494_49426


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l494_49454

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 →                             -- One side is 12 cm
  (1/2) * a * b = 54 →                 -- Area of the triangle is 54 square centimeters
  a^2 + b^2 = c^2 →                    -- Pythagorean theorem (right-angled triangle)
  c = 15 :=                            -- Hypotenuse length is 15 cm
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l494_49454


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l494_49477

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →  -- Angle of inclination
  (Real.sin α + Real.cos α = 0) →  -- Given condition
  (a - b = 0) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l494_49477


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l494_49431

/-- Calculates the total expenditure for covering a rectangular floor with mat. -/
def total_expenditure (length width cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Proves that the total expenditure for covering a 20m × 15m floor with mat at Rs. 50 per square meter is Rs. 15,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 50 = 15000 := by
  sorry

end NUMINAMATH_CALUDE_hall_mat_expenditure_l494_49431


namespace NUMINAMATH_CALUDE_center_trajectory_is_parabola_l494_49480

/-- A circle passing through a point and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : center.1^2 + (center.2 - 3)^2 = radius^2
  tangentToLine : center.2 + radius = 3

/-- The trajectory of the center of a moving circle -/
def centerTrajectory (c : TangentCircle) : Prop :=
  c.center.1^2 = 12 * c.center.2

/-- Theorem: The trajectory of the center of a circle passing through (0, 3) 
    and tangent to y + 3 = 0 is described by x^2 = 12y -/
theorem center_trajectory_is_parabola :
  ∀ c : TangentCircle, centerTrajectory c :=
sorry

end NUMINAMATH_CALUDE_center_trajectory_is_parabola_l494_49480


namespace NUMINAMATH_CALUDE_electricity_bill_theorem_l494_49447

structure MeterReading where
  peak : ℕ
  night : ℕ
  offPeak : ℕ

structure TariffPrices where
  peak : ℚ
  night : ℚ
  offPeak : ℚ

def calculatePayment (prev : MeterReading) (curr : MeterReading) (prices : TariffPrices) : ℚ :=
  (curr.peak - prev.peak) * prices.peak +
  (curr.night - prev.night) * prices.night +
  (curr.offPeak - prev.offPeak) * prices.offPeak

def maxAdditionalPayment (prev : MeterReading) (curr : MeterReading) (prices : TariffPrices) : ℚ :=
  sorry

def expectedDifference (prev : MeterReading) (curr : MeterReading) (prices : TariffPrices) : ℚ :=
  sorry

theorem electricity_bill_theorem (prev : MeterReading) (curr : MeterReading) (prices : TariffPrices) :
  maxAdditionalPayment prev curr prices = 397.34 ∧
  expectedDifference prev curr prices = 19.3 :=
by sorry

end NUMINAMATH_CALUDE_electricity_bill_theorem_l494_49447


namespace NUMINAMATH_CALUDE_opposite_absolute_values_imply_y_power_x_l494_49407

theorem opposite_absolute_values_imply_y_power_x (x y : ℝ) : 
  |2*y - 3| + |5*x - 10| = 0 → y^x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_imply_y_power_x_l494_49407


namespace NUMINAMATH_CALUDE_marks_siblings_count_l494_49401

/-- The number of Mark's siblings given the egg distribution problem -/
def marks_siblings : ℕ :=
  let total_eggs : ℕ := 24  -- two dozen eggs
  let eggs_per_person : ℕ := 6
  let total_people : ℕ := total_eggs / eggs_per_person
  total_people - 1

theorem marks_siblings_count : marks_siblings = 3 := by
  sorry

end NUMINAMATH_CALUDE_marks_siblings_count_l494_49401


namespace NUMINAMATH_CALUDE_min_sum_floor_l494_49428

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(x^2 + y^2) / z⌋ + ⌊(y^2 + z^2) / x⌋ + ⌊(z^2 + x^2) / y⌋ = 4) ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    ⌊(a^2 + b^2) / c⌋ + ⌊(b^2 + c^2) / a⌋ + ⌊(c^2 + a^2) / b⌋ ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_floor_l494_49428


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l494_49470

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 → 
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l494_49470


namespace NUMINAMATH_CALUDE_inverse_function_equality_l494_49472

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that if f^(-1)(x) = 2, then b = 1 - 2a. -/
theorem inverse_function_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := fun x : ℝ => 2 / (a * x + b)
  (∃ x, f x = 2) → b = 1 - 2 * a :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_equality_l494_49472


namespace NUMINAMATH_CALUDE_average_of_eight_numbers_l494_49494

theorem average_of_eight_numbers :
  ∀ (a₁ a₂ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ),
    (a₁ + a₂) / 2 = 20 →
    (b₁ + b₂ + b₃) / 3 = 26 →
    c₁ = c₂ - 4 →
    c₁ = c₃ - 6 →
    c₃ = 30 →
    (a₁ + a₂ + b₁ + b₂ + b₃ + c₁ + c₂ + c₃) / 8 = 25 := by
  sorry


end NUMINAMATH_CALUDE_average_of_eight_numbers_l494_49494


namespace NUMINAMATH_CALUDE_coat_price_and_tax_l494_49481

/-- Represents the price of a coat -/
structure CoatPrice where
  original : ℝ
  discounted : ℝ
  taxRate : ℝ

/-- Calculates the tax amount based on the original price and tax rate -/
def calculateTax (price : CoatPrice) : ℝ :=
  price.original * price.taxRate

theorem coat_price_and_tax (price : CoatPrice) 
  (h1 : price.discounted = 72)
  (h2 : price.discounted = (2/5) * price.original)
  (h3 : price.taxRate = 0.05) :
  price.original = 180 ∧ calculateTax price = 9 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_and_tax_l494_49481


namespace NUMINAMATH_CALUDE_length_of_24_l494_49422

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem length_of_24 : length 24 = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_of_24_l494_49422


namespace NUMINAMATH_CALUDE_divisibility_of_T_l494_49432

def T (n : ℤ) : ℤ := (2*n)^2 + (2*n+2)^2 + (2*n+4)^2

theorem divisibility_of_T :
  (∀ n : ℤ, 4 ∣ T n) ∧ (∃ n : ℤ, ¬(7 ∣ T n)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_T_l494_49432


namespace NUMINAMATH_CALUDE_cone_volume_l494_49446

theorem cone_volume (s : ℝ) (θ : ℝ) (h : s = 6 ∧ θ = 2 * π / 3) :
  ∃ (V : ℝ), V = (16 * Real.sqrt 2 / 3) * π ∧
  V = (1 / 3) * π * (s * θ / (2 * π))^2 * Real.sqrt (s^2 - (s * θ / (2 * π))^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l494_49446


namespace NUMINAMATH_CALUDE_fixed_charge_is_six_l494_49414

/-- Represents Elvin's telephone bill components and totals -/
structure PhoneBill where
  fixed_charge : ℝ  -- Fixed monthly charge for internet service
  january_call_charge : ℝ  -- Charge for calls made in January
  january_total : ℝ  -- Total bill for January
  february_total : ℝ  -- Total bill for February

/-- Theorem stating that given the conditions, the fixed monthly charge is $6 -/
theorem fixed_charge_is_six (bill : PhoneBill) 
  (h1 : bill.fixed_charge + bill.january_call_charge = bill.january_total)
  (h2 : bill.fixed_charge + 2 * bill.january_call_charge = bill.february_total)
  (h3 : bill.january_total = 48)
  (h4 : bill.february_total = 90) :
  bill.fixed_charge = 6 := by
  sorry

end NUMINAMATH_CALUDE_fixed_charge_is_six_l494_49414


namespace NUMINAMATH_CALUDE_fixed_point_difference_l494_49437

/-- Given a function f(x) = a^(2x-6) + n, where a > 0 and a ≠ 1,
    and f(m) = 2, prove that m - n = 2 -/
theorem fixed_point_difference (a n m : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  (fun x ↦ a^(2*x - 6) + n) m = 2 → m - n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_difference_l494_49437


namespace NUMINAMATH_CALUDE_inequality_proof_l494_49427

theorem inequality_proof (x : ℝ) : 
  2 < x → x < 9/2 → (10*x^2 + 15*x - 75) / ((3*x - 6)*(x + 5)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l494_49427


namespace NUMINAMATH_CALUDE_slate_rock_count_l494_49492

theorem slate_rock_count :
  let pumice_count : ℕ := 11
  let granite_count : ℕ := 4
  let total_count (slate_count : ℕ) : ℕ := slate_count + pumice_count + granite_count
  let prob_two_slate (slate_count : ℕ) : ℚ :=
    (slate_count : ℚ) / (total_count slate_count : ℚ) *
    ((slate_count - 1 : ℚ) / (total_count slate_count - 1 : ℚ))
  ∃ (slate_count : ℕ),
    prob_two_slate slate_count = 15 / 100 ∧
    slate_count = 10 :=
by sorry

end NUMINAMATH_CALUDE_slate_rock_count_l494_49492


namespace NUMINAMATH_CALUDE_star_value_l494_49457

def star (a b : ℤ) : ℚ := 1 / a + 1 / b

theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 10) (h4 : a * b = 24) :
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l494_49457


namespace NUMINAMATH_CALUDE_same_terminal_side_negative_pi_sixth_same_terminal_side_l494_49445

theorem same_terminal_side (θ₁ θ₂ : ℝ) : ∃ k : ℤ, θ₂ = θ₁ + 2 * π * k → 
  (θ₁.cos = θ₂.cos ∧ θ₁.sin = θ₂.sin) :=
by sorry

theorem negative_pi_sixth_same_terminal_side : 
  ∃ k : ℤ, (11 * π / 6 : ℝ) = -π / 6 + 2 * π * k :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_negative_pi_sixth_same_terminal_side_l494_49445


namespace NUMINAMATH_CALUDE_part_one_part_two_l494_49471

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 1|

-- Part 1
theorem part_one : 
  (∀ x : ℝ, f 2 x < 5 ↔ x ∈ Set.Ioo (-2) 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4 - |a - 1|) ↔ 
    a ∈ Set.Iic (-2) ∪ Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l494_49471
