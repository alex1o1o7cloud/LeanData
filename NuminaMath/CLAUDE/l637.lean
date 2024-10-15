import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_l637_63767

/-- Given a rectangle where the length is thrice the breadth and the perimeter is 96,
    prove that its area is 432. -/
theorem rectangle_area (breadth : ℝ) (length : ℝ) : 
  length = 3 * breadth → 
  2 * (length + breadth) = 96 → 
  length * breadth = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l637_63767


namespace NUMINAMATH_CALUDE_complement_of_M_l637_63740

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l637_63740


namespace NUMINAMATH_CALUDE_simplify_expression_l637_63766

theorem simplify_expression (x : ℝ) (h : x ≥ 0) : 
  (1/2 * x^(1/2))^4 = 1/16 * x^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l637_63766


namespace NUMINAMATH_CALUDE_two_x_power_x_eq_sqrt_two_solutions_l637_63764

theorem two_x_power_x_eq_sqrt_two_solutions (x : ℝ) :
  x > 0 ∧ 2 * (x ^ x) = Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_two_x_power_x_eq_sqrt_two_solutions_l637_63764


namespace NUMINAMATH_CALUDE_three_sqrt_two_gt_sqrt_seventeen_l637_63749

theorem three_sqrt_two_gt_sqrt_seventeen : 3 * Real.sqrt 2 > Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_three_sqrt_two_gt_sqrt_seventeen_l637_63749


namespace NUMINAMATH_CALUDE_hundredth_term_is_one_l637_63775

/-- Defines the sequence term at position n -/
def sequenceTerm (n : ℕ) : ℕ :=
  sorry

/-- The number of elements in the first n groups -/
def elementsInGroups (n : ℕ) : ℕ :=
  n^2

theorem hundredth_term_is_one :
  sequenceTerm 100 = 1 :=
sorry

end NUMINAMATH_CALUDE_hundredth_term_is_one_l637_63775


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l637_63703

/-- Given an ellipse and hyperbola with common foci F₁ and F₂, intersecting at point P -/
structure EllipseHyperbolaIntersection where
  /-- The eccentricity of the ellipse -/
  e₁ : ℝ
  /-- The eccentricity of the hyperbola -/
  e₂ : ℝ
  /-- Angle F₁PF₂ -/
  angle_F₁PF₂ : ℝ
  /-- 0 < e₁ < 1 (eccentricity of ellipse) -/
  h₁ : 0 < e₁ ∧ e₁ < 1
  /-- e₂ > 1 (eccentricity of hyperbola) -/
  h₂ : e₂ > 1
  /-- cos ∠F₁PF₂ = 3/5 -/
  h₃ : Real.cos angle_F₁PF₂ = 3/5
  /-- e₂ = 2e₁ -/
  h₄ : e₂ = 2 * e₁

/-- The eccentricity of the ellipse is √10/5 -/
theorem ellipse_eccentricity (eh : EllipseHyperbolaIntersection) : eh.e₁ = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l637_63703


namespace NUMINAMATH_CALUDE_distance_AB_DB1_is_12_div_5_l637_63737

/-- A rectangular prism with given dimensions -/
structure RectangularPrism where
  AB : ℝ
  BC : ℝ
  BB1 : ℝ

/-- The distance between AB and DB₁ in a rectangular prism -/
def distance_AB_DB1 (prism : RectangularPrism) : ℝ := sorry

theorem distance_AB_DB1_is_12_div_5 (prism : RectangularPrism) 
  (h1 : prism.AB = 5)
  (h2 : prism.BC = 4)
  (h3 : prism.BB1 = 3) :
  distance_AB_DB1 prism = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_DB1_is_12_div_5_l637_63737


namespace NUMINAMATH_CALUDE_job_total_amount_l637_63786

/-- Calculates the total amount earned for a job given the time taken by two workers and one worker's share. -/
theorem job_total_amount 
  (rahul_days : ℚ) 
  (rajesh_days : ℚ) 
  (rahul_share : ℚ) : 
  rahul_days = 3 → 
  rajesh_days = 2 → 
  rahul_share = 142 → 
  ∃ (total_amount : ℚ), total_amount = 355 := by
  sorry

#check job_total_amount

end NUMINAMATH_CALUDE_job_total_amount_l637_63786


namespace NUMINAMATH_CALUDE_stratified_sample_fourth_unit_l637_63773

/-- Represents a stratified sample from four units -/
structure StratifiedSample :=
  (total : ℕ)
  (unit_samples : Fin 4 → ℕ)
  (is_arithmetic : ∃ d : ℤ, ∀ i : Fin 3, (unit_samples i.succ : ℤ) - (unit_samples i) = d)
  (sum_to_total : (Finset.univ.sum unit_samples) = total)

/-- The theorem statement -/
theorem stratified_sample_fourth_unit 
  (sample : StratifiedSample)
  (total_collected : ℕ)
  (h_total : sample.total = 150)
  (h_collected : total_collected = 1000)
  (h_second_unit : sample.unit_samples 1 = 30) :
  sample.unit_samples 3 = 60 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_fourth_unit_l637_63773


namespace NUMINAMATH_CALUDE_equality_of_pairs_l637_63708

theorem equality_of_pairs (a b x y : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0)
  (h_sum : a + b + x + y < 2)
  (h_eq1 : a + b^2 = x + y^2)
  (h_eq2 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by
  sorry

end NUMINAMATH_CALUDE_equality_of_pairs_l637_63708


namespace NUMINAMATH_CALUDE_final_state_theorem_l637_63711

/-- Represents the state of the cage -/
structure CageState where
  crickets : ℕ
  katydids : ℕ

/-- Represents a magician's transformation -/
inductive Transformation
  | Red
  | Green

/-- Applies a single transformation to the cage state -/
def applyTransformation (state : CageState) (t : Transformation) : CageState :=
  match t with
  | Transformation.Red => 
      { crickets := state.crickets + 1, katydids := state.katydids - 2 }
  | Transformation.Green => 
      { crickets := state.crickets - 5, katydids := state.katydids + 2 }

/-- Applies a sequence of transformations to the cage state -/
def applyTransformations (state : CageState) (ts : List Transformation) : CageState :=
  match ts with
  | [] => state
  | t::rest => applyTransformations (applyTransformation state t) rest

theorem final_state_theorem (transformations : List Transformation) :
  transformations.length = 15 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).crickets = 0 →
  (applyTransformations { crickets := 21, katydids := 30 } transformations).katydids = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_final_state_theorem_l637_63711


namespace NUMINAMATH_CALUDE_toy_purchase_problem_l637_63770

theorem toy_purchase_problem (toy_cost : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  toy_cost = 3 →
  discount_rate = 0.2 →
  total_paid = 12 →
  (1 - discount_rate) * (toy_cost * (total_paid / ((1 - discount_rate) * toy_cost))) = total_paid →
  ∃ n : ℕ, n = 5 ∧ n * toy_cost = total_paid / (1 - discount_rate) := by
  sorry

end NUMINAMATH_CALUDE_toy_purchase_problem_l637_63770


namespace NUMINAMATH_CALUDE_female_democrats_count_l637_63774

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l637_63774


namespace NUMINAMATH_CALUDE_largest_number_value_l637_63754

theorem largest_number_value (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_diff_large : c - b = 10)
  (h_diff_small : b - a = 5) :
  c = 41.67 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l637_63754


namespace NUMINAMATH_CALUDE_quadratic_square_form_sum_l637_63787

theorem quadratic_square_form_sum (x : ℝ) :
  ∃ (a b c : ℤ), a > 0 ∧
  (25 * x^2 + 30 * x - 35 = 0 ↔ (a * x + b)^2 = c) ∧
  a + b + c = 52 := by sorry

end NUMINAMATH_CALUDE_quadratic_square_form_sum_l637_63787


namespace NUMINAMATH_CALUDE_set_intersection_problem_l637_63709

theorem set_intersection_problem (M N : Set ℤ) (a : ℤ) 
  (hM : M = {a, 0})
  (hN : N = {1, 2})
  (hIntersection : M ∩ N = {1}) :
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l637_63709


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l637_63707

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (hours_per_shift : ℕ) (hours_per_chair : ℕ) (days : ℕ) : ℕ :=
  (hours_per_shift * days) / hours_per_chair

/-- Proves that a worker working 8-hour shifts, taking 5 hours per chair, can build 16 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l637_63707


namespace NUMINAMATH_CALUDE_machine_work_time_equation_l637_63741

theorem machine_work_time_equation (x : ℝ) (hx : x > 0) : 
  (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x) = 1 / x) → x = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_equation_l637_63741


namespace NUMINAMATH_CALUDE_ball_attendance_l637_63739

theorem ball_attendance :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  (∃ k : ℕ, n = 20 * k ∧ m = 21 * k) →
  n + m = 41 :=
λ n m h1 h2 h3 =>
  sorry

end NUMINAMATH_CALUDE_ball_attendance_l637_63739


namespace NUMINAMATH_CALUDE_f_of_3_equals_15_l637_63760

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement
theorem f_of_3_equals_15 : f 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_15_l637_63760


namespace NUMINAMATH_CALUDE_cubic_equation_root_l637_63702

theorem cubic_equation_root (a b : ℚ) :
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 12 = 0 →
  b = -47 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l637_63702


namespace NUMINAMATH_CALUDE_sum_equals_250_l637_63783

theorem sum_equals_250 : 157 + 18 + 32 + 43 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_250_l637_63783


namespace NUMINAMATH_CALUDE_min_cards_for_four_of_a_kind_standard_deck_l637_63712

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : Nat)
  (num_ranks : Nat)
  (cards_per_rank : Nat)
  (num_jokers : Nat)

/-- Calculates the minimum number of cards needed to guarantee "a four of a kind" -/
def min_cards_for_four_of_a_kind (d : Deck) : Nat :=
  d.num_jokers + (d.num_ranks * (d.cards_per_rank - 1)) + 1

/-- Theorem stating the minimum number of cards needed for "a four of a kind" in a standard deck -/
theorem min_cards_for_four_of_a_kind_standard_deck :
  let standard_deck : Deck := {
    total_cards := 52,
    num_ranks := 13,
    cards_per_rank := 4,
    num_jokers := 2
  }
  min_cards_for_four_of_a_kind standard_deck = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_for_four_of_a_kind_standard_deck_l637_63712


namespace NUMINAMATH_CALUDE_negation_equivalence_l637_63778

theorem negation_equivalence :
  (¬ ∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x + 2 = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l637_63778


namespace NUMINAMATH_CALUDE_flag_designs_count_l637_63704

/-- The number of colors available for the flag. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs. -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27. -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l637_63704


namespace NUMINAMATH_CALUDE_square_difference_l637_63751

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l637_63751


namespace NUMINAMATH_CALUDE_gcf_lcm_300_105_l637_63733

theorem gcf_lcm_300_105 : ∃ (gcf lcm : ℕ),
  (Nat.gcd 300 105 = gcf) ∧
  (Nat.lcm 300 105 = lcm) ∧
  (gcf = 15) ∧
  (lcm = 2100) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_300_105_l637_63733


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l637_63748

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + b - c = 0) : (a^3 + b^3 + c^3) / (a * b * c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l637_63748


namespace NUMINAMATH_CALUDE_intersection_of_lines_l637_63782

/-- Proves the existence and uniqueness of the intersection point of two lines, if it exists -/
theorem intersection_of_lines (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) (h2 : c ≠ 0 ∨ d ≠ 0) 
  (h3 : a * d ≠ b * c) : 
  ∃! p : ℝ × ℝ, a * p.1 + b * p.2 + e = 0 ∧ c * p.1 + d * p.2 + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l637_63782


namespace NUMINAMATH_CALUDE_part_one_part_two_l637_63753

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 3 ↔ x ≤ 1 ∨ x ≥ 5) → a = 2 :=
sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f 2 x + f 2 (x + 4) ≥ m) → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l637_63753


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l637_63793

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- State that AB is perpendicular to x-axis
def AB_perpendicular_to_x : Prop := sorry

-- Define the perimeter of triangle AF₁B
def perimeter_AF1B : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter :
  ellipse point_A.1 point_A.2 ∧
  ellipse point_B.1 point_B.2 ∧
  AB_perpendicular_to_x →
  perimeter_AF1B = 24 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l637_63793


namespace NUMINAMATH_CALUDE_average_salary_before_manager_l637_63797

/-- Proves that the average salary of employees is 1500 given the conditions -/
theorem average_salary_before_manager (num_employees : ℕ) (manager_salary : ℕ) (avg_increase : ℕ) :
  num_employees = 20 →
  manager_salary = 12000 →
  avg_increase = 500 →
  (∃ (avg_salary : ℕ),
    (num_employees + 1) * (avg_salary + avg_increase) = num_employees * avg_salary + manager_salary ∧
    avg_salary = 1500) :=
by sorry

end NUMINAMATH_CALUDE_average_salary_before_manager_l637_63797


namespace NUMINAMATH_CALUDE_reflection_of_P_l637_63713

/-- Given a point P in a Cartesian coordinate system, 
    return its coordinates with respect to the origin -/
def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

/-- Theorem: The reflection of point P(2,1) across the origin is (-2,-1) -/
theorem reflection_of_P : reflect_point (2, 1) = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_P_l637_63713


namespace NUMINAMATH_CALUDE_perimeter_is_18_l637_63726

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the left branch
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line passing through F1, A, and B
def line_through_F1AB (p : ℝ × ℝ) : Prop := sorry

-- State that A and B are on the hyperbola
axiom A_on_hyperbola : hyperbola A.1 A.2
axiom B_on_hyperbola : hyperbola B.1 B.2

-- State that A and B are on the line passing through F1
axiom A_on_line : line_through_F1AB A
axiom B_on_line : line_through_F1AB B

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- State that the distance between A and B is 5
axiom AB_distance : distance A B = 5

-- Define the perimeter of triangle AF2B
def perimeter_AF2B : ℝ := distance A F2 + distance B F2 + distance A B

-- Theorem to prove
theorem perimeter_is_18 : perimeter_AF2B = 18 := by sorry

end NUMINAMATH_CALUDE_perimeter_is_18_l637_63726


namespace NUMINAMATH_CALUDE_intersection_equals_two_to_infinity_l637_63700

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log ((1 - x) / x)}
def N : Set ℝ := {y | ∃ x, y = x^2 + 2*x + 3}

-- Define the complement of M in ℝ
def M_complement : Set ℝ := {x | x ∉ M}

-- Define the set [2, +∞)
def two_to_infinity : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem intersection_equals_two_to_infinity : (M_complement ∩ N) = two_to_infinity := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_two_to_infinity_l637_63700


namespace NUMINAMATH_CALUDE_probability_B_outscores_A_is_correct_l637_63779

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : Rat

/-- The probability that team B finishes with more points than team A -/
def probability_B_outscores_A (tournament : SoccerTournament) : Rat :=
  793 / 2048

/-- Theorem stating the probability that team B finishes with more points than team A -/
theorem probability_B_outscores_A_is_correct (tournament : SoccerTournament) 
  (h1 : tournament.num_teams = 8)
  (h2 : tournament.games_per_team = 7)
  (h3 : tournament.win_probability = 1 / 2) : 
  probability_B_outscores_A tournament = 793 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_B_outscores_A_is_correct_l637_63779


namespace NUMINAMATH_CALUDE_zhang_income_ratio_l637_63710

/-- Represents the per capita income of a village at a given time -/
structure Income where
  amount : ℝ

/-- Represents the state of two villages' incomes at two different times -/
structure VillageIncomes where
  li_past : Income
  li_present : Income
  zhang_past : Income
  zhang_present : Income

/-- The conditions of the problem -/
def income_conditions (v : VillageIncomes) : Prop :=
  v.zhang_past.amount = 0.4 * v.li_past.amount ∧
  v.zhang_present.amount = 0.8 * v.li_present.amount ∧
  v.li_present.amount = 3 * v.li_past.amount

/-- The theorem to be proved -/
theorem zhang_income_ratio (v : VillageIncomes) 
  (h : income_conditions v) : 
  v.zhang_present.amount / v.zhang_past.amount = 6 := by
  sorry


end NUMINAMATH_CALUDE_zhang_income_ratio_l637_63710


namespace NUMINAMATH_CALUDE_certain_number_problem_l637_63762

theorem certain_number_problem (A B : ℝ) (h1 : A + B = 15) (h2 : A = 7) :
  ∃ C : ℝ, C * B = 5 * A - 11 ∧ C = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l637_63762


namespace NUMINAMATH_CALUDE_dragon_boat_festival_visitors_scientific_notation_l637_63719

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem dragon_boat_festival_visitors_scientific_notation :
  toScientificNotation 82600000 = ScientificNotation.mk 8.26 7 sorry := by
  sorry

end NUMINAMATH_CALUDE_dragon_boat_festival_visitors_scientific_notation_l637_63719


namespace NUMINAMATH_CALUDE_total_sweets_l637_63756

theorem total_sweets (red_sweets : ℕ) (green_sweets : ℕ) (other_sweets : ℕ)
  (h1 : red_sweets = 49)
  (h2 : green_sweets = 59)
  (h3 : other_sweets = 177) :
  red_sweets + green_sweets + other_sweets = 285 := by
sorry

end NUMINAMATH_CALUDE_total_sweets_l637_63756


namespace NUMINAMATH_CALUDE_neighborhood_to_gina_litter_ratio_l637_63743

/-- Given the following conditions:
  * Gina collected 2 bags of litter
  * Each bag of litter weighs 4 pounds
  * Total litter collected by everyone is 664 pounds
  Prove that the ratio of litter collected by the rest of the neighborhood
  to the amount collected by Gina is 82:1 -/
theorem neighborhood_to_gina_litter_ratio :
  let gina_bags : ℕ := 2
  let bag_weight : ℕ := 4
  let total_litter : ℕ := 664
  let gina_litter := gina_bags * bag_weight
  let neighborhood_litter := total_litter - gina_litter
  neighborhood_litter / gina_litter = 82 ∧ gina_litter ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_to_gina_litter_ratio_l637_63743


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l637_63722

theorem orthogonal_vectors (x y : ℝ) : 
  (3 * x + 4 * (-2) = 0 ∧ 3 * 1 + 4 * y = 0) ↔ (x = 8/3 ∧ y = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l637_63722


namespace NUMINAMATH_CALUDE_product_of_integers_l637_63763

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18) 
  (diff_squares_eq : x^2 - y^2 = 36) : 
  x * y = 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l637_63763


namespace NUMINAMATH_CALUDE_item_price_ratio_l637_63758

theorem item_price_ratio (c p q : ℝ) (h1 : p = 0.8 * c) (h2 : q = 1.2 * c) : q / p = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_item_price_ratio_l637_63758


namespace NUMINAMATH_CALUDE_cylinder_volume_l637_63790

/-- The volume of a solid cylinder in a cubic container --/
theorem cylinder_volume (container_side : ℝ) (exposed_height : ℝ) (base_area_ratio : ℝ) :
  container_side = 20 →
  exposed_height = 8 →
  base_area_ratio = 1/8 →
  (container_side - exposed_height) * (container_side * container_side * base_area_ratio) = 650 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l637_63790


namespace NUMINAMATH_CALUDE_infinitely_many_special_triangles_l637_63777

/-- A triangle with integer side lengths and area, where one side is 4 and the difference between the other two sides is 2. -/
structure SpecialTriangle where
  a : ℕ+  -- First side length
  b : ℕ+  -- Second side length
  c : ℕ+  -- Third side length (always 4)
  area : ℕ+  -- Area of the triangle
  h_c : c = 4  -- One side is 4
  h_diff : a - b = 2 ∨ b - a = 2  -- Difference between other two sides is 2
  h_triangle : a + b > c ∧ b + c > a ∧ a + c > b  -- Triangle inequality
  h_area : 4 * area ^ 2 = (a + b + c) * (a + b - c) * (b + c - a) * (a + c - b)  -- Heron's formula

/-- There are infinitely many special triangles. -/
theorem infinitely_many_special_triangles : ∀ n : ℕ, ∃ m > n, ∃ t : SpecialTriangle, m = t.a.val := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_special_triangles_l637_63777


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l637_63746

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x - 4| ≤ 6) ↔ (-2 ≤ x ∧ x ≤ 10) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l637_63746


namespace NUMINAMATH_CALUDE_farm_animals_l637_63731

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 2 * (cows + chickens + ducks) + 22) →
  (chickens + ducks = 2 * cows) →
  (cows = 11) := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l637_63731


namespace NUMINAMATH_CALUDE_original_sugar_percentage_l637_63788

/-- Given a solution where one fourth is replaced by a 42% sugar solution,
    resulting in an 18% sugar solution, prove that the original solution
    must have been 10% sugar. -/
theorem original_sugar_percentage
  (original : ℝ)
  (replaced : ℝ := 1/4)
  (second_solution : ℝ := 42)
  (final_solution : ℝ := 18)
  (h : (1 - replaced) * original + replaced * second_solution = final_solution) :
  original = 10 :=
sorry

end NUMINAMATH_CALUDE_original_sugar_percentage_l637_63788


namespace NUMINAMATH_CALUDE_g_value_at_five_sixths_l637_63785

/-- Given a function f and g with specific properties, prove that g(5/6) = -√3/2 -/
theorem g_value_at_five_sixths 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = Real.sqrt 2 * Real.sin (a * x + π / 4))
  (h3 : ∀ x, x ≥ 0 → g x = g (x - 1))
  (h4 : ∀ x, x < 0 → g x = Real.sin (a * x))
  (h5 : ∃ T, T > 0 ∧ T = 1 ∧ ∀ x, f (x + T) = f x) :
  g (5/6) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_five_sixths_l637_63785


namespace NUMINAMATH_CALUDE_percentage_increase_l637_63729

theorem percentage_increase (x y z : ℝ) : 
  y = 0.5 * z →  -- y is 50% less than z
  x = 0.6 * z →  -- x is 60% of z
  x = 1.2 * y    -- x is 20% more than y (equivalent to 120% of y)
  := by sorry

end NUMINAMATH_CALUDE_percentage_increase_l637_63729


namespace NUMINAMATH_CALUDE_one_point_zero_six_million_scientific_notation_l637_63794

theorem one_point_zero_six_million_scientific_notation :
  (1.06 : ℝ) * (1000000 : ℝ) = (1.06 : ℝ) * (10 ^ 6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_one_point_zero_six_million_scientific_notation_l637_63794


namespace NUMINAMATH_CALUDE_four_variable_equation_consecutive_evens_l637_63780

theorem four_variable_equation_consecutive_evens :
  ∃ (x y z w : ℕ), 
    (x + y + z + w = 100) ∧ 
    (∃ (k : ℕ), x = 2 * k) ∧
    (∃ (l : ℕ), y = 2 * l) ∧
    (∃ (m : ℕ), z = 2 * m) ∧
    (∃ (n : ℕ), w = 2 * n) ∧
    (y = x + 2) ∧
    (z = x + 4) ∧
    (w = x + 6) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) := by
  sorry

end NUMINAMATH_CALUDE_four_variable_equation_consecutive_evens_l637_63780


namespace NUMINAMATH_CALUDE_no_square_root_representation_l637_63772

theorem no_square_root_representation : ¬ ∃ (A B : ℤ), (A + B * Real.sqrt 3) ^ 2 = 99999 + 111111 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_no_square_root_representation_l637_63772


namespace NUMINAMATH_CALUDE_group_division_ways_l637_63750

theorem group_division_ways (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) : 
  Nat.choose n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_group_division_ways_l637_63750


namespace NUMINAMATH_CALUDE_smallest_lcm_four_digit_gcd_five_l637_63759

theorem smallest_lcm_four_digit_gcd_five (k l : ℕ) : 
  1000 ≤ k ∧ k < 10000 ∧ 
  1000 ≤ l ∧ l < 10000 ∧ 
  Nat.gcd k l = 5 →
  201000 ≤ Nat.lcm k l ∧ 
  ∃ (k₀ l₀ : ℕ), 1000 ≤ k₀ ∧ k₀ < 10000 ∧ 
                 1000 ≤ l₀ ∧ l₀ < 10000 ∧ 
                 Nat.gcd k₀ l₀ = 5 ∧
                 Nat.lcm k₀ l₀ = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_four_digit_gcd_five_l637_63759


namespace NUMINAMATH_CALUDE_calculator_result_l637_63796

def calculator_operation (n : ℕ) : ℕ :=
  let doubled := n * 2
  let swapped := (doubled % 10) * 10 + (doubled / 10)
  swapped + 2

def is_valid_input (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 49

theorem calculator_result :
  (∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 44) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 43) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 42) ∧
  (¬ ∃ n : ℕ, is_valid_input n ∧ calculator_operation n = 41) :=
sorry

end NUMINAMATH_CALUDE_calculator_result_l637_63796


namespace NUMINAMATH_CALUDE_marble_probability_l637_63720

theorem marble_probability : 
  let green : ℕ := 4
  let white : ℕ := 3
  let red : ℕ := 5
  let blue : ℕ := 6
  let total : ℕ := green + white + red + blue
  let favorable : ℕ := green + white
  (favorable : ℚ) / total = 7 / 18 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l637_63720


namespace NUMINAMATH_CALUDE_girls_percentage_l637_63784

/-- The percentage of girls in a school with 150 total students and 60 boys is 60%. -/
theorem girls_percentage (total : ℕ) (boys : ℕ) (h1 : total = 150) (h2 : boys = 60) :
  (total - boys : ℚ) / total * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_l637_63784


namespace NUMINAMATH_CALUDE_distance_to_big_rock_big_rock_distance_l637_63768

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) : 
  v > c ∧ v > 0 ∧ c > 0 ∧ t > 0 → 
  (v + c)⁻¹ * d + (v - c)⁻¹ * d = t → 
  d = (t * v^2 - t * c^2) / (2 * v) :=
by sorry

/-- The specific case for the given problem -/
theorem big_rock_distance : 
  let v := 6 -- rower's speed in still water
  let c := 1 -- river current speed
  let t := 1 -- total time for round trip
  let d := (t * v^2 - t * c^2) / (2 * v) -- distance to Big Rock
  d = 35 / 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_big_rock_distance_l637_63768


namespace NUMINAMATH_CALUDE_neighborhood_glass_panels_l637_63799

/-- Represents the number of houses of each type -/
def num_houses_A : ℕ := 4
def num_houses_B : ℕ := 3
def num_houses_C : ℕ := 3

/-- Represents the number of glass panels in each type of house -/
def panels_per_house_A : ℕ := 
  4 * 6 + -- double windows downstairs
  8 * 3 + -- single windows upstairs
  2 * 6 + -- bay windows
  1 * 2 + -- front door
  1 * 3   -- back door

def panels_per_house_B : ℕ := 
  8 * 5 + -- double windows downstairs
  6 * 4 + -- single windows upstairs
  1 * 7 + -- bay window
  1 * 4   -- front door

def panels_per_house_C : ℕ := 
  5 * 4 + -- double windows downstairs
  10 * 2 + -- single windows upstairs
  3 * 1   -- skylights

/-- The total number of glass panels in the neighborhood -/
def total_panels : ℕ := 
  num_houses_A * panels_per_house_A +
  num_houses_B * panels_per_house_B +
  num_houses_C * panels_per_house_C

/-- Theorem stating that the total number of glass panels in the neighborhood is 614 -/
theorem neighborhood_glass_panels : total_panels = 614 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_glass_panels_l637_63799


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_properties_l637_63742

structure RegularTriangularPyramid where
  PA : ℝ
  θ : ℝ

def distance_to_base (p : RegularTriangularPyramid) : ℝ :=
  sorry

def surface_area (p : RegularTriangularPyramid) : ℝ :=
  sorry

theorem regular_triangular_pyramid_properties
  (p : RegularTriangularPyramid)
  (h1 : p.PA = 2)
  (h2 : 0 < p.θ ∧ p.θ ≤ π / 2) :
  (distance_to_base { PA := 2, θ := π / 2 } = 2 * Real.sqrt 3 / 3) ∧
  (∀ θ₁ θ₂, 0 < θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ π / 2 →
    surface_area { PA := 2, θ := θ₁ } < surface_area { PA := 2, θ := θ₂ }) :=
sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_properties_l637_63742


namespace NUMINAMATH_CALUDE_ellipse_theorem_l637_63776

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 6 / 9
  h_major : 2 * a = 2 * Real.sqrt 3

/-- A line that intersects the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) ∧
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) ∧
    (y₁ = k * x₁ + m) ∧
    (y₂ = k * x₂ + m) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The main theorem -/
theorem ellipse_theorem (E : Ellipse) (L : IntersectingLine E) :
  (E.a^2 = 3 ∧ E.b^2 = 1) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) →
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) →
    (y₁ = L.k * x₁ + L.m) →
    (y₂ = L.k * x₂ + L.m) →
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (abs L.m / Real.sqrt (1 + L.k^2) = Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l637_63776


namespace NUMINAMATH_CALUDE_journey_length_l637_63714

theorem journey_length (first_part second_part third_part total : ℝ) 
  (h1 : first_part = (1/4) * total)
  (h2 : second_part = 30)
  (h3 : third_part = (1/3) * total)
  (h4 : total = first_part + second_part + third_part) :
  total = 72 := by
  sorry

end NUMINAMATH_CALUDE_journey_length_l637_63714


namespace NUMINAMATH_CALUDE_sum_first_ten_terms_l637_63718

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_first_ten_terms :
  sum_arithmetic_sequence (-3) 4 10 = 150 := by
sorry

end NUMINAMATH_CALUDE_sum_first_ten_terms_l637_63718


namespace NUMINAMATH_CALUDE_power_product_square_l637_63717

theorem power_product_square (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by sorry

end NUMINAMATH_CALUDE_power_product_square_l637_63717


namespace NUMINAMATH_CALUDE_g_equals_g_l637_63701

/-- Two triangles are similar isosceles triangles with vertex A and angle α -/
def similarIsoscelesA (t1 t2 : Set (ℝ × ℝ)) (A : ℝ × ℝ) (α : ℝ) : Prop :=
  sorry

/-- Two triangles are similar isosceles triangles with angle π - α at the vertex -/
def similarIsoscelesVertex (t1 t2 : Set (ℝ × ℝ)) (α : ℝ) : Prop :=
  sorry

/-- The theorem stating that G = G' given the conditions -/
theorem g_equals_g' (A K L M N G G' : ℝ × ℝ) (α : ℝ) 
    (h1 : similarIsoscelesA {A, K, L} {A, M, N} A α)
    (h2 : similarIsoscelesVertex {G, N, K} {G', L, M} α) :
    G = G' :=
  sorry

end NUMINAMATH_CALUDE_g_equals_g_l637_63701


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l637_63723

def is_valid_number (N : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    N = 100 * a + 10 * b + c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    (a + b + 1 = (a + b + c) / 3 ∨
     a + (b + 1) + 1 = (a + b + c) / 3)

theorem three_digit_sum_property :
  ∀ N : ℕ, is_valid_number N → (N = 207 ∨ N = 117 ∨ N = 108) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l637_63723


namespace NUMINAMATH_CALUDE_x_cubed_term_is_seventh_l637_63791

/-- The exponent of the binomial expansion -/
def n : ℕ := 16

/-- The general term of the expansion -/
def T (r : ℕ) : ℚ → ℚ := λ x => 2^r * Nat.choose n r * x^(8 - 5/6 * r)

/-- The index of the term containing x^3 -/
def r : ℕ := 6

theorem x_cubed_term_is_seventh :
  T r = T 6 ∧ 8 - 5/6 * r = 3 ∧ r + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_term_is_seventh_l637_63791


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l637_63744

theorem prob_at_least_one_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 5 →
  red_balls = 2 →
  white_balls = 3 →
  (red_balls + white_balls = total_balls) →
  (probability_at_least_one_red : ℚ) =
    1 - (white_balls / total_balls * (white_balls - 1) / (total_balls - 1)) →
  probability_at_least_one_red = 7 / 10 := by
  sorry

#check prob_at_least_one_red

end NUMINAMATH_CALUDE_prob_at_least_one_red_l637_63744


namespace NUMINAMATH_CALUDE_cows_husk_consumption_l637_63757

/-- The number of bags of husk eaten by a given number of cows in 45 days -/
def bags_eaten (num_cows : ℕ) : ℕ :=
  num_cows

/-- Theorem stating that 45 cows eat 45 bags of husk in 45 days -/
theorem cows_husk_consumption :
  bags_eaten 45 = 45 := by
  sorry

end NUMINAMATH_CALUDE_cows_husk_consumption_l637_63757


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l637_63745

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hno_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (¬ (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x)) ∧
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l637_63745


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l637_63732

/-- Given x > 0 and y > 0, the minimum value of (x+2y)^+ is 9 -/
theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (m : ℝ), m = 9 ∧ ∀ z, z = x + 2*y → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l637_63732


namespace NUMINAMATH_CALUDE_halfway_fraction_l637_63755

theorem halfway_fraction (a b c d : ℤ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 6) :
  (a : ℚ) / b + ((c : ℚ) / d - (a : ℚ) / b) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l637_63755


namespace NUMINAMATH_CALUDE_tangent_line_polar_equation_l637_63705

/-- Given a circle in polar form ρ = 4sinθ and a point (2√2, π/4),
    the polar equation of the tangent line passing through this point is ρcosθ = 2 -/
theorem tangent_line_polar_equation
  (ρ θ : ℝ) 
  (circle_eq : ρ = 4 * Real.sin θ) 
  (point : (ρ, θ) = (2 * Real.sqrt 2, Real.pi / 4)) :
  ∃ (k : ℝ), ρ * Real.cos θ = k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_polar_equation_l637_63705


namespace NUMINAMATH_CALUDE_b_four_lt_b_seven_l637_63747

def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (b n α + 1 / α (n + 1))

theorem b_four_lt_b_seven (α : ℕ → ℕ) (h : ∀ k, α k ≥ 1) :
  b 4 α < b 7 α := by
  sorry

end NUMINAMATH_CALUDE_b_four_lt_b_seven_l637_63747


namespace NUMINAMATH_CALUDE_negative495_terminates_as_225_l637_63738

-- Define the set of possible answers
inductive PossibleAnswer
  | angle135  : PossibleAnswer
  | angle45   : PossibleAnswer
  | angle225  : PossibleAnswer
  | angleNeg225 : PossibleAnswer

-- Define a function to convert PossibleAnswer to real number (in degrees)
def toRealDegrees (a : PossibleAnswer) : ℝ :=
  match a with
  | PossibleAnswer.angle135   => 135
  | PossibleAnswer.angle45    => 45
  | PossibleAnswer.angle225   => 225
  | PossibleAnswer.angleNeg225 => -225

-- Define what it means for two angles to terminate in the same direction
def terminatesSameDirection (a b : ℝ) : Prop :=
  ∃ k : ℤ, a - b = 360 * (k : ℝ)

-- State the theorem
theorem negative495_terminates_as_225 :
  ∃ (answer : PossibleAnswer), terminatesSameDirection (-495) (toRealDegrees answer) ∧
  answer = PossibleAnswer.angle225 :=
sorry

end NUMINAMATH_CALUDE_negative495_terminates_as_225_l637_63738


namespace NUMINAMATH_CALUDE_rachel_chairs_l637_63715

/-- The number of chairs Rachel bought -/
def num_chairs : ℕ := 7

/-- The number of tables Rachel bought -/
def num_tables : ℕ := 3

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 4

/-- The total time spent (in minutes) -/
def total_time : ℕ := 40

theorem rachel_chairs :
  num_chairs = (total_time - num_tables * time_per_furniture) / time_per_furniture :=
by sorry

end NUMINAMATH_CALUDE_rachel_chairs_l637_63715


namespace NUMINAMATH_CALUDE_quadratic_equation_always_real_roots_l637_63725

theorem quadratic_equation_always_real_roots (m : ℝ) :
  ∃ x : ℝ, m * x^2 - (5*m - 1) * x + (4*m - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_always_real_roots_l637_63725


namespace NUMINAMATH_CALUDE_gcd_15_2015_l637_63730

theorem gcd_15_2015 : Nat.gcd 15 2015 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15_2015_l637_63730


namespace NUMINAMATH_CALUDE_only_striped_has_eight_legs_l637_63721

/-- Represents the color of an octopus -/
inductive OctopusColor
  | Green
  | DarkBlue
  | Purple
  | Striped

/-- Represents an octopus with its color and number of legs -/
structure Octopus where
  color : OctopusColor
  legs : ℕ

/-- Determines if an octopus tells the truth based on its number of legs -/
def tellsTruth (o : Octopus) : Prop :=
  o.legs % 2 = 0

/-- Represents the statements made by each octopus -/
def greenStatement (green darkBlue : Octopus) : Prop :=
  green.legs = 8 ∧ darkBlue.legs = 6

def darkBlueStatement (darkBlue green : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ green.legs = 7

def purpleStatement (darkBlue purple : Octopus) : Prop :=
  darkBlue.legs = 8 ∧ purple.legs = 9

def stripedStatement (green darkBlue purple striped : Octopus) : Prop :=
  green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 ∧ striped.legs = 8

/-- The main theorem stating that only the striped octopus has 8 legs -/
theorem only_striped_has_eight_legs
  (green darkBlue purple striped : Octopus)
  (h_green : green.color = OctopusColor.Green)
  (h_darkBlue : darkBlue.color = OctopusColor.DarkBlue)
  (h_purple : purple.color = OctopusColor.Purple)
  (h_striped : striped.color = OctopusColor.Striped)
  (h_greenStatement : tellsTruth green = greenStatement green darkBlue)
  (h_darkBlueStatement : tellsTruth darkBlue = darkBlueStatement darkBlue green)
  (h_purpleStatement : tellsTruth purple = purpleStatement darkBlue purple)
  (h_stripedStatement : tellsTruth striped = stripedStatement green darkBlue purple striped) :
  striped.legs = 8 ∧ green.legs ≠ 8 ∧ darkBlue.legs ≠ 8 ∧ purple.legs ≠ 8 :=
sorry

end NUMINAMATH_CALUDE_only_striped_has_eight_legs_l637_63721


namespace NUMINAMATH_CALUDE_work_equals_2pi_l637_63771

/-- The force field F --/
def F (x y : ℝ) : ℝ × ℝ := (x - y, 1)

/-- The curve L --/
def L : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4 ∧ p.2 ≥ 0}

/-- Starting point --/
def M : ℝ × ℝ := (2, 0)

/-- Ending point --/
def N : ℝ × ℝ := (-2, 0)

/-- Work done by force F along curve L from M to N --/
noncomputable def work : ℝ := sorry

theorem work_equals_2pi : work = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_work_equals_2pi_l637_63771


namespace NUMINAMATH_CALUDE_circle_symmetry_max_ab_l637_63761

/-- Given a circle x^2 + y^2 - 4ax + 2by + b^2 = 0 (where a > 0 and b > 0) 
    symmetric about the line x - y - 1 = 0, the maximum value of ab is 1/8 -/
theorem circle_symmetry_max_ab (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 - 4*a*x + 2*b*y + b^2 = 0 → 
    ∃ x' y' : ℝ, x' - y' - 1 = 0 ∧ x^2 + y^2 - 4*a*x + 2*b*y + b^2 = (x' - x)^2 + (y' - y)^2) →
  a * b ≤ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_max_ab_l637_63761


namespace NUMINAMATH_CALUDE_light_wash_water_usage_l637_63724

/-- Represents the water usage of a washing machine -/
structure WashingMachine where
  heavyWashWater : ℕ
  regularWashWater : ℕ
  lightWashWater : ℕ
  heavyWashCount : ℕ
  regularWashCount : ℕ
  lightWashCount : ℕ
  bleachedLoadsCount : ℕ
  totalWaterUsage : ℕ

/-- Theorem stating that the light wash water usage is 2 gallons -/
theorem light_wash_water_usage 
  (wm : WashingMachine) 
  (heavy_wash : wm.heavyWashWater = 20)
  (regular_wash : wm.regularWashWater = 10)
  (wash_counts : wm.heavyWashCount = 2 ∧ wm.regularWashCount = 3 ∧ wm.lightWashCount = 1)
  (bleached_loads : wm.bleachedLoadsCount = 2)
  (total_water : wm.totalWaterUsage = 76)
  (water_balance : wm.totalWaterUsage = 
    wm.heavyWashWater * wm.heavyWashCount + 
    wm.regularWashWater * wm.regularWashCount + 
    wm.lightWashWater * (wm.lightWashCount + wm.bleachedLoadsCount)) :
  wm.lightWashWater = 2 := by
  sorry

end NUMINAMATH_CALUDE_light_wash_water_usage_l637_63724


namespace NUMINAMATH_CALUDE_stephanie_store_visits_l637_63706

/-- Represents the number of oranges Stephanie buys per store visit -/
def oranges_per_visit : ℕ := 2

/-- Represents the total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- Represents the number of times Stephanie went to the store -/
def store_visits : ℕ := total_oranges / oranges_per_visit

theorem stephanie_store_visits : store_visits = 8 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_store_visits_l637_63706


namespace NUMINAMATH_CALUDE_shoveling_time_bounds_l637_63789

/-- Represents the snow shoveling scenario -/
structure SnowShoveling where
  initialRate : ℕ  -- Initial shoveling rate in cubic yards per hour
  rateDecrease : ℕ  -- Rate decrease per hour in cubic yards
  driveWidth : ℕ  -- Driveway width in yards
  driveLength : ℕ  -- Driveway length in yards
  snowDepth : ℕ  -- Snow depth in yards

/-- Calculates the time taken to shovel the driveway clean -/
def shovelingTime (s : SnowShoveling) : ℕ :=
  sorry

/-- Theorem stating that it takes at least 9 hours and less than 10 hours to clear the driveway -/
theorem shoveling_time_bounds (s : SnowShoveling) 
  (h1 : s.initialRate = 30)
  (h2 : s.rateDecrease = 2)
  (h3 : s.driveWidth = 4)
  (h4 : s.driveLength = 10)
  (h5 : s.snowDepth = 5) :
  9 ≤ shovelingTime s ∧ shovelingTime s < 10 :=
by
  sorry

end NUMINAMATH_CALUDE_shoveling_time_bounds_l637_63789


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l637_63728

/-- A function that is symmetric about the line x=1 and the point (2,0) -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧ 
  (∀ x, f (2 + x) = -f x)

theorem symmetric_function_properties (f : ℝ → ℝ) (h : SymmetricFunction f) :
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (4 - x) = -f x) ∧
  (∀ x, f (4 + x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l637_63728


namespace NUMINAMATH_CALUDE_students_history_not_statistics_l637_63735

/-- Given a group of students with the following properties:
  * There are 150 students in total
  * 58 students are taking history
  * 42 students are taking statistics
  * 95 students are taking history or statistics or both
  Then the number of students taking history but not statistics is 53. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 150 →
  history = 58 →
  statistics = 42 →
  history_or_statistics = 95 →
  history - (history + statistics - history_or_statistics) = 53 := by
sorry

end NUMINAMATH_CALUDE_students_history_not_statistics_l637_63735


namespace NUMINAMATH_CALUDE_ricks_savings_to_gift_ratio_l637_63795

def gift_cost : ℕ := 250
def cake_cost : ℕ := 25
def erikas_savings : ℕ := 155
def money_left : ℕ := 5

def total_savings : ℕ := gift_cost + cake_cost - money_left

def ricks_savings : ℕ := total_savings - erikas_savings

theorem ricks_savings_to_gift_ratio :
  (ricks_savings : ℚ) / gift_cost = 23 / 50 := by sorry

end NUMINAMATH_CALUDE_ricks_savings_to_gift_ratio_l637_63795


namespace NUMINAMATH_CALUDE_truck_fuel_distance_l637_63727

/-- Given a truck that travels 300 miles on 10 gallons of fuel,
    prove that it will travel 450 miles on 15 gallons of fuel,
    assuming a proportional relationship between fuel consumption and distance. -/
theorem truck_fuel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15)
    (h4 : initial_fuel > 0) :
  (new_fuel / initial_fuel) * initial_distance = 450 := by
  sorry

end NUMINAMATH_CALUDE_truck_fuel_distance_l637_63727


namespace NUMINAMATH_CALUDE_min_area_rectangle_l637_63781

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 → w > 0 → 2 * (l + w) = 150 → l * w ≥ 74 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l637_63781


namespace NUMINAMATH_CALUDE_specific_rectangle_perimeter_l637_63798

/-- Represents a rectangle with two internal segments --/
structure CutRectangle where
  AD : ℝ  -- Length of side AD
  AB : ℝ  -- Length of side AB
  EF : ℝ  -- Length of internal segment EF
  GH : ℝ  -- Length of internal segment GH

/-- Calculates the total perimeter of the two shapes formed by cutting the rectangle --/
def totalPerimeter (r : CutRectangle) : ℝ :=
  2 * (r.AD + r.AB + r.EF + r.GH)

/-- Theorem stating that for a specific rectangle, the total perimeter is 40 --/
theorem specific_rectangle_perimeter :
  ∃ (r : CutRectangle), r.AD = 10 ∧ r.AB = 6 ∧ r.EF = 2 ∧ r.GH = 2 ∧ totalPerimeter r = 40 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_perimeter_l637_63798


namespace NUMINAMATH_CALUDE_meaningful_condition_l637_63792

def is_meaningful (x : ℝ) : Prop :=
  x > -1 ∧ x ≠ 1

theorem meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x + 1) / (x - 1)) ↔ is_meaningful x :=
sorry

end NUMINAMATH_CALUDE_meaningful_condition_l637_63792


namespace NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l637_63765

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def abc_to_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def bca_to_number (a b c : ℕ) : ℕ :=
  100 * b + 10 * c + a

def cab_to_number (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    n = abc_to_number a b c ∧
    2 * n = bca_to_number a b c + cab_to_number a b c

theorem three_digit_numbers_satisfying_condition :
  {n : ℕ | is_three_digit_number n ∧ satisfies_condition n} = {481, 518, 592, 629} :=
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_satisfying_condition_l637_63765


namespace NUMINAMATH_CALUDE_equation_solution_l637_63734

theorem equation_solution (x : ℝ) (number : ℝ) :
  x = 32 →
  35 - (23 - (15 - x)) = 12 * 2 / (number / 2) →
  number = -2.4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l637_63734


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l637_63769

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / i
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l637_63769


namespace NUMINAMATH_CALUDE_unique_integer_term_l637_63716

def is_integer_term (n : ℕ) : Prop :=
  ∃ k : ℤ, (n^2 + 1).factorial / ((n.factorial)^(n + 2)) = k

theorem unique_integer_term :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ is_integer_term n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_term_l637_63716


namespace NUMINAMATH_CALUDE_book_price_calculation_l637_63752

theorem book_price_calculation (discounted_price original_price : ℝ) : 
  discounted_price = 8 →
  discounted_price = (1 / 8) * original_price →
  original_price = 64 := by
sorry

end NUMINAMATH_CALUDE_book_price_calculation_l637_63752


namespace NUMINAMATH_CALUDE_jim_distance_in_24_steps_l637_63736

-- Define the number of steps for Carly and Jim to cover the same distance
def carly_steps : ℕ := 3
def jim_steps : ℕ := 4

-- Define the length of Carly's step in meters
def carly_step_length : ℚ := 1/2

-- Define the number of Jim's steps we're interested in
def jim_target_steps : ℕ := 24

-- Theorem to prove
theorem jim_distance_in_24_steps :
  (jim_target_steps : ℚ) * (carly_steps * carly_step_length) / jim_steps = 9 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_in_24_steps_l637_63736
