import Mathlib

namespace extreme_value_implies_f_2_l3365_336581

/-- A function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then f(2) = 18 -/
theorem extreme_value_implies_f_2 (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧
  f a b 1 = 10 →
  f a b 2 = 18 :=
by sorry

end extreme_value_implies_f_2_l3365_336581


namespace min_value_of_function_l3365_336530

/-- The function f(x) = 4/(x-2) + x has a minimum value of 6 for x > 2 -/
theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
  sorry

end min_value_of_function_l3365_336530


namespace inverse_of_A_l3365_336500

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/14, 1/14; -1/7, 2/7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l3365_336500


namespace jasmine_bouquet_cost_l3365_336503

/-- The cost of a bouquet of jasmines after discount -/
def bouquet_cost (n : ℕ) (base_cost : ℚ) (base_num : ℕ) (discount : ℚ) : ℚ :=
  (base_cost * n / base_num) * (1 - discount)

/-- The theorem statement -/
theorem jasmine_bouquet_cost :
  bouquet_cost 50 24 8 (1/10) = 135 := by
  sorry

end jasmine_bouquet_cost_l3365_336503


namespace geometric_sequence_first_term_l3365_336563

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

/-- The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem geometric_sequence_first_term (a : ℕ → ℚ) :
  IsGeometricSequence a →
  a 7 = factorial 8 →
  a 10 = factorial 11 →
  a 1 = 8 / 245 := by
sorry

end geometric_sequence_first_term_l3365_336563


namespace largest_inscribed_pentagon_is_regular_l3365_336544

/-- A pentagon inscribed in a circle of radius 1 --/
structure InscribedPentagon where
  /-- The vertices of the pentagon --/
  vertices : Fin 5 → ℝ × ℝ
  /-- All vertices lie on the unit circle --/
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The area of an inscribed pentagon --/
def area (p : InscribedPentagon) : ℝ :=
  sorry

/-- A regular pentagon inscribed in a circle of radius 1 --/
def regular_pentagon : InscribedPentagon :=
  sorry

theorem largest_inscribed_pentagon_is_regular :
  ∀ p : InscribedPentagon, area p ≤ area regular_pentagon :=
  sorry

end largest_inscribed_pentagon_is_regular_l3365_336544


namespace parabola_distances_arithmetic_l3365_336511

/-- A parabola with focus F and three points A, B, C on it. -/
structure Parabola where
  p : ℝ
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h_p_pos : 0 < p
  h_on_parabola_1 : y₁^2 = 2 * p * x₁
  h_on_parabola_2 : y₂^2 = 2 * p * x₂
  h_on_parabola_3 : y₃^2 = 2 * p * x₃
  h_arithmetic : ∃ d : ℝ, 
    (x₂ : ℝ) - x₁ = d ∧ 
    (x₃ : ℝ) - x₂ = d

/-- If the distances from A, B, C to the focus form an arithmetic sequence,
    then x₁, x₂, x₃ form an arithmetic sequence. -/
theorem parabola_distances_arithmetic (par : Parabola) :
  ∃ d : ℝ, (par.x₂ - par.x₁ = d) ∧ (par.x₃ - par.x₂ = d) := by
  sorry

end parabola_distances_arithmetic_l3365_336511


namespace square_side_length_l3365_336599

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 9) (h2 : rectangle_width = 16) :
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 12 := by
  sorry

end square_side_length_l3365_336599


namespace sum_of_angles_two_triangles_l3365_336556

theorem sum_of_angles_two_triangles (A B C D E F : ℝ) :
  (A + B + C = 180) → (D + E + F = 180) → (A + B + C + D + E + F = 360) := by
sorry

end sum_of_angles_two_triangles_l3365_336556


namespace max_M_value_l3365_336557

/-- Definition of J_k -/
def J (k : ℕ) : ℕ := 10^(k+2) + 128

/-- Definition of M(k) -/
def M (k : ℕ) : ℕ := (J k).factors.count 2

/-- Theorem: The maximum value of M(k) for k > 0 is 8 -/
theorem max_M_value : ∃ k > 0, M k = 8 ∧ ∀ n > 0, M n ≤ 8 := by
  sorry

end max_M_value_l3365_336557


namespace salt_solution_weight_salt_solution_weight_proof_l3365_336555

theorem salt_solution_weight (initial_concentration : Real) 
                             (final_concentration : Real) 
                             (added_salt : Real) 
                             (initial_weight : Real) : Prop :=
  initial_concentration = 0.10 ∧
  final_concentration = 0.20 ∧
  added_salt = 12.5 ∧
  initial_weight * initial_concentration + added_salt = 
    (initial_weight + added_salt) * final_concentration →
  initial_weight = 100

-- Proof
theorem salt_solution_weight_proof :
  salt_solution_weight 0.10 0.20 12.5 100 := by
  sorry

end salt_solution_weight_salt_solution_weight_proof_l3365_336555


namespace contrapositive_equivalence_l3365_336570

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) := by
sorry

end contrapositive_equivalence_l3365_336570


namespace cone_lateral_surface_area_l3365_336536

/-- The lateral surface area of a cone with base radius 2 and slant height 4 is 8π -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 4 → π * r * l = 8 * π :=
sorry

end cone_lateral_surface_area_l3365_336536


namespace age_ratio_theorem_l3365_336519

/-- The number of years until the ratio of Mike's age to Sam's age is 3:2 -/
def years_until_ratio (m s : ℕ) : ℕ :=
  9

theorem age_ratio_theorem (m s : ℕ) 
  (h1 : m - 5 = 2 * (s - 5))
  (h2 : m - 12 = 3 * (s - 12)) :
  (m + years_until_ratio m s) / (s + years_until_ratio m s) = 3 / 2 := by
  sorry

end age_ratio_theorem_l3365_336519


namespace household_survey_l3365_336525

/-- Proves that the total number of households surveyed is 240 given the specified conditions -/
theorem household_survey (neither_brand : ℕ) (only_A : ℕ) (both_brands : ℕ)
  (h1 : neither_brand = 80)
  (h2 : only_A = 60)
  (h3 : both_brands = 25) :
  neither_brand + only_A + 3 * both_brands + both_brands = 240 := by
sorry

end household_survey_l3365_336525


namespace book_page_digits_l3365_336576

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) +
  2 * (min n 99 - min n 9) +
  3 * (n - min n 99)

/-- Theorem: The total number of digits used in numbering the pages of a book with 356 pages is 960 -/
theorem book_page_digits :
  totalDigits 356 = 960 := by
  sorry

end book_page_digits_l3365_336576


namespace V_upper_bound_l3365_336533

/-- V(n; b) is the number of decompositions of n into a product of one or more positive integers greater than b -/
def V (n b : ℕ+) : ℕ := sorry

/-- For all positive integers n and b, V(n; b) < n/b -/
theorem V_upper_bound (n b : ℕ+) : V n b < (n : ℚ) / b := by sorry

end V_upper_bound_l3365_336533


namespace painting_price_increase_l3365_336518

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 15 / 100) = 93.5 / 100 → x = 10 := by
  sorry

end painting_price_increase_l3365_336518


namespace first_digit_value_l3365_336501

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem first_digit_value (x y : ℕ) : 
  x < 10 → 
  y < 10 → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 2 → 
  x = 2 := by
  sorry

end first_digit_value_l3365_336501


namespace button_probability_l3365_336537

/-- Given a jar with red and blue buttons, prove the probability of selecting two red buttons after a specific removal process. -/
theorem button_probability (initial_red initial_blue : ℕ) 
  (h1 : initial_red = 6)
  (h2 : initial_blue = 10)
  (h3 : ∃ (removed : ℕ), 
    removed ≤ initial_red ∧ 
    removed ≤ initial_blue ∧ 
    initial_red + initial_blue - 2 * removed = (3 / 4) * (initial_red + initial_blue)) :
  let total_initial := initial_red + initial_blue
  let removed := (total_initial - (3 / 4) * total_initial) / 2
  let red_a := initial_red - removed
  let total_a := (3 / 4) * total_initial
  let prob_red_a := red_a / total_a
  let prob_red_b := removed / (2 * removed)
  prob_red_a * prob_red_b = 1 / 6 := by
  sorry


end button_probability_l3365_336537


namespace investment_problem_l3365_336558

theorem investment_problem (total investment bonds stocks mutual_funds : ℕ) : 
  total = 220000 ∧ 
  stocks = 5 * bonds ∧ 
  mutual_funds = 2 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  stocks = 68750 := by
sorry

end investment_problem_l3365_336558


namespace arithmetic_sequence_problem_l3365_336589

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 14) :
  (∀ n : ℕ, a n = 3 * n - 1) ∧
  (∃ n : ℕ, n * (a 1 + a n) / 2 = 155 ∧ n = 10) :=
sorry

end arithmetic_sequence_problem_l3365_336589


namespace divisibility_condition_l3365_336583

theorem divisibility_condition (a : ℤ) : 
  0 ≤ a ∧ a < 13 ∧ (13 ∣ 51^2022 + a) → a = 12 := by sorry

end divisibility_condition_l3365_336583


namespace oranges_from_second_tree_l3365_336532

theorem oranges_from_second_tree :
  ∀ (first_tree second_tree third_tree total : ℕ),
  first_tree = 80 →
  third_tree = 120 →
  total = 260 →
  total = first_tree + second_tree + third_tree →
  second_tree = 60 := by
sorry

end oranges_from_second_tree_l3365_336532


namespace min_value_problem_l3365_336517

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = -8 * Real.sqrt 2 ∧ ∀ z : ℝ, z = x * y - 4 * (x + y) - 2 → z ≥ m :=
by sorry

end min_value_problem_l3365_336517


namespace cricket_team_captain_age_l3365_336594

theorem cricket_team_captain_age (team_size : Nat) (captain_age wicket_keeper_age : Nat) 
  (remaining_players_avg_age team_avg_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 5 →
  remaining_players_avg_age = team_avg_age - 1 →
  team_avg_age = 24 →
  (team_size - 2 : ℚ) * remaining_players_avg_age + captain_age + wicket_keeper_age = 
    team_size * team_avg_age →
  captain_age = 26 := by
sorry

end cricket_team_captain_age_l3365_336594


namespace simplified_expression_l3365_336531

theorem simplified_expression : -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := by
  sorry

end simplified_expression_l3365_336531


namespace log_condition_l3365_336506

theorem log_condition (m : ℝ) (m_pos : m > 0) (m_neq_1 : m ≠ 1) :
  (∃ a b : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 → Real.log b / Real.log a > 0) ∧
  (∃ a b : ℝ, Real.log b / Real.log a > 0 ∧ ¬(0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1)) :=
by sorry

end log_condition_l3365_336506


namespace intersection_point_parametric_equation_l3365_336528

/-- Given a triangle ABC with points D and E such that:
    - D lies on BC extended past C with BD:DC = 2:1
    - E lies on AC with AE:EC = 2:1
    - P is the intersection of BE and AD
    This theorem proves that P can be expressed as (1/7)A + (2/7)B + (4/7)C -/
theorem intersection_point_parametric_equation 
  (A B C D E P : ℝ × ℝ) : 
  (∃ t : ℝ, D = (1 - t) • B + t • C ∧ t = 2/3) →
  (∃ s : ℝ, E = (1 - s) • A + s • C ∧ s = 2/3) →
  (∃ u v : ℝ, P = (1 - u) • A + u • D ∧ P = (1 - v) • B + v • E) →
  P = (1/7) • A + (2/7) • B + (4/7) • C :=
by sorry

end intersection_point_parametric_equation_l3365_336528


namespace science_club_enrollment_l3365_336510

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120) 
  (h2 : math = 80) 
  (h3 : physics = 50) 
  (h4 : both = 15) : 
  total - (math + physics - both) = 5 := by
  sorry

end science_club_enrollment_l3365_336510


namespace function_difference_implies_a_range_l3365_336588

open Real

theorem function_difference_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → 
    (a * log x₁ + x₁^2) - (a * log x₂ + x₂^2) > 2) →
  a ≥ 1 := by
sorry

end function_difference_implies_a_range_l3365_336588


namespace tower_lights_l3365_336590

/-- Represents the number of levels in the tower -/
def levels : ℕ := 7

/-- Represents the total number of lights on the tower -/
def totalLights : ℕ := 381

/-- Represents the common ratio between adjacent levels -/
def ratio : ℕ := 2

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem tower_lights :
  ∃ (topLights : ℕ), 
    geometricSum topLights ratio levels = totalLights ∧ 
    topLights = 3 := by
  sorry

end tower_lights_l3365_336590


namespace quadratic_root_difference_sum_l3365_336553

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 13 * x - 6 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference_sum (p q : ℕ) (hp : is_square_free p) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ 
    |x₁ - x₂| = (Real.sqrt (p : ℝ)) / (q : ℝ)) →
  p + q = 294 :=
sorry

end quadratic_root_difference_sum_l3365_336553


namespace theatre_audience_girls_fraction_l3365_336504

theorem theatre_audience_girls_fraction 
  (total : ℝ) 
  (adults : ℝ) 
  (children : ℝ) 
  (boys : ℝ) 
  (girls : ℝ) 
  (h1 : adults = (1 / 6) * total) 
  (h2 : children = total - adults) 
  (h3 : boys = (2 / 5) * children) 
  (h4 : girls = children - boys) : 
  girls = (1 / 2) * total := by
sorry

end theatre_audience_girls_fraction_l3365_336504


namespace repair_charge_is_30_l3365_336547

/-- Represents the services and pricing at Cid's mechanic shop --/
structure MechanicShop where
  oil_change_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Theorem stating that the repair charge is $30 --/
theorem repair_charge_is_30 (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.car_wash_price = 5)
  (h3 : shop.oil_changes = 5)
  (h4 : shop.repairs = 10)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  (shop.total_earnings - (shop.oil_changes * shop.oil_change_price + shop.car_washes * shop.car_wash_price)) / shop.repairs = 30 :=
by
  sorry

#check repair_charge_is_30

end repair_charge_is_30_l3365_336547


namespace sum_and_ratio_to_difference_l3365_336585

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 780) 
  (ratio_eq : x / y = 1.25) : 
  x - y = 86 + 2/3 := by
sorry

end sum_and_ratio_to_difference_l3365_336585


namespace inequality_proof_l3365_336569

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end inequality_proof_l3365_336569


namespace stating_count_sequences_l3365_336596

/-- 
Given positive integers n and k where 1 ≤ k < n, T(n, k) represents the number of 
sequences of k positive integers that sum to n.
-/
def T (n k : ℕ) : ℕ := sorry

/-- 
Theorem stating that T(n, k) is equal to (n-1) choose (k-1) for 1 ≤ k < n.
-/
theorem count_sequences (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  T n k = Nat.choose (n - 1) (k - 1) := by
  sorry

end stating_count_sequences_l3365_336596


namespace geometric_sequence_sixth_term_l3365_336546

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1024)  -- First term is 1024
  (h2 : a 8 = 125)   -- 8th term is 125
  (h3 : ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a n = a 1 * r^(n-1))  -- Definition of geometric sequence
  : a 6 = 5^(5/7) * 32 :=
by sorry

end geometric_sequence_sixth_term_l3365_336546


namespace fraction_evaluation_l3365_336568

theorem fraction_evaluation : (3^4 - 3^3) / (3^(-2) + 3^(-1)) = 121.5 := by sorry

end fraction_evaluation_l3365_336568


namespace length_A_l3365_336574

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

def on_line_AC (p : ℝ × ℝ) : Prop :=
  (p.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (p.1 - A.1)

def on_line_BC (p : ℝ × ℝ) : Prop :=
  (p.2 - B.2) * (C.1 - B.1) = (C.2 - B.2) * (p.1 - B.1)

def A' : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry

theorem length_A'B'_is_4_sqrt_2 :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ on_line_AC A' ∧ on_line_BC B' →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end length_A_l3365_336574


namespace only_expr4_is_equation_l3365_336512

-- Define the four expressions
def expr1 : ℝ → Prop := λ x ↦ 3 + x < 1
def expr2 : ℝ → ℝ := λ x ↦ x - 67 + 63
def expr3 : ℝ → ℝ := λ x ↦ 4.8 + x
def expr4 : ℝ → Prop := λ x ↦ x + 0.7 = 12

-- Theorem stating that only expr4 is an equation
theorem only_expr4_is_equation :
  (∃ (x : ℝ), expr4 x) ∧
  (¬∃ (x : ℝ), expr1 x = (3 + x < 1)) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr2 x = y) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr3 x = y) :=
sorry

end only_expr4_is_equation_l3365_336512


namespace f_properties_l3365_336507

open Real

noncomputable def f (x : ℝ) := x * log x

theorem f_properties :
  ∀ (m : ℝ), m > 0 →
  (∀ (x : ℝ), x > 0 →
    (∃ (min_value : ℝ),
      (∀ (y : ℝ), y ∈ Set.Icc m (m + 2) → f y ≥ min_value) ∧
      ((0 < m ∧ m < exp (-1)) → min_value = -(exp (-1))) ∧
      (m ≥ exp (-1) → min_value = f m))) ∧
  (∀ (x : ℝ), x > 0 → f x > x / (exp x) - 2 / exp 1) :=
by sorry

end f_properties_l3365_336507


namespace parallel_vectors_t_value_l3365_336513

def vector_a (t : ℝ) : ℝ × ℝ := (1, t)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 9)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel (vector_a t) (vector_b t) → t = 3 ∨ t = -3 := by
  sorry

end parallel_vectors_t_value_l3365_336513


namespace cubic_sum_minus_product_l3365_336540

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 40) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
  sorry

end cubic_sum_minus_product_l3365_336540


namespace i_in_first_quadrant_l3365_336566

/-- The complex number i corresponds to a point in the first quadrant of the complex plane. -/
theorem i_in_first_quadrant : Complex.I.re = 0 ∧ Complex.I.im > 0 := by
  sorry

end i_in_first_quadrant_l3365_336566


namespace sqrt_inequality_l3365_336580

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end sqrt_inequality_l3365_336580


namespace largest_common_term_proof_l3365_336524

/-- The first arithmetic progression with common difference 5 -/
def ap1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression with common difference 11 -/
def ap2 (n : ℕ) : ℕ := 7 + 11 * n

/-- A common term of both arithmetic progressions -/
def common_term (k m : ℕ) : Prop := ap1 k = ap2 m

/-- The largest common term less than 1000 -/
def largest_common_term : ℕ := 964

theorem largest_common_term_proof :
  (∃ k m : ℕ, common_term k m ∧ largest_common_term = ap1 k) ∧
  (∀ n : ℕ, n < 1000 → (∃ k m : ℕ, common_term k m ∧ n = ap1 k) → n ≤ largest_common_term) :=
sorry

end largest_common_term_proof_l3365_336524


namespace arithmetic_sequence_15th_term_l3365_336582

/-- Given an arithmetic sequence where the first three terms are 3, 16, and 29,
    prove that the 15th term is 185. -/
theorem arithmetic_sequence_15th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                            -- first term
    a 2 = 16 →                           -- second term
    a 3 = 29 →                           -- third term
    a 15 = 185 := by
  sorry

end arithmetic_sequence_15th_term_l3365_336582


namespace product_one_cube_sum_inequality_l3365_336505

theorem product_one_cube_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1/a + 1/b + 1/c + 1/d) := by
  sorry

end product_one_cube_sum_inequality_l3365_336505


namespace solve_for_m_l3365_336554

theorem solve_for_m (x m : ℝ) : 2 * x + m = 1 → x = -1 → m = 3 := by
  sorry

end solve_for_m_l3365_336554


namespace min_fruits_in_platter_l3365_336526

/-- Represents the types of fruits --/
inductive Fruit
  | GreenApple
  | RedApple
  | YellowApple
  | RedOrange
  | YellowOrange
  | GreenKiwi
  | PurpleGrape
  | GreenGrape

/-- Represents the fruit platter --/
structure FruitPlatter :=
  (greenApples : ℕ)
  (redApples : ℕ)
  (yellowApples : ℕ)
  (redOranges : ℕ)
  (yellowOranges : ℕ)
  (greenKiwis : ℕ)
  (purpleGrapes : ℕ)
  (greenGrapes : ℕ)

/-- Checks if the platter satisfies all constraints --/
def isValidPlatter (p : FruitPlatter) : Prop :=
  p.greenApples + p.redApples + p.yellowApples ≥ 5 ∧
  p.redOranges + p.yellowOranges ≤ 5 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≥ 8 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≤ 12 ∧
  p.greenGrapes ≥ 1 ∧
  p.purpleGrapes ≥ 1 ∧
  p.greenApples * 2 = p.redApples ∧
  p.greenApples * 3 = p.yellowApples * 2 ∧
  p.redOranges = 1 ∧
  p.yellowOranges = 2 ∧
  p.greenKiwis = p.purpleGrapes

/-- Calculates the total number of fruits in the platter --/
def totalFruits (p : FruitPlatter) : ℕ :=
  p.greenApples + p.redApples + p.yellowApples +
  p.redOranges + p.yellowOranges +
  p.greenKiwis + p.purpleGrapes + p.greenGrapes

/-- Theorem stating that the minimum number of fruits in a valid platter is 30 --/
theorem min_fruits_in_platter :
  ∀ p : FruitPlatter, isValidPlatter p → totalFruits p ≥ 30 :=
sorry

end min_fruits_in_platter_l3365_336526


namespace max_value_problem_l3365_336561

theorem max_value_problem (x y : ℝ) (h : y^2 + x - 2 = 0) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (x' y' : ℝ), y'^2 + x' - 2 = 0 → y'^2 - x'^2 + x' + 5 ≤ M :=
by sorry

end max_value_problem_l3365_336561


namespace equation_solution_l3365_336534

-- Define the equation
def equation (y : ℝ) : Prop :=
  (15 : ℝ)^(3*2) * (7^4 - 3*2) / 5670 = y

-- State the theorem
theorem equation_solution : 
  ∃ y : ℝ, equation y ∧ abs (y - 4812498.20123) < 0.00001 := by
  sorry

end equation_solution_l3365_336534


namespace four_digit_number_problem_l3365_336591

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_problem (N : ℕ) (hN : is_four_digit N) :
  let M := reverse_number N
  (N + M = 3333 ∧ N - M = 693) → N = 2013 := by
  sorry

end four_digit_number_problem_l3365_336591


namespace circle_passes_through_points_l3365_336579

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (4, 0)
def point_C : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 := by
  sorry

end circle_passes_through_points_l3365_336579


namespace meatball_fraction_eaten_l3365_336587

/-- Given 3 plates with 3 meatballs each, if 3 people eat the same fraction of meatballs from their respective plates and 3 meatballs are left in total, then each person ate 2/3 of the meatballs on their plate. -/
theorem meatball_fraction_eaten (f : ℚ) 
  (h1 : f ≥ 0) 
  (h2 : f ≤ 1) 
  (h3 : 3 * (3 - 3 * f) = 3) : 
  f = 2 / 3 := by
sorry

end meatball_fraction_eaten_l3365_336587


namespace union_of_sets_l3365_336542

theorem union_of_sets (a : ℤ) : 
  let A : Set ℤ := {|a + 1|, 3, 5}
  let B : Set ℤ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {-5, 2, 3, 5}) :=
by
  sorry

end union_of_sets_l3365_336542


namespace shirts_sold_l3365_336523

def commission_rate : ℚ := 15 / 100
def suit_price : ℚ := 700
def suit_quantity : ℕ := 2
def shirt_price : ℚ := 50
def loafer_price : ℚ := 150
def loafer_quantity : ℕ := 2
def total_commission : ℚ := 300

theorem shirts_sold (shirt_quantity : ℕ) : 
  commission_rate * (suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity) = total_commission →
  shirt_quantity = 6 := by sorry

end shirts_sold_l3365_336523


namespace concatenated_number_500_not_divisible_by_9_l3365_336538

def concatenated_number (n : ℕ) : ℕ := sorry

theorem concatenated_number_500_not_divisible_by_9 :
  ¬ (9 ∣ concatenated_number 500) := by sorry

end concatenated_number_500_not_divisible_by_9_l3365_336538


namespace cylinder_volume_relation_l3365_336502

/-- Given two cylinders A and B, where A's radius is r and height is h,
    B's height is r and radius is h, and A's volume is twice B's volume,
    prove that A's volume can be expressed as 4π h^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : h > 0) :
  let volume_A := π * r^2 * h
  let volume_B := π * h^2 * r
  volume_A = 2 * volume_B → r = 2 * h → volume_A = 4 * π * h^3 := by
  sorry

end cylinder_volume_relation_l3365_336502


namespace geometric_series_sum_l3365_336520

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 4) : ∑' n, a / (a + b)^n = 4/5 := by
  sorry

end geometric_series_sum_l3365_336520


namespace school_students_count_l3365_336559

theorem school_students_count : ℕ :=
  let below_eight_percent : ℚ := 20 / 100
  let eight_years_count : ℕ := 12
  let above_eight_ratio : ℚ := 2 / 3
  let total_students : ℕ := 40

  have h1 : ↑eight_years_count + (↑eight_years_count * above_eight_ratio) = (1 - below_eight_percent) * total_students := by sorry

  total_students


end school_students_count_l3365_336559


namespace no_integer_solution_l3365_336522

theorem no_integer_solution (n : ℕ) (hn : n ≥ 11) :
  ¬ ∃ m : ℤ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) := by
sorry

end no_integer_solution_l3365_336522


namespace acute_triangle_probability_condition_l3365_336564

/-- The probability of forming an acute triangle from three random vertices of a regular n-gon --/
def acuteTriangleProbability (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (3 * (n / 2 - 2)) / (2 * (n - 1))
  else (3 * ((n - 1) / 2 - 1)) / (2 * (n - 1))

/-- Theorem stating that the probability of forming an acute triangle is 93/125 
    if and only if n is 376 or 127 --/
theorem acute_triangle_probability_condition (n : ℕ) :
  acuteTriangleProbability n = 93 / 125 ↔ n = 376 ∨ n = 127 := by
  sorry

end acute_triangle_probability_condition_l3365_336564


namespace unique_abc_sum_l3365_336572

theorem unique_abc_sum (x : ℝ) : 
  x = Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2) →
  ∃! (a b c : ℕ+), 
    x^80 = 2*x^78 + 8*x^76 + 9*x^74 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 ∧
    a + b + c = 151 := by
  sorry

end unique_abc_sum_l3365_336572


namespace min_value_f_when_a_is_1_range_of_a_when_solution_exists_l3365_336549

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f_when_a_is_1 :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Theorem 2: Range of a when solution set of f(x) ≤ 3 is non-empty
theorem range_of_a_when_solution_exists :
  (∃ x : ℝ, f a x ≤ 3) → |3 - a| ≤ 3 :=
sorry

end min_value_f_when_a_is_1_range_of_a_when_solution_exists_l3365_336549


namespace inequality_system_integer_solutions_l3365_336551

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 ≥ (x - 2 : ℚ) / 3 ∧ (2 * x - 5 : ℤ) < -3 * x}
  S = {-1, 0} := by sorry

end inequality_system_integer_solutions_l3365_336551


namespace runners_meeting_time_l3365_336592

/-- 
Given two runners, Danny and Steve, running towards each other from their respective houses:
* Danny's time to reach Steve's house is t minutes
* Steve's time to reach Danny's house is 2t minutes
* Steve takes 13.5 minutes longer to reach the halfway point than Danny
Prove that t = 27 minutes
-/
theorem runners_meeting_time (t : ℝ) 
  (h1 : t > 0) -- Danny's time is positive
  (h2 : 2 * t - t / 2 = 13.5) -- Difference in time to reach halfway point
  : t = 27 := by
  sorry

end runners_meeting_time_l3365_336592


namespace P_equals_Q_l3365_336545

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m*x^2 + 4*m*x - 4 < 0}

theorem P_equals_Q : P = Q := by sorry

end P_equals_Q_l3365_336545


namespace math_team_selection_count_l3365_336562

theorem math_team_selection_count :
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let boys_needed : ℕ := 2
  let girls_needed : ℕ := 3
  (Nat.choose total_boys boys_needed) * (Nat.choose total_girls girls_needed) = 2520 :=
by sorry

end math_team_selection_count_l3365_336562


namespace grasshopper_jumps_l3365_336552

/-- Given a grasshopper's initial position and first jump endpoint, calculate its final position after a second identical jump -/
theorem grasshopper_jumps (initial_pos : ℝ) (first_jump_end : ℝ) : 
  initial_pos = 8 → first_jump_end = 17.5 → 
  let jump_length := first_jump_end - initial_pos
  first_jump_end + jump_length = 27 := by
sorry

end grasshopper_jumps_l3365_336552


namespace lucky_lila_problem_l3365_336521

theorem lucky_lila_problem (a b c d e : ℤ) : 
  a = 5 → b = 3 → c = 2 → d = 6 →
  (a - b + c * d - e = a - (b + (c * (d - e)))) →
  e = 8 := by
  sorry

end lucky_lila_problem_l3365_336521


namespace q_polynomial_form_l3365_336577

/-- The function q satisfying the given equation -/
noncomputable def q : ℝ → ℝ := fun x => 4*x^4 + 16*x^3 + 36*x^2 + 10*x + 4 - (2*x^6 + 5*x^4 + 11*x^2 + 6*x)

/-- Theorem stating that q has the specified polynomial form -/
theorem q_polynomial_form : q = fun x => -2*x^6 - x^4 + 16*x^3 + 25*x^2 + 4*x + 4 := by sorry

end q_polynomial_form_l3365_336577


namespace smallest_number_properties_l3365_336516

/-- The number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The smallest natural number divisible by 35 with exactly 75 divisors -/
def smallestNumber : ℕ := 490000

theorem smallest_number_properties :
  (35 ∣ smallestNumber) ∧
  (numDivisors smallestNumber = 75) ∧
  ∀ n : ℕ, n < smallestNumber → ¬((35 ∣ n) ∧ (numDivisors n = 75)) := by
  sorry

end smallest_number_properties_l3365_336516


namespace inscribed_square_area_l3365_336508

/-- The area of a square inscribed in a quadrant of a circle with radius 10 is equal to 40 -/
theorem inscribed_square_area (r : ℝ) (s : ℝ) (h1 : r = 10) (h2 : s > 0) 
  (h3 : s^2 + (3/2 * s)^2 = r^2) : s^2 = 40 := by
  sorry

end inscribed_square_area_l3365_336508


namespace specific_trapezoid_area_l3365_336575

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- Calculates the area of a trapezoid given its side lengths -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  -- We don't implement the actual calculation here
  sorry

/-- Theorem: The area of the specific trapezoid is 450 -/
theorem specific_trapezoid_area :
  trapezoidArea { a := 16, b := 44, c := 17, d := 25 } = 450 := by
  sorry

end specific_trapezoid_area_l3365_336575


namespace zoo_trip_vans_l3365_336571

def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

theorem zoo_trip_vans : vans_needed 4 2 6 = 2 := by
  sorry

end zoo_trip_vans_l3365_336571


namespace expression_simplification_and_evaluation_l3365_336578

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ -2) (h2 : x ≠ 0) (h3 : x ≠ 2) :
  (x^2 / (x - 2) + 4 / (2 - x)) / ((x^2 + 4*x + 4) / x) = x / (x + 2) ∧
  (1 : ℝ) / (1 + 2) = (1 : ℝ) / 3 := by
sorry

end expression_simplification_and_evaluation_l3365_336578


namespace work_payment_proof_l3365_336543

/-- Calculates the total payment for a bricklayer and an electrician's work -/
def total_payment (total_hours : ℝ) (bricklayer_hours : ℝ) (bricklayer_rate : ℝ) (electrician_rate : ℝ) : ℝ :=
  let electrician_hours := total_hours - bricklayer_hours
  bricklayer_hours * bricklayer_rate + electrician_hours * electrician_rate

/-- Proves that the total payment for the given work scenario is $1170 -/
theorem work_payment_proof :
  total_payment 90 67.5 12 16 = 1170 := by
  sorry

end work_payment_proof_l3365_336543


namespace sarah_copies_360_pages_l3365_336565

/-- The number of copies per person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in each contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end sarah_copies_360_pages_l3365_336565


namespace opposite_direction_speed_l3365_336586

/-- 
Given two people traveling in opposite directions for 1.5 hours, 
with one person traveling at 5 miles per hour and ending up 19.5 miles apart, 
prove that the other person's speed must be 8 miles per hour.
-/
theorem opposite_direction_speed 
  (time : ℝ) 
  (distance : ℝ) 
  (speed_peter : ℝ) 
  (speed_juan : ℝ) : 
  time = 1.5 → 
  distance = 19.5 → 
  speed_peter = 5 → 
  distance = (speed_juan + speed_peter) * time → 
  speed_juan = 8 := by
  sorry

end opposite_direction_speed_l3365_336586


namespace max_integer_value_of_fraction_l3365_336527

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 20) / (4 * x^2 + 12 * x + 8) < 12002 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 20) / (4 * y^2 + 12 * y + 8) > 12001 - ε :=
by sorry

#check max_integer_value_of_fraction

end max_integer_value_of_fraction_l3365_336527


namespace intersection_radius_l3365_336560

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in the xz-plane -/
  xz_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in the xz-plane -/
  xz_radius : ℝ
  /-- Center of the circle in the zy-plane -/
  zy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in the zy-plane -/
  zy_radius : ℝ

/-- The radius of the circle where the sphere intersects the zy-plane is 3 -/
theorem intersection_radius (s : IntersectingSphere) 
  (h1 : s.xz_center = (3, 0, 5))
  (h2 : s.xz_radius = 2)
  (h3 : s.zy_center = (0, 3, -4)) :
  s.zy_radius = 3 := by
  sorry

end intersection_radius_l3365_336560


namespace davids_chemistry_marks_l3365_336541

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 85)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 85)
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 87 := by
  sorry

end davids_chemistry_marks_l3365_336541


namespace max_hardcover_books_l3365_336529

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The problem statement -/
theorem max_hardcover_books :
  ∀ (hardcover paperback : ℕ),
  hardcover + paperback = 36 →
  IsComposite (paperback - hardcover) →
  hardcover ≤ 16 :=
by sorry

end max_hardcover_books_l3365_336529


namespace green_packs_count_l3365_336515

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The total number of bouncy balls bought -/
def total_balls : ℕ := 160

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

theorem green_packs_count : green_packs = 4 := by
  sorry

end green_packs_count_l3365_336515


namespace diamond_eight_five_l3365_336595

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem diamond_eight_five : diamond 8 5 = 160 := by sorry

end diamond_eight_five_l3365_336595


namespace equal_cost_messages_l3365_336593

/-- Represents the cost of a text messaging plan -/
structure TextPlan where
  costPerMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.costPerMessage * messages + plan.monthlyFee

theorem equal_cost_messages : 
  let planA : TextPlan := ⟨0.25, 9⟩
  let planB : TextPlan := ⟨0.40, 0⟩
  ∃ (x : ℚ), x = 60 ∧ totalCost planA x = totalCost planB x :=
by sorry

end equal_cost_messages_l3365_336593


namespace basic_algorithm_statements_correct_l3365_336548

/-- Represents the types of algorithm statements -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem: The set of basic algorithm statements is exactly 
    {INPUT, PRINT, IF-THEN, DO, WHILE} -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = 
    {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
     AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end basic_algorithm_statements_correct_l3365_336548


namespace leah_savings_days_l3365_336514

/-- Proves that Leah saved for 20 days given the conditions of the problem -/
theorem leah_savings_days : ℕ :=
  let josiah_daily_savings : ℚ := 25 / 100
  let josiah_days : ℕ := 24
  let leah_daily_savings : ℚ := 1 / 2
  let megan_days : ℕ := 12
  let total_savings : ℚ := 28
  let leah_days : ℕ := 20

  have josiah_total : ℚ := josiah_daily_savings * josiah_days
  have megan_total : ℚ := 2 * leah_daily_savings * megan_days
  have leah_total : ℚ := leah_daily_savings * leah_days

  have savings_equation : josiah_total + leah_total + megan_total = total_savings := by sorry

  leah_days


end leah_savings_days_l3365_336514


namespace distribution_plans_count_l3365_336567

/-- The number of ways to distribute 3 volunteer teachers among 6 schools, with at most 2 teachers per school -/
def distribution_plans : ℕ := 210

/-- The number of schools -/
def num_schools : ℕ := 6

/-- The number of volunteer teachers -/
def num_teachers : ℕ := 3

/-- The maximum number of teachers allowed per school -/
def max_teachers_per_school : ℕ := 2

theorem distribution_plans_count :
  distribution_plans = 210 :=
sorry

end distribution_plans_count_l3365_336567


namespace sufficient_not_necessary_condition_l3365_336550

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l3365_336550


namespace sequence_recurrence_problem_l3365_336597

/-- Given a sequence of positive real numbers {a_n} (n ≥ 0) satisfying the recurrence relation
    a_n = a_{n-1} / (m * a_{n-2}) for n ≥ 2, where m is a real parameter,
    prove that if a_2009 = a_0 / a_1, then m = 1. -/
theorem sequence_recurrence_problem (a : ℕ → ℝ) (m : ℝ) 
    (h_positive : ∀ n, a n > 0)
    (h_recurrence : ∀ n ≥ 2, a n = a (n-1) / (m * a (n-2)))
    (h_equality : a 2009 = a 0 / a 1) :
  m = 1 := by
  sorry

end sequence_recurrence_problem_l3365_336597


namespace inequality_solution_set_l3365_336539

theorem inequality_solution_set (x y : ℝ) : 
  (∀ y > 0, (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y)) / (x + y) > 3 * x^2 * y) ↔ 
  x > 0 := by
sorry

end inequality_solution_set_l3365_336539


namespace sum_of_coordinates_reflection_l3365_336535

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all four coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end sum_of_coordinates_reflection_l3365_336535


namespace line_symmetry_l3365_336509

-- Define the lines
def l1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def l2 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Define the symmetry relation
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2)/2) ((y1 + y2)/2) ∧ x1 + y2 = 0 ∧ y1 + x2 = 0

-- Theorem statement
theorem line_symmetry :
  (∀ x y : ℝ, l1 x y ↔ l2 (-y) (-x)) →
  (∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 ∧ l2 x2 y2 → symmetric_points x1 y1 x2 y2) →
  ∀ x y : ℝ, l2 x y ↔ 2*x - y - 3 = 0 :=
sorry

end line_symmetry_l3365_336509


namespace initial_bedbug_count_l3365_336573

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem stating the initial number of bedbugs -/
theorem initial_bedbug_count : ∃ (initial : ℕ), 
  bedbug_population initial 4 = 810 ∧ initial = 10 := by
  sorry

end initial_bedbug_count_l3365_336573


namespace seven_dots_max_regions_l3365_336584

/-- The maximum number of regions formed by connecting n dots on a circle's circumference --/
def max_regions (n : ℕ) : ℕ :=
  1 + (n.choose 2) + (n.choose 4)

/-- Theorem: For 7 dots on a circle's circumference, the maximum number of regions is 57 --/
theorem seven_dots_max_regions :
  max_regions 7 = 57 := by
  sorry

end seven_dots_max_regions_l3365_336584


namespace win_sector_area_l3365_336598

/-- Given a circular spinner with radius 8 cm and probability of winning 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 8 →
  win_prob = 3 / 8 →
  win_area = win_prob * π * radius^2 →
  win_area = 24 * π := by
  sorry

#check win_sector_area

end win_sector_area_l3365_336598
