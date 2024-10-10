import Mathlib

namespace product_and_remainder_l3163_316322

theorem product_and_remainder (a b c d : ℤ) : 
  d = a * b * c → 
  1 < a → a < b → b < c → 
  233 % d = 79 → 
  a + c = 13 := by
sorry

end product_and_remainder_l3163_316322


namespace base_b_perfect_square_implies_b_greater_than_two_l3163_316345

/-- Represents a number in base b --/
def base_representation (b : ℕ) : ℕ := b^2 + 2*b + 1

/-- Checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem base_b_perfect_square_implies_b_greater_than_two :
  ∀ b : ℕ, is_perfect_square (base_representation b) → b > 2 :=
by sorry

end base_b_perfect_square_implies_b_greater_than_two_l3163_316345


namespace johns_weight_change_l3163_316397

theorem johns_weight_change (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : 
  initial_weight = 220 →
  loss_percentage = 10 →
  weight_gain = 2 →
  initial_weight * (1 - loss_percentage / 100) + weight_gain = 200 := by
  sorry

end johns_weight_change_l3163_316397


namespace smallest_m_for_divisibility_l3163_316393

theorem smallest_m_for_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 436 * 32^n) % 2001 = 0 ∧ 
    ∀ (m : ℕ), m < 436 → 
      ∀ (k : ℕ), k % 2 = 1 → (55^k + m * 32^k) % 2001 ≠ 0 :=
by sorry

end smallest_m_for_divisibility_l3163_316393


namespace card_distribution_events_l3163_316363

-- Define the set of colors
inductive Color
| Red
| Yellow
| Blue
| White

-- Define the set of people
inductive Person
| A
| B
| C
| D

-- Define the distribution of cards
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "D receives the red card"
def D_red (d : Distribution) : Prop := d Person.D = Color.Red

-- State the theorem
theorem card_distribution_events :
  -- Each person receives one card
  (∀ p : Person, ∃! c : Color, ∀ d : Distribution, d p = c) →
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(A_red d ∧ D_red d)) ∧
  -- The events are not complementary
  ¬(∀ d : Distribution, A_red d ↔ ¬(D_red d)) :=
by sorry

end card_distribution_events_l3163_316363


namespace hyperbola_asymptotes_sufficient_not_necessary_l3163_316360

/-- A hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 --/
structure Hyperbola (a b : ℝ) : Type :=
  (hap : a > 0)
  (hbp : b > 0)

/-- The asymptotes of a hyperbola --/
def asymptotes (h : Hyperbola a b) (x : ℝ) : Set ℝ :=
  {y | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The theorem stating that the hyperbola equation is a sufficient but not necessary condition for its asymptotes --/
theorem hyperbola_asymptotes_sufficient_not_necessary (a b : ℝ) :
  (∃ (h : Hyperbola a b), ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → y ∈ asymptotes h x) ∧
  (∃ a' b' : ℝ, ∃ (h : Hyperbola a' b'), ∀ x y : ℝ, y ∈ asymptotes h x ∧ (x^2 / a'^2) - (y^2 / b'^2) ≠ 1) :=
sorry

end hyperbola_asymptotes_sufficient_not_necessary_l3163_316360


namespace salary_increase_after_five_years_l3163_316338

theorem salary_increase_after_five_years (annual_raise : ℝ) (num_years : ℕ) : 
  annual_raise = 0.12 → num_years = 5 → (1 + annual_raise) ^ num_years > 1.76 := by
  sorry

end salary_increase_after_five_years_l3163_316338


namespace kangaroo_arrangement_count_l3163_316385

/-- The number of kangaroos -/
def n : ℕ := 8

/-- The number of ways to arrange the tallest and shortest kangaroos at the ends -/
def end_arrangements : ℕ := 2

/-- The number of remaining kangaroos to be arranged -/
def remaining_kangaroos : ℕ := n - 2

/-- The total number of ways to arrange the kangaroos -/
def total_arrangements : ℕ := end_arrangements * (Nat.factorial remaining_kangaroos)

theorem kangaroo_arrangement_count :
  total_arrangements = 1440 := by
  sorry

end kangaroo_arrangement_count_l3163_316385


namespace ann_age_l3163_316346

/-- Ann's age in years -/
def A : ℕ := sorry

/-- Susan's age in years -/
def S : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : A = S + 5

/-- The sum of their ages is 27 -/
axiom age_sum : A + S = 27

/-- Prove that Ann is 16 years old -/
theorem ann_age : A = 16 := by sorry

end ann_age_l3163_316346


namespace company_picnic_attendance_l3163_316349

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 35

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 100 - percentMen

/-- Represents the percentage of all employees who attended the picnic -/
def percentAttended : ℝ := 33

/-- Represents the percentage of women who attended the picnic -/
def percentWomenAttended : ℝ := 40

/-- Represents the percentage of men who attended the picnic -/
def percentMenAttended : ℝ := 20

theorem company_picnic_attendance :
  percentMenAttended * (percentMen / 100) + percentWomenAttended * (percentWomen / 100) = percentAttended / 100 :=
by sorry

end company_picnic_attendance_l3163_316349


namespace C₁_intersects_C₂_max_value_on_C₂_l3163_316309

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x + y - 1 = 0
def C₂ (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y + Real.sqrt 2 / 2)^2 = 1

-- Define a point M on C₂
structure PointOnC₂ where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y

-- Theorem 1: C₁ intersects C₂
theorem C₁_intersects_C₂ : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y :=
sorry

-- Theorem 2: Maximum value of 2x + y for points on C₂
theorem max_value_on_C₂ :
  ∃ (max : ℝ), max = Real.sqrt 2 / 2 + Real.sqrt 5 ∧
  ∀ (M : PointOnC₂), 2 * M.x + M.y ≤ max :=
sorry

end C₁_intersects_C₂_max_value_on_C₂_l3163_316309


namespace exam_items_count_l3163_316310

theorem exam_items_count :
  ∀ (X : ℕ) (E M : ℕ),
    M = 24 →
    M = E / 2 + 6 →
    X = E + 4 →
    X = 40 := by
  sorry

end exam_items_count_l3163_316310


namespace coffee_expense_l3163_316387

theorem coffee_expense (items_per_day : ℕ) (cost_per_item : ℕ) (days : ℕ) :
  items_per_day = 2 →
  cost_per_item = 2 →
  days = 30 →
  items_per_day * cost_per_item * days = 120 :=
by
  sorry

end coffee_expense_l3163_316387


namespace typing_orders_count_l3163_316383

/-- The number of letters to be typed during the day -/
def total_letters : ℕ := 12

/-- The set of letter numbers that have already been typed -/
def typed_letters : Finset ℕ := {10, 12}

/-- The set of letter numbers that could potentially be in the in-box -/
def potential_inbox : Finset ℕ := Finset.range 10 ∪ {11}

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_typing_orders : ℕ :=
  (Finset.powerset potential_inbox).sum (fun s => s.card + 1)

/-- The main theorem stating the number of possible typing orders -/
theorem typing_orders_count : possible_typing_orders = 6144 := by
  sorry

end typing_orders_count_l3163_316383


namespace square_area_with_diagonal_l3163_316336

/-- The area of a square with sides of length 12 meters is 144 square meters, 
    given that the diagonal of the square satisfies the Pythagorean theorem. -/
theorem square_area_with_diagonal (x : ℝ) : 
  (x^2 = 2 * 12^2) →  -- Pythagorean theorem for the diagonal
  (12 * 12 : ℝ) = 144 := by
  sorry

end square_area_with_diagonal_l3163_316336


namespace intersection_complement_equality_l3163_316321

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the result set
def result : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem intersection_complement_equality : P ∩ (Set.univ \ Q) = result := by
  sorry

end intersection_complement_equality_l3163_316321


namespace survey_total_is_260_l3163_316350

/-- Represents the survey results of households using different brands of soap -/
structure SoapSurvey where
  neither : Nat
  onlyA : Nat
  onlyB : Nat
  both : Nat

/-- Calculates the total number of households surveyed -/
def totalHouseholds (survey : SoapSurvey) : Nat :=
  survey.neither + survey.onlyA + survey.onlyB + survey.both

/-- Theorem stating the total number of households surveyed is 260 -/
theorem survey_total_is_260 : ∃ (survey : SoapSurvey),
  survey.neither = 80 ∧
  survey.onlyA = 60 ∧
  survey.onlyB = 3 * survey.both ∧
  survey.both = 30 ∧
  totalHouseholds survey = 260 := by
  sorry

end survey_total_is_260_l3163_316350


namespace sum_properties_l3163_316341

theorem sum_properties (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(Even (x + y))) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 6 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 9 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ (∃ k : ℤ, x + y = 9 * k)) :=
by sorry

end sum_properties_l3163_316341


namespace initial_puppies_count_l3163_316365

/-- The number of puppies Sandy's dog had initially -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has now -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 :=
by
  sorry

#check initial_puppies_count

end initial_puppies_count_l3163_316365


namespace line_perp_para_implies_plane_perp_l3163_316369

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (planePara : Plane → Plane → Prop)
variable (planePerpDir : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_para_implies_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp m α)
  (h2 : para m β) :
  planePerpDir α β :=
sorry

end line_perp_para_implies_plane_perp_l3163_316369


namespace paper_used_calculation_l3163_316359

-- Define the variables
def total_paper : ℕ := 900
def remaining_paper : ℕ := 744

-- Define the theorem
theorem paper_used_calculation : total_paper - remaining_paper = 156 := by
  sorry

end paper_used_calculation_l3163_316359


namespace exactly_one_correct_probability_l3163_316382

theorem exactly_one_correct_probability 
  (prob_a : ℝ) 
  (prob_b : ℝ) 
  (h_prob_a : prob_a = 0.7) 
  (h_prob_b : prob_b = 0.8) : 
  prob_a * (1 - prob_b) + (1 - prob_a) * prob_b = 0.38 := by
  sorry

end exactly_one_correct_probability_l3163_316382


namespace other_root_of_quadratic_l3163_316333

theorem other_root_of_quadratic (p : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + p * x = 9) ∧ 
  (7 * (-3)^2 + p * (-3) = 9) → 
  7 * (3/7)^2 + p * (3/7) = 9 :=
sorry

end other_root_of_quadratic_l3163_316333


namespace storage_wheels_count_l3163_316352

def total_wheels (bicycles tricycles unicycles four_wheelers : ℕ) : ℕ :=
  bicycles * 2 + tricycles * 3 + unicycles * 1 + four_wheelers * 4

theorem storage_wheels_count : total_wheels 16 7 10 5 = 83 := by
  sorry

end storage_wheels_count_l3163_316352


namespace x_cubed_plus_y_cubed_le_two_l3163_316394

theorem x_cubed_plus_y_cubed_le_two (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_ineq : x^2 + y^3 ≥ x^3 + y^4) : 
  x^3 + y^3 ≤ 2 := by
sorry

end x_cubed_plus_y_cubed_le_two_l3163_316394


namespace unique_solution_for_equation_l3163_316361

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ x = 11888 ∧ y = 11893 := by
  sorry

end unique_solution_for_equation_l3163_316361


namespace angle_measure_in_triangle_l3163_316356

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √2, b = 2, and sin B + cos B = √2, then the measure of angle A is π/6. -/
theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 2 →
  b = 2 →
  Real.sin B + Real.cos B = Real.sqrt 2 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = π / 6 := by
sorry

end angle_measure_in_triangle_l3163_316356


namespace representation_of_2008_l3163_316398

theorem representation_of_2008 : ∃ (a b c : ℕ), 
  2008 = a + 40 * b + 40 * c ∧ 
  (1 : ℚ) / a + (b : ℚ) / 40 + (c : ℚ) / 40 = 1 := by
  sorry

end representation_of_2008_l3163_316398


namespace inequality_proof_l3163_316342

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ 1 / (a * b * c) := by
  sorry

end inequality_proof_l3163_316342


namespace bacon_suggestion_count_l3163_316375

/-- The number of students who suggested bacon, given the total number of students
    and the number of students who suggested mashed potatoes. -/
def students_suggested_bacon (total : ℕ) (mashed_potatoes : ℕ) : ℕ :=
  total - mashed_potatoes

/-- Theorem stating that the number of students who suggested bacon is 125,
    given the total number of students and those who suggested mashed potatoes. -/
theorem bacon_suggestion_count :
  students_suggested_bacon 310 185 = 125 := by
  sorry

end bacon_suggestion_count_l3163_316375


namespace sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l3163_316320

-- Define a trihedral angle
structure TrihedralAngle where
  -- Angles between edges and opposite faces
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Plane angles at vertex
  θ₁ : ℝ
  θ₂ : ℝ
  θ₃ : ℝ
  -- Ensure all angles are positive
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ
  θ₁_pos : 0 < θ₁
  θ₂_pos : 0 < θ₂
  θ₃_pos : 0 < θ₃

-- Theorem 1: Sum of angles between edges and opposite faces is less than sum of plane angles
theorem sum_edge_face_angles_less_than_plane_angles (t : TrihedralAngle) :
  t.α + t.β + t.γ < t.θ₁ + t.θ₂ + t.θ₃ := by
  sorry

-- Theorem 2: If all plane angles are acute, sum of angles between edges and opposite faces 
-- is greater than half the sum of plane angles
theorem sum_edge_face_angles_greater_than_half_plane_angles_if_acute (t : TrihedralAngle)
  (h₁ : t.θ₁ < π/2) (h₂ : t.θ₂ < π/2) (h₃ : t.θ₃ < π/2) :
  t.α + t.β + t.γ > (t.θ₁ + t.θ₂ + t.θ₃) / 2 := by
  sorry

end sum_edge_face_angles_less_than_plane_angles_sum_edge_face_angles_greater_than_half_plane_angles_if_acute_l3163_316320


namespace sign_language_size_l3163_316362

theorem sign_language_size :
  ∀ n : ℕ,
  (n ≥ 2) →
  (n^2 - (n-2)^2 = 888) →
  n = 223 := by
sorry

end sign_language_size_l3163_316362


namespace square_of_sum_fifteen_three_l3163_316314

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end square_of_sum_fifteen_three_l3163_316314


namespace dollar_squared_diff_zero_l3163_316335

/-- Custom operation definition -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem statement -/
theorem dollar_squared_diff_zero (x y : ℝ) : dollar ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end dollar_squared_diff_zero_l3163_316335


namespace largest_divisible_by_digits_correct_l3163_316327

/-- A function that returns true if n is divisible by all of its distinct, non-zero digits -/
def divisible_by_digits (n : ℕ) : Bool :=
  let digits := n.digits 10
  digits.all (λ d => d ≠ 0 ∧ n % d = 0)

/-- The largest three-digit number divisible by all its distinct, non-zero digits -/
def largest_divisible_by_digits : ℕ := 936

theorem largest_divisible_by_digits_correct :
  (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ divisible_by_digits n → n ≤ largest_divisible_by_digits) ∧
  divisible_by_digits largest_divisible_by_digits :=
sorry

end largest_divisible_by_digits_correct_l3163_316327


namespace expression_evaluation_l3163_316377

theorem expression_evaluation : (π - 2) ^ 0 - 2 * Real.sqrt 3 * 2⁻¹ - Real.sqrt 16 + |1 - Real.sqrt 3| = -4 := by
  sorry

end expression_evaluation_l3163_316377


namespace total_girls_count_l3163_316328

theorem total_girls_count (van1_students van2_students van3_students van4_students van5_students : Nat)
                          (van1_boys van2_boys van3_boys van4_boys van5_boys : Nat)
                          (h1 : van1_students = 24) (h2 : van2_students = 30) (h3 : van3_students = 20)
                          (h4 : van4_students = 36) (h5 : van5_students = 29)
                          (h6 : van1_boys = 12) (h7 : van2_boys = 16) (h8 : van3_boys = 10)
                          (h9 : van4_boys = 18) (h10 : van5_boys = 8) :
  (van1_students - van1_boys) + (van2_students - van2_boys) + (van3_students - van3_boys) +
  (van4_students - van4_boys) + (van5_students - van5_boys) = 75 := by
  sorry

end total_girls_count_l3163_316328


namespace local_maximum_at_e_l3163_316381

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem local_maximum_at_e :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (Real.exp 1 - δ) (Real.exp 1 + δ),
    x ≠ Real.exp 1 → f x < f (Real.exp 1) := by
  sorry

end local_maximum_at_e_l3163_316381


namespace hundredth_odd_followed_by_hundredth_even_l3163_316366

/-- The nth odd positive integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The nth even positive integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem hundredth_odd_followed_by_hundredth_even :
  nth_odd 100 = 199 ∧ nth_even 100 = nth_odd 100 + 1 := by
  sorry

end hundredth_odd_followed_by_hundredth_even_l3163_316366


namespace handshake_remainder_l3163_316325

/-- The number of ways 8 people can shake hands, where each person shakes hands with exactly 2 others -/
def M : ℕ := sorry

/-- The group size -/
def group_size : ℕ := 8

/-- The number of handshakes per person -/
def handshakes_per_person : ℕ := 2

theorem handshake_remainder : M ≡ 355 [ZMOD 1000] := by sorry

end handshake_remainder_l3163_316325


namespace coffee_preference_expectation_l3163_316384

theorem coffee_preference_expectation (total_sample : ℕ) 
  (coffee_ratio : ℚ) (h1 : coffee_ratio = 3 / 7) (h2 : total_sample = 350) : 
  ℕ := by
  sorry

#check coffee_preference_expectation

end coffee_preference_expectation_l3163_316384


namespace largest_angle_convex_pentagon_l3163_316376

/-- 
Given a convex pentagon with interior angles measuring x+1, 2x, 3x, 4x, and 5x-1 degrees,
where x is a positive real number and the sum of these angles is 540 degrees,
prove that the measure of the largest angle is 179 degrees.
-/
theorem largest_angle_convex_pentagon (x : ℝ) 
  (h_positive : x > 0)
  (h_sum : (x + 1) + 2*x + 3*x + 4*x + (5*x - 1) = 540) :
  max (x + 1) (max (2*x) (max (3*x) (max (4*x) (5*x - 1)))) = 179 := by
  sorry

end largest_angle_convex_pentagon_l3163_316376


namespace complement_A_intersect_B_l3163_316308

def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {3, 5, 6}

theorem complement_A_intersect_B : (U \ A) ∩ B = {6} := by sorry

end complement_A_intersect_B_l3163_316308


namespace subtraction_division_result_l3163_316344

theorem subtraction_division_result : 1.85 - 1.85 / 1.85 = 0.85 := by
  sorry

end subtraction_division_result_l3163_316344


namespace olivia_coin_device_l3163_316311

def coin_change (start : ℕ) (change : ℕ) (target : ℕ) : Prop :=
  ∃ k : ℕ, start + k * (change - 1) = target

theorem olivia_coin_device (targets : List ℕ := [492, 776, 1248, 1520, 1984]) :
  ∀ t ∈ targets, (coin_change 1 80 t ↔ t = 1984) := by sorry

end olivia_coin_device_l3163_316311


namespace sum_of_coefficients_l3163_316399

theorem sum_of_coefficients (a b c : ℤ) : 
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 21*x + 108 = (x - b) * (x - c)) →
  a + b + c = 32 := by
sorry

end sum_of_coefficients_l3163_316399


namespace total_tomatoes_l3163_316304

def tomato_problem (plant1 plant2 plant3 : ℕ) : Prop :=
  plant1 = 24 ∧
  plant2 = (plant1 / 2) + 5 ∧
  plant3 = plant2 + 2 ∧
  plant1 + plant2 + plant3 = 60

theorem total_tomatoes :
  ∃ plant1 plant2 plant3 : ℕ, tomato_problem plant1 plant2 plant3 :=
sorry

end total_tomatoes_l3163_316304


namespace quartic_equation_real_roots_l3163_316351

theorem quartic_equation_real_roots :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∀ x : ℝ, 3 * x^4 + x^3 - 6 * x^2 + x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end quartic_equation_real_roots_l3163_316351


namespace hidden_primes_average_l3163_316357

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (visible hidden : ℕ) : ℕ := visible + hidden

theorem hidden_primes_average (h1 h2 h3 : ℕ) :
  is_prime h1 →
  is_prime h2 →
  is_prime h3 →
  card_sum 44 h1 = card_sum 59 h2 →
  card_sum 44 h1 = card_sum 38 h3 →
  (h1 + h2 + h3) / 3 = 14 :=
by sorry

end hidden_primes_average_l3163_316357


namespace arithmetic_equality_l3163_316302

theorem arithmetic_equality : 3 * (7 - 5) - 5 = 1 := by
  sorry

end arithmetic_equality_l3163_316302


namespace consecutive_product_3024_l3163_316372

theorem consecutive_product_3024 :
  ∀ n : ℕ, n > 0 →
  (n * (n + 1) * (n + 2) * (n + 3) = 3024) ↔ n = 6 := by
sorry

end consecutive_product_3024_l3163_316372


namespace additional_passengers_proof_l3163_316300

/-- The number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- The number of seats in each carriage -/
def seats_per_carriage : ℕ := 25

/-- The total number of passengers that can be accommodated in 3 trains with additional capacity -/
def total_passengers : ℕ := 420

/-- The number of trains -/
def num_trains : ℕ := 3

/-- The additional number of passengers each carriage can accommodate -/
def additional_passengers : ℕ := 10

theorem additional_passengers_proof :
  additional_passengers = 
    (total_passengers - num_trains * carriages_per_train * seats_per_carriage) / 
    (num_trains * carriages_per_train) :=
by sorry

end additional_passengers_proof_l3163_316300


namespace smallest_positive_angle_exists_l3163_316396

theorem smallest_positive_angle_exists : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  (∀ φ : ℝ, φ > 0 ∧ φ < 360 ∧ 
    Real.cos (φ * Real.pi / 180) = 
      Real.sin (45 * Real.pi / 180) + 
      Real.cos (30 * Real.pi / 180) - 
      Real.sin (18 * Real.pi / 180) - 
      Real.cos (12 * Real.pi / 180) → 
    θ ≤ φ) ∧
  Real.cos (θ * Real.pi / 180) = 
    Real.sin (45 * Real.pi / 180) + 
    Real.cos (30 * Real.pi / 180) - 
    Real.sin (18 * Real.pi / 180) - 
    Real.cos (12 * Real.pi / 180) :=
by sorry

end smallest_positive_angle_exists_l3163_316396


namespace sin_squared_sum_l3163_316303

theorem sin_squared_sum (α β : ℝ) 
  (h : Real.arcsin (Real.sin α + Real.sin β) + Real.arcsin (Real.sin α - Real.sin β) = π / 2) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 = 1 / 2 := by
  sorry

end sin_squared_sum_l3163_316303


namespace problem_1_problem_2_l3163_316388

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = -1) :
  (4 - x) * (2 * x + 1) + 3 * x * (x - 3) = 7 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 1) (hy : y = 1/2) :
  ((x + 2*y)^2 - (3*x + y)*(3*x - y) - 5*y^2) / (-1/2 * x) = 12 := by sorry

end problem_1_problem_2_l3163_316388


namespace min_sum_with_constraint_l3163_316364

theorem min_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = x * y → a + b ≤ x + y ∧ a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end min_sum_with_constraint_l3163_316364


namespace systematic_sample_third_element_l3163_316323

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_sample : ℕ
  interval : ℕ

/-- Checks if a seat number is in the systematic sample -/
def in_sample (s : SystematicSample) (seat : ℕ) : Prop :=
  ∃ k : ℕ, seat = s.first_sample + k * s.interval ∧ seat ≤ s.population_size

theorem systematic_sample_third_element 
  (s : SystematicSample)
  (h_pop : s.population_size = 45)
  (h_sample : s.sample_size = 3)
  (h_interval : s.interval = s.population_size / s.sample_size)
  (h_11 : in_sample s 11)
  (h_41 : in_sample s 41) :
  in_sample s 26 := by
  sorry

#check systematic_sample_third_element

end systematic_sample_third_element_l3163_316323


namespace percentage_not_working_l3163_316318

/-- Represents the employment status of a group --/
structure EmploymentStatus where
  fullTime : Rat
  partTime : Rat

/-- Represents the survey data --/
structure SurveyData where
  mothers : EmploymentStatus
  fathers : EmploymentStatus
  grandparents : EmploymentStatus
  womenPercentage : Rat
  menPercentage : Rat
  grandparentsPercentage : Rat

/-- Calculates the percentage of individuals not working in a given group --/
def notWorkingPercentage (status : EmploymentStatus) : Rat :=
  1 - status.fullTime - status.partTime

/-- Theorem stating the percentage of surveyed individuals not holding a job --/
theorem percentage_not_working (data : SurveyData) :
  data.mothers = { fullTime := 5/6, partTime := 1/6 } →
  data.fathers = { fullTime := 3/4, partTime := 1/8 } →
  data.grandparents = { fullTime := 1/2, partTime := 1/4 } →
  data.womenPercentage = 55/100 →
  data.menPercentage = 35/100 →
  data.grandparentsPercentage = 1/10 →
  (notWorkingPercentage data.mothers) * data.womenPercentage +
  (notWorkingPercentage data.fathers) * data.menPercentage +
  (notWorkingPercentage data.grandparents) * data.grandparentsPercentage =
  6875/100000 := by
  sorry

end percentage_not_working_l3163_316318


namespace shaded_area_is_16pi_l3163_316348

/-- Represents the pattern of semicircles as described in the problem -/
structure SemicirclePattern where
  diameter : ℝ
  length : ℝ

/-- Calculates the area of the shaded region in the semicircle pattern -/
def shaded_area (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of the given pattern is 16π square inches -/
theorem shaded_area_is_16pi (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 4)
  (h2 : pattern.length = 18) : 
  shaded_area pattern = 16 * Real.pi := by
  sorry

end shaded_area_is_16pi_l3163_316348


namespace unique_integer_solution_l3163_316371

theorem unique_integer_solution (w x y z : ℤ) :
  w^2 + 11*x^2 - 8*y^2 - 12*y*z - 10*z^2 = 0 →
  w = 0 ∧ x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end unique_integer_solution_l3163_316371


namespace largest_factor_of_9975_l3163_316315

theorem largest_factor_of_9975 : 
  ∀ n : ℕ, n ∣ 9975 ∧ n < 10000 → n ≤ 4975 :=
by
  sorry

end largest_factor_of_9975_l3163_316315


namespace comparison_of_products_l3163_316343

theorem comparison_of_products (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end comparison_of_products_l3163_316343


namespace greater_number_proof_l3163_316305

theorem greater_number_proof (x y : ℝ) (h_sum : x + y = 40) (h_diff : x - y = 12) (h_greater : x > y) : x = 26 := by
  sorry

end greater_number_proof_l3163_316305


namespace parabola_theorem_l3163_316347

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P on parabola C -/
def point_P : ℝ × ℝ := (2, 1)

/-- Focus F of parabola C -/
def focus_F : ℝ × ℝ := (0, 1)

/-- Point H where the axis of the parabola intersects the y-axis -/
def point_H : ℝ × ℝ := (0, -1)

/-- Line l passing through focus F and intersecting parabola C at points A and B -/
def line_l (x y : ℝ) : Prop := ∃ (k b : ℝ), y = k*x + b ∧ (0 = k*0 + b - 1)

/-- Points A and B on parabola C -/
def points_AB (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

/-- AB is perpendicular to HB -/
def AB_perp_HB (A B : ℝ × ℝ) : Prop :=
  (A.2 - B.2) * (B.1 - point_H.1) = -(A.1 - B.1) * (B.2 - point_H.2)

/-- Main theorem: |AF| - |BF| = 4 -/
theorem parabola_theorem (A B : ℝ × ℝ) :
  points_AB A B → AB_perp_HB A B →
  Real.sqrt ((A.1 - focus_F.1)^2 + (A.2 - focus_F.2)^2) -
  Real.sqrt ((B.1 - focus_F.1)^2 + (B.2 - focus_F.2)^2) = 4 :=
by sorry

end parabola_theorem_l3163_316347


namespace arithmetic_expression_evaluation_l3163_316373

theorem arithmetic_expression_evaluation : 8 / 2 - (3 - 5 + 7) + 3 * 4 = 11 := by
  sorry

end arithmetic_expression_evaluation_l3163_316373


namespace coltons_remaining_stickers_l3163_316312

/-- The number of stickers Colton has left after giving some away to friends. -/
def stickers_left (initial_stickers : ℕ) (stickers_per_friend : ℕ) (num_friends : ℕ) 
  (extra_to_mandy : ℕ) (less_to_justin : ℕ) : ℕ :=
  let stickers_to_friends := stickers_per_friend * num_friends
  let stickers_to_mandy := stickers_to_friends + extra_to_mandy
  let stickers_to_justin := stickers_to_mandy - less_to_justin
  let total_given_away := stickers_to_friends + stickers_to_mandy + stickers_to_justin
  initial_stickers - total_given_away

/-- Theorem stating that Colton has 42 stickers left given the problem conditions. -/
theorem coltons_remaining_stickers : 
  stickers_left 72 4 3 2 10 = 42 := by
  sorry

end coltons_remaining_stickers_l3163_316312


namespace smallest_period_of_special_function_l3163_316353

/-- A function satisfying the given condition -/
def is_special_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of a function -/
def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬ is_period f q

/-- The main theorem -/
theorem smallest_period_of_special_function (f : ℝ → ℝ) (h : is_special_function f) :
  is_smallest_positive_period f 30 :=
sorry

end smallest_period_of_special_function_l3163_316353


namespace subject_selection_methods_l3163_316313

/-- The number of subjects excluding the mandatory subject -/
def n : ℕ := 5

/-- The number of subjects to be chosen from the remaining subjects -/
def k : ℕ := 2

/-- Combination formula -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subject_selection_methods :
  combination n k = 10 :=
by sorry

end subject_selection_methods_l3163_316313


namespace polynomial_expansion_l3163_316358

theorem polynomial_expansion (t : ℝ) : 
  (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 := by sorry

end polynomial_expansion_l3163_316358


namespace least_addition_for_divisibility_l3163_316386

theorem least_addition_for_divisibility :
  ∃ (x : ℕ), x = 8 ∧ 
  (∀ (y : ℕ), y < x → ¬(37 ∣ (157639 + y))) ∧
  (37 ∣ (157639 + x)) := by
  sorry

end least_addition_for_divisibility_l3163_316386


namespace two_digit_number_product_l3163_316307

theorem two_digit_number_product (n : ℕ) (tens units : ℕ) : 
  n = tens * 10 + units →
  n = 24 →
  units = tens + 2 →
  n * (tens + units) = 144 := by
sorry

end two_digit_number_product_l3163_316307


namespace quadratic_inequality_range_l3163_316331

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3/4 < 0) ↔ -3 < k ∧ k ≤ 0 :=
by sorry

end quadratic_inequality_range_l3163_316331


namespace sum_of_divisors_143_l3163_316317

/-- The sum of all positive integer divisors of 143 is 168 -/
theorem sum_of_divisors_143 : (Finset.filter (· ∣ 143) (Finset.range 144)).sum id = 168 := by
  sorry

end sum_of_divisors_143_l3163_316317


namespace factorization_of_a_squared_minus_2a_l3163_316392

theorem factorization_of_a_squared_minus_2a (a : ℝ) : a^2 - 2*a = a*(a - 2) := by
  sorry

end factorization_of_a_squared_minus_2a_l3163_316392


namespace arccos_one_half_equals_pi_third_l3163_316326

theorem arccos_one_half_equals_pi_third : 
  Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_equals_pi_third_l3163_316326


namespace sarah_shopping_theorem_l3163_316380

/-- The amount of money Sarah started with -/
def initial_amount : ℕ := sorry

/-- The cost of one toy car -/
def toy_car_cost : ℕ := 11

/-- The number of toy cars Sarah bought -/
def num_toy_cars : ℕ := 2

/-- The cost of the scarf -/
def scarf_cost : ℕ := 10

/-- The cost of the beanie -/
def beanie_cost : ℕ := 14

/-- The amount of money Sarah has remaining after all purchases -/
def remaining_money : ℕ := 7

/-- Theorem stating that the initial amount is equal to the sum of all purchases plus the remaining money -/
theorem sarah_shopping_theorem : 
  initial_amount = toy_car_cost * num_toy_cars + scarf_cost + beanie_cost + remaining_money :=
by sorry

end sarah_shopping_theorem_l3163_316380


namespace volume_of_rotated_region_l3163_316340

/-- The volume of a solid formed by rotating a region consisting of a 6x1 rectangle
    and a 4x3 rectangle about the x-axis -/
theorem volume_of_rotated_region : ℝ := by
  -- Define the dimensions of the rectangles
  let height1 : ℝ := 6
  let width1 : ℝ := 1
  let height2 : ℝ := 3
  let width2 : ℝ := 4

  -- Define the volumes of the two cylinders
  let volume1 : ℝ := Real.pi * height1^2 * width1
  let volume2 : ℝ := Real.pi * height2^2 * width2

  -- Define the total volume
  let total_volume : ℝ := volume1 + volume2

  -- Prove that the total volume equals 72π
  have : total_volume = 72 * Real.pi := by sorry

  -- Return the result
  exact 72 * Real.pi

end volume_of_rotated_region_l3163_316340


namespace g_2016_equals_1_l3163_316319

-- Define the properties of function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

-- Define function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

-- Theorem statement
theorem g_2016_equals_1 (f : ℝ → ℝ) (h : satisfies_conditions f) :
  g f 2016 = 1 := by
  sorry

end g_2016_equals_1_l3163_316319


namespace larger_number_problem_l3163_316355

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 47) :
  max x y = 25 := by
  sorry

end larger_number_problem_l3163_316355


namespace base3_21021_equals_196_l3163_316395

def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_21021_equals_196 :
  base3_to_base10 [2, 1, 0, 2, 1] = 196 := by
  sorry

end base3_21021_equals_196_l3163_316395


namespace intersection_volume_l3163_316334

-- Define the two cubes
def cube1 (x y z : ℝ) : Prop := max (|x|) (max |y| |z|) ≤ 1
def cube2 (x y z : ℝ) : Prop := max (|x-1|) (max |y-1| |z-1|) ≤ 1

-- Define the intersection of the two cubes
def intersection (x y z : ℝ) : Prop := cube1 x y z ∧ cube2 x y z

-- Define the volume of a region
noncomputable def volume (region : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem intersection_volume : volume intersection = 1 := by sorry

end intersection_volume_l3163_316334


namespace dan_picked_nine_limes_l3163_316391

/-- The number of limes Dan gave to Sara -/
def limes_given_to_Sara : ℕ := 4

/-- The number of limes Dan has left -/
def limes_left_with_Dan : ℕ := 5

/-- The total number of limes Dan picked initially -/
def total_limes : ℕ := limes_given_to_Sara + limes_left_with_Dan

theorem dan_picked_nine_limes : total_limes = 9 := by
  sorry

end dan_picked_nine_limes_l3163_316391


namespace arithmetic_sequence_range_l3163_316390

theorem arithmetic_sequence_range (a : ℝ) :
  (∀ n : ℕ+, (1 + (a + n - 1)) / (a + n - 1) ≤ (1 + (a + 5 - 1)) / (a + 5 - 1)) →
  -4 < a ∧ a < -3 := by
  sorry

end arithmetic_sequence_range_l3163_316390


namespace max_value_is_b_l3163_316337

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (max (max (1/2) b) (2*a*b)) (a^2 + b^2) := by
  sorry

end max_value_is_b_l3163_316337


namespace more_stable_performance_l3163_316332

/-- Given two students A and B with their respective variances, 
    proves that the student with lower variance has more stable performance -/
theorem more_stable_performance (S_A_squared S_B_squared : ℝ) 
  (h1 : S_A_squared = 0.3)
  (h2 : S_B_squared = 0.1) : 
  S_B_squared < S_A_squared := by sorry


end more_stable_performance_l3163_316332


namespace cupcake_price_l3163_316370

/-- Proves that the price of each cupcake is $2 given the problem conditions --/
theorem cupcake_price (cookies_sold : ℕ) (cookie_price : ℚ) (cupcakes_sold : ℕ) 
  (spoons_bought : ℕ) (spoon_price : ℚ) (money_left : ℚ)
  (h1 : cookies_sold = 40)
  (h2 : cookie_price = 4/5)
  (h3 : cupcakes_sold = 30)
  (h4 : spoons_bought = 2)
  (h5 : spoon_price = 13/2)
  (h6 : money_left = 79) :
  let total_earned := cookies_sold * cookie_price + cupcakes_sold * (2 : ℚ)
  let total_spent := spoons_bought * spoon_price + money_left
  total_earned = total_spent := by sorry

end cupcake_price_l3163_316370


namespace probability_of_convex_quadrilateral_l3163_316389

/-- The number of points on the circle -/
def n : ℕ := 6

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords forming a convex quadrilateral -/
def probability : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability is 1/91 -/
theorem probability_of_convex_quadrilateral : probability = 1 / 91 := by
  sorry

end probability_of_convex_quadrilateral_l3163_316389


namespace arrangements_with_separation_l3163_316301

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the problem. -/
def numberOfPeople : ℕ := 5

/-- Theorem stating that the number of arrangements with at least one person between A and B is 72. -/
theorem arrangements_with_separation :
  totalArrangements numberOfPeople - adjacentArrangements numberOfPeople = 72 := by
  sorry

#eval totalArrangements numberOfPeople - adjacentArrangements numberOfPeople

end arrangements_with_separation_l3163_316301


namespace factorization_problem1_l3163_316368

theorem factorization_problem1 (a b : ℝ) :
  -3 * a^3 + 12 * a^2 * b - 12 * a * b^2 = -3 * a * (a - 2*b)^2 := by sorry

end factorization_problem1_l3163_316368


namespace dodecagon_diagonals_l3163_316339

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l3163_316339


namespace periodic_function_value_l3163_316379

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value :
  ∀ (f : ℝ → ℝ),
  is_periodic f 4 →
  (∀ x ∈ Set.Icc 0 4, f x = x) →
  f 7.6 = 3.6 := by
sorry

end periodic_function_value_l3163_316379


namespace fencing_cost_per_meter_l3163_316329

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 55)
  (h2 : breadth = 45)
  (h3 : length = breadth + 10)
  (h4 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end fencing_cost_per_meter_l3163_316329


namespace five_twelve_thirteen_pythagorean_triple_l3163_316354

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple : isPythagoreanTriple 5 12 13 := by
  sorry

end five_twelve_thirteen_pythagorean_triple_l3163_316354


namespace vegan_soy_free_menu_fraction_l3163_316367

theorem vegan_soy_free_menu_fraction 
  (total_menu : ℕ) 
  (vegan_fraction : Rat) 
  (soy_containing_vegan_fraction : Rat) 
  (h1 : vegan_fraction = 1 / 10) 
  (h2 : soy_containing_vegan_fraction = 2 / 3) : 
  (1 - soy_containing_vegan_fraction) * vegan_fraction = 1 / 30 := by
sorry

end vegan_soy_free_menu_fraction_l3163_316367


namespace min_value_sum_squares_l3163_316378

theorem min_value_sum_squares (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = k) (hk : k ≥ -1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = k → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_sum_squares_l3163_316378


namespace spring_work_compression_l3163_316316

/-- Given a spring that is compressed 1 cm by a 10 N force, 
    the work done to compress it by 10 cm is 5 J. -/
theorem spring_work_compression (k : ℝ) : 
  (10 : ℝ) = k * 1 → (∫ x in (0 : ℝ)..(10 : ℝ), k * x) = 5 := by
  sorry

end spring_work_compression_l3163_316316


namespace hyperbola_eccentricity_l3163_316374

/-- Theorem: Eccentricity of a hyperbola with specific properties --/
theorem hyperbola_eccentricity (a b c : ℝ) (h : c^2 = a^2 + b^2) : 
  let f1 : ℝ × ℝ := (-c, 0)
  let f2 : ℝ × ℝ := (c, 0)
  let A : ℝ × ℝ := (c, b^2 / a)
  let B : ℝ × ℝ := (c, -b^2 / a)
  let G : ℝ × ℝ := (c / 3, 0)
  ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    (G.1 - A.1) * (f1.1 - B.1) + (G.2 - A.2) * (f1.2 - B.2) = 0 →
    c / a = Real.sqrt 3 :=
by sorry

end hyperbola_eccentricity_l3163_316374


namespace sams_juice_consumption_l3163_316324

theorem sams_juice_consumption (total_juice : ℚ) (sams_portion : ℚ) : 
  total_juice = 3/7 → sams_portion = 4/5 → sams_portion * total_juice = 12/35 := by
  sorry

end sams_juice_consumption_l3163_316324


namespace train_ride_total_time_l3163_316306

def train_ride_duration (reading_time eating_time movie_time nap_time : ℕ) : ℕ :=
  reading_time + eating_time + movie_time + nap_time

theorem train_ride_total_time :
  let reading_time : ℕ := 2
  let eating_time : ℕ := 1
  let movie_time : ℕ := 3
  let nap_time : ℕ := 3
  train_ride_duration reading_time eating_time movie_time nap_time = 9 := by
  sorry

end train_ride_total_time_l3163_316306


namespace no_two_obtuse_angles_l3163_316330

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_180 : angle1 + angle2 + angle3 = 180
  positive : angle1 > 0 ∧ angle2 > 0 ∧ angle3 > 0

-- Define what an obtuse angle is
def isObtuse (angle : Real) : Prop := angle > 90

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : 
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle2) ∧
  ¬(isObtuse t.angle1 ∧ isObtuse t.angle3) ∧
  ¬(isObtuse t.angle2 ∧ isObtuse t.angle3) := by
  sorry


end no_two_obtuse_angles_l3163_316330
