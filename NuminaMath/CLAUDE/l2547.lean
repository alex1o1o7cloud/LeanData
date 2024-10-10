import Mathlib

namespace exam_question_distribution_l2547_254786

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks : ℕ → ℕ) (group_b_marks : ℕ → ℕ) (group_c_marks : ℕ → ℕ)
  (group_b_count : ℕ) :
  total_questions = 100 →
  group_b_count = 23 →
  (∀ n, group_a_marks n = n) →
  (∀ n, group_b_marks n = 2 * n) →
  (∀ n, group_c_marks n = 3 * n) →
  (∀ a b c, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c, a + b + c = total_questions → 
    group_a_marks a ≥ (3 * (group_a_marks a + group_b_marks b + group_c_marks c)) / 5) →
  (∀ a b c, a + b + c = total_questions → 
    group_b_marks b ≤ (group_a_marks a + group_b_marks b + group_c_marks c) / 4) →
  ∃ a c, a + group_b_count + c = total_questions ∧ c = 1 :=
by sorry

end exam_question_distribution_l2547_254786


namespace max_value_interval_range_l2547_254783

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The interval (a, 6-a^2) --/
def interval (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6 - a^2}

/-- Theorem stating the range of a for which f has a maximum on the interval --/
theorem max_value_interval_range :
  ∀ a : ℝ, (∃ x_max ∈ interval a, ∀ x ∈ interval a, f x ≤ f x_max) →
    a > -Real.sqrt 7 ∧ a ≤ -2 :=
sorry

end max_value_interval_range_l2547_254783


namespace mrs_hilt_money_left_l2547_254752

def initial_amount : ℕ := 10
def truck_cost : ℕ := 3
def pencil_case_cost : ℕ := 2

theorem mrs_hilt_money_left : 
  initial_amount - (truck_cost + pencil_case_cost) = 5 := by
  sorry

end mrs_hilt_money_left_l2547_254752


namespace bowling_ball_weight_l2547_254706

theorem bowling_ball_weight : 
  ∀ (b c : ℝ),
  5 * b = 2 * c →
  3 * c = 72 →
  b = 9.6 := by
sorry

end bowling_ball_weight_l2547_254706


namespace remaining_ribbon_length_l2547_254707

/-- Calculates the remaining ribbon length after wrapping gifts -/
theorem remaining_ribbon_length
  (num_gifts : ℕ)
  (ribbon_per_gift : ℝ)
  (initial_ribbon_length : ℝ)
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : initial_ribbon_length = 15) :
  initial_ribbon_length - (↑num_gifts * ribbon_per_gift) = 3 :=
by sorry

end remaining_ribbon_length_l2547_254707


namespace sup_good_is_ln_2_l2547_254724

/-- A positive real number d is good if there exists an infinite sequence
    a₁, a₂, a₃, ... ∈ (0,d) such that for each n, the points a₁, a₂, ..., aₙ
    partition the interval [0,d] into segments of length at most 1/n each. -/
def IsGood (d : ℝ) : Prop :=
  d > 0 ∧ ∃ a : ℕ → ℝ, ∀ n : ℕ,
    (∀ i : ℕ, i ≤ n → 0 < a i ∧ a i < d) ∧
    (∀ i : ℕ, i ≤ n → ∀ j : ℕ, j ≤ n → i ≠ j → |a i - a j| ≤ 1 / n) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ d → ∃ i : ℕ, i ≤ n ∧ |x - a i| ≤ 1 / n)

/-- The supremum of the set of all good numbers is ln 2. -/
theorem sup_good_is_ln_2 : sSup {d : ℝ | IsGood d} = Real.log 2 := by
  sorry


end sup_good_is_ln_2_l2547_254724


namespace min_squares_covering_sqrt63_l2547_254743

theorem min_squares_covering_sqrt63 :
  ∀ n : ℕ, n ≥ 2 → (4 * n - 4 ≥ Real.sqrt 63 ↔ n ≥ 3) :=
by sorry

end min_squares_covering_sqrt63_l2547_254743


namespace g_minus_one_eq_zero_iff_s_eq_neg_six_l2547_254759

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 2 * x^2 + x + s

/-- Theorem stating that g(-1) = 0 if and only if s = -6 -/
theorem g_minus_one_eq_zero_iff_s_eq_neg_six :
  ∀ s : ℝ, g s (-1) = 0 ↔ s = -6 := by sorry

end g_minus_one_eq_zero_iff_s_eq_neg_six_l2547_254759


namespace tenth_term_of_arithmetic_sequence_l2547_254767

/-- Given an arithmetic sequence with first term 2/3 and second term 4/3,
    prove that its tenth term is 20/3. -/
theorem tenth_term_of_arithmetic_sequence :
  let a₁ : ℚ := 2/3  -- First term
  let a₂ : ℚ := 4/3  -- Second term
  let d : ℚ := a₂ - a₁  -- Common difference
  let a₁₀ : ℚ := a₁ + 9 * d  -- Tenth term
  a₁₀ = 20/3 :=
by sorry

end tenth_term_of_arithmetic_sequence_l2547_254767


namespace gcd_problem_l2547_254718

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 7769) :
  Int.gcd (4 * b^2 + 81 * b + 144) (2 * b + 7) = 1 := by
  sorry

end gcd_problem_l2547_254718


namespace distance_AB_is_correct_l2547_254776

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let first_meeting_distance : ℝ := 700
  let second_meeting_distance : ℝ := 400
  -- Define the distance as a variable to be solved
  let d : ℝ := 1700
  d

theorem distance_AB_is_correct : distance_AB = 1700 := by
  -- Unfold the definition of distance_AB
  unfold distance_AB
  -- The proof goes here
  sorry


end distance_AB_is_correct_l2547_254776


namespace not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l2547_254721

-- Definition of a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Statement 1
theorem not_closed_set_1 : ¬ is_closed_set {-4, -2, 0, 2, 4} := by sorry

-- Statement 2
def positive_integers : Set Int := {n | n > 0}

theorem not_closed_set_2 : ¬ is_closed_set positive_integers := by sorry

-- Statement 3
def multiples_of_three : Set Int := {n | ∃ k : Int, n = 3 * k}

theorem closed_set_3 : is_closed_set multiples_of_three := by sorry

-- Statement 4
theorem exist_closed_sets_union_not_closed :
  ∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂) := by sorry

end not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l2547_254721


namespace no_linear_term_implies_m_equals_six_l2547_254791

theorem no_linear_term_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2*x + m) * (x - 3) = 2*x^2 - 3*m) → m = 6 := by
  sorry

end no_linear_term_implies_m_equals_six_l2547_254791


namespace tom_teaching_years_l2547_254710

/-- Represents the number of years Tom has been teaching. -/
def tom_years : ℕ := sorry

/-- Represents the number of years Devin has been teaching. -/
def devin_years : ℕ := sorry

/-- The total number of years Tom and Devin have been teaching. -/
def total_years : ℕ := 70

/-- Theorem stating that Tom has been teaching for 50 years, given the conditions. -/
theorem tom_teaching_years :
  (tom_years + devin_years = total_years) ∧
  (devin_years = tom_years / 2 - 5) →
  tom_years = 50 :=
by sorry

end tom_teaching_years_l2547_254710


namespace negation_of_existence_exponential_cube_inequality_l2547_254754

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ p x) ↔ (∀ x : ℝ, x > 0 → ¬ p x) := by sorry

theorem exponential_cube_inequality :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by sorry

end negation_of_existence_exponential_cube_inequality_l2547_254754


namespace right_triangle_hypotenuse_l2547_254726

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 24 → b = 10 → h^2 = a^2 + b^2 → h = 26 := by
  sorry

end right_triangle_hypotenuse_l2547_254726


namespace right_triangle_rotation_creates_cone_l2547_254705

/-- A right triangle is a triangle with one right angle -/
structure RightTriangle where
  -- We don't need to define the specifics of a right triangle for this statement
  mk :: 

/-- A cone is a three-dimensional geometric shape with a circular base that tapers to a point -/
structure Cone where
  -- We don't need to define the specifics of a cone for this statement
  mk ::

/-- Rotation of a right triangle around one of its legs -/
def rotateAroundLeg (t : RightTriangle) : Cone :=
  sorry

theorem right_triangle_rotation_creates_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateAroundLeg t = c :=
sorry

end right_triangle_rotation_creates_cone_l2547_254705


namespace sum_of_cubes_difference_l2547_254796

theorem sum_of_cubes_difference (p q r : ℕ+) :
  (p + q + r : ℕ)^3 - p^3 - q^3 - r^3 = 200 →
  (p : ℕ) + q + r = 7 := by
sorry

end sum_of_cubes_difference_l2547_254796


namespace strawberry_picker_l2547_254713

/-- Given three people picking strawberries, proves that one person picked 200 strawberries -/
theorem strawberry_picker (total jonathan_matthew matthew_zac : ℕ) 
  (h_total : total = 550)
  (h_jonathan_matthew : jonathan_matthew = 350)
  (h_matthew_zac : matthew_zac = 250) :
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200 := by
  sorry

end strawberry_picker_l2547_254713


namespace simple_interest_calculation_l2547_254751

/-- Simple interest calculation -/
theorem simple_interest_calculation (principal interest_rate simple_interest : ℚ) : 
  principal = 8 →
  interest_rate = 5 / 100 →
  simple_interest = 4.8 →
  ∃ (months : ℚ), months = 12 ∧ simple_interest = principal * interest_rate * months :=
by sorry

end simple_interest_calculation_l2547_254751


namespace paper_towel_savings_l2547_254714

theorem paper_towel_savings : 
  let case_price : ℚ := 9
  let individual_price : ℚ := 1
  let rolls_per_case : ℕ := 12
  let case_price_per_roll : ℚ := case_price / rolls_per_case
  let savings_per_roll : ℚ := individual_price - case_price_per_roll
  let percent_savings : ℚ := (savings_per_roll / individual_price) * 100
  percent_savings = 25 := by sorry

end paper_towel_savings_l2547_254714


namespace dandelion_count_l2547_254755

/-- Proves the original number of yellow and white dandelions given the initial and final conditions --/
theorem dandelion_count : ∀ y w : ℕ,
  y + w = 35 →
  y - 2 = 2 * (w - 6) →
  y = 20 ∧ w = 15 := by
  sorry

end dandelion_count_l2547_254755


namespace variable_order_l2547_254790

theorem variable_order (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : 
  c > a ∧ a > b ∧ b > d := by
  sorry

end variable_order_l2547_254790


namespace area_of_triangle_l2547_254720

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := Real.sqrt 7
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the angle between PF₁ and PF₂
def angle_F₁PF₂ (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let v₁ := (F₁.1 - P.1, F₁.2 - P.2)
  let v₂ := (F₂.1 - P.1, F₂.2 - P.2)
  let cos_angle := (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2))
  cos_angle = 1/2  -- cos 60° = 1/2

-- Theorem statement
theorem area_of_triangle (P F₁ F₂ : ℝ × ℝ) :
  point_on_hyperbola P →
  foci F₁ F₂ →
  angle_F₁PF₂ P F₁ F₂ →
  let a := Real.sqrt ((F₁.1 - P.1)^2 + (F₁.2 - P.2)^2)
  let b := Real.sqrt ((F₂.1 - P.1)^2 + (F₂.2 - P.2)^2)
  let s := (a + b + 2 * Real.sqrt 7) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - 2 * Real.sqrt 7)) = 3 * Real.sqrt 3 :=
sorry

end area_of_triangle_l2547_254720


namespace largest_c_for_negative_two_in_range_l2547_254717

/-- The function f(x) defined as x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The theorem stating that the largest value of c for which -2 is in the range of f is 2 -/
theorem largest_c_for_negative_two_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -2) ↔ c ≤ 2 :=
sorry

end largest_c_for_negative_two_in_range_l2547_254717


namespace lowella_score_l2547_254728

/-- Given a 100-item exam, prove that Lowella's score is 22% when:
    - Pamela's score is 20 percentage points higher than Lowella's
    - Mandy's score is twice Pamela's score
    - Mandy's score is 84% -/
theorem lowella_score (pamela_score mandy_score lowella_score : ℚ) : 
  pamela_score = lowella_score + 20 →
  mandy_score = 2 * pamela_score →
  mandy_score = 84 →
  lowella_score = 22 := by
  sorry

#check lowella_score

end lowella_score_l2547_254728


namespace quadratic_roots_relation_l2547_254745

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ ∃ y, y^2 + p*y + m = 0 ∧ x = 3*y) →
  n / p = 27 := by
sorry

end quadratic_roots_relation_l2547_254745


namespace sofia_card_theorem_l2547_254736

theorem sofia_card_theorem (y : Real) (h1 : 0 < y) (h2 : y < Real.pi / 2) 
  (h3 : Real.tan y > Real.sin y) (h4 : Real.tan y > Real.cos y) : y = Real.pi / 4 :=
by sorry

end sofia_card_theorem_l2547_254736


namespace inequality_relations_l2547_254715

theorem inequality_relations :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a^2 > b^2 → |a| > |b|) ∧
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) :=
sorry

end inequality_relations_l2547_254715


namespace delta_k_zero_iff_ge_four_l2547_254735

def u (n : ℕ) : ℕ := n^3 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => λ n => f (n + 1) - f n

def Δk : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ Δk k

theorem delta_k_zero_iff_ge_four (k : ℕ) :
  (∀ n, Δk k u n = 0) ↔ k ≥ 4 := by sorry

end delta_k_zero_iff_ge_four_l2547_254735


namespace oranges_used_proof_l2547_254785

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 30 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ) 
  (h1 : oranges_per_glass = 3) 
  (h2 : glasses = 10) : 
  total_oranges oranges_per_glass glasses = 30 := by
sorry

end oranges_used_proof_l2547_254785


namespace fifth_point_coordinate_l2547_254784

/-- A sequence of 16 numbers where each number (except the first and last) is the average of its two adjacent numbers -/
def ArithmeticSequence (a : Fin 16 → ℝ) : Prop :=
  a 0 = 2 ∧ 
  a 15 = 47 ∧ 
  ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2

theorem fifth_point_coordinate (a : Fin 16 → ℝ) (h : ArithmeticSequence a) : a 4 = 14 := by
  sorry

end fifth_point_coordinate_l2547_254784


namespace prob_no_dessert_is_35_percent_l2547_254757

/-- Represents the probability of different order combinations -/
structure OrderProbabilities where
  dessert_coffee : ℝ
  dessert_only : ℝ
  coffee_only : ℝ
  appetizer_dessert : ℝ
  appetizer_coffee : ℝ
  appetizer_dessert_coffee : ℝ

/-- Calculate the probability of not ordering dessert -/
def prob_no_dessert (p : OrderProbabilities) : ℝ :=
  1 - (p.dessert_coffee + p.dessert_only + p.appetizer_dessert + p.appetizer_dessert_coffee)

/-- Theorem: The probability of not ordering dessert is 35% -/
theorem prob_no_dessert_is_35_percent (p : OrderProbabilities)
  (h1 : p.dessert_coffee = 0.60)
  (h2 : p.dessert_only = 0.15)
  (h3 : p.coffee_only = 0.10)
  (h4 : p.appetizer_dessert = 0.05)
  (h5 : p.appetizer_coffee = 0.08)
  (h6 : p.appetizer_dessert_coffee = 0.03) :
  prob_no_dessert p = 0.35 := by
  sorry

end prob_no_dessert_is_35_percent_l2547_254757


namespace smallest_solution_of_equation_l2547_254727

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ y : ℝ, 1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
by sorry

end smallest_solution_of_equation_l2547_254727


namespace bakery_pie_distribution_l2547_254763

theorem bakery_pie_distribution (initial_pie : ℚ) (additional_percentage : ℚ) (num_employees : ℕ) :
  initial_pie = 8/9 →
  additional_percentage = 1/10 →
  num_employees = 4 →
  (initial_pie + initial_pie * additional_percentage) / num_employees = 11/45 := by
  sorry

end bakery_pie_distribution_l2547_254763


namespace joans_kittens_l2547_254778

/-- Represents the number of kittens Joan gave to her friends -/
def kittens_given_away (initial_kittens current_kittens : ℕ) : ℕ :=
  initial_kittens - current_kittens

/-- Proves that Joan gave away 2 kittens -/
theorem joans_kittens : kittens_given_away 8 6 = 2 := by
  sorry

end joans_kittens_l2547_254778


namespace no_playful_numbers_l2547_254798

/-- A two-digit positive integer is playful if it equals the sum of the cube of its tens digit and the square of its units digit. -/
def IsPlayful (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a^3 + b^2

/-- The number of playful two-digit positive integers is zero. -/
theorem no_playful_numbers : ∀ n : ℕ, ¬(IsPlayful n) := by sorry

end no_playful_numbers_l2547_254798


namespace set_a_condition_l2547_254722

theorem set_a_condition (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≤ 1 := by
sorry

end set_a_condition_l2547_254722


namespace number_of_divisors_of_60_l2547_254701

theorem number_of_divisors_of_60 : Nat.card (Nat.divisors 60) = 12 := by
  sorry

end number_of_divisors_of_60_l2547_254701


namespace prob_between_30_and_40_l2547_254772

/-- Represents the age groups in the population -/
inductive AgeGroup
  | LessThan20
  | Between20And30
  | Between30And40
  | MoreThan40

/-- Represents the population with their age distribution -/
structure Population where
  total : ℕ
  ageDist : AgeGroup → ℕ
  sum_eq_total : (ageDist AgeGroup.LessThan20) + (ageDist AgeGroup.Between20And30) + 
                 (ageDist AgeGroup.Between30And40) + (ageDist AgeGroup.MoreThan40) = total

/-- The probability of selecting a person from a specific age group -/
def prob (p : Population) (ag : AgeGroup) : ℚ :=
  (p.ageDist ag : ℚ) / (p.total : ℚ)

/-- The given population -/
def givenPopulation : Population where
  total := 200
  ageDist := fun
    | AgeGroup.LessThan20 => 20
    | AgeGroup.Between20And30 => 30
    | AgeGroup.Between30And40 => 70
    | AgeGroup.MoreThan40 => 80
  sum_eq_total := by sorry

theorem prob_between_30_and_40 : 
  prob givenPopulation AgeGroup.Between30And40 = 7 / 20 := by sorry

end prob_between_30_and_40_l2547_254772


namespace chocolate_discount_l2547_254766

/-- Calculates the discount amount given the original price and final price -/
def discount (original_price final_price : ℚ) : ℚ :=
  original_price - final_price

/-- Proves that the discount on a chocolate with original price $2.00 and final price $1.43 is $0.57 -/
theorem chocolate_discount :
  let original_price : ℚ := 2
  let final_price : ℚ := 143/100
  discount original_price final_price = 57/100 := by
sorry

end chocolate_discount_l2547_254766


namespace mishas_current_dollars_l2547_254764

theorem mishas_current_dollars (current_dollars target_dollars needed_dollars : ℕ) 
  (h1 : target_dollars = 47)
  (h2 : needed_dollars = 13)
  (h3 : current_dollars + needed_dollars = target_dollars) :
  current_dollars = 34 := by
  sorry

end mishas_current_dollars_l2547_254764


namespace marbles_distribution_l2547_254792

theorem marbles_distribution (n : ℕ) (initial_marbles : ℕ) : 
  n = 12 ∧ initial_marbles = 50 → 
  (n * (n + 1)) / 2 - initial_marbles = 28 := by
sorry

end marbles_distribution_l2547_254792


namespace thread_length_calculation_l2547_254739

/-- The total length of thread required given an original length and an additional fraction -/
def total_length (original : ℝ) (additional_fraction : ℝ) : ℝ :=
  original + original * additional_fraction

/-- Theorem: Given a 12 cm thread and an additional three-quarters requirement, the total length is 21 cm -/
theorem thread_length_calculation : total_length 12 (3/4) = 21 := by
  sorry

end thread_length_calculation_l2547_254739


namespace normal_dist_peak_l2547_254780

/-- A normal distribution with probability 0.5 of falling within the interval (0.2, +∞) -/
structure NormalDist where
  pdf : ℝ → ℝ
  cdf : ℝ → ℝ
  right_tail_prob : cdf 0.2 = 0.5

/-- The peak of the probability density function occurs at x = 0.2 -/
theorem normal_dist_peak (d : NormalDist) : 
  ∀ x : ℝ, d.pdf x ≤ d.pdf 0.2 :=
sorry

end normal_dist_peak_l2547_254780


namespace skew_parallel_relationship_l2547_254760

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- But for simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction vector
  sorry

-- Define what it means for two lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_parallel_relationship (a b c : Line3D) :
  are_skew a b → are_parallel a c → (are_skew c b ∨ do_intersect c b) :=
by
  sorry

end skew_parallel_relationship_l2547_254760


namespace tangent_line_y_intercept_l2547_254741

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 12 * Real.sqrt 17 / 17 := by
  sorry

end tangent_line_y_intercept_l2547_254741


namespace cricket_team_size_l2547_254768

/-- The number of players on a cricket team satisfying specific conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers non_throwers right_handed : ℕ),
    throwers = 37 →
    non_throwers = total_players - throwers →
    right_handed = 51 →
    right_handed = throwers + (2 * non_throwers / 3) →
    total_players = 58 := by
  sorry

end cricket_team_size_l2547_254768


namespace min_value_expression_l2547_254788

theorem min_value_expression (x y z k : ℝ) 
  (hx : -2 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 2) 
  (hz : -2 < z ∧ z < 2) 
  (hk : k > 0) :
  (k / ((2 - x) * (2 - y) * (2 - z))) + (k / ((2 + x) * (2 + y) * (2 + z))) ≥ 2 * k :=
sorry

end min_value_expression_l2547_254788


namespace power_multiplication_l2547_254711

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_multiplication_l2547_254711


namespace parabola_tangents_perpendicular_iff_P_on_line_l2547_254789

/-- Parabola C: x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l: y = -1 -/
def line (x y : ℝ) : Prop := y = -1

/-- Point P is on line l -/
def P_on_line (P : ℝ × ℝ) : Prop := line P.1 P.2

/-- PA and PB are perpendicular -/
def tangents_perpendicular (P A B : ℝ × ℝ) : Prop :=
  let slope_PA := (A.2 - P.2) / (A.1 - P.1)
  let slope_PB := (B.2 - P.2) / (B.1 - P.1)
  slope_PA * slope_PB = -1

/-- A and B are points on the parabola -/
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

/-- PA and PB are tangent to the parabola at A and B respectively -/
def tangent_lines (P A B : ℝ × ℝ) : Prop :=
  points_on_parabola A B ∧
  (A.2 - P.2) / (A.1 - P.1) = A.1 / 2 ∧
  (B.2 - P.2) / (B.1 - P.1) = B.1 / 2

theorem parabola_tangents_perpendicular_iff_P_on_line
  (P A B : ℝ × ℝ) :
  tangent_lines P A B →
  (P_on_line P ↔ tangents_perpendicular P A B) :=
sorry

end parabola_tangents_perpendicular_iff_P_on_line_l2547_254789


namespace probability_sum_less_than_product_l2547_254702

def valid_pairs : Finset (ℕ × ℕ) :=
  (Finset.range 6).product (Finset.range 6)

def satisfying_pairs : Finset (ℕ × ℕ) :=
  valid_pairs.filter (fun p => p.1 + p.2 < p.1 * p.2)

theorem probability_sum_less_than_product :
  (satisfying_pairs.card : ℚ) / valid_pairs.card = 2 / 3 := by
  sorry

end probability_sum_less_than_product_l2547_254702


namespace markus_more_marbles_l2547_254781

theorem markus_more_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) 
  (h1 : mara_bags = 12) (h2 : mara_marbles_per_bag = 2) 
  (h3 : markus_bags = 2) (h4 : markus_marbles_per_bag = 13) : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end markus_more_marbles_l2547_254781


namespace sin_15_75_simplification_l2547_254779

theorem sin_15_75_simplification : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end sin_15_75_simplification_l2547_254779


namespace system1_neither_necessary_nor_sufficient_l2547_254770

-- Define the two systems of inequalities
def system1 (x y a b : ℝ) : Prop := x > a ∧ y > b
def system2 (x y a b : ℝ) : Prop := x + y > a + b ∧ x * y > a * b

-- Theorem stating that system1 is neither necessary nor sufficient for system2
theorem system1_neither_necessary_nor_sufficient :
  ¬(∀ x y a b : ℝ, system1 x y a b → system2 x y a b) ∧
  ¬(∀ x y a b : ℝ, system2 x y a b → system1 x y a b) :=
sorry

end system1_neither_necessary_nor_sufficient_l2547_254770


namespace chris_jogging_time_l2547_254732

/-- Represents the time in minutes -/
def Time := ℝ

/-- Represents the distance in miles -/
def Distance := ℝ

/-- Chris's jogging rate in minutes per mile -/
def chris_rate : ℝ := sorry

/-- Alex's walking rate in minutes per mile -/
def alex_rate : ℝ := sorry

theorem chris_jogging_time 
  (h1 : chris_rate * 4 = 2 * alex_rate * 2)  -- Chris's 4-mile time is twice Alex's 2-mile time
  (h2 : alex_rate * 2 = 40)                  -- Alex's 2-mile time is 40 minutes
  : chris_rate * 6 = 120 :=                  -- Chris's 6-mile time is 120 minutes
sorry

end chris_jogging_time_l2547_254732


namespace A_equals_B_l2547_254769

/-- The number of digits written when listing integers from 1 to 10^(n-1) -/
def A (n : ℕ) : ℕ := sorry

/-- The number of zeros written when listing integers from 1 to 10^n -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A(n) equals B(n) for all positive integers n -/
theorem A_equals_B (n : ℕ) (h : n > 0) : A n = B n := by sorry

end A_equals_B_l2547_254769


namespace ibrahim_palace_count_l2547_254797

/-- Represents a square grid of rooms -/
structure RoomGrid where
  size : Nat
  has_door_between_rooms : Bool
  has_window_on_outer_wall : Bool

/-- Calculates the number of windows in the grid -/
def count_windows (grid : RoomGrid) : Nat :=
  if grid.has_window_on_outer_wall then
    4 * grid.size
  else
    0

/-- Calculates the number of doors in the grid -/
def count_doors (grid : RoomGrid) : Nat :=
  if grid.has_door_between_rooms then
    2 * grid.size * (grid.size - 1)
  else
    0

/-- Theorem stating the number of windows and doors in the specific 10x10 grid -/
theorem ibrahim_palace_count (grid : RoomGrid)
  (h_size : grid.size = 10)
  (h_door : grid.has_door_between_rooms = true)
  (h_window : grid.has_window_on_outer_wall = true) :
  count_windows grid = 40 ∧ count_doors grid = 180 := by
  sorry


end ibrahim_palace_count_l2547_254797


namespace quadratic_function_property_l2547_254723

/-- A quadratic function of the form y = 3(x - a)² -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := 3 * (x - a)^2

/-- The property that y increases as x increases when x > 2 -/
def increasing_when_x_gt_2 (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → quadratic_function a x₂ > quadratic_function a x₁

theorem quadratic_function_property (a : ℝ) :
  increasing_when_x_gt_2 a → a ≤ 2 := by sorry

end quadratic_function_property_l2547_254723


namespace a_plus_b_value_l2547_254750

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |a| = 4)
  (h2 : Real.sqrt (b^2) = 3)
  (h3 : a + b > 0) :
  a + b = 1 ∨ a + b = 7 := by
  sorry

end a_plus_b_value_l2547_254750


namespace smallest_k_for_cosine_equation_l2547_254787

theorem smallest_k_for_cosine_equation :
  let f : ℕ → Prop := λ k => Real.cos (k^2 + 8^2 : ℝ)^2 = 1
  ∃ (k₁ k₂ : ℕ), k₁ < k₂ ∧ f k₁ ∧ f k₂ ∧ k₁ = 10 ∧ k₂ = 12 ∧
    ∀ (k : ℕ), 0 < k ∧ k < k₁ → ¬f k :=
by sorry

end smallest_k_for_cosine_equation_l2547_254787


namespace distance_between_foci_l2547_254774

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 18

-- Define the foci
def focus1 : ℝ × ℝ := (2, 3)
def focus2 : ℝ × ℝ := (-4, 5)

-- Theorem statement
theorem distance_between_foci :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = 2 * Real.sqrt 10 := by
  sorry

end distance_between_foci_l2547_254774


namespace art_club_theorem_l2547_254782

/-- Represents the distribution of students in a school's clubs -/
structure SchoolClubs where
  total_students : ℕ
  music_students : ℕ
  recitation_offset : ℕ
  dance_offset : ℤ

/-- Calculates the number of students in the art club -/
def art_club_students (sc : SchoolClubs) : ℤ :=
  sc.total_students - sc.music_students - (sc.music_students / 2 + sc.recitation_offset) - 
  (sc.music_students + 2 * sc.recitation_offset + sc.dance_offset)

/-- Theorem stating the number of students in the art club -/
theorem art_club_theorem (sc : SchoolClubs) 
  (h1 : sc.total_students = 220)
  (h2 : sc.dance_offset = -40) :
  art_club_students sc = 260 - (5/2 : ℚ) * sc.music_students - 3 * sc.recitation_offset :=
by sorry

end art_club_theorem_l2547_254782


namespace count_with_3_or_6_in_base_7_eq_1776_l2547_254703

/-- The count of integers among the first 2401 positive integers in base 7 that use 3 or 6 as a digit -/
def count_with_3_or_6_in_base_7 : ℕ :=
  2401 - 5^4

theorem count_with_3_or_6_in_base_7_eq_1776 :
  count_with_3_or_6_in_base_7 = 1776 := by sorry

end count_with_3_or_6_in_base_7_eq_1776_l2547_254703


namespace largest_intersection_point_l2547_254730

-- Define the polynomial P(x)
def P (x b : ℝ) : ℝ := x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3

-- Define the line L(x)
def L (x c d : ℝ) : ℝ := c*x - d

-- Theorem statement
theorem largest_intersection_point (b c d : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, P x b = L x c d ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max ∧ x_max = 6) :=
by
  sorry


end largest_intersection_point_l2547_254730


namespace quadratic_root_l2547_254775

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 9*a - 3*b + c = 0) : 
  a*(-3)^2 + b*(-3) + c = 0 :=
sorry

end quadratic_root_l2547_254775


namespace number_equation_l2547_254747

theorem number_equation : ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 := by
  sorry

end number_equation_l2547_254747


namespace twelve_people_in_line_l2547_254799

/-- The number of people in a line with Jeanne, given the number of people in front and behind her -/
def people_in_line (people_in_front : ℕ) (people_behind : ℕ) : ℕ :=
  people_in_front + 1 + people_behind

/-- Theorem stating that there are 12 people in the line -/
theorem twelve_people_in_line :
  people_in_line 4 7 = 12 := by
  sorry

#check twelve_people_in_line

end twelve_people_in_line_l2547_254799


namespace intersection_point_correct_l2547_254740

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The first line: y = 3x + 4 -/
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 4

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The point through which the perpendicular line passes -/
def point : (ℚ × ℚ) := (3, 2)

/-- The perpendicular line passing through (3, 2) -/
def line₂ (x y : ℚ) : Prop := y - point.2 = m₂ * (x - point.1)

/-- The intersection point of the two lines -/
def intersection_point : (ℚ × ℚ) := (-3/10, 31/10)

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 :=
sorry

end intersection_point_correct_l2547_254740


namespace marks_fruit_consumption_l2547_254738

/-- Given that Mark had 10 pieces of fruit for the week, kept 2 for next week,
    and brought 3 to school on Friday, prove that he ate 5 pieces in the first four days. -/
theorem marks_fruit_consumption
  (total_fruit : ℕ)
  (kept_for_next_week : ℕ)
  (brought_to_school : ℕ)
  (h1 : total_fruit = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_to_school = 3) :
  total_fruit - kept_for_next_week - brought_to_school = 5 := by
  sorry

end marks_fruit_consumption_l2547_254738


namespace smallest_k_for_omega_inequality_l2547_254773

/-- ω(n) denotes the number of positive prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The theorem states that 5 is the smallest positive integer k 
    such that 2^ω(n) ≤ k∙n^(1/4) for all positive integers n -/
theorem smallest_k_for_omega_inequality : 
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ k * (n : ℝ)^(1/4)) ∧ 
  (∀ k : ℕ, 0 < k → k < 5 → ∃ n : ℕ, n > 0 ∧ (2 : ℝ)^(omega n : ℝ) > k * (n : ℝ)^(1/4)) ∧
  (∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ 5 * (n : ℝ)^(1/4)) :=
sorry

end smallest_k_for_omega_inequality_l2547_254773


namespace sum_of_cubes_l2547_254756

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (sum_prod_eq : x*y + y*z + z*x = -3) 
  (prod_eq : x*y*z = 2) : 
  x^3 + y^3 + z^3 = 32 := by
sorry

end sum_of_cubes_l2547_254756


namespace janet_practice_days_l2547_254771

def total_miles : ℕ := 72
def miles_per_day : ℕ := 8

theorem janet_practice_days : 
  total_miles / miles_per_day = 9 := by sorry

end janet_practice_days_l2547_254771


namespace johns_allowance_l2547_254749

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 92/100 = A) →
  A = 345/100 := by
sorry

end johns_allowance_l2547_254749


namespace exists_valid_coloring_l2547_254719

/-- A coloring of the edges of a complete graph on 6 vertices -/
def Coloring := Fin 6 → Fin 6 → Fin 5

/-- A valid coloring ensures that each vertex has exactly one edge of each color -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v : Fin 6, ∀ color : Fin 5,
    ∃! w : Fin 6, w ≠ v ∧ c v w = color

/-- There exists a valid coloring of the complete graph K₆ using 5 colors -/
theorem exists_valid_coloring : ∃ c : Coloring, is_valid_coloring c := by
  sorry

end exists_valid_coloring_l2547_254719


namespace miscalculation_correction_l2547_254748

theorem miscalculation_correction (x : ℝ) : 
  63 + x = 69 → 36 / x = 6 := by sorry

end miscalculation_correction_l2547_254748


namespace rock_paper_scissors_probabilities_l2547_254709

-- Define the game structure
structure RockPaperScissors where
  players : Finset Char := {'A', 'B', 'C'}

-- Define the probability of winning in a single throw
def win_prob : ℚ := 1 / 3

-- Define the probability of a tie in a single throw
def tie_prob : ℚ := 1 / 3

-- Define the probability that A wins against B with no more than two throws
def prob_A_wins_B (game : RockPaperScissors) : ℚ := sorry

-- Define the probability that C will treat after two throws
def prob_C_treats (game : RockPaperScissors) : ℚ := sorry

-- Define the probability that exactly two days out of three C will treat after two throws
def prob_C_treats_two_days (game : RockPaperScissors) : ℚ := sorry

theorem rock_paper_scissors_probabilities (game : RockPaperScissors) :
  prob_A_wins_B game = 4 / 9 ∧
  prob_C_treats game = 2 / 9 ∧
  prob_C_treats_two_days game = 28 / 243 := by sorry

end rock_paper_scissors_probabilities_l2547_254709


namespace special_function_value_l2547_254753

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 3) ≤ f x + 3) ∧
  (∀ x, f (x + 2) ≥ f x + 2) ∧
  (f 4 = 2008)

/-- The theorem to be proved -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) : f 2008 = 4012 := by
  sorry

end special_function_value_l2547_254753


namespace ellipse_vector_dot_product_range_l2547_254708

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from M to a point -/
def vector_MA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - M.1, A.2 - M.2)

theorem ellipse_vector_dot_product_range :
  ∀ A B : ℝ × ℝ,
  on_ellipse A.1 A.2 →
  on_ellipse B.1 B.2 →
  dot_product (vector_MA A) (vector_MA B) = 0 →
  ∃ x : ℝ, x = dot_product (vector_MA A) (A.1 - B.1, A.2 - B.2) ∧
           2/3 ≤ x ∧ x ≤ 9 :=
sorry

end ellipse_vector_dot_product_range_l2547_254708


namespace triangle_properties_l2547_254761

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  b + a * Real.cos C = 0 →
  Real.sin A = 2 * Real.sin (A + C) →
  C = 2 * π / 3 ∧ c / a = Real.sqrt 2 := by
  sorry

end triangle_properties_l2547_254761


namespace expression_simplification_l2547_254716

theorem expression_simplification (x y : ℝ) (hx : x = -3) (hy : y = -1) :
  (-3 * x^2 - 4*y) - (2 * x^2 - 5*y + 6) + (x^2 - 5*y - 1) = -39 := by
  sorry

end expression_simplification_l2547_254716


namespace new_person_weight_l2547_254704

theorem new_person_weight (initial_count : ℕ) (initial_average : ℝ) (weight_decrease : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  weight_decrease = 5 →
  let total_weight := initial_count * initial_average
  let new_count := initial_count + 1
  let new_average := initial_average - weight_decrease
  let new_person_weight := new_count * new_average - total_weight
  new_person_weight = 55 := by
sorry

end new_person_weight_l2547_254704


namespace mixed_fraction_product_l2547_254746

theorem mixed_fraction_product (X Y : ℕ) : 
  (X > 0) →
  (Y > 0) →
  (5 : ℚ) + 1 / X > 5 →
  (5 : ℚ) + 1 / X ≤ 11 / 2 →
  (5 + 1 / X) * (Y + 1 / 2) = 43 →
  X = 17 ∧ Y = 8 := by
sorry

end mixed_fraction_product_l2547_254746


namespace square_dissection_existence_l2547_254734

theorem square_dissection_existence :
  ∃ (S a b c : ℝ), 
    S > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    S^2 = a^2 + 3*b^2 + 5*c^2 :=
by sorry

end square_dissection_existence_l2547_254734


namespace stating_convex_polygon_decomposition_iff_centrally_symmetric_l2547_254765

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A parallelogram. -/
structure Parallelogram where
  -- Add necessary fields for a parallelogram

/-- Represents a decomposition of a polygon into parallelograms. -/
def Decomposition (p : ConvexPolygon) := List Parallelogram

/-- Checks if a decomposition is valid for a given polygon. -/
def is_valid_decomposition (p : ConvexPolygon) (d : Decomposition p) : Prop :=
  sorry

/-- Checks if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : ConvexPolygon) : Prop :=
  sorry

/-- 
Theorem stating that a convex polygon can be decomposed into a finite number of parallelograms 
if and only if it is centrally symmetric.
-/
theorem convex_polygon_decomposition_iff_centrally_symmetric (p : ConvexPolygon) :
  (∃ d : Decomposition p, is_valid_decomposition p d) ↔ is_centrally_symmetric p :=
sorry

end stating_convex_polygon_decomposition_iff_centrally_symmetric_l2547_254765


namespace henrys_game_purchase_l2547_254700

/-- Henry's money problem -/
theorem henrys_game_purchase (initial : ℕ) (birthday_gift : ℕ) (final : ℕ) 
  (h1 : initial = 11)
  (h2 : birthday_gift = 18)
  (h3 : final = 19) :
  initial + birthday_gift - final = 10 := by
  sorry

end henrys_game_purchase_l2547_254700


namespace roots_of_equation_l2547_254725

theorem roots_of_equation (x : ℝ) : 
  (3 * Real.sqrt x + 3 / Real.sqrt x = 7) ↔ 
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end roots_of_equation_l2547_254725


namespace intersection_with_complement_l2547_254762

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by
  sorry

end intersection_with_complement_l2547_254762


namespace combined_age_is_28_l2547_254777

/-- Represents the ages of Michael and his brothers -/
structure FamilyAges where
  michael : ℕ
  younger_brother : ℕ
  older_brother : ℕ

/-- Defines the conditions for the ages of Michael and his brothers -/
def valid_ages (ages : FamilyAges) : Prop :=
  ages.younger_brother = 5 ∧
  ages.older_brother = 3 * ages.younger_brother ∧
  ages.older_brother = 2 * (ages.michael - 1) + 1

/-- Theorem stating that the combined age of Michael and his brothers is 28 years -/
theorem combined_age_is_28 (ages : FamilyAges) (h : valid_ages ages) :
  ages.michael + ages.younger_brother + ages.older_brother = 28 := by
  sorry


end combined_age_is_28_l2547_254777


namespace cubic_expression_evaluation_l2547_254729

theorem cubic_expression_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end cubic_expression_evaluation_l2547_254729


namespace hotel_beds_count_l2547_254744

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
    (h1 : total_rooms = 13)
    (h2 : two_bed_rooms = 8)
    (h3 : beds_in_two_bed_room = 2)
    (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 := by
  sorry

#eval 8 * 2 + (13 - 8) * 3  -- This should output 31

end hotel_beds_count_l2547_254744


namespace power_last_digit_match_l2547_254793

theorem power_last_digit_match : ∃ (m n : ℕ), 
  100 ≤ 2^m ∧ 2^m < 1000 ∧ 
  100 ≤ 3^n ∧ 3^n < 1000 ∧ 
  2^m % 10 = 3^n % 10 ∧ 
  2^m % 10 = 3 := by
sorry

end power_last_digit_match_l2547_254793


namespace deck_size_l2547_254758

theorem deck_size (toothpicks_per_card : ℕ) (unused_cards : ℕ) (boxes : ℕ) (toothpicks_per_box : ℕ) :
  toothpicks_per_card = 75 →
  unused_cards = 16 →
  boxes = 6 →
  toothpicks_per_box = 450 →
  boxes * toothpicks_per_box / toothpicks_per_card + unused_cards = 52 :=
by
  sorry

end deck_size_l2547_254758


namespace geometric_sequence_inequality_l2547_254794

theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q ≠ 1 →  -- common ratio is not 1
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end geometric_sequence_inequality_l2547_254794


namespace parabola_intersection_ratio_l2547_254731

/-- Two parabolas with given properties -/
structure ParabolaPair where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = a * x₁^2 + b * x₁ + c  -- Vertex condition for N₁
  h₂ : y₂ = -a * x₂^2 + d * x₂ + e  -- Vertex condition for N₂
  h₃ : 21 = a * 12^2 + b * 12 + c  -- A(12, 21) lies on N₁
  h₄ : 3 = a * 28^2 + b * 28 + c  -- B(28, 3) lies on N₁
  h₅ : 21 = -a * 12^2 + d * 12 + e  -- A(12, 21) lies on N₂
  h₆ : 3 = -a * 28^2 + d * 28 + e  -- B(28, 3) lies on N₂

/-- The main theorem -/
theorem parabola_intersection_ratio (p : ParabolaPair) :
  (p.x₁ + p.x₂) / (p.y₁ + p.y₂) = 5 / 3 := by sorry

end parabola_intersection_ratio_l2547_254731


namespace cubelets_one_color_count_l2547_254737

/-- Represents a cube divided into cubelets -/
structure CubeletCube where
  size : Nat
  total_cubelets : Nat
  painted_faces : Fin 3 → Fin 6

/-- The number of cubelets painted with exactly one color -/
def cubelets_with_one_color (c : CubeletCube) : Nat :=
  6 * (c.size - 2) * (c.size - 2)

/-- Theorem: In a 6x6x6 cube painted as described, 96 cubelets are painted with exactly one color -/
theorem cubelets_one_color_count :
  ∀ c : CubeletCube, c.size = 6 → c.total_cubelets = 216 → cubelets_with_one_color c = 96 := by
  sorry

end cubelets_one_color_count_l2547_254737


namespace grape_sales_profit_l2547_254795

/-- Profit function for grape sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 510 * x - 7500

/-- Theorem stating the properties of the profit function -/
theorem grape_sales_profit :
  let w := profit_function
  (w 28 = 1040) ∧
  (∀ x, w x ≤ w (51/2)) ∧
  (w (51/2) = 1102.5) := by
  sorry

end grape_sales_profit_l2547_254795


namespace find_k_l2547_254733

def is_max_solution (k : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5 → x ≤ 3

theorem find_k : ∃! k : ℝ, is_max_solution k ∧ k = 8 := by sorry

end find_k_l2547_254733


namespace log_ratio_independence_l2547_254712

theorem log_ratio_independence (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) : 
  (Real.log P / Real.log a) / (Real.log K / Real.log a) = 
  (Real.log P / Real.log b) / (Real.log K / Real.log b) := by
  sorry

end log_ratio_independence_l2547_254712


namespace average_first_five_multiples_of_five_l2547_254742

/-- The average of the first 5 multiples of 5 is 15 -/
theorem average_first_five_multiples_of_five : 
  (List.sum (List.map (· * 5) (List.range 5))) / 5 = 15 := by
  sorry

end average_first_five_multiples_of_five_l2547_254742
