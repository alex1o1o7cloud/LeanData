import Mathlib

namespace NUMINAMATH_CALUDE_courier_speed_impossibility_l1002_100268

/-- Proves the impossibility of achieving a specific average speed given certain conditions -/
theorem courier_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_avg_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_avg_speed = 12 →
  ¬∃ (remaining_speed : ℝ),
    remaining_speed > 0 ∧
    (2/3 * total_distance / initial_speed + 1/3 * total_distance / remaining_speed) = (total_distance / target_avg_speed) :=
by sorry

end NUMINAMATH_CALUDE_courier_speed_impossibility_l1002_100268


namespace NUMINAMATH_CALUDE_distance_covered_l1002_100206

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) : 
  time_minutes = 42 → speed_km_per_hour = 10 → 
  (time_minutes / 60) * speed_km_per_hour = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_covered_l1002_100206


namespace NUMINAMATH_CALUDE_tangent_perpendicular_range_l1002_100285

theorem tangent_perpendicular_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2*x - a + 1/x = 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_range_l1002_100285


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1002_100279

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 2) :
  ∀ x < 0, f x = -x - 2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1002_100279


namespace NUMINAMATH_CALUDE_solve_for_a_l1002_100209

theorem solve_for_a (a b d : ℤ) 
  (eq1 : a + b = d) 
  (eq2 : b + d = 7) 
  (eq3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l1002_100209


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1002_100287

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (1 + Real.sqrt 17) / 4
  let x₂ : ℝ := (1 - Real.sqrt 17) / 4
  ∀ x : ℝ, 2 * x^2 - x = 2 ↔ (x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1002_100287


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l1002_100207

/-- The number of popsicles Megan eats in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes / minutes_per_popsicle : ℕ)

/-- Converts hours and minutes to total minutes -/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem megan_popsicle_consumption :
  popsicles_eaten 12 (to_minutes 6 45) = 33 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l1002_100207


namespace NUMINAMATH_CALUDE_integer_properties_l1002_100221

theorem integer_properties (m n k : ℕ) (hm : m > 0) (hn : n > 0) : 
  ∃ (a b : ℕ), 
    -- (m+n)^2 + (m-n)^2 is even
    ∃ (c : ℕ), (m + n)^2 + (m - n)^2 = 2 * c ∧
    -- ((m+n)^2 + (m-n)^2) / 2 can be expressed as the sum of squares of two positive integers
    ((m + n)^2 + (m - n)^2) / 2 = a^2 + b^2 ∧
    -- For any integer k, (2k+1)^2 - (2k-1)^2 is divisible by 8
    ∃ (d : ℕ), (2 * k + 1)^2 - (2 * k - 1)^2 = 8 * d :=
by sorry

end NUMINAMATH_CALUDE_integer_properties_l1002_100221


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l1002_100267

theorem cubic_roots_inequality (A B C : ℝ) 
  (h : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
       ∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ (x = a ∨ x = b ∨ x = c)) :
  A^2 + B^2 + 18*C > 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l1002_100267


namespace NUMINAMATH_CALUDE_inverse_proportion_graph_l1002_100262

/-- Given that point A(2,4) lies on the graph of y = k/x, prove that (4,2) also lies on the graph
    while (-2,4), (2,-4), and (-4,2) do not. -/
theorem inverse_proportion_graph (k : ℝ) (h : k ≠ 0) : 
  (4 : ℝ) = k / 2 →  -- Point A(2,4) lies on the graph
  (2 : ℝ) = k / 4 ∧  -- Point (4,2) lies on the graph
  (4 : ℝ) ≠ k / (-2) ∧  -- Point (-2,4) does not lie on the graph
  (-4 : ℝ) ≠ k / 2 ∧  -- Point (2,-4) does not lie on the graph
  (2 : ℝ) ≠ k / (-4) :=  -- Point (-4,2) does not lie on the graph
by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_graph_l1002_100262


namespace NUMINAMATH_CALUDE_card_selection_counts_l1002_100281

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (card_count : cards.card = 52)

/-- Counts the number of ways to select 4 cards with different suits and ranks -/
def count_four_different (d : Deck) : ℕ := sorry

/-- Counts the number of ways to select 6 cards with all suits represented -/
def count_six_all_suits (d : Deck) : ℕ := sorry

/-- Theorem stating the correct counts for both selections -/
theorem card_selection_counts (d : Deck) : 
  count_four_different d = 17160 ∧ count_six_all_suits d = 8682544 := by sorry

end NUMINAMATH_CALUDE_card_selection_counts_l1002_100281


namespace NUMINAMATH_CALUDE_sine_tangent_relation_l1002_100224

theorem sine_tangent_relation (α : Real) (h : 0 < α ∧ α < Real.pi) :
  (∃ β, (Real.sqrt 2 / 2 < Real.sin β ∧ Real.sin β < 1) ∧ ¬(Real.tan β > 1)) ∧
  (∀ γ, Real.tan γ > 1 → Real.sqrt 2 / 2 < Real.sin γ ∧ Real.sin γ < 1) :=
by sorry

end NUMINAMATH_CALUDE_sine_tangent_relation_l1002_100224


namespace NUMINAMATH_CALUDE_space_station_cost_sharing_l1002_100295

/-- The cost of building a space station in trillions of dollars -/
def space_station_cost : ℝ := 5

/-- The number of people sharing the cost in millions -/
def number_of_people : ℝ := 500

/-- The share of each person in dollars -/
def person_share : ℝ := 10000

theorem space_station_cost_sharing :
  (space_station_cost * 1000000) / number_of_people = person_share := by
  sorry

end NUMINAMATH_CALUDE_space_station_cost_sharing_l1002_100295


namespace NUMINAMATH_CALUDE_S_is_finite_l1002_100266

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Number of positive divisors function -/
def tau (n : ℕ) : ℕ := sorry

/-- The set of positive integers satisfying the inequality -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ phi n * tau n ≥ Real.sqrt (n^3 / 3)}

/-- Theorem stating that S is finite -/
theorem S_is_finite : Set.Finite S := by sorry

end NUMINAMATH_CALUDE_S_is_finite_l1002_100266


namespace NUMINAMATH_CALUDE_equation_always_has_two_solutions_l1002_100293

theorem equation_always_has_two_solutions (b : ℝ) (h : 1 ≤ b ∧ b ≤ 25) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  x₁^4 + 36*b^2 = (9*b^2 - 15*b)*x₁^2 ∧
  x₂^4 + 36*b^2 = (9*b^2 - 15*b)*x₂^2 :=
sorry

end NUMINAMATH_CALUDE_equation_always_has_two_solutions_l1002_100293


namespace NUMINAMATH_CALUDE_range_of_a_l1002_100256

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 2*a*x - 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (a > 0) →
  (∀ x : ℝ, p x a → q x) →
  (∃ x : ℝ, q x ∧ ¬(p x a)) →
  (0 < a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1002_100256


namespace NUMINAMATH_CALUDE_nadia_walked_18_km_l1002_100243

/-- The distance Hannah walked in kilometers -/
def hannah_distance : ℝ := sorry

/-- The distance Nadia walked in kilometers -/
def nadia_distance : ℝ := 2 * hannah_distance

/-- The total distance walked by both girls in kilometers -/
def total_distance : ℝ := 27

theorem nadia_walked_18_km :
  nadia_distance = 18 ∧ hannah_distance + nadia_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_nadia_walked_18_km_l1002_100243


namespace NUMINAMATH_CALUDE_simplify_complex_root_expression_l1002_100208

theorem simplify_complex_root_expression (x : ℝ) (h : x ≥ 0) :
  (6 * x * (5 + 2 * Real.sqrt 6)) ^ (1/4) * Real.sqrt (3 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) = Real.sqrt (6 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_root_expression_l1002_100208


namespace NUMINAMATH_CALUDE_box_surface_area_l1002_100280

/-- Proves that the surface area of a rectangular box is 975 given specific conditions -/
theorem box_surface_area (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25)
  (volume : a * b * c = 600) :
  2 * (a * b + b * c + c * a) = 975 := by
sorry

end NUMINAMATH_CALUDE_box_surface_area_l1002_100280


namespace NUMINAMATH_CALUDE_sufficient_necessary_but_not_sufficient_l1002_100286

-- Define propositions p and q
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def sufficient (p q : Prop) : Prop := p → q

-- Define what it means for p to be a necessary and sufficient condition for q
def necessary_and_sufficient (p q : Prop) : Prop := p ↔ q

-- Theorem stating that "p is a sufficient condition for q" is a necessary but not sufficient condition for "p is a necessary and sufficient condition for q"
theorem sufficient_necessary_but_not_sufficient :
  (∀ p q, necessary_and_sufficient p q → sufficient p q) ∧
  ¬(∀ p q, sufficient p q → necessary_and_sufficient p q) :=
sorry

end NUMINAMATH_CALUDE_sufficient_necessary_but_not_sufficient_l1002_100286


namespace NUMINAMATH_CALUDE_circular_board_holes_l1002_100274

/-- The number of holes on the circular board -/
def n : ℕ := 91

/-- Proposition: The number of holes on the circular board satisfies all conditions -/
theorem circular_board_holes :
  n < 100 ∧
  ∃ k : ℕ, k > 0 ∧ 2 * k ≡ 1 [ZMOD n] ∧
  ∃ m : ℕ, m > 0 ∧ 4 * m ≡ 2 * k [ZMOD n] ∧
  6 ≡ 0 [ZMOD n] :=
by sorry

end NUMINAMATH_CALUDE_circular_board_holes_l1002_100274


namespace NUMINAMATH_CALUDE_b_job_fraction_l1002_100291

/-- The fraction of the job that B completes when A and B work together to finish a job -/
theorem b_job_fraction (a_time b_time : ℝ) (a_solo_time : ℝ) : 
  a_time = 6 →
  b_time = 3 →
  a_solo_time = 1 →
  (25 : ℝ) / 54 = 
    ((1 - a_solo_time / a_time) * (1 / b_time) * 
     (1 - a_solo_time / a_time) / ((1 / a_time) + (1 / b_time))) :=
by sorry

end NUMINAMATH_CALUDE_b_job_fraction_l1002_100291


namespace NUMINAMATH_CALUDE_half_squared_equals_quarter_l1002_100269

theorem half_squared_equals_quarter : (1 / 2 : ℝ) ^ 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_half_squared_equals_quarter_l1002_100269


namespace NUMINAMATH_CALUDE_triangle_4_4_7_l1002_100212

/-- A triangle can be formed from three line segments if the sum of any two sides
    is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 4, and 7 can form a triangle. -/
theorem triangle_4_4_7 :
  can_form_triangle 4 4 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_4_7_l1002_100212


namespace NUMINAMATH_CALUDE_no_solution_functional_equation_l1002_100242

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = 2*x - f y :=
by sorry

end NUMINAMATH_CALUDE_no_solution_functional_equation_l1002_100242


namespace NUMINAMATH_CALUDE_cost_for_three_roofs_is_1215_l1002_100202

/-- Calculates the total cost of materials for building roofs with discounts applied --/
def total_cost_with_discounts (
  num_roofs : ℕ
  ) (
  metal_bars_per_roof : ℕ
  ) (
  wooden_beams_per_roof : ℕ
  ) (
  steel_rods_per_roof : ℕ
  ) (
  bars_per_set : ℕ
  ) (
  beams_per_set : ℕ
  ) (
  rods_per_set : ℕ
  ) (
  cost_per_bar : ℕ
  ) (
  cost_per_beam : ℕ
  ) (
  cost_per_rod : ℕ
  ) (
  discount_threshold : ℕ
  ) (
  discount_rate : ℚ
  ) : ℕ :=
  sorry

/-- Theorem stating that the total cost for building 3 roofs with given specifications is $1215 --/
theorem cost_for_three_roofs_is_1215 :
  total_cost_with_discounts 3 2 3 1 7 5 4 10 15 20 10 (1/10) = 1215 :=
  sorry

end NUMINAMATH_CALUDE_cost_for_three_roofs_is_1215_l1002_100202


namespace NUMINAMATH_CALUDE_book_has_2000_pages_l1002_100288

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to grab lunch (in hours) -/
def lunch_time : ℕ := 4

/-- The time it takes Juan to read the book (in hours) -/
def reading_time : ℕ := 2 * lunch_time

/-- The total number of pages in the book -/
def book_pages : ℕ := pages_per_hour * reading_time

theorem book_has_2000_pages : book_pages = 2000 := by
  sorry

end NUMINAMATH_CALUDE_book_has_2000_pages_l1002_100288


namespace NUMINAMATH_CALUDE_complex_product_example_l1002_100298

theorem complex_product_example : 
  let z₁ : ℂ := -1 + 2 * Complex.I
  let z₂ : ℂ := 2 + Complex.I
  z₁ * z₂ = -4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_example_l1002_100298


namespace NUMINAMATH_CALUDE_prime_power_implies_prime_n_l1002_100203

theorem prime_power_implies_prime_n (n : ℕ) (p : ℕ) (k : ℕ) :
  (∃ (p : ℕ), Prime p ∧ ∃ (k : ℕ), 3^n - 2^n = p^k) →
  Prime n :=
by sorry

end NUMINAMATH_CALUDE_prime_power_implies_prime_n_l1002_100203


namespace NUMINAMATH_CALUDE_brothers_combined_age_l1002_100259

theorem brothers_combined_age : 
  ∀ (x y : ℕ), (x - 6 + y - 6 = 100) → (x + y = 112) :=
by
  sorry

end NUMINAMATH_CALUDE_brothers_combined_age_l1002_100259


namespace NUMINAMATH_CALUDE_sum_of_powers_l1002_100227

theorem sum_of_powers (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a^1 + a^2 + a^3 + a^4 + a^5 + a^6 + a^7 + a^8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1002_100227


namespace NUMINAMATH_CALUDE_removed_term_is_16th_l1002_100231

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ := 2 * n^2 - n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℕ := 4 * k - 3

theorem removed_term_is_16th :
  ∀ k : ℕ,
  (S 21 - a k = 40 * 20) →
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_removed_term_is_16th_l1002_100231


namespace NUMINAMATH_CALUDE_max_x2_plus_y2_l1002_100270

theorem max_x2_plus_y2 (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3*x + y| ≤ 6) : x^2 + y^2 ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_x2_plus_y2_l1002_100270


namespace NUMINAMATH_CALUDE_inequality_proof_l1002_100251

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b * 2^a + a * 2^(-b) ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1002_100251


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_union_A_B_equals_B_l1002_100257

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A (-2) ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem union_A_B_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_union_A_B_equals_B_l1002_100257


namespace NUMINAMATH_CALUDE_exam_correct_answers_l1002_100219

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  wrong_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  total_score : ℤ

/-- Calculates the number of correctly answered questions. -/
def correct_answers (result : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that for the given exam conditions, 
    the number of correct answers is 40. -/
theorem exam_correct_answers 
  (e : Exam) 
  (r : ExamResult) 
  (h1 : e.total_questions = 80) 
  (h2 : e.correct_score = 4) 
  (h3 : e.wrong_score = -1) 
  (h4 : r.exam = e) 
  (h5 : r.total_score = 120) : 
  correct_answers r = 40 :=
sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l1002_100219


namespace NUMINAMATH_CALUDE_ceiling_abs_negative_l1002_100292

theorem ceiling_abs_negative : ⌈|(-52.7 : ℝ)|⌉ = 53 := by sorry

end NUMINAMATH_CALUDE_ceiling_abs_negative_l1002_100292


namespace NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1002_100252

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The set of outcomes where the sum of the dice is 5 -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 + 2 = 5)

/-- The probability of the sum of two fair dice being 5 -/
theorem prob_sum_five_is_one_ninth :
  (sumFiveOutcomes.card : ℚ) / outcomes.card = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1002_100252


namespace NUMINAMATH_CALUDE_f_composition_equals_constant_l1002_100277

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z ^ 2 else -3 * z ^ 2

-- State the theorem
theorem f_composition_equals_constant : f (f (f (f (1 + I)))) = (-28311552 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_constant_l1002_100277


namespace NUMINAMATH_CALUDE_sally_peaches_theorem_l1002_100255

/-- Represents the number of peaches Sally picked at the orchard -/
def peaches_picked (initial total : ℕ) : ℕ := total - initial

/-- Theorem stating that the number of peaches Sally picked is the difference between her total and initial peaches -/
theorem sally_peaches_theorem (initial total : ℕ) (h : initial ≤ total) :
  peaches_picked initial total = total - initial :=
by sorry

end NUMINAMATH_CALUDE_sally_peaches_theorem_l1002_100255


namespace NUMINAMATH_CALUDE_odd_function_value_l1002_100284

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 ∧ x < 2 then a * Real.log x - a * x + 1 else 0

-- State the theorem
theorem odd_function_value (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 2, f a x = a * Real.log x - a * x + 1) →  -- definition for x ∈ (0, 2)
  (∃ c ∈ Set.Ioo (-2) 0, ∀ x ∈ Set.Ioo (-2) 0, f a x ≥ f a c) →  -- minimum value exists in (-2, 0)
  (∃ c ∈ Set.Ioo (-2) 0, f a c = 1) →  -- minimum value is 1
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_value_l1002_100284


namespace NUMINAMATH_CALUDE_max_correct_percentage_l1002_100247

theorem max_correct_percentage
  (total : ℝ)
  (solo_portion : ℝ)
  (together_portion : ℝ)
  (chloe_solo_correct : ℝ)
  (chloe_overall_correct : ℝ)
  (max_solo_correct : ℝ)
  (h1 : solo_portion = 2/3)
  (h2 : together_portion = 1/3)
  (h3 : solo_portion + together_portion = 1)
  (h4 : chloe_solo_correct = 0.7)
  (h5 : chloe_overall_correct = 0.82)
  (h6 : max_solo_correct = 0.85)
  : max_solo_correct * solo_portion + (chloe_overall_correct - chloe_solo_correct * solo_portion) = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_percentage_l1002_100247


namespace NUMINAMATH_CALUDE_hundredth_odd_integer_l1002_100248

theorem hundredth_odd_integer : ∀ n : ℕ, n > 0 → (2 * n - 1) = 199 ↔ n = 100 := by sorry

end NUMINAMATH_CALUDE_hundredth_odd_integer_l1002_100248


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1002_100236

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  4*x + 3*y - 35 = 0

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧
  ∀ (x' y' : ℝ), circle_equation x' y' ∧ line_equation x' y' → (x', y') = (x, y) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1002_100236


namespace NUMINAMATH_CALUDE_contest_scores_l1002_100254

theorem contest_scores (x y : ℝ) : 
  (9 + 8.7 + 9.3 + x + y) / 5 = 9 →
  ((9 - 9)^2 + (8.7 - 9)^2 + (9.3 - 9)^2 + (x - 9)^2 + (y - 9)^2) / 5 = 0.1 →
  |x - y| = 0.8 := by
sorry

end NUMINAMATH_CALUDE_contest_scores_l1002_100254


namespace NUMINAMATH_CALUDE_cloud_computing_analysis_l1002_100218

/-- Cloud computing market data --/
structure MarketData :=
  (year : ℕ)
  (market_scale : ℝ)

/-- Regression equation coefficients --/
structure RegressionCoefficients :=
  (b : ℝ)
  (a : ℝ)

/-- Cloud computing market analysis --/
theorem cloud_computing_analysis 
  (data : List MarketData)
  (sum_ln_y : ℝ)
  (sum_x_ln_y : ℝ)
  (initial_error_variance : ℝ → ℝ)
  (initial_probability : ℝ)
  (new_error_variance : ℝ → ℝ) :
  ∃ (coef : RegressionCoefficients) 
    (new_probability : ℝ) 
    (cost_decrease : ℝ),
  (coef.b = 0.386 ∧ coef.a = 6.108) ∧
  (new_probability = 0.9545) ∧
  (cost_decrease = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cloud_computing_analysis_l1002_100218


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l1002_100213

theorem divisible_by_twelve (n : ℤ) : 12 ∣ n^2 * (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l1002_100213


namespace NUMINAMATH_CALUDE_function_composition_equality_l1002_100283

theorem function_composition_equality (C D : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ)
  (h_def : ∀ x, h x = C * x - 3 * D^2)
  (k_def : ∀ x, k x = D * x + 1)
  (D_neq_neg_one : D ≠ -1)
  (h_k_2_eq_zero : h (k 2) = 0) :
  C = 3 * D^2 / (2 * D + 1) := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1002_100283


namespace NUMINAMATH_CALUDE_product_eleven_sum_reciprocal_squares_l1002_100258

theorem product_eleven_sum_reciprocal_squares :
  ∀ a b : ℕ,
  a * b = 11 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 122 / 121 :=
by
  sorry

end NUMINAMATH_CALUDE_product_eleven_sum_reciprocal_squares_l1002_100258


namespace NUMINAMATH_CALUDE_probability_green_yellow_blue_l1002_100201

def total_balls : ℕ := 500
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50

theorem probability_green_yellow_blue :
  (green_balls + yellow_balls + blue_balls : ℚ) / total_balls = 220 / 500 := by
  sorry

end NUMINAMATH_CALUDE_probability_green_yellow_blue_l1002_100201


namespace NUMINAMATH_CALUDE_ant_problem_l1002_100232

/-- Represents the number of ants for each species on Day 0 -/
structure AntCounts where
  a : ℕ  -- Species A
  b : ℕ  -- Species B
  c : ℕ  -- Species C

/-- Calculates the total number of ants on a given day -/
def totalAnts (day : ℕ) (counts : AntCounts) : ℕ :=
  2^day * counts.a + 3^day * counts.b + 4^day * counts.c

theorem ant_problem (counts : AntCounts) :
  totalAnts 0 counts = 50 →
  totalAnts 4 counts = 6561 →
  4^4 * counts.c = 5632 := by
  sorry

end NUMINAMATH_CALUDE_ant_problem_l1002_100232


namespace NUMINAMATH_CALUDE_complex_sum_properties_l1002_100214

open Complex

/-- Given complex numbers z and u with the specified properties, prove the required statements -/
theorem complex_sum_properties (α β : ℝ) (z u : ℂ) 
  (hz : z = Complex.exp (I * α))  -- z = cos α + i sin α
  (hu : u = Complex.exp (I * β))  -- u = cos β + i sin β
  (hsum : z + u = (4/5 : ℂ) + (3/5 : ℂ) * I) : 
  (Complex.tan (α + β) = 24/7) ∧ (z^2 + u^2 + z*u = 0) := by
  sorry


end NUMINAMATH_CALUDE_complex_sum_properties_l1002_100214


namespace NUMINAMATH_CALUDE_g_neg_two_eq_eleven_l1002_100223

/-- The function g(x) = x^2 - 2x + 3 -/
def g (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: g(-2) = 11 -/
theorem g_neg_two_eq_eleven : g (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_eq_eleven_l1002_100223


namespace NUMINAMATH_CALUDE_cab_driver_income_l1002_100250

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 600)
  (h3 : day3 = 450)
  (h4 : day4 = 400)
  (h5 : day5 = 800)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 500) :
  day2 = 250 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1002_100250


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l1002_100260

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 4*x - 5 = 0 ↔ (x - 2)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l1002_100260


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1002_100200

theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 3 ∧ P.2 = 2 * a + 1 ∧ P.2 = 0) → a = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1002_100200


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l1002_100273

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  length : ℝ
  width : ℝ
  height : ℝ
  radius : ℝ
  length_eq_twice_height : length = 2 * height
  surface_area_eq_288 : 2 * (length * width + width * height + length * height) = 288
  edge_sum_eq_96 : 4 * (length + width + height) = 96
  inscribed_in_sphere : (2 * radius) ^ 2 = length ^ 2 + width ^ 2 + height ^ 2

/-- The radius of the sphere containing the inscribed box is 4√5 -/
theorem inscribed_box_radius (box : InscribedBox) : box.radius = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l1002_100273


namespace NUMINAMATH_CALUDE_natasha_quarters_l1002_100278

theorem natasha_quarters : ∃ n : ℕ,
  8 < n ∧ n < 80 ∧
  n % 4 = 3 ∧
  n % 5 = 1 ∧
  n % 7 = 3 ∧
  n = 31 := by
sorry

end NUMINAMATH_CALUDE_natasha_quarters_l1002_100278


namespace NUMINAMATH_CALUDE_modular_arithmetic_problems_l1002_100263

theorem modular_arithmetic_problems :
  (∃ k : ℕ, 19^10 = 6 * k + 1) ∧
  (∃ m : ℕ, 19^14 = 70 * m + 11) ∧
  (∃ n : ℕ, 17^9 = 48 * n + 17) ∧
  (∃ p : ℕ, 14^(14^14) = 100 * p + 36) := by
sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problems_l1002_100263


namespace NUMINAMATH_CALUDE_fruit_preference_ratio_l1002_100217

theorem fruit_preference_ratio (total_students : ℕ) 
  (cherries_preference : ℕ) (apple_date_ratio : ℕ) (banana_cherry_ratio : ℕ) 
  (h1 : total_students = 780)
  (h2 : cherries_preference = 60)
  (h3 : apple_date_ratio = 2)
  (h4 : banana_cherry_ratio = 3) : 
  (banana_cherry_ratio * cherries_preference) / 
  ((total_students - banana_cherry_ratio * cherries_preference - cherries_preference) / 
   (apple_date_ratio + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_fruit_preference_ratio_l1002_100217


namespace NUMINAMATH_CALUDE_expression_simplification_l1002_100229

theorem expression_simplification (x : ℝ) (h : x = (Real.sqrt 3 - 1) / 3) :
  (2 / (x - 1) + 1 / (x + 1)) * (x^2 - 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1002_100229


namespace NUMINAMATH_CALUDE_path_cost_calculation_l1002_100265

/-- Calculates the total cost of constructing a path around a rectangular field -/
def path_construction_cost (field_length field_width path_width cost_per_sqm : ℝ) : ℝ :=
  let outer_length := field_length + 2 * path_width
  let outer_width := field_width + 2 * path_width
  let total_area := outer_length * outer_width
  let field_area := field_length * field_width
  let path_area := total_area - field_area
  path_area * cost_per_sqm

/-- Theorem stating the total cost of constructing the path -/
theorem path_cost_calculation :
  path_construction_cost 75 55 2.5 10 = 6750 := by
  sorry

end NUMINAMATH_CALUDE_path_cost_calculation_l1002_100265


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l1002_100299

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 3

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 6

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := 11/35

theorem hexagon_segment_probability :
  let total := T.card
  let same_length_pairs := (num_sides.choose 2) + (num_short_diagonals.choose 2) + (num_long_diagonals.choose 2)
  prob_same_length = same_length_pairs / (total.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l1002_100299


namespace NUMINAMATH_CALUDE_oranges_per_pack_l1002_100244

/-- Proves the number of oranges in each pack given Tammy's orange selling scenario -/
theorem oranges_per_pack (trees : ℕ) (oranges_per_tree_per_day : ℕ) (price_per_pack : ℕ) 
  (total_earnings : ℕ) (days : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  price_per_pack = 2 →
  total_earnings = 840 →
  days = 21 →
  (trees * oranges_per_tree_per_day * days) / (total_earnings / price_per_pack) = 6 := by
  sorry

#check oranges_per_pack

end NUMINAMATH_CALUDE_oranges_per_pack_l1002_100244


namespace NUMINAMATH_CALUDE_batsman_average_l1002_100225

theorem batsman_average (x : ℕ) : 
  (40 * x + 30 * 10) / (x + 10) = 35 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l1002_100225


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l1002_100282

theorem ice_cream_consumption (friday_consumption : Real) (total_consumption : Real)
  (h1 : friday_consumption = 3.25)
  (h2 : total_consumption = 3.5) :
  total_consumption - friday_consumption = 0.25 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l1002_100282


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_l1002_100233

theorem negation_of_exists_greater (p : Prop) :
  (¬ ∃ (n : ℕ), 2^n > 1000) ↔ (∀ (n : ℕ), 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_l1002_100233


namespace NUMINAMATH_CALUDE_circle_area_theorem_l1002_100245

-- Define the center and point on the circle
def center : ℝ × ℝ := (-2, 5)
def point : ℝ × ℝ := (8, -4)

-- Calculate the squared distance between two points
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- Define the theorem
theorem circle_area_theorem :
  let r := Real.sqrt (distance_squared center point)
  π * r^2 = 181 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l1002_100245


namespace NUMINAMATH_CALUDE_square_sum_equals_eleven_halves_l1002_100271

theorem square_sum_equals_eleven_halves (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 4) : 
  a^2 + b^2 = 11/2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eleven_halves_l1002_100271


namespace NUMINAMATH_CALUDE_camp_grouping_l1002_100297

theorem camp_grouping (total_children : ℕ) (max_group_size : ℕ) (h1 : total_children = 30) (h2 : max_group_size = 12) :
  ∃ (group_size : ℕ) (num_groups : ℕ),
    group_size ≤ max_group_size ∧
    group_size * num_groups = total_children ∧
    ∀ (k : ℕ), k ≤ max_group_size → k * (total_children / k) = total_children → num_groups ≤ (total_children / k) :=
by
  sorry

end NUMINAMATH_CALUDE_camp_grouping_l1002_100297


namespace NUMINAMATH_CALUDE_award_distribution_probability_l1002_100205

def num_classes : ℕ := 4
def num_awards : ℕ := 8

def distribute_awards (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem award_distribution_probability :
  let total_distributions := distribute_awards (num_awards - num_classes) num_classes
  let favorable_distributions := distribute_awards ((num_awards - num_classes) - 1) (num_classes - 1)
  (favorable_distributions : ℚ) / total_distributions = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_award_distribution_probability_l1002_100205


namespace NUMINAMATH_CALUDE_even_sum_converse_true_l1002_100228

theorem even_sum_converse_true (a b : ℤ) : 
  (∀ (a b : ℤ), Even (a + b) → Even a ∧ Even b) → 
  (Even a ∧ Even b → Even (a + b)) := by sorry

end NUMINAMATH_CALUDE_even_sum_converse_true_l1002_100228


namespace NUMINAMATH_CALUDE_bamboo_nine_nodes_l1002_100272

theorem bamboo_nine_nodes (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 1 + a 3 + a 9 = 17/6 :=     -- sum of 1st, 3rd, and 9th terms
by sorry

end NUMINAMATH_CALUDE_bamboo_nine_nodes_l1002_100272


namespace NUMINAMATH_CALUDE_greg_granola_bars_l1002_100235

/-- Proves that Greg set aside 1 granola bar for each day of the week --/
theorem greg_granola_bars (total : ℕ) (traded : ℕ) (sisters : ℕ) (bars_per_sister : ℕ) (days : ℕ)
  (h_total : total = 20)
  (h_traded : traded = 3)
  (h_sisters : sisters = 2)
  (h_bars_per_sister : bars_per_sister = 5)
  (h_days : days = 7) :
  (total - traded - sisters * bars_per_sister) / days = 1 := by
  sorry

end NUMINAMATH_CALUDE_greg_granola_bars_l1002_100235


namespace NUMINAMATH_CALUDE_twirly_tea_cups_l1002_100211

theorem twirly_tea_cups (people_per_cup : ℕ) (total_people : ℕ) (num_cups : ℕ) :
  people_per_cup = 9 →
  total_people = 63 →
  num_cups * people_per_cup = total_people →
  num_cups = 7 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_l1002_100211


namespace NUMINAMATH_CALUDE_negative_three_x_squared_times_negative_three_x_l1002_100241

theorem negative_three_x_squared_times_negative_three_x (x : ℝ) :
  (-3 * x) * (-3 * x)^2 = -27 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_x_squared_times_negative_three_x_l1002_100241


namespace NUMINAMATH_CALUDE_image_property_l1002_100294

class StarOperation (T : Type) where
  star : T → T → T

variable {T : Type} [StarOperation T]

def image (a : T) : Set T :=
  {c | ∃ b, c = StarOperation.star a b}

theorem image_property (a : T) (c : T) (h : c ∈ image a) :
  StarOperation.star a c = c := by
  sorry

end NUMINAMATH_CALUDE_image_property_l1002_100294


namespace NUMINAMATH_CALUDE_father_son_age_relationship_l1002_100253

/-- Represents the age relationship between a father and his son Ronit -/
structure AgeRelationship where
  ronit_age : ℕ
  father_age : ℕ
  years_passed : ℕ

/-- The conditions of the problem -/
def age_conditions (ar : AgeRelationship) : Prop :=
  (ar.father_age = 4 * ar.ronit_age) ∧
  (ar.father_age + ar.years_passed = (5/2) * (ar.ronit_age + ar.years_passed)) ∧
  (ar.father_age + ar.years_passed + 8 = 2 * (ar.ronit_age + ar.years_passed + 8))

theorem father_son_age_relationship :
  ∃ ar : AgeRelationship, age_conditions ar ∧ ar.years_passed = 8 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_relationship_l1002_100253


namespace NUMINAMATH_CALUDE_range_of_a_l1002_100204

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 4*a}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (h : N ⊆ M a) : 1/2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1002_100204


namespace NUMINAMATH_CALUDE_allowance_theorem_l1002_100216

def initial_allowance : ℚ := 12

def first_week_spending (allowance : ℚ) : ℚ := allowance / 3

def second_week_spending (remaining : ℚ) : ℚ := remaining / 4

def final_amount (allowance : ℚ) : ℚ :=
  let after_first_week := allowance - first_week_spending allowance
  after_first_week - second_week_spending after_first_week

theorem allowance_theorem : final_amount initial_allowance = 6 := by
  sorry

end NUMINAMATH_CALUDE_allowance_theorem_l1002_100216


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1002_100234

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1002_100234


namespace NUMINAMATH_CALUDE_domino_puzzle_solution_exists_l1002_100210

-- Define the type for domino pieces
structure Domino where
  left : Fin 7
  right : Fin 7

-- Define the type for the puzzle arrangement
structure DominoPuzzle where
  rows : List (List Char)
  assignment : Char → Fin 7

-- Define the valid domino set (excluding 0-0)
def validDominoSet : List Domino := sorry

-- Check if a domino is valid (in the set)
def isValidDomino (d : Domino) : Bool := sorry

-- Check if the assignment is valid (each letter to a unique digit)
def isValidAssignment (assignment : Char → Fin 7) : Bool := sorry

-- Calculate the sum of a row given an assignment
def rowSum (row : List Char) (assignment : Char → Fin 7) : Nat := sorry

-- The main theorem
theorem domino_puzzle_solution_exists (puzzle : DominoPuzzle) : 
  (∀ row ∈ puzzle.rows, rowSum row puzzle.assignment = 24) ∧
  isValidAssignment puzzle.assignment ∧
  (∀ row ∈ puzzle.rows, ∀ pair ∈ row.zip (row.tail!), 
    isValidDomino ⟨puzzle.assignment pair.1, puzzle.assignment pair.2⟩) :=
sorry

end NUMINAMATH_CALUDE_domino_puzzle_solution_exists_l1002_100210


namespace NUMINAMATH_CALUDE_function_identity_l1002_100238

theorem function_identity (f : ℝ → ℝ) 
  (h₁ : f 0 = 1)
  (h₂ : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1002_100238


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1002_100276

theorem cos_alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 3 / 5) : 
  Real.cos α = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1002_100276


namespace NUMINAMATH_CALUDE_base8_to_base5_conversion_l1002_100215

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- The number 427 in base 8 -/
def num_base8 : ℕ := 427

/-- The number 2104 in base 5 -/
def num_base5 : ℕ := 2104

theorem base8_to_base5_conversion :
  base10ToBase5 (base8ToBase10 num_base8) = num_base5 := by sorry

end NUMINAMATH_CALUDE_base8_to_base5_conversion_l1002_100215


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l1002_100220

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m^2}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : 
  B m ⊆ A m → m = 0 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l1002_100220


namespace NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l1002_100290

-- Define the sequence a_n
def a (n : ℕ) : ℕ := n^2 - (n-1)^2

-- Define the triangle sides
def side_a : ℕ := a 2
def side_b : ℕ := a 3
def side_c : ℕ := a 4

-- State the theorem
theorem largest_angle_is_120_degrees :
  let angle := Real.arccos ((side_a^2 + side_b^2 - side_c^2) / (2 * side_a * side_b))
  angle = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l1002_100290


namespace NUMINAMATH_CALUDE_even_periodic_function_property_l1002_100261

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem even_periodic_function_property
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
  sorry

end NUMINAMATH_CALUDE_even_periodic_function_property_l1002_100261


namespace NUMINAMATH_CALUDE_sector_cone_properties_l1002_100296

/-- Represents a cone formed from a sector of a circular sheet -/
structure SectorCone where
  sheet_radius : ℝ
  num_sectors : ℕ

/-- Calculate the height of a cone formed from a sector of a circular sheet -/
def cone_height (c : SectorCone) : ℝ :=
  sorry

/-- Calculate the volume of a cone formed from a sector of a circular sheet -/
def cone_volume (c : SectorCone) : ℝ :=
  sorry

theorem sector_cone_properties (c : SectorCone) 
  (h_radius : c.sheet_radius = 12)
  (h_sectors : c.num_sectors = 4) :
  cone_height c = 3 * Real.sqrt 15 ∧ 
  cone_volume c = 9 * Real.pi * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_sector_cone_properties_l1002_100296


namespace NUMINAMATH_CALUDE_largest_m_for_cubic_quintic_inequality_l1002_100230

theorem largest_m_for_cubic_quintic_inequality :
  ∃ (m : ℝ), m = 9 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    10 * (a^3 + b^3 + c^3) - m * (a^5 + b^5 + c^5) ≥ 1) ∧
  (∀ (m' : ℝ), m' > m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      10 * (a^3 + b^3 + c^3) - m' * (a^5 + b^5 + c^5) < 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_cubic_quintic_inequality_l1002_100230


namespace NUMINAMATH_CALUDE_mickey_horses_per_week_l1002_100264

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := 7 + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mickey mounts 98 horses per week -/
theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end NUMINAMATH_CALUDE_mickey_horses_per_week_l1002_100264


namespace NUMINAMATH_CALUDE_x_values_l1002_100246

theorem x_values (x : ℝ) : (|2000 * x + 2000| = 20 * 2000) → (x = 19 ∨ x = -21) := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1002_100246


namespace NUMINAMATH_CALUDE_jackson_flight_distance_l1002_100240

theorem jackson_flight_distance (beka_distance : ℕ) (difference : ℕ) (jackson_distance : ℕ) : 
  beka_distance = 873 → 
  difference = 310 → 
  beka_distance = jackson_distance + difference → 
  jackson_distance = 563 :=
by sorry

end NUMINAMATH_CALUDE_jackson_flight_distance_l1002_100240


namespace NUMINAMATH_CALUDE_keaton_apple_harvest_interval_l1002_100289

/-- Represents Keaton's farm and harvesting schedule -/
structure Farm where
  orange_harvest_interval : ℕ  -- months between orange harvests
  orange_harvest_value : ℕ     -- value of each orange harvest in dollars
  apple_harvest_value : ℕ      -- value of each apple harvest in dollars
  total_yearly_earnings : ℕ    -- total earnings per year in dollars

/-- Calculates how often Keaton can harvest his apples -/
def apple_harvest_interval (f : Farm) : ℕ :=
  12 / ((f.total_yearly_earnings - (12 / f.orange_harvest_interval * f.orange_harvest_value)) / f.apple_harvest_value)

/-- Theorem stating that Keaton can harvest his apples every 3 months -/
theorem keaton_apple_harvest_interval :
  ∀ (f : Farm),
  f.orange_harvest_interval = 2 →
  f.orange_harvest_value = 50 →
  f.apple_harvest_value = 30 →
  f.total_yearly_earnings = 420 →
  apple_harvest_interval f = 3 := by
  sorry

end NUMINAMATH_CALUDE_keaton_apple_harvest_interval_l1002_100289


namespace NUMINAMATH_CALUDE_triangle_problem_l1002_100226

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    prove the measure of angle A and the area of the triangle 
    under specific conditions. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) →  -- Given condition
  (0 < A ∧ A < π) →                          -- Angle A is in (0, π)
  (0 < B ∧ B < π) →                          -- Angle B is in (0, π)
  (0 < C ∧ C < π) →                          -- Angle C is in (0, π)
  (A + B + C = π) →                          -- Sum of angles in a triangle
  (a * Real.sin B = b * Real.sin A) →        -- Law of sines
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) → -- Law of cosines
  (A = π / 6) ∧                              -- First part of the theorem
  ((Real.cos B = 2 * Real.sqrt 2 / 3 ∧ a = Real.sqrt 2) →
   (1 / 2 * a * b * Real.sin C = (2 * Real.sqrt 2 + Real.sqrt 3) / 9)) -- Second part
  := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1002_100226


namespace NUMINAMATH_CALUDE_hazel_fish_count_l1002_100222

theorem hazel_fish_count (total : ℕ) (father : ℕ) (hazel : ℕ) : 
  total = 94 → father = 46 → total = father + hazel → hazel = 48 := by
  sorry

end NUMINAMATH_CALUDE_hazel_fish_count_l1002_100222


namespace NUMINAMATH_CALUDE_distance_to_axes_l1002_100239

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance functions
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem distance_to_axes (Q : Point2D) (hx : Q.x = -6) (hy : Q.y = 5) :
  distToXAxis Q = 5 ∧ distToYAxis Q = 6 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_axes_l1002_100239


namespace NUMINAMATH_CALUDE_total_turtles_count_l1002_100237

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + 20

/-- The total number of turtles received by Marion and Martha -/
def total_turtles : ℕ := marion_turtles + martha_turtles

theorem total_turtles_count : total_turtles = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_count_l1002_100237


namespace NUMINAMATH_CALUDE_lemonade_price_calculation_l1002_100275

theorem lemonade_price_calculation (glasses_per_gallon : ℕ) (cost_per_gallon : ℚ) 
  (gallons_made : ℕ) (glasses_drunk : ℕ) (glasses_unsold : ℕ) (net_profit : ℚ) :
  glasses_per_gallon = 16 →
  cost_per_gallon = 7/2 →
  gallons_made = 2 →
  glasses_drunk = 5 →
  glasses_unsold = 6 →
  net_profit = 14 →
  (gallons_made * cost_per_gallon + net_profit) / 
    (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) = 1 := by
  sorry

#eval (2 * (7/2 : ℚ) + 14) / (2 * 16 - 5 - 6)

end NUMINAMATH_CALUDE_lemonade_price_calculation_l1002_100275


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_increasing_implies_a_nonnegative_l1002_100249

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Part 1: Extreme value at x = 3 implies a = 3
theorem extreme_value_implies_a (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a x ≤ f a 3) →
  a = 3 :=
sorry

-- Part 2: Increasing on (-∞, 0) implies a ∈ [0, +∞)
theorem increasing_implies_a_nonnegative (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 0 → f a x < f a y) →
  a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_increasing_implies_a_nonnegative_l1002_100249
