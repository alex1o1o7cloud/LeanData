import Mathlib

namespace x_fourth_plus_y_fourth_l2184_218420

theorem x_fourth_plus_y_fourth (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x * y = 48) : 
  x^4 + y^4 = 5392 := by
sorry

end x_fourth_plus_y_fourth_l2184_218420


namespace donuts_left_for_coworkers_l2184_218476

def total_donuts : ℕ := 30
def gluten_free_donuts : ℕ := 12
def regular_donuts : ℕ := total_donuts - gluten_free_donuts

def gluten_free_eaten_driving : ℕ := 1
def regular_eaten_driving : ℕ := 0

def gluten_free_afternoon_snack : ℕ := 2
def regular_afternoon_snack : ℕ := 4

theorem donuts_left_for_coworkers :
  total_donuts - 
  (gluten_free_eaten_driving + regular_eaten_driving + 
   gluten_free_afternoon_snack + regular_afternoon_snack) = 23 := by
  sorry

end donuts_left_for_coworkers_l2184_218476


namespace probability_of_drawing_balls_l2184_218481

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5
def drawn_balls : ℕ := 5
def drawn_black : ℕ := 3
def drawn_white : ℕ := 2

theorem probability_of_drawing_balls : 
  (Nat.choose black_balls drawn_black * Nat.choose white_balls drawn_white) / 
  Nat.choose total_balls drawn_balls = 400 / 1001 := by
sorry

end probability_of_drawing_balls_l2184_218481


namespace root_of_two_equations_l2184_218461

theorem root_of_two_equations (p q r s t k : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (eq1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
  (eq2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0) :
  k = 1 ∨ k = Complex.exp (Complex.I * π / 3) ∨ 
  k = Complex.exp (-Complex.I * π / 3) ∨ k = -1 ∨ 
  k = Complex.exp (2 * Complex.I * π / 3) ∨ 
  k = Complex.exp (-2 * Complex.I * π / 3) := by
  sorry

end root_of_two_equations_l2184_218461


namespace intersection_of_M_and_N_l2184_218472

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x - 3 < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l2184_218472


namespace derivative_product_polynomial_l2184_218482

theorem derivative_product_polynomial (x : ℝ) :
  let f : ℝ → ℝ := λ x => (2*x^2 + 3)*(3*x - 1)
  let f' : ℝ → ℝ := λ x => 18*x^2 - 4*x + 9
  HasDerivAt f (f' x) x := by sorry

end derivative_product_polynomial_l2184_218482


namespace tax_rate_65_percent_l2184_218435

/-- Given a tax rate as a percentage, calculate the equivalent dollar amount per $100.00 -/
def tax_rate_to_dollars (percent : ℝ) : ℝ :=
  percent

theorem tax_rate_65_percent : tax_rate_to_dollars 65 = 65 := by
  sorry

end tax_rate_65_percent_l2184_218435


namespace trigonometric_identities_l2184_218466

theorem trigonometric_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  ((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = 7 / 17) ∧
  (Real.sin α ^ 2 + Real.sin (2 * α) = 0) := by
  sorry

end trigonometric_identities_l2184_218466


namespace audiobook_listening_time_l2184_218469

/-- Calculates the average daily listening time for audiobooks -/
def average_daily_listening_time (num_audiobooks : ℕ) (audiobook_length : ℕ) (total_days : ℕ) : ℚ :=
  (num_audiobooks * audiobook_length : ℚ) / total_days

/-- Proves that the average daily listening time is 2 hours given the specific conditions -/
theorem audiobook_listening_time :
  let num_audiobooks : ℕ := 6
  let audiobook_length : ℕ := 30
  let total_days : ℕ := 90
  average_daily_listening_time num_audiobooks audiobook_length total_days = 2 := by
  sorry

end audiobook_listening_time_l2184_218469


namespace opposite_reciprocal_abs_l2184_218422

theorem opposite_reciprocal_abs (x : ℚ) (h : x = -4/3) : 
  (-x = 4/3) ∧ (x⁻¹ = -3/4) ∧ (|x| = 4/3) := by
  sorry

end opposite_reciprocal_abs_l2184_218422


namespace at_least_one_travels_to_beijing_l2184_218493

theorem at_least_one_travels_to_beijing 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/3) 
  (h2 : prob_B = 1/4) 
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1) 
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) : 
  1 - (1 - prob_A) * (1 - prob_B) = 1/2 := by
sorry

end at_least_one_travels_to_beijing_l2184_218493


namespace sqrt_equality_implies_relation_l2184_218417

theorem sqrt_equality_implies_relation (a b c : ℕ+) :
  (a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ) ≥ 0 →
  Real.sqrt ((a.val ^ 2 : ℝ) - (b.val : ℝ) / (c.val : ℝ)) = a.val - Real.sqrt ((b.val : ℝ) / (c.val : ℝ)) →
  b = a ^ 2 * c := by
  sorry

end sqrt_equality_implies_relation_l2184_218417


namespace initial_volume_of_solution_l2184_218424

/-- Given a solution with initial volume V, prove that V = 40 liters -/
theorem initial_volume_of_solution (V : ℝ) : 
  (0.05 * V + 3.5 = 0.11 * (V + 10)) → V = 40 := by sorry

end initial_volume_of_solution_l2184_218424


namespace count_integers_satisfying_inequality_l2184_218430

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005) ∧
    (∀ n : ℕ, 2000 < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < 2005 → n ∈ S) ∧
    Finset.card S = 5 :=
by sorry

end count_integers_satisfying_inequality_l2184_218430


namespace complex_equation_solution_l2184_218438

theorem complex_equation_solution (z : ℂ) : (2 + z) / (2 - z) = I → z = 2 * I := by
  sorry

end complex_equation_solution_l2184_218438


namespace tony_weightlifting_ratio_l2184_218418

/-- Given Tony's weightlifting capabilities, prove the ratio of his military press to curl weight. -/
theorem tony_weightlifting_ratio :
  ∀ (curl_weight military_press_weight squat_weight : ℝ),
    curl_weight = 90 →
    squat_weight = 5 * military_press_weight →
    squat_weight = 900 →
    military_press_weight / curl_weight = 2 :=
by
  sorry

end tony_weightlifting_ratio_l2184_218418


namespace koi_fish_count_l2184_218458

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end koi_fish_count_l2184_218458


namespace units_digit_of_expression_l2184_218492

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 8 * 19 * 1983 - 8^3 is 4 -/
theorem units_digit_of_expression : unitsDigit (8 * 19 * 1983 - 8^3) = 4 := by
  sorry

end units_digit_of_expression_l2184_218492


namespace mode_most_effective_l2184_218468

/-- Represents different statistical measures -/
inductive StatisticalMeasure
  | Variance
  | Mean
  | Median
  | Mode

/-- Represents a shoe model -/
structure ShoeModel where
  id : Nat
  sales : Nat

/-- Represents a shoe store -/
structure ShoeStore where
  models : List ShoeModel
  
/-- Determines the most effective statistical measure for increasing sales -/
def mostEffectiveMeasure (store : ShoeStore) : StatisticalMeasure :=
  StatisticalMeasure.Mode

/-- Theorem: The mode is the most effective statistical measure for increasing sales -/
theorem mode_most_effective (store : ShoeStore) :
  mostEffectiveMeasure store = StatisticalMeasure.Mode :=
by sorry

end mode_most_effective_l2184_218468


namespace alcohol_mixture_proof_l2184_218437

/-- Proves that mixing 200 mL of 10% alcohol solution with 600 mL of 30% alcohol solution results in a 25% alcohol mixture -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 600
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
sorry

end alcohol_mixture_proof_l2184_218437


namespace will_initial_candy_l2184_218421

/-- The amount of candy Will gave to Haley -/
def candy_given : ℕ := 6

/-- The amount of candy Will had left after giving some to Haley -/
def candy_left : ℕ := 9

/-- The initial amount of candy Will had -/
def initial_candy : ℕ := candy_given + candy_left

theorem will_initial_candy : initial_candy = 15 := by
  sorry

end will_initial_candy_l2184_218421


namespace largest_remainder_209_l2184_218463

theorem largest_remainder_209 :
  (∀ n : ℕ, n < 120 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 104) ∧
  (∃ n : ℕ, n < 120 ∧ ∃ k : ℕ, 209 = n * k + 104) ∧
  (∀ n : ℕ, n < 90 → ∃ k r : ℕ, 209 = n * k + r ∧ r < n ∧ r ≤ 69) ∧
  (∃ n : ℕ, n < 90 ∧ ∃ k : ℕ, 209 = n * k + 69) :=
by sorry

end largest_remainder_209_l2184_218463


namespace kaydence_age_l2184_218479

/-- Kaydence's family ages problem -/
theorem kaydence_age (total_age father_age mother_age brother_age sister_age kaydence_age : ℕ) :
  total_age = 200 ∧
  father_age = 60 ∧
  mother_age = father_age - 2 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  total_age = father_age + mother_age + brother_age + sister_age + kaydence_age →
  kaydence_age = 12 := by
  sorry

end kaydence_age_l2184_218479


namespace largest_of_five_consecutive_composites_l2184_218487

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem largest_of_five_consecutive_composites (a b c d e : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ is_two_digit d ∧ is_two_digit e →
  a < 40 ∧ b < 40 ∧ c < 40 ∧ d < 40 ∧ e < 40 →
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 →
  ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ ¬(is_prime d) ∧ ¬(is_prime e) →
  e = 36 :=
sorry

end largest_of_five_consecutive_composites_l2184_218487


namespace point_P_satisfies_conditions_l2184_218410

-- Define the curve C
def C (x : ℝ) : ℝ := x^3 - 10*x + 3

-- Define the derivative of C
def C' (x : ℝ) : ℝ := 3*x^2 - 10

theorem point_P_satisfies_conditions : 
  let x₀ : ℝ := -2
  let y₀ : ℝ := 15
  (x₀ < 0) ∧ 
  (C x₀ = y₀) ∧ 
  (C' x₀ = 2) := by sorry

end point_P_satisfies_conditions_l2184_218410


namespace mike_spent_500_on_plants_l2184_218414

/-- The amount Mike spent on plants for himself -/
def mike_spent_on_plants : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | total_rose_bushes, rose_bush_price, friend_rose_bushes, num_aloes, aloe_price =>
    let self_rose_bushes := total_rose_bushes - friend_rose_bushes
    let rose_bush_cost := self_rose_bushes * rose_bush_price
    let aloe_cost := num_aloes * aloe_price
    rose_bush_cost + aloe_cost

theorem mike_spent_500_on_plants :
  mike_spent_on_plants 6 75 2 2 100 = 500 := by
  sorry

end mike_spent_500_on_plants_l2184_218414


namespace product_of_two_greatest_unattainable_scores_l2184_218499

/-- A score is attainable if it can be expressed as a non-negative integer combination of 19, 9, and 8. -/
def IsAttainable (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 19 * a + 9 * b + 8 * c

/-- The set of all attainable scores. -/
def AttainableScores : Set ℕ :=
  {n : ℕ | IsAttainable n}

/-- The set of all unattainable scores. -/
def UnattainableScores : Set ℕ :=
  {n : ℕ | ¬IsAttainable n}

/-- The two greatest unattainable scores. -/
def TwoGreatestUnattainableScores : Fin 2 → ℕ :=
  fun i => if i = 0 then 39 else 31

theorem product_of_two_greatest_unattainable_scores :
  (TwoGreatestUnattainableScores 0) * (TwoGreatestUnattainableScores 1) = 1209 ∧
  (∀ n : ℕ, n ∈ UnattainableScores → n ≤ (TwoGreatestUnattainableScores 0)) ∧
  (∀ n : ℕ, n ∈ UnattainableScores ∧ n ≠ (TwoGreatestUnattainableScores 0) → n ≤ (TwoGreatestUnattainableScores 1)) :=
by sorry

end product_of_two_greatest_unattainable_scores_l2184_218499


namespace max_stores_visited_l2184_218402

theorem max_stores_visited (
  total_stores : ℕ)
  (total_shoppers : ℕ)
  (double_visitors : ℕ)
  (total_visits : ℕ)
  (h1 : total_stores = 7)
  (h2 : total_shoppers = 11)
  (h3 : double_visitors = 7)
  (h4 : total_visits = 21)
  (h5 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧
    ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end max_stores_visited_l2184_218402


namespace inequalities_hold_l2184_218452

theorem inequalities_hold (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b + 1) / (a + 1) > b / a ∧ a + 1 / b > b + 1 / a := by
  sorry

end inequalities_hold_l2184_218452


namespace arithmetic_geometric_sequence_l2184_218400

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_2, a_4, and a_8 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 4) ^ 2 = a 2 * a 8

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a → geometric_subseq a → a 4 = 12 := by
  sorry

end arithmetic_geometric_sequence_l2184_218400


namespace triangle_properties_l2184_218428

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle ABC -/
theorem triangle_properties (t : Triangle)
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c)
  (h2 : t.a = Real.sqrt 2)
  (h3 : Real.sin t.B * Real.sin t.C = (Real.sin t.A)^2) :
  t.A = π/3 ∧ (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :=
by sorry

end triangle_properties_l2184_218428


namespace jonessa_pay_l2184_218477

theorem jonessa_pay (tax_rate : ℝ) (take_home_pay : ℝ) (total_pay : ℝ) : 
  tax_rate = 0.1 →
  take_home_pay = 450 →
  take_home_pay = total_pay * (1 - tax_rate) →
  total_pay = 500 := by
sorry

end jonessa_pay_l2184_218477


namespace min_value_of_f_l2184_218459

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 * (x - 2)

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 2, m ≤ f x) ∧ (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = m) ∧ m = -64 := by
  sorry

end min_value_of_f_l2184_218459


namespace perimeter_is_22_l2184_218478

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 14 - y^2 / 11 = 1

-- Define the focus F₁
def F₁ : ℝ × ℝ := sorry

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P and Q are on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom Q_on_hyperbola : hyperbola Q.1 Q.2

-- State that P and Q are on line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l

-- State that line l passes through the origin
axiom l_through_origin : (0, 0) ∈ l

-- State that PF₁ · QF₁ = 0
axiom PF₁_perp_QF₁ : (P.1 - F₁.1, P.2 - F₁.2) • (Q.1 - F₁.1, Q.2 - F₁.2) = 0

-- Define the perimeter of triangle PF₁Q
def perimeter_PF₁Q : ℝ := sorry

-- Theorem to prove
theorem perimeter_is_22 : perimeter_PF₁Q = 22 := sorry

end perimeter_is_22_l2184_218478


namespace problem_1_problem_2_l2184_218413

-- Problem 1
theorem problem_1 : (-2)^3 / (-2)^2 * (1/2)^0 = -2 := by sorry

-- Problem 2
theorem problem_2 : 199 * 201 + 1 = 40000 := by sorry

end problem_1_problem_2_l2184_218413


namespace collinear_vectors_x_value_l2184_218429

/-- Two vectors a and b in ℝ² are collinear if there exists a scalar k such that b = k * a -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given two vectors a = (1, -2) and b = (-2, x) in ℝ², 
    if a and b are collinear, then x = 4 -/
theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  collinear a b → x = 4 := by
sorry

end collinear_vectors_x_value_l2184_218429


namespace binary_representation_of_70_has_7_digits_l2184_218490

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_70_has_7_digits :
  (decimal_to_binary 70).length = 7 := by
  sorry

end binary_representation_of_70_has_7_digits_l2184_218490


namespace initial_average_height_l2184_218498

theorem initial_average_height (n : ℕ) (wrong_height actual_height : ℝ) (actual_average : ℝ) :
  n = 35 ∧
  wrong_height = 166 ∧
  actual_height = 106 ∧
  actual_average = 181 →
  (n * actual_average + (wrong_height - actual_height)) / n = 182 + 5 / 7 :=
by sorry

end initial_average_height_l2184_218498


namespace number_problem_l2184_218480

theorem number_problem (x : ℚ) : x - (3/5) * x = 56 → x = 140 := by
  sorry

end number_problem_l2184_218480


namespace janet_stuffies_l2184_218409

theorem janet_stuffies (total : ℕ) (kept_fraction : ℚ) (given_fraction : ℚ) : 
  total = 60 →
  kept_fraction = 1/3 →
  given_fraction = 1/4 →
  (total - kept_fraction * total) * given_fraction = 10 := by
sorry

end janet_stuffies_l2184_218409


namespace exam_marks_proof_l2184_218439

theorem exam_marks_proof (T : ℝ) 
  (h1 : 0.3 * T + 50 = 199.99999999999997) 
  (passing_mark : ℝ := 199.99999999999997) 
  (second_candidate_score : ℝ := 0.45 * T) : 
  second_candidate_score - passing_mark = 25 := by
sorry

end exam_marks_proof_l2184_218439


namespace prime_sum_product_l2184_218465

theorem prime_sum_product : 
  ∃ (p q : ℕ), Prime p ∧ Prime q ∧ 2 * p + 5 * q = 36 ∧ p * q = 26 := by
  sorry

end prime_sum_product_l2184_218465


namespace percentage_problem_l2184_218427

theorem percentage_problem (P : ℝ) : P = 35 ↔ (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42 := by
  sorry

end percentage_problem_l2184_218427


namespace angle_properties_l2184_218406

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -√3x
def terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), y = -Real.sqrt 3 * x ∧ x = Real.cos α ∧ y = Real.sin α

-- Define the set S of angles with the same terminal side as α
def S : Set Real :=
  {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}

-- State the theorem
theorem angle_properties (h : terminal_side α) :
  (Real.tan α = -Real.sqrt 3) ∧
  (S = {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}) ∧
  ((Real.sqrt 3 * Real.sin (α - Real.pi) + 5 * Real.cos (2 * Real.pi - α)) /
   (-Real.sqrt 3 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (Real.pi + α)) = 4) :=
by sorry

end angle_properties_l2184_218406


namespace average_cost_is_14_cents_l2184_218450

/-- Calculates the average cost per pencil in cents, rounded to the nearest whole number -/
def average_cost_per_pencil (num_pencils : ℕ) (catalog_price shipping_cost discount : ℚ) : ℕ :=
  let total_cost_cents := (catalog_price + shipping_cost - discount) * 100
  let average_cost_cents := total_cost_cents / num_pencils
  (average_cost_cents + 1/2).floor.toNat

/-- Proves that the average cost per pencil is 14 cents given the specified conditions -/
theorem average_cost_is_14_cents :
  average_cost_per_pencil 150 15 7.5 1.5 = 14 := by
  sorry

#eval average_cost_per_pencil 150 15 7.5 1.5

end average_cost_is_14_cents_l2184_218450


namespace javier_ate_five_meat_ravioli_l2184_218486

/-- Represents the weight of each type of ravioli in ounces -/
structure RavioliWeights where
  meat : Float
  pumpkin : Float
  cheese : Float

/-- Represents the number of each type of ravioli eaten by Javier -/
structure JavierMeal where
  meat : Nat
  pumpkin : Nat
  cheese : Nat

/-- Calculates the total weight of Javier's meal -/
def mealWeight (weights : RavioliWeights) (meal : JavierMeal) : Float :=
  weights.meat * meal.meat.toFloat + weights.pumpkin * meal.pumpkin.toFloat + weights.cheese * meal.cheese.toFloat

/-- Theorem: Given the conditions, Javier ate 5 meat ravioli -/
theorem javier_ate_five_meat_ravioli (weights : RavioliWeights) (meal : JavierMeal) : 
  weights.meat = 1.5 ∧ 
  weights.pumpkin = 1.25 ∧ 
  weights.cheese = 1 ∧ 
  meal.pumpkin = 2 ∧ 
  meal.cheese = 4 ∧ 
  mealWeight weights meal = 15 → 
  meal.meat = 5 := by
  sorry


end javier_ate_five_meat_ravioli_l2184_218486


namespace wedding_decoration_cost_per_place_setting_l2184_218445

/-- Calculates the cost per place setting for wedding decorations --/
theorem wedding_decoration_cost_per_place_setting 
  (num_tables : ℕ) 
  (tablecloth_cost : ℕ) 
  (place_settings_per_table : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (rose_cost : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) 
  (total_decoration_cost : ℕ) : 
  num_tables = 20 →
  tablecloth_cost = 25 →
  place_settings_per_table = 4 →
  roses_per_centerpiece = 10 →
  rose_cost = 5 →
  lilies_per_centerpiece = 15 →
  lily_cost = 4 →
  total_decoration_cost = 3500 →
  (total_decoration_cost - 
   (num_tables * tablecloth_cost + 
    num_tables * (roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost))) / 
   (num_tables * place_settings_per_table) = 10 := by
  sorry

end wedding_decoration_cost_per_place_setting_l2184_218445


namespace river_current_speed_l2184_218403

/-- Proves that the speed of the river's current is half the swimmer's speed in still water --/
theorem river_current_speed (x y : ℝ) : 
  x > 0 → -- swimmer's speed in still water is positive
  x = 10 → -- swimmer's speed in still water is 10 km/h
  (x + y) > 0 → -- downstream speed is positive
  (x - y) > 0 → -- upstream speed is positive
  (1 / (x - y)) = (3 * (1 / (x + y))) → -- upstream time is 3 times downstream time
  y = x / 2 := by
  sorry

end river_current_speed_l2184_218403


namespace five_balls_four_boxes_l2184_218426

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to put 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes :
  distribute_balls 5 4 = 1024 := by
  sorry

end five_balls_four_boxes_l2184_218426


namespace acute_angle_condition_l2184_218443

/-- Given vectors a and b in ℝ², prove that x > -3 is a necessary but not sufficient condition
    for the angle between a and b to be acute -/
theorem acute_angle_condition (a b : ℝ × ℝ) (x : ℝ) 
    (ha : a = (2, 3)) (hb : b = (x, 2)) : 
    (∃ (y : ℝ), y > -3 ∧ y ≠ x ∧ 
      ((a.1 * b.1 + a.2 * b.2 > 0) ∧ 
       ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2))) ∧
    (x > -3 → 
      (a.1 * b.1 + a.2 * b.2 > 0) ∧ 
      ¬(∃ (k : ℝ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2)) :=
by sorry

end acute_angle_condition_l2184_218443


namespace min_value_theorem_l2184_218408

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ ∀ (z : ℝ), z = 1/(x-y) + 9/(x+5*y) → z ≥ min_val := by
  sorry

end min_value_theorem_l2184_218408


namespace largest_number_is_nineteen_l2184_218485

theorem largest_number_is_nineteen 
  (a b c : ℕ) 
  (sum_ab : a + b = 16)
  (sum_ac : a + c = 20)
  (sum_bc : b + c = 23) :
  max a (max b c) = 19 := by
  sorry

end largest_number_is_nineteen_l2184_218485


namespace square_area_with_circles_l2184_218404

theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  let d := 2 * r
  let s := 2 * d
  s^2 = 144 :=
by sorry

end square_area_with_circles_l2184_218404


namespace all_polyhedra_l2184_218425

-- Define the properties of a polyhedron
structure Polyhedron :=
  (has_flat_faces : Bool)
  (has_straight_edges : Bool)
  (has_sharp_corners : Bool)

-- Define the geometric solids
inductive GeometricSolid
  | TriangularPrism
  | SquareFrustum
  | Cube
  | HexagonalPyramid

-- Function to check if a geometric solid is a polyhedron
def is_polyhedron (solid : GeometricSolid) : Polyhedron :=
  match solid with
  | GeometricSolid.TriangularPrism => ⟨true, true, true⟩
  | GeometricSolid.SquareFrustum => ⟨true, true, true⟩
  | GeometricSolid.Cube => ⟨true, true, true⟩
  | GeometricSolid.HexagonalPyramid => ⟨true, true, true⟩

-- Theorem stating that all the given solids are polyhedra
theorem all_polyhedra :
  (is_polyhedron GeometricSolid.TriangularPrism).has_flat_faces ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_straight_edges ∧
  (is_polyhedron GeometricSolid.TriangularPrism).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_flat_faces ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_straight_edges ∧
  (is_polyhedron GeometricSolid.SquareFrustum).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.Cube).has_flat_faces ∧
  (is_polyhedron GeometricSolid.Cube).has_straight_edges ∧
  (is_polyhedron GeometricSolid.Cube).has_sharp_corners ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_flat_faces ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_straight_edges ∧
  (is_polyhedron GeometricSolid.HexagonalPyramid).has_sharp_corners :=
by sorry

end all_polyhedra_l2184_218425


namespace y_divisibility_l2184_218470

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 24 * k) ∧
  ¬(∃ k : ℕ, y = 16 * k) := by
  sorry

end y_divisibility_l2184_218470


namespace e_4i_in_third_quadrant_l2184_218467

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define Euler's formula
axiom eulers_formula (x : ℝ) : cexp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem e_4i_in_third_quadrant :
  third_quadrant (cexp (4 * Complex.I)) :=
sorry

end e_4i_in_third_quadrant_l2184_218467


namespace janet_time_saved_l2184_218436

/-- The number of minutes Janet spends looking for keys daily -/
def keys_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding keys daily -/
def complain_time : ℕ := 3

/-- The number of minutes Janet spends searching for phone daily -/
def phone_time : ℕ := 5

/-- The number of minutes Janet spends looking for wallet daily -/
def wallet_time : ℕ := 4

/-- The number of minutes Janet spends trying to remember sunglasses location daily -/
def sunglasses_time : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Janet will save 154 minutes per week by stopping all these activities -/
theorem janet_time_saved :
  (keys_time + complain_time + phone_time + wallet_time + sunglasses_time) * days_in_week = 154 := by
  sorry

end janet_time_saved_l2184_218436


namespace naomi_bike_count_l2184_218484

theorem naomi_bike_count (total_wheels : ℕ) (childrens_bikes : ℕ) (regular_bike_wheels : ℕ) (childrens_bike_wheels : ℕ) : 
  total_wheels = 58 →
  childrens_bikes = 11 →
  regular_bike_wheels = 2 →
  childrens_bike_wheels = 4 →
  ∃ (regular_bikes : ℕ), regular_bikes = 7 ∧ 
    total_wheels = regular_bikes * regular_bike_wheels + childrens_bikes * childrens_bike_wheels :=
by
  sorry

end naomi_bike_count_l2184_218484


namespace intersection_implies_m_value_l2184_218457

def A : Set ℝ := {x | x^2 + x - 12 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem intersection_implies_m_value :
  ∃ m : ℝ, A ∩ B m = {3} → m = -1/3 := by sorry

end intersection_implies_m_value_l2184_218457


namespace total_vehicles_in_yard_l2184_218497

theorem total_vehicles_in_yard (num_trucks : ℕ) (num_tanks : ℕ) : 
  num_trucks = 20 → 
  num_tanks = 5 * num_trucks → 
  num_tanks + num_trucks = 120 :=
by
  sorry

end total_vehicles_in_yard_l2184_218497


namespace g_has_two_zeros_l2184_218412

noncomputable def f (x : ℝ) : ℝ := (x - Real.sin x) / Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - 1 / (2 * Real.exp 2)

theorem g_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧
  ∀ (x : ℝ), g x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end g_has_two_zeros_l2184_218412


namespace function_properties_l2184_218474

-- Define the function f(x) = ax³ + bx - 1
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = -3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = 1) ∧
    (b = -3) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≤ 17) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = 17) ∧
    (∀ x ∈ Set.Icc (-2) 3, f a b x ≥ -3) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a b x = -3) :=
by sorry

end function_properties_l2184_218474


namespace sets_satisfying_union_condition_l2184_218432

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ M ∈ S, M ∪ {1} = {1, 2, 3}) ∧ 
    (∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end sets_satisfying_union_condition_l2184_218432


namespace log_equation_holds_l2184_218455

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 49 / Real.log 2 :=
by sorry

end log_equation_holds_l2184_218455


namespace surface_area_ratio_of_spheres_l2184_218456

theorem surface_area_ratio_of_spheres (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end surface_area_ratio_of_spheres_l2184_218456


namespace fraction_subtraction_l2184_218447

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 7) - (2 + 5 + 7) / (3 + 6 + 9) = 32 / 63 := by
  sorry

end fraction_subtraction_l2184_218447


namespace hike_length_is_83_l2184_218454

/-- Represents the length of a 5-day hike satisfying specific conditions -/
def HikeLength (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧  -- Non-negative distances
  a + b = 36 ∧                              -- First two days
  (b + c + d) / 3 = 15 ∧                    -- Average of days 2, 3, 4
  c + d + e = 45 ∧                          -- Last three days
  a + c + e = 38                            -- Days 1, 3, 5

/-- The theorem stating that the total hike length is 83 miles -/
theorem hike_length_is_83 {a b c d e : ℝ} (h : HikeLength a b c d e) :
  a + b + c + d + e = 83 := by
  sorry


end hike_length_is_83_l2184_218454


namespace intersection_point_B_coords_l2184_218475

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)
  intersection : A ≠ B

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersection_point_B_coords (circles : IntersectingCircles) : circles.B = (-8, 8) := by
  sorry

end intersection_point_B_coords_l2184_218475


namespace cosine_graph_shift_l2184_218401

/-- Given a cosine function f(x) = 3cos(2x), prove that shifting its graph π/6 units 
    to the right results in the graph of g(x) = 3cos(2x - π/3) -/
theorem cosine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * (x - π / 6)) = 3 * Real.cos (2 * x - π / 3) := by
  sorry

end cosine_graph_shift_l2184_218401


namespace intersection_points_correct_l2184_218434

/-- The number of intersection points of line segments connecting m points on the positive X-axis
    and n points on the positive Y-axis, where no three segments intersect at the same point. -/
def intersectionPoints (m n : ℕ) : ℚ :=
  (m * (m - 1) * n * (n - 1) : ℚ) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = (m * (m - 1) * n * (n - 1) : ℚ) / 4 :=
by sorry

end intersection_points_correct_l2184_218434


namespace parabola_tangent_property_fixed_point_property_l2184_218407

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define a point on the axis of the parabola
def point_on_axis (G : ℝ × ℝ) : Prop := G.2 = -1

-- Define tangent points
def tangent_points (A B : ℝ × ℝ) (G : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ point_on_axis G

-- Define the perpendicular condition
def perpendicular (A M N : ℝ × ℝ) : Prop :=
  (M.1 - A.1) * (N.1 - A.1) + (M.2 - A.2) * (N.2 - A.2) = 0

-- Main theorem
theorem parabola_tangent_property (G : ℝ × ℝ) (A B : ℝ × ℝ) :
  tangent_points A B G → A.1 * B.1 + A.2 * B.2 = -3 :=
sorry

-- Fixed point theorem
theorem fixed_point_property (G A M N : ℝ × ℝ) :
  G.1 = 0 ∧ tangent_points A (2, 1) G ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧ perpendicular A M N →
  ∃ t : ℝ, t * (M.1 - 2) + (1 - t) * (N.1 - 2) = 0 ∧
         t * (M.2 - 5) + (1 - t) * (N.2 - 5) = 0 :=
sorry

end parabola_tangent_property_fixed_point_property_l2184_218407


namespace card_area_after_shortening_l2184_218491

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_after_shortening (initial : Rectangle) :
  initial.length = 6 ∧ initial.width = 8 →
  ∃ (shortened : Rectangle), 
    (shortened.length = initial.length ∧ shortened.width = initial.width - 2 ∧ 
     area shortened = 36) →
    area { length := initial.length - 2, width := initial.width } = 32 := by
  sorry

end card_area_after_shortening_l2184_218491


namespace scientific_notation_of_small_decimal_l2184_218423

theorem scientific_notation_of_small_decimal (x : ℝ) :
  x = 0.000815 →
  ∃ (a : ℝ) (n : ℤ), x = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -4 ∧ a = 8.15 :=
by sorry

end scientific_notation_of_small_decimal_l2184_218423


namespace range_of_f_l2184_218416

def f (x : ℝ) : ℝ := -x^2 + 2*x - 3

theorem range_of_f :
  ∃ (a b : ℝ), a = -3 ∧ b = -2 ∧
  (∀ x ∈ Set.Icc 0 2, a ≤ f x ∧ f x ≤ b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc 0 2, f x = y) :=
sorry

end range_of_f_l2184_218416


namespace geometric_sequence_pairs_l2184_218415

/-- The number of ordered pairs (a, r) satisfying the given conditions -/
def num_pairs : ℕ := 26^3

/-- The base of the logarithm and the exponent in the final equation -/
def base : ℕ := 2015
def exponent : ℕ := 155

theorem geometric_sequence_pairs :
  ∃ (S : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ S ↔ 
      let (a, r) := p
      (a > 0 ∧ r > 0) ∧ (a * r^6 = base^exponent)) ∧
    Finset.card S = num_pairs :=
sorry

end geometric_sequence_pairs_l2184_218415


namespace total_albums_l2184_218494

theorem total_albums (adele bridget katrina miriam : ℕ) : 
  adele = 30 →
  bridget = adele - 15 →
  katrina = 6 * bridget →
  miriam = 5 * katrina →
  adele + bridget + katrina + miriam = 585 :=
by
  sorry

end total_albums_l2184_218494


namespace cole_drive_time_to_work_l2184_218451

/-- Proves that given the conditions of Cole's round trip, it took him 210 minutes to drive to work. -/
theorem cole_drive_time_to_work (speed_to_work : ℝ) (speed_to_home : ℝ) (total_time : ℝ) :
  speed_to_work = 75 →
  speed_to_home = 105 →
  total_time = 6 →
  (total_time * speed_to_work * speed_to_home) / (speed_to_work + speed_to_home) * (60 / speed_to_work) = 210 := by
  sorry

#check cole_drive_time_to_work

end cole_drive_time_to_work_l2184_218451


namespace triple_hash_twenty_l2184_218441

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_hash_twenty : hash (hash (hash 20)) = 4.4 := by
  sorry

end triple_hash_twenty_l2184_218441


namespace book_sales_total_l2184_218411

/-- Calculates the total amount received from book sales given the number of books and their prices -/
def totalAmountReceived (fictionBooks nonFictionBooks childrensBooks : ℕ)
                        (fictionPrice nonFictionPrice childrensPrice : ℚ)
                        (fictionSoldRatio nonFictionSoldRatio childrensSoldRatio : ℚ) : ℚ :=
  (fictionBooks : ℚ) * fictionSoldRatio * fictionPrice +
  (nonFictionBooks : ℚ) * nonFictionSoldRatio * nonFictionPrice +
  (childrensBooks : ℚ) * childrensSoldRatio * childrensPrice

/-- The total amount received from book sales is $799 -/
theorem book_sales_total : 
  totalAmountReceived 60 84 42 5 7 3 (3/4) (5/6) (2/3) = 799 := by
  sorry

end book_sales_total_l2184_218411


namespace sphere_hemisphere_volume_ratio_l2184_218462

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (r / 3)^3) = 54 := by
  sorry

end sphere_hemisphere_volume_ratio_l2184_218462


namespace red_green_peaches_count_l2184_218483

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 6

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The total number of red and green peaches in the basket -/
def total_red_green : ℕ := red_peaches + green_peaches

theorem red_green_peaches_count : total_red_green = 22 := by
  sorry

end red_green_peaches_count_l2184_218483


namespace floor_abs_negative_real_l2184_218433

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end floor_abs_negative_real_l2184_218433


namespace locus_is_ellipse_l2184_218449

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Defines the locus equation for a point P relative to an isosceles triangle -/
def locusEquation (P : Point) (triangle : IsoscelesTriangle) (k : ℝ) : Prop :=
  3 * P.x^2 + 2 * P.y^2 - 2 * triangle.height * P.y + triangle.height^2 + 
  2 * (triangle.base / 2)^2 = k * triangle.side^2

/-- States that the locus of points satisfying the equation forms an ellipse -/
theorem locus_is_ellipse (triangle : IsoscelesTriangle) (k : ℝ) 
    (h_k : k > 1) (h_side : triangle.side^2 = (triangle.base / 2)^2 + triangle.height^2) :
  ∃ (center : Point) (a b : ℝ), ∀ (P : Point),
    locusEquation P triangle k ↔ 
    (P.x - center.x)^2 / a^2 + (P.y - center.y)^2 / b^2 = 1 :=
  sorry

end locus_is_ellipse_l2184_218449


namespace sin_pi_minus_alpha_l2184_218495

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x^2 + y^2 = 25 ∧ 
   (∃ (r : Real), r > 0 ∧ x = r * (Real.cos α) ∧ y = r * (Real.sin α))) →
  Real.sin (π - α) = 3/5 := by
  sorry

end sin_pi_minus_alpha_l2184_218495


namespace rectangle_area_problem_l2184_218442

theorem rectangle_area_problem (x : ℝ) :
  x > 0 ∧
  (5 - (-1)) * (x - (-2)) = 66 →
  x = 9 := by
sorry

end rectangle_area_problem_l2184_218442


namespace B_complete_work_in_40_days_l2184_218419

/-- The number of days it takes A to complete the work alone -/
def A_days : ℝ := 45

/-- The number of days A and B work together -/
def together_days : ℝ := 9

/-- The number of days B works alone after A leaves -/
def B_alone_days : ℝ := 23

/-- The number of days it takes B to complete the work alone -/
def B_days : ℝ := 40

/-- Theorem stating that given the conditions, B can complete the work alone in 40 days -/
theorem B_complete_work_in_40_days :
  (together_days * (1 / A_days + 1 / B_days)) + (B_alone_days * (1 / B_days)) = 1 :=
by sorry

end B_complete_work_in_40_days_l2184_218419


namespace complex_modulus_equal_parts_l2184_218444

theorem complex_modulus_equal_parts (a : ℝ) :
  let z : ℂ := (1 + 2*I) * (a + I)
  (z.re = z.im) → Complex.abs z = 5 * Real.sqrt 2 := by
sorry

end complex_modulus_equal_parts_l2184_218444


namespace unique_solution_iff_a_eq_one_l2184_218405

/-- The equation has exactly one solution if and only if a = 1 -/
theorem unique_solution_iff_a_eq_one (a : ℝ) :
  (∃! x : ℝ, 5^(x^2 - 6*a*x + 9*a^2) = a*x^2 - 6*a^2*x + 9*a^3 + a^2 - 6*a + 6) ↔ a = 1 := by
  sorry

end unique_solution_iff_a_eq_one_l2184_218405


namespace ursulas_salads_l2184_218464

theorem ursulas_salads :
  ∀ (hot_dog_price salad_price : ℚ)
    (num_hot_dogs : ℕ)
    (initial_money change : ℚ),
  hot_dog_price = 3/2 →
  salad_price = 5/2 →
  num_hot_dogs = 5 →
  initial_money = 20 →
  change = 5 →
  ∃ (num_salads : ℕ),
    num_salads = 3 ∧
    initial_money - change = num_hot_dogs * hot_dog_price + num_salads * salad_price :=
by sorry

end ursulas_salads_l2184_218464


namespace min_sum_of_even_factors_l2184_218496

theorem min_sum_of_even_factors (a b : ℤ) : 
  Even a → Even b → a * b = 144 → (∀ x y : ℤ, Even x → Even y → x * y = 144 → a + b ≤ x + y) → a + b = -74 :=
sorry

end min_sum_of_even_factors_l2184_218496


namespace trig_expression_simplification_l2184_218489

theorem trig_expression_simplification :
  (Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + Real.sin (45 * π / 180) +
   Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + Real.sin (75 * π / 180) + Real.sin (85 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (15 * π / 180) * Real.cos (25 * π / 180)) = 8 := by
  sorry

end trig_expression_simplification_l2184_218489


namespace problem_statement_l2184_218440

/-- Given a function g : ℝ → ℝ satisfying the following conditions:
  1) For all x y : ℝ, 2 * x * g y = 3 * y * g x
  2) g 10 = 15
  Prove that g 5 = 45/4 -/
theorem problem_statement (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x) 
  (h2 : g 10 = 15) : 
  g 5 = 45/4 := by
  sorry

end problem_statement_l2184_218440


namespace maria_travel_fraction_l2184_218431

theorem maria_travel_fraction (total_distance : ℝ) (remaining_distance : ℝ) 
  (first_stop_fraction : ℝ) :
  total_distance = 480 →
  remaining_distance = 180 →
  remaining_distance = (1 - first_stop_fraction) * total_distance * (3/4) →
  first_stop_fraction = 1/2 := by
sorry

end maria_travel_fraction_l2184_218431


namespace stump_pulling_force_l2184_218453

/-- The force required to pull a stump varies inversely with the lever length -/
def inverse_variation (force length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem stump_pulling_force 
  (force_10 length_10 force_25 length_25 : ℝ)
  (h1 : force_10 = 180)
  (h2 : length_10 = 10)
  (h3 : length_25 = 25)
  (h4 : inverse_variation force_10 length_10)
  (h5 : inverse_variation force_25 length_25)
  : force_25 = 72 := by
sorry

end stump_pulling_force_l2184_218453


namespace min_value_sum_l2184_218473

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a) + (a^2 * b) / (18 * b * c)) ≥ 4/9 :=
by sorry

end min_value_sum_l2184_218473


namespace sqrt_combination_l2184_218488

theorem sqrt_combination (t : ℝ) : 
  (∃ k : ℝ, k * Real.sqrt 12 = Real.sqrt (2 * t - 1)) → 
  Real.sqrt 12 = 2 * Real.sqrt 3 → 
  t = 2 := by
sorry

end sqrt_combination_l2184_218488


namespace quadratic_function_property_l2184_218471

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(1) - f(-1) = -6, then b = -3 -/
theorem quadratic_function_property (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^2 + b * x + c
  (f 1 - f (-1) = -6) → b = -3 := by
  sorry

end quadratic_function_property_l2184_218471


namespace apple_orchard_composition_l2184_218448

/-- Represents the composition of an apple orchard -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- The number of pure Gala trees in an orchard with given conditions -/
def pure_gala_count (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.pure_fuji + o.cross_pollinated = 204 ∧
  o.pure_gala = 36

theorem apple_orchard_composition :
  ∃ (o : Orchard), pure_gala_count o :=
sorry

end apple_orchard_composition_l2184_218448


namespace continued_fraction_evaluation_l2184_218460

theorem continued_fraction_evaluation :
  2 + (3 / (4 + (5/6))) = 76/29 := by
  sorry

end continued_fraction_evaluation_l2184_218460


namespace union_equality_implies_a_equals_two_l2184_218446

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, 2+a}

theorem union_equality_implies_a_equals_two :
  ∀ a : ℝ, A a ∪ B a = A a → a = 2 :=
by sorry

end union_equality_implies_a_equals_two_l2184_218446
