import Mathlib

namespace paint_cans_used_l3932_393233

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  rooms : ℕ

/-- Represents the number of paint cans -/
structure PaintCans where
  count : ℕ

/-- The painting scenario -/
structure PaintingScenario where
  initialCapacity : PaintCapacity
  lostCans : PaintCans
  finalCapacity : PaintCapacity

/-- The theorem to prove -/
theorem paint_cans_used (scenario : PaintingScenario) 
  (h1 : scenario.initialCapacity.rooms = 40)
  (h2 : scenario.lostCans.count = 5)
  (h3 : scenario.finalCapacity.rooms = 30) :
  ∃ (usedCans : PaintCans), usedCans.count = 15 ∧ 
    usedCans.count * (scenario.initialCapacity.rooms / (scenario.initialCapacity.rooms - scenario.finalCapacity.rooms)) = scenario.finalCapacity.rooms :=
by sorry

end paint_cans_used_l3932_393233


namespace sixth_grade_forgot_homework_percentage_l3932_393290

/-- Represents the percentage of students who forgot their homework in a group -/
def forgot_homework_percentage (total : ℕ) (forgot : ℕ) : ℚ :=
  (forgot : ℚ) / (total : ℚ) * 100

/-- Calculates the total number of students who forgot their homework -/
def total_forgot (group_a_total : ℕ) (group_b_total : ℕ) 
  (group_a_forgot_percent : ℚ) (group_b_forgot_percent : ℚ) : ℕ :=
  (group_a_total * group_a_forgot_percent.num / group_a_forgot_percent.den).toNat +
  (group_b_total * group_b_forgot_percent.num / group_b_forgot_percent.den).toNat

theorem sixth_grade_forgot_homework_percentage :
  let group_a_total : ℕ := 20
  let group_b_total : ℕ := 80
  let group_a_forgot_percent : ℚ := 20 / 100
  let group_b_forgot_percent : ℚ := 15 / 100
  let total_students : ℕ := group_a_total + group_b_total
  let total_forgot : ℕ := total_forgot group_a_total group_b_total group_a_forgot_percent group_b_forgot_percent
  forgot_homework_percentage total_students total_forgot = 16 := by
sorry

end sixth_grade_forgot_homework_percentage_l3932_393290


namespace abc_value_l3932_393281

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (eq1 : a * (b + c) = 156)
  (eq2 : b * (c + a) = 168)
  (eq3 : c * (a + b) = 176) :
  a * b * c = 754 := by
sorry

end abc_value_l3932_393281


namespace work_completion_theorem_l3932_393248

/-- Represents the number of men originally employed -/
def original_men : ℕ := 17

/-- Represents the number of days originally required to finish the work -/
def original_days : ℕ := 8

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Theorem stating that the given conditions lead to the correct number of original men -/
theorem work_completion_theorem :
  (original_men * original_days = (original_men + additional_men) * (original_days - days_saved)) ∧
  (original_men ≥ 1) ∧
  (∀ m : ℕ, m < original_men →
    m * original_days ≠ (m + additional_men) * (original_days - days_saved)) :=
by sorry

end work_completion_theorem_l3932_393248


namespace perfect_square_factors_count_l3932_393202

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a/2 + 1) * ((b/2 + 1) * (c/2 + 1))

theorem perfect_square_factors_count :
  count_perfect_square_factors 10 12 15 = 336 := by sorry

end perfect_square_factors_count_l3932_393202


namespace f_2_equals_17_l3932_393255

/-- A function f with an extremum of 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2 - 1

/-- The function f has an extremum of 9 at x = 1 -/
def has_extremum_at_1 (a b : ℝ) : Prop :=
  f a b 1 = 9 ∧ ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1

theorem f_2_equals_17 (a b : ℝ) (h : has_extremum_at_1 a b) : f a b 2 = 17 := by
  sorry

end f_2_equals_17_l3932_393255


namespace ant_farm_problem_l3932_393264

/-- Represents the number of ants of a specific species on a given day -/
def ant_count (initial : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial * (growth_rate ^ days)

theorem ant_farm_problem :
  ∀ a b c : ℕ,
  a + b + c = 50 →
  ant_count a 2 4 + ant_count b 3 4 + ant_count c 5 4 = 6230 →
  ant_count a 2 4 = 736 :=
by
  sorry

#check ant_farm_problem

end ant_farm_problem_l3932_393264


namespace least_five_digit_palindrome_div_25_l3932_393258

/-- A function that checks if a natural number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 10201 is the least five-digit palindrome divisible by 25 -/
theorem least_five_digit_palindrome_div_25 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 25 = 0 → n ≥ 10201 :=
sorry

end least_five_digit_palindrome_div_25_l3932_393258


namespace a_5_plus_a_6_equals_152_l3932_393230

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 3 * n + 1

-- Define the partial sum S_n
def S (n : ℕ) : ℤ := n^3

-- State the theorem
theorem a_5_plus_a_6_equals_152 : a 5 + a 6 = 152 := by
  sorry

end a_5_plus_a_6_equals_152_l3932_393230


namespace lydia_apple_eating_age_l3932_393250

/-- The age at which Lydia will first eat an apple from her tree -/
def apple_eating_age (tree_maturity : ℕ) (planting_age : ℕ) : ℕ :=
  planting_age + tree_maturity

theorem lydia_apple_eating_age
  (tree_maturity : ℕ)
  (planting_age : ℕ)
  (current_age : ℕ)
  (h1 : tree_maturity = 7)
  (h2 : planting_age = 4)
  (h3 : current_age = 9) :
  apple_eating_age tree_maturity planting_age = 11 := by
sorry

end lydia_apple_eating_age_l3932_393250


namespace system_solution_l3932_393218

theorem system_solution : ∃ (x y : ℝ), 
  (2 * x^2 - 3 * x * y + y^2 = 3 ∧ x^2 + 2 * x * y - 2 * y^2 = 6) ∧
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) := by
  sorry

end system_solution_l3932_393218


namespace arithmetic_sequence_10th_term_l3932_393243

/-- An arithmetic sequence with 30 terms, first term 3, and last term 89 has its 10th term equal to 30 -/
theorem arithmetic_sequence_10th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ i j : ℕ, i < j → a j - a i = (j - i) * (a 1 - a 0)) →  -- arithmetic sequence
    (a 0 = 3) →                                               -- first term is 3
    (a 29 = 89) →                                             -- last term is 89
    (a 9 = 30) :=                                             -- 10th term (index 9) is 30
by
  sorry

end arithmetic_sequence_10th_term_l3932_393243


namespace solution_set_inequality_l3932_393266

/-- Given that the solution set of x^2 - ax + b < 0 is (1,2), 
    prove that the solution set of 1/x < b/a is (-∞, 0) ∪ (3/2, +∞) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, 1/x < b/a ↔ x < 0 ∨ 3/2 < x) := by
sorry

end solution_set_inequality_l3932_393266


namespace inequality_proof_l3932_393270

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 8*c^2 - 3*a*b - 4*b*c - 2*c*a ≥ 0 := by
  sorry

end inequality_proof_l3932_393270


namespace wage_decrease_theorem_l3932_393285

theorem wage_decrease_theorem (x : ℝ) : 
  (100 - x) * 1.5 = 75 → x = 50 := by
  sorry

end wage_decrease_theorem_l3932_393285


namespace fraction_multiplication_l3932_393294

theorem fraction_multiplication (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) :
  (3 * a * b) / x * (2 * x^2) / (9 * a * b^2) = (2 * x) / (3 * b) := by
  sorry

end fraction_multiplication_l3932_393294


namespace sum_of_n_for_perfect_square_l3932_393222

theorem sum_of_n_for_perfect_square : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n < 2023 ∧ ∃ k : ℕ, 2 * n^2 + 3 * n = k^2) ∧
  (∀ n : ℕ, n < 2023 → (∃ k : ℕ, 2 * n^2 + 3 * n = k^2) → n ∈ S) ∧
  S.sum id = 444 := by
  sorry

end sum_of_n_for_perfect_square_l3932_393222


namespace shooting_scenarios_l3932_393252

theorem shooting_scenarios (n : ℕ) (n₁ n₂ n₃ n₄ : ℕ) 
  (h_total : n = n₁ + n₂ + n₃ + n₄)
  (h_n : n = 10)
  (h_n₁ : n₁ = 2)
  (h_n₂ : n₂ = 4)
  (h_n₃ : n₃ = 3)
  (h_n₄ : n₄ = 1) :
  (Nat.factorial n) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃ * Nat.factorial n₄) = 12600 :=
by sorry

end shooting_scenarios_l3932_393252


namespace shooting_game_propositions_l3932_393246

variable (p₁ p₂ : Prop)

theorem shooting_game_propositions :
  -- Both shots hit the airplane
  (p₁ ∧ p₂) = (p₁ ∧ p₂) ∧
  -- Both shots missed the airplane
  (¬p₁ ∧ ¬p₂) = (¬p₁ ∧ ¬p₂) ∧
  -- Exactly one shot hit the airplane
  ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) = ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) ∧
  -- At least one shot hit the airplane
  (p₁ ∨ p₂) = (p₁ ∨ p₂) := by sorry

end shooting_game_propositions_l3932_393246


namespace max_perimeter_triangle_l3932_393247

/-- Given a triangle with two sides of length 7 and 9 units, and the third side of length x units
    (where x is an integer), the maximum perimeter of the triangle is 31 units. -/
theorem max_perimeter_triangle (x : ℤ) : 
  (7 : ℝ) + 9 > x ∧ (7 : ℝ) + x > 9 ∧ (9 : ℝ) + x > 7 → 
  x > 0 →
  (∀ y : ℤ, ((7 : ℝ) + 9 > y ∧ (7 : ℝ) + y > 9 ∧ (9 : ℝ) + y > 7 → y ≤ x)) →
  (7 : ℝ) + 9 + x = 31 := by
  sorry

end max_perimeter_triangle_l3932_393247


namespace forty_people_skating_wheels_l3932_393217

/-- The number of wheels on the floor when a given number of people are roller skating. -/
def wheels_on_floor (people : ℕ) : ℕ :=
  people * 2 * 4

/-- Theorem: When 40 people are roller skating, there are 320 wheels on the floor. -/
theorem forty_people_skating_wheels : wheels_on_floor 40 = 320 := by
  sorry

#eval wheels_on_floor 40

end forty_people_skating_wheels_l3932_393217


namespace brick_length_proof_l3932_393275

/-- Given a courtyard and brick specifications, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 18 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  ∃ (brick_length : ℝ),
    brick_length = 0.2 ∧
    courtyard_length * courtyard_width * 10000 = 
      total_bricks * brick_length * brick_width := by
  sorry

end brick_length_proof_l3932_393275


namespace gratuity_percentage_l3932_393228

theorem gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℝ)
  (avg_cost_before_gratuity : ℝ)
  (h1 : num_people = 7)
  (h2 : total_bill = 840)
  (h3 : avg_cost_before_gratuity = 100) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) * 100 = 20 := by
sorry

end gratuity_percentage_l3932_393228


namespace polar_to_cartesian_l3932_393283

theorem polar_to_cartesian (θ : Real) (x y : Real) :
  x = (2 * Real.sin θ + 4 * Real.cos θ) * Real.cos θ ∧
  y = (2 * Real.sin θ + 4 * Real.cos θ) * Real.sin θ →
  (x - 2)^2 + (y - 1)^2 = 5 := by
  sorry

end polar_to_cartesian_l3932_393283


namespace prob_at_least_one_passes_eq_0_995_l3932_393225

/-- The probability that at least one candidate passes the test -/
def prob_at_least_one_passes (prob_A prob_B prob_C : ℝ) : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Theorem stating that the probability of at least one candidate passing is 0.995 -/
theorem prob_at_least_one_passes_eq_0_995 :
  prob_at_least_one_passes 0.9 0.8 0.75 = 0.995 := by
  sorry

#eval prob_at_least_one_passes 0.9 0.8 0.75

end prob_at_least_one_passes_eq_0_995_l3932_393225


namespace function_properties_l3932_393239

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- State the theorem
theorem function_properties (f : ℝ → ℝ) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (is_odd (fun x => f (x - a)) ∧ is_odd (fun x => f (x + a)) → has_period f (4 * a)) ∧
  (is_odd (fun x => f (x - a)) ∧ is_even (fun x => f (x - b)) → has_period f (4 * |a - b|)) :=
by sorry

end function_properties_l3932_393239


namespace star_five_three_l3932_393235

def star (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end star_five_three_l3932_393235


namespace incorrect_statement_l3932_393237

theorem incorrect_statement :
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → ¬(p ∧ q)) :=
sorry

end incorrect_statement_l3932_393237


namespace certain_number_is_sixteen_l3932_393259

theorem certain_number_is_sixteen :
  ∃ x : ℝ, (213 * x = 3408) ∧ (21.3 * x = 340.8) → x = 16 := by
  sorry

end certain_number_is_sixteen_l3932_393259


namespace novel_pages_l3932_393287

theorem novel_pages (planned_days : ℕ) (actual_days : ℕ) (extra_pages_per_day : ℕ) 
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : extra_pages_per_day = 20) : 
  (planned_days * ((actual_days * extra_pages_per_day) / (planned_days - actual_days))) = 1200 :=
by sorry

end novel_pages_l3932_393287


namespace unique_abc_solution_l3932_393280

/-- Represents a base-5 number with two digits -/
def BaseFiveNumber (a b : Nat) : Nat := 5 * a + b

/-- Represents a three-digit number in base 10 -/
def ThreeDigitNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

theorem unique_abc_solution :
  ∀ (A B C : Nat),
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 5 → B < 5 → C < 5 →
    A ≠ B → B ≠ C → A ≠ C →
    BaseFiveNumber A B + C = BaseFiveNumber C 0 →
    BaseFiveNumber A B + BaseFiveNumber B A = BaseFiveNumber C C →
    ThreeDigitNumber A B C = 323 := by
  sorry

end unique_abc_solution_l3932_393280


namespace largest_quotient_l3932_393298

def digits : List Nat := [4, 2, 8, 1, 9]

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  (∃ (d1 d2 d3 d4 d5 : Nat),
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
    a = 100 * d1 + 10 * d2 + d3 ∧ b = 10 * d4 + d5)

theorem largest_quotient :
  ∀ (a b : Nat), is_valid_pair a b →
  ∃ (q : Nat), a / b = q ∧ q ≤ 82 ∧
  (∀ (c d : Nat), is_valid_pair c d → c / d ≤ q) :=
sorry

end largest_quotient_l3932_393298


namespace judy_pencil_cost_l3932_393293

-- Define the given conditions
def pencils_per_week : ℕ := 10
def days_per_week : ℕ := 5
def pencils_per_pack : ℕ := 30
def cost_per_pack : ℕ := 4
def total_days : ℕ := 45

-- Define the theorem
theorem judy_pencil_cost :
  let pencils_per_day : ℚ := pencils_per_week / days_per_week
  let total_pencils : ℚ := pencils_per_day * total_days
  let packs_needed : ℚ := total_pencils / pencils_per_pack
  let total_cost : ℚ := packs_needed * cost_per_pack
  total_cost = 12 := by
  sorry

end judy_pencil_cost_l3932_393293


namespace box_volume_ratio_l3932_393241

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Alex's box dimensions -/
def alex_length : ℝ := 8
def alex_width : ℝ := 6
def alex_height : ℝ := 12

/-- Felicia's box dimensions -/
def felicia_length : ℝ := 12
def felicia_width : ℝ := 6
def felicia_height : ℝ := 8

theorem box_volume_ratio :
  (box_volume alex_length alex_width alex_height) / (box_volume felicia_length felicia_width felicia_height) = 1 := by
  sorry

end box_volume_ratio_l3932_393241


namespace money_distribution_l3932_393284

theorem money_distribution (p q r : ℚ) : 
  p + q + r = 5000 →
  r = (2/3) * (p + q) →
  r = 2000 := by
sorry

end money_distribution_l3932_393284


namespace min_distance_theorem_l3932_393272

/-- Given a line segment AB of length 2 with midpoint C, where A moves on the x-axis and B moves on the y-axis. -/
def line_segment (A B C : ℝ × ℝ) : Prop :=
  norm (A - B) = 2 ∧ C = (A + B) / 2 ∧ A.2 = 0 ∧ B.1 = 0

/-- The trajectory of point C is a circle with equation x² + y² = 1 -/
def trajectory (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 = 1

/-- The line √2ax + by = 1 intersects the trajectory at points C and D -/
def intersecting_line (a b : ℝ) (C D : ℝ × ℝ) : Prop :=
  trajectory C ∧ trajectory D ∧ 
  Real.sqrt 2 * a * C.1 + b * C.2 = 1 ∧
  Real.sqrt 2 * a * D.1 + b * D.2 = 1

/-- Triangle COD is a right-angled triangle with O as the origin -/
def right_triangle (C D : ℝ × ℝ) : Prop :=
  (C.1 * D.1 + C.2 * D.2) = 0

/-- Point P has coordinates (a, b) -/
def point_P (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (a, b)

/-- The main theorem: The minimum distance between P(a, b) and (0, 1) is √2 - 1 -/
theorem min_distance_theorem (A B C D P : ℝ × ℝ) (a b : ℝ) :
  line_segment A B C →
  trajectory C →
  intersecting_line a b C D →
  right_triangle C D →
  point_P P a b →
  (∃ (min_dist : ℝ), ∀ (a' b' : ℝ), 
    norm ((a', b') - (0, 1)) ≥ min_dist ∧
    min_dist = Real.sqrt 2 - 1) :=
sorry

end min_distance_theorem_l3932_393272


namespace specific_regular_polygon_l3932_393268

/-- Properties of a regular polygon -/
structure RegularPolygon where
  perimeter : ℝ
  side_length : ℝ
  sides : ℕ
  interior_angle : ℝ

/-- The theorem about the specific regular polygon -/
theorem specific_regular_polygon :
  ∃ (p : RegularPolygon),
    p.perimeter = 180 ∧
    p.side_length = 15 ∧
    p.sides = 12 ∧
    p.interior_angle = 150 := by
  sorry

end specific_regular_polygon_l3932_393268


namespace quadratic_equation_roots_l3932_393253

theorem quadratic_equation_roots :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 1/3 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_roots_l3932_393253


namespace min_value_ab_min_value_ab_is_nine_l3932_393208

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y - x*y + 3 = 0 → a*b ≤ x*y :=
by sorry

theorem min_value_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  a*b = 9 :=
by sorry

end min_value_ab_min_value_ab_is_nine_l3932_393208


namespace complex_modulus_problem_l3932_393240

theorem complex_modulus_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 5 / (2 - i) * i
  Complex.abs z = Real.sqrt 10 := by sorry

end complex_modulus_problem_l3932_393240


namespace fraction_power_multiply_l3932_393292

theorem fraction_power_multiply (a b c : ℚ) : 
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_power_multiply_l3932_393292


namespace one_prime_between_90_and_100_l3932_393232

theorem one_prime_between_90_and_100 : 
  ∃! p, Prime p ∧ 90 < p ∧ p < 100 := by
sorry

end one_prime_between_90_and_100_l3932_393232


namespace f_properties_l3932_393265

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x - a)

-- Define the derivative f'(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (1 + a) * x + a

theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. The derivative of f(x) is f'(x)
  (∀ x, deriv (f a) x = f_prime a x) ∧
  -- 2. f(x) has two different critical points
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f_prime a x₁ = 0 ∧ f_prime a x₂ = 0) ∧
  -- 3. f(x₁) + f(x₂) ≤ 0 holds if and only if a ≥ 2
  (∀ x₁ x₂, f_prime a x₁ = 0 → f_prime a x₂ = 0 → 
    (f a x₁ + f a x₂ ≤ 0 ↔ a ≥ 2)) := by
  sorry

end f_properties_l3932_393265


namespace triangle_existence_l3932_393295

theorem triangle_existence (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 12 ∧ c = y.val^2 ∧ 
   a + b > c ∧ a + c > b ∧ b + c > a) ↔ (y = 3 ∨ y = 4) :=
by sorry

end triangle_existence_l3932_393295


namespace tangent_line_to_circle_l3932_393203

/-- Given a point A(2,4) and a circle x^2 + y^2 = 4, 
    the tangent line from A to the circle has equation x = 2 or 3x - 4y + 10 = 0 -/
theorem tangent_line_to_circle (A : ℝ × ℝ) (circle : Set (ℝ × ℝ)) :
  A = (2, 4) →
  circle = {(x, y) | x^2 + y^2 = 4} →
  (∃ (k : ℝ), (∀ (x y : ℝ), (x, y) ∈ circle → 
    (x = 2 ∨ 3*x - 4*y + 10 = 0) ↔ 
    ((x - 2)^2 + (y - 4)^2 = ((x - 0)^2 + (y - 0)^2 - 4) / 4))) := by
  sorry


end tangent_line_to_circle_l3932_393203


namespace solution_set_of_inequality_l3932_393249

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-5 : ℝ) 3 \ {3} = {x | (x + 5) / (3 - x) ≥ 0} := by
  sorry

end solution_set_of_inequality_l3932_393249


namespace four_isosceles_triangles_l3932_393289

-- Define a Point type for 2D coordinates
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Triangle type
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Function to calculate the squared distance between two points
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := distanceSquared t.a t.b
  let d2 := distanceSquared t.b t.c
  let d3 := distanceSquared t.c t.a
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangleA : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangleB : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangleC : Triangle := ⟨⟨1, 2⟩, ⟨3, 1⟩, ⟨5, 2⟩⟩
def triangleD : Triangle := ⟨⟨7, 3⟩, ⟨6, 5⟩, ⟨9, 3⟩⟩
def triangleE : Triangle := ⟨⟨8, 2⟩, ⟨9, 4⟩, ⟨10, 1⟩⟩

-- Theorem stating that exactly 4 out of 5 triangles are isosceles
theorem four_isosceles_triangles :
  (isIsosceles triangleA ∧ 
   isIsosceles triangleB ∧ 
   isIsosceles triangleC ∧ 
   ¬isIsosceles triangleD ∧ 
   isIsosceles triangleE) :=
by sorry

end four_isosceles_triangles_l3932_393289


namespace intersection_of_A_and_B_l3932_393267

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < -3} := by sorry

end intersection_of_A_and_B_l3932_393267


namespace tens_digit_of_1998_pow_2003_minus_1995_l3932_393262

theorem tens_digit_of_1998_pow_2003_minus_1995 :
  (1998^2003 - 1995) % 100 / 10 = 0 := by
  sorry

end tens_digit_of_1998_pow_2003_minus_1995_l3932_393262


namespace area_of_triangle_BCD_l3932_393220

/-- Given a triangle ABC with area 36 and base 6, and a triangle BCD sharing the same height as ABC
    with base 34, prove that the area of triangle BCD is 204. -/
theorem area_of_triangle_BCD (area_ABC : ℝ) (base_AC : ℝ) (base_CD : ℝ) (height : ℝ) :
  area_ABC = 36 →
  base_AC = 6 →
  base_CD = 34 →
  area_ABC = (1/2) * base_AC * height →
  (1/2) * base_CD * height = 204 := by
  sorry

end area_of_triangle_BCD_l3932_393220


namespace power_difference_divisibility_l3932_393273

theorem power_difference_divisibility (a b : ℤ) (h : 100 ∣ (a - b)) :
  10000 ∣ (a^100 - b^100) := by
  sorry

end power_difference_divisibility_l3932_393273


namespace largest_angle_in_triangle_l3932_393282

theorem largest_angle_in_triangle (α β γ : Real) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = (7/5) * 90 →  -- Two angles sum to 7/5 of a right angle
  β = α + 45 →  -- One angle is 45° larger than the other
  max α (max β γ) = 85.5 :=  -- The largest angle is 85.5°
by sorry

end largest_angle_in_triangle_l3932_393282


namespace tom_search_cost_l3932_393296

/-- Calculates the total cost for Tom's search service given the number of days -/
def total_cost (days : ℕ) : ℕ :=
  if days ≤ 5 then
    100 * days
  else
    500 + 60 * (days - 5)

/-- The problem statement -/
theorem tom_search_cost : total_cost 10 = 800 := by
  sorry

end tom_search_cost_l3932_393296


namespace mashed_potatoes_count_mashed_potatoes_proof_l3932_393227

theorem mashed_potatoes_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference mashed_count =>
    (bacon_count = 269) →
    (difference = 61) →
    (mashed_count = bacon_count + difference) →
    (mashed_count = 330)

-- The proof is omitted
theorem mashed_potatoes_proof : mashed_potatoes_count 269 61 330 := by
  sorry

end mashed_potatoes_count_mashed_potatoes_proof_l3932_393227


namespace abc_product_l3932_393276

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : 1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) :
  a * b * c = 1912 := by
  sorry

end abc_product_l3932_393276


namespace overlapping_segment_length_l3932_393206

theorem overlapping_segment_length (tape_length : ℝ) (total_length : ℝ) (num_tapes : ℕ) :
  tape_length = 250 →
  total_length = 925 →
  num_tapes = 4 →
  ∃ (overlap_length : ℝ),
    overlap_length * (num_tapes - 1) = num_tapes * tape_length - total_length ∧
    overlap_length = 25 := by
  sorry

end overlapping_segment_length_l3932_393206


namespace gcd_1729_587_l3932_393223

theorem gcd_1729_587 : Nat.gcd 1729 587 = 1 := by
  sorry

end gcd_1729_587_l3932_393223


namespace best_of_five_more_advantageous_l3932_393251

/-- The probability of the stronger player winning in a best-of-three format -/
def prob_best_of_three (p : ℝ) : ℝ := p^2 + 2*p^2*(1-p)

/-- The probability of the stronger player winning in a best-of-five format -/
def prob_best_of_five (p : ℝ) : ℝ := p^3 + 3*p^3*(1-p) + 6*p^3*(1-p)^2

/-- Theorem stating that the best-of-five format is more advantageous for selecting the strongest player -/
theorem best_of_five_more_advantageous (p : ℝ) (h : 0.5 < p ∧ p ≤ 1) :
  prob_best_of_three p < prob_best_of_five p :=
sorry

end best_of_five_more_advantageous_l3932_393251


namespace class1_participants_l3932_393201

/-- The number of students in Class 1 -/
def class1_students : ℕ := 40

/-- The number of students in Class 2 -/
def class2_students : ℕ := 36

/-- The number of students in Class 3 -/
def class3_students : ℕ := 44

/-- The total number of students who did not participate in the competition -/
def non_participants : ℕ := 30

/-- The proportion of students participating in the competition -/
def participation_rate : ℚ := 3/4

theorem class1_participants :
  (class1_students : ℚ) * participation_rate = 30 :=
sorry

end class1_participants_l3932_393201


namespace sum_of_15th_set_l3932_393209

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l3932_393209


namespace prove_nested_max_min_l3932_393219

/-- Given distinct real numbers p, q, r, s, t satisfying p < q < r < s < t,
    prove that M(M(p, m(q, s)), m(r, m(p, t))) = q -/
theorem prove_nested_max_min (p q r s t : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) 
  (h_order : p < q ∧ q < r ∧ r < s ∧ s < t) : 
  max (max p (min q s)) (min r (min p t)) = q := by
  sorry

end prove_nested_max_min_l3932_393219


namespace equation_solution_l3932_393278

theorem equation_solution : 
  ∀ x : ℝ, (x + 2) * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 1 := by
sorry

end equation_solution_l3932_393278


namespace josh_money_left_l3932_393226

/-- The amount of money Josh has left after selling bracelets and buying cookies -/
def money_left (cost_per_bracelet : ℚ) (sell_price : ℚ) (num_bracelets : ℕ) (cookie_cost : ℚ) : ℚ :=
  (sell_price - cost_per_bracelet) * num_bracelets - cookie_cost

/-- Theorem stating that Josh has $3 left after selling bracelets and buying cookies -/
theorem josh_money_left :
  money_left 1 1.5 12 3 = 3 := by
  sorry

end josh_money_left_l3932_393226


namespace real_roots_iff_k_leq_five_l3932_393277

theorem real_roots_iff_k_leq_five (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end real_roots_iff_k_leq_five_l3932_393277


namespace largest_equal_cost_number_l3932_393279

/-- Cost calculation for Option 1 -/
def option1Cost (n : ℕ) : ℕ :=
  n.digits 10
    |> List.foldl (fun acc d => acc + if d % 2 = 0 then 2 * d else d) 0

/-- Cost calculation for Option 2 -/
def option2Cost (n : ℕ) : ℕ :=
  n.digits 2
    |> List.foldl (fun acc d => acc + if d = 1 then 2 else 1) 0

/-- Theorem stating that 237 is the largest number less than 500 with equal costs -/
theorem largest_equal_cost_number :
  ∀ n : ℕ, n < 500 → n > 237 →
    option1Cost n ≠ option2Cost n :=
by
  sorry

#eval option1Cost 237
#eval option2Cost 237

end largest_equal_cost_number_l3932_393279


namespace factorial_ratio_equals_sixty_sevenths_l3932_393263

theorem factorial_ratio_equals_sixty_sevenths :
  (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_ratio_equals_sixty_sevenths_l3932_393263


namespace composition_equation_solution_l3932_393212

theorem composition_equation_solution (α β : ℝ → ℝ) (h1 : ∀ x, α x = 4 * x + 9) 
  (h2 : ∀ x, β x = 9 * x + 6) (h3 : α (β x) = 8) : x = -25/36 := by
  sorry

end composition_equation_solution_l3932_393212


namespace best_optimistic_coefficient_l3932_393261

theorem best_optimistic_coefficient 
  (a b c x : ℝ) 
  (h1 : a < b) 
  (h2 : 0 < x) 
  (h3 : x < 1) 
  (h4 : c = a + x * (b - a)) 
  (h5 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end best_optimistic_coefficient_l3932_393261


namespace bookshelf_theorem_l3932_393221

def bookshelf_problem (yoongi_notebooks jungkook_notebooks hoseok_notebooks : ℕ) : Prop :=
  yoongi_notebooks = 3 ∧ jungkook_notebooks = 3 ∧ hoseok_notebooks = 3 →
  yoongi_notebooks + jungkook_notebooks + hoseok_notebooks = 9

theorem bookshelf_theorem : bookshelf_problem 3 3 3 := by
  sorry

end bookshelf_theorem_l3932_393221


namespace decreasing_function_a_range_l3932_393257

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1 / 9 : ℝ) (1 / 5 : ℝ) := by
  sorry


end decreasing_function_a_range_l3932_393257


namespace game_converges_to_black_hole_l3932_393260

/-- Represents a three-digit number in the game --/
structure GameNumber :=
  (hundreds : Nat)
  (tens : Nat)
  (ones : Nat)

/-- Counts the number of even digits in a natural number --/
def countEvenDigits (n : Nat) : Nat :=
  sorry

/-- Counts the number of odd digits in a natural number --/
def countOddDigits (n : Nat) : Nat :=
  sorry

/-- Counts the total number of digits in a natural number --/
def countDigits (n : Nat) : Nat :=
  sorry

/-- Converts a natural number to a GameNumber --/
def natToGameNumber (n : Nat) : GameNumber :=
  { hundreds := countEvenDigits n,
    tens := countOddDigits n,
    ones := countDigits n }

/-- Converts a GameNumber to a natural number --/
def gameNumberToNat (g : GameNumber) : Nat :=
  g.hundreds * 100 + g.tens * 10 + g.ones

/-- Applies one step of the game rules --/
def gameStep (n : Nat) : Nat :=
  gameNumberToNat (natToGameNumber n)

/-- The final number reached in the game --/
def blackHoleNumber : Nat := 123

/-- Theorem: The game always ends with the black hole number --/
theorem game_converges_to_black_hole (start : Nat) : 
  ∃ k : Nat, (gameStep^[k] start) = blackHoleNumber :=
sorry

end game_converges_to_black_hole_l3932_393260


namespace tangent_line_at_one_l3932_393224

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x else -Real.log x + 3 * x

-- State the theorem
theorem tangent_line_at_one (h : ∀ x, f (-x) = -f x) :
  let tangent_line (x : ℝ) := 2 * x + 1
  ∀ x, tangent_line x = f 1 + (tangent_line 1 - f 1) * (x - 1) := by
  sorry

end tangent_line_at_one_l3932_393224


namespace converse_is_false_l3932_393204

theorem converse_is_false : ¬∀ x : ℝ, x > 0 → x - 3 > 0 := by sorry

end converse_is_false_l3932_393204


namespace youngest_child_age_l3932_393216

/-- Represents the age of the youngest child -/
def youngest_age : ℕ → Prop := λ x =>
  -- There are 5 children
  -- Children are born at intervals of 3 years each
  -- The sum of their ages is 60 years
  x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60

/-- Proves that the age of the youngest child is 6 years -/
theorem youngest_child_age : youngest_age 6 := by
  sorry

end youngest_child_age_l3932_393216


namespace interior_edge_sum_is_ten_l3932_393207

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  width : ℝ
  outerLength : ℝ
  outerWidth : ℝ
  frameArea : ℝ

/-- Calculates the sum of interior edge lengths of a picture frame -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * (frame.outerLength - 2 * frame.width) + 2 * (frame.outerWidth - 2 * frame.width)

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 10 inches -/
theorem interior_edge_sum_is_ten (frame : PictureFrame) 
    (h1 : frame.width = 2)
    (h2 : frame.outerLength = 7)
    (h3 : frame.frameArea = 36)
    (h4 : frame.frameArea = frame.outerLength * frame.outerWidth - (frame.outerLength - 2 * frame.width) * (frame.outerWidth - 2 * frame.width)) :
  interiorEdgeSum frame = 10 := by
  sorry

#check interior_edge_sum_is_ten

end interior_edge_sum_is_ten_l3932_393207


namespace selection_count_l3932_393254

def class_size : ℕ := 38
def selection_size : ℕ := 5
def remaining_students : ℕ := class_size - 2  -- Excluding students A and B
def remaining_selection_size : ℕ := selection_size - 1  -- We always select student A

def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count : 
  binomial remaining_students remaining_selection_size = 58905 := by
  sorry

end selection_count_l3932_393254


namespace quadratic_roots_imply_negative_a_abs_function_solutions_l3932_393271

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 3) * x + a

-- Define the absolute value function
def abs_function (x : ℝ) : ℝ := |3 - x^2|

theorem quadratic_roots_imply_negative_a (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ quadratic a x = 0 ∧ quadratic a y = 0) → a < 0 :=
sorry

theorem abs_function_solutions (a : ℝ) :
  ¬(∃! x : ℝ, abs_function x = a) :=
sorry

end quadratic_roots_imply_negative_a_abs_function_solutions_l3932_393271


namespace sqrt_fifth_power_of_sqrt5_to_4th_l3932_393286

theorem sqrt_fifth_power_of_sqrt5_to_4th : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 4 = 9765625 := by
  sorry

end sqrt_fifth_power_of_sqrt5_to_4th_l3932_393286


namespace train_crossing_time_l3932_393288

/-- Proves that a train with given length and speed takes a specific time to cross a pole --/
theorem train_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 125) 
  (h2 : train_speed_kmh = 90) : 
  train_length / (train_speed_kmh * 1000 / 3600) = 5 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3932_393288


namespace quadratic_equation_properties_l3932_393210

theorem quadratic_equation_properties (m : ℝ) :
  (∀ m, ∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0) ∧
  (∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0 ∧ x > 6 → m > 8) := by
  sorry

end quadratic_equation_properties_l3932_393210


namespace least_distinct_values_l3932_393269

/-- Given a list of positive integers with the specified properties,
    the least number of distinct values is 218. -/
theorem least_distinct_values (list : List ℕ+) : 
  (list.length = 3042) →
  (∃! m, list.count m = 15 ∧ ∀ n, list.count n ≤ list.count m) →
  (list.toFinset.card ≥ 218 ∧ ∀ k, k < 218 → ¬(list.toFinset.card = k)) := by
  sorry

end least_distinct_values_l3932_393269


namespace mans_age_twice_sons_age_l3932_393213

theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 16 →
  age_difference = 18 →
  (son_age + age_difference + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end mans_age_twice_sons_age_l3932_393213


namespace quarter_circle_roll_path_length_l3932_393291

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (π * r / 4)
  path_length = 4.5 := by sorry

end quarter_circle_roll_path_length_l3932_393291


namespace brendan_lawn_cutting_l3932_393299

/-- The number of yards Brendan could cut per day before buying the lawnmower -/
def initial_yards : ℝ := 8

/-- The increase in cutting capacity after buying the lawnmower -/
def capacity_increase : ℝ := 0.5

/-- The number of days Brendan worked with the new lawnmower -/
def days_worked : ℕ := 7

/-- The total number of yards cut with the new lawnmower -/
def total_yards_cut : ℕ := 84

theorem brendan_lawn_cutting :
  initial_yards * (1 + capacity_increase) * days_worked = total_yards_cut :=
by sorry

end brendan_lawn_cutting_l3932_393299


namespace seventh_term_of_specific_geometric_sequence_l3932_393244

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The seventh term of a geometric sequence with first term 3 and second term -3/2 is 3/64 -/
theorem seventh_term_of_specific_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -3/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 3/64 := by
sorry

end seventh_term_of_specific_geometric_sequence_l3932_393244


namespace leaves_collection_time_l3932_393229

/-- The time taken to collect leaves under given conditions -/
def collect_leaves_time (total_leaves : ℕ) (collect_rate : ℕ) (scatter_rate : ℕ) (cycle_time : ℚ) : ℚ :=
  let net_increase := collect_rate - scatter_rate
  let full_cycles := (total_leaves - net_increase) / net_increase
  (full_cycles * cycle_time + cycle_time) / 60

/-- The problem statement -/
theorem leaves_collection_time :
  collect_leaves_time 50 5 3 (45 / 60) = 75 / 4 :=
sorry

end leaves_collection_time_l3932_393229


namespace gcf_154_252_l3932_393245

theorem gcf_154_252 : Nat.gcd 154 252 = 14 := by
  sorry

end gcf_154_252_l3932_393245


namespace area_of_ω_l3932_393200

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (14, 9)

-- Assume A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Assume the intersection point of tangents is on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : 
  |circle_area ω - 154.73 * Real.pi| < 0.01 := sorry

end area_of_ω_l3932_393200


namespace absolute_value_inequality_l3932_393211

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x + 2| > 3} = solution_set :=
by sorry

end absolute_value_inequality_l3932_393211


namespace product_of_five_reals_l3932_393256

theorem product_of_five_reals (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h1 : a * b + b = a * c + a)
  (h2 : b * c + c = b * d + b)
  (h3 : c * d + d = c * e + c)
  (h4 : d * e + e = d * a + d) :
  a * b * c * d * e = 1 := by
sorry

end product_of_five_reals_l3932_393256


namespace gift_wrapping_combinations_l3932_393215

/-- Represents the number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- Represents the number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- Represents the number of types of gift cards -/
def gift_card_types : ℕ := 4

/-- Represents the number of designs of gift tags -/
def gift_tag_designs : ℕ := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_designs

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations : total_combinations = 600 := by sorry

end gift_wrapping_combinations_l3932_393215


namespace z_in_fourth_quadrant_l3932_393205

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end z_in_fourth_quadrant_l3932_393205


namespace smallest_four_digit_divisible_by_53_l3932_393234

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l3932_393234


namespace sin_negative_120_degrees_l3932_393297

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_120_degrees_l3932_393297


namespace x_equals_six_l3932_393214

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem x_equals_six : ∃ x : ℕ, x * factorial x + 2 * factorial x = 40320 ∧ x = 6 := by
  sorry

end x_equals_six_l3932_393214


namespace rectangle_tileable_iff_divisible_l3932_393236

/-- An (0, b)-tile is a 2 × b rectangle. -/
structure ZeroBTile (b : ℕ) :=
  (width : Fin 2)
  (height : Fin b)

/-- A tiling of an m × n rectangle with (0, b)-tiles. -/
def Tiling (m n b : ℕ) := List (ZeroBTile b)

/-- Predicate to check if a tiling is valid for an m × n rectangle. -/
def IsValidTiling (m n b : ℕ) (t : Tiling m n b) : Prop :=
  sorry  -- Definition of valid tiling omitted for brevity

/-- An m × n rectangle is (0, b)-tileable if there exists a valid tiling. -/
def IsTileable (m n b : ℕ) : Prop :=
  ∃ t : Tiling m n b, IsValidTiling m n b t

/-- Main theorem: An m × n rectangle is (0, b)-tileable iff 2b divides m or 2b divides n. -/
theorem rectangle_tileable_iff_divisible (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  IsTileable m n b ↔ (2 * b ∣ m) ∨ (2 * b ∣ n) :=
sorry

end rectangle_tileable_iff_divisible_l3932_393236


namespace sin_geq_x_on_unit_interval_l3932_393238

theorem sin_geq_x_on_unit_interval (x : Real) (h : x ∈ Set.Icc 0 1) :
  Real.sqrt 2 * Real.sin x ≥ x := by
  sorry

end sin_geq_x_on_unit_interval_l3932_393238


namespace square_1369_product_l3932_393231

theorem square_1369_product (x : ℤ) (h : x^2 = 1369) : (x + 3) * (x - 3) = 1360 := by
  sorry

end square_1369_product_l3932_393231


namespace coin_toss_problem_l3932_393274

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem coin_toss_problem (n : ℕ) :
  binomial_probability n 3 0.5 = 0.25 → n = 4 := by
  sorry

end coin_toss_problem_l3932_393274


namespace subtraction_preserves_inequality_l3932_393242

theorem subtraction_preserves_inequality (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end subtraction_preserves_inequality_l3932_393242
