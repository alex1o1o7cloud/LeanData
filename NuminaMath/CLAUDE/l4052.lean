import Mathlib

namespace isosceles_right_triangle_hypotenuse_l4052_405259

/-- An isosceles right triangle with perimeter 14 + 14√2 has a hypotenuse of length 28 -/
theorem isosceles_right_triangle_hypotenuse : ∀ a c : ℝ,
  a > 0 → c > 0 →
  a = c / Real.sqrt 2 →  -- Condition for isosceles right triangle
  2 * a + c = 14 + 14 * Real.sqrt 2 →  -- Perimeter condition
  c = 28 := by
  sorry

end isosceles_right_triangle_hypotenuse_l4052_405259


namespace base_subtraction_l4052_405288

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base_8_num := to_base_10 [0, 1, 2, 3, 4, 5] 8
  let base_9_num := to_base_10 [2, 3, 4, 5, 6] 9
  base_8_num - base_9_num = 136532 := by
  sorry

end base_subtraction_l4052_405288


namespace triangle_area_in_circle_l4052_405279

theorem triangle_area_in_circle (r : ℝ) (h : r = 3) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
    a = b ∧ c = a * Real.sqrt 2 ∧  -- Sides are in ratio 1:1:√2
    c = 2 * r ∧  -- Diameter of circle
    (1/2) * a * b = 9 := by  -- Area of triangle
  sorry

end triangle_area_in_circle_l4052_405279


namespace smallest_five_digit_multiple_with_16_factors_l4052_405217

-- Define a function to count the number of factors of a natural number
def count_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is five digits
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_five_digit_multiple_with_16_factors : 
  ∀ n : ℕ, is_five_digit n → n % 2014 = 0 → 
  count_factors (n % 1000) = 16 → n ≥ 24168 := by sorry

end smallest_five_digit_multiple_with_16_factors_l4052_405217


namespace factor_sum_l4052_405209

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 2*X + 5) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 31 := by
sorry

end factor_sum_l4052_405209


namespace product_equals_sum_of_squares_l4052_405210

theorem product_equals_sum_of_squares 
  (nums : List ℕ) 
  (count : nums.length = 116) 
  (sum_of_squares : (nums.map (λ x => x^2)).sum = 144) : 
  nums.prod = 144 := by
sorry

end product_equals_sum_of_squares_l4052_405210


namespace sector_angle_measure_l4052_405229

theorem sector_angle_measure (r : ℝ) (α : ℝ) 
  (h1 : α * r = 2)  -- arc length = 2
  (h2 : (1/2) * α * r^2 = 2)  -- area = 2
  : α = 1 := by
sorry

end sector_angle_measure_l4052_405229


namespace sally_lost_balloons_l4052_405231

/-- Given that Sally initially had 9 orange balloons and now has 7 orange balloons,
    prove that she lost 2 orange balloons. -/
theorem sally_lost_balloons (initial : Nat) (current : Nat) 
    (h1 : initial = 9) (h2 : current = 7) : initial - current = 2 := by
  sorry

end sally_lost_balloons_l4052_405231


namespace frustum_volume_ratio_l4052_405296

theorem frustum_volume_ratio (h₁ h₂ : ℝ) (A₁ A₂ : ℝ) (V₁ V₂ : ℝ) :
  h₁ / h₂ = 3 / 5 →
  A₁ / A₂ = 9 / 25 →
  V₁ = (1 / 3) * h₁ * A₁ →
  V₂ = (1 / 3) * h₂ * A₂ →
  V₁ / (V₂ - V₁) = 27 / 71 :=
by sorry

end frustum_volume_ratio_l4052_405296


namespace x_squared_when_y_is_4_l4052_405289

-- Define the inverse variation relationship between x² and y³
def inverse_variation (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x^2 * y^3 = k

-- State the theorem
theorem x_squared_when_y_is_4
  (h1 : ∀ x y, inverse_variation x y)
  (h2 : inverse_variation 10 2) :
  ∃ x : ℝ, inverse_variation x 4 ∧ x^2 = 12.5 := by
sorry


end x_squared_when_y_is_4_l4052_405289


namespace largest_percent_error_circle_area_l4052_405268

/-- The largest possible percent error in the computed area of a circle -/
theorem largest_percent_error_circle_area (actual_circumference : ℝ) (max_error_percent : ℝ) :
  actual_circumference = 30 →
  max_error_percent = 15 →
  ∃ (computed_area actual_area : ℝ),
    computed_area ≠ actual_area ∧
    abs ((computed_area - actual_area) / actual_area) ≤ 0.3225 ∧
    ∀ (other_area : ℝ),
      abs ((other_area - actual_area) / actual_area) ≤ abs ((computed_area - actual_area) / actual_area) :=
by sorry

end largest_percent_error_circle_area_l4052_405268


namespace zuzkas_number_l4052_405220

theorem zuzkas_number : ∃! n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ 
  10 * n + 1 = 3 * (100000 + n) := by
sorry

end zuzkas_number_l4052_405220


namespace simplify_product_of_sqrt_l4052_405293

theorem simplify_product_of_sqrt (y : ℝ) (hy : y > 0) :
  Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y) := by
  sorry

end simplify_product_of_sqrt_l4052_405293


namespace total_amount_spent_l4052_405225

/-- Calculates the total amount spent on a meal given the base food price, sales tax rate, and tip rate. -/
theorem total_amount_spent
  (food_price : ℝ)
  (sales_tax_rate : ℝ)
  (tip_rate : ℝ)
  (h1 : food_price = 150)
  (h2 : sales_tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  food_price * (1 + sales_tax_rate) * (1 + tip_rate) = 198 :=
by sorry

end total_amount_spent_l4052_405225


namespace blue_face_probability_l4052_405221

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)
  (face_sum : blue_faces + red_faces + green_faces = total_faces)

def roll_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

theorem blue_face_probability (o : Octahedron) 
  (h1 : o.total_faces = 8)
  (h2 : o.blue_faces = 4)
  (h3 : o.red_faces = 3)
  (h4 : o.green_faces = 1) :
  roll_probability o = 1/2 := by
  sorry

end blue_face_probability_l4052_405221


namespace ittymangnark_catch_l4052_405200

/-- Represents the number of fish each family member and pet receives -/
structure FishDistribution where
  ittymangnark : ℕ
  kingnook : ℕ
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Represents the distribution of fish eyes -/
structure EyeDistribution where
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Theorem stating that given the fish and eye distribution, Ittymangnark caught 21 fish -/
theorem ittymangnark_catch (fish : FishDistribution) (eyes : EyeDistribution) :
  fish.ittymangnark = 3 →
  fish.kingnook = 4 →
  fish.oomyapeck = 1 →
  fish.yurraknalik = 2 →
  fish.ankaq = 1 →
  fish.nanuq = 3 →
  eyes.oomyapeck = 24 →
  eyes.yurraknalik = 4 →
  eyes.ankaq = 6 →
  eyes.nanuq = 8 →
  fish.ittymangnark + fish.kingnook + fish.oomyapeck + fish.yurraknalik + fish.ankaq + fish.nanuq = 21 :=
by
  sorry

end ittymangnark_catch_l4052_405200


namespace no_triple_exists_l4052_405203

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  ∃ (k : ℕ+), k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 := by
  sorry

end no_triple_exists_l4052_405203


namespace two_lines_theorem_l4052_405260

/-- Two lines in the plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ
  l₂ : ℝ → ℝ → ℝ
  a : ℝ
  b : ℝ
  h₁ : ∀ x y, l₁ x y = a * x - b * y + 4
  h₂ : ∀ x y, l₂ x y = (a - 1) * x + y + 2

/-- Scenario 1: l₁ passes through (-3,-1) and is perpendicular to l₂ -/
def scenario1 (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0 ∧ 
  (lines.a / lines.b) * (1 - lines.a) = -1

/-- Scenario 2: l₁ is parallel to l₂ and has y-intercept -3 -/
def scenario2 (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a ∧
  4 / lines.b = -3

theorem two_lines_theorem (lines : TwoLines) :
  (scenario1 lines → lines.a = 2 ∧ lines.b = 2) ∧
  (scenario2 lines → lines.a = 4 ∧ lines.b = -4/3) := by
  sorry

end two_lines_theorem_l4052_405260


namespace expression_equality_l4052_405290

theorem expression_equality (x : ℝ) : 3 * x * (21 - (x + 3) * x - 3) = 54 * x - 3 * x^3 + 9 * x^2 := by
  sorry

end expression_equality_l4052_405290


namespace website_earnings_l4052_405263

/-- John's website earnings problem -/
theorem website_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ)
  (h1 : visits_per_month = 30000)
  (h2 : days_per_month = 30)
  (h3 : earnings_per_visit = 1 / 100) :
  (visits_per_month : ℚ) * earnings_per_visit / days_per_month = 10 := by
  sorry

end website_earnings_l4052_405263


namespace expansion_has_four_terms_l4052_405264

/-- The expression after substituting 2x for the asterisk and expanding -/
def expanded_expression (x : ℝ) : ℝ := x^6 + x^4 + 4*x^2 + 4

/-- The original expression with the asterisk replaced by 2x -/
def original_expression (x : ℝ) : ℝ := (x^3 - 2)^2 + (x^2 + 2*x)^2

theorem expansion_has_four_terms :
  ∀ x : ℝ, original_expression x = expanded_expression x ∧ 
  (∃ a b c d : ℝ, expanded_expression x = a*x^6 + b*x^4 + c*x^2 + d ∧ 
   a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :=
by sorry

end expansion_has_four_terms_l4052_405264


namespace problem_solution_l4052_405262

theorem problem_solution (a b c x y z : ℝ) 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5)
  (h2 : c / x + a / y + b / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := by
sorry

end problem_solution_l4052_405262


namespace prob_A_win_match_is_correct_l4052_405255

/-- The probability of player A winning a single game -/
def prob_A_win : ℝ := 0.6

/-- The probability of player B winning a single game -/
def prob_B_win : ℝ := 0.4

/-- The probability of player A winning the match after winning the first game -/
def prob_A_win_match : ℝ := prob_A_win + prob_B_win * prob_A_win

/-- Theorem stating that the probability of A winning the match after winning the first game is 0.84 -/
theorem prob_A_win_match_is_correct : prob_A_win_match = 0.84 := by
  sorry

end prob_A_win_match_is_correct_l4052_405255


namespace largest_divisor_of_five_consecutive_integers_l4052_405212

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 120 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 120 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end largest_divisor_of_five_consecutive_integers_l4052_405212


namespace prime_pairs_congruence_l4052_405269

theorem prime_pairs_congruence (p q : ℕ) : 
  Prime p ∧ Prime q →
  (∀ x : ℤ, x^(3*p*q) ≡ x [ZMOD (3*p*q)]) →
  ((p = 11 ∧ q = 17) ∨ (p = 17 ∧ q = 11)) := by
  sorry

end prime_pairs_congruence_l4052_405269


namespace election_votes_l4052_405266

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 : ℚ) / 100 * total_votes - (40 : ℚ) / 100 * total_votes = 288) : 
  (60 : ℚ) / 100 * total_votes = 864 := by
sorry

end election_votes_l4052_405266


namespace parabola_focus_coordinates_l4052_405261

/-- A parabola is defined by its vertex, directrix, and focus. -/
structure Parabola where
  vertex : ℝ × ℝ
  directrix : ℝ
  focus : ℝ × ℝ

/-- Given a parabola with vertex at (2,0) and directrix x = -1, its focus is at (5,0). -/
theorem parabola_focus_coordinates :
  ∀ p : Parabola, p.vertex = (2, 0) ∧ p.directrix = -1 → p.focus = (5, 0) := by
  sorry

end parabola_focus_coordinates_l4052_405261


namespace inequality_proof_l4052_405282

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end inequality_proof_l4052_405282


namespace fraction_of_sum_l4052_405273

theorem fraction_of_sum (m n : ℝ) (a b c : ℝ) 
  (h1 : a = (b + c) / m)
  (h2 : b = (c + a) / n)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0) :
  (m * n ≠ 1 → c / (a + b) = (m * n - 1) / (m + n + 2)) ∧
  (m = -1 ∧ n = -1 → c / (a + b) = -1) :=
sorry

end fraction_of_sum_l4052_405273


namespace quadratic_root_implies_b_l4052_405257

theorem quadratic_root_implies_b (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x - 6 = 0) ∧ (2^2 + b*2 - 6 = 0) → b = 1 := by
  sorry

end quadratic_root_implies_b_l4052_405257


namespace inequality_proof_l4052_405216

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : 1/b - 1/a > 1) :
  Real.sqrt (1 + a) > 1 / Real.sqrt (1 - b) := by
  sorry

end inequality_proof_l4052_405216


namespace inequality_solution_l4052_405241

theorem inequality_solution (x : ℝ) : x / (x^2 + 3*x + 2) ≥ 0 ↔ x ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 0 := by
  sorry

end inequality_solution_l4052_405241


namespace x_squared_range_l4052_405235

theorem x_squared_range (x : ℝ) : 
  (Real.rpow (x + 9) (1/3) - Real.rpow (x - 9) (1/3) = 3) → 
  75 < x^2 ∧ x^2 < 85 := by
sorry

end x_squared_range_l4052_405235


namespace intersection_properties_l4052_405253

-- Define the line l: ax - y + 2 - 2a = 0
def line_equation (a x y : ℝ) : Prop := a * x - y + 2 - 2 * a = 0

-- Define the circle C: (x - 4)² + (y - 1)² = r²
def circle_equation (x y r : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = r^2

-- Define the intersection condition
def intersects_at_two_points (a r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
    circle_equation x₁ y₁ r ∧ circle_equation x₂ y₂ r

theorem intersection_properties (a r : ℝ) (hr : r > 0) 
  (h_intersect : intersects_at_two_points a r) :
  -- 1. The line passes through (2, 2)
  (line_equation a 2 2) ∧
  -- 2. r > √5
  (r > Real.sqrt 5) ∧
  -- 3. When r = 3, the chord length is between 4 and 6
  (r = 3 → ∃ (l : ℝ), 4 ≤ l ∧ l ≤ 6 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 3 ∧ circle_equation x₂ y₂ 3 →
      l = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  -- 4. When r = 5, the minimum dot product is -25
  (r = 5 → ∃ (min_dot_product : ℝ), min_dot_product = -25 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      line_equation a x₁ y₁ ∧ line_equation a x₂ y₂ ∧
      circle_equation x₁ y₁ 5 ∧ circle_equation x₂ y₂ 5 →
      ((x₁ - 4) * (x₂ - 4) + (y₁ - 1) * (y₂ - 1)) ≥ min_dot_product) :=
by sorry

end intersection_properties_l4052_405253


namespace arrangement_count_is_2880_l4052_405256

/-- The number of ways to arrange 4 boys and 3 girls in a row with constraints -/
def arrangementCount : ℕ :=
  let numBoys : ℕ := 4
  let numGirls : ℕ := 3
  let waysToChooseTwoGirls : ℕ := Nat.choose numGirls 2
  let waysToArrangeBoys : ℕ := Nat.factorial numBoys
  let spacesForGirlUnits : ℕ := numBoys + 1
  let waysToInsertGirlUnits : ℕ := Nat.descFactorial spacesForGirlUnits 2
  waysToChooseTwoGirls * waysToArrangeBoys * waysToInsertGirlUnits

theorem arrangement_count_is_2880 : arrangementCount = 2880 := by
  sorry

end arrangement_count_is_2880_l4052_405256


namespace sqrt_2023_irrational_not_perfect_square_2023_l4052_405274

theorem sqrt_2023_irrational : Irrational (Real.sqrt 2023) := by sorry

theorem not_perfect_square_2023 : ¬ ∃ n : ℕ, n ^ 2 = 2023 := by sorry

end sqrt_2023_irrational_not_perfect_square_2023_l4052_405274


namespace average_age_combined_l4052_405270

-- Define the groups and their properties
def num_fifth_graders : ℕ := 40
def avg_age_fifth_graders : ℚ := 12
def num_parents : ℕ := 60
def avg_age_parents : ℚ := 35
def num_teachers : ℕ := 10
def avg_age_teachers : ℚ := 45

-- Define the theorem
theorem average_age_combined :
  let total_people := num_fifth_graders + num_parents + num_teachers
  let total_age := num_fifth_graders * avg_age_fifth_graders +
                   num_parents * avg_age_parents +
                   num_teachers * avg_age_teachers
  total_age / total_people = 27.5454545 := by
  sorry


end average_age_combined_l4052_405270


namespace button_remainder_l4052_405226

theorem button_remainder (n : ℕ) 
  (h1 : n % 2 = 1)
  (h2 : n % 3 = 1)
  (h3 : n % 4 = 3)
  (h4 : n % 5 = 3) : 
  n % 12 = 7 := by sorry

end button_remainder_l4052_405226


namespace ratio_of_sum_and_difference_l4052_405201

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end ratio_of_sum_and_difference_l4052_405201


namespace prob_not_six_in_six_rolls_l4052_405280

-- Define a fair six-sided die
def fair_die : Finset ℕ := Finset.range 6

-- Define the probability of an event for a fair die
def prob (event : Finset ℕ) : ℚ :=
  event.card / fair_die.card

-- Define the event of not rolling a 6
def not_six : Finset ℕ := Finset.range 5

-- Theorem statement
theorem prob_not_six_in_six_rolls :
  (prob not_six) ^ 6 = (5 : ℚ) / 6 ^ 6 :=
sorry

end prob_not_six_in_six_rolls_l4052_405280


namespace tangent_line_y_intercept_implies_a_range_l4052_405240

theorem tangent_line_y_intercept_implies_a_range (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * x^2
  ∀ m : ℝ, m > 1 →
    let f' : ℝ → ℝ := λ x ↦ Real.exp x + 2 * a * x
    let tangent_slope : ℝ := f' m
    let tangent_y_intercept : ℝ := f m - tangent_slope * m
    tangent_y_intercept < 1 →
    a ∈ Set.Ici (-1) :=
by sorry

end tangent_line_y_intercept_implies_a_range_l4052_405240


namespace parabola_max_triangle_area_l4052_405204

/-- Given a parabola y = ax^2 + bx + c with a ≠ 0, intersecting the x-axis at A and B
    and the y-axis at C, with its vertex on y = -1, and ABC forming a right triangle,
    prove that the maximum area of triangle ABC is 1. -/
theorem parabola_max_triangle_area (a b c : ℝ) (ha : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let vertex_y := -1
  let A := {x : ℝ | f x = 0 ∧ x < 0}
  let B := {x : ℝ | f x = 0 ∧ x > 0}
  let C := (0, c)
  (∃ x, f x = vertex_y) →
  (∃ x₁ ∈ A, ∃ x₂ ∈ B, c^2 = (-x₁) * x₂) →
  (∀ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| → S ≤ 1) ∧ 
  (∃ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| ∧ S = 1) :=
by sorry

end parabola_max_triangle_area_l4052_405204


namespace probability_no_distinct_roots_l4052_405267

def is_valid_pair (b c : ℤ) : Prop :=
  b.natAbs ≤ 4 ∧ c.natAbs ≤ 4 ∧ c ≥ 0

def has_distinct_real_roots (b c : ℤ) : Prop :=
  b^2 - 4*c > 0

def total_valid_pairs : ℕ := 45

def pairs_without_distinct_roots : ℕ := 27

theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : ℚ) / total_valid_pairs = 3 / 5 :=
sorry

end probability_no_distinct_roots_l4052_405267


namespace max_value_of_expression_l4052_405218

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 ∧ ∃ x y, x^2 + y^2 ≤ 1 ∧ |x^2 + 2*x*y - y^2| = Real.sqrt 2 :=
by sorry

end max_value_of_expression_l4052_405218


namespace sin_240_degrees_l4052_405219

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l4052_405219


namespace knight_seating_probability_correct_l4052_405271

/-- The probability of three knights seated at a round table with n chairs
    having empty chairs on both sides of each knight. -/
def knight_seating_probability (n : ℕ) : ℚ :=
  if n ≥ 6 then
    (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2))
  else
    0

/-- Theorem: The probability of three knights seated at a round table with n chairs (n ≥ 6)
    having empty chairs on both sides of each knight is (n-4)(n-5) / ((n-1)(n-2)). -/
theorem knight_seating_probability_correct (n : ℕ) (h : n ≥ 6) :
  knight_seating_probability n = (n - 4 : ℚ) * (n - 5) / ((n - 1 : ℚ) * (n - 2)) :=
by sorry

end knight_seating_probability_correct_l4052_405271


namespace quadratic_sum_l4052_405214

/-- The quadratic expression 20x^2 + 240x + 3200 can be written as a(x+b)^2+c -/
def quadratic (x : ℝ) : ℝ := 20*x^2 + 240*x + 3200

/-- The completed square form of the quadratic -/
def completed_square (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum : 
  ∃ (a b c : ℝ), (∀ x, quadratic x = completed_square x a b c) ∧ (a + b + c = 2506) := by
sorry

end quadratic_sum_l4052_405214


namespace prob_b_not_lose_l4052_405299

/-- The probability that Player A wins a chess match. -/
def prob_a_wins : ℝ := 0.5

/-- The probability of a draw in a chess match. -/
def prob_draw : ℝ := 0.1

/-- The probability that Player B does not lose is equal to 0.5. -/
theorem prob_b_not_lose : 1 - prob_a_wins = 0.5 := by sorry

end prob_b_not_lose_l4052_405299


namespace route_count_is_70_l4052_405215

-- Define the grid structure
structure Grid :=
  (levels : Nat)
  (segments_between_levels : List Nat)

-- Define a route
def Route := List (Nat × Nat)

-- Function to check if a route is valid (doesn't intersect itself)
def is_valid_route (g : Grid) (r : Route) : Bool := sorry

-- Function to generate all possible routes
def all_routes (g : Grid) : List Route := sorry

-- Function to count valid routes
def count_valid_routes (g : Grid) : Nat :=
  (all_routes g).filter (is_valid_route g) |>.length

-- Define our specific grid
def our_grid : Grid :=
  { levels := 4,
    segments_between_levels := [3, 5, 3] }

-- Theorem statement
theorem route_count_is_70 :
  count_valid_routes our_grid = 70 := by sorry

end route_count_is_70_l4052_405215


namespace cyclic_sum_inequality_l4052_405244

theorem cyclic_sum_inequality (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_one : x + y + z = 1) :
  x^2 * y + y^2 * z + z^2 * x ≤ 4/27 ∧ 
  (x^2 * y + y^2 * z + z^2 * x = 4/27 ↔ 
    ((x = 2/3 ∧ y = 1/3 ∧ z = 0) ∨ 
     (x = 0 ∧ y = 2/3 ∧ z = 1/3) ∨ 
     (x = 1/3 ∧ y = 0 ∧ z = 2/3))) :=
by sorry

end cyclic_sum_inequality_l4052_405244


namespace cos_pi_third_plus_two_alpha_l4052_405205

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_pi_third_plus_two_alpha_l4052_405205


namespace perfect_square_condition_l4052_405202

/-- 
For a polynomial of the form x^2 - 18x + k to be a perfect square binomial,
k must equal 81.
-/
theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 18*x + k = (x + a)^2 + b) ↔ k = 81 := by
  sorry

end perfect_square_condition_l4052_405202


namespace cheese_distribution_l4052_405245

/-- Represents the amount of cheese bought by the first n customers -/
def S (n : ℕ) : ℚ := 20 * n / (n + 10)

/-- The total amount of cheese available -/
def total_cheese : ℚ := 20

theorem cheese_distribution (n : ℕ) (h : n ≤ 10) :
  (total_cheese - S n = 10 * (S n / n)) ∧
  (∀ k : ℕ, k ≤ n → S k - S (k-1) > 0) ∧
  (S 10 = 10) := by sorry

#check cheese_distribution

end cheese_distribution_l4052_405245


namespace sum_of_x_and_y_l4052_405213

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 10) :
  x + y = 14 ∨ x + y = -2 := by sorry

end sum_of_x_and_y_l4052_405213


namespace world_cup_souvenir_production_l4052_405234

def planned_daily_production : ℕ := 10000

def production_deviations : List ℤ := [41, -34, -52, 127, -72, 36, -29]

def production_cost : ℕ := 35

def selling_price : ℕ := 40

theorem world_cup_souvenir_production 
  (planned_daily_production : ℕ)
  (production_deviations : List ℤ)
  (production_cost selling_price : ℕ)
  (h1 : planned_daily_production = 10000)
  (h2 : production_deviations = [41, -34, -52, 127, -72, 36, -29])
  (h3 : production_cost = 35)
  (h4 : selling_price = 40) :
  (∃ (max min : ℤ), max ∈ production_deviations ∧ 
                    min ∈ production_deviations ∧ 
                    max - min = 199) ∧
  (production_deviations.sum = 17) ∧
  ((7 * planned_daily_production + production_deviations.sum) * 
   (selling_price - production_cost) = 350085) :=
by sorry

end world_cup_souvenir_production_l4052_405234


namespace cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l4052_405284

-- Define the Cartesian product
def cartesianProduct (A B : Set α) : Set (α × α) :=
  {p | p.1 ∈ A ∧ p.2 ∈ B}

-- Statement 1
theorem cartesian_product_subset {A B C : Set α} (h : A ⊆ C) :
  cartesianProduct A B ⊆ cartesianProduct C B :=
sorry

-- Statement 2
theorem cartesian_product_intersection {A B C : Set α} :
  cartesianProduct A (B ∩ C) = cartesianProduct A B ∩ cartesianProduct A C :=
sorry

-- Statement 3
theorem y_axis_representation {R : Type} [LinearOrderedField R] :
  cartesianProduct {(0 : R)} (Set.univ : Set R) = {p : R × R | p.1 = 0} :=
sorry

end cartesian_product_subset_cartesian_product_intersection_y_axis_representation_l4052_405284


namespace triangle_radii_relations_l4052_405228

/-- Given a triangle ABC with sides a, b, c, inradius r, exradii r_a, r_b, r_c, semi-perimeter p, and area S -/
theorem triangle_radii_relations (a b c r r_a r_b r_c p S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ p > 0 ∧ S > 0)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_area_inradius : S = p * r)
  (h_area_exradius_a : S = (p - a) * r_a)
  (h_area_exradius_b : S = (p - b) * r_b)
  (h_area_exradius_c : S = (p - c) * r_c) :
  (1 / r = 1 / r_a + 1 / r_b + 1 / r_c) ∧ 
  (S = Real.sqrt (r * r_a * r_b * r_c)) := by
  sorry

end triangle_radii_relations_l4052_405228


namespace legs_per_chair_correct_l4052_405285

/-- The number of legs per office chair in Kenzo's company -/
def legs_per_chair : ℕ := 5

/-- The initial number of office chairs -/
def initial_chairs : ℕ := 80

/-- The number of round tables -/
def round_tables : ℕ := 20

/-- The number of legs per round table -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that remain after damage (as a rational number) -/
def remaining_chair_ratio : ℚ := 3/5

/-- The total number of furniture legs remaining after disposal -/
def total_remaining_legs : ℕ := 300

/-- Theorem stating that the number of legs per chair is correct given the conditions -/
theorem legs_per_chair_correct : 
  (remaining_chair_ratio * initial_chairs : ℚ).num * legs_per_chair + 
  round_tables * legs_per_table = total_remaining_legs :=
by sorry

end legs_per_chair_correct_l4052_405285


namespace elise_puzzle_cost_l4052_405249

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem elise_puzzle_cost : puzzle_cost 8 13 2 1 = 18 := by
  sorry

end elise_puzzle_cost_l4052_405249


namespace best_card_to_disprove_l4052_405294

-- Define the set of visible card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the property of being a consonant
def isConsonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- John's statement as a function
def johnsStatement (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isConsonant c → isOdd n
  | (CardSide.Number n, CardSide.Letter c) => isConsonant c → isOdd n
  | _ => True

-- The set of visible card sides
def visibleSides : List CardSide := [CardSide.Letter 'A', CardSide.Letter 'B', CardSide.Number 7, CardSide.Number 8, CardSide.Number 9]

-- The theorem to prove
theorem best_card_to_disprove (cards : List Card) :
  (∀ card ∈ cards, (CardSide.Number 8 ∈ card.1 :: card.2 :: []) →
    ¬(∀ c ∈ cards, johnsStatement c)) →
  (∀ side ∈ visibleSides, side ≠ CardSide.Number 8 →
    ∃ c ∈ cards, (side ∈ c.1 :: c.2 :: []) ∧
      (∀ card ∈ cards, (side ∈ card.1 :: card.2 :: []) →
        (∃ c' ∈ cards, ¬johnsStatement c'))) :=
by sorry

end best_card_to_disprove_l4052_405294


namespace lcm_144_132_l4052_405258

theorem lcm_144_132 : lcm 144 132 = 1584 := by
  sorry

end lcm_144_132_l4052_405258


namespace sams_cycling_speed_l4052_405283

/-- Given the cycling speeds of three friends, prove Sam's speed -/
theorem sams_cycling_speed 
  (lucas_speed : ℚ) 
  (maya_speed_ratio : ℚ) 
  (lucas_sam_ratio : ℚ)
  (h1 : lucas_speed = 5)
  (h2 : maya_speed_ratio = 4 / 5)
  (h3 : lucas_sam_ratio = 9 / 8) :
  lucas_speed * (8 / 9) = 40 / 9 := by
  sorry

#check sams_cycling_speed

end sams_cycling_speed_l4052_405283


namespace function_passes_through_point_l4052_405252

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end function_passes_through_point_l4052_405252


namespace simplify_sqrt_difference_l4052_405238

theorem simplify_sqrt_difference : 
  (Real.sqrt 704 / Real.sqrt 64) - (Real.sqrt 300 / Real.sqrt 75) = Real.sqrt 11 - 2 := by
  sorry

end simplify_sqrt_difference_l4052_405238


namespace existence_of_57_multiple_non_existence_of_58_multiple_l4052_405276

/-- Removes the first digit of a positive integer -/
def removeFirstDigit (n : ℕ) : ℕ := sorry

/-- Checks if a number satisfies the condition A = k * B, where B is A with its first digit removed -/
def satisfiesCondition (A : ℕ) (k : ℕ) : Prop :=
  A = k * removeFirstDigit A

theorem existence_of_57_multiple :
  ∃ A : ℕ, A > 0 ∧ satisfiesCondition A 57 := by sorry

theorem non_existence_of_58_multiple :
  ¬∃ A : ℕ, A > 0 ∧ satisfiesCondition A 58 := by sorry

end existence_of_57_multiple_non_existence_of_58_multiple_l4052_405276


namespace scientific_notation_4212000_l4052_405250

theorem scientific_notation_4212000 :
  4212000 = 4.212 * (10 ^ 6) := by
  sorry

end scientific_notation_4212000_l4052_405250


namespace quadratic_root_problem_l4052_405233

theorem quadratic_root_problem (m : ℝ) :
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) →
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end quadratic_root_problem_l4052_405233


namespace river_road_cars_l4052_405239

theorem river_road_cars (B C : ℕ) 
  (h1 : B * 13 = C)  -- ratio of buses to cars is 1:13
  (h2 : B = C - 60)  -- there are 60 fewer buses than cars
  : C = 65 := by sorry

end river_road_cars_l4052_405239


namespace club_officer_selection_l4052_405278

/-- The number of ways to choose distinct officers from a group -/
def chooseOfficers (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1).factorial / (n - k).factorial

theorem club_officer_selection :
  chooseOfficers 12 5 = 95040 := by
  sorry

end club_officer_selection_l4052_405278


namespace two_digit_number_property_l4052_405287

theorem two_digit_number_property (N : ℕ) : 
  (N ≥ 10 ∧ N ≤ 99) →
  (4 * (N / 10) + 2 * (N % 10) = N / 2) →
  (N = 32 ∨ N = 64 ∨ N = 96) :=
by sorry

end two_digit_number_property_l4052_405287


namespace triangle_midpoint_x_coordinate_sum_l4052_405281

theorem triangle_midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (b + c) / 2 + (c + a) / 2
  midpoint_sum = vertex_sum := by
sorry

end triangle_midpoint_x_coordinate_sum_l4052_405281


namespace investment_return_percentage_l4052_405242

/-- Calculates the return percentage for a two-venture investment --/
def calculate_return_percentage (total_investment : ℚ) (investment1 : ℚ) (investment2 : ℚ) 
  (profit_percentage1 : ℚ) (loss_percentage2 : ℚ) : ℚ :=
  let profit1 := investment1 * profit_percentage1
  let loss2 := investment2 * loss_percentage2
  let net_income := profit1 - loss2
  (net_income / total_investment) * 100

/-- Theorem stating that the return percentage is 6.5% for the given investment scenario --/
theorem investment_return_percentage : 
  calculate_return_percentage 25000 16250 16250 (15/100) (5/100) = 13/2 :=
by sorry

end investment_return_percentage_l4052_405242


namespace fractional_equation_root_l4052_405291

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end fractional_equation_root_l4052_405291


namespace intersection_with_complement_l4052_405207

def U : Set ℕ := {1, 3, 5, 7}
def A : Set ℕ := {3, 5}
def B : Set ℕ := {1, 3, 7}

theorem intersection_with_complement : A ∩ (U \ B) = {5} := by sorry

end intersection_with_complement_l4052_405207


namespace no_primes_in_factorial_range_l4052_405224

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n + 1 → ¬ Nat.Prime k :=
by sorry

end no_primes_in_factorial_range_l4052_405224


namespace refrigerator_price_l4052_405232

theorem refrigerator_price (P : ℝ) 
  (h1 : 1.1 * P = 21725)  -- Selling price for 10% profit
  (h2 : 0.8 * P + 125 + 250 = 16175)  -- Price paid by buyer
  : True :=
by sorry

end refrigerator_price_l4052_405232


namespace sqrt_identity_l4052_405236

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sqrt_identity_l4052_405236


namespace textbook_weight_difference_l4052_405251

/-- Given the weights of four textbooks, prove that the difference between
    the sum of the middle two weights and the difference between the
    largest and smallest weights is 2.5 pounds. -/
theorem textbook_weight_difference
  (chemistry_weight geometry_weight calculus_weight biology_weight : ℝ)
  (h1 : chemistry_weight = 7.125)
  (h2 : geometry_weight = 0.625)
  (h3 : calculus_weight = 5.25)
  (h4 : biology_weight = 3.75)
  : (calculus_weight + biology_weight) - (chemistry_weight - geometry_weight) = 2.5 := by
  sorry

end textbook_weight_difference_l4052_405251


namespace juice_reduction_fraction_l4052_405208

/-- Proves that the fraction of the original volume that the juice was reduced to is 1/12 --/
theorem juice_reduction_fraction (original_volume : ℚ) (quart_to_cup : ℚ) (sugar_added : ℚ) (final_volume : ℚ) :
  original_volume = 6 →
  quart_to_cup = 4 →
  sugar_added = 1 →
  final_volume = 3 →
  (final_volume - sugar_added) / (original_volume * quart_to_cup) = 1 / 12 := by
sorry

end juice_reduction_fraction_l4052_405208


namespace sufficient_not_necessary_l4052_405286

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the given conditions
variable (h1 : B → A)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem to prove
theorem sufficient_not_necessary : (C → A) ∧ ¬(A → C) := by sorry

end sufficient_not_necessary_l4052_405286


namespace train_length_l4052_405230

theorem train_length (tree_time platform_time platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 230 →
  platform_length = 1100 →
  ∃ (train_length : ℝ),
    train_length / tree_time = (train_length + platform_length) / platform_time ∧
    train_length = 1200 := by
  sorry

end train_length_l4052_405230


namespace quadratic_and_related_function_properties_l4052_405237

/-- Given a quadratic function f and its derivative, prove properties about its coefficients and a related function g --/
theorem quadratic_and_related_function_properties
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ x, f x = a * x^2 + b * x + 3)
  (h₂ : a ≠ 0)
  (h₃ : ∀ x, deriv f x = 2 * x - 8)
  (g : ℝ → ℝ)
  (h₄ : ∀ x, g x = Real.exp x * Real.sin x + f x) :
  a = 1 ∧ b = -8 ∧
  (∃ m c : ℝ, m = 7 ∧ c = -3 ∧ ∀ x y, y = deriv g 0 * (x - 0) + g 0 ↔ m * x + y + c = 0) :=
sorry

end quadratic_and_related_function_properties_l4052_405237


namespace simplify_sqrt_sum_l4052_405247

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l4052_405247


namespace candy_box_price_increase_l4052_405265

theorem candy_box_price_increase (current_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  current_price = 10 ∧ 
  increase_percentage = 25 ∧ 
  current_price = original_price * (1 + increase_percentage / 100) →
  original_price = 8 := by
sorry

end candy_box_price_increase_l4052_405265


namespace max_value_complex_expression_l4052_405222

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ 16 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 16 * Real.sqrt 2 :=
by sorry

end max_value_complex_expression_l4052_405222


namespace lock_settings_count_l4052_405243

/-- The number of digits on each dial of the lock -/
def num_digits : ℕ := 8

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of different settings possible for the lock -/
def num_settings : ℕ := 1680

/-- Theorem stating that the number of different settings for the lock
    with the given conditions is equal to 1680 -/
theorem lock_settings_count :
  (num_digits.factorial) / ((num_digits - num_dials).factorial) = num_settings :=
sorry

end lock_settings_count_l4052_405243


namespace queen_mary_legs_l4052_405275

/-- The total number of legs on the Queen Mary II -/
def total_legs : ℕ := 41

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The number of cats on the ship -/
def num_cats : ℕ := 7

/-- The number of legs a cat has -/
def cat_legs : ℕ := 4

/-- The number of legs a normal human has -/
def human_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- Theorem stating the total number of legs on the ship -/
theorem queen_mary_legs : 
  total_legs = 
    (num_cats * cat_legs) + 
    ((total_heads - num_cats - 1) * human_legs) + 
    captain_legs :=
by sorry

end queen_mary_legs_l4052_405275


namespace least_subtraction_l4052_405248

theorem least_subtraction (n : ℕ) : n = 10 ↔ 
  (∀ m : ℕ, m < n → ¬(
    (2590 - n) % 9 = 6 ∧ 
    (2590 - n) % 11 = 6 ∧ 
    (2590 - n) % 13 = 6
  )) ∧
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6 :=
by sorry

end least_subtraction_l4052_405248


namespace min_fruits_problem_l4052_405223

theorem min_fruits_problem : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 4 = 3 → m % 5 = 4 → m % 6 = 5 → m ≥ n) ∧
  n = 59 := by
sorry

end min_fruits_problem_l4052_405223


namespace hyperbola_eccentricity_range_l4052_405227

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes and intersection points
def asymptote_intersections (h : Hyperbola) (x : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x = h.a^2 / h.c ∧ (y = h.b * x / h.a ∨ y = -h.b * x / h.a)}

-- Define the angle AFB
def angle_AFB (h : Hyperbola) (A B : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def eccentricity (h : Hyperbola) : ℝ := sorry

-- State the theorem
theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (A B : ℝ × ℝ) (hA : A ∈ asymptote_intersections h (h.a^2 / h.c)) 
  (hB : B ∈ asymptote_intersections h (h.a^2 / h.c))
  (hAngle : π/3 < angle_AFB h A B ∧ angle_AFB h A B < π/2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end hyperbola_eccentricity_range_l4052_405227


namespace third_motorcyclist_speed_l4052_405298

theorem third_motorcyclist_speed 
  (v1 : ℝ) (v2 : ℝ) (v3 : ℝ) (t_delay : ℝ) (t_diff : ℝ) :
  v1 = 80 →
  v2 = 60 →
  t_delay = 0.5 →
  t_diff = 1.25 →
  v3 * (v3 * t_diff / (v3 - v1) - t_delay) = v1 * (v3 * t_diff / (v3 - v1)) →
  v3 * (v3 * t_diff / (v3 - v1) - v3 * t_diff / (v3 - v2) - t_delay) = 
    v2 * (v3 * t_diff / (v3 - v1) - t_delay) →
  v3 = 100 := by
sorry


end third_motorcyclist_speed_l4052_405298


namespace triangle_ABC_properties_l4052_405211

theorem triangle_ABC_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.cos A = Real.sin A * (a * Real.cos C + c * Real.cos A) →
  a = 2 * Real.sqrt 3 →
  (5 * Real.sqrt 3) / 4 = (1 / 2) * a * b * Real.sin C →
  (A = π / 3) ∧ (a + b + c = 5 * Real.sqrt 3) := by
sorry

end triangle_ABC_properties_l4052_405211


namespace fixed_point_quadratic_l4052_405277

theorem fixed_point_quadratic (k : ℝ) : 
  200 = 8 * (5 : ℝ)^2 + 3 * k * 5 - 5 * k := by sorry

end fixed_point_quadratic_l4052_405277


namespace f_even_implies_increasing_l4052_405295

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

theorem f_even_implies_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  ∀ a b, 0 < a → a < b → f m a < f m b :=
by sorry

end f_even_implies_increasing_l4052_405295


namespace giyoons_chocolates_l4052_405297

theorem giyoons_chocolates (initial_friends : ℕ) (absent_friends : ℕ) (extra_per_person : ℕ) (leftover : ℕ) :
  initial_friends = 8 →
  absent_friends = 2 →
  extra_per_person = 1 →
  leftover = 4 →
  ∃ (total_chocolates : ℕ),
    total_chocolates = (initial_friends - absent_friends) * ((total_chocolates / initial_friends) + extra_per_person) + leftover ∧
    total_chocolates = 40 :=
by sorry

end giyoons_chocolates_l4052_405297


namespace circles_shortest_distance_l4052_405272

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 6*y = 8

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 2*y = 1

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := -0.68

/-- Theorem stating that the shortest distance between the two circles is -0.68 -/
theorem circles_shortest_distance :
  ∃ (d : ℝ), d = shortest_distance ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    circle1 x₁ y₁ → circle2 x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
  sorry

end circles_shortest_distance_l4052_405272


namespace sqrt_product_difference_of_squares_l4052_405254

-- Problem 1
theorem sqrt_product : Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10 := by sorry

-- Problem 2
theorem difference_of_squares : (3 + Real.sqrt 6) * (3 - Real.sqrt 6) = 3 := by sorry

end sqrt_product_difference_of_squares_l4052_405254


namespace product_of_primes_with_conditions_l4052_405206

theorem product_of_primes_with_conditions :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    (r - q = 2 * p) ∧
    (r * q + p^2 = 676) ∧
    (p * q * r = 2001) := by
  sorry

end product_of_primes_with_conditions_l4052_405206


namespace problem_solutions_l4052_405292

theorem problem_solutions : 
  (1) * (1/3 - 1/4 - 1/2) / (-1/24) = 10 ∧ 
  -(3^2) - (-2/3) * 6 + (-2)^3 = -13 := by sorry

end problem_solutions_l4052_405292


namespace smallest_bob_number_l4052_405246

def alice_number : Nat := 30

def has_all_prime_factors_of (n m : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p ∣ n → p ∣ m

def has_additional_prime_factor (n m : Nat) : Prop :=
  ∃ p : Nat, Nat.Prime p ∧ p ∣ m ∧ ¬(p ∣ n)

theorem smallest_bob_number :
  ∃ bob_number : Nat,
    has_all_prime_factors_of alice_number bob_number ∧
    has_additional_prime_factor alice_number bob_number ∧
    (∀ m : Nat, m < bob_number →
      ¬(has_all_prime_factors_of alice_number m ∧
        has_additional_prime_factor alice_number m)) ∧
    bob_number = 210 := by
  sorry

end smallest_bob_number_l4052_405246
