import Mathlib

namespace line_plane_perpendicularity_l824_82446

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (distinct : Plane → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : distinct α β)
  (h_distinct_lines : distinct_lines m n) :
  (parallel m n ∧ perpendicular m α) → perpendicular n α :=
by sorry

end line_plane_perpendicularity_l824_82446


namespace guitar_price_l824_82496

theorem guitar_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) :
  upfront_percentage = 0.20 →
  upfront_payment = 240 →
  upfront_percentage * total_price = upfront_payment →
  total_price = 1200 := by
  sorry

end guitar_price_l824_82496


namespace vector_operation_proof_l824_82449

def vector_operation : Prop :=
  let v1 : Fin 2 → ℝ := ![5, -6]
  let v2 : Fin 2 → ℝ := ![-2, 13]
  let v3 : Fin 2 → ℝ := ![1, -2]
  v1 + v2 - 3 • v3 = ![0, 13]

theorem vector_operation_proof : vector_operation := by
  sorry

end vector_operation_proof_l824_82449


namespace triangle_side_length_l824_82434

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  Real.cos (B - C) = 47 / 50 →
  a = Real.sqrt 54.4 :=
by
  sorry

end triangle_side_length_l824_82434


namespace m_range_l824_82468

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end m_range_l824_82468


namespace glen_village_impossibility_l824_82441

theorem glen_village_impossibility : ¬ ∃ (h c : ℕ), 21 * h + 6 * c = 96 := by
  sorry

#check glen_village_impossibility

end glen_village_impossibility_l824_82441


namespace irrational_sum_of_roots_l824_82447

theorem irrational_sum_of_roots (n : ℤ) : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (n - 1) + Real.sqrt (n + 1) = p / q :=
sorry

end irrational_sum_of_roots_l824_82447


namespace find_number_l824_82411

theorem find_number (x : ℝ) : 6 + (1/2) * (1/3) * (1/5) * x = (1/15) * x → x = 180 := by
  sorry

end find_number_l824_82411


namespace quadratic_inequality_condition_l824_82420

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 ≤ a ∧ a ≤ 4) ∧
  ¬(0 ≤ a ∧ a ≤ 4 → ∀ x : ℝ, x^2 + a*x + a > 0) :=
by sorry

end quadratic_inequality_condition_l824_82420


namespace equation_represents_two_hyperbolas_l824_82423

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^4 - 6*x^4 = 3*y^2 - 2

-- Define what a hyperbola equation looks like
def is_hyperbola_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  y^2 - a*x^2 = c ∧ b ≠ 0

-- Theorem statement
theorem equation_represents_two_hyperbolas :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, equation x y ↔ 
      (is_hyperbola_equation a₁ b₁ c₁ x y ∨ is_hyperbola_equation a₂ b₂ c₂ x y)) ∧
    b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ (a₁ ≠ a₂ ∨ c₁ ≠ c₂) :=
sorry

end equation_represents_two_hyperbolas_l824_82423


namespace dinner_cost_is_36_l824_82415

/-- Represents the farming scenario with kids planting corn --/
structure FarmingScenario where
  kids : ℕ
  ears_per_row : ℕ
  seeds_per_bag : ℕ
  seeds_per_ear : ℕ
  pay_per_row : ℚ
  bags_per_kid : ℕ

/-- Calculates the cost of dinner per kid based on the farming scenario --/
def dinner_cost_per_kid (scenario : FarmingScenario) : ℚ :=
  let ears_per_bag := scenario.seeds_per_bag / scenario.seeds_per_ear
  let total_ears := scenario.bags_per_kid * ears_per_bag
  let rows_planted := total_ears / scenario.ears_per_row
  let earnings := rows_planted * scenario.pay_per_row
  earnings / 2

/-- Theorem stating that the dinner cost per kid is $36 given the specific scenario --/
theorem dinner_cost_is_36 (scenario : FarmingScenario) 
  (h1 : scenario.kids = 4)
  (h2 : scenario.ears_per_row = 70)
  (h3 : scenario.seeds_per_bag = 48)
  (h4 : scenario.seeds_per_ear = 2)
  (h5 : scenario.pay_per_row = 3/2)
  (h6 : scenario.bags_per_kid = 140) :
  dinner_cost_per_kid scenario = 36 := by
  sorry

end dinner_cost_is_36_l824_82415


namespace fifteen_ways_to_divide_books_l824_82490

/-- The number of ways to divide 6 different books into 3 groups -/
def divide_books : ℕ :=
  Nat.choose 6 4 * Nat.choose 2 1 * Nat.choose 1 1 / Nat.factorial 2

/-- Theorem stating that there are 15 ways to divide the books -/
theorem fifteen_ways_to_divide_books : divide_books = 15 := by
  sorry

end fifteen_ways_to_divide_books_l824_82490


namespace sign_determination_l824_82482

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end sign_determination_l824_82482


namespace simplify_polynomial_l824_82410

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end simplify_polynomial_l824_82410


namespace t_shape_perimeter_l824_82445

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shape formed by two rectangles -/
structure TShape where
  top : Rectangle
  bottom : Rectangle

/-- Calculates the perimeter of a T-shape -/
def TShape.perimeter (t : TShape) : ℝ :=
  t.top.perimeter + t.bottom.perimeter - 2 * t.top.width

theorem t_shape_perimeter : 
  let t : TShape := {
    top := { width := 1, height := 4 },
    bottom := { width := 5, height := 2 }
  }
  TShape.perimeter t = 20 := by sorry

end t_shape_perimeter_l824_82445


namespace product_of_three_numbers_l824_82400

/-- Given three numbers a, b, and c satisfying certain conditions, 
    prove that their product is equal to 369912000/4913 -/
theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 180 ∧ 
  8 * a = m ∧ 
  b - 10 = m ∧ 
  c + 10 = m → 
  a * b * c = 369912000 / 4913 := by
  sorry

end product_of_three_numbers_l824_82400


namespace cycle_price_problem_l824_82443

/-- Given a cycle sold at a 25% loss for Rs. 2100, prove that the original price was Rs. 2800. -/
theorem cycle_price_problem (selling_price : ℝ) (loss_percentage : ℝ) 
    (h1 : selling_price = 2100)
    (h2 : loss_percentage = 25) : 
  ∃ original_price : ℝ, 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 2800 := by
  sorry

end cycle_price_problem_l824_82443


namespace triangle_properties_l824_82462

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom side_a : a = 4
axiom side_c : c = Real.sqrt 13
axiom sin_relation : Real.sin A = 4 * Real.sin B

-- State the theorem
theorem triangle_properties : b = 1 ∧ C = Real.pi / 3 := by sorry

end triangle_properties_l824_82462


namespace power_of_three_expression_equals_zero_l824_82403

theorem power_of_three_expression_equals_zero :
  3^2003 - 5 * 3^2002 + 6 * 3^2001 = 0 := by
  sorry

end power_of_three_expression_equals_zero_l824_82403


namespace functional_equation_solution_l824_82413

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = x + f y) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by sorry

end functional_equation_solution_l824_82413


namespace mushroom_collection_proof_l824_82407

theorem mushroom_collection_proof :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 6 ∧
    x₁ + x₃ = 7 ∧
    x₁ + x₄ = 9 ∧
    x₂ + x₃ = 9 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 12 ∧
    x₁ = 2 ∧
    x₂ = 4 ∧
    x₃ = 5 ∧
    x₄ = 7 := by
  sorry

end mushroom_collection_proof_l824_82407


namespace inequality_solution_set_l824_82440

theorem inequality_solution_set (a b : ℝ) : 
  a > 2 → 
  (∀ x, ax + 3 < 2*x + b ↔ x < 0) → 
  b = 3 := by
sorry

end inequality_solution_set_l824_82440


namespace chessboard_number_property_l824_82412

theorem chessboard_number_property (n : ℕ) (X : Matrix (Fin n) (Fin n) ℝ) 
  (h : ∀ (i j k : Fin n), X i j + X j k + X k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ (i j : Fin n), X i j = t i - t j := by
sorry

end chessboard_number_property_l824_82412


namespace digit_ratio_l824_82494

theorem digit_ratio (x : ℕ) (a b c : ℕ) : 
  x ≥ 100 ∧ x < 1000 →  -- x is a 3-digit integer
  a > 0 →               -- a > 0
  x = 100 * a + 10 * b + c →  -- x is composed of digits a, b, c
  (999 : ℕ) - x = 241 →  -- difference between largest possible value and x is 241
  (b : ℚ) / c = 5 / 8 :=  -- ratio of b to c is 5:8
by sorry

end digit_ratio_l824_82494


namespace hyperbola_focal_length_l824_82455

/-- The focal length of a hyperbola with equation x² - y² = 1 is 2√2 -/
theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ (f : ℝ), (f = 2 * Real.sqrt 2 ∧ 
    ∀ (c : ℝ), (c^2 = 2 → f = 2*c) ∧
    ∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2)) :=
by sorry

end hyperbola_focal_length_l824_82455


namespace monkey_escape_time_l824_82426

/-- Proves that a monkey running at 15 feet/second for t seconds, then swinging at 10 feet/second for 10 seconds, covering 175 feet total, ran for 5 seconds. -/
theorem monkey_escape_time (run_speed : ℝ) (swing_speed : ℝ) (swing_time : ℝ) (total_distance : ℝ) :
  run_speed = 15 →
  swing_speed = 10 →
  swing_time = 10 →
  total_distance = 175 →
  ∃ t : ℝ, t * run_speed + swing_time * swing_speed = total_distance ∧ t = 5 :=
by
  sorry

#check monkey_escape_time

end monkey_escape_time_l824_82426


namespace total_cost_is_122_4_l824_82442

/-- Calculates the total cost of Zoe's app usage over a year -/
def total_cost (initial_app_cost monthly_fee annual_discount in_game_cost upgrade_cost membership_discount : ℝ) : ℝ :=
  let first_two_months := 2 * monthly_fee
  let annual_plan_cost := (12 * monthly_fee) * (1 - annual_discount)
  let discounted_in_game := in_game_cost * (1 - membership_discount)
  let discounted_upgrade := upgrade_cost * (1 - membership_discount)
  initial_app_cost + first_two_months + annual_plan_cost + discounted_in_game + discounted_upgrade

/-- Theorem stating that the total cost is $122.4 given the specified conditions -/
theorem total_cost_is_122_4 :
  total_cost 5 8 0.15 10 12 0.1 = 122.4 := by
  sorry

#eval total_cost 5 8 0.15 10 12 0.1

end total_cost_is_122_4_l824_82442


namespace regular_triangle_counts_l824_82433

/-- Regular triangle with sides divided into n segments -/
structure RegularTriangle (n : ℕ) where
  -- Add any necessary fields

/-- Count of regular triangles in a RegularTriangle -/
def countRegularTriangles (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n + 1)) / 8
  else
    ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8

/-- Count of rhombuses in a RegularTriangle -/
def countRhombuses (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n - 1)) / 8
  else
    ((n - 1) * (n + 1) * (2 * n + 3)) / 8

/-- Theorem stating the counts are correct -/
theorem regular_triangle_counts (n : ℕ) (t : RegularTriangle n) :
  (countRegularTriangles t = if n % 2 = 0 then (n * (n + 2) * (2 * n + 1)) / 8
                             else ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8) ∧
  (countRhombuses t = if n % 2 = 0 then (n * (n + 2) * (2 * n - 1)) / 8
                      else ((n - 1) * (n + 1) * (2 * n + 3)) / 8) :=
by sorry

end regular_triangle_counts_l824_82433


namespace fraction_zero_implies_x_negative_three_l824_82453

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x ≠ 3) → ((x^2 - 9) / (x - 3) = 0) → x = -3 := by
  sorry

end fraction_zero_implies_x_negative_three_l824_82453


namespace number_puzzle_l824_82458

theorem number_puzzle : ∃ x : ℝ, 35 + 3 * x = 50 ∧ x - 4 = 1 := by
  sorry

end number_puzzle_l824_82458


namespace smallest_odd_five_primes_l824_82404

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_five_prime_factors (n : ℕ) : Prop := sorry

def smallest_odd_with_five_prime_factors : ℕ := 15015

theorem smallest_odd_five_primes :
  has_exactly_five_prime_factors smallest_odd_with_five_prime_factors ∧
  ∀ m : ℕ, m < smallest_odd_with_five_prime_factors →
    ¬(has_exactly_five_prime_factors m ∧ Odd m) :=
sorry

end smallest_odd_five_primes_l824_82404


namespace sequence_formula_l824_82429

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := 2 * n^2 + n

-- Theorem statement
theorem sequence_formula (n : ℕ) : a n = 4 * n - 1 := by
  sorry

end sequence_formula_l824_82429


namespace copper_alloy_percentages_l824_82402

theorem copper_alloy_percentages
  (x y : ℝ)  -- Percentages of copper in first and second alloys
  (m₁ m₂ : ℝ)  -- Masses of first and second alloys
  (h₁ : y = x + 40)  -- First alloy's copper percentage is 40% less than the second
  (h₂ : x * m₁ / 100 = 6)  -- First alloy contains 6 kg of copper
  (h₃ : y * m₂ / 100 = 12)  -- Second alloy contains 12 kg of copper
  (h₄ : 36 * (m₁ + m₂) / 100 = 18)  -- Mixture contains 36% copper
  : x = 20 ∧ y = 60 := by
  sorry

end copper_alloy_percentages_l824_82402


namespace sock_pairs_count_l824_82464

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  white * brown + brown * blue + white * blue

/-- Theorem: Given 5 white socks, 4 brown socks, and 3 blue socks,
    there are 47 ways to choose a pair of socks of different colors -/
theorem sock_pairs_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end sock_pairs_count_l824_82464


namespace min_circular_arrangement_with_shared_digit_l824_82476

/-- A function that checks if two natural numbers share a common digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop := sorry

/-- A function that represents a circular arrangement of numbers from 1 to n -/
def circular_arrangement (n : ℕ) : (ℕ → ℕ) := sorry

/-- The main theorem stating that 29 is the smallest number satisfying the conditions -/
theorem min_circular_arrangement_with_shared_digit :
  ∀ n : ℕ, n ≥ 2 →
  (∃ arr : ℕ → ℕ, 
    (∀ i : ℕ, arr i ≤ n) ∧ 
    (∀ i : ℕ, share_digit (arr i) (arr (i + 1))) ∧
    (∀ k : ℕ, k ≤ n → ∃ i : ℕ, arr i = k)) →
  n ≥ 29 :=
sorry

end min_circular_arrangement_with_shared_digit_l824_82476


namespace regular_polygon_sides_l824_82469

/-- A regular polygon with side length 10 and perimeter 60 has 6 sides -/
theorem regular_polygon_sides (s : ℕ) (side_length perimeter : ℝ) : 
  side_length = 10 → perimeter = 60 → s * side_length = perimeter → s = 6 := by
  sorry

end regular_polygon_sides_l824_82469


namespace area_of_triangle_GAB_l824_82470

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define points P, Q, and G
def point_P : ℝ × ℝ := (2, 0)
def point_Q : ℝ × ℝ := (0, -2)
def point_G : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem area_of_triangle_GAB :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧
    curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧
    line_l B.1 B.2 ∧
    line_l point_Q.1 point_Q.2 →
    let area := (1/2) * ‖A - B‖ * (2 * Real.sqrt 2)
    area = 16 * Real.sqrt 2 := by
  sorry


end area_of_triangle_GAB_l824_82470


namespace hat_number_sum_l824_82481

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem hat_number_sum : 
  ∀ (A B : ℕ),
  (A ≥ 2 ∧ A ≤ 49) →  -- Alice's number is between 2 and 49
  (B > 10 ∧ is_prime B) →  -- Bob's number is prime and greater than 10
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → ¬(k > B)) →  -- Alice can't tell who has the larger number
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → (k > B ∨ B > k)) →  -- Bob can tell who has the larger number
  (∃ (k : ℕ), 50 * B + A = k * k) →  -- The result is a perfect square
  A + B = 37 :=
by sorry

end hat_number_sum_l824_82481


namespace percentage_of_sikh_boys_l824_82451

/-- Proves that the percentage of Sikh boys is 10% given the specified conditions --/
theorem percentage_of_sikh_boys
  (total_boys : ℕ)
  (muslim_percentage : ℚ)
  (hindu_percentage : ℚ)
  (other_boys : ℕ)
  (h1 : total_boys = 850)
  (h2 : muslim_percentage = 44/100)
  (h3 : hindu_percentage = 32/100)
  (h4 : other_boys = 119) :
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1/10 := by
  sorry

end percentage_of_sikh_boys_l824_82451


namespace purchasing_ways_l824_82473

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 7

/-- The number of different milk flavors --/
def milk_flavors : ℕ := 4

/-- The total number of products they purchase --/
def total_products : ℕ := 4

/-- Charlie's purchasing strategy: no repeats, can buy both oreos and milk --/
def charlie_strategy (k : ℕ) : ℕ := Nat.choose (oreo_flavors + milk_flavors) k

/-- Delta's purchasing strategy: only oreos, can have repeats --/
def delta_strategy (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else if k = 3 then Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else Nat.choose oreo_flavors 4 + oreo_flavors * (oreo_flavors - 1) + 
       (oreo_flavors * (oreo_flavors - 1)) / 2 + oreo_flavors

/-- The total number of ways to purchase 4 products --/
def total_ways : ℕ := 
  (charlie_strategy 4 * delta_strategy 0) +
  (charlie_strategy 3 * delta_strategy 1) +
  (charlie_strategy 2 * delta_strategy 2) +
  (charlie_strategy 1 * delta_strategy 3) +
  (charlie_strategy 0 * delta_strategy 4)

theorem purchasing_ways : total_ways = 4054 := by
  sorry

end purchasing_ways_l824_82473


namespace slope_of_line_l824_82437

/-- The slope of a line represented by the equation 3y = 4x + 9 is 4/3 -/
theorem slope_of_line (x y : ℝ) : 3 * y = 4 * x + 9 → (y - 3) / x = 4 / 3 := by
  sorry

end slope_of_line_l824_82437


namespace smallest_number_with_remainder_l824_82438

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  n > 300 →
  n % 25 = 24 →
  (∀ m : ℕ, m > 300 ∧ m % 25 = 24 → n ≤ m) →
  n = 324 := by
  sorry

end smallest_number_with_remainder_l824_82438


namespace percentage_d_grades_l824_82486

def scores : List ℕ := [89, 65, 55, 96, 73, 93, 82, 70, 77, 65, 81, 79, 67, 85, 88, 61, 84, 71, 73, 90]

def is_d_grade (score : ℕ) : Bool :=
  65 ≤ score ∧ score ≤ 75

def count_d_grades (scores : List ℕ) : ℕ :=
  scores.filter is_d_grade |>.length

theorem percentage_d_grades :
  (count_d_grades scores : ℚ) / scores.length * 100 = 35 := by
  sorry

end percentage_d_grades_l824_82486


namespace unique_pizza_combinations_l824_82483

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 5

/-- Binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The number of unique five-topping pizzas with 8 available toppings is 56 -/
theorem unique_pizza_combinations : binomial n k = 56 := by
  sorry

end unique_pizza_combinations_l824_82483


namespace rhombus_not_necessarily_planar_l824_82405

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A shape in 3D space -/
class Shape where
  vertices : List Point3D

/-- A triangle is always planar -/
def Triangle (a b c : Point3D) : Shape :=
  { vertices := [a, b, c] }

/-- A trapezoid is always planar -/
def Trapezoid (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A parallelogram is always planar -/
def Parallelogram (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A rhombus (quadrilateral with equal sides) -/
def Rhombus (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- Predicate to check if a shape is planar -/
def isPlanar (s : Shape) : Prop :=
  sorry

/-- Theorem stating that a rhombus is not necessarily planar -/
theorem rhombus_not_necessarily_planar :
  ∃ (a b c d : Point3D), ¬(isPlanar (Rhombus a b c d)) ∧
    (∀ (x y z : Point3D), isPlanar (Triangle x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Trapezoid w x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Parallelogram w x y z)) :=
  sorry

end rhombus_not_necessarily_planar_l824_82405


namespace geometric_sequence_property_l824_82477

/-- A geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 2 * a 12) = -Real.sqrt 3 := by
  sorry

end geometric_sequence_property_l824_82477


namespace winning_candidate_vote_percentage_l824_82466

/-- Given an association with total members, votes cast, and the winning candidate's votes as a percentage of total membership, calculate the percentage of votes cast that the winning candidate received. -/
theorem winning_candidate_vote_percentage
  (total_members : ℕ)
  (votes_cast : ℕ)
  (winning_votes_percentage_of_total : ℚ)
  (h1 : total_members = 1600)
  (h2 : votes_cast = 525)
  (h3 : winning_votes_percentage_of_total = 19.6875 / 100) :
  (winning_votes_percentage_of_total * total_members) / votes_cast = 60 / 100 :=
by sorry

end winning_candidate_vote_percentage_l824_82466


namespace intersection_A_B_l824_82419

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end intersection_A_B_l824_82419


namespace min_quotient_three_digit_number_l824_82475

theorem min_quotient_three_digit_number (a : ℕ) :
  a ≠ 0 ∧ a ≠ 7 ∧ a ≠ 8 →
  (∀ x : ℕ, x ≠ 0 ∧ x ≠ 7 ∧ x ≠ 8 →
    (100 * a + 78 : ℚ) / (a + 15) ≤ (100 * x + 78 : ℚ) / (x + 15)) →
  (100 * a + 78 : ℚ) / (a + 15) = 11125 / 1000 :=
sorry

end min_quotient_three_digit_number_l824_82475


namespace product_sum_theorem_l824_82428

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a + b + c = 16) : 
  a * b + b * c + a * c = 50 := by
sorry

end product_sum_theorem_l824_82428


namespace jonathan_exercise_distance_l824_82491

/-- Represents Jonathan's exercise routine for a week -/
structure ExerciseRoutine where
  monday_speed : ℝ
  wednesday_speed : ℝ
  friday_speed : ℝ
  total_time : ℝ

/-- Theorem stating that if Jonathan travels the same distance each day and 
    spends a total of 6 hours exercising in a week, given his speeds on different days, 
    he travels 6 miles on each exercise day. -/
theorem jonathan_exercise_distance (routine : ExerciseRoutine) 
  (h1 : routine.monday_speed = 2)
  (h2 : routine.wednesday_speed = 3)
  (h3 : routine.friday_speed = 6)
  (h4 : routine.total_time = 6)
  (h5 : ∃ d : ℝ, d > 0 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time) :
  ∃ d : ℝ, d = 6 ∧ 
    d / routine.monday_speed + 
    d / routine.wednesday_speed + 
    d / routine.friday_speed = routine.total_time := by
  sorry

end jonathan_exercise_distance_l824_82491


namespace probability_A_and_B_selected_l824_82435

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B when choosing 3 students out of 5 -/
def prob_select_A_and_B : ℚ := 3 / 10

theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2)) / 
  (Nat.choose total_students selected_students) = prob_select_A_and_B :=
sorry

end probability_A_and_B_selected_l824_82435


namespace two_digit_number_solution_l824_82465

/-- A two-digit number with unit digit greater than tens digit by 2 and less than 30 -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 10 = (n / 10) + 2 ∧  -- unit digit greater than tens digit by 2
  n < 30  -- less than 30

theorem two_digit_number_solution :
  ∀ n : ℕ, TwoDigitNumber n → (n = 13 ∨ n = 24) :=
by sorry

end two_digit_number_solution_l824_82465


namespace min_value_expression_l824_82425

theorem min_value_expression (a b : ℝ) (h : a * b > 0) :
  (a^4 + 4*b^4 + 1) / (a*b) ≥ 4 := by
  sorry

end min_value_expression_l824_82425


namespace max_product_with_digits_1_to_5_l824_82460

def digit := Fin 5

def valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : digit), n = d₁.val + 1 + 10 * (d₂.val + 1) + 100 * (d₃.val + 1)

def valid_product (p : ℕ) : Prop :=
  ∃ (n₁ n₂ : ℕ), valid_number n₁ ∧ valid_number n₂ ∧ p = n₁ * n₂

theorem max_product_with_digits_1_to_5 :
  ∀ p, valid_product p → p ≤ 22412 :=
sorry

end max_product_with_digits_1_to_5_l824_82460


namespace proportional_y_value_l824_82418

/-- Given that y is directly proportional to x+1 and y=4 when x=1, 
    prove that y=6 when x=2 -/
theorem proportional_y_value (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = k * (x + 1)) →  -- y is directly proportional to x+1
  (4 = k * (1 + 1)) →                    -- when x=1, y=4
  (6 = k * (2 + 1)) :=                   -- prove y=6 when x=2
by
  sorry


end proportional_y_value_l824_82418


namespace square_triangle_area_equality_l824_82427

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 32 / 3 := by
  sorry

end square_triangle_area_equality_l824_82427


namespace odd_function_g_l824_82492

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f in terms of g -/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem odd_function_g (g : ℝ → ℝ) :
  IsOdd (f g) → (f g 1 = 1) → g = fun x ↦ x^5 - x^2 := by
  sorry

end odd_function_g_l824_82492


namespace base_conversion_and_operation_l824_82487

-- Define the base conversions
def base9_to_10 (n : ℕ) : ℕ := n

def base4_to_10 (n : ℕ) : ℕ := n

def base8_to_10 (n : ℕ) : ℕ := n

-- Define the operation
def operation (a b c d : ℕ) : ℕ := a / b - c + d

-- Theorem statement
theorem base_conversion_and_operation :
  operation (base9_to_10 1357) (base4_to_10 100) (base8_to_10 2460) (base9_to_10 5678) = 2938 := by
  sorry

-- Additional lemmas for individual base conversions
lemma base9_1357 : base9_to_10 1357 = 1024 := by sorry
lemma base4_100 : base4_to_10 100 = 16 := by sorry
lemma base8_2460 : base8_to_10 2460 = 1328 := by sorry
lemma base9_5678 : base9_to_10 5678 = 4202 := by sorry

end base_conversion_and_operation_l824_82487


namespace bowling_team_weight_problem_l824_82431

theorem bowling_team_weight_problem (original_players : ℕ) (original_avg_weight : ℝ)
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 121 →
  new_player1_weight = 110 →
  new_avg_weight = 113 →
  ∃ new_player2_weight : ℝ,
    new_player2_weight = 60 ∧
    (original_players : ℝ) * original_avg_weight + new_player1_weight + new_player2_weight =
      ((original_players : ℝ) + 2) * new_avg_weight :=
by sorry

end bowling_team_weight_problem_l824_82431


namespace adjacent_same_tribe_l824_82471

-- Define the four tribes
inductive Tribe
| Human
| Dwarf
| Elf
| Goblin

-- Define the seating arrangement
def Seating := Fin 33 → Tribe

-- Define the condition that humans cannot sit next to goblins
def NoHumanNextToGoblin (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Human ∧ s (i + 1) = Tribe.Goblin) ∨
                (s i = Tribe.Goblin ∧ s (i + 1) = Tribe.Human) → False

-- Define the condition that elves cannot sit next to dwarves
def NoElfNextToDwarf (s : Seating) : Prop :=
  ∀ i : Fin 33, (s i = Tribe.Elf ∧ s (i + 1) = Tribe.Dwarf) ∨
                (s i = Tribe.Dwarf ∧ s (i + 1) = Tribe.Elf) → False

-- Define the property of having adjacent same tribe representatives
def HasAdjacentSameTribe (s : Seating) : Prop :=
  ∃ i : Fin 33, s i = s (i + 1)

-- State the theorem
theorem adjacent_same_tribe (s : Seating) 
  (no_human_goblin : NoHumanNextToGoblin s) 
  (no_elf_dwarf : NoElfNextToDwarf s) : 
  HasAdjacentSameTribe s :=
sorry

end adjacent_same_tribe_l824_82471


namespace mixture_ratio_theorem_l824_82421

/-- Represents the components of the mixture -/
inductive Component
  | Milk
  | Water
  | Juice

/-- Calculates the amount of a component in the initial mixture -/
def initial_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => 60 * (5 / 8)
  | Component.Water => 60 * (2 / 8)
  | Component.Juice => 60 * (1 / 8)

/-- Calculates the amount of a component after adding water and juice -/
def final_amount (c : Component) : ℚ :=
  match c with
  | Component.Milk => initial_amount Component.Milk
  | Component.Water => initial_amount Component.Water + 15
  | Component.Juice => initial_amount Component.Juice + 5

/-- Represents the final ratio of the mixture components -/
def final_ratio : Fin 3 → ℕ
  | 0 => 15  -- Milk
  | 1 => 12  -- Water
  | 2 => 5   -- Juice
  | _ => 0   -- This case is unreachable, but needed for completeness

theorem mixture_ratio_theorem :
  ∃ (k : ℚ), k > 0 ∧
    (final_amount Component.Milk = k * final_ratio 0) ∧
    (final_amount Component.Water = k * final_ratio 1) ∧
    (final_amount Component.Juice = k * final_ratio 2) :=
sorry

end mixture_ratio_theorem_l824_82421


namespace sixth_term_of_sequence_l824_82489

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem sixth_term_of_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 6) :
  geometric_sequence a₁ (a₂ / a₁) 6 = 96 := by
sorry

end sixth_term_of_sequence_l824_82489


namespace jordyn_zrinka_age_ratio_l824_82461

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove the ratio of Jordyn's to Zrinka's age -/
theorem jordyn_zrinka_age_ratio :
  ∀ (mehki_age jordyn_age zrinka_age : ℕ),
  mehki_age = 22 →
  zrinka_age = 6 →
  mehki_age = jordyn_age + 10 →
  (jordyn_age : ℚ) / (zrinka_age : ℚ) = 2 := by
sorry

end jordyn_zrinka_age_ratio_l824_82461


namespace juice_fraction_is_one_fourth_l824_82422

/-- Represents the contents of a cup -/
structure CupContents where
  milk : ℚ
  juice : ℚ

/-- Represents the state of both cups -/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups := {
  cup1 := { milk := 6, juice := 0 },
  cup2 := { milk := 0, juice := 6 }
}

def transfer_milk (state : TwoCups) : TwoCups := {
  cup1 := { milk := state.cup1.milk * 2/3, juice := state.cup1.juice },
  cup2 := { milk := state.cup2.milk + state.cup1.milk * 1/3, juice := state.cup2.juice }
}

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total2 := state.cup2.milk + state.cup2.juice
  let transfer_amount := total2 * 1/4
  let milk_fraction := state.cup2.milk / total2
  let juice_fraction := state.cup2.juice / total2
  {
    cup1 := {
      milk := state.cup1.milk + transfer_amount * milk_fraction,
      juice := state.cup1.juice + transfer_amount * juice_fraction
    },
    cup2 := {
      milk := state.cup2.milk - transfer_amount * milk_fraction,
      juice := state.cup2.juice - transfer_amount * juice_fraction
    }
  }

def final_state : TwoCups :=
  transfer_mixture (transfer_milk initial_state)

theorem juice_fraction_is_one_fourth :
  (final_state.cup1.juice) / (final_state.cup1.milk + final_state.cup1.juice) = 1/4 := by
  sorry


end juice_fraction_is_one_fourth_l824_82422


namespace quadratic_equation_roots_l824_82454

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ ≠ x₁ ∧ 
   ∀ x : ℝ, x^2 - 6*x + k = 0 ↔ (x = x₁ ∨ x = x₂)) → 
  k = 8 ∧ ∃ x₂ : ℝ, x₂ = 4 :=
by sorry

end quadratic_equation_roots_l824_82454


namespace possible_last_ball_A_impossible_last_ball_C_l824_82452

/-- Represents the types of balls in the simulator -/
inductive BallType
  | A
  | B
  | C

/-- Represents the state of the simulator -/
structure SimulatorState :=
  (countA : Nat)
  (countB : Nat)
  (countC : Nat)

/-- Represents a collision between two ball types -/
def collision (a b : BallType) : BallType :=
  match a, b with
  | BallType.A, BallType.A => BallType.C
  | BallType.B, BallType.B => BallType.C
  | BallType.C, BallType.C => BallType.C
  | BallType.A, BallType.B => BallType.C
  | BallType.B, BallType.A => BallType.C
  | BallType.A, BallType.C => BallType.B
  | BallType.C, BallType.A => BallType.B
  | BallType.B, BallType.C => BallType.A
  | BallType.C, BallType.B => BallType.A

/-- The initial state of the simulator -/
def initialState : SimulatorState :=
  { countA := 12, countB := 9, countC := 10 }

/-- Predicate to check if a state has only one ball left -/
def hasOneBallLeft (state : SimulatorState) : Prop :=
  state.countA + state.countB + state.countC = 1

/-- Theorem stating that it's possible for the last ball to be type A -/
theorem possible_last_ball_A :
  ∃ (finalState : SimulatorState),
    hasOneBallLeft finalState ∧ finalState.countA = 1 :=
sorry

/-- Theorem stating that it's impossible for the last ball to be type C -/
theorem impossible_last_ball_C :
  ∀ (finalState : SimulatorState),
    hasOneBallLeft finalState → finalState.countC = 0 :=
sorry

end possible_last_ball_A_impossible_last_ball_C_l824_82452


namespace valid_representation_characterization_l824_82459

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2

theorem valid_representation_characterization :
  ∀ n : ℕ, is_valid_representation n ↔ (n = 8 ∨ n = 20) :=
sorry

end valid_representation_characterization_l824_82459


namespace symmetric_point_example_l824_82424

/-- Given a point (x, y) in a 2D coordinate system, this function returns the point that is symmetric to (x, y) with respect to the origin. -/
def symmetricPointOrigin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem stating that the point symmetric to (-2, 5) with respect to the origin is (2, -5). -/
theorem symmetric_point_example : symmetricPointOrigin (-2) 5 = (2, -5) := by
  sorry

end symmetric_point_example_l824_82424


namespace absolute_value_greater_than_one_l824_82457

theorem absolute_value_greater_than_one (a b : ℝ) 
  (h1 : b * (a + b + 1) < 0) 
  (h2 : b * (a + b - 1) < 0) : 
  |a| > 1 := by
sorry

end absolute_value_greater_than_one_l824_82457


namespace division_4073_by_38_l824_82472

theorem division_4073_by_38 : ∃ (q r : ℕ), 4073 = 38 * q + r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end division_4073_by_38_l824_82472


namespace smallest_k_inequality_half_satisfies_inequality_l824_82493

theorem smallest_k_inequality (k : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + k * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  k ≥ (1 / 2 : ℝ) :=
sorry

theorem half_satisfies_inequality : 
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
    Real.sqrt (x * y) + (1 / 2 : ℝ) * Real.sqrt (|x - y|) ≥ (x + y) / 2 :=
sorry

end smallest_k_inequality_half_satisfies_inequality_l824_82493


namespace sample_size_is_140_l824_82450

/-- Represents a school with students and a height measurement study -/
structure School where
  total_students : ℕ
  measured_students : ℕ
  measured_students_le_total : measured_students ≤ total_students

/-- The sample size of a height measurement study in a school -/
def sample_size (s : School) : ℕ := s.measured_students

/-- Theorem stating that for a school with 1740 students and 140 measured students, the sample size is 140 -/
theorem sample_size_is_140 (s : School) 
  (h1 : s.total_students = 1740) 
  (h2 : s.measured_students = 140) : 
  sample_size s = 140 := by sorry

end sample_size_is_140_l824_82450


namespace sqrt_3_irrational_l824_82432

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l824_82432


namespace student_number_proof_l824_82497

theorem student_number_proof : 
  ∃ x : ℝ, (2 * x - 138 = 102) ∧ (x = 120) := by
  sorry

end student_number_proof_l824_82497


namespace algebraic_expression_value_l824_82498

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m - n = -2) 
  (h2 : m * n = 3) : 
  -m^3*n + 2*m^2*n^2 - m*n^3 = -12 := by
  sorry

end algebraic_expression_value_l824_82498


namespace algebra_test_average_l824_82414

theorem algebra_test_average : ∀ (male_count female_count : ℕ) 
  (male_avg female_avg overall_avg : ℚ),
  male_count = 8 →
  female_count = 28 →
  male_avg = 83 →
  female_avg = 92 →
  overall_avg = (male_count * male_avg + female_count * female_avg) / (male_count + female_count) →
  overall_avg = 90 :=
by
  sorry

end algebra_test_average_l824_82414


namespace container_capacity_increase_l824_82456

/-- Proves that quadrupling all dimensions of a container increases its capacity by a factor of 64 -/
theorem container_capacity_increase (original_capacity new_capacity : ℝ) : 
  original_capacity = 5 → new_capacity = 320 → new_capacity = original_capacity * 64 := by
  sorry

end container_capacity_increase_l824_82456


namespace f_two_zeros_iff_a_in_range_l824_82401

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - (a + 2) * x

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z : ℝ, z > 0 → f a z = 0 → (z = x ∨ z = y)

theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_exactly_two_zeros a ↔ -1 < a ∧ a < 0 :=
sorry

end f_two_zeros_iff_a_in_range_l824_82401


namespace problem_statement_l824_82448

theorem problem_statement : (1 / ((-5^2)^3)) * ((-5)^8) * Real.sqrt 5 = 5^(5/2) := by
  sorry

end problem_statement_l824_82448


namespace joan_has_16_seashells_l824_82409

/-- The number of seashells Joan has now, given that she found 79 and gave away 63. -/
def joans_remaining_seashells (found : ℕ) (gave_away : ℕ) : ℕ :=
  found - gave_away

/-- Theorem stating that Joan has 16 seashells now. -/
theorem joan_has_16_seashells : 
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end joan_has_16_seashells_l824_82409


namespace max_remainder_problem_l824_82406

theorem max_remainder_problem :
  ∃ (n : ℕ) (r : ℕ),
    2013 ≤ n ∧ n ≤ 2156 ∧
    n % 5 = r ∧ n % 11 = r ∧ n % 13 = r ∧
    r ≤ 4 ∧
    ∀ (m : ℕ) (s : ℕ),
      2013 ≤ m ∧ m ≤ 2156 ∧
      m % 5 = s ∧ m % 11 = s ∧ m % 13 = s →
      s ≤ r :=
by sorry

end max_remainder_problem_l824_82406


namespace custom_op_two_three_l824_82499

-- Define the custom operation
def customOp (x y : ℕ) : ℕ := x + y^2

-- Theorem statement
theorem custom_op_two_three : customOp 2 3 = 11 := by
  sorry

end custom_op_two_three_l824_82499


namespace range_of_a_l824_82485

def A (a : ℝ) : Set ℝ := {x | a * x < 1}
def B : Set ℝ := {x | |x - 1| < 2}

theorem range_of_a (a : ℝ) (h : A a ∪ B = A a) : a ∈ Set.Icc (-1 : ℝ) (1/3) := by
  sorry

end range_of_a_l824_82485


namespace smallest_distance_between_circle_and_ellipse_l824_82417

theorem smallest_distance_between_circle_and_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 16) = 1}
  ∃ (d : ℝ), d = (Real.sqrt 35 - 2) / 2 ∧
    (∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle → p₂ ∈ ellipse →
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) ≥ d) ∧
    (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle ∧ p₂ ∈ ellipse ∧
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = d) :=
by
  sorry

end smallest_distance_between_circle_and_ellipse_l824_82417


namespace inequality_proof_l824_82416

theorem inequality_proof (k m n : ℕ) (hk : k > 0) (hm : m > 0) (hn : n > 0) 
  (hkm : k ≠ m) (hkn : k ≠ n) (hmn : m ≠ n) : 
  (k - 1 / k) * (m - 1 / m) * (n - 1 / n) ≤ k * m * n - (k + m + n) := by
  sorry

end inequality_proof_l824_82416


namespace shares_multiple_l824_82484

/-- Represents the shares of money for three children -/
structure Shares where
  anusha : ℕ
  babu : ℕ
  esha : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem shares_multiple (s : Shares) 
  (h1 : 12 * s.anusha = 8 * s.babu)
  (h2 : ∃ k : ℕ, 8 * s.babu = k * s.esha)
  (h3 : s.anusha + s.babu + s.esha = 378)
  (h4 : s.anusha = 84) :
  ∃ k : ℕ, 8 * s.babu = 6 * s.esha := by
  sorry


end shares_multiple_l824_82484


namespace expression_is_integer_l824_82495

theorem expression_is_integer (n : ℤ) : ∃ k : ℤ, (n / 3 : ℚ) + (n^2 / 2 : ℚ) + (n^3 / 6 : ℚ) = k := by
  sorry

end expression_is_integer_l824_82495


namespace expression_simplification_and_evaluation_l824_82408

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ 1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (1 - 3 / (a + 2)) / ((a^2 - 2*a + 1) / (a^2 - 4)) = (a - 2) / (a - 1) ∧
  (0 - 2) / (0 - 1) = 2 := by
sorry

end expression_simplification_and_evaluation_l824_82408


namespace abcd_hex_binary_digits_l824_82463

-- Define the hexadecimal number ABCD₁₆ as its decimal equivalent
def abcd_hex : ℕ := 43981

-- Theorem stating that the binary representation of ABCD₁₆ requires 16 bits
theorem abcd_hex_binary_digits : 
  (Nat.log 2 abcd_hex).succ = 16 := by sorry

end abcd_hex_binary_digits_l824_82463


namespace unique_fraction_representation_l824_82467

theorem unique_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end unique_fraction_representation_l824_82467


namespace f_increasing_on_positive_reals_l824_82430

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_increasing_on_positive_reals :
  ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_increasing_on_positive_reals_l824_82430


namespace max_value_part1_l824_82479

theorem max_value_part1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/2) * x * (1 - 2*x) ≤ 1/16 := by sorry

end max_value_part1_l824_82479


namespace equation_solutions_range_l824_82444

theorem equation_solutions_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) →
  m ∈ Set.Icc 0 8 :=
sorry

end equation_solutions_range_l824_82444


namespace two_integers_sum_and_lcm_l824_82439

theorem two_integers_sum_and_lcm : ∃ (m n : ℕ), 
  m > 0 ∧ n > 0 ∧ m + n = 60 ∧ Nat.lcm m n = 273 ∧ m = 21 ∧ n = 39 := by
  sorry

end two_integers_sum_and_lcm_l824_82439


namespace triangle_problem_l824_82480

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 3 * a * Real.sin C = c * Real.cos A)
  (h_B : B = π / 4)
  (h_area : 1 / 2 * a * c * Real.sin B = 9) :
  Real.sin A = Real.sqrt 10 / 10 ∧ a = 3 := by
  sorry


end triangle_problem_l824_82480


namespace inequality_equivalence_l824_82478

theorem inequality_equivalence (x : ℝ) : 
  3 * x - 6 > 12 - 2 * x + x^2 ↔ -1 < x ∧ x < 6 := by
  sorry

end inequality_equivalence_l824_82478


namespace box_volume_l824_82436

/-- A rectangular box with given face areas and length-height relationship has a volume of 120 cubic inches -/
theorem box_volume (l w h : ℝ) (area1 : l * w = 30) (area2 : w * h = 20) (area3 : l * h = 12) (length_height : l = h + 1) :
  l * w * h = 120 := by
  sorry

end box_volume_l824_82436


namespace arithmetic_mean_1_5_l824_82474

theorem arithmetic_mean_1_5 (m : ℝ) : m = (1 + 5) / 2 → m = 3 := by
  sorry

end arithmetic_mean_1_5_l824_82474


namespace max_ab_value_l824_82488

theorem max_ab_value (a b : ℝ) : 
  (∃! x, x^2 - 2*a*x - b^2 + 12 ≤ 0) → 
  ∀ c, a*b ≤ c → c ≤ 6 :=
by sorry

end max_ab_value_l824_82488
