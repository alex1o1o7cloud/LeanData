import Mathlib

namespace distance_between_points_l184_18411

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 7)
  let p2 : ℝ × ℝ := (3, -2)
  dist p1 p2 = 9 := by sorry

end distance_between_points_l184_18411


namespace triangle_max_area_l184_18475

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  (Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) →
  (a + b + c = 12) →
  (∀ a' b' c' : ℝ, a' + b' + c' = 12 → 
    a' * b' * Real.sin C / 2 ≤ 36 * (3 - 2 * Real.sqrt 2)) :=
by sorry

end triangle_max_area_l184_18475


namespace inequality_addition_l184_18433

theorem inequality_addition {a b c d : ℝ} (hab : a > b) (hcd : c > d) (hc : c ≠ 0) (hd : d ≠ 0) :
  a + c > b + d := by
  sorry

end inequality_addition_l184_18433


namespace vector_sum_magnitude_l184_18428

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 4) →
  (Real.sqrt (b.1^2 + b.2^2) = 3) →
  (angle_between a b = Real.pi / 3) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 37 :=
by sorry

end vector_sum_magnitude_l184_18428


namespace sin_product_equals_neg_two_fifths_l184_18453

theorem sin_product_equals_neg_two_fifths (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -2/5 := by
  sorry

end sin_product_equals_neg_two_fifths_l184_18453


namespace angle_equality_l184_18431

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ + 2 * Real.sin θ) : 
  θ = 10 * π / 180 := by
  sorry

end angle_equality_l184_18431


namespace congruence_implies_b_zero_l184_18418

theorem congruence_implies_b_zero (a b c m : ℤ) (h_m : m > 1) 
  (h_cong : ∀ n : ℕ, (a^n + b*n + c) % m = 0) : 
  b % m = 0 ∧ (b^2) % m = 0 := by
  sorry

end congruence_implies_b_zero_l184_18418


namespace correct_distribution_l184_18429

/-- Represents the amount of logs contributed by each person -/
structure Contribution where
  troykina : ℕ
  pyatorkina : ℕ
  bestoplivny : ℕ

/-- Represents the payment made by Bestoplivny in kopecks -/
def bestoplivny_payment : ℕ := 80

/-- Calculates the fair distribution of the payment -/
def calculate_distribution (c : Contribution) : ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of the payment -/
theorem correct_distribution (c : Contribution) 
  (h1 : c.troykina = 3)
  (h2 : c.pyatorkina = 5)
  (h3 : c.bestoplivny = 0) :
  calculate_distribution c = (10, 70) := by sorry

end correct_distribution_l184_18429


namespace inequality_proof_l184_18465

theorem inequality_proof (x : ℝ) : x > -4/3 → 3 - 1/(3*x + 4) < 5 := by
  sorry

end inequality_proof_l184_18465


namespace parabola_y_axis_intersection_l184_18401

/-- The parabola is defined by the equation y = -x^2 + 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 3*x - 4

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -4 :=
sorry

end parabola_y_axis_intersection_l184_18401


namespace smallest_number_problem_l184_18404

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 30 →
  b = 31 →
  a ≤ b ∧ b ≤ c →
  c = b + 6 →
  a = 22 := by
sorry

end smallest_number_problem_l184_18404


namespace rhombus_area_l184_18410

/-- A rhombus with side length √113 and diagonal difference 8 has area 194. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = (Real.sqrt 210)^2 - 4^2 →
  area = 194 := by sorry

end rhombus_area_l184_18410


namespace barbara_winning_condition_l184_18400

/-- The game rules for Alberto and Barbara --/
structure GameRules where
  alberto_choice : ℕ → ℕ
  barbara_choice : ℕ → ℕ → ℕ
  alberto_move : ℕ → ℕ
  max_moves : ℕ

/-- Barbara's winning condition --/
def barbara_wins (n : ℕ) (rules : GameRules) : Prop :=
  ∃ (strategy : ℕ → ℕ), ∀ (alberto_plays : ℕ → ℕ),
    ∃ (m : ℕ), m ≤ rules.max_moves ∧
    (strategy (alberto_plays m)) = n

/-- The main theorem --/
theorem barbara_winning_condition (n : ℕ) (h : n > 1) :
  (∃ (rules : GameRules), barbara_wins n rules) ↔ (∃ (k : ℕ), n = 6 * k) :=
sorry

end barbara_winning_condition_l184_18400


namespace negation_of_existence_negation_of_proposition_l184_18496

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x₀ : ℝ, x₀ - 2 > Real.log x₀) ↔ (∀ x : ℝ, x - 2 ≤ Real.log x) :=
by sorry

end negation_of_existence_negation_of_proposition_l184_18496


namespace book_sales_l184_18440

theorem book_sales (wednesday_sales : ℕ) : 
  wednesday_sales + 3 * wednesday_sales + 3 * wednesday_sales / 5 = 69 → 
  wednesday_sales = 15 := by
sorry

end book_sales_l184_18440


namespace seating_arrangements_eq_384_l184_18459

/-- Represents the number of executives -/
def num_executives : ℕ := 5

/-- Represents the total number of people (executives + partners) -/
def total_people : ℕ := 2 * num_executives

/-- Calculates the number of distinct seating arrangements -/
def seating_arrangements : ℕ :=
  (List.range num_executives).foldl (λ acc i => acc * (total_people - 2 * i)) 1 / total_people

/-- Theorem stating that the number of distinct seating arrangements is 384 -/
theorem seating_arrangements_eq_384 : seating_arrangements = 384 := by sorry

end seating_arrangements_eq_384_l184_18459


namespace total_sum_lent_l184_18469

/-- Proves that the total sum lent is 2665 given the specified conditions --/
theorem total_sum_lent (first_part second_part : ℕ) : 
  second_part = 1640 →
  (first_part * 8 * 3) = (second_part * 3 * 5) →
  first_part + second_part = 2665 := by
  sorry

#check total_sum_lent

end total_sum_lent_l184_18469


namespace tom_fruit_purchase_l184_18468

/-- Given Tom's purchase of apples and mangoes, prove the amount of mangoes bought -/
theorem tom_fruit_purchase (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 70)
  (h4 : total_paid = 1190) :
  (total_paid - apple_kg * apple_rate) / mango_rate = 9 := by
  sorry

end tom_fruit_purchase_l184_18468


namespace apple_price_is_two_l184_18462

/-- The cost of items in Fabian's shopping basket -/
def shopping_cost (apple_price : ℝ) : ℝ :=
  5 * apple_price +  -- 5 kg of apples
  3 * (apple_price - 1) +  -- 3 packs of sugar
  0.5 * 6  -- 500g of walnuts

/-- Theorem: The price of apples is $2 per kg -/
theorem apple_price_is_two :
  ∃ (apple_price : ℝ), apple_price = 2 ∧ shopping_cost apple_price = 16 :=
by
  sorry

end apple_price_is_two_l184_18462


namespace intersection_point_determines_d_l184_18488

theorem intersection_point_determines_d : ∀ d : ℝ,
  (∃ x y : ℝ, 3 * x - 4 * y = d ∧ 6 * x + 8 * y = -d ∧ x = 2 ∧ y = -3) →
  d = 18 := by
sorry

end intersection_point_determines_d_l184_18488


namespace line_through_three_points_l184_18455

/-- A line contains the points (-2, 7), (7, k), and (21, 4). This theorem proves that k = 134/23. -/
theorem line_through_three_points (k : ℚ) : 
  (∃ (m b : ℚ), 
    (7 : ℚ) = m * (-2 : ℚ) + b ∧ 
    k = m * (7 : ℚ) + b ∧ 
    (4 : ℚ) = m * (21 : ℚ) + b) → 
  k = 134 / 23 := by
sorry

end line_through_three_points_l184_18455


namespace least_tiles_required_l184_18461

def room_length : ℕ := 544
def room_width : ℕ := 374

theorem least_tiles_required (length width : ℕ) (h1 : length = room_length) (h2 : width = room_width) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    length % tile_size = 0 ∧
    width % tile_size = 0 ∧
    (length / tile_size) * (width / tile_size) = 176 :=
sorry

end least_tiles_required_l184_18461


namespace basketball_probability_l184_18409

theorem basketball_probability (free_throw high_school pro : ℝ) 
  (h1 : free_throw = 4/5)
  (h2 : high_school = 1/2)
  (h3 : pro = 1/3) :
  1 - (1 - free_throw) * (1 - high_school) * (1 - pro) = 14/15 := by
  sorry

end basketball_probability_l184_18409


namespace larger_integer_value_l184_18442

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 ∨ b = 21 :=
sorry

end larger_integer_value_l184_18442


namespace top_layer_blocks_l184_18427

/-- Represents a four-layer pyramid with a specific block distribution -/
structure Pyramid :=
  (top : ℕ)  -- Number of blocks in the top layer

/-- The total number of blocks in the pyramid -/
def Pyramid.total (p : Pyramid) : ℕ :=
  p.top + 3 * p.top + 9 * p.top + 27 * p.top

theorem top_layer_blocks (p : Pyramid) :
  p.total = 40 → p.top = 1 := by
  sorry

#check top_layer_blocks

end top_layer_blocks_l184_18427


namespace trig_simplification_l184_18441

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos x ^ 2 := by
  sorry

end trig_simplification_l184_18441


namespace eggs_division_l184_18478

theorem eggs_division (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) :
  total_eggs = 15 →
  num_groups = 3 →
  eggs_per_group * num_groups = total_eggs →
  eggs_per_group = 5 := by
  sorry

end eggs_division_l184_18478


namespace lte_lemma_largest_power_of_two_dividing_difference_l184_18432

-- Define the valuation function v_2
def v_2 (n : ℕ) : ℕ := sorry

-- Define the Lifting The Exponent Lemma
theorem lte_lemma (a b : ℕ) (h : Odd a ∧ Odd b) :
  v_2 (a^4 - b^4) = v_2 (a - b) + v_2 4 + v_2 (a + b) - 1 := sorry

-- Main theorem
theorem largest_power_of_two_dividing_difference :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) := by sorry

end lte_lemma_largest_power_of_two_dividing_difference_l184_18432


namespace range_of_f_l184_18425

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end range_of_f_l184_18425


namespace min_sum_of_primes_l184_18424

theorem min_sum_of_primes (p q : ℕ) : 
  p > 1 → q > 1 → Nat.Prime p → Nat.Prime q → 
  17 * (p + 1) = 21 * (q + 1) → 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → Nat.Prime p' → Nat.Prime q' → 
    17 * (p' + 1) = 21 * (q' + 1) → p + q ≤ p' + q') → 
  p + q = 70 := by
sorry

end min_sum_of_primes_l184_18424


namespace room_occupancy_l184_18454

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (2 : ℚ) / 3 * people = (3 : ℚ) / 4 * chairs ∧ 
  chairs - (3 : ℚ) / 4 * chairs = 6 →
  people = 27 := by
sorry

end room_occupancy_l184_18454


namespace line_passes_first_third_quadrants_iff_positive_slope_l184_18458

/-- A line passes through the first and third quadrants if and only if its slope is positive -/
theorem line_passes_first_third_quadrants_iff_positive_slope (k : ℝ) :
  (k ≠ 0 ∧ ∀ x y : ℝ, y = k * x → (x > 0 → y > 0) ∧ (x < 0 → y < 0)) ↔ k > 0 := by
  sorry

end line_passes_first_third_quadrants_iff_positive_slope_l184_18458


namespace exponent_calculation_l184_18417

theorem exponent_calculation : 3^3 * 5^3 * 3^5 * 5^5 = 15^8 := by
  sorry

end exponent_calculation_l184_18417


namespace parallel_vectors_y_value_l184_18439

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given vectors a and b, with a parallel to b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
    (ha : a = (2, 3)) 
    (hb : b = (4, -1 + y)) 
    (h_parallel : parallel a b) : 
  y = 7 := by
  sorry

end parallel_vectors_y_value_l184_18439


namespace peters_remaining_money_l184_18482

/-- Calculates Peter's remaining money after market purchases -/
theorem peters_remaining_money
  (initial_amount : ℕ)
  (potato_quantity potato_price : ℕ)
  (tomato_quantity tomato_price : ℕ)
  (cucumber_quantity cucumber_price : ℕ)
  (banana_quantity banana_price : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_quantity = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_quantity = 9)
  (h5 : tomato_price = 3)
  (h6 : cucumber_quantity = 5)
  (h7 : cucumber_price = 4)
  (h8 : banana_quantity = 3)
  (h9 : banana_price = 5) :
  initial_amount - (potato_quantity * potato_price +
                    tomato_quantity * tomato_price +
                    cucumber_quantity * cucumber_price +
                    banana_quantity * banana_price) = 426 := by
  sorry

end peters_remaining_money_l184_18482


namespace min_value_reciprocal_sum_l184_18472

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 → 1/a + 4/b ≥ 9/2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 4/b = 9/2) :=
by sorry

end min_value_reciprocal_sum_l184_18472


namespace no_integer_solution_l184_18474

def is_all_twos (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2

theorem no_integer_solution : ¬ ∃ (N : ℤ), is_all_twos (2008 * N.natAbs) :=
  sorry

end no_integer_solution_l184_18474


namespace triangle_min_value_l184_18476

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB - bcosA = c/3, then the minimum value of (acosA + bcosB) / (acosB) is √2. -/
theorem triangle_min_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin B →
  a * Real.cos B - b * Real.cos A = c / 3 →
  ∃ (x : ℝ), x = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) ∧
    x ≥ Real.sqrt 2 ∧
    ∀ (y : ℝ), y = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) → y ≥ x :=
by sorry

end triangle_min_value_l184_18476


namespace average_of_remaining_numbers_l184_18402

theorem average_of_remaining_numbers 
  (total : ℕ) 
  (subset : ℕ) 
  (remaining : ℕ) 
  (total_sum : ℚ) 
  (subset_sum : ℚ) 
  (h1 : total = 5) 
  (h2 : subset = 3) 
  (h3 : remaining = total - subset) 
  (h4 : total_sum / total = 6) 
  (h5 : subset_sum / subset = 4) : 
  (total_sum - subset_sum) / remaining = 9 := by
sorry

end average_of_remaining_numbers_l184_18402


namespace expression_equality_l184_18405

theorem expression_equality : 
  (Real.sqrt 12) / 2 + |Real.sqrt 3 - 2| - Real.tan (π / 3) = 2 - Real.sqrt 3 := by
  sorry

end expression_equality_l184_18405


namespace room_area_calculation_l184_18412

/-- The area of a rectangular room with width 8 feet and length 1.5 feet is 12 square feet. -/
theorem room_area_calculation (width length area : ℝ) : 
  width = 8 → length = 1.5 → area = width * length → area = 12 := by
sorry

end room_area_calculation_l184_18412


namespace system_solutions_product_l184_18435

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x^3 - 5*x*y^2 = 21 ∧ y^3 - 5*x^2*y = 28

-- Define the theorem
theorem system_solutions_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  system x₁ y₁ ∧ system x₂ y₂ ∧ system x₃ y₃ ∧
  (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃) →
  (11 - x₁/y₁) * (11 - x₂/y₂) * (11 - x₃/y₃) = 1729 :=
by sorry

end system_solutions_product_l184_18435


namespace time_for_A_to_reach_B_l184_18448

-- Define the total distance between points A and B
variable (S : ℝ) 

-- Define the speeds of A and B
variable (v_A v_B : ℝ)

-- Define the time when B catches up to A for the first time
variable (t : ℝ)

-- Theorem statement
theorem time_for_A_to_reach_B 
  (h1 : v_A * (t + 48/60) = v_B * t) 
  (h2 : v_A * (t + 48/60) = 2/3 * S) 
  (h3 : v_A * (t + 48/60 + 1/2 * t + 6/60) + 6/60 * v_B = S) 
  : (108 : ℝ) - (96 : ℝ) = 12 := by
  sorry


end time_for_A_to_reach_B_l184_18448


namespace donut_ratio_l184_18486

/-- Given a total of 40 donuts shared among Delta, Beta, and Gamma,
    where Delta takes 8 donuts and Gamma takes 8 donuts,
    prove that the ratio of Beta's donuts to Gamma's donuts is 3:1. -/
theorem donut_ratio :
  ∀ (total delta gamma beta : ℕ),
    total = 40 →
    delta = 8 →
    gamma = 8 →
    beta = total - delta - gamma →
    beta / gamma = 3 := by
  sorry

end donut_ratio_l184_18486


namespace smallest_n_congruence_l184_18460

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 8 ∧ ∀ (m : ℕ), m > 0 → (3 * m) % 26 = 8 → n ≤ m :=
by sorry

end smallest_n_congruence_l184_18460


namespace intersection_A_B_C_subset_intersection_A_B_l184_18408

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 3*a*x + 2*a^2 < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

-- Theorem for the range of a such that C is a subset of A ∩ B
theorem C_subset_intersection_A_B (a : ℝ) : 
  C a ⊆ (A ∩ B) ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := by sorry

end intersection_A_B_C_subset_intersection_A_B_l184_18408


namespace min_bags_l184_18466

theorem min_bags (total_objects : ℕ) (red_boxes blue_boxes : ℕ) 
  (objects_per_red : ℕ) (objects_per_blue : ℕ) :
  total_objects = 731 ∧ 
  red_boxes = 17 ∧ 
  blue_boxes = 43 ∧ 
  objects_per_red = 43 ∧ 
  objects_per_blue = 17 →
  ∃ (n : ℕ), n > 0 ∧ 
    (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
      objects_per_red * a + objects_per_blue * b = total_objects) ∧
    (∀ (m : ℕ), m > 0 ∧ 
      (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
        objects_per_red * a + objects_per_blue * b = total_objects) → 
      n ≤ m) ∧
    n = 17 := by
  sorry

end min_bags_l184_18466


namespace points_on_line_l184_18452

/-- Given a line defined by x = (y^2 / 3) - (2 / 5), if three points (m, n), (m + p, n + 9), and (m + q, n + 18) lie on this line, then p = 6n + 27 and q = 12n + 108 -/
theorem points_on_line (m n p q : ℝ) : 
  (m = n^2 / 3 - 2 / 5) →
  (m + p = (n + 9)^2 / 3 - 2 / 5) →
  (m + q = (n + 18)^2 / 3 - 2 / 5) →
  (p = 6 * n + 27 ∧ q = 12 * n + 108) := by
sorry

end points_on_line_l184_18452


namespace james_units_per_semester_l184_18491

/-- Given that James pays $2000 for 2 semesters and each unit costs $50,
    prove that he takes 20 units per semester. -/
theorem james_units_per_semester 
  (total_cost : ℕ) 
  (num_semesters : ℕ) 
  (unit_cost : ℕ) 
  (h1 : total_cost = 2000) 
  (h2 : num_semesters = 2) 
  (h3 : unit_cost = 50) : 
  (total_cost / num_semesters) / unit_cost = 20 := by
  sorry

end james_units_per_semester_l184_18491


namespace angle_coincidence_l184_18492

def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem angle_coincidence (α : ℝ) 
  (h1 : is_obtuse_angle α) 
  (h2 : (4 * α) % 360 = α % 360) : 
  α = 120 := by
  sorry

end angle_coincidence_l184_18492


namespace pen_pencil_difference_is_1500_l184_18437

/-- Represents the stationery order problem --/
structure StationeryOrder where
  pencilBoxes : ℕ
  pencilsPerBox : ℕ
  penCost : ℕ
  pencilCost : ℕ
  totalCost : ℕ

/-- Calculates the difference between pens and pencils ordered --/
def penPencilDifference (order : StationeryOrder) : ℕ :=
  let totalPencils := order.pencilBoxes * order.pencilsPerBox
  let totalPenCost := order.totalCost - order.pencilCost * totalPencils
  let totalPens := totalPenCost / order.penCost
  totalPens - totalPencils

/-- Theorem stating the difference between pens and pencils ordered --/
theorem pen_pencil_difference_is_1500 (order : StationeryOrder) 
  (h1 : order.pencilBoxes = 15)
  (h2 : order.pencilsPerBox = 80)
  (h3 : order.penCost = 5)
  (h4 : order.pencilCost = 4)
  (h5 : order.totalCost = 18300)
  (h6 : order.penCost * (penPencilDifference order + order.pencilBoxes * order.pencilsPerBox) > 
        2 * order.pencilCost * (order.pencilBoxes * order.pencilsPerBox)) :
  penPencilDifference order = 1500 := by
  sorry

#eval penPencilDifference { pencilBoxes := 15, pencilsPerBox := 80, penCost := 5, pencilCost := 4, totalCost := 18300 }

end pen_pencil_difference_is_1500_l184_18437


namespace tax_growth_equation_l184_18420

/-- Represents the average annual growth rate of taxes paid by a company over two years -/
def average_annual_growth_rate (initial_tax : ℝ) (final_tax : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_tax * (1 + x)^years = final_tax

/-- Theorem stating that the equation 40(1+x)^2 = 48.4 correctly represents the average annual growth rate -/
theorem tax_growth_equation (x : ℝ) :
  average_annual_growth_rate 40 48.4 2 x ↔ 40 * (1 + x)^2 = 48.4 :=
by sorry

end tax_growth_equation_l184_18420


namespace total_salary_proof_l184_18436

def salary_n : ℝ := 260

def salary_m : ℝ := 1.2 * salary_n

def total_salary : ℝ := salary_m + salary_n

theorem total_salary_proof : total_salary = 572 := by
  sorry

end total_salary_proof_l184_18436


namespace three_zeros_implies_a_eq_neg_e_l184_18483

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x
  else if x = 0 then 0
  else Real.exp (-x) - a * x

-- State the theorem
theorem three_zeros_implies_a_eq_neg_e (a : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a = -Real.exp 1 := by
  sorry


end three_zeros_implies_a_eq_neg_e_l184_18483


namespace smallest_valid_number_l184_18443

def is_valid (A : ℕ+) : Prop :=
  ∃ (a b : ℕ), 
    A = 2^a * 3^b ∧
    (a + 1) * (b + 1) = 3 * a * b

theorem smallest_valid_number : 
  is_valid 12 ∧ ∀ A : ℕ+, A < 12 → ¬is_valid A :=
sorry

end smallest_valid_number_l184_18443


namespace geometric_sequence_common_ratio_l184_18467

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by sorry

end geometric_sequence_common_ratio_l184_18467


namespace probability_two_red_shoes_l184_18481

/-- The probability of drawing two red shoes from a set of 6 red shoes and 4 green shoes is 1/3. -/
theorem probability_two_red_shoes (total_shoes : ℕ) (red_shoes : ℕ) (green_shoes : ℕ) :
  total_shoes = red_shoes + green_shoes →
  red_shoes = 6 →
  green_shoes = 4 →
  (red_shoes : ℚ) / total_shoes * ((red_shoes - 1) : ℚ) / (total_shoes - 1) = 1 / 3 :=
by sorry

end probability_two_red_shoes_l184_18481


namespace largest_sum_is_185_l184_18477

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of two two-digit numbers formed by three digits -/
def sum_XYZ (X Y Z : Digit) : ℕ := 10 * X.val + 11 * Y.val + Z.val

/-- The largest possible sum given the constraints -/
def largest_sum : ℕ := 185

/-- Theorem stating that 185 is the largest possible sum -/
theorem largest_sum_is_185 :
  ∀ X Y Z : Digit,
    X.val > Y.val →
    Y.val > Z.val →
    X ≠ Y →
    Y ≠ Z →
    X ≠ Z →
    sum_XYZ X Y Z ≤ largest_sum :=
sorry

end largest_sum_is_185_l184_18477


namespace vector_BC_l184_18422

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2) = (-7, -4) := by
  sorry

end vector_BC_l184_18422


namespace arithmetic_sequence_sum_ratio_l184_18419

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    if a₄/a₈ = 2/3, then S₇/S₁₅ = 14/45 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n))
  (h_ratio : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 := by
  sorry

end arithmetic_sequence_sum_ratio_l184_18419


namespace units_digit_of_3_power_2020_l184_18445

theorem units_digit_of_3_power_2020 : ∃ n : ℕ, 3^2020 ≡ 1 [ZMOD 10] :=
sorry

end units_digit_of_3_power_2020_l184_18445


namespace fraction_transformation_l184_18430

theorem fraction_transformation (x : ℚ) : 
  (3 + 2*x) / (4 + 3*x) = 5/9 → x = -7/3 := by
  sorry

end fraction_transformation_l184_18430


namespace blue_marble_probability_l184_18406

/-- Represents the probability of selecting a blue marble from a bag with specific conditions. -/
theorem blue_marble_probability (total : ℕ) (yellow : ℕ) (h1 : total = 60) (h2 : yellow = 20) :
  let green := yellow / 2
  let remaining := total - yellow - green
  let blue := remaining / 2
  (blue : ℚ) / total = 1/4 := by sorry

end blue_marble_probability_l184_18406


namespace most_cost_effective_boat_rental_l184_18489

/-- Represents the cost and capacity of a boat type -/
structure BoatType where
  capacity : Nat
  cost : Nat

/-- Represents a combination of boats -/
structure BoatCombination where
  largeboats : Nat
  smallboats : Nat

def totalPeople (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.capacity + b.smallboats * small.capacity

def totalCost (b : BoatCombination) (large : BoatType) (small : BoatType) : Nat :=
  b.largeboats * large.cost + b.smallboats * small.cost

def isSufficient (b : BoatCombination) (large : BoatType) (small : BoatType) (people : Nat) : Prop :=
  totalPeople b large small ≥ people

def isMoreCostEffective (b1 b2 : BoatCombination) (large : BoatType) (small : BoatType) : Prop :=
  totalCost b1 large small < totalCost b2 large small

theorem most_cost_effective_boat_rental :
  let large : BoatType := { capacity := 6, cost := 24 }
  let small : BoatType := { capacity := 4, cost := 20 }
  let people : Nat := 46
  let optimal : BoatCombination := { largeboats := 7, smallboats := 1 }
  (isSufficient optimal large small people) ∧
  (∀ b : BoatCombination, 
    isSufficient b large small people → 
    totalCost optimal large small ≤ totalCost b large small) := by
  sorry

#check most_cost_effective_boat_rental

end most_cost_effective_boat_rental_l184_18489


namespace divisibility_implication_l184_18495

theorem divisibility_implication (x y : ℤ) : 
  (∃ k : ℤ, 4*x - y = 3*k) → (∃ m : ℤ, 4*x^2 + 7*x*y - 2*y^2 = 9*m) :=
by sorry

end divisibility_implication_l184_18495


namespace park_area_theorem_l184_18480

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- Calculates the area of a rectangular park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- Calculates the perimeter of a rectangular park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- Calculates the fencing cost for a rectangular park -/
def fencingCost (park : RectangularPark) (costPerMeter : ℝ) : ℝ :=
  perimeter park * costPerMeter

theorem park_area_theorem (park : RectangularPark) :
  fencingCost park 0.5 = 155 → area park = 5766 := by
  sorry

end park_area_theorem_l184_18480


namespace larry_channels_l184_18498

/-- Calculates the final number of channels Larry has after a series of changes. -/
def final_channels (initial : ℕ) (removed1 : ℕ) (added1 : ℕ) (removed2 : ℕ) (added2 : ℕ) (added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that given the specific changes to Larry's channel package, he ends up with 147 channels. -/
theorem larry_channels : final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end larry_channels_l184_18498


namespace toy_value_proof_l184_18403

theorem toy_value_proof (total_toys : ℕ) (total_value : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_value = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    (total_toys - 1) * other_toy_value + special_toy_value = total_value ∧
    other_toy_value = 5 := by
  sorry

end toy_value_proof_l184_18403


namespace regression_line_intercept_l184_18444

/-- Given a regression line with slope 1.23 passing through (4, 5), prove its y-intercept is 0.08 -/
theorem regression_line_intercept (slope : ℝ) (x₀ y₀ : ℝ) (h1 : slope = 1.23) (h2 : x₀ = 4) (h3 : y₀ = 5) :
  y₀ = slope * x₀ + 0.08 := by
  sorry

end regression_line_intercept_l184_18444


namespace tangent_lines_slope_4_tangent_line_at_point_2_neg6_l184_18493

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_lines_slope_4 (x y : ℝ) :
  (4 * x - y - 18 = 0 ∨ 4 * x - y - 14 = 0) →
  ∃ x₀ : ℝ, f' x₀ = 4 ∧ y = f x₀ + 4 * (x - x₀) :=
sorry

theorem tangent_line_at_point_2_neg6 (x y : ℝ) :
  13 * x - y - 32 = 0 →
  y = f 2 + f' 2 * (x - 2) :=
sorry

end tangent_lines_slope_4_tangent_line_at_point_2_neg6_l184_18493


namespace port_distance_equation_l184_18447

/-- The distance between two ports satisfies a specific equation based on ship and river speeds --/
theorem port_distance_equation (ship_speed : ℝ) (current_speed : ℝ) (time_difference : ℝ) 
  (h1 : ship_speed = 26)
  (h2 : current_speed = 2)
  (h3 : time_difference = 3) :
  ∃ x : ℝ, x / (ship_speed + current_speed) = x / (ship_speed - current_speed) - time_difference :=
by sorry

end port_distance_equation_l184_18447


namespace factor_count_of_n_l184_18434

-- Define the number we're working with
def n : ℕ := 8^2 * 9^3 * 10^4

-- Define a function to count distinct natural-number factors
def count_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem factor_count_of_n : count_factors n = 385 := by sorry

end factor_count_of_n_l184_18434


namespace ramp_cost_is_2950_l184_18423

/-- Calculate the total cost of installing a ramp --/
def total_ramp_cost (permit_cost : ℝ) (contractor_hourly_rate : ℝ) 
  (contractor_days : ℕ) (contractor_hours_per_day : ℕ) 
  (inspector_discount_percent : ℝ) : ℝ :=
  let contractor_total_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hourly_rate * contractor_total_hours
  let inspector_cost := contractor_cost * (1 - inspector_discount_percent)
  permit_cost + contractor_cost + inspector_cost

/-- Theorem stating the total cost of installing a ramp is $2950 --/
theorem ramp_cost_is_2950 : 
  total_ramp_cost 250 150 3 5 0.8 = 2950 := by
  sorry

end ramp_cost_is_2950_l184_18423


namespace red_light_estimation_l184_18490

theorem red_light_estimation (total_surveyed : ℕ) (yes_answers : ℕ) :
  total_surveyed = 600 →
  yes_answers = 180 →
  let prob_odd_id := (1 : ℚ) / 2
  let prob_yes := (yes_answers : ℚ) / total_surveyed
  let prob_red_light := 2 * prob_yes - prob_odd_id
  ⌊total_surveyed * prob_red_light⌋ = 60 := by
  sorry

end red_light_estimation_l184_18490


namespace randy_block_difference_l184_18438

/-- Randy's block building problem -/
theorem randy_block_difference :
  ∀ (total_blocks house_blocks tower_blocks : ℕ),
    total_blocks = 90 →
    house_blocks = 89 →
    tower_blocks = 63 →
    house_blocks - tower_blocks = 26 :=
by
  sorry

end randy_block_difference_l184_18438


namespace arccos_negative_half_l184_18464

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end arccos_negative_half_l184_18464


namespace smallest_single_discount_l184_18473

theorem smallest_single_discount (m : ℕ) : m = 29 ↔ 
  (∀ k : ℕ, k < m → 
    ((1 - k / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.08)^3 ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.12)^2)) ∧
  ((1 - m / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - m / 100 : ℝ) < (1 - 0.08)^3 ∧
   (1 - m / 100 : ℝ) < (1 - 0.12)^2) :=
by sorry

end smallest_single_discount_l184_18473


namespace binary_1010_to_decimal_l184_18416

/-- Converts a list of binary digits to its decimal representation. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₂ -/
def binary_1010 : List Bool := [false, true, false, true]

/-- Theorem stating that the decimal representation of 1010₂ is 10 -/
theorem binary_1010_to_decimal :
  binary_to_decimal binary_1010 = 10 := by
  sorry

end binary_1010_to_decimal_l184_18416


namespace right_triangle_area_l184_18470

theorem right_triangle_area (a b : ℝ) (h : Real.sqrt (a - 5) + (b - 4)^2 = 0) :
  let area := (1/2) * a * b
  area = 6 ∨ area = 10 := by
sorry

end right_triangle_area_l184_18470


namespace total_pies_sold_l184_18485

/-- Represents a type of pie --/
inductive PieType
| Shepherds
| ChickenPot
| VegetablePot
| BeefPot

/-- Represents the size of a pie --/
inductive PieSize
| Small
| Large

/-- Represents the number of pieces a pie is cut into --/
def pieceCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 4
  | PieType.Shepherds, PieSize.Large => 8
  | PieType.ChickenPot, PieSize.Small => 5
  | PieType.ChickenPot, PieSize.Large => 10
  | PieType.VegetablePot, PieSize.Small => 6
  | PieType.VegetablePot, PieSize.Large => 12
  | PieType.BeefPot, PieSize.Small => 7
  | PieType.BeefPot, PieSize.Large => 14

/-- Represents the number of customers who ordered each type and size of pie --/
def customerCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 52
  | PieType.Shepherds, PieSize.Large => 76
  | PieType.ChickenPot, PieSize.Small => 80
  | PieType.ChickenPot, PieSize.Large => 130
  | PieType.VegetablePot, PieSize.Small => 42
  | PieType.VegetablePot, PieSize.Large => 96
  | PieType.BeefPot, PieSize.Small => 35
  | PieType.BeefPot, PieSize.Large => 105

/-- Calculates the number of pies sold for a given type and size --/
def piesSold (t : PieType) (s : PieSize) : ℕ :=
  (customerCount t s + pieceCount t s - 1) / pieceCount t s

/-- Theorem: The total number of pies sold is 80 --/
theorem total_pies_sold :
  (piesSold PieType.Shepherds PieSize.Small +
   piesSold PieType.Shepherds PieSize.Large +
   piesSold PieType.ChickenPot PieSize.Small +
   piesSold PieType.ChickenPot PieSize.Large +
   piesSold PieType.VegetablePot PieSize.Small +
   piesSold PieType.VegetablePot PieSize.Large +
   piesSold PieType.BeefPot PieSize.Small +
   piesSold PieType.BeefPot PieSize.Large) = 80 :=
by sorry

end total_pies_sold_l184_18485


namespace fraction_sum_equals_one_l184_18499

theorem fraction_sum_equals_one (x : ℝ) : x / (x + 1) + 1 / (x + 1) = 1 := by
  sorry

end fraction_sum_equals_one_l184_18499


namespace quadratic_inequality_condition_l184_18471

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end quadratic_inequality_condition_l184_18471


namespace alvin_coconut_trees_l184_18407

/-- The number of coconuts each tree yields -/
def coconuts_per_tree : ℕ := 5

/-- The price of each coconut in dollars -/
def price_per_coconut : ℕ := 3

/-- The amount Alvin needs to earn in dollars -/
def target_earnings : ℕ := 90

/-- The number of coconut trees Alvin needs to harvest -/
def trees_to_harvest : ℕ := 6

theorem alvin_coconut_trees :
  trees_to_harvest * coconuts_per_tree * price_per_coconut = target_earnings :=
sorry

end alvin_coconut_trees_l184_18407


namespace problem_statement_l184_18449

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_equation : 2 * (Real.sqrt (log10 a) + Real.sqrt (log10 b)) + log10 (Real.sqrt a) + log10 (Real.sqrt b) = 108)
  (h_int_sqrt_log_a : ∃ m : ℕ, Real.sqrt (log10 a) = m)
  (h_int_sqrt_log_b : ∃ n : ℕ, Real.sqrt (log10 b) = n)
  (h_int_log_sqrt_a : ∃ k : ℕ, log10 (Real.sqrt a) = k)
  (h_int_log_sqrt_b : ∃ l : ℕ, log10 (Real.sqrt b) = l) :
  a * b = 10^116 := by
sorry

end problem_statement_l184_18449


namespace invalid_votes_percentage_l184_18414

theorem invalid_votes_percentage 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_votes = 357000) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end invalid_votes_percentage_l184_18414


namespace final_value_is_990_l184_18497

def loop_calculation (s i : ℕ) : ℕ :=
  if i ≥ 9 then loop_calculation (s * i) (i - 1)
  else s

theorem final_value_is_990 : loop_calculation 1 11 = 990 := by
  sorry

end final_value_is_990_l184_18497


namespace smallest_perfect_cube_divisor_l184_18451

theorem smallest_perfect_cube_divisor
  (p q r : ℕ)
  (hp : Prime p)
  (hq : Prime q)
  (hr : Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h1_not_prime : ¬ Prime 1)
  (n : ℕ)
  (hn : n = p * q^3 * r^6) :
  ∃ (m : ℕ), m^3 = p^3 * q^3 * r^6 ∧
    ∀ (k : ℕ), (k^3 ≥ n) → (k^3 ≥ m^3) :=
by sorry

end smallest_perfect_cube_divisor_l184_18451


namespace least_subtraction_for_divisibility_problem_solution_l184_18456

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (r : ℕ), r < d ∧ (n - r) % d = 0 ∧ ∀ (s : ℕ), s < r → (n - s) % d ≠ 0 := by
  sorry

theorem problem_solution :
  ∃ (r : ℕ), r = 43 ∧ (62575 - r) % 99 = 0 ∧ ∀ (s : ℕ), s < r → (62575 - s) % 99 ≠ 0 := by
  sorry

end least_subtraction_for_divisibility_problem_solution_l184_18456


namespace kale_spring_mowings_l184_18487

/-- The number of times Kale mowed his lawn in the spring -/
def spring_mowings : ℕ := sorry

/-- The number of times Kale mowed his lawn in the summer -/
def summer_mowings : ℕ := 5

/-- The difference between spring and summer mowings -/
def mowing_difference : ℕ := 3

/-- Theorem stating that Kale mowed his lawn 8 times in the spring -/
theorem kale_spring_mowings :
  spring_mowings = 8 ∧
  summer_mowings = 5 ∧
  spring_mowings - summer_mowings = mowing_difference :=
sorry

end kale_spring_mowings_l184_18487


namespace cos_42_cos_78_minus_sin_42_sin_78_l184_18457

theorem cos_42_cos_78_minus_sin_42_sin_78 :
  Real.cos (42 * π / 180) * Real.cos (78 * π / 180) -
  Real.sin (42 * π / 180) * Real.sin (78 * π / 180) = -1/2 := by
  sorry

end cos_42_cos_78_minus_sin_42_sin_78_l184_18457


namespace complex_equation_solution_l184_18484

theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) + 2 * Complex.I * z = (2 : ℂ) - 5 * Complex.I * z ∧ z = (1 / 7 : ℂ) * Complex.I :=
by sorry

end complex_equation_solution_l184_18484


namespace correct_initial_chips_l184_18479

/-- The number of chips Marnie ate initially to see if she likes them -/
def initial_chips : ℕ := 5

/-- The total number of chips in the bag -/
def total_chips : ℕ := 100

/-- The number of chips Marnie eats per day starting from the second day -/
def daily_chips : ℕ := 10

/-- The total number of days it takes Marnie to finish the bag -/
def total_days : ℕ := 10

/-- Theorem stating that the initial number of chips Marnie ate is correct -/
theorem correct_initial_chips :
  2 * initial_chips + (total_days - 1) * daily_chips = total_chips :=
by sorry

end correct_initial_chips_l184_18479


namespace min_tokens_correct_l184_18450

/-- The minimum number of tokens required to fill an n × m grid -/
def min_tokens (n m : ℕ) : ℕ :=
  if n % 2 = 0 ∨ m % 2 = 0 then
    (n + 1) / 2 + (m + 1) / 2
  else
    (n + 1) / 2 + (m + 1) / 2 - 1

/-- A function that determines if a grid can be filled given initial token placement -/
def can_fill_grid (n m : ℕ) (initial_tokens : Finset (ℕ × ℕ)) : Prop :=
  sorry

theorem min_tokens_correct (n m : ℕ) :
  ∀ (k : ℕ), k < min_tokens n m →
    ¬∃ (initial_tokens : Finset (ℕ × ℕ)),
      initial_tokens.card = k ∧
      can_fill_grid n m initial_tokens :=
  sorry

end min_tokens_correct_l184_18450


namespace fifteenth_row_seats_l184_18415

/-- Represents the number of seats in a given row of the sports palace. -/
def seats_in_row (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row of the sports palace has 33 seats. -/
theorem fifteenth_row_seats : seats_in_row 15 = 33 := by
  sorry

end fifteenth_row_seats_l184_18415


namespace relay_race_average_time_l184_18446

/-- Calculates the average time per leg in a two-leg relay race -/
def average_time_per_leg (time_y time_z : ℕ) : ℚ :=
  (time_y + time_z : ℚ) / 2

/-- Theorem: The average time per leg for the given relay race is 42 seconds -/
theorem relay_race_average_time :
  average_time_per_leg 58 26 = 42 := by
  sorry

end relay_race_average_time_l184_18446


namespace difference_of_squares_l184_18426

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end difference_of_squares_l184_18426


namespace min_value_of_function_l184_18413

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 ∧
  ∃ y : ℝ, y > -1 ∧ (y^2 + 7*y + 10) / (y + 1) = 9 :=
by
  sorry

#check min_value_of_function

end min_value_of_function_l184_18413


namespace hard_candy_colouring_amount_l184_18494

/-- Represents the candy store's daily production and food colouring usage --/
structure CandyStore where
  lollipop_colouring : ℕ  -- ml of food colouring per lollipop
  lollipops_made : ℕ      -- number of lollipops made
  hard_candies_made : ℕ   -- number of hard candies made
  total_colouring : ℕ     -- total ml of food colouring used

/-- Calculates the amount of food colouring needed for each hard candy --/
def hard_candy_colouring (store : CandyStore) : ℕ :=
  (store.total_colouring - store.lollipop_colouring * store.lollipops_made) / store.hard_candies_made

/-- Theorem stating the amount of food colouring needed for each hard candy --/
theorem hard_candy_colouring_amount (store : CandyStore)
  (h1 : store.lollipop_colouring = 5)
  (h2 : store.lollipops_made = 100)
  (h3 : store.hard_candies_made = 5)
  (h4 : store.total_colouring = 600) :
  hard_candy_colouring store = 20 := by
  sorry

end hard_candy_colouring_amount_l184_18494


namespace factorial_calculation_l184_18421

theorem factorial_calculation : (Nat.factorial 15) / ((Nat.factorial 7) * (Nat.factorial 8)) * 2 = 1286 := by
  sorry

end factorial_calculation_l184_18421


namespace water_evaporation_rate_l184_18463

/-- Given a bowl with water that experiences evaporation over time, 
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.03)
  : (initial_water * evaporation_percentage) / evaporation_period = 0.06 := by
  sorry


end water_evaporation_rate_l184_18463
