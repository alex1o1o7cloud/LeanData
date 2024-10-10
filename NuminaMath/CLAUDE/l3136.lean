import Mathlib

namespace rectangle_increase_l3136_313635

/-- Proves that for a rectangle with length increased by 10% and area increased by 37.5%,
    the breadth must be increased by 25% -/
theorem rectangle_increase (L B : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) : 
  ∃ p : ℝ, 
    (1.1 * L) * (B * (1 + p / 100)) = 1.375 * (L * B) ∧ 
    p = 25 := by
  sorry

end rectangle_increase_l3136_313635


namespace john_biking_distance_l3136_313675

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 [2, 5, 6, 3] = 1360 := by
  sorry

end john_biking_distance_l3136_313675


namespace not_perfect_square_infinitely_often_l3136_313684

theorem not_perfect_square_infinitely_often (a b : ℕ+) (h : ∃ p : ℕ, Nat.Prime p ∧ b = a + p) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ¬∃ k : ℕ, (a^n + a + 1) * (b^n + b + 1) = k^2 := by
  sorry

end not_perfect_square_infinitely_often_l3136_313684


namespace rectangle_area_l3136_313680

-- Define the rectangle
def Rectangle (length width : ℝ) : Prop :=
  width = (2/3) * length ∧ 2 * (length + width) = 148

-- State the theorem
theorem rectangle_area (l w : ℝ) (h : Rectangle l w) : l * w = 1314.24 := by
  sorry

end rectangle_area_l3136_313680


namespace x_is_twenty_percent_greater_than_80_l3136_313663

/-- If x is 20 percent greater than 80, then x equals 96. -/
theorem x_is_twenty_percent_greater_than_80 : ∀ x : ℝ, x = 80 * (1 + 20 / 100) → x = 96 :=
by
  sorry

end x_is_twenty_percent_greater_than_80_l3136_313663


namespace f_increasing_condition_l3136_313622

/-- The quadratic function f(x) = 3x^2 - ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - a * x + 4

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 6 * x - a

/-- The theorem stating the condition for f(x) to be increasing on [-5, +∞) -/
theorem f_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ -5 → (f_deriv a x ≥ 0)) ↔ a ≤ -30 := by sorry

end f_increasing_condition_l3136_313622


namespace probability_sum_greater_than_six_l3136_313668

/-- Box A contains ping-pong balls numbered 1 and 2 -/
def box_A : Finset ℕ := {1, 2}

/-- Box B contains ping-pong balls numbered 3, 4, 5, and 6 -/
def box_B : Finset ℕ := {3, 4, 5, 6}

/-- The set of all possible outcomes when drawing one ball from each box -/
def all_outcomes : Finset (ℕ × ℕ) :=
  box_A.product box_B

/-- The set of favorable outcomes (sum greater than 6) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 > 6)

/-- The probability of drawing balls with sum greater than 6 -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem probability_sum_greater_than_six : probability = 3/8 := by
  sorry

end probability_sum_greater_than_six_l3136_313668


namespace beach_population_evening_l3136_313692

/-- The number of people at the beach in the evening -/
def beach_population (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the total number of people at the beach in the evening -/
theorem beach_population_evening :
  beach_population 3 100 40 = 63 := by
  sorry

end beach_population_evening_l3136_313692


namespace computer_price_increase_l3136_313618

theorem computer_price_increase (original_price : ℝ) : 
  original_price + 0.2 * original_price = 351 → 
  2 * original_price = 585 := by
  sorry

end computer_price_increase_l3136_313618


namespace fishing_ratio_l3136_313658

/-- Given that Jordan caught 4 fish and after losing one-fourth of their total catch, 
    9 fish remain, prove that the ratio of Perry's catch to Jordan's catch is 2:1 -/
theorem fishing_ratio : 
  let jordan_catch : ℕ := 4
  let remaining_fish : ℕ := 9
  let total_catch : ℕ := remaining_fish * 4 / 3
  let perry_catch : ℕ := total_catch - jordan_catch
  (perry_catch : ℚ) / jordan_catch = 2 / 1 := by
sorry


end fishing_ratio_l3136_313658


namespace inequality_system_solution_l3136_313652

theorem inequality_system_solution :
  let S : Set ℝ := {x | (x - 1) / 2 + 2 > x ∧ 2 * (x - 2) ≤ 3 * x - 5}
  S = {x | 1 ≤ x ∧ x < 3} := by
  sorry

end inequality_system_solution_l3136_313652


namespace min_value_sin_function_l3136_313621

theorem min_value_sin_function (x : Real) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  (2 * Real.sin x ^ 2 + 1) / Real.sin (2 * x) ≥ Real.sqrt 3 := by
  sorry

end min_value_sin_function_l3136_313621


namespace infinitely_many_circled_l3136_313666

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Predicate that checks if a number in the sequence is circled -/
def IsCircled (a : Sequence) (n : ℕ) : Prop := a n ≥ n

/-- The main theorem stating that infinitely many numbers are circled -/
theorem infinitely_many_circled (a : Sequence) : 
  Set.Infinite {n : ℕ | IsCircled a n} := by
  sorry

end infinitely_many_circled_l3136_313666


namespace gcf_of_lcm_sum_and_difference_l3136_313665

theorem gcf_of_lcm_sum_and_difference : Nat.gcd (Nat.lcm 9 15 + 5) (Nat.lcm 10 21 - 7) = 1 := by
  sorry

end gcf_of_lcm_sum_and_difference_l3136_313665


namespace cube_intersection_probability_l3136_313625

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  vertices : Finset (Fin 8)
  faces : Finset (Finset (Fin 4))
  vertex_count : vertices.card = 8
  face_count : faces.card = 6

/-- A function that determines if three vertices form a plane intersecting the cube's interior -/
def plane_intersects_interior (c : Cube) (v1 v2 v3 : Fin 8) : Prop :=
  sorry

/-- The probability of three randomly chosen distinct vertices forming a plane
    that intersects the interior of the cube -/
def intersection_probability (c : Cube) : ℚ :=
  sorry

theorem cube_intersection_probability (c : Cube) :
  intersection_probability c = 4/7 := by
  sorry

end cube_intersection_probability_l3136_313625


namespace parabola_through_origin_l3136_313696

/-- A parabola passing through the origin can be represented by the equation y = ax^2 + bx,
    where a and b are real numbers and at least one of them is non-zero. -/
theorem parabola_through_origin :
  ∀ (f : ℝ → ℝ), (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0) →
  (f 0 = 0) →
  (∀ x, ∃ y, f x = y ∧ y = a * x^2 + b * x) →
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ x, f x = a * x^2 + b * x) :=
by sorry

end parabola_through_origin_l3136_313696


namespace min_value_theorem_l3136_313673

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_solution : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ Set.Ioo x₁ x₂) :
  ∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y → y ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end min_value_theorem_l3136_313673


namespace quadratic_roots_theorem_l3136_313606

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - 4*x - 2*k + 8

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0

-- Define the additional condition
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^3 * x₂ + x₁ * x₂^3 = 24

-- Theorem statement
theorem quadratic_roots_theorem :
  ∀ k : ℝ, has_two_real_roots k →
  (∃ x₁ x₂ : ℝ, quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 ∧ roots_condition x₁ x₂) →
  k = 3 :=
sorry

end quadratic_roots_theorem_l3136_313606


namespace independence_test_not_always_correct_l3136_313653

-- Define the independence test
def independence_test (sample : Type) : Prop := True

-- Define the principle of small probability
def principle_of_small_probability : Prop := True

-- Define the concept of different samples
def different_samples (s1 s2 : Type) : Prop := s1 ≠ s2

-- Define the concept of different conclusions
def different_conclusions (c1 c2 : Prop) : Prop := c1 ≠ c2

-- Define other methods for determining categorical variable relationships
def other_methods_exist : Prop := True

-- Theorem statement
theorem independence_test_not_always_correct :
  (∀ (s : Type), independence_test s → principle_of_small_probability) →
  (∀ (s1 s2 : Type), different_samples s1 s2 → 
    ∃ (c1 c2 : Prop), different_conclusions c1 c2) →
  other_methods_exist →
  ¬(∀ (s : Type), independence_test s → 
    ∀ (conclusion : Prop), conclusion) :=
by sorry

end independence_test_not_always_correct_l3136_313653


namespace three_thousand_six_hundred_factorization_l3136_313677

theorem three_thousand_six_hundred_factorization (a b c d : ℕ+) 
  (h1 : 3600 = 2^(a.val) * 3^(b.val) * 4^(c.val) * 5^(d.val))
  (h2 : a.val + b.val + c.val + d.val = 7) : c.val = 1 := by
  sorry

end three_thousand_six_hundred_factorization_l3136_313677


namespace complex_fraction_equality_l3136_313601

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (7 - Complex.I) / (3 + Complex.I) = 2 - Complex.I := by
  sorry

end complex_fraction_equality_l3136_313601


namespace maxRegions_correct_maxRegions_is_maximal_l3136_313678

/-- The maximal number of regions a circle can be divided into by segments joining n points on its boundary -/
def maxRegions (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose n 2 + 1

/-- Theorem stating that maxRegions gives the correct number of regions -/
theorem maxRegions_correct (n : ℕ) : 
  maxRegions n = Nat.choose n 4 + Nat.choose n 2 + 1 := by
  sorry

/-- Theorem stating that maxRegions indeed gives the maximal number of regions -/
theorem maxRegions_is_maximal (n : ℕ) :
  ∀ k : ℕ, k ≤ maxRegions n := by
  sorry

end maxRegions_correct_maxRegions_is_maximal_l3136_313678


namespace complex_fraction_modulus_l3136_313682

theorem complex_fraction_modulus (a b : ℝ) (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i) / (a + b*i) = 2 - i → Complex.abs (a - b*i) = 1 := by
  sorry

end complex_fraction_modulus_l3136_313682


namespace sum_of_squares_minus_linear_l3136_313676

theorem sum_of_squares_minus_linear : ∀ x y : ℝ, 
  x ≠ y → 
  x^2 - 2000*x = y^2 - 2000*y → 
  x + y = 2000 := by
sorry

end sum_of_squares_minus_linear_l3136_313676


namespace triangle_inequality_cube_root_l3136_313636

theorem triangle_inequality_cube_root (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 := by
  sorry

end triangle_inequality_cube_root_l3136_313636


namespace inequality_proof_l3136_313629

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end inequality_proof_l3136_313629


namespace unique_equidistant_point_l3136_313632

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  sideLength : ℝ

/-- Checks if a point is inside or on the diagonal face BDD₁B₁ of the cube -/
def isOnDiagonalFace (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the distance from a point to a plane -/
def distToPlane (p : Point3D) (plane : Point3D → Prop) : ℝ := sorry

/-- The plane ABC of the cube -/
def planeABC (c : Cube) : Point3D → Prop := sorry

/-- The plane ABA₁ of the cube -/
def planeABA1 (c : Cube) : Point3D → Prop := sorry

/-- The plane ADA₁ of the cube -/
def planeADA1 (c : Cube) : Point3D → Prop := sorry

theorem unique_equidistant_point (c : Cube) : 
  ∃! p : Point3D, 
    isOnDiagonalFace c p ∧ 
    distToPlane p (planeABC c) = distToPlane p (planeABA1 c) ∧
    distToPlane p (planeABC c) = distToPlane p (planeADA1 c) :=
sorry

end unique_equidistant_point_l3136_313632


namespace binomial_n_minus_two_l3136_313664

theorem binomial_n_minus_two (n : ℕ) (h : n ≥ 2) : 
  (n.choose (n - 2)) = n * (n - 1) / 2 :=
by sorry

end binomial_n_minus_two_l3136_313664


namespace inverse_of_3_mod_1013_l3136_313647

theorem inverse_of_3_mod_1013 : ∃ x : ℕ, 0 ≤ x ∧ x < 1013 ∧ (3 * x) % 1013 = 1 :=
by
  use 338
  sorry

end inverse_of_3_mod_1013_l3136_313647


namespace port_vessel_ratio_l3136_313613

theorem port_vessel_ratio :
  ∀ (cargo sailboats fishing : ℕ),
    cargo + 4 + sailboats + fishing = 28 →
    sailboats = cargo + 6 →
    sailboats = 7 * fishing →
    cargo = 2 * 4 :=
by sorry

end port_vessel_ratio_l3136_313613


namespace factorization_of_2m_squared_minus_18_l3136_313611

theorem factorization_of_2m_squared_minus_18 (m : ℝ) : 2 * m^2 - 18 = 2 * (m + 3) * (m - 3) := by
  sorry

end factorization_of_2m_squared_minus_18_l3136_313611


namespace expression_simplification_l3136_313697

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 2) :
  a^2 / (a^2 + 2*a) - (a^2 - 2*a + 1) / (a + 2) / ((a^2 - 1) / (a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3136_313697


namespace coupon_savings_difference_l3136_313654

/-- Represents the savings from Coupon A (20% off total price) -/
def savingsA (price : ℝ) : ℝ := 0.2 * price

/-- Represents the savings from Coupon B (flat $40 discount) -/
def savingsB : ℝ := 40

/-- Represents the savings from Coupon C (30% off amount exceeding $120) -/
def savingsC (price : ℝ) : ℝ := 0.3 * (price - 120)

/-- Theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference (minPrice maxPrice : ℝ) : 
  (minPrice > 120) →
  (maxPrice > 120) →
  (∀ p : ℝ, minPrice ≤ p → p ≤ maxPrice → 
    savingsA p ≥ max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p > maxPrice → 
    savingsA p < max (savingsB) (savingsC p)) →
  (∃ p : ℝ, p < minPrice → p > 120 → 
    savingsA p < max (savingsB) (savingsC p)) →
  maxPrice - minPrice = 160 := by
sorry

end coupon_savings_difference_l3136_313654


namespace product_of_real_parts_l3136_313688

theorem product_of_real_parts (x : ℂ) : 
  x^2 - 6*x = -8 + 2*I → 
  ∃ (x₁ x₂ : ℂ), x₁ ≠ x₂ ∧ 
    x₁^2 - 6*x₁ = -8 + 2*I ∧ 
    x₂^2 - 6*x₂ = -8 + 2*I ∧
    (x₁.re * x₂.re = 9 - Real.sqrt 5 / 2) :=
by sorry

end product_of_real_parts_l3136_313688


namespace estimate_pi_random_simulation_l3136_313650

/-- Estimate pi using a random simulation method with a square paper and inscribed circle -/
theorem estimate_pi_random_simulation (total_seeds : ℕ) (seeds_in_circle : ℕ) :
  total_seeds = 1000 →
  seeds_in_circle = 778 →
  ∃ (pi_estimate : ℝ), pi_estimate = 4 * (seeds_in_circle : ℝ) / (total_seeds : ℝ) ∧ 
                        abs (pi_estimate - 3.112) < 0.001 :=
by
  sorry

end estimate_pi_random_simulation_l3136_313650


namespace park_trees_count_l3136_313610

theorem park_trees_count : ∃! n : ℕ, 
  80 < n ∧ n < 150 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 4 ∧ 
  n = 98 := by sorry

end park_trees_count_l3136_313610


namespace conic_section_union_l3136_313616

/-- The equation y^4 - 6x^4 = 3y^2 - 2 represents the union of a hyperbola and an ellipse -/
theorem conic_section_union (x y : ℝ) : 
  (y^4 - 6*x^4 = 3*y^2 - 2) ↔ 
  ((y^2 - 3*x^2 = 2 ∨ y^2 - 2*x^2 = 1) ∨ (y^2 + 3*x^2 = 2 ∨ y^2 + 2*x^2 = 1)) :=
sorry

end conic_section_union_l3136_313616


namespace divide_powers_of_nineteen_l3136_313656

theorem divide_powers_of_nineteen : 19^12 / 19^8 = 130321 := by sorry

end divide_powers_of_nineteen_l3136_313656


namespace chocolate_bar_cost_l3136_313679

/-- The number of chocolate bars in the box initially -/
def initial_bars : ℕ := 13

/-- The number of bars Rachel didn't sell -/
def unsold_bars : ℕ := 4

/-- The total amount Rachel made from selling the bars -/
def total_revenue : ℚ := 18

/-- The cost of each chocolate bar -/
def bar_cost : ℚ := 2

theorem chocolate_bar_cost :
  (initial_bars - unsold_bars) * bar_cost = total_revenue :=
sorry

end chocolate_bar_cost_l3136_313679


namespace square_field_area_l3136_313600

theorem square_field_area (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end square_field_area_l3136_313600


namespace beaded_corset_cost_l3136_313644

/-- The number of rows of purple beads -/
def purple_rows : ℕ := 50

/-- The number of beads per row of purple beads -/
def purple_beads_per_row : ℕ := 20

/-- The number of rows of blue beads -/
def blue_rows : ℕ := 40

/-- The number of beads per row of blue beads -/
def blue_beads_per_row : ℕ := 18

/-- The number of gold beads -/
def gold_beads : ℕ := 80

/-- The cost of beads in dollars per 10 beads -/
def cost_per_10_beads : ℚ := 1

/-- The total cost of all beads in dollars -/
def total_cost : ℚ := 180

theorem beaded_corset_cost :
  (purple_rows * purple_beads_per_row + blue_rows * blue_beads_per_row + gold_beads) / 10 * cost_per_10_beads = total_cost := by
  sorry

end beaded_corset_cost_l3136_313644


namespace unique_solution_system_l3136_313628

theorem unique_solution_system (x y : ℝ) : 
  (2 * x + y = 3 ∧ x - y = 3) ↔ (x = 2 ∧ y = -1) :=
by sorry

end unique_solution_system_l3136_313628


namespace shekars_social_studies_score_l3136_313617

/-- Given Shekar's scores in four subjects and his average marks, prove his score in social studies -/
theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 75)
  (h5 : average_score = 73)
  (h6 : average_score = (math_score + science_score + english_score + biology_score + social_studies_score) / 5) :
  social_studies_score = 82 := by
  sorry

end shekars_social_studies_score_l3136_313617


namespace circle_condition_l3136_313669

theorem circle_condition (m : ℝ) : 
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  (m < 1/4 ∨ m > 1) :=
sorry

end circle_condition_l3136_313669


namespace baseball_cards_pages_l3136_313667

def organize_baseball_cards (new_cards : ℕ) (old_cards : ℕ) (cards_per_page : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

theorem baseball_cards_pages :
  organize_baseball_cards 8 10 3 = 6 := by
  sorry

end baseball_cards_pages_l3136_313667


namespace gcd_condition_iff_prime_representation_l3136_313605

theorem gcd_condition_iff_prime_representation (x y : ℕ) : 
  (∀ n : ℕ, Nat.gcd (n * (Nat.factorial x - x * y - x - y + 2) + 2) 
                    (n * (Nat.factorial x - x * y - x - y + 3) + 3) > 1) ↔
  (∃ q : ℕ, Prime q ∧ q > 3 ∧ x = q - 1 ∧ y = (Nat.factorial (q - 1) - (q - 1)) / q) :=
by sorry

end gcd_condition_iff_prime_representation_l3136_313605


namespace expression_value_l3136_313657

theorem expression_value (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end expression_value_l3136_313657


namespace tampa_bay_bucs_problem_l3136_313603

/-- The Tampa Bay Bucs team composition problem -/
theorem tampa_bay_bucs_problem 
  (initial_football_players : ℕ)
  (initial_cheerleaders : ℕ)
  (quitting_football_players : ℕ)
  (quitting_cheerleaders : ℕ)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : quitting_football_players = 10)
  (h4 : quitting_cheerleaders = 4) :
  (initial_football_players - quitting_football_players) + 
  (initial_cheerleaders - quitting_cheerleaders) = 15 := by
  sorry

end tampa_bay_bucs_problem_l3136_313603


namespace volume_ratio_l3136_313687

/-- The domain S bounded by two curves -/
structure Domain (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The volume formed by revolving the domain around the x-axis -/
noncomputable def volume_x (d : Domain a b) : ℝ := sorry

/-- The volume formed by revolving the domain around the y-axis -/
noncomputable def volume_y (d : Domain a b) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (a b : ℝ) (d : Domain a b) :
  (volume_x d) / (volume_y d) = 14 / 5 := by sorry

end volume_ratio_l3136_313687


namespace A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l3136_313615

/-- Represents a hexagonal grid game. -/
structure HexGame where
  k : ℕ
  -- Add other necessary components of the game state

/-- Defines a valid move for Player A. -/
def valid_move_A (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player A
  sorry

/-- Defines a valid move for Player B. -/
def valid_move_B (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player B
  sorry

/-- Defines the winning condition for Player A. -/
def A_wins (game : HexGame) : Prop :=
  -- Define the condition when Player A wins
  sorry

/-- States that Player A can win in a finite number of moves for k = 5. -/
theorem A_can_win_with_5 : 
  ∃ (game : HexGame), game.k = 5 ∧ A_wins game :=
sorry

/-- States that Player A cannot win in a finite number of moves for k ≥ 6. -/
theorem A_cannot_win_with_6_or_more :
  ∀ (game : HexGame), game.k ≥ 6 → ¬(A_wins game) :=
sorry

/-- The main theorem stating that 6 is the minimum value of k for which
    Player A cannot win in a finite number of moves. -/
theorem min_k_for_A_cannot_win : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (game : HexGame), game.k ≥ k → ¬(A_wins game)) ∧
  (∀ (k' : ℕ), k' < k → ∃ (game : HexGame), game.k = k' ∧ A_wins game) :=
sorry

end A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l3136_313615


namespace percentage_enrolled_in_biology_l3136_313637

theorem percentage_enrolled_in_biology (total_students : ℕ) (not_enrolled : ℕ) 
  (h1 : total_students = 880) (h2 : not_enrolled = 638) :
  (((total_students - not_enrolled) : ℚ) / total_students) * 100 = 27.5 := by
  sorry

end percentage_enrolled_in_biology_l3136_313637


namespace a_m_prime_factors_l3136_313660

def a_m (m : ℕ) : ℕ := (2^(2*m+1))^2 + 1

def has_at_most_two_prime_factors (n : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ ∀ (r : ℕ), Nat.Prime r → r ∣ n → r = p ∨ r = q

theorem a_m_prime_factors (m : ℕ) :
  has_at_most_two_prime_factors (a_m m) ↔ m = 0 ∨ m = 1 ∨ m = 2 := by sorry

end a_m_prime_factors_l3136_313660


namespace total_numbers_l3136_313602

theorem total_numbers (average : ℝ) (first_six_average : ℝ) (last_eight_average : ℝ) (eighth_number : ℝ)
  (h1 : average = 60)
  (h2 : first_six_average = 57)
  (h3 : last_eight_average = 61)
  (h4 : eighth_number = 50) :
  ∃ n : ℕ, n = 13 ∧ 
    average * n = first_six_average * 6 + last_eight_average * 8 - eighth_number :=
by
  sorry


end total_numbers_l3136_313602


namespace smallest_integer_with_remainders_l3136_313686

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 6 = 5) ∧ 
  (n % 8 = 7) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 6 = 5 ∧ m % 8 = 7 → m ≥ n) ∧
  n = 23 := by
  sorry

end smallest_integer_with_remainders_l3136_313686


namespace exponent_division_l3136_313646

theorem exponent_division (a : ℝ) : a^10 / a^9 = a := by
  sorry

end exponent_division_l3136_313646


namespace sum_in_M_l3136_313659

/-- Define the set Mα for a positive real number α -/
def M (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x2 > x1 → 
    -α * (x2 - x1) < f x2 - f x1 ∧ f x2 - f x1 < α * (x2 - x1)

/-- Theorem: If f ∈ Mα1 and g ∈ Mα2, then f + g ∈ Mα1+α2 -/
theorem sum_in_M (α1 α2 : ℝ) (f g : ℝ → ℝ) 
    (hα1 : α1 > 0) (hα2 : α2 > 0) 
    (hf : M α1 f) (hg : M α2 g) : 
  M (α1 + α2) (f + g) := by
  sorry

end sum_in_M_l3136_313659


namespace non_officers_count_l3136_313645

/-- Prove the number of non-officers in an office given salary information -/
theorem non_officers_count (avg_salary : ℝ) (officer_salary : ℝ) (non_officer_salary : ℝ) 
  (officer_count : ℕ) (h1 : avg_salary = 120) (h2 : officer_salary = 430) 
  (h3 : non_officer_salary = 110) (h4 : officer_count = 15) : 
  ∃ (non_officer_count : ℕ), 
    avg_salary * (officer_count + non_officer_count) = 
    officer_salary * officer_count + non_officer_salary * non_officer_count ∧ 
    non_officer_count = 465 := by
  sorry

end non_officers_count_l3136_313645


namespace milk_water_mixture_l3136_313699

theorem milk_water_mixture (total_weight : ℝ) (added_water : ℝ) (new_ratio : ℝ) :
  total_weight = 85 →
  added_water = 5 →
  new_ratio = 3 →
  let initial_water := (total_weight - new_ratio * added_water) / (new_ratio + 1)
  let initial_milk := total_weight - initial_water
  (initial_milk / initial_water) = 27 / 7 := by
  sorry

end milk_water_mixture_l3136_313699


namespace solve_for_b_l3136_313633

/-- Given two functions p and q, prove that if p(q(3)) = 31, then b has two specific values. -/
theorem solve_for_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x^2 - 7)
  (hq : ∀ x, q x = 4 * x - b)
  (h_pq3 : p (q 3) = 31) :
  b = 12 + Real.sqrt 19 ∨ b = 12 - Real.sqrt 19 := by
  sorry

#check solve_for_b

end solve_for_b_l3136_313633


namespace f_increasing_condition_and_range_f_range_on_interval_l3136_313604

/-- The function f(x) = x^2 - 4x -/
def f (x : ℝ) : ℝ := x^2 - 4*x

theorem f_increasing_condition_and_range (a : ℝ) :
  (∀ x ≥ 2*a - 1, MonotoneOn f (Set.Ici (2*a - 1))) → a ≥ 3/2 :=
sorry

theorem f_range_on_interval :
  Set.image f (Set.Icc 1 7) = Set.Icc (-4) 21 :=
sorry

end f_increasing_condition_and_range_f_range_on_interval_l3136_313604


namespace min_side_length_l3136_313609

/-- An isosceles triangle with a perpendicular line from vertex to base -/
structure IsoscelesTriangleWithPerp where
  -- The length of two equal sides
  side : ℝ
  -- The length of CD
  cd : ℕ
  -- Assertion that BD^2 = 77
  h_bd_sq : side^2 - cd^2 = 77

/-- The theorem stating the minimal possible integer value for AC -/
theorem min_side_length (t : IsoscelesTriangleWithPerp) : 
  ∃ (min : ℕ), (∀ (t' : IsoscelesTriangleWithPerp), (t'.side : ℝ) ≥ min) ∧ min = 12 := by
  sorry

end min_side_length_l3136_313609


namespace range_of_a_l3136_313670

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a-1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → a ≥ 1 := by
  sorry

end range_of_a_l3136_313670


namespace intersection_and_coefficients_l3136_313624

def A : Set ℝ := {x | x^2 < 9}
def B : Set ℝ := {x | (x-2)*(x+4) < 0}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -3 < x ∧ x < 2}) ∧
  (∃ a b : ℝ, ∀ x : ℝ, (x ∈ A ∪ B) ↔ (2*x^2 + a*x + b < 0) ∧ a = 2 ∧ b = -24) :=
by sorry

end intersection_and_coefficients_l3136_313624


namespace physics_marks_correct_l3136_313612

/-- Given a student's marks in four subjects and their average across five subjects,
    calculate the marks in the fifth subject. -/
def calculate_physics_marks (e m c b : ℕ) (avg : ℚ) (n : ℕ) : ℚ :=
  n * avg - (e + m + c + b)

/-- Theorem stating that the calculated physics marks are correct given the problem conditions. -/
theorem physics_marks_correct 
  (e m c b : ℕ) 
  (avg : ℚ) 
  (n : ℕ) 
  (h1 : e = 70) 
  (h2 : m = 60) 
  (h3 : c = 60) 
  (h4 : b = 65) 
  (h5 : avg = 66.6) 
  (h6 : n = 5) : 
  calculate_physics_marks e m c b avg n = 78 := by
sorry

#eval calculate_physics_marks 70 60 60 65 66.6 5

end physics_marks_correct_l3136_313612


namespace second_week_rainfall_l3136_313691

/-- Proves that the rainfall during the second week of January was 15 inches,
    given the total rainfall and the relationship between the two weeks. -/
theorem second_week_rainfall (total_rainfall : ℝ) (first_week : ℝ) (second_week : ℝ) : 
  total_rainfall = 25 →
  second_week = 1.5 * first_week →
  total_rainfall = first_week + second_week →
  second_week = 15 := by
  sorry


end second_week_rainfall_l3136_313691


namespace password_generation_l3136_313661

def polynomial (x y : ℤ) : ℤ := 32 * x^3 - 8 * x * y^2

def factor1 (x : ℤ) : ℤ := 8 * x
def factor2 (x y : ℤ) : ℤ := 2 * x + y
def factor3 (x y : ℤ) : ℤ := 2 * x - y

def concatenate (a b c : ℤ) : ℤ := a * 100000 + b * 1000 + c

theorem password_generation (x y : ℤ) (h1 : x = 10) (h2 : y = 10) :
  concatenate (factor1 x) (factor2 x y) (factor3 x y) = 803010 :=
by sorry

end password_generation_l3136_313661


namespace average_grade_year_before_l3136_313620

/-- Calculates the average grade for the year before last given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 86 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 86 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 69.2 :=
by
  sorry

#check average_grade_year_before

end average_grade_year_before_l3136_313620


namespace distinct_roots_rectangle_perimeter_l3136_313698

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 1)*x + 4*k - 3

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := (2*k + 1)^2 - 4*(4*k - 3)

-- Statement 1: The equation always has two distinct real roots
theorem distinct_roots (k : ℝ) : discriminant k > 0 := by sorry

-- Define the sum and product of roots
def sum_of_roots (k : ℝ) : ℝ := 2*k + 1
def product_of_roots (k : ℝ) : ℝ := 4*k - 3

-- Statement 2: When roots represent rectangle sides with diagonal √31, perimeter is 14
theorem rectangle_perimeter (k : ℝ) 
  (h1 : sum_of_roots k^2 + product_of_roots k = 31) 
  (h2 : k > 0) : 
  2 * sum_of_roots k = 14 := by sorry

end distinct_roots_rectangle_perimeter_l3136_313698


namespace problem_1_problem_2_l3136_313639

-- Problem 1
theorem problem_1 : 
  (2 + 1/4)^(1/2) - (-0.96)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) = 1/2 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : 
  (2 * (a^2)^(1/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4*a := by sorry

end problem_1_problem_2_l3136_313639


namespace largest_expression_l3136_313631

theorem largest_expression : 
  let a := (1 : ℚ) / 2
  let b := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let c := (1 : ℚ) / 4 + (1 : ℚ) / 5 + (1 : ℚ) / 6
  let d := (1 : ℚ) / 5 + (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8
  let e := (1 : ℚ) / 6 + (1 : ℚ) / 7 + (1 : ℚ) / 8 + (1 : ℚ) / 9 + (1 : ℚ) / 10
  e > a ∧ e > b ∧ e > c ∧ e > d := by
  sorry

end largest_expression_l3136_313631


namespace parking_fee_range_l3136_313634

/-- Represents the parking fee function --/
def parking_fee (x : ℝ) : ℝ := -5 * x + 12000

/-- Theorem: The parking fee range is [6900, 8100] given the problem conditions --/
theorem parking_fee_range :
  ∀ x : ℝ,
  0 ≤ x ∧ x ≤ 1200 ∧
  1200 * 0.65 ≤ x ∧ x ≤ 1200 * 0.85 →
  6900 ≤ parking_fee x ∧ parking_fee x ≤ 8100 :=
by sorry

end parking_fee_range_l3136_313634


namespace complex_magnitude_l3136_313643

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l3136_313643


namespace complex_equation_sum_l3136_313642

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 3 * i) / i = b - 2 * i → a + b = 5 := by
  sorry

end complex_equation_sum_l3136_313642


namespace root_ratio_implies_k_value_l3136_313630

theorem root_ratio_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + k = 0 ∧ s^2 - 4*s + k = 0) →
  k = 3 := by
sorry

end root_ratio_implies_k_value_l3136_313630


namespace equation1_solutions_equation2_solution_l3136_313671

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 1)^2 - 1 = 15
def equation2 (x : ℝ) : Prop := (1/3) * (x + 3)^3 - 9 = 0

-- Theorem for equation 1
theorem equation1_solutions : 
  (∃ x : ℝ, equation1 x) ↔ (equation1 5 ∧ equation1 (-3)) :=
sorry

-- Theorem for equation 2
theorem equation2_solution : 
  (∃ x : ℝ, equation2 x) ↔ equation2 0 :=
sorry

end equation1_solutions_equation2_solution_l3136_313671


namespace brownie_division_l3136_313623

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 30

-- Define the dimensions of each brownie piece
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- Define the number of pieces
def num_pieces : ℕ := 60

-- Theorem statement
theorem brownie_division :
  pan_length * pan_width = num_pieces * piece_length * piece_width :=
by sorry

end brownie_division_l3136_313623


namespace number_of_boys_who_love_marbles_l3136_313683

def total_marbles : ℕ := 35
def marbles_per_boy : ℕ := 7

theorem number_of_boys_who_love_marbles : 
  total_marbles / marbles_per_boy = 5 := by
  sorry

end number_of_boys_who_love_marbles_l3136_313683


namespace toms_average_increase_l3136_313690

/-- Calculates the increase in average score given four exam scores -/
def increase_in_average (score1 score2 score3 score4 : ℚ) : ℚ :=
  let initial_average := (score1 + score2 + score3) / 3
  let new_average := (score1 + score2 + score3 + score4) / 4
  new_average - initial_average

/-- Theorem: The increase in Tom's average score is 3.25 -/
theorem toms_average_increase :
  increase_in_average 72 78 81 90 = 13/4 := by
  sorry

end toms_average_increase_l3136_313690


namespace digit_count_700_l3136_313607

def count_digit (d : Nat) (n : Nat) : Nat :=
  (n / 100 + 1) * 10

theorem digit_count_700 : 
  (count_digit 9 700 + count_digit 8 700) = 280 := by
  sorry

end digit_count_700_l3136_313607


namespace solution_value_l3136_313649

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c^2

/-- Theorem stating that 5/19 is the value of a that satisfies the equation -/
theorem solution_value : ∃ (a : ℝ), F a 3 2 = F a 2 3 ∧ a = 5/19 := by
  sorry

end solution_value_l3136_313649


namespace complex_magnitude_problem_l3136_313608

theorem complex_magnitude_problem (z : ℂ) (h : z / (2 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end complex_magnitude_problem_l3136_313608


namespace unique_three_digit_cube_sum_l3136_313640

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem unique_three_digit_cube_sum : ∃! n : ℕ, 
  is_three_digit_number n ∧ n = (digit_sum n)^3 :=
by
  sorry

end unique_three_digit_cube_sum_l3136_313640


namespace product_simplification_l3136_313681

theorem product_simplification (y : ℝ) (h : y ≠ 0) :
  (21 * y^3) * (9 * y^2) * (1 / (7*y)^2) = 27/7 * y^3 := by
  sorry

end product_simplification_l3136_313681


namespace binary_of_34_l3136_313674

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 34 is 100010 -/
theorem binary_of_34 : toBinary 34 = [false, true, false, false, false, true] := by
  sorry

end binary_of_34_l3136_313674


namespace allison_wins_prob_l3136_313685

/-- Represents a 6-sided cube with specified face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube configuration -/
def allison_cube : Cube :=
  { faces := λ _ => 7 }

/-- Brian's cube configuration -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube configuration -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 2 then 3 else 5 }

/-- The probability of rolling a specific value or less on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's -/
theorem allison_wins_prob : 
  prob_roll_le brian_cube 6 * prob_roll_le noah_cube 6 = 1 := by
  sorry


end allison_wins_prob_l3136_313685


namespace symmetry_of_point_l3136_313655

/-- Given a point P and a line L, this function returns the point symmetric to P with respect to L -/
def symmetricPoint (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x + y = 1 -/
def lineXPlusYEq1 (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

theorem symmetry_of_point :
  symmetricPoint (2, 5) lineXPlusYEq1 = (-4, -1) := by sorry

end symmetry_of_point_l3136_313655


namespace equation_solution_l3136_313672

theorem equation_solution : ∃ x : ℚ, (1 / 4 : ℚ) + 1 / x = 7 / 8 ∧ x = 8 / 5 := by
  sorry

end equation_solution_l3136_313672


namespace angle_sum_from_tan_roots_l3136_313662

theorem angle_sum_from_tan_roots (α β : Real) :
  (∃ x y : Real, x^2 + 6*x + 7 = 0 ∧ y^2 + 6*y + 7 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  α + β = -3*π/4 := by
sorry

end angle_sum_from_tan_roots_l3136_313662


namespace factor_difference_of_squares_l3136_313689

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4*y) * (9 + 4*y) := by
  sorry

end factor_difference_of_squares_l3136_313689


namespace diamond_45_15_l3136_313626

/-- The diamond operation on positive real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ :=
  x / y

/-- Axioms for the diamond operation -/
axiom diamond_positive (x y : ℝ) : 0 < x → 0 < y → 0 < diamond x y

axiom diamond_prop1 (x y : ℝ) : 0 < x → 0 < y → diamond (x * y) y = x * diamond y y

axiom diamond_prop2 (x : ℝ) : 0 < x → diamond (diamond x 1) x = diamond x 1

axiom diamond_def (x y : ℝ) : 0 < x → 0 < y → diamond x y = x / y

axiom diamond_one : diamond 1 1 = 1

/-- Theorem: 45 ◇ 15 = 3 -/
theorem diamond_45_15 : diamond 45 15 = 3 := by
  sorry

end diamond_45_15_l3136_313626


namespace inequality_preservation_l3136_313648

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end inequality_preservation_l3136_313648


namespace cubic_expression_evaluation_l3136_313695

theorem cubic_expression_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end cubic_expression_evaluation_l3136_313695


namespace not_perfect_square_l3136_313614

theorem not_perfect_square (a b : ℕ+) (h : (a.val^2 - b.val^2) % 4 ≠ 0) :
  ¬ ∃ (k : ℕ), (a.val + 3*b.val) * (5*a.val + 7*b.val) = k^2 := by
  sorry

end not_perfect_square_l3136_313614


namespace reappearance_line_l3136_313651

def letter_cycle_length : ℕ := 5
def digit_cycle_length : ℕ := 4

theorem reappearance_line : Nat.lcm letter_cycle_length digit_cycle_length = 20 := by
  sorry

end reappearance_line_l3136_313651


namespace school_club_revenue_l3136_313693

/-- Represents the revenue from full-price tickets in a school club event. -/
def revenue_full_price (total_tickets : ℕ) (total_revenue : ℚ) : ℚ :=
  let full_price : ℚ := 30
  let full_price_tickets : ℕ := 45
  full_price * full_price_tickets

/-- Proves that the revenue from full-price tickets is $1350 given the conditions. -/
theorem school_club_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (h_tickets : total_tickets = 160)
  (h_revenue : total_revenue = 2500) :
  revenue_full_price total_tickets total_revenue = 1350 := by
  sorry

#eval revenue_full_price 160 2500

end school_club_revenue_l3136_313693


namespace solution_exists_in_interval_l3136_313619

theorem solution_exists_in_interval : ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x + x = 2 := by
  sorry


end solution_exists_in_interval_l3136_313619


namespace x_equals_neg_x_valid_l3136_313694

/-- Represents a variable in a programming context -/
structure Variable where
  name : String

/-- Represents an expression in a programming context -/
inductive Expression where
  | Var : Variable → Expression
  | Num : Int → Expression
  | Neg : Expression → Expression
  | Add : Expression → Expression → Expression
  | Str : String → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an assignment is valid -/
def is_valid_assignment (a : Assignment) : Prop :=
  match a.lhs with
  | Expression.Var _ => True
  | _ => False

/-- Theorem stating that x = -x is a valid assignment -/
theorem x_equals_neg_x_valid :
  ∃ (x : Variable),
    is_valid_assignment { lhs := Expression.Var x, rhs := Expression.Neg (Expression.Var x) } ∧
    ¬is_valid_assignment { lhs := Expression.Num 5, rhs := Expression.Str "M" } ∧
    ¬is_valid_assignment { lhs := Expression.Add (Expression.Var ⟨"x"⟩) (Expression.Var ⟨"y"⟩), rhs := Expression.Num 0 } :=
by
  sorry


end x_equals_neg_x_valid_l3136_313694


namespace geneticallyModifiedMicroorganismsAllocation_l3136_313627

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysics : ℝ
  geneticallyModifiedMicroorganisms : ℝ

/-- The total budget percentage --/
def totalBudgetPercentage : ℝ := 100

/-- The total degrees in a circle --/
def totalDegrees : ℝ := 360

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem geneticallyModifiedMicroorganismsAllocation (budget : BudgetAllocation) : 
  budget.microphotonics = 12 ∧ 
  budget.homeElectronics = 24 ∧ 
  budget.foodAdditives = 15 ∧ 
  budget.industrialLubricants = 8 ∧ 
  budget.basicAstrophysics * (totalBudgetPercentage / totalDegrees) = 12 ∧
  budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
    budget.industrialLubricants + budget.basicAstrophysics + 
    budget.geneticallyModifiedMicroorganisms = totalBudgetPercentage →
  budget.geneticallyModifiedMicroorganisms = 29 := by
  sorry


end geneticallyModifiedMicroorganismsAllocation_l3136_313627


namespace james_tree_problem_l3136_313638

/-- Represents the number of trees James initially has -/
def initial_trees : ℕ := 2

/-- Represents the percentage of seeds planted -/
def planting_rate : ℚ := 60 / 100

/-- Represents the number of new trees planted -/
def new_trees : ℕ := 24

/-- Represents the number of plants per tree -/
def plants_per_tree : ℕ := 20

theorem james_tree_problem :
  plants_per_tree * initial_trees * planting_rate = new_trees :=
sorry

end james_tree_problem_l3136_313638


namespace identical_differences_l3136_313641

theorem identical_differences (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_bound : ∀ i, a i < 70) : 
  ∃ (d : ℕ) (i j k l : Fin 19), 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    a (i.succ) - a i = d ∧ 
    a (j.succ) - a j = d ∧ 
    a (k.succ) - a k = d ∧ 
    a (l.succ) - a l = d :=
sorry

end identical_differences_l3136_313641
