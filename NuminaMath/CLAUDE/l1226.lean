import Mathlib

namespace NUMINAMATH_CALUDE_prism_volume_l1226_122696

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : y * z = 8) 
  (h3 : x * z = 3) : 
  x * y * z = 24 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1226_122696


namespace NUMINAMATH_CALUDE_weight_replacement_l1226_122695

theorem weight_replacement (total_weight : ℝ) (replaced_weight : ℝ) : 
  (8 : ℝ) * ((total_weight - replaced_weight + 77) / 8 - total_weight / 8) = 1.5 →
  replaced_weight = 65 := by
sorry

end NUMINAMATH_CALUDE_weight_replacement_l1226_122695


namespace NUMINAMATH_CALUDE_time_to_bernards_house_l1226_122668

/-- Given June's biking rate and the distance to Bernard's house, prove the time to bike there --/
theorem time_to_bernards_house 
  (distance_to_julia : ℝ) 
  (time_to_julia : ℝ) 
  (distance_to_bernard : ℝ) 
  (h1 : distance_to_julia = 2) 
  (h2 : time_to_julia = 8) 
  (h3 : distance_to_bernard = 6) : 
  (time_to_julia / distance_to_julia) * distance_to_bernard = 24 := by
  sorry

end NUMINAMATH_CALUDE_time_to_bernards_house_l1226_122668


namespace NUMINAMATH_CALUDE_shepherds_sheep_count_l1226_122647

theorem shepherds_sheep_count :
  ∀ a b : ℕ,
  (∃ n : ℕ, a = n * n) →  -- a is a perfect square
  (∃ m : ℕ, b = m * m) →  -- b is a perfect square
  97 ≤ a + b →            -- lower bound of total sheep
  a + b ≤ 108 →           -- upper bound of total sheep
  a > b →                 -- Noémie has more sheep than Tristan
  a ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  b ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  Odd (a + b) →           -- Total number of sheep is odd
  a = 81 ∧ b = 16 :=      -- Conclusion: Noémie has 81 sheep, Tristan has 16 sheep
by sorry

end NUMINAMATH_CALUDE_shepherds_sheep_count_l1226_122647


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1226_122601

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1226_122601


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1226_122656

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 6 / 9
  h_major : 2 * a = 2 * Real.sqrt 3

/-- A line that intersects the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) ∧
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) ∧
    (y₁ = k * x₁ + m) ∧
    (y₂ = k * x₂ + m) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- The main theorem -/
theorem ellipse_theorem (E : Ellipse) (L : IntersectingLine E) :
  (E.a^2 = 3 ∧ E.b^2 = 1) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1) →
    (x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1) →
    (y₁ = L.k * x₁ + L.m) →
    (y₂ = L.k * x₂ + L.m) →
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (abs L.m / Real.sqrt (1 + L.k^2) = Real.sqrt 3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1226_122656


namespace NUMINAMATH_CALUDE_poplar_tree_count_l1226_122684

theorem poplar_tree_count : ∃ (poplar willow : ℕ),
  poplar + willow = 120 ∧ poplar + 10 = willow ∧ poplar = 55 := by
  sorry

end NUMINAMATH_CALUDE_poplar_tree_count_l1226_122684


namespace NUMINAMATH_CALUDE_graph_is_pair_of_lines_l1226_122653

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

/-- The graph is a pair of straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ,
    (is_straight_line f ∧ is_straight_line g) ∧
    ∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_lines_l1226_122653


namespace NUMINAMATH_CALUDE_painted_cube_probability_l1226_122658

/-- The size of the cube's side -/
def cube_side : ℕ := 5

/-- The total number of unit cubes in the larger cube -/
def total_cubes : ℕ := cube_side ^ 3

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces : ℕ := 1

/-- The number of unit cubes with no painted faces -/
def no_painted_faces : ℕ := (cube_side - 2) ^ 3

/-- The number of ways to choose two cubes out of the total -/
def total_combinations : ℕ := total_cubes.choose 2

/-- The number of successful outcomes -/
def successful_outcomes : ℕ := three_painted_faces * no_painted_faces

theorem painted_cube_probability :
  (successful_outcomes : ℚ) / total_combinations = 9 / 2583 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l1226_122658


namespace NUMINAMATH_CALUDE_alphametic_puzzle_unique_solution_l1226_122690

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the alphametic puzzle IDA + ME = ORA -/
def AlphameticPuzzle (I D A M E R O : Digit) : Prop :=
  (100 * I.val + 10 * D.val + A.val) + (10 * M.val + E.val) = 
  (100 * O.val + 10 * R.val + A.val)

/-- The main theorem stating that there exists a unique solution to the puzzle -/
theorem alphametic_puzzle_unique_solution : 
  ∃! (I D A M E R O : Digit), 
    AlphameticPuzzle I D A M E R O ∧ 
    I ≠ D ∧ I ≠ A ∧ I ≠ M ∧ I ≠ E ∧ I ≠ R ∧ I ≠ O ∧
    D ≠ A ∧ D ≠ M ∧ D ≠ E ∧ D ≠ R ∧ D ≠ O ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ R ∧ A ≠ O ∧
    M ≠ E ∧ M ≠ R ∧ M ≠ O ∧
    E ≠ R ∧ E ≠ O ∧
    R ≠ O ∧
    R.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_alphametic_puzzle_unique_solution_l1226_122690


namespace NUMINAMATH_CALUDE_exists_parallel_line_l1226_122631

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_line_l1226_122631


namespace NUMINAMATH_CALUDE_coreys_weekend_goal_l1226_122643

/-- Corey's goal for the number of golf balls to find every weekend -/
def coreys_goal (saturday_balls sunday_balls remaining_balls : ℕ) : ℕ :=
  saturday_balls + sunday_balls + remaining_balls

/-- Theorem stating Corey's goal for the number of golf balls to find every weekend -/
theorem coreys_weekend_goal :
  coreys_goal 16 18 14 = 48 := by
  sorry

end NUMINAMATH_CALUDE_coreys_weekend_goal_l1226_122643


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1226_122629

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1226_122629


namespace NUMINAMATH_CALUDE_negative_calculation_l1226_122694

theorem negative_calculation : 
  ((-4) + (-5) < 0) ∧ 
  ((-4) - (-5) ≥ 0) ∧ 
  ((-4) * (-5) ≥ 0) ∧ 
  ((-4) / (-5) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_calculation_l1226_122694


namespace NUMINAMATH_CALUDE_janice_purchase_l1226_122625

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 23 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l1226_122625


namespace NUMINAMATH_CALUDE_professors_women_tenured_or_both_l1226_122685

theorem professors_women_tenured_or_both (
  women_percentage : Real)
  (tenured_percentage : Real)
  (men_tenured_percentage : Real)
  (h1 : women_percentage = 0.69)
  (h2 : tenured_percentage = 0.70)
  (h3 : men_tenured_percentage = 0.52) :
  women_percentage + tenured_percentage - (tenured_percentage - men_tenured_percentage * (1 - women_percentage)) = 0.8512 := by
  sorry

end NUMINAMATH_CALUDE_professors_women_tenured_or_both_l1226_122685


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1226_122648

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

-- The sum-to-product identity for sine
axiom sin_sum_to_product_identity (a b : ℝ) : 
  Real.sin a + Real.sin b = 2 * Real.sin ((a + b) / 2) * Real.cos ((a - b) / 2)

end NUMINAMATH_CALUDE_sin_sum_to_product_l1226_122648


namespace NUMINAMATH_CALUDE_characterize_no_solution_set_l1226_122674

/-- The set of real numbers a for which the equation has no solution -/
def NoSolutionSet : Set ℝ :=
  {a | ∀ x, 9 * |x - 4*a| + |x - a^2| + 8*x - 4*a ≠ 0}

/-- The theorem stating the characterization of the set where the equation has no solution -/
theorem characterize_no_solution_set :
  NoSolutionSet = {a | a < -24 ∨ a > 0} :=
by sorry

end NUMINAMATH_CALUDE_characterize_no_solution_set_l1226_122674


namespace NUMINAMATH_CALUDE_dog_shampoo_count_l1226_122612

def clean_time : ℕ := 55
def hose_time : ℕ := 10
def shampoo_time : ℕ := 15

theorem dog_shampoo_count : 
  ∃ n : ℕ, n * shampoo_time + hose_time = clean_time ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_dog_shampoo_count_l1226_122612


namespace NUMINAMATH_CALUDE_coats_collected_from_high_schools_l1226_122626

theorem coats_collected_from_high_schools 
  (total_coats : ℕ) 
  (elementary_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : elementary_coats = 2515) :
  total_coats - elementary_coats = 6922 := by
sorry

end NUMINAMATH_CALUDE_coats_collected_from_high_schools_l1226_122626


namespace NUMINAMATH_CALUDE_final_state_correct_l1226_122602

/-- Represents the state of variables A, B, and C -/
structure State :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)

/-- Executes the assignment statements and returns the final state -/
def executeAssignments : State := by
  let s1 : State := { A := 0, B := 0, C := 2 }  -- C ← 2
  let s2 : State := { A := s1.A, B := 1, C := s1.C }  -- B ← 1
  let s3 : State := { A := 2, B := s2.B, C := s2.C }  -- A ← 2
  exact s3

/-- Theorem stating that the final values of A, B, and C are 2, 1, and 2 respectively -/
theorem final_state_correct : 
  let final := executeAssignments
  final.A = 2 ∧ final.B = 1 ∧ final.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_final_state_correct_l1226_122602


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l1226_122661

theorem triangle_angle_ratio (A B C : ℝ) : 
  A = 60 → B = 80 → A + B + C = 180 → B / C = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l1226_122661


namespace NUMINAMATH_CALUDE_unique_prime_digit_product_l1226_122672

def is_prime_digit (d : Nat) : Prop :=
  d ∈ [2, 3, 5, 7]

def all_prime_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_prime_digit d

theorem unique_prime_digit_product : 
  ∃! (a b : Nat), 
    100 ≤ a ∧ a < 1000 ∧
    10 ≤ b ∧ b < 100 ∧
    all_prime_digits a ∧
    all_prime_digits b ∧
    1000 ≤ a * b ∧ a * b < 10000 ∧
    all_prime_digits (a * b) ∧
    a = 775 ∧ b = 33 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_digit_product_l1226_122672


namespace NUMINAMATH_CALUDE_range_a_theorem_l1226_122618

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (a > -2 ∧ a < -1) ∨ a ≥ 1

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1226_122618


namespace NUMINAMATH_CALUDE_problem_solution_l1226_122640

theorem problem_solution (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12) 
  (h2 : a^2 + b^2 + c^2 = a*b + b*c + c*a) : 
  a + b^2 + c^3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1226_122640


namespace NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l1226_122666

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_diff_eq_sum_abs_iff_product_nonpositive_l1226_122666


namespace NUMINAMATH_CALUDE_circle_tangent_origin_l1226_122608

/-- A circle in the xy-plane -/
structure Circle where
  G : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a circle is tangent to the x-axis at the origin -/
def isTangentAtOrigin (c : Circle) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + c.G * x + c.E * y + c.F = 0 ∧
                (x = 0 ∧ y = 0) ∧
                ∀ (x' y' : ℝ), x' ≠ 0 → (x'^2 + y'^2 + c.G * x' + c.E * y' + c.F > 0)

theorem circle_tangent_origin (c : Circle) :
  isTangentAtOrigin c → c.G = 0 ∧ c.F = 0 ∧ c.E ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_origin_l1226_122608


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1226_122607

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 11 ∧ 
  (∀ (y : ℕ), (2000 - y : ℤ) % 17 = 0 → y ≥ x) ∧ 
  (2000 - x : ℤ) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1226_122607


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1226_122610

theorem polynomial_expansion (x : ℝ) :
  (1 + x^2 + 2*x - x^4) * (3 - x^3 + 2*x^2 - 5*x) =
  x^7 - 2*x^6 + 4*x^5 - 3*x^4 - 2*x^3 - 4*x^2 + x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1226_122610


namespace NUMINAMATH_CALUDE_almond_butter_servings_l1226_122652

-- Define the total amount of almond butter in cups
def total_almond_butter : ℚ := 17 + 1/3

-- Define the serving size in cups
def serving_size : ℚ := 1 + 1/2

-- Theorem: The number of servings in the container is 11 5/9
theorem almond_butter_servings :
  total_almond_butter / serving_size = 11 + 5/9 := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l1226_122652


namespace NUMINAMATH_CALUDE_min_value_theorem_l1226_122698

/-- A line that bisects a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ x y : ℝ, a * x + b * y - 2 = 0 → 
    (x - 3)^2 + (y - 2)^2 = 25 → (x - 3)^2 + (y - 2)^2 ≤ 25

/-- The theorem stating the minimum value of 3/a + 2/b -/
theorem min_value_theorem (l : BisectingLine) : 
  (∀ k : BisectingLine, 3 / l.a + 2 / l.b ≤ 3 / k.a + 2 / k.b) → 
  3 / l.a + 2 / l.b = 25 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1226_122698


namespace NUMINAMATH_CALUDE_sara_hotdog_cost_l1226_122667

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.1
  hotdog := 5.36

/-- Theorem: Sara's hotdog cost $5.36 -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

#check sara_hotdog_cost

end NUMINAMATH_CALUDE_sara_hotdog_cost_l1226_122667


namespace NUMINAMATH_CALUDE_difference_61st_terms_l1226_122679

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def sequenceC (n : ℕ) : ℝ := arithmeticSequence 20 15 n

def sequenceD (n : ℕ) : ℝ := arithmeticSequence 20 (-15) n

theorem difference_61st_terms :
  |sequenceC 61 - sequenceD 61| = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_61st_terms_l1226_122679


namespace NUMINAMATH_CALUDE_correct_transformation_l1226_122693

theorem correct_transformation (x : ℝ) : 2*x = 3*x + 4 → 2*x - 3*x = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l1226_122693


namespace NUMINAMATH_CALUDE_article_percentage_loss_l1226_122638

theorem article_percentage_loss 
  (selling_price : ℝ) 
  (selling_price_with_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price = 136 →
  selling_price_with_gain = 192 →
  gain_percentage = 20 →
  let cost_price := selling_price_with_gain / (1 + gain_percentage / 100)
  let loss := cost_price - selling_price
  let percentage_loss := (loss / cost_price) * 100
  percentage_loss = 15 := by
sorry

end NUMINAMATH_CALUDE_article_percentage_loss_l1226_122638


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1226_122691

theorem system_of_equations_sum (x y z : ℝ) :
  (y + z = 20 - 4*x) ∧ 
  (x + z = 1 - 4*y) ∧ 
  (x + y = -12 - 4*z) →
  3*x + 3*y + 3*z = 9/2 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1226_122691


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1226_122676

theorem log_sum_equals_two : Real.log 3 / Real.log 6 + Real.log 4 / Real.log 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1226_122676


namespace NUMINAMATH_CALUDE_hexagon_problem_l1226_122630

-- Define the regular hexagon
structure RegularHexagon :=
  (side_length : ℝ)
  (A B C D E F : ℝ × ℝ)

-- Define the intersection point L
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Define point K
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Function to check if a point is outside the hexagon
def is_outside (hex : RegularHexagon) (point : ℝ × ℝ) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (hex : RegularHexagon) 
  (h1 : hex.side_length = 2) :
  is_outside hex (K hex) ∧ 
  distance (K hex) hex.B = (2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_problem_l1226_122630


namespace NUMINAMATH_CALUDE_division_remainder_seventeen_by_two_l1226_122636

theorem division_remainder_seventeen_by_two :
  ∃ (q : ℕ), 17 = 2 * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_seventeen_by_two_l1226_122636


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1226_122611

theorem cosine_sine_identity : 
  Real.cos (20 * π / 180) * Real.cos (385 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (155 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1226_122611


namespace NUMINAMATH_CALUDE_simplify_expression_l1226_122632

theorem simplify_expression (x : ℝ) : (2*x)^4 + (3*x)*(x^3) = 19*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1226_122632


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l1226_122686

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (a b c d e f : ℝ) : Prop := a * e = b * d

-- Theorem statement
theorem intersection_and_parallel_line :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    intersection_point x y →
    parallel_lines 2 3 k 2 3 5 →
    2 * x + 3 * y + k = 0 →
    k = -7 :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l1226_122686


namespace NUMINAMATH_CALUDE_rectangular_field_area_increase_l1226_122617

theorem rectangular_field_area_increase 
  (original_length : ℝ) 
  (original_width : ℝ) 
  (length_increase : ℝ) : 
  original_length = 20 →
  original_width = 5 →
  length_increase = 10 →
  (original_length + length_increase) * original_width - original_length * original_width = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_increase_l1226_122617


namespace NUMINAMATH_CALUDE_multiples_of_three_is_closed_set_l1226_122673

-- Define a closed set
def is_closed_set (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

-- Define the set A
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

-- Theorem statement
theorem multiples_of_three_is_closed_set : is_closed_set A := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_three_is_closed_set_l1226_122673


namespace NUMINAMATH_CALUDE_train_length_proof_l1226_122621

/-- Given a train that crosses an electric pole in 30 seconds at a speed of 43.2 km/h,
    prove that its length is 360 meters. -/
theorem train_length_proof (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 30 →
  speed_kmh = 43.2 →
  length = speed_kmh * 1000 / 3600 * crossing_time →
  length = 360 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l1226_122621


namespace NUMINAMATH_CALUDE_camera_cost_proof_l1226_122628

/-- The cost of the old camera model --/
def old_camera_cost : ℝ := 4000

/-- The cost of the new camera model --/
def new_camera_cost : ℝ := old_camera_cost * 1.3

/-- The original price of the lens --/
def lens_original_price : ℝ := 400

/-- The discount on the lens --/
def lens_discount : ℝ := 200

/-- The discounted price of the lens --/
def lens_discounted_price : ℝ := lens_original_price - lens_discount

/-- The total amount paid for the new camera and the discounted lens --/
def total_paid : ℝ := 5400

theorem camera_cost_proof : 
  new_camera_cost + lens_discounted_price = total_paid ∧ 
  old_camera_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_camera_cost_proof_l1226_122628


namespace NUMINAMATH_CALUDE_justin_age_l1226_122655

/-- Prove that Justin's age is 26 years -/
theorem justin_age :
  ∀ (justin_age jessica_age james_age : ℕ),
  (jessica_age = justin_age + 6) →
  (james_age = jessica_age + 7) →
  (james_age + 5 = 44) →
  justin_age = 26 := by
sorry

end NUMINAMATH_CALUDE_justin_age_l1226_122655


namespace NUMINAMATH_CALUDE_compound_mass_proof_l1226_122637

/-- The atomic mass of Carbon in g/mol -/
def atomic_mass_C : ℝ := 12.01

/-- The atomic mass of Hydrogen in g/mol -/
def atomic_mass_H : ℝ := 1.008

/-- The atomic mass of Oxygen in g/mol -/
def atomic_mass_O : ℝ := 16.00

/-- The atomic mass of Nitrogen in g/mol -/
def atomic_mass_N : ℝ := 14.01

/-- The atomic mass of Bromine in g/mol -/
def atomic_mass_Br : ℝ := 79.90

/-- The molecular formula of the compound -/
def compound_formula := "C8H10O2NBr2"

/-- The number of moles of the compound -/
def moles_compound : ℝ := 3

/-- The total mass of the compound in grams -/
def total_mass : ℝ := 938.91

/-- Theorem stating that the total mass of 3 moles of C8H10O2NBr2 is 938.91 grams -/
theorem compound_mass_proof :
  moles_compound * (8 * atomic_mass_C + 10 * atomic_mass_H + 2 * atomic_mass_O + atomic_mass_N + 2 * atomic_mass_Br) = total_mass := by
  sorry

end NUMINAMATH_CALUDE_compound_mass_proof_l1226_122637


namespace NUMINAMATH_CALUDE_elise_remaining_money_l1226_122669

/-- Calculates the remaining money in dollars for Elise --/
def remaining_money (initial_amount : ℝ) (saved_euros : ℝ) (euro_to_dollar : ℝ) 
                    (comic_cost : ℝ) (puzzle_cost_pounds : ℝ) (pound_to_dollar : ℝ) : ℝ :=
  initial_amount + saved_euros * euro_to_dollar - comic_cost - puzzle_cost_pounds * pound_to_dollar

/-- Theorem stating that Elise's remaining money is $1.04 --/
theorem elise_remaining_money :
  remaining_money 8 11 1.18 2 13 1.38 = 1.04 := by
  sorry

end NUMINAMATH_CALUDE_elise_remaining_money_l1226_122669


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l1226_122657

theorem power_of_three_mod_five : 3^17 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l1226_122657


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1226_122641

theorem quadrilateral_area (S_ABCD S_OKSL S_ONAM S_OMBK : ℝ) 
  (h1 : S_ABCD = 4 * (S_OKSL + S_ONAM))
  (h2 : S_OKSL = 6)
  (h3 : S_ONAM = 12)
  (h4 : S_OMBK = S_ABCD - S_OKSL - 24 - S_ONAM) :
  S_ABCD = 72 ∧ S_OMBK = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1226_122641


namespace NUMINAMATH_CALUDE_solve_for_m_l1226_122675

/-- 
If 2x + m = 6 and x = 2, then m = 2
-/
theorem solve_for_m (x m : ℝ) (eq : 2 * x + m = 6) (sol : x = 2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1226_122675


namespace NUMINAMATH_CALUDE_sea_horse_penguin_ratio_l1226_122634

/-- The number of sea horses at the zoo -/
def sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_horses + 85

/-- The ratio of sea horses to penguins -/
def ratio : ℕ × ℕ := (14, 31)

/-- Theorem stating that the ratio of sea horses to penguins is 14:31 -/
theorem sea_horse_penguin_ratio :
  (sea_horses : ℚ) / (penguins : ℚ) = (ratio.1 : ℚ) / (ratio.2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sea_horse_penguin_ratio_l1226_122634


namespace NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l1226_122663

/-- Given distinct positive rational numbers a and b, if a^n - b^n is a positive integer
    for infinitely many positive integers n, then a and b are positive integers. -/
theorem rational_power_difference_integer_implies_integer
  (a b : ℚ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_distinct : a ≠ b)
  (h_infinite : ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ∃ k : ℤ, k > 0 ∧ a^n - b^n = k) :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ a = m ∧ b = n :=
sorry

end NUMINAMATH_CALUDE_rational_power_difference_integer_implies_integer_l1226_122663


namespace NUMINAMATH_CALUDE_square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1226_122683

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem special_case_square_root_three (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (25 + 4 * Real.sqrt 6) = 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1226_122683


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1226_122623

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -(b / a) ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2023 * x - 2024
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -2023 ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1226_122623


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1226_122699

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  (a 1 + a 2 + a 3 + a 4 = 1) →  -- sum of first 4 terms is 1
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1226_122699


namespace NUMINAMATH_CALUDE_joint_completion_time_l1226_122624

/-- Given two people A and B who can complete a task in x and y hours respectively,
    the time it takes for them to complete the task together is xy/(x+y) hours. -/
theorem joint_completion_time (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x⁻¹ + y⁻¹)⁻¹ = x * y / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_joint_completion_time_l1226_122624


namespace NUMINAMATH_CALUDE_probability_three_correct_is_one_sixth_l1226_122622

/-- The probability of exactly 3 out of 5 packages being delivered to the correct houses in a random delivery -/
def probability_three_correct_deliveries : ℚ :=
  (Nat.choose 5 3 * 2) / Nat.factorial 5

/-- Theorem stating that the probability of exactly 3 out of 5 packages being delivered to the correct houses is 1/6 -/
theorem probability_three_correct_is_one_sixth :
  probability_three_correct_deliveries = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_three_correct_is_one_sixth_l1226_122622


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l1226_122649

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(2) = 40640 -/
theorem polynomial_root_relation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ (x : ℝ), g x = 0 → f x = 0) →
  f 2 = 40640 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l1226_122649


namespace NUMINAMATH_CALUDE_neg_a_cubed_times_a_squared_l1226_122678

theorem neg_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_neg_a_cubed_times_a_squared_l1226_122678


namespace NUMINAMATH_CALUDE_cid_earnings_l1226_122692

def oil_change_price : ℕ := 20
def repair_price : ℕ := 30
def car_wash_price : ℕ := 5

def oil_changes_performed : ℕ := 5
def repairs_performed : ℕ := 10
def car_washes_performed : ℕ := 15

def total_earnings : ℕ := 
  oil_change_price * oil_changes_performed + 
  repair_price * repairs_performed + 
  car_wash_price * car_washes_performed

theorem cid_earnings : total_earnings = 475 := by
  sorry

end NUMINAMATH_CALUDE_cid_earnings_l1226_122692


namespace NUMINAMATH_CALUDE_parabola_vertex_l1226_122681

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -9 * (x - 7)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (7, 0)

/-- Theorem: The vertex of the parabola y = -9(x-7)^2 is at the point (7, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1226_122681


namespace NUMINAMATH_CALUDE_a_4_equals_zero_l1226_122606

def sequence_a (n : ℕ) : ℤ := n^2 - 2*n - 8

theorem a_4_equals_zero : sequence_a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_zero_l1226_122606


namespace NUMINAMATH_CALUDE_sum_of_powers_l1226_122687

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1226_122687


namespace NUMINAMATH_CALUDE_carol_first_six_probability_l1226_122671

/-- The probability of rolling a 6 on any single roll. -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on any single roll. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of rolls, where 0 represents Alice, 1 represents Bob, and 2 represents Carol. -/
def roll_sequence : ℕ → Fin 3
  | n => n % 3

/-- The probability that Carol is the first to roll a 6. -/
def prob_carol_first_six : ℚ :=
  let a : ℚ := prob_not_six ^ 2 * prob_six
  let r : ℚ := prob_not_six ^ 3
  a / (1 - r)

theorem carol_first_six_probability :
  prob_carol_first_six = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_carol_first_six_probability_l1226_122671


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1226_122639

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1226_122639


namespace NUMINAMATH_CALUDE_factorization_problems_l1226_122619

theorem factorization_problems (x : ℝ) : 
  (9 * x^2 - 6 * x + 1 = (3 * x - 1)^2) ∧ 
  (x^3 - x = x * (x + 1) * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1226_122619


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l1226_122642

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the radius of the inscribed sphere to the height of a regular tetrahedron is 1:4 -/
theorem inscribed_sphere_radius_to_height_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_to_height_ratio_l1226_122642


namespace NUMINAMATH_CALUDE_additional_telephone_lines_l1226_122654

theorem additional_telephone_lines :
  (9 * 10^6 : ℕ) - (9 * 10^5 : ℕ) = 81 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_additional_telephone_lines_l1226_122654


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1226_122615

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p / 2

-- Define the left focus of the hyperbola
def left_focus_hyperbola (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) :
  (∃ x y : ℝ, directrix p x ∧ left_focus_hyperbola x y) →
  p = 4 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l1226_122615


namespace NUMINAMATH_CALUDE_line_slope_from_parametric_equation_l1226_122609

/-- Given a line l with parametric equations x = 1 - (3/5)t and y = (4/5)t,
    prove that the slope of the line is -4/3 -/
theorem line_slope_from_parametric_equation :
  ∀ (l : ℝ → ℝ × ℝ),
  (∀ t, l t = (1 - 3/5 * t, 4/5 * t)) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b ∧ m = -4/3) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_from_parametric_equation_l1226_122609


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l1226_122600

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The left segment of the larger base divided by the height -/
  leftSegment : ℝ
  /-- The right segment of the larger base divided by the height -/
  rightSegment : ℝ
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sum of segments equals the larger base -/
  segment_sum : leftSegment + rightSegment = largerBase
  /-- The ratio of segments is 2:3 -/
  segment_ratio : leftSegment / rightSegment = 2 / 3

/-- 
If the height of an isosceles trapezoid divides the larger base into segments 
with a ratio of 2:3, then the ratio of the larger base to the smaller base is 5:1
-/
theorem isosceles_trapezoid_base_ratio (t : IsoscelesTrapezoid) : 
  t.largerBase / t.smallerBase = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l1226_122600


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l1226_122689

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101(2) -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110(2) -/
def binary2 : List Bool := [false, true, true]

/-- Statement: The sum of the decimal representations of 101(2) and 110(2) is 11 -/
theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l1226_122689


namespace NUMINAMATH_CALUDE_three_roots_symmetric_about_two_l1226_122660

/-- A function f: ℝ → ℝ that satisfies f(2+x) = f(2-x) for all x ∈ ℝ -/
def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

/-- The set of roots of f -/
def roots (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

theorem three_roots_symmetric_about_two (f : ℝ → ℝ) :
  symmetric_about_two f →
  (∃ a b : ℝ, roots f = {0, a, b} ∧ a ≠ b ∧ a ≠ 0 ∧ b ≠ 0) →
  roots f = {0, 2, 4} :=
sorry

end NUMINAMATH_CALUDE_three_roots_symmetric_about_two_l1226_122660


namespace NUMINAMATH_CALUDE_mets_fans_count_l1226_122677

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The conditions of the problem -/
def fan_conditions (fc : FanCounts) : Prop :=
  -- Ratio of Yankees to Mets fans is 3:2
  3 * fc.mets = 2 * fc.yankees ∧
  -- Ratio of Mets to Red Sox fans is 4:5
  4 * fc.red_sox = 5 * fc.mets ∧
  -- Total number of fans is 330
  fc.yankees + fc.mets + fc.red_sox = 330

/-- The theorem to prove -/
theorem mets_fans_count (fc : FanCounts) : 
  fan_conditions fc → fc.mets = 88 := by
  sorry


end NUMINAMATH_CALUDE_mets_fans_count_l1226_122677


namespace NUMINAMATH_CALUDE_quadrilateral_properties_l1226_122664

/-- Definition of a quadrilateral with given vertices -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Function to calculate the intersection point of diagonals -/
def diagonalIntersection (q : Quadrilateral) : ℝ × ℝ := sorry

/-- Function to calculate the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating the properties of the given quadrilateral -/
theorem quadrilateral_properties :
  let q := Quadrilateral.mk (5, 6) (-1, 2) (-2, -1) (4, -5)
  diagonalIntersection q = (-1/6, 5/6) ∧
  quadrilateralArea q = 42 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_properties_l1226_122664


namespace NUMINAMATH_CALUDE_correct_decision_probability_l1226_122680

theorem correct_decision_probability (p : ℝ) (h : p = 0.8) :
  let n := 3  -- number of consultants
  let prob_two_correct := Nat.choose n 2 * p^2 * (1 - p)
  let prob_three_correct := Nat.choose n 3 * p^3
  prob_two_correct + prob_three_correct = 0.896 :=
sorry

end NUMINAMATH_CALUDE_correct_decision_probability_l1226_122680


namespace NUMINAMATH_CALUDE_flu_infection_rate_l1226_122605

/-- The average number of people infected by one person in each round -/
def average_infections : ℝ := 4

/-- The number of people initially infected -/
def initial_infected : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 50

theorem flu_infection_rate :
  initial_infected +
  initial_infected * average_infections +
  (initial_infected + initial_infected * average_infections) * average_infections =
  total_infected :=
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l1226_122605


namespace NUMINAMATH_CALUDE_car_distance_calculation_l1226_122604

/-- The distance a car needs to cover given initial time and new speed requirements -/
theorem car_distance_calculation (initial_time : ℝ) (new_speed : ℝ) : 
  initial_time = 6 →
  new_speed = 36 →
  (new_speed * (3/2 * initial_time)) = 324 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l1226_122604


namespace NUMINAMATH_CALUDE_incorrect_number_correction_l1226_122644

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 16)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 17) :
  let correct_num := incorrect_num - (n * correct_avg - n * incorrect_avg)
  correct_num = 15 := by sorry

end NUMINAMATH_CALUDE_incorrect_number_correction_l1226_122644


namespace NUMINAMATH_CALUDE_line_intersect_xz_plane_l1226_122670

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersect_xz_plane (p₁ p₂ intersection : ℝ × ℝ × ℝ) :
  p₁ = (1, 2, 3) →
  p₂ = (4, 0, -1) →
  intersection = (4, 0, -1) →
  (∃ t : ℝ, intersection = p₁ + t • (p₂ - p₁)) ∧
  (intersection.2 = 0) := by
  sorry

#check line_intersect_xz_plane

end NUMINAMATH_CALUDE_line_intersect_xz_plane_l1226_122670


namespace NUMINAMATH_CALUDE_mcpherson_rent_contribution_l1226_122635

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mcpherson_rent_contribution 
  (total_rent : ℕ) 
  (mrs_mcpherson_percentage : ℚ) 
  (h1 : total_rent = 1200)
  (h2 : mrs_mcpherson_percentage = 30 / 100) : 
  total_rent - (mrs_mcpherson_percentage * total_rent).floor = 840 := by
sorry

end NUMINAMATH_CALUDE_mcpherson_rent_contribution_l1226_122635


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1226_122697

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 15*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 5 / 2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1226_122697


namespace NUMINAMATH_CALUDE_exponent_division_l1226_122650

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1226_122650


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_l1226_122659

/-- A cubic function with parameters m and n. -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x. -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_l1226_122659


namespace NUMINAMATH_CALUDE_find_number_l1226_122665

theorem find_number : ∃! x : ℝ, (x + 82 + 90 + 88 + 84) / 5 = 88 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1226_122665


namespace NUMINAMATH_CALUDE_uniform_price_l1226_122603

def full_year_salary : ℕ := 500
def months_worked : ℕ := 9
def payment_received : ℕ := 300

theorem uniform_price : 
  ∃ (uniform_price : ℕ), 
    (uniform_price + payment_received = (months_worked * full_year_salary) / 12) ∧
    uniform_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_l1226_122603


namespace NUMINAMATH_CALUDE_people_on_boats_l1226_122682

theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) :
  num_boats = 5 → people_per_boat = 3 → num_boats * people_per_boat = 15 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l1226_122682


namespace NUMINAMATH_CALUDE_school_demographics_l1226_122616

theorem school_demographics (total_students : ℕ) (avg_age_boys avg_age_girls avg_age_school : ℚ) : 
  total_students = 640 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 160 ∧ 
    (total_students - num_girls) * avg_age_boys + num_girls * avg_age_girls = total_students * avg_age_school :=
by sorry

end NUMINAMATH_CALUDE_school_demographics_l1226_122616


namespace NUMINAMATH_CALUDE_intersection_M_N_l1226_122688

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1226_122688


namespace NUMINAMATH_CALUDE_A_inter_B_equals_open_interval_l1226_122620

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| < 2}

theorem A_inter_B_equals_open_interval : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_equals_open_interval_l1226_122620


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1226_122651

-- Define the quadratic function
def f (x : ℝ) := x^2 - 5*x + 6

-- Define the solution set
def solution_set := { x : ℝ | 2 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem quadratic_inequality_solution :
  { x : ℝ | f x ≤ 0 } = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1226_122651


namespace NUMINAMATH_CALUDE_find_divisor_with_remainder_relation_l1226_122646

theorem find_divisor_with_remainder_relation : ∃ (A : ℕ), 
  (312 % A = 2 * (270 % A)) ∧ 
  (270 % A = 2 * (211 % A)) ∧ 
  (A = 19) := by
sorry

end NUMINAMATH_CALUDE_find_divisor_with_remainder_relation_l1226_122646


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1226_122662

theorem min_value_of_expression (x y : ℝ) :
  Real.sqrt (2 * x^2 - 6 * x + 5) + Real.sqrt (y^2 - 4 * y + 5) + Real.sqrt (2 * x^2 - 2 * x * y + y^2) ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1226_122662


namespace NUMINAMATH_CALUDE_crocodile_count_correct_l1226_122633

/-- Represents the number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- Represents the number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- Represents the number of eyes each animal (frog or crocodile) has -/
def eyes_per_animal : ℕ := 2

/-- Represents the total number of animal eyes in the pond -/
def total_eyes : ℕ := 60

/-- Theorem stating that the number of crocodiles is correct given the conditions -/
theorem crocodile_count_correct :
  num_crocodiles * eyes_per_animal + num_frogs * eyes_per_animal = total_eyes :=
sorry

end NUMINAMATH_CALUDE_crocodile_count_correct_l1226_122633


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1226_122613

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l1226_122613


namespace NUMINAMATH_CALUDE_fair_rides_l1226_122627

theorem fair_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) 
  (h1 : initial_tickets = 79)
  (h2 : spent_tickets = 23)
  (h3 : tickets_per_ride = 7) :
  (initial_tickets - spent_tickets) / tickets_per_ride = 8 := by
  sorry

end NUMINAMATH_CALUDE_fair_rides_l1226_122627


namespace NUMINAMATH_CALUDE_batsman_average_l1226_122614

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : 
  total_innings = 17 → 
  last_innings_score = 85 → 
  average_increase = 3 → 
  (((total_innings - 1) * ((total_innings * (37 - average_increase)) / total_innings) + last_innings_score) / total_innings : ℚ) = 37 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l1226_122614


namespace NUMINAMATH_CALUDE_copy_machine_rate_copy_machine_rate_proof_l1226_122645

/-- Given two copy machines working together for 30 minutes to produce 3000 copies,
    where one machine produces 65 copies per minute, prove that the other machine
    must produce 35 copies per minute. -/
theorem copy_machine_rate : ℕ → Prop :=
  fun x =>
    -- x is the number of copies per minute for the first machine
    -- 65 is the number of copies per minute for the second machine
    -- 30 is the number of minutes they work
    -- 3000 is the total number of copies produced
    30 * x + 30 * 65 = 3000 →
    x = 35

-- The proof would go here, but we're skipping it as requested
theorem copy_machine_rate_proof : copy_machine_rate 35 := by sorry

end NUMINAMATH_CALUDE_copy_machine_rate_copy_machine_rate_proof_l1226_122645
