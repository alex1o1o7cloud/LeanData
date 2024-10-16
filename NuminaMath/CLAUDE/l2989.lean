import Mathlib

namespace NUMINAMATH_CALUDE_expand_square_root_two_l2989_298994

theorem expand_square_root_two (a b : ℚ) : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_square_root_two_l2989_298994


namespace NUMINAMATH_CALUDE_down_jacket_price_l2989_298942

/-- The marked price of a down jacket -/
def marked_price : ℝ := 550

/-- The cost price of the down jacket -/
def cost_price : ℝ := 350

/-- The selling price as a percentage of the marked price -/
def selling_percentage : ℝ := 0.8

/-- The profit made on the sale of the jacket -/
def profit : ℝ := 90

/-- Theorem stating that the marked price satisfies the given conditions -/
theorem down_jacket_price : 
  selling_percentage * marked_price - cost_price = profit :=
by sorry

end NUMINAMATH_CALUDE_down_jacket_price_l2989_298942


namespace NUMINAMATH_CALUDE_overtaking_distance_l2989_298920

/-- Represents a vehicle with a given length -/
structure Vehicle where
  length : ℝ

/-- Represents the overtaking scenario on a highway -/
structure OvertakingScenario where
  sedan : Vehicle
  truck : Vehicle

/-- The additional distance traveled by the sedan during overtaking -/
def additionalDistance (scenario : OvertakingScenario) : ℝ :=
  scenario.sedan.length + scenario.truck.length

theorem overtaking_distance (scenario : OvertakingScenario) :
  additionalDistance scenario = scenario.sedan.length + scenario.truck.length := by
  sorry

end NUMINAMATH_CALUDE_overtaking_distance_l2989_298920


namespace NUMINAMATH_CALUDE_marsh_bird_difference_l2989_298914

theorem marsh_bird_difference (canadian_geese mallard_ducks great_egrets red_winged_blackbirds : ℕ) 
  (h1 : canadian_geese = 58)
  (h2 : mallard_ducks = 37)
  (h3 : great_egrets = 21)
  (h4 : red_winged_blackbirds = 15) :
  canadian_geese - mallard_ducks = 21 := by
  sorry

end NUMINAMATH_CALUDE_marsh_bird_difference_l2989_298914


namespace NUMINAMATH_CALUDE_sequence_integer_count_l2989_298916

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ k : ℕ, ∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) ∧
  (∀ k : ℕ, (∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) → k = 5) :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l2989_298916


namespace NUMINAMATH_CALUDE_binomial_coefficient_x5_in_1_plus_x_7_l2989_298902

theorem binomial_coefficient_x5_in_1_plus_x_7 :
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * (1 ^ (7 - k)) * (x ^ k)) =
  21 * x^5 + (Finset.range 8).sum (fun k => if k ≠ 5 then (Nat.choose 7 k) * (1 ^ (7 - k)) * (x ^ k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x5_in_1_plus_x_7_l2989_298902


namespace NUMINAMATH_CALUDE_sqrt_product_minus_one_equals_546_l2989_298939

theorem sqrt_product_minus_one_equals_546 : 
  Real.sqrt ((25 : ℝ) * 24 * 23 * 22 - 1) = 546 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_minus_one_equals_546_l2989_298939


namespace NUMINAMATH_CALUDE_inequality_condition_l2989_298932

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 2 → x^2 - 2*x + a < 0) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2989_298932


namespace NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l2989_298962

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  let geometric_mean := Real.sqrt (a * b)
  geometric_mean = 1 ∨ geometric_mean = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_of_sqrt2_plus_minus_one_l2989_298962


namespace NUMINAMATH_CALUDE_log_inequality_l2989_298926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality (a x₁ x₂ : ℝ) (ha : a > 0 ∧ a ≠ 1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  (a > 1 → (f a x₁ + f a x₂) / 2 ≤ f a ((x₁ + x₂) / 2)) ∧
  (0 < a ∧ a < 1 → (f a x₁ + f a x₂) / 2 ≥ f a ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2989_298926


namespace NUMINAMATH_CALUDE_property_implies_increasing_l2989_298963

-- Define the property that (f(a) - f(b)) / (a - b) > 0 for all distinct a and b
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem property_implies_increasing (f : ℝ → ℝ) :
  satisfies_property f → is_increasing f :=
by
  sorry

end NUMINAMATH_CALUDE_property_implies_increasing_l2989_298963


namespace NUMINAMATH_CALUDE_special_polyhedron_ratio_l2989_298988

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : Nat
  x : ℝ
  y : ℝ
  isIsosceles : Bool
  vertexDegrees : Finset Nat
  dihedralAnglesEqual : Bool

/-- The conditions for our special polyhedron -/
def specialPolyhedronConditions (p : SpecialPolyhedron) : Prop :=
  p.faces = 12 ∧
  p.isIsosceles = true ∧
  p.vertexDegrees = {3, 6} ∧
  p.dihedralAnglesEqual = true

/-- The theorem stating the ratio of x to y for our special polyhedron -/
theorem special_polyhedron_ratio (p : SpecialPolyhedron) 
  (h : specialPolyhedronConditions p) : p.x / p.y = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_special_polyhedron_ratio_l2989_298988


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2989_298976

theorem cube_volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : 
  b^3 / a^3 = 125 / 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2989_298976


namespace NUMINAMATH_CALUDE_sqrt_inequality_and_fraction_bound_l2989_298917

theorem sqrt_inequality_and_fraction_bound : 
  (Real.sqrt 5 + Real.sqrt 7 > 1 + Real.sqrt 13) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y > 1 → 
    min ((1 + x) / y) ((1 + y) / x) < 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_and_fraction_bound_l2989_298917


namespace NUMINAMATH_CALUDE_min_perimeter_area_l2989_298904

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F
def rightFocus : ℝ × ℝ := (3, 0)

-- Define point A
def A : ℝ × ℝ := (0, 4)

-- Define a point P on the left branch of the hyperbola
def P : ℝ × ℝ := sorry

-- Define the perimeter of triangle APF
def perimeter (p : ℝ × ℝ) : ℝ := sorry

-- Define the area of triangle APF
def area (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem min_perimeter_area :
  ∃ (p : ℝ × ℝ), 
    hyperbola p.1 p.2 ∧ 
    p.1 < 0 ∧ 
    (∀ q : ℝ × ℝ, hyperbola q.1 q.2 ∧ q.1 < 0 → perimeter p ≤ perimeter q) ∧
    area p = 36/7 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_area_l2989_298904


namespace NUMINAMATH_CALUDE_monster_family_eyes_l2989_298925

/-- A monster family with a specific number of eyes for each member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  kid_eyes : ℕ

/-- Calculate the total number of eyes in a monster family -/
def total_eyes (family : MonsterFamily) : ℕ :=
  family.mom_eyes + family.dad_eyes + family.num_kids * family.kid_eyes

/-- Theorem: The total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ (family : MonsterFamily),
    family.mom_eyes = 1 ∧
    family.dad_eyes = 3 ∧
    family.num_kids = 3 ∧
    family.kid_eyes = 4 ∧
    total_eyes family = 16 := by
  sorry

end NUMINAMATH_CALUDE_monster_family_eyes_l2989_298925


namespace NUMINAMATH_CALUDE_bisection_method_structures_l2989_298984

/-- Bisection method for finding the approximate root of x^2 - 5 = 0 -/
def bisection_method (f : ℝ → ℝ) (a b : ℝ) (ε : ℝ) : ℝ := sorry

/-- The equation to solve -/
def equation (x : ℝ) : ℝ := x^2 - 5

theorem bisection_method_structures :
  ∃ (sequential conditional loop : Bool),
    sequential ∧ conditional ∧ loop ∧
    (∀ (a b ε : ℝ), ε > 0 → 
      ∃ (result : ℝ), 
        bisection_method equation a b ε = result ∧ 
        |equation result| < ε) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_structures_l2989_298984


namespace NUMINAMATH_CALUDE_intersection_A_B_l2989_298921

-- Define set A
def A : Set ℝ := {x : ℝ | x * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {0, 1, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2989_298921


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2989_298991

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  1 / a + 1 / b = 65 / 6 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2989_298991


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2989_298907

theorem volleyball_team_selection (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 16 →
  k = 7 →
  m = 2 →
  (Nat.choose (n - m) k) + (Nat.choose (n - m) (k - m)) = 5434 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2989_298907


namespace NUMINAMATH_CALUDE_chad_odd_jobs_income_l2989_298997

theorem chad_odd_jobs_income 
  (savings_rate : Real)
  (mowing_income : Real)
  (birthday_income : Real)
  (videogame_income : Real)
  (total_savings : Real)
  (h1 : savings_rate = 0.4)
  (h2 : mowing_income = 600)
  (h3 : birthday_income = 250)
  (h4 : videogame_income = 150)
  (h5 : total_savings = 460) :
  ∃ (odd_jobs_income : Real),
    odd_jobs_income = 150 ∧
    total_savings = savings_rate * (mowing_income + birthday_income + videogame_income + odd_jobs_income) :=
by
  sorry


end NUMINAMATH_CALUDE_chad_odd_jobs_income_l2989_298997


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2989_298946

/-- The equation 3x^2 - 9y^2 - 18y = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0),
    ∀ (x y : ℝ), 3 * x^2 - 9 * y^2 - 18 * y = 0 ↔
      ((y + c)^2 / a^2) - (x^2 / b^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2989_298946


namespace NUMINAMATH_CALUDE_second_class_average_l2989_298940

/-- Proves that given two classes with specified student counts and averages,
    the average mark of the second class is 90. -/
theorem second_class_average (students1 students2 : ℕ) (avg1 avg_combined : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_combined = 71.25 →
  (students1 * avg1 + students2 * (90 : ℚ)) / (students1 + students2 : ℚ) = avg_combined :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_l2989_298940


namespace NUMINAMATH_CALUDE_modulo_congruence_problem_l2989_298999

theorem modulo_congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 49325 % 31 = n % 31 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_problem_l2989_298999


namespace NUMINAMATH_CALUDE_product_x_y_is_32_l2989_298981

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  is_parallelogram : EF = GH ∧ FG = HE

/-- The product of x and y in the given parallelogram is 32 -/
theorem product_x_y_is_32 (p : Parallelogram)
  (h1 : p.EF = 42)
  (h2 : ∃ y, p.FG = 4 * y^3)
  (h3 : ∃ x, p.GH = 2 * x + 10)
  (h4 : p.HE = 32) :
  ∃ x y, x * y = 32 ∧ p.FG = 4 * y^3 ∧ p.GH = 2 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_product_x_y_is_32_l2989_298981


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2989_298964

/-- The coefficient of x^3 in the expansion of (x^2 - 4)(x + 1/x)^9 -/
def coefficient_x_cubed : ℤ := -210

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x_cubed = binomial 9 4 - 4 * binomial 9 3 :=
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2989_298964


namespace NUMINAMATH_CALUDE_exam_mean_score_l2989_298966

theorem exam_mean_score (score_below mean score_above : ℝ) 
  (h1 : score_below = mean - 2 * (score_above - mean) / 5)
  (h2 : score_above = mean + 3 * (score_above - mean) / 5)
  (h3 : score_below = 60)
  (h4 : score_above = 100) : 
  mean = 76 := by
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2989_298966


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l2989_298912

/-- The number of non-birch trees -/
def non_birch_trees : ℕ := 9

/-- The number of birch trees -/
def birch_trees : ℕ := 3

/-- The total number of trees -/
def total_trees : ℕ := non_birch_trees + birch_trees

/-- The number of slots available for birch trees -/
def available_slots : ℕ := non_birch_trees + 1

/-- The probability of no two birch trees being adjacent when randomly arranged -/
theorem birch_tree_arrangement_probability :
  (Nat.choose available_slots birch_trees : ℚ) / (Nat.choose total_trees birch_trees : ℚ) = 6 / 11 := by
  sorry

#eval Nat.choose available_slots birch_trees + Nat.choose total_trees birch_trees

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l2989_298912


namespace NUMINAMATH_CALUDE_book_purchase_problem_l2989_298957

theorem book_purchase_problem :
  ∀ (total_A total_B only_A only_B both : ℕ),
    total_A = 2 * total_B →
    both = 500 →
    both = 2 * only_B →
    total_A = only_A + both →
    total_B = only_B + both →
    only_A = 1000 := by
  sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l2989_298957


namespace NUMINAMATH_CALUDE_watermelon_cost_undetermined_l2989_298975

-- Define the given constants
def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_bought : ℕ := 2

-- Define the amount spent on watermelons
def watermelon_spent : ℕ := total_spent - (pineapple_cost * pineapples_bought)

-- Theorem stating that the cost of each watermelon cannot be determined
theorem watermelon_cost_undetermined : 
  ∀ (n : ℕ), n > 0 → ∃ (cost : ℕ), cost > 0 ∧ n * cost = watermelon_spent :=
sorry

end NUMINAMATH_CALUDE_watermelon_cost_undetermined_l2989_298975


namespace NUMINAMATH_CALUDE_round_robin_six_teams_l2989_298977

/-- The number of matches in a round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 6 teams, 15 matches are played -/
theorem round_robin_six_teams :
  num_matches 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_six_teams_l2989_298977


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2989_298928

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 121 * Real.pi) :
  A = Real.pi * r^2 → r = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2989_298928


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l2989_298970

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x + 3 / (x^3) ≥ 12 :=
by sorry

theorem equality_condition : 9 * 1 + 3 / (1^3) = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l2989_298970


namespace NUMINAMATH_CALUDE_integer_condition_l2989_298910

theorem integer_condition (m k n : ℕ) (h1 : 0 < m) (h2 : 0 < k) (h3 : 0 < n)
  (h4 : k < n - 1) (h5 : m ≤ n) :
  ∃ z : ℤ, (n - 3 * k + m : ℚ) / (k + m : ℚ) * (n.choose k : ℚ) = z ↔ 
  ∃ t : ℕ, 2 * m = t * (k + m) :=
by sorry

end NUMINAMATH_CALUDE_integer_condition_l2989_298910


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2989_298960

theorem x_squared_plus_reciprocal (x : ℝ) (h : 15 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2989_298960


namespace NUMINAMATH_CALUDE_geometric_progression_formula_l2989_298903

/-- A geometric progression with positive terms, where a₁ = 1 and a₂ + a₃ = 6 -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) ∧
  a 1 = 1 ∧
  a 2 + a 3 = 6

/-- The general term of the geometric progression is 2^(n-1) -/
theorem geometric_progression_formula (a : ℕ → ℝ) (h : GeometricProgression a) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_formula_l2989_298903


namespace NUMINAMATH_CALUDE_white_line_length_l2989_298952

theorem white_line_length 
  (blue_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : blue_length = 3.33) 
  (h2 : length_difference = 4.33) : 
  blue_length + length_difference = 7.66 := by
sorry

end NUMINAMATH_CALUDE_white_line_length_l2989_298952


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2989_298980

/-- The equation (x - 3)(x - 5) = k - 4x has exactly one real solution if and only if k = 11 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (x - 3) * (x - 5) = k - 4 * x) ↔ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2989_298980


namespace NUMINAMATH_CALUDE_root_approximation_l2989_298911

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_approximation (root : ℕ+) :
  (f root = 0) →
  (f 1 = -2) →
  (f 1.5 = 0.625) →
  (f 1.25 = -0.984) →
  (f 1.375 = -0.260) →
  (f 1.4375 = 0.162) →
  (f 1.40625 = -0.054) →
  ∃ x : ℝ, x ∈ (Set.Ioo 1.375 1.4375) ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_approximation_l2989_298911


namespace NUMINAMATH_CALUDE_parabola_equation_l2989_298943

/-- Given a parabola with axis of symmetry x = -2, its standard form equation is y^2 = 8x. -/
theorem parabola_equation (p : ℝ) (h : p > 0) (axis : ℝ → Prop) : 
  (axis = fun x ↦ x = -2) → 
  (fun x y ↦ y^2 = 2*p*x) = fun x y ↦ y^2 = 8*x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2989_298943


namespace NUMINAMATH_CALUDE_line_equation_proof_l2989_298969

/-- The parabola y^2 = (5/2)x -/
def parabola (x y : ℝ) : Prop := y^2 = (5/2) * x

/-- Point O is the origin (0,0) -/
def O : ℝ × ℝ := (0, 0)

/-- Point through which the line passes -/
def P : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

/-- Two points are perpendicular with respect to the origin -/
def perpendicular_to_origin (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem line_equation_proof :
  ∃ (A B : ℝ × ℝ),
    A ≠ O ∧ B ≠ O ∧
    A ≠ B ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    perpendicular_to_origin A B ∧
    on_line A.1 A.2 ∧
    on_line B.1 B.2 ∧
    on_line P.1 P.2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2989_298969


namespace NUMINAMATH_CALUDE_cubic_root_simplification_l2989_298968

theorem cubic_root_simplification (s : ℝ) : s = 1 / (2 - Real.rpow 3 (1/3)) → s = 2 + Real.rpow 3 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_simplification_l2989_298968


namespace NUMINAMATH_CALUDE_salary_increase_l2989_298948

-- Define the salary function
def salary (x : ℝ) : ℝ := 60 + 90 * x

-- State the theorem
theorem salary_increase (x : ℝ) :
  salary (x + 1) - salary x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l2989_298948


namespace NUMINAMATH_CALUDE_water_usage_calculation_l2989_298983

/-- Calculates the weekly water usage for baths given the specified parameters. -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (baths_per_week : ℕ) : ℕ :=
  let total_capacity := bucket_capacity * buckets_to_fill
  let water_removed := bucket_capacity * buckets_removed
  let water_per_bath := total_capacity - water_removed
  water_per_bath * baths_per_week

/-- Theorem stating that the weekly water usage is 9240 ounces given the specified parameters. -/
theorem water_usage_calculation :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_calculation_l2989_298983


namespace NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_3_l2989_298930

-- Define the equation
def equation (m x : ℝ) : Prop :=
  m / (x - 4) - (1 - x) / (4 - x) = 0

-- Define what it means for x to be a positive root
def is_positive_root (m x : ℝ) : Prop :=
  equation m x ∧ x > 0

-- Theorem statement
theorem equation_positive_root_implies_m_equals_3 :
  ∀ m : ℝ, (∃ x : ℝ, is_positive_root m x) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_positive_root_implies_m_equals_3_l2989_298930


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l2989_298941

/-- Given two monomials 3x^m*y and -x^3*y^n that are like terms, prove that m - n = 2 -/
theorem like_terms_exponent_difference (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^m * y = -x^3 * y^n) → m - n = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l2989_298941


namespace NUMINAMATH_CALUDE_negation_exactly_one_even_l2989_298965

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ IsEven b ∧ ¬IsEven c) ∨
  (¬IsEven a ∧ ¬IsEven b ∧ IsEven c)

/-- The main theorem stating that the negation of "exactly one even" is equivalent to "all odd or at least two even" -/
theorem negation_exactly_one_even (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ (¬IsEven a ∧ ¬IsEven b ∧ ¬IsEven c) ∨ (IsEven a ∧ IsEven b) ∨ (IsEven a ∧ IsEven c) ∨ (IsEven b ∧ IsEven c) :=
sorry


end NUMINAMATH_CALUDE_negation_exactly_one_even_l2989_298965


namespace NUMINAMATH_CALUDE_problem_solution_l2989_298933

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2989_298933


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l2989_298986

/-- The inradius of a right triangle with side lengths 7, 24, and 25 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 7 ∧ b = 24 ∧ c = 25 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l2989_298986


namespace NUMINAMATH_CALUDE_other_coin_denomination_l2989_298996

/-- Proves that given the problem conditions, the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination (total_coins : ℕ) (total_value : ℕ) (twenty_paise_coins : ℕ) 
  (h1 : total_coins = 324)
  (h2 : total_value = 7100)  -- 71 Rs in paise
  (h3 : twenty_paise_coins = 200) :
  (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l2989_298996


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_pow_5_minus_5_pow_4_l2989_298923

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_pow_5_minus_5_pow_4_l2989_298923


namespace NUMINAMATH_CALUDE_fewer_threes_for_hundred_l2989_298993

-- Define a type for arithmetic expressions
inductive Expr
  | num : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

-- Function to count the number of threes in an expression
def countThrees : Expr → Nat
  | Expr.num 3 => 1
  | Expr.num _ => 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Theorem: There exists an expression using fewer than ten threes that evaluates to 100
theorem fewer_threes_for_hundred : ∃ e : Expr, eval e = 100 ∧ countThrees e < 10 := by
  sorry


end NUMINAMATH_CALUDE_fewer_threes_for_hundred_l2989_298993


namespace NUMINAMATH_CALUDE_georgie_prank_ways_l2989_298937

/-- The number of windows in the mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can accomplish the prank -/
def prank_ways : ℕ := num_windows * (num_windows - 1) * (num_windows - 2)

/-- Theorem stating that the number of ways Georgie can accomplish the prank is 336 -/
theorem georgie_prank_ways : prank_ways = 336 := by
  sorry

end NUMINAMATH_CALUDE_georgie_prank_ways_l2989_298937


namespace NUMINAMATH_CALUDE_triangulated_rectangle_has_36_triangles_l2989_298995

/-- Represents a rectangle divided into triangles -/
structure TriangulatedRectangle where
  smallest_triangles : ℕ
  has_isosceles_triangles : Bool
  has_large_right_triangles : Bool

/-- Counts the total number of triangles in a triangulated rectangle -/
def count_triangles (rect : TriangulatedRectangle) : ℕ :=
  sorry

/-- Theorem: A rectangle divided into 16 smallest right triangles contains 36 total triangles -/
theorem triangulated_rectangle_has_36_triangles :
  ∀ (rect : TriangulatedRectangle),
    rect.smallest_triangles = 16 →
    rect.has_isosceles_triangles = true →
    rect.has_large_right_triangles = true →
    count_triangles rect = 36 :=
  sorry

end NUMINAMATH_CALUDE_triangulated_rectangle_has_36_triangles_l2989_298995


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_function_properties_l2989_298908

-- Define the function f(x)
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Minimum value of f(x) is 2
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5 := by sorry

-- Part 2: Inequality holds for x ∈ [0, 1]
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2 := by sorry

-- Combined theorem
theorem function_properties (a : ℝ) :
  ((∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) → a = -1 ∨ a = -5) ∧
  ((∀ x ∈ Set.Icc 0 1, f a x ≤ |5 + x|) → a ∈ Set.Icc (-1) 2) := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_a_range_function_properties_l2989_298908


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2989_298953

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2989_298953


namespace NUMINAMATH_CALUDE_complex_division_equals_i_l2989_298913

theorem complex_division_equals_i : (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equals_i_l2989_298913


namespace NUMINAMATH_CALUDE_sixth_test_score_l2989_298950

def average_score : ℝ := 84
def num_tests : ℕ := 6
def known_scores : List ℝ := [83, 77, 92, 85, 89]

theorem sixth_test_score :
  let total_sum := average_score * num_tests
  let sum_of_known_scores := known_scores.sum
  total_sum - sum_of_known_scores = 78 := by
  sorry

end NUMINAMATH_CALUDE_sixth_test_score_l2989_298950


namespace NUMINAMATH_CALUDE_select_gloves_count_l2989_298982

/-- The number of ways to select 4 gloves from 5 pairs of gloves with exactly one pair of the same color -/
def select_gloves (n : ℕ) : ℕ :=
  let total_pairs := 5
  let select_size := 4
  let pair_combinations := Nat.choose total_pairs 1
  let remaining_gloves := 2 * (total_pairs - 1)
  let other_combinations := Nat.choose remaining_gloves 2
  let same_color_pair := Nat.choose (total_pairs - 1) 1
  pair_combinations * (other_combinations - same_color_pair)

/-- Theorem stating that the number of ways to select 4 gloves from 5 pairs of gloves 
    with exactly one pair of the same color is 120 -/
theorem select_gloves_count : select_gloves 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_gloves_count_l2989_298982


namespace NUMINAMATH_CALUDE_lottery_winning_probability_l2989_298989

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def drawnWinnerBallCount : ℕ := 6

theorem lottery_winning_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / (winnerBallCount.choose drawnWinnerBallCount) = 1 / 476721000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l2989_298989


namespace NUMINAMATH_CALUDE_books_sold_and_remaining_l2989_298934

/-- Given that a person sells 45 books and has 6 books remaining, prove that they initially had 51 books. -/
theorem books_sold_and_remaining (books_sold : ℕ) (books_remaining : ℕ) : 
  books_sold = 45 → books_remaining = 6 → books_sold + books_remaining = 51 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_and_remaining_l2989_298934


namespace NUMINAMATH_CALUDE_fraction_inequality_l2989_298901

theorem fraction_inequality (x : ℝ) : (x + 8) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2989_298901


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l2989_298909

/-- Represents the number of acrobats in the zoo. -/
def acrobats : ℕ := 5

/-- Represents the number of elephants in the zoo. -/
def elephants : ℕ := sorry

/-- Represents the number of camels in the zoo. -/
def camels : ℕ := sorry

/-- The total number of legs in the zoo. -/
def total_legs : ℕ := 58

/-- The total number of heads in the zoo. -/
def total_heads : ℕ := 17

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  acrobats * 2 + elephants * 4 + camels * 4 = total_legs ∧
  acrobats + elephants + camels = total_heads :=
by sorry

end NUMINAMATH_CALUDE_acrobats_count_correct_l2989_298909


namespace NUMINAMATH_CALUDE_tamika_drove_farther_l2989_298945

-- Define the given conditions
def tamika_time : ℝ := 8
def tamika_speed : ℝ := 45
def logan_time : ℝ := 5
def logan_speed : ℝ := 55

-- Define the distance calculation function
def distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

-- Theorem statement
theorem tamika_drove_farther : 
  distance tamika_time tamika_speed - distance logan_time logan_speed = 85 := by
  sorry

end NUMINAMATH_CALUDE_tamika_drove_farther_l2989_298945


namespace NUMINAMATH_CALUDE_horner_V₃_eq_9_l2989_298947

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 3x^4 + 7x^3 - 9x^2 + 4x - 10 -/
def f : List ℚ := [2, -3, 7, -9, 4, -10]

/-- V₃ in Horner's method for f(x) at x = 2 -/
def V₃ : ℚ := horner [2, -3, 7] 2

theorem horner_V₃_eq_9 : V₃ = 9 := by
  sorry

end NUMINAMATH_CALUDE_horner_V₃_eq_9_l2989_298947


namespace NUMINAMATH_CALUDE_sin_cos_power_six_bounds_l2989_298978

theorem sin_cos_power_six_bounds :
  ∀ x : ℝ, (1 : ℝ) / 4 ≤ Real.sin x ^ 6 + Real.cos x ^ 6 ∧
            Real.sin x ^ 6 + Real.cos x ^ 6 ≤ 1 ∧
            (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 6 = (1 : ℝ) / 4) ∧
            (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_power_six_bounds_l2989_298978


namespace NUMINAMATH_CALUDE_pen_distribution_l2989_298961

theorem pen_distribution (total_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) :
  total_pencils = 928 →
  num_students = 16 →
  total_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 16 * k :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l2989_298961


namespace NUMINAMATH_CALUDE_rectangle_area_constant_l2989_298959

theorem rectangle_area_constant (d : ℝ) (h : d > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w / l = 3 / 5 ∧ w ^ 2 + l ^ 2 = (10 * d) ^ 2 ∧ w * l = (750 / 17) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_constant_l2989_298959


namespace NUMINAMATH_CALUDE_polynomial_nonzero_coeffs_l2989_298936

/-- A polynomial has at least n+1 nonzero coefficients if its degree is at least n -/
def HasAtLeastNPlusOneNonzeroCoeffs (p : Polynomial ℝ) (n : ℕ) : Prop :=
  (Finset.filter (· ≠ 0) p.support).card ≥ n + 1

/-- The main theorem statement -/
theorem polynomial_nonzero_coeffs
  (a : ℝ) (k : ℕ) (Q : Polynomial ℝ) 
  (ha : a ≠ 0) (hQ : Q ≠ 0) :
  let W := (Polynomial.X - Polynomial.C a)^k * Q
  HasAtLeastNPlusOneNonzeroCoeffs W k := by
sorry

end NUMINAMATH_CALUDE_polynomial_nonzero_coeffs_l2989_298936


namespace NUMINAMATH_CALUDE_ping_pong_practice_time_l2989_298990

theorem ping_pong_practice_time 
  (total_students : ℕ) 
  (practicing_simultaneously : ℕ) 
  (total_time : ℕ) 
  (h1 : total_students = 5)
  (h2 : practicing_simultaneously = 2)
  (h3 : total_time = 90) :
  (total_time * practicing_simultaneously) / total_students = 36 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_practice_time_l2989_298990


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2989_298992

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) / (3 - Complex.I) - Complex.I / (3 + Complex.I) = (1 + Complex.I) / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2989_298992


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2989_298922

/-- Given a rectangle with perimeter 60 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 162/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 60) →  -- Perimeter condition
  (length = (5/2) * width) →     -- Ratio condition
  Real.sqrt (length^2 + width^2) = 162/7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2989_298922


namespace NUMINAMATH_CALUDE_p_is_converse_of_r_l2989_298967

-- Define propositions as functions from some type α to Prop
variable {α : Type}
variable (p q r : α → Prop)

-- Define the relationships between p, q, and r
axiom contrapositive : (∀ x, p x → q x) ↔ (∀ x, ¬q x → ¬p x)
axiom negation : (∀ x, q x) ↔ (∀ x, ¬r x)

-- Theorem to prove
theorem p_is_converse_of_r : (∀ x, p x → r x) ↔ (∀ x, r x → p x) := by sorry

end NUMINAMATH_CALUDE_p_is_converse_of_r_l2989_298967


namespace NUMINAMATH_CALUDE_problem_solution_l2989_298972

theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (eq1 : a * Real.sqrt a + b * Real.sqrt b = 183)
  (eq2 : a * Real.sqrt b + b * Real.sqrt a = 182) :
  9 / 5 * (a + b) = 657 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2989_298972


namespace NUMINAMATH_CALUDE_money_division_l2989_298905

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * p = 7 * q →
  7 * q = 12 * r →
  q - p = 4500 →
  r - q = 4500 →
  q - p = 3600 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2989_298905


namespace NUMINAMATH_CALUDE_contrapositive_square_inequality_l2989_298985

theorem contrapositive_square_inequality (x y : ℝ) :
  (¬(x > y) → ¬(x^2 > y^2)) ↔ (x ≤ y → x^2 ≤ y^2) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_square_inequality_l2989_298985


namespace NUMINAMATH_CALUDE_set_intersection_example_l2989_298944

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∩ B = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l2989_298944


namespace NUMINAMATH_CALUDE_irrational_among_given_numbers_l2989_298931

theorem irrational_among_given_numbers :
  let a : ℝ := -1/7
  let b : ℝ := Real.sqrt 11
  let c : ℝ := 0.3
  let d : ℝ := Real.sqrt 25
  Irrational b ∧ ¬(Irrational a ∨ Irrational c ∨ Irrational d) := by
  sorry

end NUMINAMATH_CALUDE_irrational_among_given_numbers_l2989_298931


namespace NUMINAMATH_CALUDE_square_of_97_l2989_298958

theorem square_of_97 : 97^2 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l2989_298958


namespace NUMINAMATH_CALUDE_power_of_32_l2989_298927

theorem power_of_32 (n : ℕ) : 
  2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n → n = 81 := by
  sorry

end NUMINAMATH_CALUDE_power_of_32_l2989_298927


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2989_298971

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 3 * (T - 3 * N)) : T / N = 4 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2989_298971


namespace NUMINAMATH_CALUDE_snow_probability_l2989_298938

theorem snow_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^5 = 1023/1024 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l2989_298938


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2989_298951

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) :
  ({x : ℝ | f a x ≤ 2*x} = {x : ℝ | x ≥ 1}) → (a = 0 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2989_298951


namespace NUMINAMATH_CALUDE_different_color_prob_l2989_298919

def bag_prob (p_red_red p_white_white : ℚ) : Prop :=
  p_red_red = 2/15 ∧ p_white_white = 1/3

theorem different_color_prob (p_red_red p_white_white : ℚ) 
  (h : bag_prob p_red_red p_white_white) : 
  1 - (p_red_red + p_white_white) = 8/15 :=
sorry

end NUMINAMATH_CALUDE_different_color_prob_l2989_298919


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2989_298929

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2989_298929


namespace NUMINAMATH_CALUDE_book_price_calculation_l2989_298974

/-- Given a book with a suggested retail price, this theorem proves that
    if the marked price is 60% of the suggested retail price, and a customer
    pays 60% of the marked price, then the customer pays 36% of the
    suggested retail price. -/
theorem book_price_calculation (suggested_retail_price : ℝ) :
  let marked_price := 0.6 * suggested_retail_price
  let customer_price := 0.6 * marked_price
  customer_price = 0.36 * suggested_retail_price := by
  sorry

#check book_price_calculation

end NUMINAMATH_CALUDE_book_price_calculation_l2989_298974


namespace NUMINAMATH_CALUDE_solve_equation_l2989_298915

theorem solve_equation (x : ℝ) : (x ^ 3).sqrt = 9 * (81 ^ (1 / 9 : ℝ)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2989_298915


namespace NUMINAMATH_CALUDE_compute_expression_l2989_298924

theorem compute_expression : 
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2/3) = 41.65 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l2989_298924


namespace NUMINAMATH_CALUDE_trimmed_square_area_l2989_298900

/-- The area of a rectangle formed by trimming a square --/
theorem trimmed_square_area (original_side : ℝ) (trim1 : ℝ) (trim2 : ℝ) 
  (h1 : original_side = 18)
  (h2 : trim1 = 4)
  (h3 : trim2 = 3) :
  (original_side - trim1) * (original_side - trim2) = 210 := by
sorry

end NUMINAMATH_CALUDE_trimmed_square_area_l2989_298900


namespace NUMINAMATH_CALUDE_correct_division_incorrect_others_l2989_298987

theorem correct_division_incorrect_others :
  ((-8) / (-4) = 8 / 4) ∧
  ¬((-5) + 9 = -(9 - 5)) ∧
  ¬(7 - (-10) = 7 - 10) ∧
  ¬((-5) * 0 = -5) := by
  sorry

end NUMINAMATH_CALUDE_correct_division_incorrect_others_l2989_298987


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l2989_298935

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) :
  n = 12 → k = 4 → Nat.choose n k = 495 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilaterals_from_circle_points_l2989_298935


namespace NUMINAMATH_CALUDE_smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l2989_298979

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 2 = 7 * k) → n ≥ 2 :=
by
  sorry

theorem two_boxes_satisfies : 
  ∃ k : ℕ, 15 * 2 - 2 = 7 * k :=
by
  sorry

theorem two_is_smallest : 
  ∀ m : ℕ, m < 2 → ¬(∃ k : ℕ, 15 * m - 2 = 7 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l2989_298979


namespace NUMINAMATH_CALUDE_consecutive_fibonacci_coprime_l2989_298973

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem consecutive_fibonacci_coprime (n : ℕ) (h : n ≥ 1) : 
  Nat.gcd (fib n) (fib (n - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_fibonacci_coprime_l2989_298973


namespace NUMINAMATH_CALUDE_student_uniform_cost_l2989_298955

/-- Calculates the total cost for a student's uniforms including discounts, fees, and taxes -/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (socks_cost : ℚ) (shoes_cost : ℚ) 
  (uniform_fee : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let jacket_cost := 3 * shirt_cost
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost + jacket_cost + shoes_cost
  let subtotal := num_uniforms * uniform_cost * (1 - discount_rate) + uniform_fee
  subtotal * (1 + tax_rate)

/-- The total cost for a student buying 5 uniforms is $1117.77 -/
theorem student_uniform_cost : 
  uniform_cost 5 20 3 40 15 (10/100) (6/100) = 1117.77 := by
  sorry

end NUMINAMATH_CALUDE_student_uniform_cost_l2989_298955


namespace NUMINAMATH_CALUDE_probability_of_two_pairs_l2989_298954

def number_of_dice : ℕ := 7
def sides_per_die : ℕ := 6

def total_outcomes : ℕ := sides_per_die ^ number_of_dice

def ways_to_choose_pair_numbers : ℕ := Nat.choose 6 2
def ways_to_choose_dice_for_pairs : ℕ := Nat.choose number_of_dice 4
def ways_to_arrange_pairs : ℕ := 6  -- 4! / (2! * 2!)
def ways_to_choose_remaining_numbers : ℕ := 4 * 3 * 2

def successful_outcomes : ℕ := 
  ways_to_choose_pair_numbers * ways_to_choose_dice_for_pairs * 
  ways_to_arrange_pairs * ways_to_choose_remaining_numbers

theorem probability_of_two_pairs (h : successful_outcomes = 151200 ∧ total_outcomes = 279936) :
  (successful_outcomes : ℚ) / total_outcomes = 175 / 324 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_pairs_l2989_298954


namespace NUMINAMATH_CALUDE_mark_speeding_ticket_cost_l2989_298998

/-- Calculate the total cost of Mark's speeding ticket --/
def speeding_ticket_cost (base_fine : ℕ) (fine_increase_per_mph : ℕ) 
  (mark_speed : ℕ) (speed_limit : ℕ) (court_costs : ℕ) 
  (lawyer_fee_per_hour : ℕ) (lawyer_hours : ℕ) : ℕ := 
  let speed_difference := mark_speed - speed_limit
  let speed_fine := base_fine + fine_increase_per_mph * speed_difference
  let doubled_fine := 2 * speed_fine
  let total_without_lawyer := doubled_fine + court_costs
  let lawyer_cost := lawyer_fee_per_hour * lawyer_hours
  total_without_lawyer + lawyer_cost

theorem mark_speeding_ticket_cost : 
  speeding_ticket_cost 50 2 75 30 300 80 3 = 820 := by
  sorry

end NUMINAMATH_CALUDE_mark_speeding_ticket_cost_l2989_298998


namespace NUMINAMATH_CALUDE_bcm_hens_count_l2989_298918

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  ↑(total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
  sorry

end NUMINAMATH_CALUDE_bcm_hens_count_l2989_298918


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l2989_298906

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of teachers --/
def num_teachers : ℕ := 5

/-- Represents the constraint that each school must have at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- The function that calculates the number of ways to distribute teachers --/
def distribute_teachers : ℕ := sorry

/-- The theorem stating that the number of ways to distribute teachers is 150 --/
theorem distribute_teachers_count : distribute_teachers = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l2989_298906


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2989_298956

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2989_298956


namespace NUMINAMATH_CALUDE_prob_queen_of_diamonds_l2989_298949

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)

/-- Represents a specific card -/
structure Card :=
  (rank : String)
  (suit : String)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4 }

/-- Probability of drawing a specific card from a deck -/
def prob_draw_card (d : Deck) (c : Card) : ℚ :=
  1 / d.total_cards

/-- Queen of Diamonds card -/
def queen_of_diamonds : Card :=
  { rank := "Queen",
    suit := "Diamonds" }

/-- Theorem: Probability of drawing Queen of Diamonds from a standard deck is 1/52 -/
theorem prob_queen_of_diamonds :
  prob_draw_card standard_deck queen_of_diamonds = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_queen_of_diamonds_l2989_298949
