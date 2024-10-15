import Mathlib

namespace NUMINAMATH_CALUDE_steven_fruit_difference_l1413_141335

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℕ := steven_apples - steven_peaches

theorem steven_fruit_difference : apple_peach_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_steven_fruit_difference_l1413_141335


namespace NUMINAMATH_CALUDE_chessboard_nail_configuration_l1413_141319

/-- Represents a point on the chessboard --/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Checks if three points are collinear --/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A configuration of 16 points on the chessboard --/
def Configuration := Fin 16 → Point

/-- Predicate to check if a configuration is valid --/
def valid_configuration (config : Configuration) : Prop :=
  (∀ i j k : Fin 16, i ≠ j → j ≠ k → i ≠ k → ¬collinear (config i) (config j) (config k))

theorem chessboard_nail_configuration :
  ∃ (config : Configuration), valid_configuration config :=
sorry

end NUMINAMATH_CALUDE_chessboard_nail_configuration_l1413_141319


namespace NUMINAMATH_CALUDE_student_permutations_l1413_141370

/-- Represents the number of students --/
def n : ℕ := 5

/-- The factorial function --/
def factorial (m : ℕ) : ℕ := 
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The number of permutations of n elements --/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of permutations not in alphabetical order --/
def permutations_not_alphabetical (n : ℕ) : ℕ := permutations n - 1

/-- The number of permutations where two specific elements are consecutive --/
def permutations_consecutive_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

theorem student_permutations :
  (permutations n = 120) ∧
  (permutations_not_alphabetical n = 119) ∧
  (permutations_consecutive_pair n = 48) := by
  sorry

end NUMINAMATH_CALUDE_student_permutations_l1413_141370


namespace NUMINAMATH_CALUDE_spherical_cap_height_theorem_l1413_141301

/-- The height of a spherical cap -/
def spherical_cap_height (R : ℝ) (c : ℝ) : Set ℝ :=
  {h | h = 2*R*(c-1)/c ∨ h = 2*R*(c-2)/(c-1)}

/-- Theorem: The height of a spherical cap with radius R, whose surface area is c times 
    the area of its circular base (c > 1), is either 2R(c-1)/c or 2R(c-2)/(c-1) -/
theorem spherical_cap_height_theorem (R c : ℝ) (hR : R > 0) (hc : c > 1) :
  ∃ h ∈ spherical_cap_height R c,
    (∃ S_cap S_base : ℝ, 
      S_cap = c * S_base ∧
      ((S_cap = 2 * π * R * h ∧ S_base = π * (2*R*h - h^2)) ∨
       (S_cap = 2 * π * R * h + π * (2*R*h - h^2) ∧ S_base = π * (2*R*h - h^2)))) :=
by
  sorry

end NUMINAMATH_CALUDE_spherical_cap_height_theorem_l1413_141301


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l1413_141318

theorem sum_and_equal_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 108)
  (equal_after_changes : a + 8 = b - 4 ∧ b - 4 = 6 * c) :
  b = 724 / 13 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l1413_141318


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_conditions_l1413_141300

/-- Represents the equation (x^2)/(2m) - (y^2)/(m-6) = 1 as an ellipse with foci on the y-axis -/
def proposition_p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ x y : ℝ, x^2 / (2*m) - y^2 / (m-6) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- Represents the equation (x^2)/(m+1) + (y^2)/(m-1) = 1 as a hyperbola -/
def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / (m+1) + y^2 / (m-1) = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1)

/-- Theorem stating the conditions for proposition_p and proposition_q -/
theorem ellipse_hyperbola_conditions (m : ℝ) :
  (proposition_p m ↔ 0 < m ∧ m < 2) ∧
  (¬proposition_q m ↔ m ≤ -1 ∨ m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_conditions_l1413_141300


namespace NUMINAMATH_CALUDE_tax_rate_as_percent_l1413_141321

/-- Given a tax rate of $82 per $100.00, prove that the tax rate expressed as a percent is 82%. -/
theorem tax_rate_as_percent (tax_amount : ℝ) (base_amount : ℝ) :
  tax_amount = 82 ∧ base_amount = 100 →
  (tax_amount / base_amount) * 100 = 82 := by
sorry

end NUMINAMATH_CALUDE_tax_rate_as_percent_l1413_141321


namespace NUMINAMATH_CALUDE_third_of_ten_given_metaphorical_quarter_l1413_141346

-- Define the metaphorical relationship
def metaphorical_quarter (x : ℚ) : ℚ := x / 5

-- Define the actual third
def actual_third (x : ℚ) : ℚ := x / 3

-- Theorem statement
theorem third_of_ten_given_metaphorical_quarter :
  metaphorical_quarter 20 = 4 → actual_third 10 = 8/3 :=
by
  sorry

end NUMINAMATH_CALUDE_third_of_ten_given_metaphorical_quarter_l1413_141346


namespace NUMINAMATH_CALUDE_egyptian_fraction_odd_divisor_l1413_141320

theorem egyptian_fraction_odd_divisor (n : ℕ) (h_n : n > 1) (h_odd : Odd n) :
  (∃ x y : ℕ, (4 : ℚ) / n = 1 / x + 1 / y) ↔
  (∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∃ k : ℕ, p = 4 * k - 1) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_odd_divisor_l1413_141320


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1413_141354

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64/9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l1413_141354


namespace NUMINAMATH_CALUDE_min_value_when_a_neg_one_max_value_case1_max_value_case2_l1413_141391

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem for the minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∀ x ∈ Set.Icc 0 2, f (-1) x ≥ -2 :=
sorry

-- Theorem for the maximum value when -2 ≤ a ≤ -1/4
theorem max_value_case1 (a : ℝ) (h : a ∈ Set.Icc (-2) (-1/4)) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ -1 / (4 * a) :=
sorry

-- Theorem for the maximum value when -1/4 < a ≤ 0
theorem max_value_case2 (a : ℝ) (h : a ∈ Set.Ioo (-1/4) 0) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ 4 * a + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_neg_one_max_value_case1_max_value_case2_l1413_141391


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1413_141307

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1375 →
  L = 1632 →
  L = 6 * S + R →
  R < S →
  R = 90 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1413_141307


namespace NUMINAMATH_CALUDE_exponential_function_property_l1413_141333

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∀ (x₁ x₂ : ℝ), (fun x ↦ a^x) (x₁ + x₂) = (fun x ↦ a^x) x₁ * (fun x ↦ a^x) x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_property_l1413_141333


namespace NUMINAMATH_CALUDE_work_rate_problem_l1413_141355

theorem work_rate_problem (x y k : ℝ) : 
  x = k * y → 
  y = 1 / 80 → 
  x + y = 1 / 20 → 
  k = 3 := by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l1413_141355


namespace NUMINAMATH_CALUDE_train_carriage_seats_l1413_141339

theorem train_carriage_seats : 
  ∀ (seats_per_carriage : ℕ),
  (3 * 4 * (seats_per_carriage + 10) = 420) →
  seats_per_carriage = 25 := by
sorry

end NUMINAMATH_CALUDE_train_carriage_seats_l1413_141339


namespace NUMINAMATH_CALUDE_square_of_negative_two_times_a_cubed_l1413_141343

theorem square_of_negative_two_times_a_cubed (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_times_a_cubed_l1413_141343


namespace NUMINAMATH_CALUDE_f_not_prime_l1413_141398

def f (n : ℕ+) : ℤ := n.val^4 - 400 * n.val^2 + 600

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l1413_141398


namespace NUMINAMATH_CALUDE_cube_collinear_points_l1413_141356

/-- Represents a point in a cube -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | CubeCenter

/-- Represents a line in a cube -/
structure CubeLine where
  points : Finset CubePoint
  collinear : points.card = 3

/-- The set of all points in the cube -/
def cubePoints : Finset CubePoint := sorry

/-- The set of all lines in the cube -/
def cubeLines : Finset CubeLine := sorry

/-- The number of vertices in a cube -/
def numVertices : Nat := 8

/-- The number of edge midpoints in a cube -/
def numEdgeMidpoints : Nat := 12

/-- The number of face centers in a cube -/
def numFaceCenters : Nat := 6

/-- The number of cube centers in a cube -/
def numCubeCenters : Nat := 1

theorem cube_collinear_points :
  cubePoints.card = numVertices + numEdgeMidpoints + numFaceCenters + numCubeCenters ∧
  cubeLines.card = 49 := by sorry

end NUMINAMATH_CALUDE_cube_collinear_points_l1413_141356


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1413_141324

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  total_marks = 120 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1413_141324


namespace NUMINAMATH_CALUDE_mitzi_amusement_park_money_l1413_141305

def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23
def remaining_money : ℕ := 9

theorem mitzi_amusement_park_money :
  ticket_cost + food_cost + tshirt_cost + remaining_money = 75 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_amusement_park_money_l1413_141305


namespace NUMINAMATH_CALUDE_original_number_proof_l1413_141393

theorem original_number_proof : 
  ∃ N : ℕ, N ≥ 118 ∧ (N - 31) % 87 = 0 ∧ ∀ M : ℕ, M < N → (M - 31) % 87 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1413_141393


namespace NUMINAMATH_CALUDE_jean_kept_fraction_l1413_141386

theorem jean_kept_fraction (total : ℕ) (janet_got : ℕ) (janet_fraction : ℚ) :
  total = 60 →
  janet_got = 10 →
  janet_fraction = 1/4 →
  (total - (janet_got / janet_fraction)) / total = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_jean_kept_fraction_l1413_141386


namespace NUMINAMATH_CALUDE_min_operations_to_exceed_1000_l1413_141353

-- Define the operation of repeated squaring
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

-- State the theorem
theorem min_operations_to_exceed_1000 :
  (∃ n : ℕ, repeated_square 3 n > 1000) ∧
  (∀ m : ℕ, repeated_square 3 m > 1000 → m ≥ 3) ∧
  (repeated_square 3 3 > 1000) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_exceed_1000_l1413_141353


namespace NUMINAMATH_CALUDE_inequality_proof_l1413_141332

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1413_141332


namespace NUMINAMATH_CALUDE_b_parallel_same_direction_as_a_l1413_141352

/-- Two vectors are parallel and in the same direction if one is a positive scalar multiple of the other -/
def parallel_same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Given vector a -/
def a : ℝ × ℝ := (1, -1)

/-- Vector b to be proven parallel and in the same direction as a -/
def b : ℝ × ℝ := (2, -2)

/-- Theorem stating that b is parallel and in the same direction as a -/
theorem b_parallel_same_direction_as_a : parallel_same_direction a b := by
  sorry

end NUMINAMATH_CALUDE_b_parallel_same_direction_as_a_l1413_141352


namespace NUMINAMATH_CALUDE_problem_solution_l1413_141373

def f (a x : ℝ) := |a*x - 1| - (a - 1) * |x|

theorem problem_solution :
  (∀ x : ℝ, f 2 x > 2 ↔ x < -1 ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 1 2, f a x < a + 1) → a ≥ 2/5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1413_141373


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l1413_141388

theorem eight_digit_divisibility (n : ℕ) (h : 1000 ≤ n ∧ n < 10000) :
  ∃ k : ℕ, 10001 * n = k * (10000 * n + n) :=
sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l1413_141388


namespace NUMINAMATH_CALUDE_manager_team_selection_l1413_141326

theorem manager_team_selection : Nat.choose 10 6 = 210 := by
  sorry

end NUMINAMATH_CALUDE_manager_team_selection_l1413_141326


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1413_141311

theorem final_sum_after_operations (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1413_141311


namespace NUMINAMATH_CALUDE_laundry_detergent_problem_l1413_141382

def standard_weight : ℕ := 450
def price_per_bag : ℕ := 3
def weight_deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def qualification_criterion : ℤ → Bool := λ x => x.natAbs ≤ 4

theorem laundry_detergent_problem :
  let total_weight := (weight_deviations.sum + standard_weight * weight_deviations.length : ℤ)
  let qualified_bags := weight_deviations.filter qualification_criterion
  let total_sales := qualified_bags.length * price_per_bag
  (total_weight = 3598 ∧ total_sales = 18) := by sorry

end NUMINAMATH_CALUDE_laundry_detergent_problem_l1413_141382


namespace NUMINAMATH_CALUDE_circle_bounded_area_l1413_141380

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def boundedArea (c1 c2 : Circle) : ℝ :=
  sorry

theorem circle_bounded_area :
  let c1 : Circle := { center := (5, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 5 }
  boundedArea c1 c2 = 50 - 12.5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_bounded_area_l1413_141380


namespace NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l1413_141331

/-- Represents a chromosome in a human cell -/
structure Chromosome where
  parent : Bool  -- true for paternal, false for maternal

/-- Represents a pair of homologous chromosomes -/
structure HomologousPair where
  chromosome1 : Chromosome
  chromosome2 : Chromosome

/-- Represents a human cell -/
structure HumanCell where
  chromosomePairs : List HomologousPair

/-- Axiom: Humans reproduce sexually -/
axiom human_sexual_reproduction : True

/-- Axiom: Fertilization involves fusion of sperm and egg cells -/
axiom fertilization_fusion : True

/-- Axiom: Meiosis occurs in formation of reproductive cells -/
axiom meiosis_in_reproduction : True

/-- Axiom: Zygote chromosome count is restored to somatic cell count -/
axiom zygote_chromosome_restoration : True

/-- Axiom: Half of zygote chromosomes from sperm, half from egg -/
axiom zygote_chromosome_origin : True

/-- Theorem: Each pair of homologous chromosomes is provided by both parents -/
theorem homologous_pair_from_both_parents (cell : HumanCell) : 
  ∀ pair ∈ cell.chromosomePairs, pair.chromosome1.parent ≠ pair.chromosome2.parent := by
  sorry


end NUMINAMATH_CALUDE_homologous_pair_from_both_parents_l1413_141331


namespace NUMINAMATH_CALUDE_article_cost_price_l1413_141374

theorem article_cost_price (original_selling_price original_cost_price new_selling_price new_cost_price : ℝ) :
  original_selling_price = 1.25 * original_cost_price →
  new_cost_price = 0.8 * original_cost_price →
  new_selling_price = original_selling_price - 12.60 →
  new_selling_price = 1.3 * new_cost_price →
  original_cost_price = 60 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l1413_141374


namespace NUMINAMATH_CALUDE_set_operations_l1413_141330

def A : Set ℝ := {x | x < -2 ∨ x > 5}
def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem set_operations :
  (Aᶜ : Set ℝ) = {x | -2 ≤ x ∧ x ≤ 5} ∧
  (Bᶜ : Set ℝ) = {x | x < 4 ∨ x > 6} ∧
  (A ∩ B : Set ℝ) = {x | 5 < x ∧ x ≤ 6} ∧
  ((A ∪ B)ᶜ : Set ℝ) = {x | -2 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1413_141330


namespace NUMINAMATH_CALUDE_treaty_to_university_founding_l1413_141385

theorem treaty_to_university_founding (treaty_day : Nat) (founding_day : Nat) : 
  treaty_day % 7 = 2 → -- Tuesday is represented as 2 (0 = Sunday, 1 = Monday, etc.)
  founding_day = treaty_day + 1204 →
  founding_day % 7 = 5 -- Friday is represented as 5
  := by sorry

end NUMINAMATH_CALUDE_treaty_to_university_founding_l1413_141385


namespace NUMINAMATH_CALUDE_ski_lift_time_l1413_141337

theorem ski_lift_time (ski_down_time : ℝ) (num_trips : ℕ) (total_time : ℝ) 
  (h1 : ski_down_time = 5)
  (h2 : num_trips = 6)
  (h3 : total_time = 120) : 
  (total_time - num_trips * ski_down_time) / num_trips = 15 := by
sorry

end NUMINAMATH_CALUDE_ski_lift_time_l1413_141337


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1413_141369

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b = 2*a ∧ c = 3*a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l1413_141369


namespace NUMINAMATH_CALUDE_unique_solution_l1413_141350

theorem unique_solution : ∃! (x : ℕ+), (1 : ℕ)^(x.val + 2) + 2^(x.val + 1) + 3^(x.val - 1) + 4^x.val = 1170 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1413_141350


namespace NUMINAMATH_CALUDE_sqrt_simplification_l1413_141304

theorem sqrt_simplification : Real.sqrt 32 + Real.sqrt 8 - Real.sqrt 50 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l1413_141304


namespace NUMINAMATH_CALUDE_polyhedron_volume_l1413_141308

/-- The volume of a polyhedron formed by a cube and a tetrahedron -/
theorem polyhedron_volume (cube_side : ℝ) (tetra_base_area : ℝ) (tetra_height : ℝ) :
  cube_side = 2 →
  tetra_base_area = 2 →
  tetra_height = 2 →
  cube_side ^ 3 + (1/3) * tetra_base_area * tetra_height = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l1413_141308


namespace NUMINAMATH_CALUDE_car_trading_profit_l1413_141328

/-- Calculates the profit percentage for a car trading scenario -/
theorem car_trading_profit (original_price : ℝ) (h : original_price > 0) :
  let trader_buy_price := original_price * (1 - 0.2)
  let dealer_buy_price := trader_buy_price * (1 + 0.3)
  let customer_buy_price := dealer_buy_price * (1 + 0.5)
  let trader_final_price := customer_buy_price * (1 - 0.1)
  let profit := trader_final_price - trader_buy_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60.4 := by
sorry


end NUMINAMATH_CALUDE_car_trading_profit_l1413_141328


namespace NUMINAMATH_CALUDE_nabla_problem_l1413_141303

-- Define the operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1413_141303


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l1413_141309

theorem coffee_shop_sales (teas : ℕ) (lattes : ℕ) : 
  teas = 6 → lattes = 4 * teas + 8 → lattes = 32 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l1413_141309


namespace NUMINAMATH_CALUDE_ellipse_range_theorem_l1413_141351

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / (16/3) = 1

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The expression OP · OQ + MP · MQ -/
def expr (P Q : ℝ × ℝ) : ℝ :=
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

/-- The theorem to be proved -/
theorem ellipse_range_theorem :
  ∀ P Q : ℝ × ℝ,
  is_on_ellipse P.1 P.2 →
  is_on_ellipse Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ expr P Q ∧ expr P Q ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_theorem_l1413_141351


namespace NUMINAMATH_CALUDE_dot_product_range_l1413_141344

/-- Given a fixed point M(0, 4) and a point P(x, y) on the circle x^2 + y^2 = 4,
    the dot product of MP⃗ and OP⃗ is bounded between -4 and 12. -/
theorem dot_product_range (x y : ℝ) : 
  x^2 + y^2 = 4 → 
  -4 ≤ x * x + y * y - 4 * y ∧ x * x + y * y - 4 * y ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_range_l1413_141344


namespace NUMINAMATH_CALUDE_circle_symmetric_points_line_l1413_141316

/-- Circle with center (-1, 3) and radius 3 -/
def Circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9

/-- Line with equation x + my + 4 = 0 -/
def Line (m x y : ℝ) : Prop := x + m * y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def SymmetricPoints (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  Line m ((P.1 + Q.1) / 2) ((P.2 + Q.2) / 2)

theorem circle_symmetric_points_line (m : ℝ) :
  (∃ P Q : ℝ × ℝ, Circle P.1 P.2 ∧ Circle Q.1 Q.2 ∧ SymmetricPoints P Q m) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_points_line_l1413_141316


namespace NUMINAMATH_CALUDE_glove_ratio_for_43_participants_l1413_141348

/-- The ratio of the minimum number of gloves needed to the number of participants -/
def glove_ratio (participants : ℕ) : ℚ :=
  2

theorem glove_ratio_for_43_participants :
  glove_ratio 43 = 2 := by
  sorry

end NUMINAMATH_CALUDE_glove_ratio_for_43_participants_l1413_141348


namespace NUMINAMATH_CALUDE_shift_left_3_units_l1413_141390

-- Define the original function
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the shifted function
def g (x : ℝ) : ℝ := (x + 2)^2

-- Define the shift operation
def shift (h : ℝ → ℝ) (s : ℝ) : ℝ → ℝ := fun x ↦ h (x + s)

-- Theorem statement
theorem shift_left_3_units :
  shift f 3 = g := by sorry

end NUMINAMATH_CALUDE_shift_left_3_units_l1413_141390


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l1413_141340

theorem number_of_divisors_of_30 : Nat.card {d : ℕ | d > 0 ∧ 30 % d = 0} = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l1413_141340


namespace NUMINAMATH_CALUDE_jack_recycling_earnings_l1413_141313

/-- The amount Jack gets per bottle in dollars -/
def bottle_amount : ℚ := sorry

/-- The amount Jack gets per can in dollars -/
def can_amount : ℚ := 5 / 100

/-- The number of bottles Jack recycled -/
def num_bottles : ℕ := 80

/-- The number of cans Jack recycled -/
def num_cans : ℕ := 140

/-- The total amount Jack made in dollars -/
def total_amount : ℚ := 15

theorem jack_recycling_earnings :
  bottle_amount * num_bottles + can_amount * num_cans = total_amount ∧
  bottle_amount = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_jack_recycling_earnings_l1413_141313


namespace NUMINAMATH_CALUDE_negation_equivalence_l1413_141360

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 - 2*x₀ - 2 > 0) ↔ 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1413_141360


namespace NUMINAMATH_CALUDE_feed_has_greatest_value_l1413_141341

/-- The value of a letter in the alphabet (A to F) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | _ => 0

/-- The value of a word, which is the sum of its letter values -/
def word_value (w : String) : ℕ :=
  w.data.map letter_value |>.sum

/-- The list of words to compare -/
def words : List String := ["BEEF", "FADE", "FEED", "FACE", "DEAF"]

theorem feed_has_greatest_value :
  ∀ w ∈ words, word_value "FEED" ≥ word_value w :=
by sorry

end NUMINAMATH_CALUDE_feed_has_greatest_value_l1413_141341


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1413_141394

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x, f a c x > 0 ↔ 1 < x ∧ x < 3

-- Define the sufficient condition
def sufficient_condition (a c m : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * x + 4 * c > 0 → x + m > 0

-- Define the not necessary condition
def not_necessary_condition (a c m : ℝ) : Prop :=
  ∃ x, x + m > 0 ∧ ¬(a * x^2 + 2 * x + 4 * c > 0)

theorem quadratic_inequality_problem (a c m : ℝ) :
  solution_set a c →
  sufficient_condition a c m →
  not_necessary_condition a c m →
  (a = -1/4 ∧ c = -3/4) ∧ (m ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1413_141394


namespace NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l1413_141368

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 50 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l1413_141368


namespace NUMINAMATH_CALUDE_bank_withdrawal_total_l1413_141334

theorem bank_withdrawal_total (x y : ℕ) : 
  x / 20 + y / 20 = 30 → x + y = 600 := by
  sorry

end NUMINAMATH_CALUDE_bank_withdrawal_total_l1413_141334


namespace NUMINAMATH_CALUDE_farmer_land_usage_l1413_141306

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l1413_141306


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1413_141384

/-- Given a line ax + by + c = 0 where ac > 0 and bc < 0, 
    the line does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (h1 : a * c > 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l1413_141384


namespace NUMINAMATH_CALUDE_celias_rent_l1413_141387

/-- Celia's monthly budget -/
structure MonthlyBudget where
  food : ℕ
  streaming : ℕ
  cellPhone : ℕ
  rent : ℕ
  savings : ℕ

/-- Celia's budget satisfies the given conditions -/
def validBudget (b : MonthlyBudget) : Prop :=
  b.food = 400 ∧
  b.streaming = 30 ∧
  b.cellPhone = 50 ∧
  b.savings = 198 ∧
  b.savings * 10 = b.food + b.streaming + b.cellPhone + b.rent

/-- Theorem: Celia's rent is $1500 -/
theorem celias_rent (b : MonthlyBudget) (h : validBudget b) : b.rent = 1500 := by
  sorry


end NUMINAMATH_CALUDE_celias_rent_l1413_141387


namespace NUMINAMATH_CALUDE_ratio_product_theorem_l1413_141375

theorem ratio_product_theorem (a b c : ℝ) : 
  a / b = 3 / 4 ∧ b / c = 4 / 6 ∧ c = 18 → a * b * c = 1944 := by
  sorry

end NUMINAMATH_CALUDE_ratio_product_theorem_l1413_141375


namespace NUMINAMATH_CALUDE_square_root_of_1_5625_l1413_141329

theorem square_root_of_1_5625 : Real.sqrt 1.5625 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1_5625_l1413_141329


namespace NUMINAMATH_CALUDE_product_of_decimals_l1413_141315

theorem product_of_decimals : (0.4 : ℝ) * 0.6 = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1413_141315


namespace NUMINAMATH_CALUDE_base_conversion_puzzle_l1413_141399

theorem base_conversion_puzzle :
  ∀ (n : ℕ+) (C D : ℕ),
    C < 8 ∧ D < 8 ∧  -- C and D are single digits in base 8
    C < 5 ∧ D < 5 ∧  -- C and D are single digits in base 5
    n = 8 * C + D ∧  -- base 8 representation
    n = 5 * D + C    -- base 5 representation
    → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_puzzle_l1413_141399


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l1413_141302

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14620 := by
sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l1413_141302


namespace NUMINAMATH_CALUDE_square_rotation_cylinder_volume_l1413_141383

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem square_rotation_cylinder_volume (side_length : ℝ) (volume : ℝ) :
  side_length = 10 →
  volume = Real.pi * (side_length / 2)^2 * side_length →
  volume = 250 * Real.pi :=
by
  sorry

#check square_rotation_cylinder_volume

end NUMINAMATH_CALUDE_square_rotation_cylinder_volume_l1413_141383


namespace NUMINAMATH_CALUDE_camper_difference_is_nine_l1413_141357

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The difference in the number of campers rowing in the afternoon compared to the morning -/
def camper_difference : ℕ := afternoon_campers - morning_campers

/-- Theorem stating that the difference in campers is 9 -/
theorem camper_difference_is_nine : camper_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_camper_difference_is_nine_l1413_141357


namespace NUMINAMATH_CALUDE_triangle_inequality_l1413_141376

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) :
  x > 0 → y > 0 → z > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C ≤ 
  (1/2) * (x*y + y*z + z*x) * Real.sqrt ((x + y + z)/(x*y*z)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1413_141376


namespace NUMINAMATH_CALUDE_trophy_cost_l1413_141345

def total_cost (a b : ℕ) : ℚ := (a * 1000 + 999 + b) / 10

theorem trophy_cost (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : (a * 1000 + 999 + b) % 8 = 0) 
  (h4 : (a + 9 + 9 + 9 + b) % 9 = 0) : 
  (total_cost a b) / 72 = 11.11 := by
  sorry

end NUMINAMATH_CALUDE_trophy_cost_l1413_141345


namespace NUMINAMATH_CALUDE_hall_length_l1413_141362

/-- A rectangular hall with breadth two-thirds of its length and area 2400 sq meters has a length of 60 meters. -/
theorem hall_length (length breadth : ℝ) : 
  breadth = (2 / 3) * length →
  length * breadth = 2400 →
  length = 60 := by
sorry

end NUMINAMATH_CALUDE_hall_length_l1413_141362


namespace NUMINAMATH_CALUDE_negation_equivalence_l1413_141364

theorem negation_equivalence :
  ¬(∃ x : ℝ, x > 1 ∧ x^2 - x > 0) ↔ (∀ x : ℝ, x > 1 → x^2 - x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1413_141364


namespace NUMINAMATH_CALUDE_probability_no_adjacent_seating_l1413_141317

def num_chairs : ℕ := 9
def num_people : ℕ := 4

def total_arrangements (n m : ℕ) : ℕ :=
  (n - 1) * (n - 2) * (n - 3)

def favorable_arrangements (n m : ℕ) : ℕ :=
  (n - m + 1) * m

theorem probability_no_adjacent_seating :
  (favorable_arrangements num_chairs num_people : ℚ) / 
  (total_arrangements num_chairs num_people : ℚ) = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_seating_l1413_141317


namespace NUMINAMATH_CALUDE_bread_roll_flour_usage_l1413_141392

theorem bread_roll_flour_usage
  (original_rolls : ℕ) (original_flour_per_roll : ℚ)
  (new_rolls : ℕ) (new_flour_per_roll : ℚ)
  (h1 : original_rolls = 24)
  (h2 : original_flour_per_roll = 1 / 8)
  (h3 : new_rolls = 16)
  (h4 : original_rolls * original_flour_per_roll = new_rolls * new_flour_per_roll) :
  new_flour_per_roll = 3 / 16 := by
sorry

end NUMINAMATH_CALUDE_bread_roll_flour_usage_l1413_141392


namespace NUMINAMATH_CALUDE_equation_solutions_l1413_141363

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧ 
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 6 ∧ 
    (x₁ + 3)*(x₁ - 3) = 3*(x₁ + 3) ∧ (x₂ + 3)*(x₂ - 3) = 3*(x₂ + 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1413_141363


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1413_141358

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1413_141358


namespace NUMINAMATH_CALUDE_rational_function_value_at_two_l1413_141312

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_neg_one_neg_two : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value_at_two (f : RationalFunction) : f.p 2 / f.q 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_two_l1413_141312


namespace NUMINAMATH_CALUDE_polynomial_identity_l1413_141359

/-- Given a polynomial P(m) that satisfies P(m) - 3m = 5m^2 - 3m - 5,
    prove that P(m) = 5m^2 - 5 -/
theorem polynomial_identity (m : ℝ) (P : ℝ → ℝ) 
    (h : ∀ m, P m - 3*m = 5*m^2 - 3*m - 5) : 
    P m = 5*m^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1413_141359


namespace NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l1413_141396

theorem nested_sqrt_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l1413_141396


namespace NUMINAMATH_CALUDE_function_property_l1413_141342

theorem function_property (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x - f y) - f (f x) = -f y - 1) →
  (∀ x : ℤ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1413_141342


namespace NUMINAMATH_CALUDE_vector_dot_product_result_l1413_141327

theorem vector_dot_product_result :
  let a : ℝ × ℝ := (Real.cos (45 * π / 180), Real.sin (45 * π / 180))
  let b : ℝ × ℝ := (Real.cos (15 * π / 180), Real.sin (15 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_dot_product_result_l1413_141327


namespace NUMINAMATH_CALUDE_fairGame_l1413_141395

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  yellow : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (count : BallCount) : ℕ :=
  count.yellow + count.black + count.red

/-- Determines if the game is fair given the current ball count -/
def isFair (count : BallCount) : Prop :=
  count.yellow = count.black

/-- Represents the action of replacing black balls with yellow balls -/
def replaceBalls (count : BallCount) (n : ℕ) : BallCount :=
  { yellow := count.yellow + n
    black := count.black - n
    red := count.red }

/-- The main theorem stating that replacing 4 black balls with yellow balls makes the game fair -/
theorem fairGame (initialCount : BallCount)
    (h1 : initialCount.yellow = 5)
    (h2 : initialCount.black = 13)
    (h3 : initialCount.red = 22) :
    isFair (replaceBalls initialCount 4) := by
  sorry

end NUMINAMATH_CALUDE_fairGame_l1413_141395


namespace NUMINAMATH_CALUDE_expression_equals_one_tenth_l1413_141378

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := ⌈x⌉

-- Define the expression
def expression : ℚ := 
  (ceiling ((25 : ℚ) / 11 - ceiling ((35 : ℚ) / 19))) / 
  (ceiling ((35 : ℚ) / 11 + ceiling ((11 * 19 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_tenth_l1413_141378


namespace NUMINAMATH_CALUDE_alloy_composition_proof_l1413_141371

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 0.12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 0.08

/-- The amount of the first alloy used in kg -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 0.092

/-- The amount of the second alloy used in kg -/
def amount_2 : ℝ := 35

theorem alloy_composition_proof :
  chromium_percent_1 * amount_1 + chromium_percent_2 * amount_2 =
  chromium_percent_new * (amount_1 + amount_2) := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_proof_l1413_141371


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l1413_141366

-- Define the cubic function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Theorem 1: Tangent line equation
theorem tangent_line_at_zero (a b c : ℝ) :
  ∃ m k, ∀ x, m*x + k = f a b c x + (f a b c 0 - f a b c x) / x :=
sorry

-- Theorem 2: Range of c when a = b = 4 and f has three distinct zeros
theorem range_of_c_for_three_zeros :
  ∃ c, 0 < c ∧ c < 32/27 ∧
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f 4 4 c x = 0 ∧ f 4 4 c y = 0 ∧ f 4 4 c z = 0) :=
sorry

-- Theorem 3: Necessary but not sufficient condition for three distinct zeros
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →
  a^2 - 3*b > 0 :=
sorry

theorem condition_not_sufficient :
  ∃ a b c, a^2 - 3*b > 0 ∧
  ¬(∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l1413_141366


namespace NUMINAMATH_CALUDE_complement_of_union_l1413_141347

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (A ∪ B)ᶜ = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1413_141347


namespace NUMINAMATH_CALUDE_specific_frustum_lateral_surface_area_l1413_141381

/-- The lateral surface area of a frustum of a cone --/
def lateralSurfaceArea (slantHeight : ℝ) (radiusRatio : ℝ) (centralAngle : ℝ) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a specific frustum of a cone --/
theorem specific_frustum_lateral_surface_area :
  lateralSurfaceArea 10 (2/5) 216 = 252 * Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_frustum_lateral_surface_area_l1413_141381


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1413_141377

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1413_141377


namespace NUMINAMATH_CALUDE_N_minus_M_eq_six_l1413_141367

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem N_minus_M_eq_six : N \ M = {6} := by sorry

end NUMINAMATH_CALUDE_N_minus_M_eq_six_l1413_141367


namespace NUMINAMATH_CALUDE_fermat_point_distance_sum_l1413_141336

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (3, 5)
def P : ℝ × ℝ := (5, 3)

theorem fermat_point_distance_sum :
  let AP := distance A.1 A.2 P.1 P.2
  let BP := distance B.1 B.2 P.1 P.2
  let CP := distance C.1 C.2 P.1 P.2
  AP + BP + CP = Real.sqrt 34 + Real.sqrt 58 + 2 * Real.sqrt 2 ∧
  (1 : ℕ) + (1 : ℕ) + (2 : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fermat_point_distance_sum_l1413_141336


namespace NUMINAMATH_CALUDE_log_equation_equals_zero_l1413_141349

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_equation_equals_zero : 2 * log5 10 + log5 0.25 = 0 := by sorry

end NUMINAMATH_CALUDE_log_equation_equals_zero_l1413_141349


namespace NUMINAMATH_CALUDE_anna_gets_more_candy_l1413_141397

/-- Calculates the difference in candy pieces between Anna and Billy --/
def candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) : ℕ :=
  anna_per_house * anna_houses - billy_per_house * billy_houses

/-- Proves that Anna gets 15 more pieces of candy than Billy --/
theorem anna_gets_more_candy : 
  candy_difference 14 11 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_anna_gets_more_candy_l1413_141397


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l1413_141372

/-- Calculates the final amount after two years of compound interest with different rates each year -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated -/
theorem compound_interest_calculation 
  (initial_amount : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_amount = 8736) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) : 
  final_amount initial_amount rate1 rate2 = 9539.712 := by
  sorry

#eval final_amount 8736 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l1413_141372


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1413_141361

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * 1^3 + b * 1^2 + c * 1 + d = 0)
  (h4 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = 49 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1413_141361


namespace NUMINAMATH_CALUDE_intersection_distance_l1413_141322

theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 8) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 8) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1413_141322


namespace NUMINAMATH_CALUDE_no_rational_roots_for_odd_coefficients_l1413_141338

theorem no_rational_roots_for_odd_coefficients (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ¬∃ (x : ℚ), x^2 + 2*↑p*x + 2*↑q = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_odd_coefficients_l1413_141338


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l1413_141389

/-- Given two sets A and B with specific elements, prove that if A = B, then a = 1 -/
theorem set_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l1413_141389


namespace NUMINAMATH_CALUDE_fibonacci_like_sequence_l1413_141323

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

def increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem fibonacci_like_sequence (a : ℕ → ℕ) 
  (h1 : sequence_property a) 
  (h2 : increasing_sequence a)
  (h3 : a 7 = 120) : 
  a 8 = 194 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_like_sequence_l1413_141323


namespace NUMINAMATH_CALUDE_A_on_axes_l1413_141314

def A (a : ℝ) : ℝ × ℝ := (a - 3, a^2 - 4)

theorem A_on_axes :
  (∀ a : ℝ, (A a).2 = 0 → (A a = (-1, 0) ∨ A a = (-5, 0))) ∧
  (∀ a : ℝ, (A a).1 = 0 → A a = (0, 5)) := by
  sorry

end NUMINAMATH_CALUDE_A_on_axes_l1413_141314


namespace NUMINAMATH_CALUDE_base5_412_equals_base7_212_l1413_141365

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : Nat) : Nat :=
  sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : Nat) : Nat :=
  sorry

/-- Theorem stating that 412₅ is equal to 212₇ --/
theorem base5_412_equals_base7_212 :
  decimalToBase7 (base5ToDecimal 412) = 212 :=
sorry

end NUMINAMATH_CALUDE_base5_412_equals_base7_212_l1413_141365


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1413_141379

/-- Represents the composition of a population --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Represents a stratified sample --/
structure StratifiedSample where
  population : Population
  sampleSize : Nat
  youngInSample : Nat

/-- Theorem stating the relationship between the sample size and the number of young people in the sample --/
theorem stratified_sample_size 
  (sample : StratifiedSample) 
  (h1 : sample.population = { elderly := 20, middleAged := 120, young := 100 })
  (h2 : sample.youngInSample = 10) : 
  sample.sampleSize = 24 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1413_141379


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_range_of_a_l1413_141310

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 2 * (3 * a + 1) ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3} ∪ {-1}

-- State the theorem
theorem sufficient_condition_implies_range_of_a (a : ℝ) :
  A a ⊆ B a → a ∈ RangeOfA := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_range_of_a_l1413_141310


namespace NUMINAMATH_CALUDE_ellipse_equation_slope_product_sum_of_squares_l1413_141325

/-- An ellipse with eccentricity √2/2 and foci on the unit circle -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (a^2 - b^2) / a^2 = 1/2
  h4 : a^2 - b^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (e : SpecialEllipse) where
  x : ℝ
  y : ℝ
  h : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : SpecialEllipse) :
  ∀ (p : PointOnEllipse e), p.x^2 / 2 + p.y^2 = 1 := by sorry

theorem slope_product (e : SpecialEllipse) (p q : PointOnEllipse e) 
  (hp : p.x ≠ 0) (hq : q.x ≠ 0) :
  (p.y / p.x) * (q.y / q.x) = -1/2 := by sorry

theorem sum_of_squares (e : SpecialEllipse) (p q : PointOnEllipse e) :
  p.x^2 + p.y^2 + q.x^2 + q.y^2 = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_slope_product_sum_of_squares_l1413_141325
