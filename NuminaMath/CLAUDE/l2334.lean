import Mathlib

namespace NUMINAMATH_CALUDE_flower_town_coin_impossibility_l2334_233411

/-- Represents the number of inhabitants in Flower Town -/
def num_inhabitants : ℕ := 1990

/-- Represents the number of coins each inhabitant must give -/
def coins_per_inhabitant : ℕ := 10

/-- Represents a meeting between two inhabitants -/
structure Meeting where
  giver : Fin num_inhabitants
  receiver : Fin num_inhabitants
  giver_gives_10 : Bool

/-- The main theorem stating the impossibility of the scenario -/
theorem flower_town_coin_impossibility :
  ¬ ∃ (meetings : List Meeting),
    (∀ i : Fin num_inhabitants, 
      (meetings.filter (λ m => m.giver = i ∨ m.receiver = i)).length = coins_per_inhabitant) ∧
    (∀ m : Meeting, m ∈ meetings → m.giver ≠ m.receiver) :=
by
  sorry


end NUMINAMATH_CALUDE_flower_town_coin_impossibility_l2334_233411


namespace NUMINAMATH_CALUDE_max_k_value_l2334_233428

theorem max_k_value (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℝ, (4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) → k ≤ 9) ∧ 
  (∃ k : ℝ, k = 9 ∧ 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l2334_233428


namespace NUMINAMATH_CALUDE_race_head_start_l2334_233491

/-- Given two runners A and B, where A's speed is 20/16 times B's speed,
    this theorem proves that A should give B a head start of 1/5 of the
    total race length for the race to end in a dead heat. -/
theorem race_head_start (v_B : ℝ) (L : ℝ) (h_pos_v : v_B > 0) (h_pos_L : L > 0) :
  let v_A := (20 / 16) * v_B
  let x := 1 / 5
  L / v_A = (L - x * L) / v_B := by
  sorry

#check race_head_start

end NUMINAMATH_CALUDE_race_head_start_l2334_233491


namespace NUMINAMATH_CALUDE_correct_verbs_for_sentence_l2334_233458

-- Define the structure of a sentence with two blanks
structure SentenceWithBlanks where
  first_blank : String
  second_blank : String

-- Define the concept of subject-verb agreement
def subjectVerbAgrees (subject : String) (verb : String) : Prop := sorry

-- Define the specific sentence structure
def remoteAreasNeed : String := "Remote areas need"
def childrenNeed : String := "children need"

-- Theorem to prove
theorem correct_verbs_for_sentence :
  ∃ (s : SentenceWithBlanks),
    subjectVerbAgrees remoteAreasNeed s.first_blank ∧
    subjectVerbAgrees childrenNeed s.second_blank ∧
    s.first_blank = "is" ∧
    s.second_blank = "are" := by sorry

end NUMINAMATH_CALUDE_correct_verbs_for_sentence_l2334_233458


namespace NUMINAMATH_CALUDE_chloe_earnings_l2334_233476

/-- Chloe's earnings over two weeks -/
theorem chloe_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 26 →
  extra_earnings = 65.45 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 360 :=
by sorry

end NUMINAMATH_CALUDE_chloe_earnings_l2334_233476


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l2334_233440

/-- Represents the number of cakes Baker made -/
def cakes_made : ℕ := sorry

/-- Represents the number of pastries Baker made -/
def pastries_made : ℕ := 153

/-- Represents the number of pastries Baker sold -/
def pastries_sold : ℕ := 8

/-- Represents the number of cakes Baker sold -/
def cakes_sold : ℕ := 97

/-- Represents the difference between cakes sold and pastries sold -/
def difference_sold : ℕ := 89

theorem baker_cakes_theorem : 
  pastries_made = 153 ∧ 
  pastries_sold = 8 ∧ 
  cakes_sold = 97 ∧ 
  difference_sold = 89 ∧ 
  cakes_sold - pastries_sold = difference_sold → 
  cakes_made = 97 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l2334_233440


namespace NUMINAMATH_CALUDE_trigonometric_identity_equivalence_l2334_233455

theorem trigonometric_identity_equivalence (x : ℝ) :
  (1 + Real.cos (4 * x)) * Real.sin (2 * x) = (Real.cos (2 * x))^2 ↔
  (∃ k : ℤ, x = (-1)^k * (π / 12) + k * (π / 2)) ∨
  (∃ n : ℤ, x = π / 4 * (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_equivalence_l2334_233455


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2334_233454

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetricXAxis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- The given point N. -/
def N : Point2D :=
  ⟨2, 3⟩

theorem symmetric_point_coordinates :
  symmetricXAxis N = ⟨2, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2334_233454


namespace NUMINAMATH_CALUDE_max_area_fence_enclosure_l2334_233464

/-- Represents a rectangular fence enclosure --/
structure FenceEnclosure where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 200
  length_constraint : length ≥ 90
  width_constraint : width ≥ 50
  ratio_constraint : length ≤ 2 * width

/-- The area of a fence enclosure --/
def area (f : FenceEnclosure) : ℝ := f.length * f.width

/-- Theorem stating the maximum area of the fence enclosure --/
theorem max_area_fence_enclosure :
  ∃ (f : FenceEnclosure), ∀ (g : FenceEnclosure), area f ≥ area g ∧ area f = 10000 :=
sorry

end NUMINAMATH_CALUDE_max_area_fence_enclosure_l2334_233464


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2334_233414

/-- Given the purchase of blankets with known and unknown rates, prove the unknown rate -/
theorem unknown_blanket_rate (total_blankets : ℕ) (known_rate1 known_rate2 avg_rate : ℚ) 
  (count1 count2 count_unknown : ℕ) :
  total_blankets = count1 + count2 + count_unknown →
  count1 = 1 →
  count2 = 5 →
  count_unknown = 2 →
  known_rate1 = 100 →
  known_rate2 = 150 →
  avg_rate = 150 →
  (count1 * known_rate1 + count2 * known_rate2 + count_unknown * ((total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown)) / total_blankets = avg_rate →
  (total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown = 175 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l2334_233414


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l2334_233400

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 + 5 * x

-- Define the three points
def p1 : ℝ × ℝ := (0, 0)
def p2 : ℝ × ℝ := (-1, -1)
def p3 : ℝ × ℝ := (1, 9)

-- Theorem statement
theorem quadratic_function_passes_through_points :
  f p1.1 = p1.2 ∧ f p2.1 = p2.2 ∧ f p3.1 = p3.2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l2334_233400


namespace NUMINAMATH_CALUDE_only_135_and_144_satisfy_l2334_233416

/-- Represents a 3-digit positive integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The decimal representation of abc --/
def decimal_rep (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The product of digits multiplied by their sum --/
def digit_product_sum (n : ThreeDigitInt) : Nat :=
  n.a * n.b * n.c * (n.a + n.b + n.c)

/-- The theorem stating that only 135 and 144 satisfy the equation --/
theorem only_135_and_144_satisfy :
  ∀ n : ThreeDigitInt, decimal_rep n = digit_product_sum n ↔ decimal_rep n = 135 ∨ decimal_rep n = 144 := by
  sorry

end NUMINAMATH_CALUDE_only_135_and_144_satisfy_l2334_233416


namespace NUMINAMATH_CALUDE_room_width_proof_l2334_233409

/-- Given a rectangular room with known length, paving cost, and paving rate per square meter,
    prove that the width of the room is 2.75 meters. -/
theorem room_width_proof (length : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  length = 6.5 →
  paving_cost = 10725 →
  paving_rate = 600 →
  paving_cost / paving_rate / length = 2.75 := by
  sorry


end NUMINAMATH_CALUDE_room_width_proof_l2334_233409


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2334_233434

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2334_233434


namespace NUMINAMATH_CALUDE_georges_calculation_l2334_233488

theorem georges_calculation (y : ℝ) : y / 7 = 30 → y + 70 = 280 := by
  sorry

end NUMINAMATH_CALUDE_georges_calculation_l2334_233488


namespace NUMINAMATH_CALUDE_geometry_problem_l2334_233422

/-- Two lines l₁ and l₂, and a point P -/
structure GeometrySetup where
  -- Line l₁: 2x - y - 5 = 0
  l₁ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ 2 * x - y - 5 = 0
  -- Line l₂: x + y - 5 = 0
  l₂ : ℝ → ℝ → Prop
  l₂_eq : ∀ x y, l₂ x y ↔ x + y - 5 = 0
  -- Point P(3, 0)
  P : ℝ × ℝ
  P_def : P = (3, 0)

/-- The distance function from a point to a line -/
noncomputable def distance_point_to_line (P : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ := sorry

/-- The line m passing through P and intersecting l₁ and l₂ -/
def line_m (setup : GeometrySetup) : ℝ → ℝ → Prop := sorry

theorem geometry_problem (setup : GeometrySetup) :
  -- (I) The distance from P to l₁ is √5/5
  distance_point_to_line setup.P setup.l₁ = Real.sqrt 5 / 5 ∧
  -- (II) The equation of line m is x - y/10 - 3 = 0
  ∀ x y, line_m setup x y ↔ x - y / 10 - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_geometry_problem_l2334_233422


namespace NUMINAMATH_CALUDE_discount_order_matters_l2334_233462

def original_price : ℚ := 50
def fixed_discount : ℚ := 10
def percentage_discount : ℚ := 0.25

def price_fixed_then_percentage : ℚ := (original_price - fixed_discount) * (1 - percentage_discount)
def price_percentage_then_fixed : ℚ := (original_price * (1 - percentage_discount)) - fixed_discount

theorem discount_order_matters :
  price_percentage_then_fixed < price_fixed_then_percentage ∧
  (price_fixed_then_percentage - price_percentage_then_fixed) * 100 = 250 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_matters_l2334_233462


namespace NUMINAMATH_CALUDE_smaller_tv_diagonal_l2334_233483

/-- Given two square televisions where the larger one has a 28-inch diagonal
    and its screen area is 79.5 square inches greater than the smaller one,
    prove that the smaller television's diagonal is 25 inches. -/
theorem smaller_tv_diagonal (d : ℝ) : 
  d > 0 → -- d is positive (diagonal length)
  (28 / Real.sqrt 2) ^ 2 = (d / Real.sqrt 2) ^ 2 + 79.5 →
  d = 25 := by
sorry

end NUMINAMATH_CALUDE_smaller_tv_diagonal_l2334_233483


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2334_233445

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2334_233445


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l2334_233499

theorem units_digit_of_fraction : (30 * 31 * 32 * 33 * 34 * 35) / 5000 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l2334_233499


namespace NUMINAMATH_CALUDE_all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l2334_233436

-- Define the type for geometric bodies
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cube
  | HexagonalPyramid
  | Sphere
  | Cone
  | Frustum
  | Hemisphere

-- Define a predicate for polyhedra
def isPolyhedron : GeometricBody → Prop
  | GeometricBody.TriangularPrism => True
  | GeometricBody.QuadrangularPyramid => True
  | GeometricBody.Cube => True
  | GeometricBody.HexagonalPyramid => True
  | _ => False

-- Define the set of geometric bodies in option C
def setC : List GeometricBody :=
  [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid,
   GeometricBody.Cube, GeometricBody.HexagonalPyramid]

-- Theorem: All elements in setC are polyhedra
theorem all_setC_are_polyhedra : ∀ x ∈ setC, isPolyhedron x := by
  sorry

-- Main theorem: setC consists entirely of polyhedra
theorem setC_consists_entirely_of_polyhedra : 
  (∀ x ∈ setC, isPolyhedron x) ∧ (setC ≠ []) := by
  sorry

end NUMINAMATH_CALUDE_all_setC_are_polyhedra_setC_consists_entirely_of_polyhedra_l2334_233436


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2334_233437

/-- Given a total spend, sales tax, and tax rate, calculate the cost of tax-free items -/
def cost_of_tax_free_items (total_spend : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_spend - sales_tax / tax_rate

/-- Theorem: Given the specific values from the problem, the cost of tax-free items is 22 rupees -/
theorem tax_free_items_cost :
  let total_spend : ℚ := 25
  let sales_tax : ℚ := 30 / 100 -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100 -- 10%
  cost_of_tax_free_items total_spend sales_tax tax_rate = 22 := by
  sorry

#eval cost_of_tax_free_items 25 (30/100) (10/100)

end NUMINAMATH_CALUDE_tax_free_items_cost_l2334_233437


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2334_233489

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation ax - by + 2 = 0 -/
def line_equation (a b x y : ℝ) : Prop :=
  a*x - b*y + 2 = 0

/-- The chord length is 4 -/
def chord_length (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation a b x₁ y₁ ∧ line_equation a b x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  chord_length a b → (1/a + 1/b ≥ 3/2 + Real.sqrt 2) ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ chord_length a₀ b₀ ∧ 1/a₀ + 1/b₀ = 3/2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2334_233489


namespace NUMINAMATH_CALUDE_club_size_after_four_years_l2334_233479

def club_size (initial_members : ℕ) (years : ℕ) : ℕ :=
  let active_members := initial_members - 3
  let growth_factor := 4
  (growth_factor ^ years) * active_members + 3

theorem club_size_after_four_years :
  club_size 21 4 = 4611 := by sorry

end NUMINAMATH_CALUDE_club_size_after_four_years_l2334_233479


namespace NUMINAMATH_CALUDE_count_80_in_scores_l2334_233492

def scores : List ℕ := [80, 90, 80, 80, 100, 70]

theorem count_80_in_scores : (scores.filter (· = 80)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_80_in_scores_l2334_233492


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2334_233426

theorem triangle_max_perimeter :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b = 4 * a ∧ 
    a + b > 18 ∧ 
    a + 18 > b ∧ 
    b + 18 > a ∧
    ∀ (x y : ℕ), 
      x > 0 → 
      y > 0 → 
      y = 4 * x → 
      x + y > 18 → 
      x + 18 > y → 
      y + 18 > x → 
      a + b + 18 ≥ x + y + 18 ∧
    a + b + 18 = 43 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2334_233426


namespace NUMINAMATH_CALUDE_min_jugs_to_fill_container_l2334_233424

/-- The capacity of a regular water jug in milliliters -/
def regular_jug_capacity : ℕ := 300

/-- The capacity of a giant water container in milliliters -/
def giant_container_capacity : ℕ := 1800

/-- The minimum number of regular jugs needed to fill a giant container -/
def min_jugs_needed : ℕ := giant_container_capacity / regular_jug_capacity

theorem min_jugs_to_fill_container : min_jugs_needed = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_jugs_to_fill_container_l2334_233424


namespace NUMINAMATH_CALUDE_smallest_positive_period_sin_cos_l2334_233470

/-- The smallest positive period of f(x) = sin x cos x is π -/
theorem smallest_positive_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x * Real.cos x) :
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = π :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_sin_cos_l2334_233470


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l2334_233463

/-- The number of ways to select 3 different digits from 0 to 9 -/
def select_three_digits : ℕ := Nat.choose 10 3

/-- The number of four-digit numbers formed by selecting three different digits from 0 to 9,
    where one digit may appear twice -/
def four_digit_numbers : ℕ := 3888

/-- Theorem stating that the number of four-digit numbers formed by selecting
    three different digits from 0 to 9 (where one digit may appear twice) is 3888 -/
theorem count_four_digit_numbers :
  four_digit_numbers = 3888 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l2334_233463


namespace NUMINAMATH_CALUDE_cards_kept_away_is_two_l2334_233466

/-- The number of cards in a standard deck of playing cards. -/
def standard_deck_size : ℕ := 52

/-- The number of cards used for playing. -/
def cards_used : ℕ := 50

/-- The number of cards kept away. -/
def cards_kept_away : ℕ := standard_deck_size - cards_used

theorem cards_kept_away_is_two : cards_kept_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_cards_kept_away_is_two_l2334_233466


namespace NUMINAMATH_CALUDE_lunch_spending_difference_l2334_233447

/-- Given a lunch scenario where two people spent a total of $15,
    with one person spending $10, prove that the difference in
    spending between the two people is $5. -/
theorem lunch_spending_difference :
  ∀ (your_spending friend_spending : ℕ),
  your_spending + friend_spending = 15 →
  friend_spending = 10 →
  friend_spending > your_spending →
  friend_spending - your_spending = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_difference_l2334_233447


namespace NUMINAMATH_CALUDE_pencil_count_l2334_233404

/-- Given the ratio of pens to pencils and their difference, calculate the number of pencils -/
theorem pencil_count (x : ℕ) (h1 : 6 * x = 5 * x + 6) : 6 * x = 36 := by
  sorry

#check pencil_count

end NUMINAMATH_CALUDE_pencil_count_l2334_233404


namespace NUMINAMATH_CALUDE_gift_purchase_cost_l2334_233478

def total_cost (items : List (ℕ × ℚ)) (sales_tax_rate : ℚ) (credit_card_rebate : ℚ)
  (book_discount_rate : ℚ) (sneaker_discount_rate : ℚ) : ℚ :=
  sorry

theorem gift_purchase_cost :
  let items : List (ℕ × ℚ) := [
    (3, 26), (2, 83), (1, 90), (4, 7), (3, 15), (2, 22), (5, 8), (1, 65)
  ]
  let sales_tax_rate : ℚ := 6.5 / 100
  let credit_card_rebate : ℚ := 12
  let book_discount_rate : ℚ := 10 / 100
  let sneaker_discount_rate : ℚ := 15 / 100
  total_cost items sales_tax_rate credit_card_rebate book_discount_rate sneaker_discount_rate = 564.96 :=
by sorry

end NUMINAMATH_CALUDE_gift_purchase_cost_l2334_233478


namespace NUMINAMATH_CALUDE_solution_sum_l2334_233432

theorem solution_sum (P q : ℝ) : 
  (2^2 - P*2 + 6 = 0) → (2^2 + 6*2 - q = 0) → P + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l2334_233432


namespace NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_not_always_parallel_l2334_233448

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_parallel_planes_not_always_parallel 
  (m n : Line) (α β : Plane) : 
  ¬(∀ m n α β, subset m α ∧ subset n β ∧ parallel_planes α β → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_not_always_parallel_l2334_233448


namespace NUMINAMATH_CALUDE_room_length_is_19_l2334_233497

/-- Represents the dimensions of a rectangular room with a surrounding veranda. -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda given the room dimensions. -/
def verandaArea (room : RoomWithVeranda) : ℝ :=
  (room.roomLength + 2 * room.verandaWidth) * (room.roomWidth + 2 * room.verandaWidth) -
  room.roomLength * room.roomWidth

/-- Theorem: The length of the room is 19 meters given the specified conditions. -/
theorem room_length_is_19 (room : RoomWithVeranda)
  (h1 : room.roomWidth = 12)
  (h2 : room.verandaWidth = 2)
  (h3 : verandaArea room = 140) :
  room.roomLength = 19 := by
  sorry

end NUMINAMATH_CALUDE_room_length_is_19_l2334_233497


namespace NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_thirty_sixth_l2334_233490

/-- The probability of rolling a specific number on a fair 6-sided die -/
def prob_single : ℚ := 1 / 6

/-- The set of numbers on a 6-sided die -/
def die_numbers : Set ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of numbers on a 6-sided die that are divisible by 4 -/
def divisible_by_four : Set ℕ := {n ∈ die_numbers | n % 4 = 0}

/-- The probability that both dice show numbers divisible by 4 -/
def prob_both_divisible_by_four : ℚ := prob_single * prob_single

theorem prob_both_divisible_by_four_is_one_thirty_sixth :
  prob_both_divisible_by_four = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_thirty_sixth_l2334_233490


namespace NUMINAMATH_CALUDE_fifteen_children_pencil_count_l2334_233402

/-- Given a number of children and pencils per child, calculates the total number of pencils -/
def total_pencils (num_children : ℕ) (pencils_per_child : ℕ) : ℕ :=
  num_children * pencils_per_child

/-- Proves that 15 children with 2 pencils each have 30 pencils in total -/
theorem fifteen_children_pencil_count :
  total_pencils 15 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_children_pencil_count_l2334_233402


namespace NUMINAMATH_CALUDE_tuna_cost_theorem_l2334_233487

/-- Calculates the cost of a single can of tuna in cents -/
def tuna_cost_cents (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
                    (amount_paid : ℕ) (change_received : ℕ) : ℕ :=
  let total_paid := amount_paid - change_received
  let coupon_discount := num_coupons * coupon_value
  let total_cost := total_paid * 100 + coupon_discount
  total_cost / num_cans

theorem tuna_cost_theorem : 
  tuna_cost_cents 9 5 25 2000 550 = 175 := by
  sorry

end NUMINAMATH_CALUDE_tuna_cost_theorem_l2334_233487


namespace NUMINAMATH_CALUDE_min_value_is_884_l2334_233460

/-- A type representing a permutation of the numbers 1 to 9 -/
def Perm9 := { f : Fin 9 → Fin 9 // Function.Bijective f }

/-- The expression we want to minimize -/
def expr (p : Perm9) : ℕ :=
  let x₁ := (p.val 0).val + 1
  let x₂ := (p.val 1).val + 1
  let x₃ := (p.val 2).val + 1
  let y₁ := (p.val 3).val + 1
  let y₂ := (p.val 4).val + 1
  let y₃ := (p.val 5).val + 1
  let z₁ := (p.val 6).val + 1
  let z₂ := (p.val 7).val + 1
  let z₃ := (p.val 8).val + 1
  x₁ * x₂ * x₃ + y₁ * y₂ * y₃ + z₁ * z₂ * z₃ + x₁ * y₁ * z₁

/-- The theorem stating that the minimum value of the expression is 884 -/
theorem min_value_is_884 : ∀ p : Perm9, expr p ≥ 884 := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_884_l2334_233460


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2334_233486

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (4 + 1 / 5)) = 21 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2334_233486


namespace NUMINAMATH_CALUDE_school_competition_selections_l2334_233450

theorem school_competition_selections (n m : ℕ) (hn : n = 5) (hm : m = 3) : 
  (n.choose m) * m.factorial = 60 := by
  sorry

end NUMINAMATH_CALUDE_school_competition_selections_l2334_233450


namespace NUMINAMATH_CALUDE_inequality_proof_l2334_233449

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2334_233449


namespace NUMINAMATH_CALUDE_minimum_raft_capacity_l2334_233493

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The main theorem stating the minimum raft capacity -/
theorem minimum_raft_capacity 
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canTransportAll r (mice ++ moles ++ hamsters) ∧
    ∀ (r' : Raft), r'.capacity < 140 → ¬canTransportAll r' (mice ++ moles ++ hamsters) :=
sorry

end NUMINAMATH_CALUDE_minimum_raft_capacity_l2334_233493


namespace NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_24_cubed_l2334_233412

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_42_cubed_plus_24_cubed :
  unitsDigit (42^3 + 24^3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_24_cubed_l2334_233412


namespace NUMINAMATH_CALUDE_range_of_m_l2334_233471

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2334_233471


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l2334_233425

def container_X : ℕ × ℕ := (7, 3)  -- (blue balls, yellow balls)
def container_Y : ℕ × ℕ := (5, 5)
def container_Z : ℕ × ℕ := (8, 2)

def total_balls (c : ℕ × ℕ) : ℕ := c.1 + c.2

def prob_yellow (c : ℕ × ℕ) : ℚ := c.2 / (total_balls c)

def prob_container : ℚ := 1 / 3

theorem yellow_ball_probability :
  prob_container * prob_yellow container_X +
  prob_container * prob_yellow container_Y +
  prob_container * prob_yellow container_Z = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l2334_233425


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_is_integer_l2334_233406

theorem sqrt_sum_squares_is_integer : ∃ (z : ℕ), z * z = 25530 * 25530 + 29464 * 29464 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_is_integer_l2334_233406


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2334_233438

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2334_233438


namespace NUMINAMATH_CALUDE_population_decrease_l2334_233498

theorem population_decrease (k : ℝ) (P₀ : ℝ) (n : ℕ) 
  (h1 : -1 < k) (h2 : k < 0) (h3 : P₀ > 0) : 
  P₀ * (1 + k)^(n + 1) < P₀ * (1 + k)^n := by
  sorry

#check population_decrease

end NUMINAMATH_CALUDE_population_decrease_l2334_233498


namespace NUMINAMATH_CALUDE_graphics_cards_sold_l2334_233419

/-- Represents the number of graphics cards sold. -/
def graphics_cards : ℕ := sorry

/-- Represents the number of hard drives sold. -/
def hard_drives : ℕ := 14

/-- Represents the number of CPUs sold. -/
def cpus : ℕ := 8

/-- Represents the number of RAM pairs sold. -/
def ram_pairs : ℕ := 4

/-- Represents the price of a single graphics card in dollars. -/
def graphics_card_price : ℕ := 600

/-- Represents the price of a single hard drive in dollars. -/
def hard_drive_price : ℕ := 80

/-- Represents the price of a single CPU in dollars. -/
def cpu_price : ℕ := 200

/-- Represents the price of a pair of RAM in dollars. -/
def ram_pair_price : ℕ := 60

/-- Represents the total earnings of the store in dollars. -/
def total_earnings : ℕ := 8960

/-- Theorem stating that the number of graphics cards sold is 10. -/
theorem graphics_cards_sold : graphics_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_graphics_cards_sold_l2334_233419


namespace NUMINAMATH_CALUDE_old_clock_slower_by_12_minutes_l2334_233465

/-- Represents the time interval between consecutive coincidences of hour and minute hands -/
def coincidence_interval : ℕ := 66

/-- Represents the number of coincidences in a 24-hour period -/
def coincidences_per_day : ℕ := 22

/-- Represents the number of minutes in a standard 24-hour day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of minutes in the old clock's 24-hour period -/
def old_clock_day_minutes : ℕ := coincidence_interval * coincidences_per_day

theorem old_clock_slower_by_12_minutes :
  old_clock_day_minutes - standard_day_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slower_by_12_minutes_l2334_233465


namespace NUMINAMATH_CALUDE_square_difference_equals_square_l2334_233446

theorem square_difference_equals_square (x : ℝ) : (10 - x)^2 = x^2 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_square_l2334_233446


namespace NUMINAMATH_CALUDE_fred_gave_156_sheets_l2334_233431

/-- The number of sheets Fred gave to Charles -/
def sheets_given_to_charles (initial_sheets : ℕ) (received_sheets : ℕ) (final_sheets : ℕ) : ℕ :=
  initial_sheets + received_sheets - final_sheets

/-- Theorem stating that Fred gave 156 sheets to Charles -/
theorem fred_gave_156_sheets :
  sheets_given_to_charles 212 307 363 = 156 := by
  sorry

end NUMINAMATH_CALUDE_fred_gave_156_sheets_l2334_233431


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2334_233477

/-- Given two cubes with side length b joined to form a cuboid, 
    the surface area of the resulting cuboid is 10b^2 -/
theorem cuboid_surface_area (b : ℝ) (h : b > 0) : 
  2 * (2*b*b + b*b + b*(2*b)) = 10 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l2334_233477


namespace NUMINAMATH_CALUDE_jill_bouncy_balls_difference_l2334_233408

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := balls_per_pack * red_packs

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := balls_per_pack * yellow_packs

/-- The difference between the number of red and yellow bouncy balls -/
def difference : ℕ := total_red_balls - total_yellow_balls

theorem jill_bouncy_balls_difference :
  difference = 18 := by sorry

end NUMINAMATH_CALUDE_jill_bouncy_balls_difference_l2334_233408


namespace NUMINAMATH_CALUDE_bookstore_location_l2334_233418

/-- The floor number of the academy -/
def academy_floor : ℕ := 7

/-- The number of floors the reading room is above the academy -/
def reading_room_above_academy : ℕ := 4

/-- The number of floors the bookstore is below the reading room -/
def bookstore_below_reading_room : ℕ := 9

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := academy_floor + reading_room_above_academy - bookstore_below_reading_room

theorem bookstore_location : bookstore_floor = 2 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_location_l2334_233418


namespace NUMINAMATH_CALUDE_smith_bought_six_boxes_l2334_233495

/-- Calculates the number of new boxes of markers bought by Mr. Smith -/
def new_boxes_bought (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) : ℕ :=
  (final_markers - initial_markers) / markers_per_box

/-- Proves that Mr. Smith bought 6 new boxes of markers -/
theorem smith_bought_six_boxes :
  new_boxes_bought 32 9 86 = 6 := by
  sorry

#eval new_boxes_bought 32 9 86

end NUMINAMATH_CALUDE_smith_bought_six_boxes_l2334_233495


namespace NUMINAMATH_CALUDE_average_score_is_94_l2334_233482

def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

def total_score : ℕ := june_score + patty_score + josh_score + henry_score
def num_children : ℕ := 4

theorem average_score_is_94 : total_score / num_children = 94 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_94_l2334_233482


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2334_233439

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two angles equal
  α = β →
  -- One of the equal angles is 50°
  α = 50 →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 80°
  max α (max β γ) = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l2334_233439


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l2334_233457

theorem halfway_between_one_third_and_one_fifth : 
  (1 / 3 + 1 / 5) / 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l2334_233457


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_foci_and_eccentricity_l2334_233469

/-- A hyperbola with given foci and eccentricity -/
structure Hyperbola where
  foci : ℝ × ℝ × ℝ × ℝ  -- Represents (x₁, y₁, x₂, y₂)
  eccentricity : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- Theorem stating that a hyperbola with given foci and eccentricity has the specified equation -/
theorem hyperbola_equation_from_foci_and_eccentricity (h : Hyperbola)
    (h_foci : h.foci = (-2, 0, 2, 0))
    (h_eccentricity : h.eccentricity = Real.sqrt 2) :
    ∀ x y, hyperbola_equation h x y :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_foci_and_eccentricity_l2334_233469


namespace NUMINAMATH_CALUDE_apple_probability_l2334_233484

theorem apple_probability (p_less_200 p_not_less_350 : ℝ) 
  (h1 : p_less_200 = 0.25) 
  (h2 : p_not_less_350 = 0.22) : 
  1 - p_less_200 - p_not_less_350 = 0.53 := by
  sorry

end NUMINAMATH_CALUDE_apple_probability_l2334_233484


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l2334_233427

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 20) (h2 : 5 * b = 4 * a) : 
  Nat.lcm a b = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l2334_233427


namespace NUMINAMATH_CALUDE_nested_expression_value_l2334_233461

theorem nested_expression_value : (3*(3*(2*(2*(2*(3+2)+1)+1)+2)+1)+1) = 436 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2334_233461


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2334_233496

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_increasing : IsIncreasingGeometricSequence a)
    (h_a3 : a 3 = 4)
    (h_sum : 1 / a 1 + 1 / a 5 = 5 / 8) :
  a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2334_233496


namespace NUMINAMATH_CALUDE_circle_and_line_proof_l2334_233433

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y + 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 5 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define point B
def point_B : ℝ × ℝ := (2, -2)

-- Define point D
def point_D : ℝ × ℝ := (-1, -1)

-- Define the line m (both possible equations)
def line_m (x y : ℝ) : Prop := x = -1 ∨ 3*x + 4*y + 7 = 0

theorem circle_and_line_proof :
  ∀ (x y : ℝ),
  (circle_C x y ↔ (x - point_A.1)^2 + (y - point_A.2)^2 = 25 ∧ 
                  (x - point_B.1)^2 + (y - point_B.2)^2 = 25) ∧
  (∃ (cx cy : ℝ), line_l cx cy ∧ circle_C cx cy) ∧
  (∃ (mx my : ℝ), line_m mx my ∧ point_D = (mx, my) ∧
    ∃ (x1 y1 x2 y2 : ℝ),
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      line_m x1 y1 ∧ line_m x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 4 * 21) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_proof_l2334_233433


namespace NUMINAMATH_CALUDE_pencils_per_child_l2334_233421

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 11) 
  (h2 : total_pencils = 22) : 
  total_pencils / num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l2334_233421


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_primes_l2334_233410

def is_divisible_by_primes (n : ℕ) : Prop :=
  ∃ k : ℕ, k * (2213 * 3323 * 6121) = (n / 2).factorial * 2^(n / 2)

theorem smallest_n_divisible_by_primes :
  (∀ m : ℕ, m < 12242 → ¬(is_divisible_by_primes m)) ∧
  (is_divisible_by_primes 12242) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_primes_l2334_233410


namespace NUMINAMATH_CALUDE_growth_rate_is_ten_percent_l2334_233467

def turnover_may : ℝ := 1
def turnover_july : ℝ := 1.21

def growth_rate (r : ℝ) : Prop :=
  turnover_may * (1 + r)^2 = turnover_july

theorem growth_rate_is_ten_percent :
  ∃ (r : ℝ), growth_rate r ∧ r = 0.1 :=
sorry

end NUMINAMATH_CALUDE_growth_rate_is_ten_percent_l2334_233467


namespace NUMINAMATH_CALUDE_road_sign_ratio_l2334_233423

theorem road_sign_ratio (s₁ s₂ s₃ s₄ : ℕ) : 
  s₁ = 40 →
  s₂ > s₁ →
  s₃ = 2 * s₂ →
  s₄ = s₃ - 20 →
  s₁ + s₂ + s₃ + s₄ = 270 →
  s₂ / s₁ = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_road_sign_ratio_l2334_233423


namespace NUMINAMATH_CALUDE_circle_properties_l2334_233468

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y + 4 = 0) :
  (∃ (k : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≤ k ∧ k = 4/3) ∧
  (∃ (m : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≥ m ∧ m = 0) ∧
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → x' + y' ≤ M ∧ M = 3 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2334_233468


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l2334_233415

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) : 
  last_year_harvest + harvest_increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l2334_233415


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2334_233494

/-- Calculates the total number of people who can ride a Ferris wheel -/
theorem ferris_wheel_capacity 
  (capacity : ℕ)           -- Number of people per ride
  (ride_duration : ℕ)      -- Duration of one ride in minutes
  (operation_time : ℕ) :   -- Total operation time in hours
  capacity * (60 / ride_duration) * operation_time = 1260 :=
by
  sorry

#check ferris_wheel_capacity 70 20 6

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2334_233494


namespace NUMINAMATH_CALUDE_min_value_expression_l2334_233473

theorem min_value_expression (x y : ℝ) (hx : x ≥ 4) (hy : y ≥ -3) :
  x^2 + y^2 - 8*x + 6*y + 20 ≥ -5 ∧
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 4 ∧ y₀ ≥ -3 ∧ x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 20 = -5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2334_233473


namespace NUMINAMATH_CALUDE_count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l2334_233407

/-- A function that returns the count of four-digit numbers with all even digits in descending order -/
def count_descending_even_digits : ℕ :=
  5

/-- A function that returns the count of four-digit numbers with all even digits in ascending order -/
def count_ascending_even_digits : ℕ :=
  1

/-- Theorem stating the count of four-digit numbers with all even digits in descending order -/
theorem count_descending_even_digits_is_five :
  count_descending_even_digits = 5 := by sorry

/-- Theorem stating the count of four-digit numbers with all even digits in ascending order -/
theorem count_ascending_even_digits_is_one :
  count_ascending_even_digits = 1 := by sorry

end NUMINAMATH_CALUDE_count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l2334_233407


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2334_233480

theorem smallest_number_with_conditions : ∃ A : ℕ,
  (A % 10 = 6) ∧
  (4 * A = 6 * (A / 10)) ∧
  (∀ B : ℕ, B < A → ¬(B % 10 = 6 ∧ 4 * B = 6 * (B / 10))) ∧
  A = 153846 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2334_233480


namespace NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l2334_233485

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if a triangle has perimeter 8 -/
def has_perimeter_8 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 8

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The main theorem to be proved -/
theorem unique_triangle_with_perimeter_8 :
  ∃! t : IntTriangle, has_perimeter_8 t ∧
  (∀ t' : IntTriangle, has_perimeter_8 t' → are_congruent t t') :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_with_perimeter_8_l2334_233485


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2334_233441

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 25 * x^2 + n * x + 4 = 0) ↔ n = 20 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2334_233441


namespace NUMINAMATH_CALUDE_A_time_to_complete_l2334_233452

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom AB_time : rA + rB = 1 / 2
axiom BC_time : rB + rC = 1 / 4
axiom AC_time : rA + rC = 5 / 12

-- Define the theorem
theorem A_time_to_complete : 1 / rA = 3 := by
  sorry

end NUMINAMATH_CALUDE_A_time_to_complete_l2334_233452


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2334_233451

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 20 < 0 ↔ 4 < y ∧ y < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2334_233451


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l2334_233413

noncomputable def y (x : ℝ) : ℝ := Real.rpow (x - Real.log x - 1) (1/3)

theorem function_satisfies_equation (x : ℝ) (h : x > 0) :
  Real.log x + (y x)^3 - 3 * x * (y x)^2 * (deriv y x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l2334_233413


namespace NUMINAMATH_CALUDE_cyclist_motorcyclist_speed_l2334_233459

theorem cyclist_motorcyclist_speed : ∀ (motorcyclist_speed : ℝ) (cyclist_speed : ℝ),
  motorcyclist_speed > 0 ∧
  cyclist_speed > 0 ∧
  cyclist_speed = motorcyclist_speed - 30 ∧
  120 / motorcyclist_speed + 2 = 120 / cyclist_speed →
  motorcyclist_speed = 60 ∧ cyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_motorcyclist_speed_l2334_233459


namespace NUMINAMATH_CALUDE_water_missing_calculation_l2334_233475

/-- Calculates the amount of water missing from a tank's maximum capacity after a series of leaks and refilling. -/
def water_missing (initial_capacity : ℕ) (leak_rate1 leak_duration1 : ℕ) (leak_rate2 leak_duration2 : ℕ) (fill_rate fill_duration : ℕ) : ℕ :=
  let total_leak := leak_rate1 * leak_duration1 + leak_rate2 * leak_duration2
  let remaining_water := initial_capacity - total_leak
  let filled_water := fill_rate * fill_duration
  let final_water := remaining_water + filled_water
  initial_capacity - final_water

/-- Theorem stating that the amount of water missing from the tank's maximum capacity is 140,000 gallons. -/
theorem water_missing_calculation :
  water_missing 350000 32000 5 10000 10 40000 3 = 140000 := by
  sorry

end NUMINAMATH_CALUDE_water_missing_calculation_l2334_233475


namespace NUMINAMATH_CALUDE_james_age_l2334_233403

/-- Represents the ages of Dan, James, and Lisa --/
structure Ages where
  dan : ℕ
  james : ℕ
  lisa : ℕ

/-- The conditions of the problem --/
def age_conditions (ages : Ages) : Prop :=
  ∃ (k : ℕ),
    ages.dan = 6 * k ∧
    ages.james = 5 * k ∧
    ages.lisa = 4 * k ∧
    ages.dan + 4 = 28 ∧
    ages.james + ages.lisa = 3 * (ages.james - ages.lisa)

/-- The theorem to prove --/
theorem james_age (ages : Ages) :
  age_conditions ages → ages.james = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_age_l2334_233403


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l2334_233435

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l2334_233435


namespace NUMINAMATH_CALUDE_shipping_cost_formula_l2334_233453

/-- The shipping cost function for a parcel and flat-rate envelope -/
def shippingCost (P : ℝ) : ℝ :=
  let firstPoundFee : ℝ := 12
  let additionalPoundFee : ℝ := 5
  let flatRateEnvelopeFee : ℝ := 20
  firstPoundFee + additionalPoundFee * (P - 1) + flatRateEnvelopeFee

theorem shipping_cost_formula (P : ℝ) :
  shippingCost P = 5 * P + 27 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_formula_l2334_233453


namespace NUMINAMATH_CALUDE_campaign_fund_family_contribution_percentage_l2334_233456

/-- Calculates the percentage of family contribution in a campaign fund scenario -/
theorem campaign_fund_family_contribution_percentage 
  (total_funds : ℝ) 
  (friends_percentage : ℝ) 
  (president_savings : ℝ) : 
  total_funds = 10000 →
  friends_percentage = 40 →
  president_savings = 4200 →
  let friends_contribution := (friends_percentage / 100) * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := remaining_after_friends - president_savings
  (family_contribution / remaining_after_friends) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_campaign_fund_family_contribution_percentage_l2334_233456


namespace NUMINAMATH_CALUDE_solve_equation_l2334_233442

theorem solve_equation (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2334_233442


namespace NUMINAMATH_CALUDE_base_is_ten_l2334_233443

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

-- Define a function to check if the equation holds in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [5, 7, 3, 4] h + to_decimal [6, 4, 2, 1] h = to_decimal [1, 4, 1, 5, 5] h

-- Theorem statement
theorem base_is_ten : ∃ h, h = 10 ∧ equation_holds h := by
  sorry

end NUMINAMATH_CALUDE_base_is_ten_l2334_233443


namespace NUMINAMATH_CALUDE_solution_equality_l2334_233401

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x - 1| + |x + 2| < 5}

-- State the theorem
theorem solution_equality : solution_set = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l2334_233401


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2334_233430

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-3 * a > -3 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2334_233430


namespace NUMINAMATH_CALUDE_fishing_and_camping_l2334_233420

/-- Represents the fishing and camping problem -/
theorem fishing_and_camping
  (total_fish_weight : ℝ)
  (wastage_percentage : ℝ)
  (adult_consumption : ℝ)
  (child_consumption : ℝ)
  (adult_child_ratio : ℚ)
  (max_campers : ℕ)
  (h1 : total_fish_weight = 44)
  (h2 : wastage_percentage = 0.2)
  (h3 : adult_consumption = 3)
  (h4 : child_consumption = 1)
  (h5 : adult_child_ratio = 2 / 5)
  (h6 : max_campers = 12) :
  ∃ (adult_campers child_campers : ℕ),
    adult_campers = 2 ∧
    child_campers = 5 ∧
    adult_campers + child_campers ≤ max_campers ∧
    (adult_campers : ℚ) / (child_campers : ℚ) = adult_child_ratio ∧
    (adult_campers : ℝ) * adult_consumption + (child_campers : ℝ) * child_consumption ≤
      total_fish_weight * (1 - wastage_percentage) :=
by sorry

end NUMINAMATH_CALUDE_fishing_and_camping_l2334_233420


namespace NUMINAMATH_CALUDE_different_color_probability_l2334_233474

theorem different_color_probability (blue_chips yellow_chips : ℕ) 
  (h_blue : blue_chips = 5) (h_yellow : yellow_chips = 7) :
  let total_chips := blue_chips + yellow_chips
  let p_blue := blue_chips / total_chips
  let p_yellow := yellow_chips / total_chips
  p_blue * p_yellow + p_yellow * p_blue = 35 / 72 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2334_233474


namespace NUMINAMATH_CALUDE_taxi_fare_for_100_miles_l2334_233429

/-- Represents the taxi fare system -/
structure TaxiFare where
  fixedCharge : ℝ
  fixedDistance : ℝ
  proportionalRate : ℝ

/-- Calculates the fare for a given distance -/
def calculateFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.fixedCharge + tf.proportionalRate * (distance - tf.fixedDistance)

theorem taxi_fare_for_100_miles 
  (tf : TaxiFare)
  (h1 : tf.fixedCharge = 20)
  (h2 : tf.fixedDistance = 10)
  (h3 : calculateFare tf 80 = 160) :
  calculateFare tf 100 = 200 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_for_100_miles_l2334_233429


namespace NUMINAMATH_CALUDE_climb_out_of_well_l2334_233481

/-- The number of days it takes for a man to climb out of a well -/
def daysToClimbWell (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let dailyProgress := climbUp - slipDown
  let daysForMostOfWell := (wellDepth - 1) / dailyProgress
  let remainingDistance := (wellDepth - 1) % dailyProgress
  if remainingDistance = 0 then
    daysForMostOfWell + 1
  else
    daysForMostOfWell + 2

/-- Theorem stating that it takes 30 days to climb out of a 30-meter well 
    when climbing 4 meters up and slipping 3 meters down each day -/
theorem climb_out_of_well : daysToClimbWell 30 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_climb_out_of_well_l2334_233481


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2334_233472

-- Define the cone's properties
def base_radius : ℝ := 6

-- Define the theorem
theorem cone_generatrix_length :
  ∀ (generatrix : ℝ),
  (2 * Real.pi * base_radius = Real.pi * generatrix) →
  generatrix = 12 := by
sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l2334_233472


namespace NUMINAMATH_CALUDE_range_of_a_l2334_233444

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 0 1 ∧ 2^x₀ * (3 * x₀ + a) < 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2334_233444


namespace NUMINAMATH_CALUDE_equation_solution_l2334_233417

theorem equation_solution : 
  ∃ y : ℝ, (4 : ℝ) * 8^3 = 4^y ∧ y = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2334_233417


namespace NUMINAMATH_CALUDE_father_age_proof_l2334_233405

/-- The age of the father -/
def father_age : ℕ := 48

/-- The age of the son -/
def son_age : ℕ := 75 - father_age

/-- The time difference between when the father was the son's current age and now -/
def time_difference : ℕ := father_age - son_age

theorem father_age_proof :
  (father_age + son_age = 75) ∧
  (father_age = 8 * (son_age - time_difference)) ∧
  (father_age - time_difference = son_age) →
  father_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_father_age_proof_l2334_233405
