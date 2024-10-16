import Mathlib

namespace NUMINAMATH_CALUDE_product_and_remainder_l3383_338398

theorem product_and_remainder (a b c d : ℤ) : 
  d = a * b * c → 
  1 < a → a < b → b < c → 
  233 % d = 79 → 
  a + c = 13 := by
sorry

end NUMINAMATH_CALUDE_product_and_remainder_l3383_338398


namespace NUMINAMATH_CALUDE_acute_angle_equation_l3383_338334

theorem acute_angle_equation (x : Real) : 
  x = π/3 → (Real.sin (2*x) + Real.cos x) * (Real.sin x - Real.cos x) = Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_equation_l3383_338334


namespace NUMINAMATH_CALUDE_sibling_age_difference_l3383_338300

/-- Given the ages of three siblings, prove the age difference between two of them. -/
theorem sibling_age_difference (juliet maggie ralph : ℕ) : 
  juliet > maggie ∧ 
  juliet = ralph - 2 ∧ 
  juliet = 10 ∧ 
  maggie + ralph = 19 → 
  juliet - maggie = 3 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_difference_l3383_338300


namespace NUMINAMATH_CALUDE_bug_meeting_point_l3383_338361

/-- Triangle PQR with side lengths PQ = 7, QR = 8, PR = 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (PQ_eq : PQ = 7)
  (QR_eq : QR = 8)
  (PR_eq : PR = 9)

/-- Point S where bugs meet -/
def S (t : Triangle) : ℝ := sorry

/-- QS is the distance from Q to S -/
def QS (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 -/
theorem bug_meeting_point (t : Triangle) : QS t = 5 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l3383_338361


namespace NUMINAMATH_CALUDE_sqrt_not_arithmetic_if_geometric_not_arithmetic_l3383_338316

theorem sqrt_not_arithmetic_if_geometric_not_arithmetic
  (a b c : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (geometric_sequence : b^2 = a * c)
  (not_arithmetic_sequence : ¬(a + c = 2 * b)) :
  ¬(Real.sqrt a + Real.sqrt c = 2 * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_not_arithmetic_if_geometric_not_arithmetic_l3383_338316


namespace NUMINAMATH_CALUDE_icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l3383_338379

/-- An icosidodecahedron is a convex polyhedron with 20 triangular faces and 12 pentagonal faces. -/
structure Icosidodecahedron where
  /-- The number of triangular faces -/
  triangular_faces : ℕ
  /-- The number of pentagonal faces -/
  pentagonal_faces : ℕ
  /-- The icosidodecahedron is a convex polyhedron -/
  is_convex : Bool
  /-- The number of triangular faces is 20 -/
  triangular_faces_eq : triangular_faces = 20
  /-- The number of pentagonal faces is 12 -/
  pentagonal_faces_eq : pentagonal_faces = 12

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices (i : Icosidodecahedron) : ℕ := 30

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices_proof (i : Icosidodecahedron) : 
  icosidodecahedron_vertices i = 30 := by
  sorry

end NUMINAMATH_CALUDE_icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l3383_338379


namespace NUMINAMATH_CALUDE_smallest_valid_m_l3383_338388

def is_valid (m : ℕ+) : Prop :=
  ∃ k₁ k₂ : ℕ, k₁ ≤ m ∧ k₂ ≤ m ∧ 
  (m^2 + m) % k₁ = 0 ∧ 
  (m^2 + m) % k₂ ≠ 0

theorem smallest_valid_m :
  (∀ m : ℕ+, m < 4 → ¬(is_valid m)) ∧ 
  is_valid 4 := by sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l3383_338388


namespace NUMINAMATH_CALUDE_sum_equals_5070_l3383_338329

theorem sum_equals_5070 (P : ℕ) : 
  1010 + 1012 + 1014 + 1016 + 1018 = 5100 - P → P = 30 :=
by sorry

end NUMINAMATH_CALUDE_sum_equals_5070_l3383_338329


namespace NUMINAMATH_CALUDE_infinite_good_pairs_l3383_338301

/-- A number is "good" if every prime factor in its prime factorization appears with an exponent of at least 2 -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p ^ k ∣ n)

/-- The sequence of "good" numbers -/
def good_sequence : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * good_sequence n * (good_sequence n + 1)

/-- Theorem stating the existence of infinitely many pairs of consecutive "good" numbers -/
theorem infinite_good_pairs :
  ∀ n : ℕ, is_good (good_sequence n) ∧ is_good (good_sequence n + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_good_pairs_l3383_338301


namespace NUMINAMATH_CALUDE_rectangle_area_l3383_338373

theorem rectangle_area (d : ℝ) (h : d > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = d^2 ∧ w * l = (3 / 10) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3383_338373


namespace NUMINAMATH_CALUDE_cubic_root_product_l3383_338337

theorem cubic_root_product : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 9 * x^2 + 5 * x - 10
  ∀ a b c : ℝ, f a = 0 → f b = 0 → f c = 0 → a * b * c = 10 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_product_l3383_338337


namespace NUMINAMATH_CALUDE_howard_purchase_l3383_338326

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents40 : ℕ
  dollars3 : ℕ
  dollars4 : ℕ

/-- The problem constraints -/
def satisfiesConstraints (counts : ItemCounts) : Prop :=
  counts.cents40 + counts.dollars3 + counts.dollars4 = 40 ∧
  40 * counts.cents40 + 300 * counts.dollars3 + 400 * counts.dollars4 = 5000

/-- The theorem to prove -/
theorem howard_purchase :
  ∃ (counts : ItemCounts), satisfiesConstraints counts ∧ counts.cents40 = 20 :=
by sorry

end NUMINAMATH_CALUDE_howard_purchase_l3383_338326


namespace NUMINAMATH_CALUDE_real_y_condition_l3383_338382

theorem real_y_condition (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 10 = 0) ↔ (x ≤ -17/9 ∨ x ≥ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l3383_338382


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l3383_338306

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of roles to be filled --/
def k : ℕ := 4

/-- The number of ways to arrange k people in k positions --/
def arrange (k : ℕ) : ℕ := Nat.factorial k

/-- The number of ways to choose k people from n people --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of ways to select and arrange volunteers for roles --/
def totalWays : ℕ :=
  arrange (k - 1) + choose (n - 1) (k - 1) * (k - 1) * arrange (k - 1)

theorem volunteer_selection_theorem : totalWays = 96 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l3383_338306


namespace NUMINAMATH_CALUDE_fraction_value_l3383_338342

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3383_338342


namespace NUMINAMATH_CALUDE_cassandra_pies_l3383_338303

/-- Calculates the number of apple pies Cassandra made -/
def number_of_pies (apples_bought : ℕ) (slices_per_pie : ℕ) (apples_per_slice : ℕ) : ℕ :=
  (apples_bought / apples_per_slice) / slices_per_pie

theorem cassandra_pies :
  let apples_bought := 4 * 12 -- four dozen
  let slices_per_pie := 6
  let apples_per_slice := 2
  number_of_pies apples_bought slices_per_pie apples_per_slice = 4 := by
  sorry

#eval number_of_pies (4 * 12) 6 2

end NUMINAMATH_CALUDE_cassandra_pies_l3383_338303


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l3383_338378

/-- Represents the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese_initial : ℕ) (house_initial : ℕ) (cats_left : ℕ) : ℕ :=
  siamese_initial + house_initial - cats_left

/-- Theorem stating that 19 cats were sold during the sale. -/
theorem cats_sold_during_sale :
  cats_sold 15 49 45 = 19 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l3383_338378


namespace NUMINAMATH_CALUDE_total_points_earned_l3383_338323

def enemies_defeated : ℕ := 6
def points_per_enemy : ℕ := 9
def level_completion_points : ℕ := 8

theorem total_points_earned :
  enemies_defeated * points_per_enemy + level_completion_points = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_points_earned_l3383_338323


namespace NUMINAMATH_CALUDE_wide_tall_difference_l3383_338362

/-- Represents a cupboard for storing glasses --/
structure Cupboard where
  capacity : ℕ

/-- Represents the collection of cupboards --/
structure CupboardCollection where
  tall : Cupboard
  wide : Cupboard
  narrow : Cupboard

/-- The problem setup --/
def setup : CupboardCollection where
  tall := { capacity := 20 }
  wide := { capacity := 0 }  -- We don't know the capacity, so we set it to 0
  narrow := { capacity := 10 }  -- After breaking one shelf

/-- The theorem to prove --/
theorem wide_tall_difference (w : ℕ) : 
  w = setup.wide.capacity → w - setup.tall.capacity = w - 20 := by
  sorry

/-- The main result --/
def result : ℕ → ℕ
  | w => w - 20

#check result

end NUMINAMATH_CALUDE_wide_tall_difference_l3383_338362


namespace NUMINAMATH_CALUDE_min_value_sum_l3383_338328

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a) + (a^2 * b) / (18 * b * c)) ≥ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l3383_338328


namespace NUMINAMATH_CALUDE_ursulas_salads_l3383_338318

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

end NUMINAMATH_CALUDE_ursulas_salads_l3383_338318


namespace NUMINAMATH_CALUDE_original_deck_size_l3383_338395

/-- Represents a deck of cards with red and black cards only -/
structure Deck where
  red : ℕ
  black : ℕ

/-- The probability of selecting a red card from the deck -/
def redProbability (d : Deck) : ℚ :=
  d.red / (d.red + d.black)

theorem original_deck_size :
  ∀ d : Deck,
  redProbability d = 1/4 →
  redProbability {red := d.red, black := d.black + 6} = 1/5 →
  d.red + d.black = 24 := by
sorry

end NUMINAMATH_CALUDE_original_deck_size_l3383_338395


namespace NUMINAMATH_CALUDE_baseball_card_ratio_l3383_338390

/-- Proves the ratio of cards Maria took to initial cards is 8:5 --/
theorem baseball_card_ratio : 
  ∀ (initial final maria_taken peter_given : ℕ),
  initial = 15 →
  peter_given = 1 →
  final = 18 →
  maria_taken = 3 * (initial - peter_given) - final →
  (maria_taken : ℚ) / initial = 8 / 5 := by
    sorry

end NUMINAMATH_CALUDE_baseball_card_ratio_l3383_338390


namespace NUMINAMATH_CALUDE_equal_utility_days_l3383_338372

/-- Daniel's utility function -/
def utility (reading : ℚ) (soccer : ℚ) : ℚ := reading * soccer

/-- Time spent on Wednesday -/
def wednesday (t : ℚ) : ℚ × ℚ := (10 - t, t)

/-- Time spent on Thursday -/
def thursday (t : ℚ) : ℚ × ℚ := (t + 4, 4 - t)

/-- The theorem stating that t = 8/5 makes the utility equal on both days -/
theorem equal_utility_days (t : ℚ) : 
  t = 8/5 ↔ 
  utility (wednesday t).1 (wednesday t).2 = utility (thursday t).1 (thursday t).2 := by
sorry

end NUMINAMATH_CALUDE_equal_utility_days_l3383_338372


namespace NUMINAMATH_CALUDE_freds_carrots_l3383_338364

/-- Given that Sally grew 6 carrots and the total number of carrots is 10,
    prove that Fred grew 4 carrots. -/
theorem freds_carrots (sally_carrots : ℕ) (total_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : total_carrots = 10) :
  total_carrots - sally_carrots = 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_carrots_l3383_338364


namespace NUMINAMATH_CALUDE_domain_of_sqrt_fraction_l3383_338313

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (Real.sqrt (8 - x))) ↔ x ∈ Set.Ici (-3) ∩ Set.Iio 8 := by
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_fraction_l3383_338313


namespace NUMINAMATH_CALUDE_factorization_example_l3383_338307

-- Define factorization
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the left-hand side of the equation
def lhs (m : ℝ) : ℝ := m^2 - 4

-- Define the right-hand side of the equation
def rhs (m : ℝ) : ℝ := (m + 2) * (m - 2)

-- Theorem statement
theorem factorization_example : is_factorization lhs rhs := by sorry

end NUMINAMATH_CALUDE_factorization_example_l3383_338307


namespace NUMINAMATH_CALUDE_farm_width_is_15km_l3383_338339

/-- A rectangular farm with given properties has a width of 15 kilometers. -/
theorem farm_width_is_15km (length width : ℝ) : 
  length > 0 →
  width > 0 →
  2 * (length + width) = 46 →
  width = length + 7 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_farm_width_is_15km_l3383_338339


namespace NUMINAMATH_CALUDE_melanie_dimes_given_to_dad_l3383_338333

theorem melanie_dimes_given_to_dad (initial_dimes : ℕ) (dimes_from_mother : ℕ) (final_dimes : ℕ) :
  initial_dimes = 7 →
  dimes_from_mother = 4 →
  final_dimes = 3 →
  initial_dimes + dimes_from_mother - final_dimes = 8 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_given_to_dad_l3383_338333


namespace NUMINAMATH_CALUDE_point_transformation_l3383_338321

/-- Rotates a point (x, y) by 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualsX rotated.1 rotated.2
  final = (5, -1) → b - a = -4 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l3383_338321


namespace NUMINAMATH_CALUDE_fourth_root_equation_l3383_338304

theorem fourth_root_equation (y : ℝ) :
  (y * (y^5)^(1/2))^(1/4) = 4 → y = 2^(16/7) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l3383_338304


namespace NUMINAMATH_CALUDE_train_speed_problem_l3383_338302

/-- Proves that given a train journey where the distance is covered in 276 minutes
    at speed S1, and the same distance can be covered in 69 minutes at 16 kmph,
    then S1 = 4 kmph -/
theorem train_speed_problem (S1 : ℝ) : 
  (276 : ℝ) * S1 = 69 * 16 → S1 = 4 := by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3383_338302


namespace NUMINAMATH_CALUDE_pond_area_l3383_338376

/-- Given a square garden with a perimeter of 48 meters and an area not occupied by a pond of 124 square meters, the area of the pond is 20 square meters. -/
theorem pond_area (garden_perimeter : ℝ) (non_pond_area : ℝ) : 
  garden_perimeter = 48 →
  non_pond_area = 124 →
  (garden_perimeter / 4)^2 - non_pond_area = 20 := by
sorry

end NUMINAMATH_CALUDE_pond_area_l3383_338376


namespace NUMINAMATH_CALUDE_symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l3383_338351

-- Definition of symmetric equations
def is_symmetric (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₁ + a₂ = 0 ∧ b₁ = b₂ ∧ c₁ + c₂ = 0

-- Theorem 1: Symmetric equation of x² - 4x + 3 = 0
theorem symmetric_equation_example : 
  is_symmetric 1 (-4) 3 (-1) (-4) (-3) :=
sorry

-- Theorem 2: Finding m and n for symmetric equations
theorem symmetric_equation_values (m n : ℝ) :
  is_symmetric 3 (m - 1) (-n) (-3) (-1) 1 → m = 0 ∧ n = 1 :=
sorry

-- Theorem 3: Solutions of the quadratic equation
theorem quadratic_equation_solutions :
  let x₁ := (1 + Real.sqrt 13) / 6
  let x₂ := (1 - Real.sqrt 13) / 6
  3 * x₁^2 - x₁ - 1 = 0 ∧ 3 * x₂^2 - x₂ - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_equation_example_symmetric_equation_values_quadratic_equation_solutions_l3383_338351


namespace NUMINAMATH_CALUDE_l_shaped_area_specific_l3383_338389

/-- Calculates the area of an 'L'-shaped figure formed by removing a smaller rectangle from a larger rectangle. -/
def l_shaped_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

theorem l_shaped_area_specific : l_shaped_area 10 7 4 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_specific_l3383_338389


namespace NUMINAMATH_CALUDE_quadratic_solution_l3383_338331

theorem quadratic_solution : ∃ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3383_338331


namespace NUMINAMATH_CALUDE_purchase_plan_monthly_payment_l3383_338355

theorem purchase_plan_monthly_payment 
  (purchase_price : ℝ) 
  (down_payment : ℝ) 
  (num_payments : ℕ) 
  (interest_rate : ℝ) 
  (h1 : purchase_price = 118)
  (h2 : down_payment = 18)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 0.15254237288135593) :
  let total_interest : ℝ := purchase_price * interest_rate
  let total_paid : ℝ := purchase_price + total_interest
  let monthly_payment : ℝ := (total_paid - down_payment) / num_payments
  monthly_payment = 9.833333333333334 := by sorry

end NUMINAMATH_CALUDE_purchase_plan_monthly_payment_l3383_338355


namespace NUMINAMATH_CALUDE_symmetrical_point_not_in_third_quadrant_l3383_338341

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the symmetrical point with respect to the y-axis -/
def symmetricalPointY (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem to prove -/
theorem symmetrical_point_not_in_third_quadrant :
  let p := Point.mk (-3) 4
  let symmetricalP := symmetricalPointY p
  ¬(isInThirdQuadrant symmetricalP) := by
  sorry


end NUMINAMATH_CALUDE_symmetrical_point_not_in_third_quadrant_l3383_338341


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l3383_338347

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 29 →                 -- Median is 29
  c = b + 7 →              -- Largest number is 7 more than median
  a < b ∧ b < c →          -- Ensuring order: a < b < c
  a = 25 :=                -- Smallest number is 25
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l3383_338347


namespace NUMINAMATH_CALUDE_area_MNKP_l3383_338320

/-- The area of quadrilateral MNKP given the area of quadrilateral ABCD -/
theorem area_MNKP (S_ABCD : ℝ) (h1 : S_ABCD = (180 + 50 * Real.sqrt 3) / 6)
  (h2 : ∃ S_MNKP : ℝ, S_MNKP = S_ABCD / 2) :
  ∃ S_MNKP : ℝ, S_MNKP = (90 + 25 * Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_MNKP_l3383_338320


namespace NUMINAMATH_CALUDE_total_price_theorem_l3383_338354

/-- The price of a pear in dollars -/
def pear_price : ℕ := 90

/-- The total cost of an orange and a pear in dollars -/
def orange_pear_total : ℕ := 120

/-- The price of an orange in dollars -/
def orange_price : ℕ := orange_pear_total - pear_price

/-- The price of a banana in dollars -/
def banana_price : ℕ := pear_price - orange_price

/-- The number of bananas to buy -/
def num_bananas : ℕ := 200

/-- The number of oranges to buy -/
def num_oranges : ℕ := 2 * num_bananas

theorem total_price_theorem : 
  banana_price * num_bananas + orange_price * num_oranges = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_price_theorem_l3383_338354


namespace NUMINAMATH_CALUDE_mixed_number_properties_l3383_338391

/-- Represents a mixed number as a pair of integers (whole, numerator, denominator) -/
structure MixedNumber where
  whole : ℤ
  numerator : ℕ
  denominator : ℕ
  h_pos : denominator > 0
  h_proper : numerator < denominator

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Converts a mixed number to a rational number -/
def mixed_to_rational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem mixed_number_properties (m : MixedNumber) 
  (h_m : m = ⟨3, 2, 7, by norm_num, by norm_num⟩) : 
  ∃ (fractional_unit : ℚ) (num_units : ℕ) (units_to_add : ℕ),
    fractional_unit = 1 / 7 ∧ 
    num_units = 23 ∧
    units_to_add = 5 ∧
    mixed_to_rational m = num_units * fractional_unit ∧
    mixed_to_rational m + units_to_add * fractional_unit = smallest_composite := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_properties_l3383_338391


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3383_338324

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (7, -5),
    the sum of coordinates of the other endpoint is -10. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (10 + x) / 2 = 7 → 
  (4 + y) / 2 = -5 → 
  x + y = -10 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3383_338324


namespace NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_plus_one_equals_negative_three_l3383_338336

theorem x_squared_minus_four_y_squared_plus_one_equals_negative_three 
  (x y : ℝ) (h1 : x + 2*y = 4) (h2 : x - 2*y = -1) : 
  x^2 - 4*y^2 + 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_four_y_squared_plus_one_equals_negative_three_l3383_338336


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3383_338352

theorem tan_45_degrees : Real.tan (45 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3383_338352


namespace NUMINAMATH_CALUDE_probability_one_black_ball_l3383_338369

/-- The probability of drawing exactly one black ball when drawing two balls without replacement from a box containing 3 white balls and 2 black balls -/
theorem probability_one_black_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 5)
  (h3 : white_balls = 3)
  (h4 : black_balls = 2) : 
  (white_balls * black_balls) / ((total_balls * (total_balls - 1)) / 2) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_one_black_ball_l3383_338369


namespace NUMINAMATH_CALUDE_line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l3383_338363

-- Define points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (3, 5)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (-1, -1)
def E : ℝ × ℝ := (0, 4)
def K : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (3, 2)
def P : ℝ × ℝ := (6, 3)

-- Define line equations
def line1 (x : ℝ) : Prop := x = 3
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -2 * x + 4
def line4 (x y : ℝ) : Prop := y = (1/3) * x + 1

-- Theorem statements
theorem line_through_A_and_B : 
  ∀ x y : ℝ, (x, y) = A ∨ (x, y) = B → line1 x := by sorry

theorem line_through_C_and_D : 
  ∀ x y : ℝ, (x, y) = C ∨ (x, y) = D → line2 x y := by sorry

theorem line_through_E_and_K : 
  ∀ x y : ℝ, (x, y) = E ∨ (x, y) = K → line3 x y := by sorry

theorem line_through_M_and_P : 
  ∀ x y : ℝ, (x, y) = M ∨ (x, y) = P → line4 x y := by sorry

end NUMINAMATH_CALUDE_line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l3383_338363


namespace NUMINAMATH_CALUDE_digital_earth_correct_application_l3383_338338

/-- Represents the capabilities of the digital Earth -/
structure DigitalEarthCapabilities where
  resourceOptimization : Bool
  informationAccess : Bool

/-- Represents possible applications of the digital Earth -/
inductive DigitalEarthApplication
  | crimeControl
  | projectDecisionSupport
  | precipitationControl
  | disasterControl

/-- Determines if an application is correct given the capabilities of the digital Earth -/
def isCorrectApplication (capabilities : DigitalEarthCapabilities) (application : DigitalEarthApplication) : Prop :=
  capabilities.resourceOptimization ∧ 
  capabilities.informationAccess ∧ 
  application = DigitalEarthApplication.projectDecisionSupport

theorem digital_earth_correct_application (capabilities : DigitalEarthCapabilities) 
  (h1 : capabilities.resourceOptimization = true) 
  (h2 : capabilities.informationAccess = true) :
  isCorrectApplication capabilities DigitalEarthApplication.projectDecisionSupport :=
sorry

end NUMINAMATH_CALUDE_digital_earth_correct_application_l3383_338338


namespace NUMINAMATH_CALUDE_people_lifting_weights_l3383_338319

/-- The number of people in the gym at the start of Bethany's shift -/
def initial_people : ℕ := sorry

/-- The number of people who arrived during Bethany's shift -/
def arrivals : ℕ := 5

/-- The number of people who left during Bethany's shift -/
def departures : ℕ := 2

/-- The total number of people in the gym after the changes -/
def final_people : ℕ := 19

theorem people_lifting_weights : initial_people = 16 :=
  by sorry

end NUMINAMATH_CALUDE_people_lifting_weights_l3383_338319


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l3383_338315

/-- The ratio of the total volume of two cones to the volume of a cylinder -/
theorem cone_cylinder_volume_ratio :
  let r : ℝ := 4 -- radius of cylinder and cones
  let h_cyl : ℝ := 18 -- height of cylinder
  let h_cone1 : ℝ := 6 -- height of first cone
  let h_cone2 : ℝ := 9 -- height of second cone
  let v_cyl := π * r^2 * h_cyl -- volume of cylinder
  let v_cone1 := (1/3) * π * r^2 * h_cone1 -- volume of first cone
  let v_cone2 := (1/3) * π * r^2 * h_cone2 -- volume of second cone
  let v_cones := v_cone1 + v_cone2 -- total volume of cones
  v_cones / v_cyl = 5 / 18 := by
sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l3383_338315


namespace NUMINAMATH_CALUDE_factorization_problem1_l3383_338387

theorem factorization_problem1 (a b : ℝ) :
  -3 * a^3 + 12 * a^2 * b - 12 * a * b^2 = -3 * a * (a - 2*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3383_338387


namespace NUMINAMATH_CALUDE_start_ratio_black_to_white_l3383_338310

/-- The ratio of black rats to white rats that reached the goal -/
def black_rat_success_rate : ℚ := 56 / 100

/-- The ratio of white rats to white rats that reached the goal -/
def white_rat_success_rate : ℚ := 84 / 100

/-- The ratio of black rats to white rats at the goal -/
def goal_ratio : ℚ := 1 / 2

/-- Theorem stating the ratio of black rats to white rats at the start -/
theorem start_ratio_black_to_white :
  ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧
  (black_rat_success_rate * x) / (white_rat_success_rate * y) = goal_ratio ∧
  x / y = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_start_ratio_black_to_white_l3383_338310


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l3383_338358

/-- Represents a four-digit number in the form x28x --/
def fourDigitNumber (x : ℕ) : ℕ := x * 1000 + 280 + x

/-- Checks if a natural number is a single digit (0-9) --/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem unique_divisible_by_18 :
  ∃! x : ℕ, isSingleDigit x ∧ (fourDigitNumber x % 18 = 0) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l3383_338358


namespace NUMINAMATH_CALUDE_circle_area_irrational_if_rational_diameter_l3383_338340

theorem circle_area_irrational_if_rational_diameter :
  ∀ d : ℚ, d > 0 → ∃ A : ℝ, A = π * (d / 2)^2 ∧ Irrational A := by
  sorry

end NUMINAMATH_CALUDE_circle_area_irrational_if_rational_diameter_l3383_338340


namespace NUMINAMATH_CALUDE_expression_factorization_l3383_338312

theorem expression_factorization (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l3383_338312


namespace NUMINAMATH_CALUDE_distance_from_negative_three_point_two_l3383_338371

theorem distance_from_negative_three_point_two (x : ℝ) : 
  (|x + 3.2| = 4) ↔ (x = 0.8 ∨ x = -7.2) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_three_point_two_l3383_338371


namespace NUMINAMATH_CALUDE_total_paint_used_l3383_338356

/-- The amount of paint Joe uses at two airports over two weeks -/
def paint_used (paint1 paint2 : ℝ) (week1_ratio1 week2_ratio1 week1_ratio2 week2_ratio2 : ℝ) : ℝ :=
  let remaining1 := paint1 * (1 - week1_ratio1)
  let used1 := paint1 * week1_ratio1 + remaining1 * week2_ratio1
  let remaining2 := paint2 * (1 - week1_ratio2)
  let used2 := paint2 * week1_ratio2 + remaining2 * week2_ratio2
  used1 + used2

/-- Theorem stating the total amount of paint Joe uses at both airports -/
theorem total_paint_used :
  paint_used 360 600 (1/4) (1/6) (1/3) (1/5) = 415 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_used_l3383_338356


namespace NUMINAMATH_CALUDE_competition_total_students_l3383_338311

theorem competition_total_students 
  (rank_from_right : Nat) 
  (rank_from_left : Nat) 
  (h1 : rank_from_right = 18) 
  (h2 : rank_from_left = 12) : 
  rank_from_right + rank_from_left - 1 = 29 := by
sorry

end NUMINAMATH_CALUDE_competition_total_students_l3383_338311


namespace NUMINAMATH_CALUDE_arccos_one_eq_zero_l3383_338308

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_eq_zero_l3383_338308


namespace NUMINAMATH_CALUDE_triangle_side_length_l3383_338397

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Side length b is 2√2
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3383_338397


namespace NUMINAMATH_CALUDE_max_additional_spheres_is_two_l3383_338314

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents the configuration of spheres in the truncated cone -/
structure SphereConfiguration where
  cone : TruncatedCone
  O₁ : Sphere
  O₂ : Sphere

/-- Calculates the maximum number of additional spheres that can be placed in the cone -/
def maxAdditionalSpheres (config : SphereConfiguration) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_additional_spheres_is_two (config : SphereConfiguration) :
  config.cone.height = 8 ∧
  config.O₁.radius = 2 ∧
  config.O₂.radius = 3 →
  maxAdditionalSpheres config = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_spheres_is_two_l3383_338314


namespace NUMINAMATH_CALUDE_coin_value_calculation_l3383_338377

theorem coin_value_calculation (total_coins : ℕ) (dimes : ℕ) (nickels : ℕ) : 
  total_coins = 36 → 
  dimes = 26 → 
  nickels = total_coins - dimes → 
  (dimes * 10 + nickels * 5 : ℚ) / 100 = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l3383_338377


namespace NUMINAMATH_CALUDE_union_of_sets_l3383_338348

/-- Given sets M and N, prove that their union is equal to the set of all x between -1 and 5 inclusive -/
theorem union_of_sets (M N : Set ℝ) (hM : M = {x : ℝ | -1 ≤ x ∧ x < 3}) (hN : N = {x : ℝ | 2 < x ∧ x ≤ 5}) :
  M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3383_338348


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_l3383_338366

theorem sum_of_squared_differences : (302^2 - 298^2) + (152^2 - 148^2) = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_l3383_338366


namespace NUMINAMATH_CALUDE_product_units_digit_base6_l3383_338330

/-- The units digit in base 6 of a number -/
def unitsDigitBase6 (n : ℕ) : ℕ := n % 6

/-- The product of 168 and 59 -/
def product : ℕ := 168 * 59

theorem product_units_digit_base6 :
  unitsDigitBase6 product = 0 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base6_l3383_338330


namespace NUMINAMATH_CALUDE_ab_nonpositive_l3383_338393

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l3383_338393


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3383_338305

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 1 - Complex.I → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3383_338305


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3383_338375

theorem division_remainder_problem (larger smaller : ℕ) : 
  larger - smaller = 1515 →
  larger = 1600 →
  larger / smaller = 16 →
  larger % smaller = 240 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3383_338375


namespace NUMINAMATH_CALUDE_simplify_expression_l3383_338365

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 6) - (2*x + 7)*(3*x - 2) = -7*x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3383_338365


namespace NUMINAMATH_CALUDE_factorization_coefficient_sum_l3383_338394

theorem factorization_coefficient_sum : 
  ∃ (A B C D E F G H J K : ℤ),
    (125 : ℤ) * X^9 - 216 * Y^9 = 
      (A * X + B * Y) * 
      (C * X^3 + D * X * Y^2 + E * Y^3) * 
      (F * X + G * Y) * 
      (H * X^3 + J * X * Y^2 + K * Y^3) ∧
    A + B + C + D + E + F + G + H + J + K = 24 :=
by sorry

end NUMINAMATH_CALUDE_factorization_coefficient_sum_l3383_338394


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l3383_338385

/-- Regression model for child height based on age -/
def height_model (x : ℝ) : ℝ := 7.2 * x + 74

/-- The model is valid for children aged 3 to 9 years -/
def valid_age_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

/-- Prediction is considered approximate if within 1cm of the calculated value -/
def is_approximate (predicted : ℝ) (actual : ℝ) : Prop := abs (predicted - actual) ≤ 1

theorem height_prediction_at_10 :
  is_approximate (height_model 10) 146 :=
sorry

end NUMINAMATH_CALUDE_height_prediction_at_10_l3383_338385


namespace NUMINAMATH_CALUDE_yacht_distance_squared_l3383_338370

theorem yacht_distance_squared (AB BC : ℝ) (angle_B : ℝ) (AC_squared : ℝ) : 
  AB = 15 → 
  BC = 25 → 
  angle_B = 150 * Real.pi / 180 →
  AC_squared = AB^2 + BC^2 - 2 * AB * BC * Real.cos angle_B →
  AC_squared = 850 - 375 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_yacht_distance_squared_l3383_338370


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l3383_338357

def number_to_factorize : Nat := 30030

/-- The number of positive divisors of 30030 is 64 -/
theorem number_of_divisors_30030 : 
  (Nat.divisors number_to_factorize).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l3383_338357


namespace NUMINAMATH_CALUDE_factorization_1_l3383_338380

theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_l3383_338380


namespace NUMINAMATH_CALUDE_best_approximation_l3383_338360

def f (x : ℝ) := x^2 + 2*x

def table_values : List ℝ := [1.63, 1.64, 1.65, 1.66]

def target_value : ℝ := 6

theorem best_approximation :
  ∀ x ∈ table_values, 
    abs (f 1.65 - target_value) ≤ abs (f x - target_value) ∧
    (∀ y ∈ table_values, abs (f y - target_value) < abs (f 1.65 - target_value) → y = 1.65) :=
by sorry

end NUMINAMATH_CALUDE_best_approximation_l3383_338360


namespace NUMINAMATH_CALUDE_sin_15_times_sin_75_l3383_338399

theorem sin_15_times_sin_75 : Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_times_sin_75_l3383_338399


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l3383_338346

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 1 ∨ a = -1 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-1) 1, m^2 - |m| - f x a < 0) → -2 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_implies_m_range_l3383_338346


namespace NUMINAMATH_CALUDE_intersection_M_N_l3383_338322

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | Real.log x / Real.log 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3383_338322


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3383_338381

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 3, 5, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3383_338381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_number_remainder_l3383_338327

/-- A function that generates a number from an arithmetic sequence of digits -/
def arithmeticSequenceNumber (firstDigit : Nat) (commonDifference : Nat) (length : Nat) : Nat :=
  sorry

/-- The remainder when dividing by 47 -/
def modulo47 (n : Nat) : Nat :=
  n % 47

theorem arithmetic_sequence_number_remainder :
  ∀ n : Nat,
    (n > 0) →
    (arithmeticSequenceNumber 8 2 4 = n) →
    (modulo47 n = 16) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_number_remainder_l3383_338327


namespace NUMINAMATH_CALUDE_stick_length_average_l3383_338325

theorem stick_length_average (total_sticks : ℕ) (all_avg : ℝ) (two_avg : ℝ) :
  total_sticks = 11 →
  all_avg = 145.7 →
  two_avg = 142.1 →
  let remaining_sticks := total_sticks - 2
  let total_length := all_avg * total_sticks
  let two_length := two_avg * 2
  let remaining_length := total_length - two_length
  remaining_length / remaining_sticks = 146.5 := by
sorry

end NUMINAMATH_CALUDE_stick_length_average_l3383_338325


namespace NUMINAMATH_CALUDE_marble_difference_l3383_338332

theorem marble_difference : ∀ (total_marbles : ℕ),
  -- Conditions
  (total_marbles > 0) →  -- Ensure there are marbles
  (∃ (blue1 green1 blue2 green2 : ℕ),
    -- Jar 1 ratio
    7 * green1 = 3 * blue1 ∧
    -- Jar 2 ratio
    5 * green2 = 4 * blue2 ∧
    -- Same total in each jar
    blue1 + green1 = blue2 + green2 ∧
    -- Total green marbles
    green1 + green2 = 140 ∧
    -- Total marbles in each jar
    blue1 + green1 = total_marbles) →
  -- Conclusion
  ∃ (blue1 blue2 : ℕ), blue1 - blue2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3383_338332


namespace NUMINAMATH_CALUDE_solve_for_a_l3383_338367

theorem solve_for_a (a b d : ℤ) 
  (eq1 : a + b = d) 
  (eq2 : b + d = 7) 
  (eq3 : d = 4) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l3383_338367


namespace NUMINAMATH_CALUDE_games_for_512_players_l3383_338383

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of games needed to determine the champion in a single-elimination tournament -/
def games_to_champion (t : SingleEliminationTournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to determine the champion -/
theorem games_for_512_players :
  let t : SingleEliminationTournament := ⟨512, by norm_num⟩
  games_to_champion t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_512_players_l3383_338383


namespace NUMINAMATH_CALUDE_vegan_soy_free_menu_fraction_l3383_338386

theorem vegan_soy_free_menu_fraction 
  (total_menu : ℕ) 
  (vegan_fraction : Rat) 
  (soy_containing_vegan_fraction : Rat) 
  (h1 : vegan_fraction = 1 / 10) 
  (h2 : soy_containing_vegan_fraction = 2 / 3) : 
  (1 - soy_containing_vegan_fraction) * vegan_fraction = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_vegan_soy_free_menu_fraction_l3383_338386


namespace NUMINAMATH_CALUDE_f_is_cubic_l3383_338392

/-- A polynomial function of degree 4 -/
def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄

/-- The function reaches its maximum at x = -1 -/
def max_at_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x ≤ f a₀ a₁ a₂ a₃ a₄ (-1)

/-- The graph of y = f(x + 1) is symmetric about (-1, 0) -/
def symmetric_about_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ (x + 1) = f a₀ a₁ a₂ a₃ a₄ (-x + 1)

theorem f_is_cubic (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  max_at_neg_one a₀ a₁ a₂ a₃ a₄ →
  symmetric_about_neg_one a₀ a₁ a₂ a₃ a₄ →
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x = x^3 - x :=
sorry

end NUMINAMATH_CALUDE_f_is_cubic_l3383_338392


namespace NUMINAMATH_CALUDE_selenas_book_pages_l3383_338359

theorem selenas_book_pages : 
  ∀ (s : ℕ), 
  (s / 2 : ℕ) - 20 = 180 → 
  s = 400 := by
sorry

end NUMINAMATH_CALUDE_selenas_book_pages_l3383_338359


namespace NUMINAMATH_CALUDE_equivalence_of_divisibility_conditions_l3383_338349

theorem equivalence_of_divisibility_conditions (f : ℕ → ℕ) :
  (∀ m n : ℕ+, m ≤ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) ↔
  (∀ m n : ℕ+, m ≥ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_of_divisibility_conditions_l3383_338349


namespace NUMINAMATH_CALUDE_infinitely_many_zeros_sin_log_l3383_338345

/-- The function f(x) = sin(log x) has infinitely many zeros in the interval (0, 1) -/
theorem infinitely_many_zeros_sin_log :
  ∃ (S : Set ℝ), S.Infinite ∧ S ⊆ Set.Ioo (0 : ℝ) 1 ∧ ∀ x ∈ S, Real.sin (Real.log x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_zeros_sin_log_l3383_338345


namespace NUMINAMATH_CALUDE_coffee_price_coffee_price_is_12_l3383_338350

/-- The regular price for a half-pound of coffee, given a 25% discount and 
    quarter-pound bags sold for $4.50 after the discount. -/
theorem coffee_price : ℝ :=
  let discount_rate : ℝ := 0.25
  let discounted_price_quarter_pound : ℝ := 4.50
  let regular_price_half_pound : ℝ := 12

  have h1 : discounted_price_quarter_pound = 
    regular_price_half_pound / 2 * (1 - discount_rate) := by sorry

  regular_price_half_pound

/-- Proof that the regular price for a half-pound of coffee is $12 -/
theorem coffee_price_is_12 : coffee_price = 12 := by sorry

end NUMINAMATH_CALUDE_coffee_price_coffee_price_is_12_l3383_338350


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3383_338344

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | (x-2)*(x-5) < 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3383_338344


namespace NUMINAMATH_CALUDE_income_comparison_l3383_338309

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by sorry

end NUMINAMATH_CALUDE_income_comparison_l3383_338309


namespace NUMINAMATH_CALUDE_school_emblem_estimate_l3383_338374

/-- Estimates the number of students who like a design in the entire school population
    based on a sample survey. -/
def estimate_liking (total_students : ℕ) (sample_size : ℕ) (sample_liking : ℕ) : ℕ :=
  (sample_liking * total_students) / sample_size

/-- Theorem stating that the estimated number of students liking design A
    in a school of 2000 students is 1200, given a survey where 60 out of 100
    students liked design A. -/
theorem school_emblem_estimate :
  let total_students : ℕ := 2000
  let sample_size : ℕ := 100
  let sample_liking : ℕ := 60
  estimate_liking total_students sample_size sample_liking = 1200 := by
sorry

end NUMINAMATH_CALUDE_school_emblem_estimate_l3383_338374


namespace NUMINAMATH_CALUDE_expression_equality_l3383_338396

theorem expression_equality : 
  (2^3 ≠ 3^2) ∧ 
  ((-2)^3 ≠ (-3)^2) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  ((-2)^3 = (-2^3)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3383_338396


namespace NUMINAMATH_CALUDE_larger_number_proof_l3383_338343

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 5) : L = 1637 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3383_338343


namespace NUMINAMATH_CALUDE_even_number_divisor_sum_l3383_338368

theorem even_number_divisor_sum (n : ℕ) : 
  Even n →
  (∃ (divs : Finset ℕ), divs = {d : ℕ | d ∣ n} ∧ 
    (divs.sum (λ d => (1 : ℚ) / d) = 1620 / 1003)) →
  ∃ k : ℕ, n = 2006 * k :=
by sorry

end NUMINAMATH_CALUDE_even_number_divisor_sum_l3383_338368


namespace NUMINAMATH_CALUDE_min_skew_edge_distance_l3383_338353

/-- A regular octahedron with edge length a -/
structure RegularOctahedron (a : ℝ) where
  edge_length : a > 0

/-- A point on an edge of the octahedron -/
structure EdgePoint (O : RegularOctahedron a) where
  -- Additional properties can be added if needed

/-- The distance between two points on skew edges of the octahedron -/
def skew_edge_distance (O : RegularOctahedron a) (p q : EdgePoint O) : ℝ := sorry

/-- The theorem stating the minimal distance between points on skew edges -/
theorem min_skew_edge_distance (a : ℝ) (O : RegularOctahedron a) :
  ∃ (p q : EdgePoint O), 
    skew_edge_distance O p q = a * Real.sqrt 6 / 3 ∧
    ∀ (r s : EdgePoint O), skew_edge_distance O r s ≥ a * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_skew_edge_distance_l3383_338353


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3383_338384

/-- The ratio of the area of a rectangle to the area of a triangle -/
theorem rectangle_triangle_area_ratio 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ) 
  (triangle_area : ℝ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 4)
  (h3 : triangle_area = 60) :
  (rectangle_length * rectangle_width) / triangle_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l3383_338384


namespace NUMINAMATH_CALUDE_problem_statement_l3383_338335

/-- Given a function g : ℝ → ℝ satisfying the following conditions:
  1) For all x y : ℝ, 2 * x * g y = 3 * y * g x
  2) g 10 = 15
  Prove that g 5 = 45/4 -/
theorem problem_statement (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x) 
  (h2 : g 10 = 15) : 
  g 5 = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3383_338335


namespace NUMINAMATH_CALUDE_fraction_simplification_l3383_338317

theorem fraction_simplification : (1922^2 - 1913^2) / (1930^2 - 1905^2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3383_338317
