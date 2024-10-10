import Mathlib

namespace root_value_theorem_l554_55489

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 1 = 0 → 4 * m^2 - 6 * m + 2021 = 2023 := by
  sorry

end root_value_theorem_l554_55489


namespace save_fraction_is_one_seventh_l554_55418

/-- Represents the worker's financial situation over a year --/
structure WorkerFinances where
  monthly_pay : ℝ
  save_fraction : ℝ
  months : ℕ := 12

/-- The conditions of the problem as described --/
def valid_finances (w : WorkerFinances) : Prop :=
  w.monthly_pay > 0 ∧
  w.save_fraction > 0 ∧
  w.save_fraction < 1 ∧
  w.months * w.save_fraction * w.monthly_pay = 2 * (1 - w.save_fraction) * w.monthly_pay

/-- The main theorem stating that the save fraction is 1/7 --/
theorem save_fraction_is_one_seventh (w : WorkerFinances) 
  (h : valid_finances w) : w.save_fraction = 1 / 7 := by
  sorry

#check save_fraction_is_one_seventh

end save_fraction_is_one_seventh_l554_55418


namespace orthocenters_collinear_l554_55487

-- Define the basic geometric objects
variable (A B C D O : Point)

-- Define the quadrilateral ABCD
def quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle O
def inscribedCircle (O : Point) (A B C D : Point) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (P Q R : Point) : Point := sorry

-- Define collinearity of points
def collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem orthocenters_collinear 
  (h1 : quadrilateral A B C D) 
  (h2 : inscribedCircle O A B C D) : 
  collinear 
    (orthocenter O A B) 
    (orthocenter O B C) 
    (orthocenter O C D) ∧ 
  collinear 
    (orthocenter O C D) 
    (orthocenter O D A) 
    (orthocenter O A B) :=
sorry

end orthocenters_collinear_l554_55487


namespace range_of_y_l554_55460

theorem range_of_y (y : ℝ) (h1 : 1 / y < 3) (h2 : 1 / y > -4) : y > 1 / 3 := by
  sorry

end range_of_y_l554_55460


namespace cash_percentage_proof_l554_55426

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def machinery : ℝ := 200

theorem cash_percentage_proof :
  let spent := raw_materials + machinery
  let cash := total_amount - spent
  let percentage := (cash / total_amount) * 100
  ∀ ε > 0, |percentage - 29.99| < ε :=
by sorry

end cash_percentage_proof_l554_55426


namespace smallest_integer_satisfying_inequality_l554_55494

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 15*m + 56 ≤ 0 → n ≤ m) ∧ (n^2 - 15*n + 56 ≤ 0) ∧ n = 7 := by
  sorry

end smallest_integer_satisfying_inequality_l554_55494


namespace barn_hoot_difference_l554_55485

/-- The number of hoots one barnyard owl makes per minute -/
def hoots_per_owl : ℕ := 5

/-- The number of hoots heard per minute from the barn -/
def hoots_heard : ℕ := 20

/-- The number of owls we're comparing to -/
def num_owls : ℕ := 3

/-- The difference between the hoots heard and the hoots from a specific number of owls -/
def hoot_difference (heard : ℕ) (owls : ℕ) : ℤ :=
  heard - (owls * hoots_per_owl)

theorem barn_hoot_difference :
  hoot_difference hoots_heard num_owls = 5 := by
  sorry

end barn_hoot_difference_l554_55485


namespace rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l554_55401

-- Define the set of real number roots of x^2 = 2
def rootsOfTwo : Set ℝ := {x : ℝ | x^2 = 2}

-- Theorem stating that rootsOfTwo is a well-defined set
theorem rootsOfTwo_is_well_defined_set : 
  ∃ (S : Set ℝ), S = rootsOfTwo ∧ (∀ x : ℝ, x ∈ S ↔ x^2 = 2) :=
by
  sorry

-- Theorem stating that rootsOfTwo contains exactly two elements
theorem rootsOfTwo_has_two_elements :
  ∃ (a b : ℝ), a ≠ b ∧ rootsOfTwo = {a, b} :=
by
  sorry

end rootsOfTwo_is_well_defined_set_rootsOfTwo_has_two_elements_l554_55401


namespace average_problem_l554_55471

theorem average_problem (x : ℝ) (h : (47 + x) / 2 = 53) : 
  x = 59 ∧ |x - 47| = 12 ∧ x + 47 = 106 := by
  sorry

end average_problem_l554_55471


namespace single_elimination_tournament_games_l554_55444

/-- The number of games required in a single-elimination tournament -/
def games_required (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 21 teams, 20 games are required to declare a winner -/
theorem single_elimination_tournament_games :
  games_required 21 = 20 := by
  sorry

end single_elimination_tournament_games_l554_55444


namespace cherry_tomatoes_weight_l554_55492

/-- Calculates the total weight of cherry tomatoes in grams -/
def total_weight_grams (initial_kg : ℝ) (additional_g : ℝ) : ℝ :=
  initial_kg * 1000 + additional_g

/-- Theorem: The total weight of cherry tomatoes is 2560 grams -/
theorem cherry_tomatoes_weight :
  total_weight_grams 2 560 = 2560 := by
  sorry

end cherry_tomatoes_weight_l554_55492


namespace cos_arcsin_eight_seventeenths_l554_55497

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end cos_arcsin_eight_seventeenths_l554_55497


namespace prob_success_constant_l554_55456

/-- Represents the probability of finding the correct key on the kth attempt
    given n total keys. -/
def prob_success (n : ℕ) (k : ℕ) : ℚ :=
  if 1 ≤ k ∧ k ≤ n then 1 / n else 0

/-- Theorem stating that the probability of success on any attempt
    is 1/n for any valid k. -/
theorem prob_success_constant (n : ℕ) (k : ℕ) (h1 : n > 0) (h2 : 1 ≤ k) (h3 : k ≤ n) :
  prob_success n k = 1 / n :=
by sorry

end prob_success_constant_l554_55456


namespace number_of_daughters_l554_55448

theorem number_of_daughters (a : ℕ) : 
  (a.Prime) → 
  (64 + a^2 = 16*a + 1) → 
  a = 7 := by sorry

end number_of_daughters_l554_55448


namespace hiking_team_gloves_l554_55482

/-- The minimum number of gloves required for a hiking team -/
def min_gloves (participants : ℕ) : ℕ := 2 * participants

/-- Theorem: For 82 participants, the minimum number of gloves required is 164 -/
theorem hiking_team_gloves : min_gloves 82 = 164 := by
  sorry

end hiking_team_gloves_l554_55482


namespace num_distinct_configurations_l554_55476

/-- The group of cube rotations -/
def CubeRotations : Type := Unit

/-- The number of elements in the group of cube rotations -/
def numRotations : ℕ := 4

/-- The number of configurations fixed by the identity rotation -/
def fixedByIdentity : ℕ := 56

/-- The number of configurations fixed by each 180-degree rotation -/
def fixedBy180Rotation : ℕ := 6

/-- The number of 180-degree rotations -/
def num180Rotations : ℕ := 3

/-- The total number of fixed points across all rotations -/
def totalFixedPoints : ℕ := fixedByIdentity + num180Rotations * fixedBy180Rotation

/-- The theorem stating the number of distinct configurations -/
theorem num_distinct_configurations : 
  (totalFixedPoints : ℚ) / numRotations = 19 / 2 := by sorry

end num_distinct_configurations_l554_55476


namespace sqrt_5_minus_2_squared_l554_55438

theorem sqrt_5_minus_2_squared : (Real.sqrt 5 - 2)^2 = 9 - 4 * Real.sqrt 5 := by
  sorry

end sqrt_5_minus_2_squared_l554_55438


namespace two_red_cards_probability_l554_55415

/-- The number of cards in the deck -/
def total_cards : ℕ := 65

/-- The number of red cards in the deck -/
def red_cards : ℕ := 39

/-- The number of ways to choose 2 cards from the deck -/
def total_combinations : ℕ := total_cards.choose 2

/-- The number of ways to choose 2 red cards from the red cards -/
def red_combinations : ℕ := red_cards.choose 2

/-- The probability of drawing two red cards in the first two draws -/
def probability : ℚ := red_combinations / total_combinations

theorem two_red_cards_probability :
  probability = 741 / 2080 := by sorry

end two_red_cards_probability_l554_55415


namespace smallest_number_divisible_by_all_l554_55493

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 823 = 0 ∧
  (n + 1) % 618 = 0 ∧
  (n + 1) % 3648 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 3917 = 0 ∧
  (n + 1) % 4203 = 0

theorem smallest_number_divisible_by_all :
  ∃ n : ℕ, is_divisible_by_all n ∧ ∀ m : ℕ, m < n → ¬is_divisible_by_all m :=
sorry

end smallest_number_divisible_by_all_l554_55493


namespace function_value_theorem_l554_55406

/-- Given a function f(x) = ax³ - bx + |x| - 1 where f(-8) = 3, prove that f(8) = 11 -/
theorem function_value_theorem (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 - b * x + |x| - 1
  (f (-8) = 3) → (f 8 = 11) := by
sorry

end function_value_theorem_l554_55406


namespace number_of_arrangements_l554_55447

/-- The number of people in the line -/
def total_people : ℕ := 6

/-- The number of people in the adjacent group (Xiao Kai and 2 elderly) -/
def adjacent_group : ℕ := 3

/-- The number of volunteers -/
def volunteers : ℕ := 3

/-- The number of ways to arrange the adjacent group internally -/
def adjacent_group_arrangements : ℕ := 2

/-- The number of possible positions for the adjacent group in the line -/
def adjacent_group_positions : ℕ := total_people - adjacent_group - 1

/-- The number of arrangements for the volunteers -/
def volunteer_arrangements : ℕ := 6

theorem number_of_arrangements :
  adjacent_group_arrangements * adjacent_group_positions * volunteer_arrangements = 48 := by
  sorry

end number_of_arrangements_l554_55447


namespace fishing_problem_l554_55420

/-- The number of fish Xiaohua caught -/
def xiaohua_fish : ℕ := 26

/-- The number of fish Xiaobai caught -/
def xiaobai_fish : ℕ := 4

/-- The condition when Xiaohua gives 2 fish to Xiaobai -/
def condition1 (x y : ℕ) : Prop :=
  y - 2 = 4 * (x + 2)

/-- The condition when Xiaohua gives 6 fish to Xiaobai -/
def condition2 (x y : ℕ) : Prop :=
  y - 6 = 2 * (x + 6)

theorem fishing_problem :
  condition1 xiaobai_fish xiaohua_fish ∧
  condition2 xiaobai_fish xiaohua_fish := by
  sorry


end fishing_problem_l554_55420


namespace hypotenuse_length_l554_55400

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-2, 4)
def O : ℝ × ℝ := (0, 0)

-- Define the properties
theorem hypotenuse_length :
  -- A and B are on the graph of y = x^2
  (A.2 = A.1^2) →
  (B.2 = B.1^2) →
  -- Triangle ABO forms a right triangle at O
  ((A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0) →
  -- A and B are symmetric about the y-axis
  (A.1 = -B.1) →
  (A.2 = B.2) →
  -- The x-coordinate of A is 2
  (A.1 = 2) →
  -- The length of hypotenuse AB is 4
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
sorry

end hypotenuse_length_l554_55400


namespace distance_to_origin_l554_55423

/-- The distance from the point (3, -4) to the origin (0, 0) in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + (-4)^2) = 5 := by
  sorry

end distance_to_origin_l554_55423


namespace dressing_ratio_l554_55412

def ranch_cases : ℕ := 28
def caesar_cases : ℕ := 4

theorem dressing_ratio : 
  (ranch_cases / caesar_cases : ℚ) = 7 / 1 := by
  sorry

end dressing_ratio_l554_55412


namespace complex_inequality_l554_55439

theorem complex_inequality (a : ℝ) : 
  (1 - Complex.I) + (1 + Complex.I) * a ≠ 0 → a ≠ -1 ∧ a ≠ 1 := by
  sorry

end complex_inequality_l554_55439


namespace range_of_m_l554_55417

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ¬(Real.sin x + Real.cos x > m)) ∧ 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) ↔ 
  -Real.sqrt 2 ≤ m ∧ m < 2 := by
sorry

end range_of_m_l554_55417


namespace hyperbola_eccentricity_l554_55432

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if there exists a point C(0, √(2b)) such that the perpendicular bisector of AC
    (where A is the left vertex) passes through B (the right vertex),
    then the eccentricity of the hyperbola is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (a, 0)
  let C : ℝ × ℝ := (0, Real.sqrt (2 * b^2))
  f B = 1 ∧ f A = 1 ∧ f C = -1 ∧
  (∃ M : ℝ × ℝ, M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2 ∧
    (B.2 - M.2) * (C.1 - A.1) = (B.1 - M.1) * (C.2 - A.2)) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 10 / 2 := by
sorry


end hyperbola_eccentricity_l554_55432


namespace inequality_proof_l554_55473

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (2/y) ≥ 25 / (1 + 48*x*y^2) := by
  sorry

end inequality_proof_l554_55473


namespace inequality_and_equality_condition_l554_55478

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) ≥ Real.sqrt ((a + c)^2 + (b + d)^2)) ∧
  (Real.sqrt (a^2 + b^2) + Real.sqrt (c^2 + d^2) = Real.sqrt ((a + c)^2 + (b + d)^2) ↔ a * d = b * c) :=
by sorry

end inequality_and_equality_condition_l554_55478


namespace volumes_equal_l554_55495

/-- The volume of a solid of revolution obtained by rotating a region about the y-axis -/
noncomputable def VolumeOfRevolution (region : Set (ℝ × ℝ)) : ℝ := sorry

/-- The region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def Region1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region consisting of points (x, y) that satisfy x²y ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def Region2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 * p.2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The theorem stating that the volumes of revolution of the two regions are equal -/
theorem volumes_equal : VolumeOfRevolution Region1 = VolumeOfRevolution Region2 := by
  sorry

end volumes_equal_l554_55495


namespace cubic_roots_determinant_l554_55425

theorem cubic_roots_determinant (p q r : ℝ) (a b c : ℝ) : 
  (a^3 - p*a^2 + q*a - r = 0) →
  (b^3 - p*b^2 + q*b - r = 0) →
  (c^3 - p*c^2 + q*c - r = 0) →
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, 1; 1, b, 1; 1, 1, c]
  Matrix.det matrix = r - p + 2 := by sorry

end cubic_roots_determinant_l554_55425


namespace least_positive_difference_l554_55472

def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_sequence (b₁ d : ℝ) (n : ℕ) : ℝ := b₁ + (n - 1) * d

def sequence_A (n : ℕ) : ℝ := geometric_sequence 3 2 n

def sequence_B (n : ℕ) : ℝ := arithmetic_sequence 15 15 n

def valid_term_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_term_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_positive_difference :
  ∃ (m n : ℕ), 
    valid_term_A m ∧ 
    valid_term_B n ∧ 
    (∀ (i j : ℕ), valid_term_A i → valid_term_B j → 
      |sequence_A i - sequence_B j| ≥ |sequence_A m - sequence_B n|) ∧
    |sequence_A m - sequence_B n| = 3 :=
sorry

end least_positive_difference_l554_55472


namespace profit_percentage_calculation_l554_55477

/-- Given the cost price and selling price of an article, calculate the profit percentage. -/
theorem profit_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 500 → selling_price = 675 → 
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end profit_percentage_calculation_l554_55477


namespace vector_sum_zero_l554_55462

variable {E : Type*} [NormedAddCommGroup E]

/-- Given vectors CE, AC, DE, and AD in a normed additive commutative group E,
    prove that CE + AC - DE - AD = 0 -/
theorem vector_sum_zero (CE AC DE AD : E) :
  CE + AC - DE - AD = (0 : E) := by sorry

end vector_sum_zero_l554_55462


namespace only_prop2_and_prop3_true_l554_55457

-- Define the propositions
def proposition1 : Prop :=
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) →
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2))

def proposition2 : Prop :=
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0)

def proposition3 (m : ℝ) : Prop :=
  (m = 1/2) →
  ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)

def proposition4 (m n : ℝ) : Prop :=
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0) →
  (m > 0 ∧ n > 0)

-- Theorem stating that only propositions 2 and 3 are true
theorem only_prop2_and_prop3_true :
  ¬proposition1 ∧ proposition2 ∧ (∃ m : ℝ, proposition3 m) ∧ ¬(∀ m n : ℝ, proposition4 m n) :=
sorry

end only_prop2_and_prop3_true_l554_55457


namespace benny_initial_books_l554_55484

/-- The number of books Benny had initially -/
def benny_initial : ℕ := sorry

/-- The number of books Tim has -/
def tim_books : ℕ := 33

/-- The number of books Sandy received from Benny -/
def sandy_received : ℕ := 10

/-- The total number of books they have together now -/
def total_books : ℕ := 47

theorem benny_initial_books : 
  benny_initial = 24 := by sorry

end benny_initial_books_l554_55484


namespace roots_sum_of_squares_reciprocal_l554_55402

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 7*x^2 + 10*x - 6

-- State the theorem
theorem roots_sum_of_squares_reciprocal :
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (1 / a^2 + 1 / b^2 + 1 / c^2 = 46 / 9) := by
  sorry

end roots_sum_of_squares_reciprocal_l554_55402


namespace x_intercept_of_line_l554_55470

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end x_intercept_of_line_l554_55470


namespace inverse_36_mod_101_l554_55490

theorem inverse_36_mod_101 : ∃ x : ℤ, 36 * x ≡ 1 [ZMOD 101] :=
by
  use 87
  sorry

end inverse_36_mod_101_l554_55490


namespace vector_ratio_theorem_l554_55421

theorem vector_ratio_theorem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  (∃ (d : ℝ), magnitude b = magnitude a + d ∧ magnitude sum = magnitude a + 2*d) →
  (a.1 * b.1 + a.2 * b.2 = magnitude a * magnitude b * Real.cos angle) →
  ∃ (k : ℝ), k > 0 ∧ magnitude a = 3*k ∧ magnitude b = 5*k ∧ magnitude sum = 7*k := by
sorry

end vector_ratio_theorem_l554_55421


namespace triangle_longest_side_l554_55488

theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 6) + (3 * y + 5) = 49 →
  max 10 (max (y + 6) (3 * y + 5)) = 26 :=
by sorry

end triangle_longest_side_l554_55488


namespace average_decrease_rate_proof_optimal_price_reduction_proof_l554_55422

-- Define the initial price, final price, and years of decrease
def initial_price : ℝ := 200
def final_price : ℝ := 162
def years_of_decrease : ℕ := 2

-- Define the daily sales and profit parameters
def initial_daily_sales : ℕ := 20
def price_reduction_step : ℝ := 5
def sales_increase_per_step : ℕ := 10
def daily_profit : ℝ := 1150

-- Define the average yearly decrease rate
def average_decrease_rate : ℝ := 0.1

-- Define the optimal price reduction
def optimal_price_reduction : ℝ := 15

-- Theorem for the average yearly decrease rate
theorem average_decrease_rate_proof :
  initial_price * (1 - average_decrease_rate) ^ years_of_decrease = final_price :=
sorry

-- Theorem for the optimal price reduction
theorem optimal_price_reduction_proof :
  let new_price := initial_price - optimal_price_reduction
  let new_sales := initial_daily_sales + (optimal_price_reduction / price_reduction_step) * sales_increase_per_step
  (new_price - final_price) * new_sales = daily_profit :=
sorry

end average_decrease_rate_proof_optimal_price_reduction_proof_l554_55422


namespace min_value_quadratic_l554_55496

theorem min_value_quadratic :
  ∃ (z_min : ℝ), z_min = -44 ∧ ∀ (x : ℝ), x^2 + 16*x + 20 ≥ z_min :=
by
  sorry

end min_value_quadratic_l554_55496


namespace quadratic_root_problem_l554_55441

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_root_problem (a b c : ℤ) (m n : ℕ) :
  (∀ x : ℤ, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m ≠ n →
  m > 0 →
  n > 0 →
  is_prime (a + b + c) →
  (∃ x : ℤ, a * x^2 + b * x + c = -55) →
  m = 2 →
  n = 17 :=
sorry

end quadratic_root_problem_l554_55441


namespace absent_children_l554_55430

/-- Proves that the number of absent children is 70 given the conditions of the problem -/
theorem absent_children (total_children : ℕ) (sweets_per_child : ℕ) (extra_sweets : ℕ) : 
  total_children = 190 →
  sweets_per_child = 38 →
  extra_sweets = 14 →
  (total_children - (total_children - sweets_per_child * total_children / (sweets_per_child - extra_sweets))) = 70 := by
  sorry

end absent_children_l554_55430


namespace sqrt_three_diamond_sqrt_three_l554_55443

-- Define the operation ¤
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_three_diamond_sqrt_three : diamond (Real.sqrt 3) (Real.sqrt 3) = 12 := by
  sorry

end sqrt_three_diamond_sqrt_three_l554_55443


namespace rhinestones_needed_proof_l554_55429

/-- Given a total number of rhinestones needed, calculate the number still needed
    after buying one-third and finding one-fifth of the total. -/
def rhinestones_still_needed (total : ℕ) : ℕ :=
  total - (total / 3) - (total / 5)

/-- Theorem stating that for 45 rhinestones, the number still needed is 21. -/
theorem rhinestones_needed_proof :
  rhinestones_still_needed 45 = 21 := by
  sorry

#eval rhinestones_still_needed 45

end rhinestones_needed_proof_l554_55429


namespace min_value_theorem_l554_55440

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_theorem_l554_55440


namespace x_root_of_quadratic_with_integer_coeff_l554_55435

/-- Given distinct real numbers x and y with equal fractional parts and equal fractional parts of their cubes,
    x is a root of a quadratic equation with integer coefficients. -/
theorem x_root_of_quadratic_with_integer_coeff
  (x y : ℝ)
  (h_distinct : x ≠ y)
  (h_frac_eq : x - ⌊x⌋ = y - ⌊y⌋)
  (h_frac_cube_eq : x^3 - ⌊x^3⌋ = y^3 - ⌊y^3⌋) :
  ∃ (a b c : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ (a * x^2 + b * x + c : ℝ) = 0 :=
sorry

end x_root_of_quadratic_with_integer_coeff_l554_55435


namespace trig_identity_l554_55475

theorem trig_identity : 
  (Real.cos (20 * π / 180)) / (Real.cos (35 * π / 180) * Real.sqrt (1 - Real.sin (20 * π / 180))) = Real.sqrt 2 := by
  sorry

end trig_identity_l554_55475


namespace simplify_expression_l554_55453

theorem simplify_expression (x : ℝ) : 3 * (4 * x^2)^4 = 768 * x^8 := by
  sorry

end simplify_expression_l554_55453


namespace integral_2x_minus_1_l554_55403

theorem integral_2x_minus_1 : ∫ x in (0 : ℝ)..3, (2*x - 1) = 6 := by sorry

end integral_2x_minus_1_l554_55403


namespace prob_non_defective_pencils_l554_55405

/-- The probability of selecting 5 non-defective pencils from a box of 12 pencils
    where 4 are defective is 7/99. -/
theorem prob_non_defective_pencils :
  let total_pencils : ℕ := 12
  let defective_pencils : ℕ := 4
  let selected_pencils : ℕ := 5
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  (Nat.choose non_defective_pencils selected_pencils : ℚ) /
  (Nat.choose total_pencils selected_pencils : ℚ) = 7 / 99 := by
sorry

end prob_non_defective_pencils_l554_55405


namespace smallest_positive_value_cubic_expression_l554_55481

theorem smallest_positive_value_cubic_expression (a b c : ℕ+) :
  a^3 + b^3 + c^3 - 3*a*b*c ≥ 4 ∧ ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 4 :=
sorry

end smallest_positive_value_cubic_expression_l554_55481


namespace employment_after_growth_and_new_category_l554_55461

/-- Represents the employment data for town X -/
structure TownEmployment where
  initial_rate : ℝ
  annual_growth : ℝ
  years : ℕ
  male_percentage : ℝ
  tourism_percentage : ℝ
  female_edu_percentage : ℝ

/-- Theorem about employment percentages after growth and new category introduction -/
theorem employment_after_growth_and_new_category 
  (town : TownEmployment)
  (h_initial : town.initial_rate = 0.64)
  (h_growth : town.annual_growth = 0.02)
  (h_years : town.years = 5)
  (h_male : town.male_percentage = 0.55)
  (h_tourism : town.tourism_percentage = 0.1)
  (h_female_edu : town.female_edu_percentage = 0.6) :
  let final_rate := town.initial_rate + town.annual_growth * town.years
  let female_percentage := 1 - town.male_percentage
  (female_percentage = 0.45) ∧ 
  (town.female_edu_percentage > 0.5) := by
  sorry

#check employment_after_growth_and_new_category

end employment_after_growth_and_new_category_l554_55461


namespace min_teachers_is_six_l554_55413

/-- Represents the number of subjects for each discipline -/
structure SubjectCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Represents the constraints of the teaching system -/
structure TeachingSystem where
  subjects : SubjectCounts
  max_subjects_per_teacher : Nat
  specialized : Bool

/-- Calculates the minimum number of teachers required -/
def min_teachers_required (system : TeachingSystem) : Nat :=
  if system.specialized then
    let maths_teachers := (system.subjects.maths + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let physics_teachers := (system.subjects.physics + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    let chemistry_teachers := (system.subjects.chemistry + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher
    maths_teachers + physics_teachers + chemistry_teachers
  else
    let total_subjects := system.subjects.maths + system.subjects.physics + system.subjects.chemistry
    (total_subjects + system.max_subjects_per_teacher - 1) / system.max_subjects_per_teacher

/-- The main theorem stating that the minimum number of teachers required is 6 -/
theorem min_teachers_is_six (system : TeachingSystem) 
  (h1 : system.subjects = { maths := 6, physics := 5, chemistry := 5 })
  (h2 : system.max_subjects_per_teacher = 4)
  (h3 : system.specialized = true) : 
  min_teachers_required system = 6 := by
  sorry

end min_teachers_is_six_l554_55413


namespace favorite_number_is_25_l554_55433

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_diff (n : ℕ) : ℕ := Int.natAbs ((n / 10) - (n % 10))

def has_unique_digit (n : ℕ) : Prop :=
  ∀ m : ℕ, is_two_digit m → is_perfect_square m → m ≠ n →
    (n / 10 ≠ m / 10 ∧ n / 10 ≠ m % 10) ∨ (n % 10 ≠ m / 10 ∧ n % 10 ≠ m % 10)

def non_unique_sum (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_sum m = digit_sum n

def non_unique_diff (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_two_digit m ∧ digit_diff m = digit_diff n

theorem favorite_number_is_25 :
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ has_unique_digit n ∧
    non_unique_sum n ∧ non_unique_diff n ∧ n = 25 :=
by sorry

end favorite_number_is_25_l554_55433


namespace pizza_toppings_l554_55486

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (mushroom_slices : ℕ) :
  total_slices = 10 →
  cheese_slices = 5 →
  mushroom_slices = 7 →
  cheese_slices + mushroom_slices - total_slices = 2 :=
by
  sorry

end pizza_toppings_l554_55486


namespace exchange_equality_l554_55452

theorem exchange_equality (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : a₁^2 + b₁^2 = 1)
  (h2 : a₂^2 + b₂^2 = 1)
  (h3 : a₁*a₂ + b₁*b₂ = 0) :
  (a₁^2 + a₂^2 = 1) ∧ (b₁^2 + b₂^2 = 1) ∧ (a₁*b₁ + a₂*b₂ = 0) := by
sorry

end exchange_equality_l554_55452


namespace minimal_edge_count_l554_55427

/-- A graph with 7 vertices satisfying the given conditions -/
structure MinimalGraph where
  -- The set of vertices
  V : Finset ℕ
  -- The set of edges
  E : Finset (Finset ℕ)
  -- There are exactly 7 vertices
  vertex_count : V.card = 7
  -- Each edge connects exactly two vertices
  edge_valid : ∀ e ∈ E, e.card = 2 ∧ e ⊆ V
  -- Among any three vertices, at least two are connected
  connected_condition : ∀ {a b c}, a ∈ V → b ∈ V → c ∈ V → a ≠ b → b ≠ c → a ≠ c →
    {a, b} ∈ E ∨ {b, c} ∈ E ∨ {a, c} ∈ E

/-- The theorem stating that the minimal number of edges is 9 -/
theorem minimal_edge_count (G : MinimalGraph) : G.E.card = 9 := by
  sorry

end minimal_edge_count_l554_55427


namespace alien_martian_limb_difference_l554_55410

/-- The number of arms an alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs an alien has -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs a Martian has -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of aliens and Martians being compared -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end alien_martian_limb_difference_l554_55410


namespace arithmetic_sequence_sum_l554_55446

/-- Given that a, 6, and b form an arithmetic sequence in that order, prove that a + b = 12 -/
theorem arithmetic_sequence_sum (a b : ℝ) 
  (h : ∃ d : ℝ, a + d = 6 ∧ b = a + 2*d) : 
  a + b = 12 := by
sorry

end arithmetic_sequence_sum_l554_55446


namespace price_decrease_sales_increase_l554_55404

/-- Given a price decrease and revenue increase, calculate the increase in number of items sold -/
theorem price_decrease_sales_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (revenue_increase_percentage : ℝ)
  (h_price_decrease : price_decrease_percentage = 20)
  (h_revenue_increase : revenue_increase_percentage = 28.000000000000025)
  (h_positive_price : original_price > 0)
  (h_positive_quantity : original_quantity > 0) :
  let new_price := original_price * (1 - price_decrease_percentage / 100)
  let new_quantity := original_quantity * (1 + revenue_increase_percentage / 100) / (1 - price_decrease_percentage / 100)
  let quantity_increase_percentage := (new_quantity / original_quantity - 1) * 100
  ∃ ε > 0, |quantity_increase_percentage - 60| < ε :=
sorry

end price_decrease_sales_increase_l554_55404


namespace guesthouse_fixed_rate_l554_55445

/-- A guesthouse charging system with a fixed rate for the first night and an additional fee for subsequent nights. -/
structure Guesthouse where
  first_night : ℕ  -- Fixed rate for the first night
  subsequent : ℕ  -- Fee for each subsequent night

/-- The total cost for a stay at the guesthouse. -/
def total_cost (g : Guesthouse) (nights : ℕ) : ℕ :=
  g.first_night + g.subsequent * (nights - 1)

theorem guesthouse_fixed_rate :
  ∃ (g : Guesthouse),
    total_cost g 5 = 220 ∧
    total_cost g 8 = 370 ∧
    g.first_night = 20 := by
  sorry

end guesthouse_fixed_rate_l554_55445


namespace banana_orange_equivalence_l554_55474

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 4 : ℚ) * 16 * banana_value = 6 * orange_value →
  (1 / 3 : ℚ) * 9 * banana_value = (3 / 2 : ℚ) * orange_value :=
by
  sorry

end banana_orange_equivalence_l554_55474


namespace y_equal_at_one_y_diff_five_at_two_l554_55431

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -2 * x + 3
def y₂ (x : ℝ) : ℝ := 3 * x - 2

-- Theorem 1: y₁ = y₂ when x = 1
theorem y_equal_at_one : y₁ 1 = y₂ 1 := by sorry

-- Theorem 2: y₁ + 5 = y₂ when x = 2
theorem y_diff_five_at_two : y₁ 2 + 5 = y₂ 2 := by sorry

end y_equal_at_one_y_diff_five_at_two_l554_55431


namespace two_roses_more_expensive_than_three_carnations_l554_55414

/-- The price of a single rose in yuan -/
def rose_price : ℝ := sorry

/-- The price of a single carnation in yuan -/
def carnation_price : ℝ := sorry

/-- The combined price of 6 roses and 3 carnations -/
def combined_price_1 : ℝ := 6 * rose_price + 3 * carnation_price

/-- The combined price of 4 roses and 5 carnations -/
def combined_price_2 : ℝ := 4 * rose_price + 5 * carnation_price

/-- Theorem stating that the price of 2 roses is higher than the price of 3 carnations -/
theorem two_roses_more_expensive_than_three_carnations 
  (h1 : combined_price_1 > 24)
  (h2 : combined_price_2 < 22) :
  2 * rose_price > 3 * carnation_price :=
by sorry

end two_roses_more_expensive_than_three_carnations_l554_55414


namespace solve_manuscript_typing_l554_55409

def manuscript_typing_problem (total_pages : ℕ) (twice_revised : ℕ) (first_typing_cost : ℕ) (revision_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (once_revised : ℕ),
    once_revised + twice_revised ≤ total_pages ∧
    first_typing_cost * total_pages + revision_cost * once_revised + 2 * revision_cost * twice_revised = total_cost ∧
    once_revised = 30

theorem solve_manuscript_typing :
  manuscript_typing_problem 100 20 5 4 780 :=
sorry

end solve_manuscript_typing_l554_55409


namespace sin_cos_equality_relation_l554_55469

open Real

theorem sin_cos_equality_relation :
  (∃ (α β : ℝ), (sin α = sin β ∧ cos α = cos β) ∧ α ≠ β) ∧
  (∀ (α β : ℝ), α = β → (sin α = sin β ∧ cos α = cos β)) :=
by sorry

end sin_cos_equality_relation_l554_55469


namespace remaining_students_count_l554_55480

/-- The number of groups with 15 students -/
def groups_15 : ℕ := 4

/-- The number of groups with 18 students -/
def groups_18 : ℕ := 2

/-- The number of students in each of the first 4 groups -/
def students_per_group_15 : ℕ := 15

/-- The number of students in each of the last 2 groups -/
def students_per_group_18 : ℕ := 18

/-- The number of students who left early from the first 4 groups -/
def left_early_15 : ℕ := 8

/-- The number of students who left early from the last 2 groups -/
def left_early_18 : ℕ := 5

/-- The total number of remaining students -/
def remaining_students : ℕ := 
  (groups_15 * students_per_group_15 - left_early_15) + 
  (groups_18 * students_per_group_18 - left_early_18)

theorem remaining_students_count : remaining_students = 83 := by
  sorry

end remaining_students_count_l554_55480


namespace max_profit_increase_2008_l554_55419

def profit_growth : Fin 10 → ℝ
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 40
  | ⟨2, _⟩ => 60
  | ⟨3, _⟩ => 65
  | ⟨4, _⟩ => 80
  | ⟨5, _⟩ => 85
  | ⟨6, _⟩ => 90
  | ⟨7, _⟩ => 95
  | ⟨8, _⟩ => 100
  | ⟨9, _⟩ => 80

def year_from_index (i : Fin 10) : ℕ := 2000 + 2 * i.val

def profit_increase (i : Fin 9) : ℝ := profit_growth (i.succ) - profit_growth i

theorem max_profit_increase_2008 :
  ∃ (i : Fin 9), year_from_index i.succ = 2008 ∧
  ∀ (j : Fin 9), profit_increase i ≥ profit_increase j :=
by sorry

end max_profit_increase_2008_l554_55419


namespace binomial_expansion_terms_l554_55411

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 * x^(n-1) * a = 56) →
  (Nat.choose n 2 * x^(n-2) * a^2 = 168) →
  (Nat.choose n 3 * x^(n-3) * a^3 = 336) →
  n = 5 := by sorry

end binomial_expansion_terms_l554_55411


namespace sum_of_square_areas_l554_55468

/-- Given a square with side length 4 cm, and an infinite series of squares where each subsequent
    square is formed by joining the midpoints of the sides of the previous square,
    the sum of the areas of all squares is 32 cm². -/
theorem sum_of_square_areas (first_square_side : ℝ) (h : first_square_side = 4) :
  let area_sequence : ℕ → ℝ := λ n => first_square_side^2 / 2^n
  ∑' n, area_sequence n = 32 :=
sorry

end sum_of_square_areas_l554_55468


namespace elena_allowance_spending_l554_55459

theorem elena_allowance_spending (A : ℝ) : ∃ (m s : ℝ),
  m = (1/4) * (A - s) ∧
  s = (1/10) * (A - m) ∧
  m + s = (4/13) * A :=
by sorry

end elena_allowance_spending_l554_55459


namespace dog_park_ratio_l554_55455

theorem dog_park_ratio (total : ℕ) (running : ℕ) (doing_nothing : ℕ) 
  (h1 : total = 88)
  (h2 : running = 12)
  (h3 : doing_nothing = 10)
  (h4 : total / 4 = total / 4) : -- This represents that 1/4 of dogs are barking
  (total - running - (total / 4) - doing_nothing) / total = 1 / 2 := by
  sorry

end dog_park_ratio_l554_55455


namespace pyramid_solution_l554_55408

structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_right : ℕ
  row3_left : ℕ
  row3_middle : ℕ
  row3_right : ℕ

def is_valid_pyramid (p : NumberPyramid) : Prop :=
  p.row2_left = p.row1_left + p.row1_right ∧
  p.row2_right = p.row1_right + 660 ∧
  p.row3_left = p.row2_left * p.row1_left ∧
  p.row3_middle = p.row2_left * p.row2_right ∧
  p.row3_right = p.row2_right * 660

theorem pyramid_solution :
  ∃ (p : NumberPyramid), is_valid_pyramid p ∧ 
    p.row3_left = 28 ∧ p.row3_right = 630 ∧ p.row2_left = 13 := by
  sorry

end pyramid_solution_l554_55408


namespace equilateral_triangle_not_unique_from_angles_l554_55463

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- The theorem stating that two angles do not uniquely determine an equilateral triangle -/
theorem equilateral_triangle_not_unique_from_angles :
  ∃ (t1 t2 : EquilateralTriangle), t1 ≠ t2 ∧ 
  (∀ (θ : ℝ), 0 < θ ∧ θ < π → 
    (θ = π/3 ↔ (∃ (i : Fin 3), θ = π/3))) :=
sorry

end equilateral_triangle_not_unique_from_angles_l554_55463


namespace anas_dresses_l554_55450

theorem anas_dresses (ana lisa : ℕ) : 
  lisa = ana + 18 → 
  ana + lisa = 48 → 
  ana = 15 := by
sorry

end anas_dresses_l554_55450


namespace victors_stickers_l554_55464

theorem victors_stickers (flower_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 →
  animal_stickers = flower_stickers - 2 →
  flower_stickers + animal_stickers = 14 :=
by sorry

end victors_stickers_l554_55464


namespace hugo_climb_count_l554_55458

def hugo_mountain_elevation : ℕ := 10000
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2500
def boris_climb_count : ℕ := 4

theorem hugo_climb_count : 
  ∃ (x : ℕ), x * hugo_mountain_elevation = boris_climb_count * boris_mountain_elevation ∧ x = 3 :=
by sorry

end hugo_climb_count_l554_55458


namespace not_nth_power_of_sum_of_powers_l554_55498

theorem not_nth_power_of_sum_of_powers (p n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
  ¬ ∃ m : ℕ, (2^p : ℕ) + (3^p : ℕ) = m^n :=
sorry

end not_nth_power_of_sum_of_powers_l554_55498


namespace triangle_side_not_eight_l554_55407

/-- A triangle with side lengths a, b, and c exists if and only if the sum of any two sides is greater than the third side for all combinations. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In a triangle with side lengths 3, 5, and x, x cannot be 8. -/
theorem triangle_side_not_eight :
  ¬ (triangle_inequality 3 5 8) :=
sorry

end triangle_side_not_eight_l554_55407


namespace tangent_line_equation_l554_55416

/-- A line that is tangent to a circle and intersects a parabola -/
structure TangentLine where
  -- The line equation: ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The circle equation: x^4 + y^2 = 8
  circle : (x y : ℝ) → x^4 + y^2 = 8
  -- The parabola equation: y^2 = 4x
  parabola : (x y : ℝ) → y^2 = 4*x
  -- The line is tangent to the circle
  is_tangent : ∃ (x y : ℝ), a*x + b*y + c = 0 ∧ x^4 + y^2 = 8
  -- The line intersects the parabola at two points
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    a*x₁ + b*y₁ + c = 0 ∧ y₁^2 = 4*x₁ ∧
    a*x₂ + b*y₂ + c = 0 ∧ y₂^2 = 4*x₂
  -- The circle passes through the origin
  origin_on_circle : 0^4 + 0^2 = 8

/-- The theorem stating the equation of the tangent line -/
theorem tangent_line_equation (l : TangentLine) : 
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -4) := by
  sorry

end tangent_line_equation_l554_55416


namespace lev_number_pairs_l554_55479

theorem lev_number_pairs : 
  ∀ a b : ℕ, a + b + a * b = 1000 → 
  ((a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
   (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
   (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12)) :=
by sorry

end lev_number_pairs_l554_55479


namespace equation_transformation_l554_55451

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - x^3 - 2*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 4) = 0 :=
by sorry

end equation_transformation_l554_55451


namespace min_sum_given_reciprocal_sum_l554_55424

theorem min_sum_given_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + y ≥ 6 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a + b = 6 :=
by sorry

end min_sum_given_reciprocal_sum_l554_55424


namespace unique_three_digit_number_l554_55442

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  ones_range : ones ≥ 0 ∧ ones ≤ 9
  hundreds_odd : Odd hundreds

/-- Converts a ThreeDigitNumber to its decimal representation -/
def toDecimal (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Sums all permutations of a ThreeDigitNumber -/
def sumPermutations (n : ThreeDigitNumber) : Nat :=
  toDecimal n +
  (100 * n.hundreds + n.tens + 10 * n.ones) +
  (100 * n.tens + 10 * n.ones + n.hundreds) +
  (100 * n.tens + n.hundreds + 10 * n.ones) +
  (100 * n.ones + 10 * n.hundreds + n.tens) +
  (100 * n.ones + 10 * n.tens + n.hundreds)

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber, sumPermutations n = 3300 → toDecimal n = 192 := by
  sorry

end unique_three_digit_number_l554_55442


namespace tommy_initial_balloons_l554_55437

/-- The number of balloons Tommy had initially -/
def initial_balloons : ℕ := 26

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The total number of balloons Tommy had after receiving more from his mom -/
def total_balloons : ℕ := 60

/-- Theorem: Tommy had 26 balloons to start with -/
theorem tommy_initial_balloons : 
  initial_balloons + mom_balloons = total_balloons :=
by sorry

end tommy_initial_balloons_l554_55437


namespace consecutive_product_divisibility_l554_55483

theorem consecutive_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2) * (k + 3)
  (∃ m : ℤ, n = 11 * m) → 
  (∃ m : ℤ, n = 44 * m) ∧ 
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) * (k + 3) = 66 * m) :=
by sorry

end consecutive_product_divisibility_l554_55483


namespace chicken_pasta_pieces_is_two_l554_55499

/-- Represents the number of chicken pieces in different orders and the total needed -/
structure ChickenOrders where
  barbecue_pieces : ℕ
  fried_dinner_pieces : ℕ
  fried_dinner_orders : ℕ
  chicken_pasta_orders : ℕ
  barbecue_orders : ℕ
  total_pieces : ℕ

/-- Calculates the number of chicken pieces in a Chicken Pasta order -/
def chicken_pasta_pieces (orders : ChickenOrders) : ℕ :=
  (orders.total_pieces -
   (orders.fried_dinner_pieces * orders.fried_dinner_orders +
    orders.barbecue_pieces * orders.barbecue_orders)) /
  orders.chicken_pasta_orders

/-- Theorem stating that the number of chicken pieces in a Chicken Pasta order is 2 -/
theorem chicken_pasta_pieces_is_two (orders : ChickenOrders)
  (h1 : orders.barbecue_pieces = 3)
  (h2 : orders.fried_dinner_pieces = 8)
  (h3 : orders.fried_dinner_orders = 2)
  (h4 : orders.chicken_pasta_orders = 6)
  (h5 : orders.barbecue_orders = 3)
  (h6 : orders.total_pieces = 37) :
  chicken_pasta_pieces orders = 2 := by
  sorry

end chicken_pasta_pieces_is_two_l554_55499


namespace losing_candidate_vote_percentage_l554_55465

/-- Given the total number of votes and the margin of loss, 
    calculate the percentage of votes received by the losing candidate. -/
theorem losing_candidate_vote_percentage
  (total_votes : ℕ)
  (loss_margin : ℕ)
  (h1 : total_votes = 7800)
  (h2 : loss_margin = 2340) :
  (total_votes - loss_margin) * 100 / total_votes = 70 := by
  sorry

end losing_candidate_vote_percentage_l554_55465


namespace business_loss_l554_55467

/-- Proves that the total loss in a business partnership is 1600 given the specified conditions -/
theorem business_loss (ashok_capital pyarelal_capital pyarelal_loss : ℚ) : 
  ashok_capital = (1 : ℚ) / 9 * pyarelal_capital →
  pyarelal_loss = 1440 →
  ashok_capital / pyarelal_capital * pyarelal_loss + pyarelal_loss = 1600 :=
by sorry

end business_loss_l554_55467


namespace equation_solution_l554_55491

theorem equation_solution : 
  ∀ x y : ℕ, x^2 + x*y = y + 92 ↔ (x = 2 ∧ y = 88) ∨ (x = 8 ∧ y = 4) := by
  sorry

end equation_solution_l554_55491


namespace type_b_soda_cans_l554_55436

/-- The number of cans of type B soda that can be purchased for a given amount of money -/
theorem type_b_soda_cans 
  (T : ℕ) -- number of type A cans
  (P : ℕ) -- price in quarters for T cans of type A
  (R : ℚ) -- amount of dollars available
  (h1 : P > 0) -- ensure division by P is valid
  (h2 : T > 0) -- ensure division by T is valid
  : (2 * R * T.cast) / P.cast = (4 * R * T.cast) / (2 * P.cast) := by
  sorry

end type_b_soda_cans_l554_55436


namespace parabola_directrix_l554_55428

/-- Given a parabola with equation y² = 16x, its directrix has equation x = -4 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 16*x) → (∃ p : ℝ, p = 4 ∧ x = -p) :=
by sorry

end parabola_directrix_l554_55428


namespace smallest_number_divisible_by_all_l554_55466

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 12) % 8 = 0 ∧ (n - 12) % 12 = 0 ∧ (n - 12) % 22 = 0 ∧ (n - 12) % 24 = 0

theorem smallest_number_divisible_by_all : 
  (is_divisible_by_all 252) ∧ 
  (∀ m : ℕ, m < 252 → ¬(is_divisible_by_all m)) :=
by sorry

end smallest_number_divisible_by_all_l554_55466


namespace river_joe_pricing_l554_55434

/-- River Joe's Seafood Diner pricing problem -/
theorem river_joe_pricing
  (total_orders : ℕ)
  (total_revenue : ℚ)
  (catfish_price : ℚ)
  (popcorn_shrimp_orders : ℕ)
  (h1 : total_orders = 26)
  (h2 : total_revenue = 133.5)
  (h3 : catfish_price = 6)
  (h4 : popcorn_shrimp_orders = 9) :
  ∃ (popcorn_shrimp_price : ℚ),
    popcorn_shrimp_price = 3.5 ∧
    total_revenue = (total_orders - popcorn_shrimp_orders) * catfish_price +
                    popcorn_shrimp_orders * popcorn_shrimp_price :=
by sorry

end river_joe_pricing_l554_55434


namespace a_value_is_negative_one_l554_55454

/-- The coefficient of x^2 in the expansion of (1+ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ :=
  (Nat.choose 5 2 : ℝ) + a * (Nat.choose 5 1 : ℝ)

/-- The theorem stating that a = -1 given the coefficient of x^2 is 5 -/
theorem a_value_is_negative_one :
  ∃ a : ℝ, coefficient_x_squared a = 5 ∧ a = -1 :=
sorry

end a_value_is_negative_one_l554_55454


namespace sheetrock_width_l554_55449

/-- Given a rectangular piece of sheetrock with length 6 feet and area 30 square feet, its width is 5 feet. -/
theorem sheetrock_width (length : ℝ) (area : ℝ) (width : ℝ) : 
  length = 6 → area = 30 → area = length * width → width = 5 := by
  sorry

end sheetrock_width_l554_55449
