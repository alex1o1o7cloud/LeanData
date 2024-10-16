import Mathlib

namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l1864_186462

theorem hexagon_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = 2 * a) :
  (3 * Real.sqrt 3 / 2 * a^2) / (3 * Real.sqrt 3 / 2 * b^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l1864_186462


namespace NUMINAMATH_CALUDE_at_least_two_positive_l1864_186408

theorem at_least_two_positive (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c > 0) (h5 : a * b + b * c + c * a > 0) :
  (a > 0 ∧ b > 0) ∨ (b > 0 ∧ c > 0) ∨ (c > 0 ∧ a > 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_positive_l1864_186408


namespace NUMINAMATH_CALUDE_quadratic_properties_l1864_186429

theorem quadratic_properties (a b c : ℝ) (ha : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (a * c < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (a * b > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1864_186429


namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l1864_186414

/-- Parabola C₁ with equation x² = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : x^2 = 2 * C.p * y

/-- External point M -/
structure ExternalPoint (C : Parabola) where
  a : ℝ
  y : ℝ
  hy : y = -2 * C.p

/-- Theorem stating the main results -/
theorem parabola_tangent_theorem (C : Parabola) (M : ExternalPoint C) :
  -- Part 1: If a line through focus with x-intercept 2 intersects C₁ at Q and N 
  -- such that |Q'N'| = 2√5, then p = 2
  (∃ (Q N : ParabolaPoint C), 
    (Q.x / 2 + 2 * Q.y / C.p = 1) ∧ 
    (N.x / 2 + 2 * N.y / C.p = 1) ∧ 
    ((Q.x - N.x)^2 = 20)) →
  C.p = 2 ∧
  -- Part 2: If A and B are tangent points, then k₁ · k₂ = -4
  (∀ (A B : ParabolaPoint C),
    (A.y - M.y = (A.x / C.p) * (A.x - M.a)) →
    (B.y - M.y = (B.x / C.p) * (B.x - M.a)) →
    ((A.x / C.p) * (B.x / C.p) = -4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l1864_186414


namespace NUMINAMATH_CALUDE_recover_sequence_l1864_186415

/-- A sequence of six positive integers in arithmetic progression. -/
def ArithmeticSequence : Type := Fin 6 → ℕ+

/-- The given sequence with one number omitted and one miscopied. -/
def GivenSequence : Fin 5 → ℕ+ := ![113, 137, 149, 155, 173]

/-- The correct sequence. -/
def CorrectSequence : ArithmeticSequence := ![113, 125, 137, 149, 161, 173]

/-- Checks if a sequence is in arithmetic progression. -/
def isArithmeticProgression (s : ArithmeticSequence) : Prop :=
  ∃ d : ℕ+, ∀ i : Fin 5, s (i + 1) = s i + d

/-- Checks if a sequence matches the given sequence except for one miscopied number. -/
def matchesGivenSequence (s : ArithmeticSequence) : Prop :=
  ∃ j : Fin 5, ∀ i : Fin 5, i ≠ j → s i = GivenSequence i

theorem recover_sequence :
  isArithmeticProgression CorrectSequence ∧
  matchesGivenSequence CorrectSequence :=
sorry

end NUMINAMATH_CALUDE_recover_sequence_l1864_186415


namespace NUMINAMATH_CALUDE_fisherman_catch_l1864_186400

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (salmon : ℕ) (pike : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  blue_gill = 2 * bass →
  salmon = bass + bass / 3 →
  pike = (bass + trout + blue_gill + salmon) / 5 →
  bass + trout + blue_gill + salmon + pike = 138 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l1864_186400


namespace NUMINAMATH_CALUDE_expected_stones_approx_l1864_186439

/-- The width of the river (scaled to 1) -/
def river_width : ℝ := 1

/-- The maximum jump distance (scaled to 0.01) -/
def jump_distance : ℝ := 0.01

/-- The probability that we cannot cross the river after n throws -/
noncomputable def P (n : ℕ) : ℝ :=
  ∑' i, (-1)^(i-1) * (n+1).choose i * (max (1 - i * jump_distance) 0)^n

/-- The expected number of stones needed to cross the river -/
noncomputable def expected_stones : ℝ :=
  ∑' n, P n

/-- Theorem stating the approximation of the expected number of stones -/
theorem expected_stones_approx :
  ∃ ε > 0, |expected_stones - 712.811| < ε :=
sorry

end NUMINAMATH_CALUDE_expected_stones_approx_l1864_186439


namespace NUMINAMATH_CALUDE_gcd_175_100_65_l1864_186437

theorem gcd_175_100_65 : Nat.gcd 175 (Nat.gcd 100 65) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_175_100_65_l1864_186437


namespace NUMINAMATH_CALUDE_tony_lego_purchase_l1864_186450

/-- Represents the purchase of toys by Tony -/
structure ToyPurchase where
  lego_price : ℕ
  sword_price : ℕ
  dough_price : ℕ
  sword_count : ℕ
  dough_count : ℕ
  total_paid : ℕ

/-- Calculates the number of Lego sets bought -/
def lego_sets_bought (purchase : ToyPurchase) : ℕ :=
  (purchase.total_paid - purchase.sword_price * purchase.sword_count - purchase.dough_price * purchase.dough_count) / purchase.lego_price

/-- Theorem stating that Tony bought 3 sets of Lego blocks -/
theorem tony_lego_purchase : 
  ∀ (purchase : ToyPurchase), 
  purchase.lego_price = 250 ∧ 
  purchase.sword_price = 120 ∧ 
  purchase.dough_price = 35 ∧ 
  purchase.sword_count = 7 ∧ 
  purchase.dough_count = 10 ∧ 
  purchase.total_paid = 1940 → 
  lego_sets_bought purchase = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_lego_purchase_l1864_186450


namespace NUMINAMATH_CALUDE_value_of_M_l1864_186420

theorem value_of_M : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) - Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 - 1) + Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (3 - Real.sqrt 6 + Real.sqrt 42) / 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l1864_186420


namespace NUMINAMATH_CALUDE_power_multiplication_l1864_186448

theorem power_multiplication (a : ℝ) : a^5 * a^2 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1864_186448


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1864_186493

theorem adult_tickets_sold (adult_price child_price total_tickets total_amount : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 2)
  (h3 : total_tickets = 85)
  (h4 : total_amount = 275) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_amount ∧ 
    adult_tickets = 35 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l1864_186493


namespace NUMINAMATH_CALUDE_candy_chocolate_price_difference_l1864_186481

def candy_bar_original_price : ℝ := 6
def candy_bar_discount : ℝ := 0.25
def chocolate_original_price : ℝ := 3
def chocolate_discount : ℝ := 0.10

def discounted_price (original_price discount : ℝ) : ℝ :=
  original_price * (1 - discount)

theorem candy_chocolate_price_difference :
  discounted_price candy_bar_original_price candy_bar_discount -
  discounted_price chocolate_original_price chocolate_discount = 1.80 := by
sorry

end NUMINAMATH_CALUDE_candy_chocolate_price_difference_l1864_186481


namespace NUMINAMATH_CALUDE_pencil_distribution_result_l1864_186411

/-- Represents the pencil distribution problem --/
structure PencilDistribution where
  gloria_initial : ℕ
  lisa_initial : ℕ
  tim_initial : ℕ

/-- Calculates the final pencil counts after Lisa's distribution --/
def final_counts (pd : PencilDistribution) : ℕ × ℕ × ℕ :=
  let lisa_half := pd.lisa_initial / 2
  (pd.gloria_initial + lisa_half, 0, pd.tim_initial + lisa_half)

/-- Theorem stating the final pencil counts after distribution --/
theorem pencil_distribution_result (pd : PencilDistribution)
  (h1 : pd.gloria_initial = 2500)
  (h2 : pd.lisa_initial = 75800)
  (h3 : pd.tim_initial = 1950) :
  final_counts pd = (40400, 0, 39850) := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_result_l1864_186411


namespace NUMINAMATH_CALUDE_divisor_of_2p_plus_1_l1864_186480

theorem divisor_of_2p_plus_1 (p k : ℕ) (h_prime : Prime p) (h_k_gt_3 : k > 3) 
  (h_divides : k ∣ (2^p + 1)) : k ≥ 2*p + 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_2p_plus_1_l1864_186480


namespace NUMINAMATH_CALUDE_least_third_side_length_l1864_186428

theorem least_third_side_length (a b c : ℝ) : 
  a = 8 → b = 6 → c > 0 → a^2 + b^2 ≤ c^2 → c ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_least_third_side_length_l1864_186428


namespace NUMINAMATH_CALUDE_no_linear_factor_l1864_186416

/-- The polynomial p(x,y,z) = x^2-y^2+2yz-z^2+2x-y-3z has no linear factor with integer coefficients. -/
theorem no_linear_factor (x y z : ℤ) : 
  ¬ ∃ (a b c d : ℤ), (a*x + b*y + c*z + d) ∣ (x^2 - y^2 + 2*y*z - z^2 + 2*x - y - 3*z) :=
sorry

end NUMINAMATH_CALUDE_no_linear_factor_l1864_186416


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1864_186463

theorem simplify_trig_expression (θ : Real) (h : θ = 160 * π / 180) :
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1864_186463


namespace NUMINAMATH_CALUDE_cassy_jars_proof_l1864_186431

def initial_jars (boxes_type1 boxes_type2 jars_per_box1 jars_per_box2 leftover_jars : ℕ) : ℕ :=
  boxes_type1 * jars_per_box1 + boxes_type2 * jars_per_box2 + leftover_jars

theorem cassy_jars_proof :
  initial_jars 10 30 12 10 80 = 500 := by
  sorry

end NUMINAMATH_CALUDE_cassy_jars_proof_l1864_186431


namespace NUMINAMATH_CALUDE_y1_value_l1864_186402

theorem y1_value (y1 y2 y3 : Real) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1-y1)^2 + 2*(y1-y2)^2 + 2*(y2-y3)^2 + y3^2 = 1/2) :
  y1 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_y1_value_l1864_186402


namespace NUMINAMATH_CALUDE_range_of_4a_minus_2b_l1864_186436

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : -1 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 2 ≤ a + b ∧ a + b ≤ 4) : 
  ∃ (x : ℝ), -1 ≤ x ∧ x ≤ 7 ∧ x = 4*a - 2*b :=
by sorry

end NUMINAMATH_CALUDE_range_of_4a_minus_2b_l1864_186436


namespace NUMINAMATH_CALUDE_max_flour_mass_difference_l1864_186454

/-- The mass of a bag of flour in kg -/
structure FlourBag where
  mass : ℝ
  mass_range : mass ∈ Set.Icc (25 - 0.2) (25 + 0.2)

/-- The maximum difference in mass between two bags of flour -/
def max_mass_difference (bag1 bag2 : FlourBag) : ℝ :=
  |bag1.mass - bag2.mass|

/-- Theorem stating the maximum possible difference in mass between two bags of flour -/
theorem max_flour_mass_difference :
  ∃ (bag1 bag2 : FlourBag), max_mass_difference bag1 bag2 = 0.4 ∧
  ∀ (bag3 bag4 : FlourBag), max_mass_difference bag3 bag4 ≤ 0.4 := by
sorry

end NUMINAMATH_CALUDE_max_flour_mass_difference_l1864_186454


namespace NUMINAMATH_CALUDE_simplify_expression_l1864_186494

theorem simplify_expression : (7^4 + 4^5) * (2^3 - (-2)^2)^2 = 54800 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1864_186494


namespace NUMINAMATH_CALUDE_lanas_tulips_l1864_186409

/-- The number of tulips Lana picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Lana picked -/
def total_flowers : ℕ := sorry

/-- The number of flowers used for bouquets -/
def used_flowers : ℕ := 70

/-- The number of leftover flowers -/
def leftover_flowers : ℕ := 3

/-- The number of roses Lana picked -/
def roses : ℕ := 37

theorem lanas_tulips :
  (total_flowers = tulips + roses) ∧
  (total_flowers = used_flowers + leftover_flowers) →
  tulips = 36 := by sorry

end NUMINAMATH_CALUDE_lanas_tulips_l1864_186409


namespace NUMINAMATH_CALUDE_committee_formation_count_l1864_186418

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 13

/-- The size of the committee to be formed --/
def committee_size : ℕ := 4

/-- The number of players to be chosen after including player A --/
def remaining_to_choose : ℕ := committee_size - 1

/-- The number of players to choose from after excluding player A --/
def players_to_choose_from : ℕ := total_players - 1

theorem committee_formation_count :
  choose players_to_choose_from remaining_to_choose = 220 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1864_186418


namespace NUMINAMATH_CALUDE_newton_basketball_league_members_l1864_186424

theorem newton_basketball_league_members :
  let headband_cost : ℕ := 3
  let jersey_cost : ℕ := headband_cost + 7
  let items_per_member : ℕ := 2  -- 2 headbands and 2 jerseys
  let total_cost : ℕ := 2700
  (total_cost = (headband_cost * items_per_member + jersey_cost * items_per_member) * 103) :=
by sorry

end NUMINAMATH_CALUDE_newton_basketball_league_members_l1864_186424


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1864_186466

/-- Given two digits A and B in base d > 7 such that AB̅_d + AA̅_d = 172_d, prove that A_d - B_d = 5 --/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) (h1 : d > 7) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d^2 + 7 * d + 2) : 
  A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1864_186466


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1864_186459

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3*(a - b)^2) / (a * b * (1 - a - b)) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1864_186459


namespace NUMINAMATH_CALUDE_divisibility_problem_l1864_186485

theorem divisibility_problem (m p a : ℕ) (hp : Prime p) 
  (hm : p ∣ (m^2 - 2)) (ha : ∃ a : ℕ, p ∣ (a^2 + m - 2)) :
  ∃ b : ℕ, p ∣ (b^2 - m - 2) :=
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1864_186485


namespace NUMINAMATH_CALUDE_range_of_a_l1864_186468

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1864_186468


namespace NUMINAMATH_CALUDE_unfactorable_quartic_l1864_186487

theorem unfactorable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_quartic_l1864_186487


namespace NUMINAMATH_CALUDE_no_exact_two_champions_l1864_186410

-- Define the tournament structure
structure Tournament where
  teams : Type
  plays : teams → teams → Prop
  beats : teams → teams → Prop

-- Define the superiority relation
def superior (t : Tournament) (a b : t.teams) : Prop :=
  t.beats a b ∨ ∃ c, t.beats a c ∧ t.beats c b

-- Define a champion
def is_champion (t : Tournament) (a : t.teams) : Prop :=
  ∀ b : t.teams, b ≠ a → superior t a b

-- Theorem statement
theorem no_exact_two_champions (t : Tournament) :
  ¬∃ (a b : t.teams), a ≠ b ∧
    is_champion t a ∧ is_champion t b ∧
    (∀ c : t.teams, is_champion t c → (c = a ∨ c = b)) :=
sorry

end NUMINAMATH_CALUDE_no_exact_two_champions_l1864_186410


namespace NUMINAMATH_CALUDE_common_course_probability_l1864_186472

/-- Represents the set of all possible course selections for a student -/
def CourseSelection : Type := Fin 10

/-- The total number of possible course selections for three students -/
def totalCombinations : ℕ := 1000

/-- The number of favorable combinations where at least two students share two courses -/
def favorableCombinations : ℕ := 280

/-- The probability that any one student will have at least two elective courses in common with the other two students -/
def commonCourseProbability : ℚ := 79 / 250

theorem common_course_probability :
  (favorableCombinations : ℚ) / totalCombinations = commonCourseProbability := by
  sorry

end NUMINAMATH_CALUDE_common_course_probability_l1864_186472


namespace NUMINAMATH_CALUDE_f_minimum_value_l1864_186417

noncomputable def f (x : ℝ) : ℝ := ((2 * x - 1) * Real.exp x) / (x - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min = 3 / 2 ∧
  (∀ x : ℝ, x ≠ 1 → f x ≥ f x_min) ∧
  f x_min = 4 * Real.exp (3 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1864_186417


namespace NUMINAMATH_CALUDE_inequality_generalization_l1864_186471

theorem inequality_generalization (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n / x > n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_generalization_l1864_186471


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1864_186486

theorem divisibility_by_five (a b : ℕ) (n : ℕ) : 
  (5 ∣ n^2 - 1) → (5 ∣ a ∨ 5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1864_186486


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1864_186474

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧
    (∀ k > 0, p k = 2 * p (k - 1) + 1 ∨ p k = 2 * p (k - 1) - 1) ∧
    (∀ m, ∃ n > m, p n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l1864_186474


namespace NUMINAMATH_CALUDE_thumbtack_probability_estimate_l1864_186419

-- Define the structure for the frequency table entry
structure FrequencyEntry :=
  (throws : ℕ)
  (touchingGround : ℕ)
  (frequency : ℚ)

-- Define the frequency table
def frequencyTable : List FrequencyEntry := [
  ⟨40, 20, 1/2⟩,
  ⟨120, 50, 417/1000⟩,
  ⟨320, 146, 456/1000⟩,
  ⟨480, 219, 456/1000⟩,
  ⟨720, 328, 456/1000⟩,
  ⟨800, 366, 458/1000⟩,
  ⟨920, 421, 458/1000⟩,
  ⟨1000, 463, 463/1000⟩
]

-- Define the function to estimate the probability
def estimateProbability (table : List FrequencyEntry) : ℚ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem thumbtack_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateProbability frequencyTable - 46/100| < ε :=
sorry

end NUMINAMATH_CALUDE_thumbtack_probability_estimate_l1864_186419


namespace NUMINAMATH_CALUDE_minimum_computer_units_l1864_186457

theorem minimum_computer_units (x : ℕ) : x ≥ 105 ↔ 
  (5500 * 60 + 5000 * (x - 60) > 550000) := by sorry

end NUMINAMATH_CALUDE_minimum_computer_units_l1864_186457


namespace NUMINAMATH_CALUDE_lcm_perfect_square_l1864_186406

theorem lcm_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_perfect_square_l1864_186406


namespace NUMINAMATH_CALUDE_triangle_side_expression_l1864_186453

/-- Given a triangle with sides a, b, and c, 
    prove that |a-b-c| + |b-c-a| + |c+a-b| = 3c + a - b -/
theorem triangle_side_expression 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq1 : a + b > c) 
  (h_ineq2 : b + c > a) 
  (h_ineq3 : c + a > b) : 
  |a - b - c| + |b - c - a| + |c + a - b| = 3 * c + a - b :=
sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l1864_186453


namespace NUMINAMATH_CALUDE_pencil_distribution_l1864_186461

/-- Given a class with 8 students and 120 pencils, prove that when the pencils are divided equally,
    each student receives 15 pencils. -/
theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) (pencils_per_student : ℕ) 
    (h1 : num_students = 8)
    (h2 : num_pencils = 120)
    (h3 : num_pencils = num_students * pencils_per_student) :
  pencils_per_student = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1864_186461


namespace NUMINAMATH_CALUDE_rs_value_l1864_186455

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 5/8) : r * s = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rs_value_l1864_186455


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1864_186490

theorem wire_cut_ratio (x y : ℝ) : 
  x > 0 → y > 0 → -- Ensure positive lengths
  (4 * (x / 4) = 5 * (y / 5)) → -- Equal perimeters condition
  x / y = 1 := by
sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1864_186490


namespace NUMINAMATH_CALUDE_baker_a_remaining_pastries_l1864_186443

/-- Baker A's initial number of cakes -/
def baker_a_initial_cakes : ℕ := 7

/-- Baker A's initial number of pastries -/
def baker_a_initial_pastries : ℕ := 148

/-- Baker B's initial number of cakes -/
def baker_b_initial_cakes : ℕ := 10

/-- Baker B's initial number of pastries -/
def baker_b_initial_pastries : ℕ := 200

/-- Number of pastries Baker A sold -/
def baker_a_sold_pastries : ℕ := 103

/-- Theorem: Baker A will have 71 pastries left after redistribution and selling -/
theorem baker_a_remaining_pastries :
  (baker_a_initial_pastries + baker_b_initial_pastries) / 2 - baker_a_sold_pastries = 71 := by
  sorry

end NUMINAMATH_CALUDE_baker_a_remaining_pastries_l1864_186443


namespace NUMINAMATH_CALUDE_problem_solution_l1864_186475

theorem problem_solution : ∃ Y : ℚ, 
  let A : ℚ := 2010 / 3
  let B : ℚ := A / 3
  Y = A + B ∧ Y = 893 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1864_186475


namespace NUMINAMATH_CALUDE_a_minus_b_value_l1864_186447

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) :
  a - b = 7 ∨ a - b = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l1864_186447


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1864_186499

/-- The range of m for which the line y = kx + 2 (k ∈ ℝ) always intersects the ellipse x² + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), ∃ (x y : ℝ), 
    y = k * x + 2 ∧ 
    x^2 + y^2 / m = 1 ↔ 
    m ∈ Set.Ici (4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1864_186499


namespace NUMINAMATH_CALUDE_triangle_shape_l1864_186438

theorem triangle_shape (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.cos A + 2 * Real.cos C) / (Real.cos A + 2 * Real.cos B) = Real.sin B / Real.sin C) :
  A = π / 2 ∨ B = C :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l1864_186438


namespace NUMINAMATH_CALUDE_dan_remaining_limes_l1864_186433

/-- The number of limes Dan has after giving some to Sara -/
def limes_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 5 limes remaining -/
theorem dan_remaining_limes :
  limes_remaining 9 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_remaining_limes_l1864_186433


namespace NUMINAMATH_CALUDE_constant_difference_function_property_l1864_186464

/-- A linear function with constant difference -/
def ConstantDifferenceFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ d : ℝ, f (d + 2) - f d = 6)

theorem constant_difference_function_property (f : ℝ → ℝ) 
  (h : ConstantDifferenceFunction f) : f 1 - f 7 = -18 := by
  sorry

end NUMINAMATH_CALUDE_constant_difference_function_property_l1864_186464


namespace NUMINAMATH_CALUDE_cow_count_l1864_186444

/-- Given a group of ducks and cows, where the total number of legs is 36 more than
    twice the number of heads, prove that the number of cows is 18. -/
theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 36) → cows = 18 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l1864_186444


namespace NUMINAMATH_CALUDE_unique_solution_l1864_186446

theorem unique_solution : ∃! (a b c d : ℤ),
  (a^2 - b^2 - c^2 - d^2 = c - b - 2) ∧
  (2*a*b = a - d - 32) ∧
  (2*a*c = 28 - a - d) ∧
  (2*a*d = b + c + 31) ∧
  (a = 5) ∧ (b = -3) ∧ (c = 2) ∧ (d = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1864_186446


namespace NUMINAMATH_CALUDE_intersection_line_intersection_distance_l1864_186430

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B

-- Theorem for the line equation
theorem intersection_line (A B : ℝ × ℝ) (h : intersection_points A B) :
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

-- Theorem for the distance between intersection points
theorem intersection_distance (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_intersection_distance_l1864_186430


namespace NUMINAMATH_CALUDE_group_size_is_nine_l1864_186405

/-- The number of people in the original group -/
def n : ℕ := sorry

/-- The age of the person joining the group -/
def joining_age : ℕ := 34

/-- The original average age of the group -/
def original_average : ℕ := 14

/-- The new average age after the person joins -/
def new_average : ℕ := 16

/-- The minimum age in the group -/
def min_age : ℕ := 10

/-- There are two sets of twins in the group -/
axiom twin_sets : ∃ (a b : ℕ), (2 * a + 2 * b ≤ n)

/-- All individuals in the group are at least 10 years old -/
axiom all_above_min_age : ∀ (age : ℕ), age ≥ min_age

/-- The sum of ages in the original group -/
def original_sum : ℕ := n * original_average

/-- The sum of ages after the new person joins -/
def new_sum : ℕ := original_sum + joining_age

theorem group_size_is_nine :
  n * original_average + joining_age = new_average * (n + 1) →
  n = 9 := by sorry

end NUMINAMATH_CALUDE_group_size_is_nine_l1864_186405


namespace NUMINAMATH_CALUDE_cube_root_negative_27_l1864_186440

theorem cube_root_negative_27 :
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) ∧
  (¬ (∀ x : ℝ, x^2 = 64 → x = 8 ∨ x = -8)) ∧
  (¬ ((-Real.sqrt 2)^2 = 4)) ∧
  (¬ (Real.sqrt ((-5)^2) = -5)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_negative_27_l1864_186440


namespace NUMINAMATH_CALUDE_james_total_toys_l1864_186432

/-- The number of toy cars James buys to maximize his discount -/
def num_cars : ℕ := 26

/-- The number of toy soldiers James buys -/
def num_soldiers : ℕ := 2 * num_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := num_cars + num_soldiers

theorem james_total_toys :
  (num_soldiers = 2 * num_cars) ∧ 
  (num_cars > 25) ∧
  (∀ n : ℕ, n > num_cars → n > 25) →
  total_toys = 78 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l1864_186432


namespace NUMINAMATH_CALUDE_prob_sum_five_l1864_186449

/-- The number of sides on each die -/
def dice_sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def roll_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range dice_sides) (Finset.range dice_sides)

/-- The set of outcomes that sum to 5 -/
def sum_five : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 + p.2 = 5) roll_outcomes

/-- The probability of rolling a sum of 5 with two fair dice -/
theorem prob_sum_five :
  (Finset.card sum_five : ℚ) / (Finset.card roll_outcomes : ℚ) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_five_l1864_186449


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l1864_186426

theorem bernoulli_inequality (x : ℝ) (n : ℕ) 
  (h1 : x > -1) (h2 : x ≠ 0) (h3 : n > 1) : 
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l1864_186426


namespace NUMINAMATH_CALUDE_hippos_satisfy_conditions_l1864_186423

/-- The number of hippos in a herd that satisfies the given conditions -/
def initial_hippos : ℕ := 35

/-- The conditions of the herd -/
def herd_conditions (h : ℕ) : Prop :=
  let initial_elephants : ℕ := 20
  let female_hippos : ℕ := (5 * h) / 7
  let newborn_hippos : ℕ := 5 * female_hippos
  let newborn_elephants : ℕ := newborn_hippos + 10
  let total_animals : ℕ := 315
  initial_elephants + h + newborn_hippos + newborn_elephants = total_animals

/-- Theorem stating that the initial number of hippos satisfies the herd conditions -/
theorem hippos_satisfy_conditions : herd_conditions initial_hippos := by
  sorry


end NUMINAMATH_CALUDE_hippos_satisfy_conditions_l1864_186423


namespace NUMINAMATH_CALUDE_thirtieth_term_is_119_l1864_186470

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := fun n => a₁ + (n - 1) * d

/-- The first term of our sequence -/
def a₁ : ℝ := 3

/-- The second term of our sequence -/
def a₂ : ℝ := 7

/-- The third term of our sequence -/
def a₃ : ℝ := 11

/-- The common difference of our sequence -/
def d : ℝ := a₂ - a₁

/-- The 30th term of our sequence -/
def a₃₀ : ℝ := arithmeticSequence a₁ d 30

theorem thirtieth_term_is_119 : a₃₀ = 119 := by sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_119_l1864_186470


namespace NUMINAMATH_CALUDE_larger_number_problem_l1864_186451

theorem larger_number_problem (a b : ℝ) : 
  a + b = 40 → a - b = 10 → a > b → a = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1864_186451


namespace NUMINAMATH_CALUDE_complement_of_28_45_l1864_186484

/-- The complement of an angle is the difference between 90 degrees and the angle. -/
def complement (angle : ℚ) : ℚ := 90 - angle

/-- Converts degrees and minutes to a rational number representation of degrees. -/
def toDecimalDegrees (degrees : ℕ) (minutes : ℕ) : ℚ :=
  degrees + (minutes : ℚ) / 60

theorem complement_of_28_45 :
  let α := toDecimalDegrees 28 45
  complement α = toDecimalDegrees 61 15 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_28_45_l1864_186484


namespace NUMINAMATH_CALUDE_square_mod_five_not_three_l1864_186498

theorem square_mod_five_not_three (n : ℕ) : (n ^ 2) % 5 ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_five_not_three_l1864_186498


namespace NUMINAMATH_CALUDE_inequality_proof_l1864_186465

theorem inequality_proof (x : ℝ) (h : x ≥ (1/2 : ℝ)) :
  Real.sqrt (9*x + 7) < Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) ∧
  Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) < Real.sqrt (9*x + 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1864_186465


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l1864_186473

/-- Represents a 24-hour digital clock with specific display errors -/
structure ErrorClock where
  /-- The clock displays 9 instead of 1 -/
  error_one : Nat
  /-- The clock displays 5 instead of 2 -/
  error_two : Nat

/-- Calculates the fraction of the day an ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  sorry

/-- Theorem stating that the ErrorClock shows the correct time for 7/36 of the day -/
theorem error_clock_correct_time_fraction :
  ∀ (clock : ErrorClock),
  clock.error_one = 9 ∧ clock.error_two = 5 →
  correct_time_fraction clock = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l1864_186473


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1864_186492

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1864_186492


namespace NUMINAMATH_CALUDE_lives_lost_l1864_186452

theorem lives_lost (initial_lives gained_lives final_lives : ℕ) : 
  initial_lives = 43 → gained_lives = 27 → final_lives = 56 → 
  ∃ (lost_lives : ℕ), initial_lives - lost_lives + gained_lives = final_lives ∧ lost_lives = 14 := by
sorry

end NUMINAMATH_CALUDE_lives_lost_l1864_186452


namespace NUMINAMATH_CALUDE_base5_digits_of_1234_l1864_186404

/-- The number of digits in the base-5 representation of a positive integer n -/
def base5Digits (n : ℕ+) : ℕ :=
  Nat.log 5 n + 1

/-- Theorem: The number of digits in the base-5 representation of 1234 is 5 -/
theorem base5_digits_of_1234 : base5Digits 1234 = 5 := by
  sorry

end NUMINAMATH_CALUDE_base5_digits_of_1234_l1864_186404


namespace NUMINAMATH_CALUDE_circle_equation_l1864_186476

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def circle_passes_through_vertices (cx cy r : ℝ) : Prop :=
  (cx - 4)^2 + cy^2 = r^2 ∧
  cx^2 + (cy - 2)^2 = r^2 ∧
  cx^2 + (cy + 2)^2 = r^2

def center_on_negative_x_axis (cx cy : ℝ) : Prop :=
  cx < 0 ∧ cy = 0

theorem circle_equation (cx cy r : ℝ) :
  ellipse 4 0 ∧ ellipse 0 2 ∧ ellipse 0 (-2) ∧
  circle_passes_through_vertices cx cy r ∧
  center_on_negative_x_axis cx cy →
  cx = -3/2 ∧ cy = 0 ∧ r = 5/2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1864_186476


namespace NUMINAMATH_CALUDE_right_triangle_revolution_is_cone_l1864_186495

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Solid of revolution generated by rotating a right-angled triangle about one of its legs -/
def solidOfRevolution (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: The solid of revolution generated by rotating a right-angled triangle
    about one of its legs is a cone -/
theorem right_triangle_revolution_is_cone (t : RightTriangle) :
  ∃ (c : Cone), solidOfRevolution t = c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_revolution_is_cone_l1864_186495


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1864_186488

theorem remainder_divisibility (N : ℤ) : N % 17 = 2 → N % 357 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1864_186488


namespace NUMINAMATH_CALUDE_area_triangle_BCD_l1864_186469

/-- Given a triangle ABC with area 50 square units and base AC of 6 units,
    and an extension of AC to point D such that CD is 36 units long,
    prove that the area of triangle BCD is 300 square units. -/
theorem area_triangle_BCD (h : ℝ) : 
  (1/2 : ℝ) * 6 * h = 50 →  -- Area of triangle ABC
  (1/2 : ℝ) * 36 * h = 300  -- Area of triangle BCD
  := by sorry

end NUMINAMATH_CALUDE_area_triangle_BCD_l1864_186469


namespace NUMINAMATH_CALUDE_cube_remainder_mod_nine_l1864_186489

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7) → n^3 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_mod_nine_l1864_186489


namespace NUMINAMATH_CALUDE_linear_system_fraction_sum_l1864_186496

theorem linear_system_fraction_sum (a b c u v w : ℝ) 
  (eq1 : 17 * u + b * v + c * w = 0)
  (eq2 : a * u + 29 * v + c * w = 0)
  (eq3 : a * u + b * v + 56 * w = 0)
  (ha : a ≠ 17)
  (hu : u ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_fraction_sum_l1864_186496


namespace NUMINAMATH_CALUDE_negation_equivalence_l1864_186441

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1864_186441


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1864_186456

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1864_186456


namespace NUMINAMATH_CALUDE_bryden_receives_amount_l1864_186434

/-- The face value of a state quarter in dollars -/
def quarterValue : ℚ := 1 / 4

/-- The number of quarters Bryden has -/
def brydenQuarters : ℕ := 5

/-- The percentage multiplier offered by the collector -/
def collectorMultiplier : ℚ := 25

/-- The total amount Bryden will receive in dollars -/
def brydenReceives : ℚ := brydenQuarters * quarterValue * collectorMultiplier

theorem bryden_receives_amount : brydenReceives = 125 / 4 := by sorry

end NUMINAMATH_CALUDE_bryden_receives_amount_l1864_186434


namespace NUMINAMATH_CALUDE_unique_solution_l1864_186401

/-- The function f(x) = x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 1

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (a b : ℝ) : 
  a ≠ 0 ∧ 
  (∀ x : ℝ, f x - f a = (x - b) * (x - a)^2) → 
  a = -2 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1864_186401


namespace NUMINAMATH_CALUDE_current_speed_l1864_186425

/-- Calculates the speed of the current given the rowing speed in still water and the time taken to cover a distance downstream. -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  rowing_speed = 30 →
  distance = 100 →
  time = 9.99920006399488 →
  (distance / time) * 3.6 - rowing_speed = 6 := by
  sorry

#eval (100 / 9.99920006399488) * 3.6 - 30

end NUMINAMATH_CALUDE_current_speed_l1864_186425


namespace NUMINAMATH_CALUDE_lcm_24_30_40_50_l1864_186427

theorem lcm_24_30_40_50 : Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 50)) = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_50_l1864_186427


namespace NUMINAMATH_CALUDE_root_interval_sum_l1864_186412

def f (x : ℝ) := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_root_interval_sum_l1864_186412


namespace NUMINAMATH_CALUDE_prob_fourth_six_after_three_ones_l1864_186467

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a specific number on a given die --/
def prob_roll (d : Die) (n : Nat) : ℚ :=
  match d, n with
  | Die.Fair, _ => 1/6
  | Die.Biased, 1 => 1/3
  | Die.Biased, 6 => 1/3
  | Die.Biased, _ => 1/15

/-- Probability of rolling three ones in a row on a given die --/
def prob_three_ones (d : Die) : ℚ :=
  (prob_roll d 1) ^ 3

/-- Prior probability of choosing each die --/
def prior_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a six on the fourth roll
    after observing three ones --/
theorem prob_fourth_six_after_three_ones :
  let posterior_fair := (prior_prob * prob_three_ones Die.Fair) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  let posterior_biased := (prior_prob * prob_three_ones Die.Biased) /
    (prior_prob * prob_three_ones Die.Fair + prior_prob * prob_three_ones Die.Biased)
  posterior_fair * (prob_roll Die.Fair 6) + posterior_biased * (prob_roll Die.Biased 6) = 17/54 := by
  sorry

/-- The sum of numerator and denominator in the final probability --/
def result : ℕ := 17 + 54

#eval result  -- Should output 71

end NUMINAMATH_CALUDE_prob_fourth_six_after_three_ones_l1864_186467


namespace NUMINAMATH_CALUDE_max_value_of_ab_l1864_186445

theorem max_value_of_ab (a b : ℝ) (h1 : b > 0) (h2 : 3 * a + 4 * b = 2) :
  a * b ≤ 1 / 12 ∧ ∃ (a₀ b₀ : ℝ), b₀ > 0 ∧ 3 * a₀ + 4 * b₀ = 2 ∧ a₀ * b₀ = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l1864_186445


namespace NUMINAMATH_CALUDE_lunks_needed_for_20_apples_l1864_186442

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (1/2) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks required to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (2/4) * kunks

theorem lunks_needed_for_20_apples : 
  lunks_for_apples 20 = 24 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_20_apples_l1864_186442


namespace NUMINAMATH_CALUDE_twice_product_of_sum_and_difference_l1864_186460

theorem twice_product_of_sum_and_difference (x y : ℝ) 
  (sum_eq : x + y = 80) 
  (diff_eq : x - y = 10) : 
  2 * x * y = 3150 := by
sorry

end NUMINAMATH_CALUDE_twice_product_of_sum_and_difference_l1864_186460


namespace NUMINAMATH_CALUDE_circle_equation_k_l1864_186422

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
theorem circle_equation_k (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  k = -16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_l1864_186422


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1864_186403

theorem real_part_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 3*i) * i
  Complex.re z = -3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1864_186403


namespace NUMINAMATH_CALUDE_part_I_part_II_l1864_186413

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 3*y^2 = 4
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  ellipse t.A.1 t.A.2 ∧ 
  ellipse t.B.1 t.B.2 ∧
  line_l t.C.1 t.C.2 ∧
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1) = 1

-- Theorem for part I
theorem part_I (t : Triangle) (h : triangle_conditions t) 
  (h_origin : t.A.1 = 0 ∧ t.A.2 = 0) :
  (∃ (AB_length area : ℝ), 
    AB_length = 2 * Real.sqrt 2 ∧ 
    area = 2) :=
sorry

-- Theorem for part II
theorem part_II (t : Triangle) (h : triangle_conditions t) 
  (h_right_angle : (t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2) = 0)
  (h_max_AC : ∀ (t' : Triangle), triangle_conditions t' → 
    (t'.C.1 - t'.A.1)^2 + (t'.C.2 - t'.A.2)^2 ≤ (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) :
  (∃ (m : ℝ), m = -1 ∧ t.B.2 - t.A.2 = t.B.1 - t.A.1 ∧ t.A.2 = t.A.1 + m) :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1864_186413


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1864_186497

/-- Sum of digits function -/
def sum_of_digits (n : ℤ) : ℤ := sorry

theorem divisible_by_nine (x : ℤ) 
  (h : sum_of_digits x = sum_of_digits (3 * x)) : 
  9 ∣ x := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1864_186497


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1864_186407

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1864_186407


namespace NUMINAMATH_CALUDE_remainder_problem_l1864_186435

theorem remainder_problem : (55^55 + 15) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1864_186435


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1864_186491

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of five consecutive terms starting from the third term is 250 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 250

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 2 + a 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1864_186491


namespace NUMINAMATH_CALUDE_quadratic_local_symmetry_exponential_local_symmetry_range_l1864_186458

/-- A function f has a local symmetry point at x₀ if f(-x₀) = -f(x₀) -/
def has_local_symmetry_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (-x₀) = -f x₀

/-- Theorem 1: The quadratic function ax² + bx - a has a local symmetry point -/
theorem quadratic_local_symmetry
  (a b : ℝ) (ha : a ≠ 0) :
  ∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ a * x^2 + b * x - a) x₀ :=
sorry

/-- Theorem 2: Range of m for which 4^x - m * 2^(n+1) + m - 3 has a local symmetry point -/
theorem exponential_local_symmetry_range (n : ℝ) :
  ∃ m_min m_max : ℝ,
    (∀ m : ℝ, (∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ 4^x - m * 2^(n+1) + m - 3) x₀)
              ↔ m_min ≤ m ∧ m ≤ m_max) ∧
    m_min = 1 - Real.sqrt 3 ∧
    m_max = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_local_symmetry_exponential_local_symmetry_range_l1864_186458


namespace NUMINAMATH_CALUDE_square_ends_same_digits_l1864_186483

theorem square_ends_same_digits (n : ℕ) : 
  (10 ≤ n ∧ n < 100) → 
  (n^2 % 100 = n ↔ n = 25 ∨ n = 76) := by
sorry

end NUMINAMATH_CALUDE_square_ends_same_digits_l1864_186483


namespace NUMINAMATH_CALUDE_walters_chores_l1864_186477

theorem walters_chores (normal_pay exceptional_pay total_days total_earnings : ℕ) 
  (h1 : normal_pay = 3)
  (h2 : exceptional_pay = 6)
  (h3 : total_days = 10)
  (h4 : total_earnings = 42) :
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_walters_chores_l1864_186477


namespace NUMINAMATH_CALUDE_triangle_stack_sum_impossible_l1864_186478

theorem triangle_stack_sum_impossible : ¬ ∃ k : ℕ+, 6 * k = 165 := by
  sorry

end NUMINAMATH_CALUDE_triangle_stack_sum_impossible_l1864_186478


namespace NUMINAMATH_CALUDE_max_set_size_with_prime_triple_sums_l1864_186421

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if the sum of any three elements in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The main theorem -/
theorem max_set_size_with_prime_triple_sums :
  ∀ (l : List ℕ), (∀ x ∈ l, x > 0) → l.Nodup → allTripleSumsPrime l → l.length ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_set_size_with_prime_triple_sums_l1864_186421


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l1864_186482

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (-1, -1)
  f2 : ℝ × ℝ := (-1, -3)
  -- Point on the ellipse
  p : ℝ × ℝ := (4, -2)
  -- Constants in the ellipse equation
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- a and b are positive
  a_pos : a > 0
  b_pos : b > 0
  -- The point p satisfies the ellipse equation
  eq_satisfied : (((p.1 - h)^2 / a^2) + ((p.2 - k)^2 / b^2)) = 1

/-- Theorem stating that a + k = 3 for the given ellipse -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l1864_186482


namespace NUMINAMATH_CALUDE_negation_of_zero_product_property_l1864_186479

theorem negation_of_zero_product_property :
  (¬ ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0) ↔
  (∃ (x y : ℝ), x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_zero_product_property_l1864_186479
