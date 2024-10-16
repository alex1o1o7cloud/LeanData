import Mathlib

namespace NUMINAMATH_CALUDE_jared_popcorn_order_l2839_283993

/-- Calculates the minimum number of popcorn servings needed for a group -/
def min_popcorn_servings (pieces_per_serving : ℕ) (jared_consumption : ℕ) 
  (friend_group1_size : ℕ) (friend_group1_consumption : ℕ)
  (friend_group2_size : ℕ) (friend_group2_consumption : ℕ) : ℕ :=
  let total_consumption := jared_consumption + 
    friend_group1_size * friend_group1_consumption +
    friend_group2_size * friend_group2_consumption
  (total_consumption + pieces_per_serving - 1) / pieces_per_serving

/-- The minimum number of popcorn servings needed for Jared and his friends is 21 -/
theorem jared_popcorn_order : 
  min_popcorn_servings 50 120 5 90 3 150 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jared_popcorn_order_l2839_283993


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l2839_283984

theorem opposite_of_negative_one_half : 
  -((-1 : ℚ) / 2) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l2839_283984


namespace NUMINAMATH_CALUDE_stone_pile_combination_l2839_283972

/-- Two piles are considered similar if their sizes differ by at most a factor of two -/
def similar (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A combining operation takes two piles and creates a new pile with their combined size -/
def combine (x y : ℕ) : ℕ := x + y

/-- A sequence of combining operations -/
def combineSequence : List (ℕ × ℕ) → List ℕ
  | [] => []
  | (x, y) :: rest => combine x y :: combineSequence rest

/-- The theorem states that for any number of stones, there exists a sequence of
    combining operations that results in a single pile, using only similar piles -/
theorem stone_pile_combination (n : ℕ) :
  ∃ (seq : List (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ seq → similar x y) ∧
    (combineSequence seq = [n]) ∧
    (seq.foldl (λ acc (x, y) => acc - 1) n = 1) :=
  sorry

end NUMINAMATH_CALUDE_stone_pile_combination_l2839_283972


namespace NUMINAMATH_CALUDE_prom_tip_percentage_l2839_283931

theorem prom_tip_percentage : 
  let ticket_cost : ℕ := 100
  let dinner_cost : ℕ := 120
  let limo_hourly_rate : ℕ := 80
  let limo_hours : ℕ := 6
  let total_cost : ℕ := 836
  let tip_percentage : ℚ := (total_cost - (2 * ticket_cost + dinner_cost + limo_hourly_rate * limo_hours)) / dinner_cost * 100
  tip_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_prom_tip_percentage_l2839_283931


namespace NUMINAMATH_CALUDE_book_collection_problem_l2839_283921

theorem book_collection_problem (shared_books books_alice books_bob_unique : ℕ) 
  (h1 : shared_books = 12)
  (h2 : books_alice = 26)
  (h3 : books_bob_unique = 8) :
  books_alice - shared_books + books_bob_unique = 22 := by
  sorry

end NUMINAMATH_CALUDE_book_collection_problem_l2839_283921


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2839_283970

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Checks if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- Checks if a natural number contains the digit 3 -/
def containsThree (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 3 ∨ b = 3 ∨ c = 3)

theorem sum_of_numbers (A : ThreeDigitNumber) (B C : TwoDigitNumber) :
  ((containsSeven A.val ∧ containsSeven B.val) ∨
   (containsSeven A.val ∧ containsSeven C.val) ∨
   (containsSeven B.val ∧ containsSeven C.val)) →
  (containsThree B.val ∧ containsThree C.val) →
  (A.val + B.val + C.val = 208) →
  (B.val + C.val = 76) →
  A.val + B.val + C.val = 247 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2839_283970


namespace NUMINAMATH_CALUDE_sum_of_b_values_l2839_283986

theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 9 * x^2 + b₁ * x + 15 * x + 16 = 0) ∧
  (∃! x, 9 * x^2 + b₂ * x + 15 * x + 16 = 0) →
  b₁ + b₂ = -30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_b_values_l2839_283986


namespace NUMINAMATH_CALUDE_problem_1_l2839_283965

theorem problem_1 (x : ℕ) : 2 * 8^x * 16^x = 2^22 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2839_283965


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l2839_283909

/-- Represents a digit in a given base --/
def IsDigit (d : ℕ) (base : ℕ) : Prop := d < base

/-- Converts a two-digit number in a given base to base 10 --/
def ToBase10 (d : ℕ) (base : ℕ) : ℕ := base * d + d

/-- The problem statement --/
theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (c d : ℕ),
    IsDigit c 6 ∧
    IsDigit d 8 ∧
    ToBase10 c 6 = n ∧
    ToBase10 d 8 = n ∧
    (∀ (m : ℕ) (c' d' : ℕ),
      IsDigit c' 6 →
      IsDigit d' 8 →
      ToBase10 c' 6 = m →
      ToBase10 d' 8 = m →
      n ≤ m) ∧
    n = 63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l2839_283909


namespace NUMINAMATH_CALUDE_ab_greater_than_b_squared_l2839_283967

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_b_squared_l2839_283967


namespace NUMINAMATH_CALUDE_maries_age_l2839_283928

theorem maries_age (marie_age marco_age : ℕ) : 
  marco_age = 2 * marie_age + 1 →
  marie_age + marco_age = 37 →
  marie_age = 12 := by
sorry

end NUMINAMATH_CALUDE_maries_age_l2839_283928


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2839_283975

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a + b + c ≥ 0 ∧ a*b + a*c + b*c = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2839_283975


namespace NUMINAMATH_CALUDE_acute_triangle_on_perpendicular_lines_l2839_283992

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute_a : a^2 < b^2 + c^2
  acute_b : b^2 < a^2 + c^2
  acute_c : c^2 < a^2 + b^2

-- Theorem statement
theorem acute_triangle_on_perpendicular_lines (t : AcuteTriangle) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 = t.c^2 ∧
  x^2 + z^2 = t.b^2 ∧
  y^2 + z^2 = t.a^2 :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_on_perpendicular_lines_l2839_283992


namespace NUMINAMATH_CALUDE_finleys_age_l2839_283917

/-- Proves Finley's age given the conditions in the problem -/
theorem finleys_age (jill_age : ℕ) (roger_age : ℕ) (finley_age : ℕ) : 
  jill_age = 20 →
  roger_age = 2 * jill_age + 5 →
  (roger_age + 15) - (jill_age + 15) = finley_age - 30 →
  finley_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_finleys_age_l2839_283917


namespace NUMINAMATH_CALUDE_cone_volume_l2839_283947

/-- Given a cone with generatrix length 2 and unfolded side sector area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (generatrix : ℝ) (sector_area : ℝ) :
  generatrix = 2 →
  sector_area = 2 * Real.pi →
  ∃ (volume : ℝ), volume = (Real.sqrt 3 * Real.pi) / 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2839_283947


namespace NUMINAMATH_CALUDE_equation_equality_l2839_283966

theorem equation_equality : 3 * 6524 = 8254 * 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2839_283966


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2839_283950

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = surface_area →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2839_283950


namespace NUMINAMATH_CALUDE_soda_price_calculation_l2839_283938

/-- Proves that the original price of each soda is $20/9 given the conditions of the problem -/
theorem soda_price_calculation (num_sodas : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_sodas = 3 →
  discount_rate = 1/10 →
  total_paid = 6 →
  ∃ (original_price : ℚ), 
    original_price = 20/9 ∧ 
    num_sodas * (original_price * (1 - discount_rate)) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l2839_283938


namespace NUMINAMATH_CALUDE_election_winner_votes_l2839_283936

theorem election_winner_votes (total_votes : ℕ) (candidates : ℕ) 
  (difference1 : ℕ) (difference2 : ℕ) (difference3 : ℕ) :
  total_votes = 963 →
  candidates = 4 →
  difference1 = 53 →
  difference2 = 79 →
  difference3 = 105 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - difference1) + 
    (winner_votes - difference2) + (winner_votes - difference3) = total_votes ∧
    winner_votes = 300 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2839_283936


namespace NUMINAMATH_CALUDE_triangle_radii_area_relations_l2839_283957

/-- Given a triangle with side lengths a, b, c, and semiperimeter p = (a + b + c) / 2,
    let r be the inradius, and r_a, r_b, r_c be the exradii opposite to sides a, b, c respectively.
    S represents the area of the triangle. -/
theorem triangle_radii_area_relations (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let p := (a + b + c) / 2
  let r := Real.sqrt (((p - a) * (p - b) * (p - c)) / p)
  let r_a := p / (p - a) * r
  let r_b := p / (p - b) * r
  let r_c := p / (p - c) * r
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  (r * p = r_a * (p - a) ∧ 
   r * r_a = (p - b) * (p - c) ∧ 
   r_b * r_c = p * (p - a)) ∧
  S^2 = p * (p - a) * (p - b) * (p - c) ∧
  S^2 = r * r_a * r_b * r_c :=
by sorry


end NUMINAMATH_CALUDE_triangle_radii_area_relations_l2839_283957


namespace NUMINAMATH_CALUDE_range_of_a_l2839_283961

/-- Proposition p: The function y=(a-1)^x is increasing with respect to x -/
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 1) ^ x < (a - 1) ^ y

/-- Proposition q: The inequality -3^x ≤ a is true for all positive real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x > 0 → -3 ^ x ≤ a

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : -1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2839_283961


namespace NUMINAMATH_CALUDE_sum_k_squared_over_3_to_k_l2839_283908

open Real

/-- The sum of the infinite series k^2 / 3^k from k = 1 to infinity is 7 -/
theorem sum_k_squared_over_3_to_k (S : ℝ) : 
  (∑' k, (k : ℝ)^2 / 3^k) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_k_squared_over_3_to_k_l2839_283908


namespace NUMINAMATH_CALUDE_l₃_symmetric_to_l₁_wrt_l₂_l2839_283927

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (f g h : ℝ → ℝ → Prop) : Prop :=
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    f x₁ y₁ → h x₂ y₂ → 
    ∃ (x₀ y₀ : ℝ), g x₀ y₀ ∧ 
      x₀ = (x₁ + x₂) / 2 ∧ 
      y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem l₃_symmetric_to_l₁_wrt_l₂ : symmetric_wrt l₁ l₂ l₃ := by
  sorry

end NUMINAMATH_CALUDE_l₃_symmetric_to_l₁_wrt_l₂_l2839_283927


namespace NUMINAMATH_CALUDE_division_problem_l2839_283987

theorem division_problem (divisor : ℕ) : 
  (171 / divisor = 8) ∧ (171 % divisor = 3) → divisor = 21 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2839_283987


namespace NUMINAMATH_CALUDE_delores_remaining_money_l2839_283934

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (computer : ℕ) (printer : ℕ) : ℕ :=
  initial - (computer + printer)

/-- Proves that Delores has $10 left after her purchases -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_remaining_money_l2839_283934


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_general_form_with_remainder_one_l2839_283903

theorem smallest_integer_with_remainder_one : ∃ (a : ℕ), a > 0 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) ∧
  (∀ b : ℕ, b > 0 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → b % k = 1) → a ≤ b) ∧
  a = 2521 :=
sorry

theorem general_form_with_remainder_one :
  ∀ (a : ℕ), a > 0 →
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 9 → a % k = 1) →
  ∃ (n : ℕ), a = 2520 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_general_form_with_remainder_one_l2839_283903


namespace NUMINAMATH_CALUDE_min_cost_is_128_l2839_283948

/-- Represents the cost of each type of flower -/
structure FlowerCost where
  sunflower : ℕ
  tulip : ℕ
  orchid : ℚ
  rose : ℕ
  hydrangea : ℕ

/-- Represents the areas of different regions in the garden -/
structure GardenRegions where
  small_region1 : ℕ
  small_region2 : ℕ
  medium_region : ℕ
  large_region : ℕ

/-- Calculates the minimum cost of the garden given the flower costs and garden regions -/
def min_garden_cost (costs : FlowerCost) (regions : GardenRegions) : ℚ :=
  costs.sunflower * regions.small_region1 +
  costs.sunflower * regions.small_region2 +
  costs.tulip * regions.medium_region +
  costs.hydrangea * regions.large_region

theorem min_cost_is_128 (costs : FlowerCost) (regions : GardenRegions) :
  costs.sunflower = 1 ∧ 
  costs.tulip = 2 ∧ 
  costs.orchid = 5/2 ∧ 
  costs.rose = 3 ∧ 
  costs.hydrangea = 4 ∧
  regions.small_region1 = 8 ∧
  regions.small_region2 = 8 ∧
  regions.medium_region = 6 ∧
  regions.large_region = 25 →
  min_garden_cost costs regions = 128 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_is_128_l2839_283948


namespace NUMINAMATH_CALUDE_specific_competition_scores_l2839_283911

/-- Represents a mathematics competition with the given scoring rules. -/
structure MathCompetition where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unattempted_points : ℤ

/-- Calculates the number of different total scores possible in the competition. -/
def countPossibleScores (comp : MathCompetition) : ℕ := sorry

/-- The specific competition described in the problem. -/
def specificCompetition : MathCompetition :=
  { total_problems := 30
  , correct_points := 4
  , incorrect_points := -1
  , unattempted_points := 0 }

/-- Theorem stating that the number of different total scores in the specific competition is 145. -/
theorem specific_competition_scores :
  countPossibleScores specificCompetition = 145 := by sorry

end NUMINAMATH_CALUDE_specific_competition_scores_l2839_283911


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2839_283922

/-- Given that 7 oranges weigh the same as 5 apples, prove that 28 oranges weigh the same as 20 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    7 * orange_weight = 5 * apple_weight →
    28 * orange_weight = 20 * apple_weight :=
by sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l2839_283922


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2839_283990

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.33125 ↔ n = 53 ∧ d = 160 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2839_283990


namespace NUMINAMATH_CALUDE_smallest_positive_angle_proof_l2839_283901

/-- The smallest positive angle with the same terminal side as 400° -/
def smallest_positive_angle : ℝ := 40

/-- The set of angles with the same terminal side as 400° -/
def angle_set (k : ℤ) : ℝ := 400 + k * 360

theorem smallest_positive_angle_proof :
  ∀ k : ℤ, angle_set k > 0 → smallest_positive_angle ≤ angle_set k :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_proof_l2839_283901


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2839_283997

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_pos : a 1 > 0) (h_prod : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2839_283997


namespace NUMINAMATH_CALUDE_greatest_integer_X_l2839_283904

theorem greatest_integer_X (Z₀ : ℝ) (h1 : 5/3 ≤ |Z₀|) (h2 : |Z₀| < 5/2) :
  (∃ X : ℤ, (∀ Y : ℤ, |Y * Z₀| ≤ 5 → Y ≤ X) ∧ |X * Z₀| ≤ 5) ∧
  (∀ X : ℤ, (∀ Y : ℤ, |Y * Z₀| ≤ 5 → Y ≤ X) ∧ |X * Z₀| ≤ 5 → X = 2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_X_l2839_283904


namespace NUMINAMATH_CALUDE_secret_room_number_l2839_283939

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def has_digit_8 (n : ℕ) : Prop := (n / 10 = 8) ∨ (n % 10 = 8)

def exactly_three_true (p q r s : Prop) : Prop :=
  (p ∧ q ∧ r ∧ ¬s) ∨ (p ∧ q ∧ ¬r ∧ s) ∨ (p ∧ ¬q ∧ r ∧ s) ∨ (¬p ∧ q ∧ r ∧ s)

theorem secret_room_number (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : exactly_three_true (divisible_by_4 n) (is_odd n) (sum_of_digits n = 12) (has_digit_8 n)) :
  n % 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_secret_room_number_l2839_283939


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2839_283919

theorem polynomial_factorization (y : ℤ) :
  5 * (y + 4) * (y + 7) * (y + 9) * (y + 11) - 4 * y^2 =
  (y + 1) * (y + 9) * (5 * y^2 + 33 * y + 441) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2839_283919


namespace NUMINAMATH_CALUDE_beta_value_l2839_283962

theorem beta_value (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2)
  (h_sin_α : Real.sin α = Real.sqrt 5 / 5)
  (h_sin_α_β : Real.sin (α - β) = -(Real.sqrt 10) / 10) :
  β = π/4 := by
sorry

end NUMINAMATH_CALUDE_beta_value_l2839_283962


namespace NUMINAMATH_CALUDE_teacher_pen_cost_l2839_283914

/-- The total cost of pens purchased by a teacher -/
theorem teacher_pen_cost : 
  let black_pens : ℕ := 7
  let blue_pens : ℕ := 9
  let red_pens : ℕ := 5
  let black_pen_cost : ℚ := 125/100
  let blue_pen_cost : ℚ := 150/100
  let red_pen_cost : ℚ := 175/100
  (black_pens : ℚ) * black_pen_cost + 
  (blue_pens : ℚ) * blue_pen_cost + 
  (red_pens : ℚ) * red_pen_cost = 31 :=
by sorry

end NUMINAMATH_CALUDE_teacher_pen_cost_l2839_283914


namespace NUMINAMATH_CALUDE_haley_necklace_count_l2839_283989

/-- The number of necklaces Haley, Jason, and Josh have satisfy the given conditions -/
def NecklaceProblem (h j q : ℕ) : Prop :=
  (h = j + 5) ∧ (q = j / 2) ∧ (h = q + 15)

/-- Theorem: If the necklace counts satisfy the given conditions, then Haley has 25 necklaces -/
theorem haley_necklace_count
  (h j q : ℕ) (hcond : NecklaceProblem h j q) : h = 25 := by
  sorry

end NUMINAMATH_CALUDE_haley_necklace_count_l2839_283989


namespace NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l2839_283974

/-- The cost price of an article given profit and loss conditions -/
theorem cost_price_from_profit_loss_equality (cost_price : ℝ) : 
  (66 - cost_price = cost_price - 22) → cost_price = 44 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_from_profit_loss_equality_l2839_283974


namespace NUMINAMATH_CALUDE_new_observations_sum_l2839_283980

theorem new_observations_sum (initial_count : ℕ) (initial_avg : ℚ) (new_count : ℕ) (new_avg : ℚ) :
  initial_count = 9 →
  initial_avg = 15 →
  new_count = 3 →
  new_avg = 13 →
  (initial_count * initial_avg + new_count * (3 * new_avg - initial_count * initial_avg)) / new_count = 21 :=
by sorry

end NUMINAMATH_CALUDE_new_observations_sum_l2839_283980


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2839_283978

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ (k ≤ 2 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2839_283978


namespace NUMINAMATH_CALUDE_simplify_expression_l2839_283918

theorem simplify_expression (a : ℚ) : ((2 * a + 6) - 3 * a) / 2 = -a / 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2839_283918


namespace NUMINAMATH_CALUDE_inequalities_hold_l2839_283954

theorem inequalities_hold (a b : ℝ) (h : a * b > 0) :
  (2 * (a^2 + b^2) ≥ (a + b)^2) ∧
  (b / a + a / b ≥ 2) ∧
  ((a + 1 / a) * (b + 1 / b) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2839_283954


namespace NUMINAMATH_CALUDE_equation_holds_for_all_x_l2839_283941

theorem equation_holds_for_all_x : ∃ (a b c : ℝ), ∀ (x : ℝ), 
  (x + a)^2 + (2*x + b)^2 + (2*x + c)^2 = (3*x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_x_l2839_283941


namespace NUMINAMATH_CALUDE_surface_area_of_cut_and_rearranged_solid_l2839_283930

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the cuts made on the solid -/
structure Cuts where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the surface area of the new solid formed by cutting and rearranging -/
def surfaceArea (d : Dimensions) (c : Cuts) : ℝ :=
  2 * d.length * d.width +  -- top and bottom
  2 * d.length * d.height +  -- front and back
  2 * d.width * d.height    -- sides

/-- The main theorem to prove -/
theorem surface_area_of_cut_and_rearranged_solid
  (d : Dimensions)
  (c : Cuts)
  (h1 : d.length = 2 ∧ d.width = 1 ∧ d.height = 1)
  (h2 : c.first = 1/4 ∧ c.second = 5/12 ∧ c.third = 19/36) :
  surfaceArea d c = 10 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cut_and_rearranged_solid_l2839_283930


namespace NUMINAMATH_CALUDE_bowling_tournament_prize_orders_l2839_283996

/-- Represents a bowling tournament with 6 players and a specific playoff structure. -/
structure BowlingTournament :=
  (num_players : Nat)
  (playoff_structure : List (Nat × Nat))

/-- Calculates the number of possible prize order combinations in a bowling tournament. -/
def possiblePrizeOrders (tournament : BowlingTournament) : Nat :=
  2^(tournament.num_players - 1)

/-- Theorem stating that the number of possible prize order combinations
    in the given 6-player bowling tournament is 32. -/
theorem bowling_tournament_prize_orders :
  ∃ (t : BowlingTournament),
    t.num_players = 6 ∧
    t.playoff_structure = [(6, 5), (4, 0), (3, 0), (2, 0), (1, 0)] ∧
    possiblePrizeOrders t = 32 :=
by
  sorry


end NUMINAMATH_CALUDE_bowling_tournament_prize_orders_l2839_283996


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_real_solution_l2839_283910

/-- The quadratic inequality -/
def quadratic_inequality (k x : ℝ) : Prop := k * x^2 - 2 * x + 6 * k < 0

/-- The solution set for part 1 -/
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

/-- The solution set for part 2 -/
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ solution_set_1 x) → k = -2/5 :=
sorry

theorem quadratic_inequality_real_solution (k : ℝ) :
  (k ≠ 0 ∧ ∀ x, quadratic_inequality k x ↔ x ∈ solution_set_2) → 
  k < 0 ∧ k < -Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_real_solution_l2839_283910


namespace NUMINAMATH_CALUDE_complex_product_real_l2839_283912

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + x * Complex.I
  (z₁ * z₂).im = 0 → x = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2839_283912


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l2839_283915

theorem min_sum_of_primes (p q r s : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (30 ∣ p * q - r * s) →
  54 ≤ p + q + r + s :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l2839_283915


namespace NUMINAMATH_CALUDE_apartment_cost_l2839_283973

/-- The cost of a room on the first floor of Krystiana's apartment building. -/
def first_floor_cost : ℝ := 8.75

/-- The number of rooms on each floor. -/
def rooms_per_floor : ℕ := 3

/-- The additional cost for a room on the second floor compared to the first floor. -/
def second_floor_additional_cost : ℝ := 20

theorem apartment_cost (total_earnings : ℝ) 
  (h_total : total_earnings = 165) :
  first_floor_cost * rooms_per_floor + 
  (first_floor_cost + second_floor_additional_cost) * rooms_per_floor + 
  (2 * first_floor_cost) * rooms_per_floor = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_l2839_283973


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2839_283959

theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  x₀^2 = 28 * y₀ →                           -- Point is on the parabola
  (y₀ + 7/2)^2 + x₀^2 = 9 * y₀^2 →           -- Distance to focus is 3 times distance to x-axis
  y₀ = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2839_283959


namespace NUMINAMATH_CALUDE_specific_cube_structure_surface_area_l2839_283960

/-- A solid structure composed of unit cubes -/
structure CubeStructure :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)
  (total_cubes : ℕ)

/-- Calculate the surface area of a CubeStructure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific CubeStructure has a surface area of 78 square units -/
theorem specific_cube_structure_surface_area :
  ∃ (s : CubeStructure), s.length = 5 ∧ s.width = 3 ∧ s.height = 3 ∧ s.total_cubes = 15 ∧ surface_area s = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_specific_cube_structure_surface_area_l2839_283960


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l2839_283956

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3)
  (h2 : 2 < a - b ∧ a - b < 4) :
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l2839_283956


namespace NUMINAMATH_CALUDE_impossibleTransformation_l2839_283902

-- Define the type for a card
def Card := ℤ × ℤ

-- Define the operations of the machines
def machine1 (c : Card) : Card :=
  (c.1 + 1, c.2 + 1)

def machine2 (c : Card) : Option Card :=
  if c.1 % 2 = 0 ∧ c.2 % 2 = 0 then some (c.1 / 2, c.2 / 2) else none

def machine3 (c1 c2 : Card) : Option Card :=
  if c1.2 = c2.1 then some (c1.1, c2.2) else none

-- Define the property that the difference is divisible by 7
def diffDivisibleBy7 (c : Card) : Prop :=
  (c.1 - c.2) % 7 = 0

-- Theorem stating the impossibility of the transformation
theorem impossibleTransformation :
  ∀ (c : Card), diffDivisibleBy7 c →
  ¬∃ (sequence : List (Card → Card)), 
    (List.foldl (λ acc f => f acc) c sequence = (1, 1988)) :=
by sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l2839_283902


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l2839_283933

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no intersection points in the real plane. -/
theorem line_circle_no_intersection :
  ¬ ∃ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l2839_283933


namespace NUMINAMATH_CALUDE_p_satisfies_equation_l2839_283907

/-- The polynomial p(x) that satisfies the given equation -/
def p (x : ℝ) : ℝ := -2*x^4 - 2*x^3 + 5*x^2 - 2*x + 2

/-- The theorem stating that p(x) satisfies the given equation -/
theorem p_satisfies_equation (x : ℝ) :
  4*x^4 + 2*x^3 - 6*x + 4 + p x = 2*x^4 + 5*x^2 - 8*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_equation_l2839_283907


namespace NUMINAMATH_CALUDE_fourth_term_is_2016_l2839_283999

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  second_term : a 2 = 606
  sum_first_four : a 1 + a 2 + a 3 + a 4 = 3834

/-- The fourth term of the arithmetic sequence is 2016 -/
theorem fourth_term_is_2016 (seq : ArithmeticSequence) : seq.a 4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_2016_l2839_283999


namespace NUMINAMATH_CALUDE_angle_ratio_theorem_l2839_283943

theorem angle_ratio_theorem (α : Real) (m : Real) :
  m < 0 →
  let P : Real × Real := (4 * m, -3 * m)
  (P.1 / (Real.sqrt (P.1^2 + P.2^2)) = -4/5) →
  (P.2 / (Real.sqrt (P.1^2 + P.2^2)) = 3/5) →
  (2 * (P.2 / (Real.sqrt (P.1^2 + P.2^2))) + (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) /
  ((P.2 / (Real.sqrt (P.1^2 + P.2^2))) - (P.1 / (Real.sqrt (P.1^2 + P.2^2)))) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_angle_ratio_theorem_l2839_283943


namespace NUMINAMATH_CALUDE_coordinates_of_point_A_l2839_283983

-- Define a Point type for 2D coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem coordinates_of_point_A (A B : Point) : 
  (B.x = 2 ∧ B.y = 4) →  -- Coordinates of point B
  (A.y = B.y) →  -- AB is parallel to x-axis
  ((A.x - B.x)^2 + (A.y - B.y)^2 = 3^2) →  -- Length of AB is 3
  ((A.x = 5 ∧ A.y = 4) ∨ (A.x = -1 ∧ A.y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_A_l2839_283983


namespace NUMINAMATH_CALUDE_buns_left_is_two_l2839_283976

/-- The number of buns initially on the plate -/
def initial_buns : ℕ := 15

/-- Karlsson takes three times as many buns as Little Boy -/
def karlsson_multiplier : ℕ := 3

/-- Bimbo takes three times fewer buns than Little Boy -/
def bimbo_divisor : ℕ := 3

/-- The number of buns Bimbo takes -/
def bimbo_buns : ℕ := 1

/-- The number of buns Little Boy takes -/
def little_boy_buns : ℕ := bimbo_buns * bimbo_divisor

/-- The number of buns Karlsson takes -/
def karlsson_buns : ℕ := little_boy_buns * karlsson_multiplier

/-- The total number of buns taken -/
def total_taken : ℕ := bimbo_buns + little_boy_buns + karlsson_buns

/-- The number of buns left on the plate -/
def buns_left : ℕ := initial_buns - total_taken

theorem buns_left_is_two : buns_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_buns_left_is_two_l2839_283976


namespace NUMINAMATH_CALUDE_minimum_implies_a_range_l2839_283920

/-- A function f with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if f has a minimum in (0,2), then a is in (0,4) -/
theorem minimum_implies_a_range (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 0 2, ∀ x ∈ Set.Ioo 0 2, f a x₀ ≤ f a x) →
  a ∈ Set.Ioo 0 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_implies_a_range_l2839_283920


namespace NUMINAMATH_CALUDE_angle_equality_l2839_283964

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l2839_283964


namespace NUMINAMATH_CALUDE_x_value_when_y_is_one_l2839_283969

theorem x_value_when_y_is_one (x y : ℝ) :
  y = 1 / (3 * x + 1) → y = 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_one_l2839_283969


namespace NUMINAMATH_CALUDE_calculate_F_2_f_3_l2839_283971

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 3*a + 2
def F (a b : ℝ) : ℝ := b^2 + a + 1

-- State the theorem
theorem calculate_F_2_f_3 : F 2 (f 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculate_F_2_f_3_l2839_283971


namespace NUMINAMATH_CALUDE_crushers_win_probability_l2839_283923

theorem crushers_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 4/5) :
  p^n = 4096/15625 := by
  sorry

end NUMINAMATH_CALUDE_crushers_win_probability_l2839_283923


namespace NUMINAMATH_CALUDE_count_valid_triples_l2839_283900

def valid_triple (x y z : ℕ+) : Prop :=
  Nat.lcm x.val y.val = 180 ∧
  Nat.lcm x.val z.val = 420 ∧
  Nat.lcm y.val z.val = 1260

theorem count_valid_triples :
  ∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)),
    (∀ t ∈ s, valid_triple t.1 t.2.1 t.2.2) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l2839_283900


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l2839_283995

theorem andreas_living_room_area :
  ∀ (floor_area carpet_area : ℝ),
    carpet_area = 4 * 9 →
    0.75 * floor_area = carpet_area →
    floor_area = 48 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l2839_283995


namespace NUMINAMATH_CALUDE_panda_babies_l2839_283979

theorem panda_babies (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 → 
  pregnancy_rate = 1/4 → 
  (total_pandas / 2 : ℚ) * pregnancy_rate = 2 :=
sorry

end NUMINAMATH_CALUDE_panda_babies_l2839_283979


namespace NUMINAMATH_CALUDE_seller_took_weight_l2839_283926

/-- Given 10 weights with masses n, n+1, ..., n+9, if the sum of 9 of these weights is 1457,
    then the missing weight is 158. -/
theorem seller_took_weight (n : ℕ) (x : ℕ) (h1 : x ≤ 9) 
    (h2 : (10 * n + 45) - (n + x) = 1457) : n + x = 158 := by
  sorry

end NUMINAMATH_CALUDE_seller_took_weight_l2839_283926


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2839_283982

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2839_283982


namespace NUMINAMATH_CALUDE_amanda_marbles_l2839_283981

theorem amanda_marbles (katrina_marbles : ℕ) (amanda_marbles : ℕ) (mabel_marbles : ℕ) : 
  mabel_marbles = 5 * katrina_marbles →
  mabel_marbles = 85 →
  mabel_marbles = amanda_marbles + 63 →
  2 * katrina_marbles - amanda_marbles = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_amanda_marbles_l2839_283981


namespace NUMINAMATH_CALUDE_compute_F_3_f_5_l2839_283988

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^3 - a

-- State the theorem
theorem compute_F_3_f_5 : F 3 (f 5) = 24 := by sorry

end NUMINAMATH_CALUDE_compute_F_3_f_5_l2839_283988


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l2839_283991

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), IsLocalMin f x₀ ∧ x₀ = 2 := by sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l2839_283991


namespace NUMINAMATH_CALUDE_max_value_on_interval_l2839_283968

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem max_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a x ≤ f a y) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 3 ≤ f a x) →
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a y ≤ f a x ∧ f a x = 57) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l2839_283968


namespace NUMINAMATH_CALUDE_gcd_of_37500_and_61250_l2839_283955

theorem gcd_of_37500_and_61250 : Nat.gcd 37500 61250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_37500_and_61250_l2839_283955


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2839_283985

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2839_283985


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_l2839_283929

theorem sum_of_sixth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^3 + b^3 + c^3 = 8)
  (h3 : a^5 + b^5 + c^5 = 32) :
  a^6 + b^6 + c^6 = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_l2839_283929


namespace NUMINAMATH_CALUDE_linear_function_intersection_l2839_283958

/-- A linear function passing through (-1, -2) intersects the y-axis at (0, -1) -/
theorem linear_function_intersection (k : ℝ) : 
  (∀ x y, y = k * (x - 1) → (x = -1 ∧ y = -2) → (0 = k * (-1 - 1) + 2)) → 
  (∃ y, y = k * (0 - 1) ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l2839_283958


namespace NUMINAMATH_CALUDE_fruit_salad_ratio_l2839_283944

def total_salads : ℕ := 600
def alaya_salads : ℕ := 200

theorem fruit_salad_ratio :
  let angel_salads := total_salads - alaya_salads
  (angel_salads : ℚ) / alaya_salads = 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_ratio_l2839_283944


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2839_283977

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 2)
  (h_fifth : a 5 = a 4 + 2) :
  a 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2839_283977


namespace NUMINAMATH_CALUDE_andrews_donation_l2839_283946

/-- The age when Andrew started donating -/
def start_age : ℕ := 11

/-- Andrew's current age -/
def current_age : ℕ := 29

/-- The amount Andrew donates each year in thousands -/
def yearly_donation : ℕ := 7

/-- Calculate the total amount Andrew has donated -/
def total_donation : ℕ := (current_age - start_age) * yearly_donation

/-- Theorem stating that Andrew's total donation is 126k -/
theorem andrews_donation : total_donation = 126 := by sorry

end NUMINAMATH_CALUDE_andrews_donation_l2839_283946


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2839_283998

/-- Given an arithmetic sequence {a_n} where a_3 = 20 - a_6, prove that S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →               -- sum formula
  a 3 = 20 - a 6 →                                   -- given condition
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2839_283998


namespace NUMINAMATH_CALUDE_wax_remaining_l2839_283953

/-- The amount of wax remaining after detailing vehicles -/
def remaining_wax (initial : ℕ) (spilled : ℕ) (car : ℕ) (suv : ℕ) : ℕ :=
  initial - spilled - car - suv

/-- Theorem stating the remaining wax after detailing vehicles -/
theorem wax_remaining :
  remaining_wax 11 2 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_remaining_l2839_283953


namespace NUMINAMATH_CALUDE_multitive_function_thirtysix_l2839_283916

/-- A function satisfying f(a · b) = f(a) + f(b) -/
def MultitiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a * b) = f a + f b

/-- Theorem: Given a multitive function f with f(2) = p and f(3) = q, prove f(36) = 2(p + q) -/
theorem multitive_function_thirtysix
  (f : ℝ → ℝ) (p q : ℝ)
  (hf : MultitiveFunction f)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 36 = 2 * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_multitive_function_thirtysix_l2839_283916


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2839_283952

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (h1 : area = 27 * Real.sqrt 3) 
  (h2 : area = 3 * s^2 * (1/2)) : 
  s = 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2839_283952


namespace NUMINAMATH_CALUDE_amc_scoring_l2839_283906

theorem amc_scoring (total_problems : Nat) (correct_points : Int) (incorrect_points : Int) 
  (unanswered_points : Int) (attempted : Nat) (unanswered : Nat) (min_score : Int) : 
  let min_correct := ((min_score - unanswered * unanswered_points) - 
    (attempted * incorrect_points)) / (correct_points - incorrect_points)
  ⌈min_correct⌉ = 17 :=
by
  sorry

#check amc_scoring 30 7 (-1) 2 25 5 120

end NUMINAMATH_CALUDE_amc_scoring_l2839_283906


namespace NUMINAMATH_CALUDE_cave_depth_l2839_283994

theorem cave_depth (total_depth remaining_distance current_depth : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : remaining_distance = 369)
  (h3 : current_depth = total_depth - remaining_distance) :
  current_depth = 849 := by
  sorry

end NUMINAMATH_CALUDE_cave_depth_l2839_283994


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2839_283913

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > y ∧ y > 0 → x/y > 1) ∧
  ∃ a b : ℝ, a/b > 1 ∧ ¬(a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2839_283913


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2839_283925

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem geometric_progression_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 5^(1/3 : ℝ)) 
  (h2 : a * r = 5^(1/5 : ℝ)) 
  (h3 : a * r^2 = 5^(1/15 : ℝ)) : 
  a * r^3 = 5^(-1/15 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l2839_283925


namespace NUMINAMATH_CALUDE_class_notification_problem_l2839_283940

theorem class_notification_problem (n : ℕ) : 
  (1 + n + n^2 = 43) ↔ (n = 6) :=
by sorry

end NUMINAMATH_CALUDE_class_notification_problem_l2839_283940


namespace NUMINAMATH_CALUDE_point_translation_l2839_283924

/-- Given a point A(-5, 6) in a Cartesian coordinate system, 
    moving it 5 units right and 6 units up results in point A₁(0, 12) -/
theorem point_translation :
  let A : ℝ × ℝ := (-5, 6)
  let right_shift : ℝ := 5
  let up_shift : ℝ := 6
  let A₁ : ℝ × ℝ := (A.1 + right_shift, A.2 + up_shift)
  A₁ = (0, 12) := by
sorry

end NUMINAMATH_CALUDE_point_translation_l2839_283924


namespace NUMINAMATH_CALUDE_factory_uses_systematic_sampling_factory_sampling_is_systematic_l2839_283942

-- Define the characteristics of the sampling method
structure SamplingMethod where
  regular_intervals : Bool
  fixed_position : Bool
  continuous_process : Bool

-- Define Systematic Sampling
def SystematicSampling : SamplingMethod :=
  { regular_intervals := true
  , fixed_position := true
  , continuous_process := true }

-- Define the factory's sampling method
def FactorySamplingMethod : SamplingMethod :=
  { regular_intervals := true  -- Every 5 minutes
  , fixed_position := true     -- Fixed position on conveyor belt
  , continuous_process := true -- Conveyor belt process
  }

-- Theorem to prove
theorem factory_uses_systematic_sampling :
  FactorySamplingMethod = SystematicSampling := by
  sorry

-- Additional theorem to show that the factory's method is indeed Systematic Sampling
theorem factory_sampling_is_systematic :
  FactorySamplingMethod.regular_intervals ∧
  FactorySamplingMethod.fixed_position ∧
  FactorySamplingMethod.continuous_process := by
  sorry

end NUMINAMATH_CALUDE_factory_uses_systematic_sampling_factory_sampling_is_systematic_l2839_283942


namespace NUMINAMATH_CALUDE_first_month_sale_l2839_283932

theorem first_month_sale (sales_2 sales_3 sales_4 sales_5 sales_6 : ℕ)
  (h1 : sales_2 = 6927)
  (h2 : sales_3 = 6855)
  (h3 : sales_4 = 7230)
  (h4 : sales_5 = 6562)
  (h5 : sales_6 = 4791)
  (desired_average : ℕ)
  (h6 : desired_average = 6500)
  (num_months : ℕ)
  (h7 : num_months = 6) :
  ∃ (sales_1 : ℕ), sales_1 = 6635 ∧
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = desired_average :=
by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l2839_283932


namespace NUMINAMATH_CALUDE_butterfly_count_l2839_283905

/-- Given a number of butterflies, each with 12 black dots, and a total of 4764 black dots,
    prove that the number of butterflies is 397. -/
theorem butterfly_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ)
    (h1 : total_black_dots = 4764)
    (h2 : black_dots_per_butterfly = 12) :
    total_black_dots / black_dots_per_butterfly = 397 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_count_l2839_283905


namespace NUMINAMATH_CALUDE_blue_sky_project_expo_course_l2839_283963

theorem blue_sky_project_expo_course (n m : ℕ) (hn : n = 6) (hm : m = 6) :
  (Nat.choose n 2) * (m - 1) ^ (n - 2) = 
    (Nat.choose 6 2) * 5^4 :=
sorry

end NUMINAMATH_CALUDE_blue_sky_project_expo_course_l2839_283963


namespace NUMINAMATH_CALUDE_two_x_equals_y_l2839_283949

theorem two_x_equals_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : x + 2*y = 5) : 
  2*x = y := by sorry

end NUMINAMATH_CALUDE_two_x_equals_y_l2839_283949


namespace NUMINAMATH_CALUDE_second_player_wins_l2839_283937

/-- A game played on a circle with 2n + 1 equally spaced points. -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A player in the game. -/
inductive Player
  | First
  | Second

/-- A strategy for a player. -/
def Strategy (g : CircleGame) := List (Fin (2 * g.n + 1)) → Fin (2 * g.n + 1)

/-- Predicate to check if all remaining triangles are obtuse. -/
def AllTrianglesObtuse (g : CircleGame) (remaining : List (Fin (2 * g.n + 1))) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for a player. -/
def IsWinningStrategy (g : CircleGame) (p : Player) (s : Strategy g) : Prop :=
  sorry

/-- Theorem stating that the second player has a winning strategy. -/
theorem second_player_wins (g : CircleGame) :
  ∃ (s : Strategy g), IsWinningStrategy g Player.Second s :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2839_283937


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2839_283945

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2839_283945


namespace NUMINAMATH_CALUDE_find_b_value_l2839_283935

/-- Given two functions p and q, where p(x) = 2x - 11 and q(x) = 5x - b,
    prove that b = 8 when p(q(3)) = 3. -/
theorem find_b_value (b : ℝ) : 
  let p : ℝ → ℝ := λ x ↦ 2 * x - 11
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 3 → b = 8 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l2839_283935


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2839_283951

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def prob_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_diagonals 2) / Nat.choose S.card 2

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2839_283951
