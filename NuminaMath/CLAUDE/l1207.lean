import Mathlib

namespace NUMINAMATH_CALUDE_abc_sum_sixteen_l1207_120718

theorem abc_sum_sixteen (a b c : ℤ) 
  (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4)
  (h4 : ¬(a = b ∧ b = c))
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) :
  a + b + c = 16 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_sixteen_l1207_120718


namespace NUMINAMATH_CALUDE_sandwich_cost_is_90_cents_l1207_120731

/-- The cost of making a sandwich with two slices of bread, one slice of ham, and one slice of cheese -/
def sandwich_cost (bread_cost cheese_cost ham_cost : ℚ) : ℚ :=
  2 * bread_cost + cheese_cost + ham_cost

/-- Theorem stating that the cost of making a sandwich is 90 cents -/
theorem sandwich_cost_is_90_cents :
  sandwich_cost 0.15 0.35 0.25 * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_90_cents_l1207_120731


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l1207_120770

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l1207_120770


namespace NUMINAMATH_CALUDE_two_Z_one_eq_one_l1207_120748

/-- The Z operation on two real numbers -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: 2 Z 1 = 1 -/
theorem two_Z_one_eq_one : Z 2 1 = 1 := by sorry

end NUMINAMATH_CALUDE_two_Z_one_eq_one_l1207_120748


namespace NUMINAMATH_CALUDE_chocolate_manufacturer_min_price_l1207_120753

/-- Calculates the minimum selling price per unit for a chocolate manufacturer --/
theorem chocolate_manufacturer_min_price
  (units : ℕ)
  (cost_per_unit : ℝ)
  (min_profit : ℝ)
  (h1 : units = 400)
  (h2 : cost_per_unit = 40)
  (h3 : min_profit = 40000) :
  (min_profit + units * cost_per_unit) / units = 140 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_manufacturer_min_price_l1207_120753


namespace NUMINAMATH_CALUDE_worksheets_graded_l1207_120749

/-- 
Given:
- There are 9 worksheets in total
- Each worksheet has 4 problems
- There are 16 problems left to grade

Prove that the number of worksheets already graded is 5.
-/
theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) :
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets * problems_per_worksheet - problems_left = 5 * problems_per_worksheet :=
by
  sorry

#check worksheets_graded

end NUMINAMATH_CALUDE_worksheets_graded_l1207_120749


namespace NUMINAMATH_CALUDE_jerry_log_count_l1207_120797

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_log_count : totalLogs = 1220 := by
  sorry

end NUMINAMATH_CALUDE_jerry_log_count_l1207_120797


namespace NUMINAMATH_CALUDE_johns_age_l1207_120781

theorem johns_age (john dad brother : ℕ) 
  (h1 : john + 28 = dad)
  (h2 : john + dad = 76)
  (h3 : john + 5 = 2 * (brother + 5)) : 
  john = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l1207_120781


namespace NUMINAMATH_CALUDE_unique_n_exists_l1207_120742

theorem unique_n_exists : ∃! n : ℤ,
  0 ≤ n ∧ n < 17 ∧
  -150 ≡ n [ZMOD 17] ∧
  102 % n = 0 ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_exists_l1207_120742


namespace NUMINAMATH_CALUDE_min_color_changes_l1207_120733

/-- Represents a 10x10 board with colored chips -/
def Board := Fin 10 → Fin 10 → Fin 100

/-- Checks if a chip is unique in its row or column -/
def is_unique (b : Board) (i j : Fin 10) : Prop :=
  (∀ k : Fin 10, k ≠ j → b i k ≠ b i j) ∨
  (∀ k : Fin 10, k ≠ i → b k j ≠ b i j)

/-- Represents a valid color change operation -/
def valid_change (b1 b2 : Board) : Prop :=
  ∃ i j : Fin 10, 
    (∀ x y : Fin 10, (x ≠ i ∨ y ≠ j) → b1 x y = b2 x y) ∧
    is_unique b1 i j ∧
    b1 i j ≠ b2 i j

/-- Represents a sequence of valid color changes -/
def valid_sequence (n : ℕ) : Prop :=
  ∃ (seq : Fin (n + 1) → Board),
    (∀ i : Fin 10, ∀ j : Fin 10, seq 0 i j = i.val * 10 + j.val) ∧
    (∀ k : Fin n, valid_change (seq k) (seq (k + 1))) ∧
    (∀ i j : Fin 10, ¬is_unique (seq n) i j)

/-- The main theorem stating the minimum number of color changes -/
theorem min_color_changes : 
  (∃ n : ℕ, valid_sequence n) ∧ 
  (∀ m : ℕ, m < 75 → ¬valid_sequence m) :=
sorry

end NUMINAMATH_CALUDE_min_color_changes_l1207_120733


namespace NUMINAMATH_CALUDE_smallest_of_four_consecutive_integers_product_2520_l1207_120757

theorem smallest_of_four_consecutive_integers_product_2520 :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧
  ∀ (m : ℕ), m > 0 → m * (m + 1) * (m + 2) * (m + 3) = 2520 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_four_consecutive_integers_product_2520_l1207_120757


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l1207_120780

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 100 * Real.pi) :
  A = Real.pi * r^2 → r = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l1207_120780


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1207_120751

/-- Represents the fraction of orange juice in a mixture -/
def orange_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_oj_fraction pitcher2_oj_fraction : ℚ) : ℚ :=
  let total_oj := pitcher1_capacity * pitcher1_oj_fraction + pitcher2_capacity * pitcher2_oj_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_oj / total_volume

/-- Proves that the fraction of orange juice in the combined mixture is 29167/100000 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 800 800 (1/4) (1/3) = 29167/100000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1207_120751


namespace NUMINAMATH_CALUDE_bonnie_sticker_count_l1207_120759

/-- Calculates Bonnie's initial sticker count given the problem conditions -/
def bonnies_initial_stickers (june_initial : ℕ) (grandparents_gift : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (june_initial + 2 * grandparents_gift)

/-- Theorem stating that Bonnie's initial sticker count is 63 given the problem conditions -/
theorem bonnie_sticker_count :
  bonnies_initial_stickers 76 25 189 = 63 := by
  sorry

#eval bonnies_initial_stickers 76 25 189

end NUMINAMATH_CALUDE_bonnie_sticker_count_l1207_120759


namespace NUMINAMATH_CALUDE_part_one_part_two_l1207_120782

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.A + t.c * Real.sin t.C - Real.sqrt 2 * t.a * Real.sin t.C = t.b * Real.sin t.B

-- Theorem for part (I)
theorem part_one (t : Triangle) (h : given_condition t) : t.B = Real.pi / 4 := by
  sorry

-- Theorem for part (II)
theorem part_two (t : Triangle) (h1 : given_condition t) (h2 : t.A = 5 * Real.pi / 12) (h3 : t.b = 2) :
  t.a = 1 + Real.sqrt 3 ∧ t.c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1207_120782


namespace NUMINAMATH_CALUDE_mapping_has_output_l1207_120788

-- Define sets M and N
variable (M N : Type)

-- Define the mapping f from M to N
variable (f : M → N)

-- Theorem statement
theorem mapping_has_output : ∀ (x : M), ∃ (y : N), f x = y := by
  sorry

end NUMINAMATH_CALUDE_mapping_has_output_l1207_120788


namespace NUMINAMATH_CALUDE_on_time_speed_l1207_120701

-- Define the variables
def distance : ℝ → ℝ → ℝ := λ speed time => speed * time

-- Define the conditions
def early_arrival (d : ℝ) (T : ℝ) : Prop := distance 20 (T - 0.5) = d
def late_arrival (d : ℝ) (T : ℝ) : Prop := distance 12 (T + 0.5) = d

-- Define the theorem
theorem on_time_speed (d : ℝ) (T : ℝ) :
  early_arrival d T → late_arrival d T → distance 15 T = d :=
by sorry

end NUMINAMATH_CALUDE_on_time_speed_l1207_120701


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1207_120767

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

/-- The magnitude (length) of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60° in radians
  (h2 : a = (2, 0))
  (h3 : magnitude b = 1) :
  magnitude (a + 2 • b) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1207_120767


namespace NUMINAMATH_CALUDE_product_expression_value_l1207_120708

def product_expression : ℚ :=
  (3^3 - 2^3) / (3^3 + 2^3) *
  (4^3 - 3^3) / (4^3 + 3^3) *
  (5^3 - 4^3) / (5^3 + 4^3) *
  (6^3 - 5^3) / (6^3 + 5^3) *
  (7^3 - 6^3) / (7^3 + 6^3)

theorem product_expression_value : product_expression = 17 / 901 := by
  sorry

end NUMINAMATH_CALUDE_product_expression_value_l1207_120708


namespace NUMINAMATH_CALUDE_election_winner_votes_l1207_120730

/-- In an election with two candidates, where the winner received 62% of votes
    and won by 408 votes, the number of votes cast for the winning candidate is 1054. -/
theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.62 - (total_votes : ℝ) * 0.38 = 408 →
  (total_votes : ℝ) * 0.62 = 1054 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1207_120730


namespace NUMINAMATH_CALUDE_min_M_n_value_l1207_120793

def M_n (n k : ℕ+) : ℚ :=
  max (40 / n) (max (80 / (k * n)) (60 / (200 - n - k * n)))

theorem min_M_n_value :
  ∀ k : ℕ+, (∃ n : ℕ+, n + k * n ≤ 200) →
    (∀ n : ℕ+, n + k * n ≤ 200 → M_n n k ≥ 10/11) ∧
    (∃ n : ℕ+, n + k * n ≤ 200 ∧ M_n n k = 10/11) :=
by sorry

end NUMINAMATH_CALUDE_min_M_n_value_l1207_120793


namespace NUMINAMATH_CALUDE_prize_stickers_l1207_120735

/-- The number of stickers Christine already has -/
def current_stickers : ℕ := 11

/-- The number of additional stickers Christine needs -/
def additional_stickers : ℕ := 19

/-- The total number of stickers needed for a prize -/
def total_stickers : ℕ := current_stickers + additional_stickers

theorem prize_stickers : total_stickers = 30 := by sorry

end NUMINAMATH_CALUDE_prize_stickers_l1207_120735


namespace NUMINAMATH_CALUDE_parabola_equation_l1207_120714

/-- A parabola with vertex at the origin and directrix x = 2 -/
structure Parabola where
  /-- The equation of the parabola in the form y² = kx -/
  equation : ℝ → ℝ → Prop
  /-- The vertex of the parabola is at the origin -/
  vertex_at_origin : equation 0 0
  /-- The directrix of the parabola has equation x = 2 -/
  directrix_at_two : ∀ y, ¬ equation 2 y

/-- The equation of the parabola is y² = -16x -/
theorem parabola_equation (C : Parabola) : 
  C.equation = fun x y ↦ y^2 = -16*x := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1207_120714


namespace NUMINAMATH_CALUDE_crayon_count_l1207_120776

theorem crayon_count (blue : ℕ) (red : ℕ) (green : ℕ) : 
  blue = 3 → 
  red = 4 * blue → 
  green = 2 * red → 
  blue + red + green = 39 := by
sorry

end NUMINAMATH_CALUDE_crayon_count_l1207_120776


namespace NUMINAMATH_CALUDE_range_of_a_for_two_integer_solutions_l1207_120750

/-- A system of inequalities has exactly two integer solutions -/
def has_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (↑x : ℝ)^2 - (↑x : ℝ) + a - a^2 < 0 ∧ (↑x : ℝ) + 2*a > 1 ∧
    (↑y : ℝ)^2 - (↑y : ℝ) + a - a^2 < 0 ∧ (↑y : ℝ) + 2*a > 1 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y →
      ¬((↑z : ℝ)^2 - (↑z : ℝ) + a - a^2 < 0 ∧ (↑z : ℝ) + 2*a > 1)

/-- The range of a that satisfies the conditions -/
theorem range_of_a_for_two_integer_solutions :
  ∀ a : ℝ, has_two_integer_solutions a ↔ 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_integer_solutions_l1207_120750


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_property_l1207_120700

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The property given in the problem -/
def property (n : ℕ) : Prop :=
  binomial (n + 1) 7 - binomial n 7 = binomial n 8

/-- The theorem to be proved -/
theorem smallest_n_satisfying_property : 
  ∀ n : ℕ, n > 0 → (property n ↔ n ≥ 14) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_property_l1207_120700


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l1207_120705

/-- The distance between the centers of two circular pulleys with an uncrossed belt -/
theorem pulley_centers_distance (r₁ r₂ d : ℝ) (hr₁ : r₁ = 10) (hr₂ : r₂ = 6) (hd : d = 30) :
  Real.sqrt ((r₁ - r₂)^2 + d^2) = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l1207_120705


namespace NUMINAMATH_CALUDE_factorial_of_factorial_div_factorial_l1207_120764

theorem factorial_of_factorial_div_factorial :
  (Nat.factorial (Nat.factorial 3)) / (Nat.factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_div_factorial_l1207_120764


namespace NUMINAMATH_CALUDE_unique_solution_fractional_equation_l1207_120768

theorem unique_solution_fractional_equation :
  ∃! x : ℚ, (1 : ℚ) / (x - 3) = (3 : ℚ) / (x - 6) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fractional_equation_l1207_120768


namespace NUMINAMATH_CALUDE_percent_of_number_l1207_120785

theorem percent_of_number (x : ℝ) : 120 = 1.5 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l1207_120785


namespace NUMINAMATH_CALUDE_digit_add_sequence_contains_even_l1207_120744

/-- A sequence of natural numbers where each term is obtained from the previous term
    by adding one of its nonzero digits. -/
def DigitAddSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ d : ℕ, d > 0 ∧ d < 10 ∧ d ∣ a n ∧ a (n + 1) = a n + d

/-- The theorem stating that a DigitAddSequence contains an even number. -/
theorem digit_add_sequence_contains_even (a : ℕ → ℕ) (h : DigitAddSequence a) :
  ∃ n : ℕ, Even (a n) :=
sorry

end NUMINAMATH_CALUDE_digit_add_sequence_contains_even_l1207_120744


namespace NUMINAMATH_CALUDE_point_distance_on_curve_l1207_120789

theorem point_distance_on_curve (e c d : ℝ) : 
  e > 0 →
  c ≠ d →
  c^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * c + 1 →
  d^2 + (Real.sqrt e)^6 = 3 * (Real.sqrt e)^3 * d + 1 →
  |c - d| = |Real.sqrt (5 * e^3 + 4)| :=
by sorry

end NUMINAMATH_CALUDE_point_distance_on_curve_l1207_120789


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l1207_120717

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l1207_120717


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_7_l1207_120796

def is_multiple_of_7 (n : ℕ) : Prop :=
  (5 * (n - 3)^6 - 2 * n^3 + 20 * n - 35) % 7 = 0

theorem largest_n_multiple_of_7 :
  ∀ n : ℕ, n < 100000 →
    (is_multiple_of_7 n → n ≤ 99998) ∧
    (n > 99998 → ¬is_multiple_of_7 n) ∧
    is_multiple_of_7 99998 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_7_l1207_120796


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1207_120779

theorem consecutive_integers_sum_of_squares : 
  ∀ a : ℤ, (a - 2) * a * (a + 2) = 36 * a → 
  (a - 2)^2 + a^2 + (a + 2)^2 = 200 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1207_120779


namespace NUMINAMATH_CALUDE_hunter_journey_l1207_120741

theorem hunter_journey (swamp_speed forest_speed highway_speed : ℝ)
  (total_time total_distance : ℝ) (swamp_time forest_time highway_time : ℝ) :
  swamp_speed = 2 →
  forest_speed = 4 →
  highway_speed = 6 →
  total_time = 4 →
  total_distance = 15 →
  swamp_time + forest_time + highway_time = total_time →
  swamp_speed * swamp_time + forest_speed * forest_time + highway_speed * highway_time = total_distance →
  swamp_time > highway_time := by
  sorry

end NUMINAMATH_CALUDE_hunter_journey_l1207_120741


namespace NUMINAMATH_CALUDE_number_ratio_problem_l1207_120772

theorem number_ratio_problem (x : ℝ) : 3 * (2 * x + 5) = 111 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l1207_120772


namespace NUMINAMATH_CALUDE_buddy_fraction_l1207_120725

theorem buddy_fraction (s₆ : ℕ) (n₉ : ℕ) : 
  s₆ > 0 ∧ n₉ > 0 →  -- Ensure positive numbers of students
  (n₉ : ℚ) / 4 = (s₆ : ℚ) / 3 →  -- 1/4 of ninth graders paired with 1/3 of sixth graders
  (s₆ : ℚ) / 3 / ((4 * s₆ : ℚ) / 3 + s₆) = 1 / 7 :=
by sorry

#check buddy_fraction

end NUMINAMATH_CALUDE_buddy_fraction_l1207_120725


namespace NUMINAMATH_CALUDE_area_minimized_at_k_equals_one_l1207_120754

/-- Represents a planar region defined by a system of inequalities -/
def PlanarRegion := Set (ℝ × ℝ)

/-- Computes the area of a planar region -/
noncomputable def area (Ω : PlanarRegion) : ℝ := sorry

/-- The system of inequalities that defines Ω -/
def systemOfInequalities (k : ℝ) : PlanarRegion := sorry

theorem area_minimized_at_k_equals_one (k : ℝ) (hk : k ≥ 0) :
  let Ω := systemOfInequalities k
  ∀ k' ≥ 0, area Ω ≤ area (systemOfInequalities k') → k = 1 :=
sorry

end NUMINAMATH_CALUDE_area_minimized_at_k_equals_one_l1207_120754


namespace NUMINAMATH_CALUDE_determinant_scaling_l1207_120729

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 5 →
  Matrix.det ![![2 * a, 2 * b], ![2 * c, 2 * d]] = 20 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l1207_120729


namespace NUMINAMATH_CALUDE_second_divisor_is_nine_l1207_120786

theorem second_divisor_is_nine (least_number : Nat) (second_divisor : Nat) : 
  least_number = 282 →
  least_number % 31 = 3 →
  least_number % second_divisor = 3 →
  second_divisor ≠ 31 →
  second_divisor = 9 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_is_nine_l1207_120786


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_l1207_120787

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c }

/-- Reflects a parabola about the x-axis -/
def reflect (p : Parabola) : Parabola :=
  { a := -p.a
  , b := -p.b
  , c := -p.c }

/-- Adds two parabolas coefficient-wise -/
def add (p q : Parabola) : Parabola :=
  { a := p.a + q.a
  , b := p.b + q.b
  , c := p.c + q.c }

theorem parabola_transformation_sum (p : Parabola) :
  let p1 := translate p 4
  let p2 := translate (reflect p) (-4)
  (add p1 p2).b = -16 * p.a ∧ (add p1 p2).a = 0 ∧ (add p1 p2).c = 0 := by
  sorry

#check parabola_transformation_sum

end NUMINAMATH_CALUDE_parabola_transformation_sum_l1207_120787


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1207_120722

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  B = π / 3 ∧ a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1207_120722


namespace NUMINAMATH_CALUDE_smallest_b_value_l1207_120777

theorem smallest_b_value (b : ℤ) (Q : ℤ → ℤ) : 
  b > 0 →
  (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), Q x = a₀ * x^2 + a₁ * x + a₂) →
  Q 1 = b ∧ Q 4 = b ∧ Q 7 = b ∧ Q 10 = b →
  Q 2 = -b ∧ Q 5 = -b ∧ Q 8 = -b ∧ Q 11 = -b →
  (∀ c : ℤ, c > 0 ∧ 
    (∃ (P : ℤ → ℤ), (∀ x : ℤ, ∃ (a₀ a₁ a₂ : ℤ), P x = a₀ * x^2 + a₁ * x + a₂) ∧
      P 1 = c ∧ P 4 = c ∧ P 7 = c ∧ P 10 = c ∧
      P 2 = -c ∧ P 5 = -c ∧ P 8 = -c ∧ P 11 = -c) →
    c ≥ b) →
  b = 1260 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1207_120777


namespace NUMINAMATH_CALUDE_price_to_relatives_is_correct_l1207_120739

-- Define the given quantities
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def peaches_sold_to_relatives : ℕ := 4
def peaches_kept : ℕ := 1
def price_per_peach_to_friends : ℚ := 2
def total_earnings : ℚ := 25

-- Define the function to calculate the price per peach sold to relatives
def price_per_peach_to_relatives : ℚ :=
  (total_earnings - price_per_peach_to_friends * peaches_sold_to_friends) / peaches_sold_to_relatives

-- Theorem statement
theorem price_to_relatives_is_correct : price_per_peach_to_relatives = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_price_to_relatives_is_correct_l1207_120739


namespace NUMINAMATH_CALUDE_f_has_one_min_no_max_l1207_120799

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

-- State the theorem
theorem f_has_one_min_no_max :
  ∃! x : ℝ, IsLocalMin f x ∧ ∀ y : ℝ, ¬IsLocalMax f y :=
by sorry

end NUMINAMATH_CALUDE_f_has_one_min_no_max_l1207_120799


namespace NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l1207_120710

/-- Given a line that bisects a circle, prove the minimum value of a certain expression -/
theorem line_bisecting_circle_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, 2*a*x + b*y - 2 = 0 → x^2 + y^2 - 2*x - 4*y - 6 = 0) →
  (∃ x₀ y₀ : ℝ, 2*a*x₀ + b*y₀ - 2 = 0 ∧ x₀ = 1 ∧ y₀ = 2) →
  (∀ k : ℝ, 2/a + 1/b ≥ k) →
  k = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_bisecting_circle_min_value_l1207_120710


namespace NUMINAMATH_CALUDE_circular_garden_max_area_l1207_120715

theorem circular_garden_max_area (fence_length : ℝ) (h : fence_length = 200) :
  let radius := fence_length / (2 * Real.pi)
  let area := Real.pi * radius ^ 2
  area = 10000 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_max_area_l1207_120715


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l1207_120706

theorem parabola_intersection_condition (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l1207_120706


namespace NUMINAMATH_CALUDE_scaled_recipe_correct_l1207_120784

/-- Represents a cookie recipe -/
structure CookieRecipe where
  cookies : ℕ
  flour : ℕ
  eggs : ℕ

/-- Scales a cookie recipe by a given factor -/
def scaleRecipe (recipe : CookieRecipe) (factor : ℕ) : CookieRecipe :=
  { cookies := recipe.cookies * factor
  , flour := recipe.flour * factor
  , eggs := recipe.eggs * factor }

theorem scaled_recipe_correct (original : CookieRecipe) (scaled : CookieRecipe) :
  original.cookies = 40 ∧
  original.flour = 3 ∧
  original.eggs = 2 ∧
  scaled = scaleRecipe original 3 →
  scaled.cookies = 120 ∧
  scaled.flour = 9 ∧
  scaled.eggs = 6 := by
  sorry

#check scaled_recipe_correct

end NUMINAMATH_CALUDE_scaled_recipe_correct_l1207_120784


namespace NUMINAMATH_CALUDE_no_real_solutions_for_square_rectangle_area_relation_l1207_120778

theorem no_real_solutions_for_square_rectangle_area_relation :
  ¬ ∃ x : ℝ, (x + 2) * (x - 5) = 2 * (x - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_square_rectangle_area_relation_l1207_120778


namespace NUMINAMATH_CALUDE_ball_radius_is_10_ball_surface_area_is_400pi_l1207_120726

/-- Represents a spherical ball floating on water that leaves a circular hole in ice --/
structure FloatingBall where
  /-- The radius of the circular hole left in the ice --/
  holeRadius : ℝ
  /-- The depth of the hole left in the ice --/
  holeDepth : ℝ
  /-- The radius of the ball --/
  ballRadius : ℝ

/-- The properties of the floating ball problem --/
def floatingBallProblem : FloatingBall where
  holeRadius := 6
  holeDepth := 2
  ballRadius := 10

/-- Theorem stating that the radius of the ball is 10 cm --/
theorem ball_radius_is_10 (ball : FloatingBall) :
  ball.holeRadius = 6 ∧ ball.holeDepth = 2 → ball.ballRadius = 10 := by sorry

/-- Theorem stating that the surface area of the ball is 400π cm² --/
theorem ball_surface_area_is_400pi (ball : FloatingBall) :
  ball.ballRadius = 10 → 4 * Real.pi * ball.ballRadius ^ 2 = 400 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ball_radius_is_10_ball_surface_area_is_400pi_l1207_120726


namespace NUMINAMATH_CALUDE_percentage_of_360_l1207_120719

theorem percentage_of_360 : (32 / 100) * 360 = 115.2 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_l1207_120719


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1207_120766

theorem constant_term_expansion (n : ℕ+) 
  (h : (2 : ℝ)^(n : ℝ) = 32) : 
  Nat.choose n.val 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1207_120766


namespace NUMINAMATH_CALUDE_girls_in_class_l1207_120720

/-- In a class with the following properties:
    - There are 18 boys over 160 cm tall
    - These 18 boys constitute 3/4 of all boys
    - The total number of boys is 2/3 of the total number of students
    Then the number of girls in the class is 12 -/
theorem girls_in_class (tall_boys : ℕ) (total_boys : ℕ) (total_students : ℕ) 
  (h1 : tall_boys = 18)
  (h2 : tall_boys = (3 / 4 : ℚ) * total_boys)
  (h3 : total_boys = (2 / 3 : ℚ) * total_students) :
  total_students - total_boys = 12 := by
  sorry

#check girls_in_class

end NUMINAMATH_CALUDE_girls_in_class_l1207_120720


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1207_120734

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) :
  (∀ x, x^2 - 16*x + 63 = (x - a)*(x - b)) →
  3*b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1207_120734


namespace NUMINAMATH_CALUDE_infinitely_many_terms_same_prime_factors_l1207_120763

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- The set of prime factors of a natural number -/
def primeFactors (n : ℕ) : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ∣ n}

/-- There are infinitely many terms in an arithmetic progression with the same prime factors -/
theorem infinitely_many_terms_same_prime_factors (a d : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, primeFactors (arithmeticProgression a d n) = primeFactors a :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_terms_same_prime_factors_l1207_120763


namespace NUMINAMATH_CALUDE_max_log_sum_l1207_120746

/-- Given that xyz + y + z = 12, the maximum value of log₄x + log₂y + log₂z is 3 -/
theorem max_log_sum (x y z : ℝ) (h : x * y * z + y + z = 12) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.log x / Real.log 4) + (Real.log y / Real.log 2) + (Real.log z / Real.log 2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_l1207_120746


namespace NUMINAMATH_CALUDE_seventh_twenty_ninth_725th_digit_l1207_120736

def decimal_representation (n d : ℕ) : List ℕ := sorry

theorem seventh_twenty_ninth_725th_digit :
  let rep := decimal_representation 7 29
  -- The decimal representation has a period of 28 digits
  ∀ i, rep.get? i = rep.get? (i + 28)
  -- The 725th digit is 6
  → rep.get? 724 = some 6 := by
  sorry

end NUMINAMATH_CALUDE_seventh_twenty_ninth_725th_digit_l1207_120736


namespace NUMINAMATH_CALUDE_domino_pile_sum_theorem_l1207_120721

/-- Definition of a domino set -/
def DominoSet := { n : ℕ | n ≤ 28 }

/-- The total sum of points on all domino pieces -/
def totalSum : ℕ := 168

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if four numbers are consecutive -/
def areConsecutive (a b c d : ℕ) : Prop := b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- The main theorem to be proved -/
theorem domino_pile_sum_theorem :
  ∃ (a b c d : ℕ), 
    isPrime a ∧ isPrime b ∧ isPrime c ∧ isPrime d ∧
    areConsecutive a b c d ∧
    a + b + c + d = totalSum :=
sorry

end NUMINAMATH_CALUDE_domino_pile_sum_theorem_l1207_120721


namespace NUMINAMATH_CALUDE_sequence_sum_l1207_120709

theorem sequence_sum (A B C D E F G H : ℝ) 
  (h1 : C = 7)
  (h2 : ∀ (X Y Z : ℝ), (X = A ∧ Y = B ∧ Z = C) ∨ 
                        (X = B ∧ Y = C ∧ Z = D) ∨ 
                        (X = C ∧ Y = D ∧ Z = E) ∨ 
                        (X = D ∧ Y = E ∧ Z = F) ∨ 
                        (X = E ∧ Y = F ∧ Z = G) ∨ 
                        (X = F ∧ Y = G ∧ Z = H) → X + Y + Z = 36) : 
  A + H = 29 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l1207_120709


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1207_120755

theorem arithmetic_sequence_sum (N : ℤ) : 
  (1001 : ℤ) + 1004 + 1007 + 1010 + 1013 = 5050 - N → N = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1207_120755


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l1207_120702

theorem number_exceeds_fraction : ∃ x : ℚ, x = (3/8) * x + 30 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l1207_120702


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1207_120747

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), (4 * π * r^2 = 144 * π) → ((4/3) * π * r^3 = 288 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1207_120747


namespace NUMINAMATH_CALUDE_ceiling_square_minus_fraction_l1207_120743

theorem ceiling_square_minus_fraction : ⌈((-7/4)^2 - 1/8)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ceiling_square_minus_fraction_l1207_120743


namespace NUMINAMATH_CALUDE_train_distance_trains_ab_distance_l1207_120713

/-- The distance between two trains' starting points given their speed and meeting point -/
theorem train_distance (speed : ℝ) (distance_a : ℝ) : speed > 0 → distance_a > 0 →
  2 * distance_a = (distance_a * speed + distance_a * speed) / speed := by
  sorry

/-- The specific problem of trains A and B -/
theorem trains_ab_distance : 
  let speed : ℝ := 50
  let distance_a : ℝ := 225
  2 * distance_a = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_trains_ab_distance_l1207_120713


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1207_120711

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23)
  (h2 : L * B = 2520) :
  2 * (L + B) = 206 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1207_120711


namespace NUMINAMATH_CALUDE_pet_store_profit_percentage_l1207_120723

-- Define the types of animals
inductive AnimalType
| Gecko
| Parrot
| Tarantula

-- Define the structure for animal sales
structure AnimalSale where
  animalType : AnimalType
  quantity : Nat
  purchasePrice : Nat

-- Define the bulk discount function
def bulkDiscount (quantity : Nat) : Rat :=
  if quantity ≥ 5 then 0.1 else 0

-- Define the selling price function
def sellingPrice (animalType : AnimalType) (purchasePrice : Nat) : Nat :=
  match animalType with
  | AnimalType.Gecko => 3 * purchasePrice + 5
  | AnimalType.Parrot => 2 * purchasePrice + 10
  | AnimalType.Tarantula => 4 * purchasePrice + 15

-- Define the profit percentage calculation
def profitPercentage (sales : List AnimalSale) : Rat :=
  let totalCost := sales.foldl (fun acc sale =>
    acc + sale.quantity * sale.purchasePrice * (1 - bulkDiscount sale.quantity)) 0
  let totalRevenue := sales.foldl (fun acc sale =>
    acc + sale.quantity * sellingPrice sale.animalType sale.purchasePrice) 0
  let profit := totalRevenue - totalCost
  (profit / totalCost) * 100

-- Theorem statement
theorem pet_store_profit_percentage :
  let sales := [
    { animalType := AnimalType.Gecko, quantity := 6, purchasePrice := 100 },
    { animalType := AnimalType.Parrot, quantity := 3, purchasePrice := 200 },
    { animalType := AnimalType.Tarantula, quantity := 10, purchasePrice := 50 }
  ]
  abs (profitPercentage sales - 227.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_percentage_l1207_120723


namespace NUMINAMATH_CALUDE_remainder_of_a_83_mod_49_l1207_120756

theorem remainder_of_a_83_mod_49 : (6^83 + 8^83) % 49 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_a_83_mod_49_l1207_120756


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1207_120752

/-- Given a triangle with sides in ratio 3:4:5 and perimeter 60, prove its side lengths are 15, 20, and 25 -/
theorem triangle_side_lengths (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) 
  (h_perimeter : a + b + c = 60) : a = 15 ∧ b = 20 ∧ c = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1207_120752


namespace NUMINAMATH_CALUDE_optimal_road_network_l1207_120728

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a configuration of 4 observation stations -/
structure Configuration where
  stations : Fin 4 → Point
  valid : ∀ i, 0 ≤ (stations i).x ∧ (stations i).x ≤ 10 ∧ 0 ≤ (stations i).y ∧ (stations i).y ≤ 10

/-- Represents a network of roads -/
structure RoadNetwork where
  horizontal : List ℝ  -- y-coordinates of horizontal roads
  vertical : List ℝ    -- x-coordinates of vertical roads

/-- Checks if a road network connects all stations to both top and bottom edges -/
def connects (c : Configuration) (n : RoadNetwork) : Prop :=
  ∀ i, ∃ h v,
    h ∈ n.horizontal ∧ v ∈ n.vertical ∧
    ((c.stations i).x = v ∨ (c.stations i).y = h)

/-- Calculates the total length of a road network -/
def networkLength (n : RoadNetwork) : ℝ :=
  (n.horizontal.length * 10 : ℝ) + (n.vertical.sum : ℝ)

/-- The main theorem to be proved -/
theorem optimal_road_network :
  (∀ c : Configuration, ∃ n : RoadNetwork, connects c n ∧ networkLength n ≤ 25) ∧
  (∀ ε > 0, ∃ c : Configuration, ∀ n : RoadNetwork, connects c n → networkLength n > 25 - ε) :=
sorry

end NUMINAMATH_CALUDE_optimal_road_network_l1207_120728


namespace NUMINAMATH_CALUDE_debt_average_payment_l1207_120771

/-- Calculates the average payment for a debt paid in installments over a year. -/
theorem debt_average_payment
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (payment_increase : ℚ)
  (h1 : total_installments = 104)
  (h2 : first_payment_count = 24)
  (h3 : first_payment_amount = 520)
  (h4 : payment_increase = 95) :
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount +
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 593.08 := by
sorry


end NUMINAMATH_CALUDE_debt_average_payment_l1207_120771


namespace NUMINAMATH_CALUDE_line_through_points_l1207_120773

/-- Given a line passing through points (-1, -4) and (x, k), where the slope
    of the line is equal to k and k = 1, prove that x = 4. -/
theorem line_through_points (x : ℝ) :
  let k : ℝ := 1
  let slope : ℝ := (k - (-4)) / (x - (-1))
  slope = k → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1207_120773


namespace NUMINAMATH_CALUDE_brothers_age_in_6_years_l1207_120738

/-- The combined age of 4 brothers in a given number of years from now -/
def combined_age (years_from_now : ℕ) : ℕ :=
  sorry

theorem brothers_age_in_6_years :
  combined_age 15 = 107 → combined_age 6 = 71 :=
by sorry

end NUMINAMATH_CALUDE_brothers_age_in_6_years_l1207_120738


namespace NUMINAMATH_CALUDE_equation_solution_l1207_120716

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1207_120716


namespace NUMINAMATH_CALUDE_minimal_discs_to_separate_l1207_120737

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a disc in a plane -/
structure Disc where
  center : Point
  radius : ℝ

/-- A function that checks if a disc separates two points -/
def separates (d : Disc) (p1 p2 : Point) : Prop :=
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 < d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 > d.radius^2)) ∨
  (((p1.x - d.center.x)^2 + (p1.y - d.center.y)^2 > d.radius^2) ∧
   ((p2.x - d.center.x)^2 + (p2.y - d.center.y)^2 < d.radius^2))

/-- The main theorem stating the minimal number of discs needed -/
theorem minimal_discs_to_separate (points : Finset Point) 
  (h : points.card = 2019) :
  ∃ (discs : Finset Disc), discs.card = 1010 ∧
    ∀ p1 p2 : Point, p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      ∃ d ∈ discs, separates d p1 p2 :=
sorry

end NUMINAMATH_CALUDE_minimal_discs_to_separate_l1207_120737


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1207_120703

theorem unique_quadratic_solution (k : ℝ) (x : ℝ) :
  (∀ y : ℝ, 8 * y^2 + 36 * y + k = 0 ↔ y = x) →
  k = 40.5 ∧ x = -2.25 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1207_120703


namespace NUMINAMATH_CALUDE_prob_sum_five_twice_l1207_120769

/-- A die with 4 sides numbered 1 to 4. -/
def FourSidedDie : Finset ℕ := {1, 2, 3, 4}

/-- The set of all possible outcomes when rolling two 4-sided dice. -/
def TwoDiceOutcomes : Finset (ℕ × ℕ) :=
  FourSidedDie.product FourSidedDie

/-- The sum of two dice rolls. -/
def diceSum (roll : ℕ × ℕ) : ℕ := roll.1 + roll.2

/-- The set of all rolls that sum to 5. -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  TwoDiceOutcomes.filter (λ roll => diceSum roll = 5)

/-- The probability of rolling a sum of 5 with two 4-sided dice. -/
def probSumFive : ℚ :=
  (sumFiveOutcomes.card : ℚ) / (TwoDiceOutcomes.card : ℚ)

theorem prob_sum_five_twice :
  probSumFive * probSumFive = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_five_twice_l1207_120769


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1207_120758

/-- Proves that the ratio of Jake's weight after losing 33 pounds to his sister's weight is 2:1 -/
theorem jake_sister_weight_ratio :
  let jakes_current_weight : ℕ := 113
  let combined_weight : ℕ := 153
  let weight_loss : ℕ := 33
  let jakes_new_weight : ℕ := jakes_current_weight - weight_loss
  let sisters_weight : ℕ := combined_weight - jakes_current_weight
  (jakes_new_weight : ℚ) / (sisters_weight : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l1207_120758


namespace NUMINAMATH_CALUDE_power_of_two_equation_l1207_120761

theorem power_of_two_equation (m : ℤ) :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = m * 2^2006 →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l1207_120761


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1207_120727

/-- A line in the plane can be represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if its coordinates satisfy the line equation -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.yIntercept

theorem y_intercept_of_parallel_line_through_point 
  (l1 : Line) (p : Point) :
  l1.slope = 3 →
  parallel l1 { slope := 3, yIntercept := -2 } →
  pointOnLine p l1 →
  p.x = 5 →
  p.y = 7 →
  l1.yIntercept = -8 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l1207_120727


namespace NUMINAMATH_CALUDE_value_of_b_l1207_120762

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1207_120762


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l1207_120794

/-- Calculates the cost of each pen given the initial amount, notebook cost, and remaining amount --/
theorem pen_cost_calculation (initial_amount : ℚ) (notebook_cost : ℚ) (num_notebooks : ℕ) 
  (num_pens : ℕ) (remaining_amount : ℚ) : 
  initial_amount = 15 → 
  notebook_cost = 4 → 
  num_notebooks = 2 → 
  num_pens = 2 → 
  remaining_amount = 4 → 
  (initial_amount - remaining_amount - (num_notebooks : ℚ) * notebook_cost) / (num_pens : ℚ) = 1.5 := by
  sorry

#eval (15 : ℚ) - 4 - 2 * 4

#eval ((15 : ℚ) - 4 - 2 * 4) / 2

end NUMINAMATH_CALUDE_pen_cost_calculation_l1207_120794


namespace NUMINAMATH_CALUDE_probability_two_yellow_one_red_l1207_120760

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 5

/-- The number of orange marbles in the jar -/
def orange_marbles : ℕ := 4

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + yellow_marbles + orange_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 3

/-- Calculates the number of combinations of n items taken k at a time -/
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- The probability of choosing 2 yellow and 1 red marble from the jar -/
theorem probability_two_yellow_one_red : 
  (combination yellow_marbles 2 * combination red_marbles 1) / 
  (combination total_marbles chosen_marbles) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_yellow_one_red_l1207_120760


namespace NUMINAMATH_CALUDE_cheryl_walking_distance_l1207_120712

/-- Calculates the total distance Cheryl walked based on her journey segments -/
def total_distance_walked (
  speed1 : ℝ) (time1 : ℝ)
  (speed2 : ℝ) (time2 : ℝ)
  (speed3 : ℝ) (time3 : ℝ)
  (speed4 : ℝ) (time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that Cheryl's total walking distance is 32 miles -/
theorem cheryl_walking_distance :
  total_distance_walked 2 3 4 2 1 3 3 5 = 32 := by
  sorry

#eval total_distance_walked 2 3 4 2 1 3 3 5

end NUMINAMATH_CALUDE_cheryl_walking_distance_l1207_120712


namespace NUMINAMATH_CALUDE_statement_is_proposition_l1207_120792

def is_proposition (statement : Prop) : Prop :=
  statement ∨ ¬statement

theorem statement_is_proposition : is_proposition (20 - 5 * 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l1207_120792


namespace NUMINAMATH_CALUDE_percentage_of_women_professors_l1207_120795

-- Define the percentage of professors who are women
variable (W : ℝ)

-- Define the percentage of professors who are tenured
def T : ℝ := 70

-- Define the principle of inclusion-exclusion
axiom inclusion_exclusion : W + T - (W * T / 100) = 90

-- Define the percentage of men who are tenured
axiom men_tenured : (100 - W) * 52 / 100 = T - (W * T / 100)

-- Theorem to prove
theorem percentage_of_women_professors : ∃ ε > 0, abs (W - 79.17) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_professors_l1207_120795


namespace NUMINAMATH_CALUDE_arithmetic_sequences_problem_l1207_120783

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequences_problem 
  (a b : ℕ → ℝ) (d₁ d₂ : ℝ) 
  (ha : arithmetic_sequence a d₁)
  (hb : arithmetic_sequence b d₂)
  (A : ℕ → ℝ)
  (B : ℕ → ℝ)
  (hA : ∀ n, A n = a n + b n)
  (hB : ∀ n, B n = a n * b n)
  (hA₁ : A 1 = 1)
  (hA₂ : A 2 = 3)
  (hB_arith : arithmetic_sequence B (B 2 - B 1)) :
  (∀ n, A n = 2 * n - 1) ∧ d₁ * d₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_problem_l1207_120783


namespace NUMINAMATH_CALUDE_impossibility_of_transformation_l1207_120707

/-- Represents a four-digit number --/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_bound : a < 10
  b_bound : b < 10
  c_bound : c < 10
  d_bound : d < 10

/-- The invariant quantity M for a four-digit number --/
def invariant_M (n : FourDigitNumber) : Int :=
  (n.d + n.b) - (n.a + n.c)

/-- The allowed operations on four-digit numbers --/
inductive Operation
  | AddAdjacent (i : Fin 3)
  | SubtractAdjacent (i : Fin 3)

/-- Applying an operation to a four-digit number --/
def apply_operation (n : FourDigitNumber) (op : Operation) : Option FourDigitNumber :=
  sorry

/-- The main theorem: it's impossible to transform 1234 into 2002 --/
theorem impossibility_of_transformation :
  ∀ (ops : List Operation),
    let start := FourDigitNumber.mk 1 2 3 4 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    let target := FourDigitNumber.mk 2 0 0 2 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ∀ (result : FourDigitNumber),
      (ops.foldl (fun n op => (apply_operation n op).getD n) start = result) →
      result ≠ target :=
by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_transformation_l1207_120707


namespace NUMINAMATH_CALUDE_difference_of_fractions_l1207_120704

/-- Proves that the difference between 1/10 of 8000 and 1/20% of 8000 is equal to 796 -/
theorem difference_of_fractions : 
  (8000 / 10) - (8000 * (1 / 20) / 100) = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_l1207_120704


namespace NUMINAMATH_CALUDE_investment_calculation_l1207_120745

/-- Calculates the investment amount given share details and dividend received -/
theorem investment_calculation (face_value premium dividend_rate total_dividend : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  total_dividend = 600 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l1207_120745


namespace NUMINAMATH_CALUDE_cannot_determine_dracula_state_l1207_120775

-- Define the possible states for the Transylvanian and Count Dracula
inductive State : Type
  | Human : State
  | Undead : State
  | Alive : State
  | Dead : State

-- Define the Transylvanian's statement
def transylvanianStatement (transylvanian : State) (dracula : State) : Prop :=
  (transylvanian = State.Human) → (dracula = State.Alive)

-- Define the theorem
theorem cannot_determine_dracula_state :
  ∀ (transylvanian : State) (dracula : State),
    transylvanianStatement transylvanian dracula →
    ¬(∀ (dracula' : State), dracula' = State.Alive ∨ dracula' = State.Dead) :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_dracula_state_l1207_120775


namespace NUMINAMATH_CALUDE_no_solution_equation_l1207_120724

theorem no_solution_equation :
  ∀ (x : ℝ), x^2 + x ≠ 0 ∧ x + 1 ≠ 0 →
  (5*x + 2) / (x^2 + x) ≠ 3 / (x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1207_120724


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1207_120790

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 175/11 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1207_120790


namespace NUMINAMATH_CALUDE_paving_stone_width_l1207_120732

/-- Given a rectangular courtyard and paving stones with specified dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (num_stones : ℕ)
  (h1 : courtyard_length = 70)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : num_stones = 231)
  : ∃ (stone_width : ℝ),
    stone_width = 2 ∧
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1207_120732


namespace NUMINAMATH_CALUDE_ellipse_point_exists_l1207_120791

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (0, 1)

-- Define the line x = 4
def line_x_4 (x y : ℝ) : Prop := x = 4

-- Define the distance ratio condition
def distance_ratio (A M : ℝ × ℝ) : Prop :=
  let (ax, ay) := A
  let (mx, my) := M
  (ax^2 + (ay - 1)^2) / ((mx - 0)^2 + (my - 1)^2) = 1/9

theorem ellipse_point_exists : 
  ∃ (A : ℝ × ℝ), 
    let (m, n) := A
    ellipse m n ∧ 
    m > 0 ∧
    ellipse P.1 P.2 ∧
    ∃ (M : ℝ × ℝ), 
      line_x_4 M.1 M.2 ∧
      (n - 1) / m * (M.1 - 0) + 1 = M.2 ∧
      distance_ratio A M :=
by sorry

end NUMINAMATH_CALUDE_ellipse_point_exists_l1207_120791


namespace NUMINAMATH_CALUDE_triangle_dimensions_l1207_120765

theorem triangle_dimensions (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (eq1 : 2 * a / 3 = b) (eq2 : 2 * c = a) (eq3 : b - 2 = c) :
  a = 12 ∧ b = 8 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_dimensions_l1207_120765


namespace NUMINAMATH_CALUDE_sum_vertices_penta_hexa_prism_is_22_l1207_120740

/-- The number of vertices in a polygon -/
def vertices_in_polygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with polygonal bases -/
def vertices_in_prism (base_vertices : ℕ) : ℕ := 2 * base_vertices

/-- The sum of vertices in a pentagonal prism and a hexagonal prism -/
def sum_vertices_penta_hexa_prism : ℕ :=
  vertices_in_prism (vertices_in_polygon 5) + vertices_in_prism (vertices_in_polygon 6)

theorem sum_vertices_penta_hexa_prism_is_22 :
  sum_vertices_penta_hexa_prism = 22 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_penta_hexa_prism_is_22_l1207_120740


namespace NUMINAMATH_CALUDE_bag_production_l1207_120798

/-- Given that 15 machines produce 45 bags per minute, 
    prove that 150 machines will produce 3600 bags in 8 minutes. -/
theorem bag_production 
  (machines : ℕ) 
  (bags_per_minute : ℕ) 
  (time : ℕ) 
  (h1 : machines = 15) 
  (h2 : bags_per_minute = 45) 
  (h3 : time = 8) :
  (150 : ℕ) * bags_per_minute * time / machines = 3600 :=
sorry

end NUMINAMATH_CALUDE_bag_production_l1207_120798


namespace NUMINAMATH_CALUDE_operator_value_l1207_120774

/-- The operator definition -/
def operator (a : ℝ) (x : ℝ) : ℝ := x * (a - x)

/-- Theorem stating the value of 'a' in the operator definition -/
theorem operator_value :
  ∃ a : ℝ, (∀ p : ℝ, p = 1 → p + 1 = operator a (p + 1)) → a = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_operator_value_l1207_120774
