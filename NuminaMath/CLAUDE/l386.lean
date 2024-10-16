import Mathlib

namespace NUMINAMATH_CALUDE_height_comparison_l386_38690

theorem height_comparison (a b : ℝ) (h : a = b * 0.6) :
  (b - a) / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l386_38690


namespace NUMINAMATH_CALUDE_pennsylvania_quarter_percentage_l386_38617

theorem pennsylvania_quarter_percentage 
  (total_quarters : ℕ) 
  (state_quarter_ratio : ℚ) 
  (pennsylvania_quarters : ℕ) 
  (h1 : total_quarters = 35)
  (h2 : state_quarter_ratio = 2 / 5)
  (h3 : pennsylvania_quarters = 7) :
  (pennsylvania_quarters : ℚ) / ((state_quarter_ratio * total_quarters) : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_pennsylvania_quarter_percentage_l386_38617


namespace NUMINAMATH_CALUDE_xyz_value_l386_38682

theorem xyz_value (a b c x y z : ℂ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (sum_prod : x * y + x * z + y * z = 7)
  (sum : x + y + z = 4) :
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l386_38682


namespace NUMINAMATH_CALUDE_at_least_two_same_connections_l386_38695

-- Define the type for interns
def Intern : Type := ℕ

-- Define the knowing relation
def knows : Intern → Intern → Prop := sorry

-- The number of interns
def num_interns : ℕ := 80

-- The knowing relation is symmetric
axiom knows_symmetric : ∀ (a b : Intern), knows a b ↔ knows b a

-- Function to count how many interns a given intern knows
def num_known (i : Intern) : ℕ := sorry

-- Theorem statement
theorem at_least_two_same_connections : 
  ∃ (i j : Intern), i ≠ j ∧ num_known i = num_known j :=
sorry

end NUMINAMATH_CALUDE_at_least_two_same_connections_l386_38695


namespace NUMINAMATH_CALUDE_altons_weekly_profit_l386_38608

/-- Calculates the weekly profit for a business owner given daily earnings and weekly rent. -/
def weekly_profit (daily_earnings : ℕ) (weekly_rent : ℕ) : ℕ :=
  daily_earnings * 7 - weekly_rent

/-- Theorem stating that given specific daily earnings and weekly rent, the weekly profit is 36. -/
theorem altons_weekly_profit :
  weekly_profit 8 20 = 36 := by
  sorry

end NUMINAMATH_CALUDE_altons_weekly_profit_l386_38608


namespace NUMINAMATH_CALUDE_reflection_squared_is_identity_l386_38676

open Matrix

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : Fin 2 → ℝ) (h : v ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ := sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity 
  (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (reflection_matrix v h) ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_reflection_squared_is_identity_l386_38676


namespace NUMINAMATH_CALUDE_ornamental_bangles_pairs_l386_38662

/-- The number of bangles in a dozen -/
def bangles_per_dozen : ℕ := 12

/-- The number of dozens in a box -/
def dozens_per_box : ℕ := 2

/-- The number of boxes needed -/
def num_boxes : ℕ := 20

/-- The number of bangles in a pair -/
def bangles_per_pair : ℕ := 2

theorem ornamental_bangles_pairs :
  (num_boxes * dozens_per_box * bangles_per_dozen) / bangles_per_pair = 240 := by
  sorry

end NUMINAMATH_CALUDE_ornamental_bangles_pairs_l386_38662


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l386_38618

theorem real_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.re ((1 + i) / (1 - i)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l386_38618


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l386_38634

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l386_38634


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l386_38642

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (∀ θ : ℝ, θ = 160 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 18 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l386_38642


namespace NUMINAMATH_CALUDE_sum_of_ten_numbers_l386_38669

theorem sum_of_ten_numbers (numbers : Finset ℕ) (group_of_ten : Finset ℕ) (group_of_207 : Finset ℕ) :
  numbers = Finset.range 217 →
  numbers = group_of_ten ∪ group_of_207 →
  group_of_ten.card = 10 →
  group_of_207.card = 207 →
  group_of_ten ∩ group_of_207 = ∅ →
  (Finset.sum group_of_ten id) / 10 = (Finset.sum group_of_207 id) / 207 →
  Finset.sum group_of_ten id = 1090 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_ten_numbers_l386_38669


namespace NUMINAMATH_CALUDE_amount_second_shop_is_340_l386_38664

/-- The amount spent on books from the second shop -/
def amount_second_shop (books_first : ℕ) (amount_first : ℕ) (books_second : ℕ) (total_books : ℕ) (avg_price : ℕ) : ℕ :=
  total_books * avg_price - amount_first

/-- Theorem: The amount spent on the second shop is 340 -/
theorem amount_second_shop_is_340 :
  amount_second_shop 55 1500 60 115 16 = 340 := by
  sorry

end NUMINAMATH_CALUDE_amount_second_shop_is_340_l386_38664


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l386_38643

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l386_38643


namespace NUMINAMATH_CALUDE_negation_of_proposition_l386_38672

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l386_38672


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l386_38696

def sequence_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 = a n * a (n + 2) - 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_condition a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l386_38696


namespace NUMINAMATH_CALUDE_cycle_selling_price_l386_38644

theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 900 →
  gain_percentage = 22.22222222222222 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1100 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l386_38644


namespace NUMINAMATH_CALUDE_shaded_area_proof_l386_38652

def circle_radius : ℝ := 3

def pi_value : ℝ := 3

theorem shaded_area_proof :
  let circle_area := pi_value * circle_radius^2
  let square_side := circle_radius * Real.sqrt 2
  let square_area := square_side^2
  let total_square_area := 2 * square_area
  circle_area - total_square_area = 9 := by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l386_38652


namespace NUMINAMATH_CALUDE_z_takes_at_most_two_values_l386_38688

/-- Given two distinct real numbers x and y with absolute values not less than 2,
    prove that z = uv + (uv)⁻¹ can take at most 2 distinct values,
    where u + u⁻¹ = x and v + v⁻¹ = y. -/
theorem z_takes_at_most_two_values (x y : ℝ) (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hxy : x ≠ y) :
  ∃ (z₁ z₂ : ℝ), ∀ (u v : ℝ),
    (u + u⁻¹ = x) → (v + v⁻¹ = y) → (u * v + (u * v)⁻¹ = z₁ ∨ u * v + (u * v)⁻¹ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_z_takes_at_most_two_values_l386_38688


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l386_38607

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + a - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l386_38607


namespace NUMINAMATH_CALUDE_solution_comparison_l386_38699

theorem solution_comparison (c d p q : ℝ) (hc : c ≠ 0) (hp : p ≠ 0) :
  -d / c < -q / p ↔ q / p < d / c := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l386_38699


namespace NUMINAMATH_CALUDE_first_group_selection_is_five_l386_38686

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_count : ℕ
  group_size : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Calculates the position of a number within its group -/
def position_in_group (s : SystematicSampling) : ℕ :=
  s.selected_number - (s.selected_group - 1) * s.group_size

/-- Calculates the number selected from the first group -/
def first_group_selection (s : SystematicSampling) : ℕ :=
  position_in_group s

/-- Theorem stating the correct number selected from the first group -/
theorem first_group_selection_is_five (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_count = 20)
  (h4 : s.group_size = 8)
  (h5 : s.selected_number = 125)
  (h6 : s.selected_group = 16) : 
  first_group_selection s = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_selection_is_five_l386_38686


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l386_38625

/-- Given a polynomial p(x) = ax³ + bx² + cx + d -/
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_remainder_theorem (a b c d : ℝ) :
  (∃ q₁ : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * q₁ x + 1) →
  (∃ q₂ : ℝ → ℝ, ∀ x, p a b c d x = (x - 2) * q₂ x + 3) →
  ∃ q : ℝ → ℝ, ∀ x, p a b c d x = (x - 1) * (x - 2) * q x + (2 * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l386_38625


namespace NUMINAMATH_CALUDE_sequence_a_property_l386_38624

def sequence_a (n : ℕ+) : ℚ := 1 / ((n + 1) * (n + 2))

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := (n * (n + 1) : ℚ) / 2 * a n

theorem sequence_a_property (a : ℕ+ → ℚ) : 
  a 1 = 1/6 → 
  (∀ n : ℕ+, S n a = (n * (n + 1) : ℚ) / 2 * a n) → 
  ∀ n : ℕ+, a n = sequence_a n :=
sorry

end NUMINAMATH_CALUDE_sequence_a_property_l386_38624


namespace NUMINAMATH_CALUDE_complex_sum_equals_i_l386_38654

theorem complex_sum_equals_i : Complex.I^2 = -1 → (1 : ℂ) + Complex.I + Complex.I^2 = Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_i_l386_38654


namespace NUMINAMATH_CALUDE_power_sqrt_abs_calculation_l386_38692

theorem power_sqrt_abs_calculation : 2^0 + Real.sqrt 9 - |(-4)| = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sqrt_abs_calculation_l386_38692


namespace NUMINAMATH_CALUDE_simplify_absolute_expression_l386_38601

theorem simplify_absolute_expression : abs (-4^2 - 3 + 6) = 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_expression_l386_38601


namespace NUMINAMATH_CALUDE_equal_perimeter_ratio_l386_38693

/-- Given a square and an equilateral triangle with equal perimeters, 
    the ratio of the triangle's side length to the square's side length is 4/3 -/
theorem equal_perimeter_ratio (s t : ℝ) (hs : s > 0) (ht : t > 0) 
  (h_equal_perimeter : 4 * s = 3 * t) : t / s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_perimeter_ratio_l386_38693


namespace NUMINAMATH_CALUDE_lemonade_juice_requirement_l386_38681

/-- The amount of lemon juice required for a lemonade mixture -/
def lemon_juice_required (total_volume : ℚ) (water_parts : ℕ) (juice_parts : ℕ) : ℚ :=
  (total_volume * juice_parts) / (water_parts + juice_parts)

/-- Conversion from gallons to quarts -/
def gallons_to_quarts (gallons : ℚ) : ℚ := 4 * gallons

theorem lemonade_juice_requirement :
  let total_volume := (3 : ℚ) / 2  -- 1.5 gallons
  let water_parts := 5
  let juice_parts := 3
  lemon_juice_required (gallons_to_quarts total_volume) water_parts juice_parts = (9 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_juice_requirement_l386_38681


namespace NUMINAMATH_CALUDE_factors_of_n_l386_38680

-- Define the number in question
def n : Nat := 3^4 * 5^3 * 7^2

-- Define a function to count distinct factors
def count_factors (x : Nat) : Nat :=
  (Finset.range 5).card * (Finset.range 4).card * (Finset.range 3).card

-- Theorem statement
theorem factors_of_n : count_factors n = 60 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_n_l386_38680


namespace NUMINAMATH_CALUDE_orange_calorie_distribution_l386_38623

theorem orange_calorie_distribution 
  (num_oranges : ℕ) 
  (pieces_per_orange : ℕ) 
  (num_people : ℕ) 
  (calories_per_orange : ℕ) 
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_orange = 80) :
  (num_oranges * calories_per_orange) / (num_people) = 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_calorie_distribution_l386_38623


namespace NUMINAMATH_CALUDE_abs_plus_exp_zero_equals_three_l386_38691

theorem abs_plus_exp_zero_equals_three :
  |(-2 : ℝ)| + (3 - Real.sqrt 5) ^ (0 : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_exp_zero_equals_three_l386_38691


namespace NUMINAMATH_CALUDE_candy_bar_cost_l386_38653

/-- The cost of a candy bar, given that it costs $1 less than a chocolate that costs $3. -/
theorem candy_bar_cost : ℝ := by
  -- Define the cost of the candy bar
  let candy_cost : ℝ := 2

  -- Define the cost of the chocolate
  let chocolate_cost : ℝ := 3

  -- Assert that the chocolate costs $1 more than the candy bar
  have h1 : chocolate_cost = candy_cost + 1 := by sorry

  -- Prove that the candy bar costs $2
  have h2 : candy_cost = 2 := by sorry

  -- Return the cost of the candy bar
  exact candy_cost


end NUMINAMATH_CALUDE_candy_bar_cost_l386_38653


namespace NUMINAMATH_CALUDE_negation_of_existence_tan_equals_one_l386_38673

theorem negation_of_existence_tan_equals_one :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_tan_equals_one_l386_38673


namespace NUMINAMATH_CALUDE_square_of_five_power_plus_four_l386_38602

theorem square_of_five_power_plus_four (n : ℕ) : 
  (∃ m : ℕ, 5^n + 4 = m^2) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_square_of_five_power_plus_four_l386_38602


namespace NUMINAMATH_CALUDE_raise_upper_bound_l386_38641

/-- Represents a percentage as a real number between 0 and 1 -/
def Percentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The lower bound of the raise -/
def lower_bound : Percentage := ⟨0.05, by sorry⟩

/-- A possible raise value within the range -/
def possible_raise : Percentage := ⟨0.08, by sorry⟩

/-- The upper bound of the raise -/
def upper_bound : Percentage := ⟨0.09, by sorry⟩

theorem raise_upper_bound :
  lower_bound.val < possible_raise.val ∧
  possible_raise.val < upper_bound.val ∧
  ∀ (p : Percentage), lower_bound.val < p.val → p.val < upper_bound.val →
    p.val ≤ possible_raise.val ∨ possible_raise.val < p.val :=
by sorry

end NUMINAMATH_CALUDE_raise_upper_bound_l386_38641


namespace NUMINAMATH_CALUDE_malvina_card_sum_l386_38616

open Real MeasureTheory

theorem malvina_card_sum : ∀ x : ℝ,
  90 * π / 180 < x ∧ x < π →
  (∀ y : ℝ, 90 * π / 180 < y ∧ y < π →
    sin y > 0 ∧ cos y < 0 ∧ tan y < 0) →
  (∫ y in Set.Icc (90 * π / 180) π, sin y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_malvina_card_sum_l386_38616


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l386_38697

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h_prop : InverselyProportional x y) 
  (h_init : x 8 = 40 ∧ y 8 = 8) :
  x 10 = 32 ∧ y 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l386_38697


namespace NUMINAMATH_CALUDE_vasya_wins_l386_38635

/-- Represents a game state with a number of piles --/
structure GameState where
  piles : Nat

/-- Represents a player --/
inductive Player
  | Petya
  | Vasya

/-- The initial game state --/
def initialState : GameState :=
  { piles := 3 }

/-- The final game state --/
def finalState : GameState :=
  { piles := 119 }

/-- Calculates the number of moves required to reach the final state --/
def movesRequired (initial final : GameState) : Nat :=
  (final.piles - initial.piles) / 2

/-- Determines the player who makes the last move --/
def lastMovePlayer (moves : Nat) : Player :=
  if moves % 2 == 0 then Player.Vasya else Player.Petya

/-- The main theorem stating that Vasya (the second player) always wins --/
theorem vasya_wins :
  lastMovePlayer (movesRequired initialState finalState) = Player.Vasya :=
sorry


end NUMINAMATH_CALUDE_vasya_wins_l386_38635


namespace NUMINAMATH_CALUDE_recurrence_sequence_theorem_l386_38659

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ∀ n, (a (n + 1))^2 = (a n) * (a (n + 2)) + k

/-- Three terms form an arithmetic sequence -/
def IsArithmeticSequence (x y z : ℝ) : Prop := y - x = z - y

theorem recurrence_sequence_theorem (a : ℕ → ℝ) (k : ℝ) 
  (h : RecurrenceSequence a k) :
  (k = (a 2 - a 1)^2 → IsArithmeticSequence (a 1) (a 2) (a 3)) ∧ 
  (k = 0 → IsArithmeticSequence (a 2) (a 4) (a 5) → 
    (a 2) / (a 1) = 1 ∨ (a 2) / (a 1) = (1 + Real.sqrt 5) / 2) := by
  sorry


end NUMINAMATH_CALUDE_recurrence_sequence_theorem_l386_38659


namespace NUMINAMATH_CALUDE_schedule_count_eq_42_l386_38657

/-- The number of employees -/
def n : ℕ := 6

/-- The number of days -/
def d : ℕ := 3

/-- The number of employees working each day -/
def k : ℕ := 2

/-- Calculates the number of ways to schedule employees with given restrictions -/
def schedule_count : ℕ :=
  Nat.choose n (2 * k) * Nat.choose (n - 2 * k) k - 
  2 * (Nat.choose (n - 1) k * Nat.choose (n - 1 - k) k) +
  Nat.choose (n - 2) k * Nat.choose (n - 2 - k) k

theorem schedule_count_eq_42 : schedule_count = 42 := by
  sorry

end NUMINAMATH_CALUDE_schedule_count_eq_42_l386_38657


namespace NUMINAMATH_CALUDE_song_count_difference_l386_38626

/- Define the problem parameters -/
def total_days_in_june : ℕ := 30
def weekend_days : ℕ := 8
def vivian_daily_songs : ℕ := 10
def total_monthly_songs : ℕ := 396

/- Calculate the number of days they played songs -/
def playing_days : ℕ := total_days_in_june - weekend_days

/- Calculate Vivian's total songs for the month -/
def vivian_monthly_songs : ℕ := vivian_daily_songs * playing_days

/- Calculate Clara's total songs for the month -/
def clara_monthly_songs : ℕ := total_monthly_songs - vivian_monthly_songs

/- Calculate Clara's daily song count -/
def clara_daily_songs : ℕ := clara_monthly_songs / playing_days

/- Theorem to prove -/
theorem song_count_difference : vivian_daily_songs - clara_daily_songs = 2 := by
  sorry

end NUMINAMATH_CALUDE_song_count_difference_l386_38626


namespace NUMINAMATH_CALUDE_largest_n_for_binomial_equality_l386_38677

theorem largest_n_for_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_binomial_equality_l386_38677


namespace NUMINAMATH_CALUDE_max_books_borrowed_l386_38684

theorem max_books_borrowed (total_students : ℕ) (zero_book_students : ℕ) (one_book_students : ℕ) (two_book_students : ℕ) (avg_books : ℕ) :
  total_students = 40 →
  zero_book_students = 2 →
  one_book_students = 12 →
  two_book_students = 12 →
  avg_books = 2 →
  ∃ (max_books : ℕ),
    max_books = 5 ∧
    ∀ (student_books : ℕ),
      student_books ≤ max_books ∧
      (total_students * avg_books =
        0 * zero_book_students +
        1 * one_book_students +
        2 * two_book_students +
        (total_students - zero_book_students - one_book_students - two_book_students) * 3 +
        (max_books - 3)) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l386_38684


namespace NUMINAMATH_CALUDE_tangent_points_concyclic_l386_38647

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define a predicate for a point being outside a circle
variable (is_outside : Point → Circle → Prop)

-- Define a predicate for two circles being concentric
variable (concentric : Circle → Circle → Prop)

-- Define a predicate for a line being tangent to a circle at a point
variable (is_tangent : Point → Point → Circle → Prop)

-- Define a predicate for points being concyclic
variable (concyclic : List Point → Prop)

-- State the theorem
theorem tangent_points_concyclic 
  (O : Point) (c1 c2 : Circle) (M A B C D : Point) :
  center c1 = O →
  center c2 = O →
  concentric c1 c2 →
  is_outside M c1 →
  is_outside M c2 →
  is_tangent M A c1 →
  is_tangent M B c1 →
  is_tangent M C c2 →
  is_tangent M D c2 →
  concyclic [M, A, B, C, D] :=
sorry

end NUMINAMATH_CALUDE_tangent_points_concyclic_l386_38647


namespace NUMINAMATH_CALUDE_rational_equation_solution_l386_38640

theorem rational_equation_solution (x : ℚ) :
  x ≠ 3 →
  (x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2 →
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l386_38640


namespace NUMINAMATH_CALUDE_subtract_like_terms_l386_38612

theorem subtract_like_terms (a : ℝ) : 7 * a - 3 * a = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l386_38612


namespace NUMINAMATH_CALUDE_circles_common_chord_l386_38650

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem circles_common_chord :
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y →
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) ↔ common_chord x y :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l386_38650


namespace NUMINAMATH_CALUDE_sum_of_products_l386_38611

theorem sum_of_products : 5 * 7 + 6 * 12 + 15 * 4 + 4 * 9 = 203 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l386_38611


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l386_38661

/-- The problem of Jerry's action figures -/
theorem jerrys_action_figures 
  (total : ℕ) -- Total number of action figures after adding
  (added : ℕ) -- Number of added action figures
  (h1 : total = 10) -- Given: The total number of action figures after adding is 10
  (h2 : added = 6) -- Given: The number of added action figures is 6
  : total - added = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l386_38661


namespace NUMINAMATH_CALUDE_decimal_places_of_fraction_l386_38675

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 125) = (n : ℚ) / 10^5 ∧ 
  0 < n ∧ 
  n < 10^5 :=
by sorry

end NUMINAMATH_CALUDE_decimal_places_of_fraction_l386_38675


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l386_38615

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l386_38615


namespace NUMINAMATH_CALUDE_factorial_difference_l386_38683

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l386_38683


namespace NUMINAMATH_CALUDE_problem_1_l386_38622

theorem problem_1 : |-2| + (1/3)⁻¹ - (Real.sqrt 3 - 2021)^0 - Real.sqrt 3 * Real.tan (π/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l386_38622


namespace NUMINAMATH_CALUDE_no_integer_solutions_l386_38603

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 2*x*y + 3*y^2 - z^2 = 17) ∧ 
    (-x^2 + 4*y*z + z^2 = 28) ∧ 
    (x^2 + 2*x*y + 5*z^2 = 42) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l386_38603


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l386_38609

theorem correct_average_after_error_correction (n : ℕ) (initial_avg : ℚ) (wrong_value correct_value : ℚ) :
  n = 10 →
  initial_avg = 23 →
  wrong_value = 26 →
  correct_value = 36 →
  (n : ℚ) * initial_avg + (correct_value - wrong_value) = n * 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l386_38609


namespace NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l386_38679

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a2 : a 2 = 2) : 
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l386_38679


namespace NUMINAMATH_CALUDE_parents_in_program_l386_38637

theorem parents_in_program (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 803) (h2 : pupils = 698) :
  total_people - pupils = 105 := by
  sorry

end NUMINAMATH_CALUDE_parents_in_program_l386_38637


namespace NUMINAMATH_CALUDE_yellow_balls_unchanged_yellow_balls_count_l386_38619

/-- Represents the contents of a box with colored balls -/
structure BoxContents where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Removes one blue ball from the box -/
def removeOneBlueBall (box : BoxContents) : BoxContents :=
  { box with blue := box.blue - 1 }

/-- Theorem stating that the number of yellow balls remains unchanged after removing a blue ball -/
theorem yellow_balls_unchanged (initialBox : BoxContents) :
  (removeOneBlueBall initialBox).yellow = initialBox.yellow :=
by
  sorry

/-- The main theorem proving that the number of yellow balls remains 5 after removing a blue ball -/
theorem yellow_balls_count (initialBox : BoxContents)
  (h1 : initialBox.red = 3)
  (h2 : initialBox.blue = 2)
  (h3 : initialBox.yellow = 5) :
  (removeOneBlueBall initialBox).yellow = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_unchanged_yellow_balls_count_l386_38619


namespace NUMINAMATH_CALUDE_typhoon_tree_problem_l386_38648

theorem typhoon_tree_problem :
  ∀ (total survived died : ℕ),
    total = 14 →
    died = survived + 4 →
    survived + died = total →
    died = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_typhoon_tree_problem_l386_38648


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l386_38660

/-- Proves that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 12,
    then the principal amount is 1200. -/
theorem interest_difference_implies_principal
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Interest rate (as a decimal)
  (t : ℝ)  -- Time period in years
  (h1 : r = 0.1)  -- Interest rate is 10%
  (h2 : t = 2)    -- Time period is 2 years
  (h3 : P * (1 + r)^t - P - (P * r * t) = 12)  -- Difference between CI and SI is 12
  : P = 1200 :=
by sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l386_38660


namespace NUMINAMATH_CALUDE_c_completion_time_l386_38678

-- Define the work rates of A, B, and C
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 11

-- Theorem statement
theorem c_completion_time (h1 : condition1 A B) (h2 : condition2 A B C) :
  1 / C = 41.25 := by sorry

end NUMINAMATH_CALUDE_c_completion_time_l386_38678


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l386_38627

theorem complex_number_quadrant (z : ℂ) : z = (2 + Complex.I) / 3 → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l386_38627


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_subset_condition_l386_38630

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1: When m = 1, A ∩ B = {x | 3 ≤ x < 4}
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2: A ⊆ B if and only if m ≥ 3 or m ≤ -3
theorem subset_condition (m : ℝ) :
  A m ⊆ B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_subset_condition_l386_38630


namespace NUMINAMATH_CALUDE_population_increase_l386_38667

theorem population_increase (increase_0_to_2 increase_2_to_5 : ℝ) 
  (h1 : increase_0_to_2 = 0.1) 
  (h2 : increase_2_to_5 = 0.2) : 
  (1 + increase_0_to_2) * (1 + increase_2_to_5) - 1 = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l386_38667


namespace NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l386_38671

theorem divisible_by_seven_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, n % 10 = d → (7 ∣ n ↔ 7 ∣ d) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_last_digit_l386_38671


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l386_38636

/-- The number of seashells Joan found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Joan gave to Mike -/
def seashells_given : ℕ := 63

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 16

/-- Theorem: The total number of seashells Joan found is equal to the sum of seashells given to Mike and seashells left with Joan -/
theorem total_seashells_eq_sum : 
  total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l386_38636


namespace NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l386_38670

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_eq_quarter_l386_38670


namespace NUMINAMATH_CALUDE_milk_pouring_l386_38666

theorem milk_pouring (total : ℚ) (portion : ℚ) (result : ℚ) : 
  total = 3/7 → portion = 5/8 → result = portion * total → result = 15/56 := by
  sorry

end NUMINAMATH_CALUDE_milk_pouring_l386_38666


namespace NUMINAMATH_CALUDE_product_remainder_l386_38674

theorem product_remainder (n : ℤ) : (12 - 2*n) * (n + 5) ≡ -2*n^2 + 2*n + 5 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l386_38674


namespace NUMINAMATH_CALUDE_second_largest_is_five_l386_38631

def number_set : Finset ℕ := {5, 8, 4, 3, 2}

theorem second_largest_is_five :
  ∃ (x : ℕ), x ∈ number_set ∧ 
  (∀ y ∈ number_set, y ≠ x → y ≤ x) ∧
  (∃ z ∈ number_set, z > x) ∧
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_is_five_l386_38631


namespace NUMINAMATH_CALUDE_xiaoxi_has_largest_result_l386_38621

def start_number : ℕ := 8

def laura_result (n : ℕ) : ℕ := ((n - 2) * 3) + 3

def navin_result (n : ℕ) : ℕ := (n * 3 - 2) + 3

def xiaoxi_result (n : ℕ) : ℕ := ((n - 2) + 3) * 3

theorem xiaoxi_has_largest_result :
  xiaoxi_result start_number > laura_result start_number ∧
  xiaoxi_result start_number > navin_result start_number :=
by sorry

end NUMINAMATH_CALUDE_xiaoxi_has_largest_result_l386_38621


namespace NUMINAMATH_CALUDE_distance_product_on_curve_l386_38610

/-- The curve C defined by xy = 2 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 2}

/-- The theorem stating that the product of distances from any point on C to the axes is 2 -/
theorem distance_product_on_curve (p : ℝ × ℝ) (h : p ∈ C) :
  |p.1| * |p.2| = 2 := by
  sorry


end NUMINAMATH_CALUDE_distance_product_on_curve_l386_38610


namespace NUMINAMATH_CALUDE_book_exchange_ways_l386_38646

theorem book_exchange_ways (n₁ n₂ k : ℕ) (h₁ : n₁ = 6) (h₂ : n₂ = 8) (h₃ : k = 3) : 
  (n₁.choose k) * (n₂.choose k) = 1120 := by
  sorry

end NUMINAMATH_CALUDE_book_exchange_ways_l386_38646


namespace NUMINAMATH_CALUDE_min_value_on_line_l386_38638

theorem min_value_on_line (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l386_38638


namespace NUMINAMATH_CALUDE_mistaken_divisor_l386_38663

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 32 * correct_divisor →
  dividend = 56 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l386_38663


namespace NUMINAMATH_CALUDE_max_value_of_b_l386_38689

theorem max_value_of_b (a b c : ℝ) : 
  (∃ q : ℝ, a = b / q ∧ c = b * q) →  -- geometric sequence condition
  (b + 2 = (a + 6 + c + 1) / 2) →     -- arithmetic sequence condition
  b ≤ 3/4 :=                          -- maximum value of b
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l386_38689


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l386_38639

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of F_{F₁₅} is 5 -/
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l386_38639


namespace NUMINAMATH_CALUDE_pants_price_calculation_l386_38600

/-- The price of a T-shirt in dollars -/
def tshirt_price : ℚ := 5

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The price of a refurbished T-shirt in dollars -/
def refurbished_tshirt_price : ℚ := tshirt_price / 2

/-- The total income from the sales in dollars -/
def total_income : ℚ := 53

/-- The number of T-shirts sold -/
def tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

theorem pants_price_calculation :
  pants_price = total_income - 
    (tshirts_sold * tshirt_price + 
     skirts_sold * skirt_price + 
     refurbished_tshirts_sold * refurbished_tshirt_price) :=
by sorry

end NUMINAMATH_CALUDE_pants_price_calculation_l386_38600


namespace NUMINAMATH_CALUDE_turtleneck_profit_l386_38633

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) : 
  let initial_price := C * 1.20
  let new_year_price := initial_price * 1.25
  let february_price := new_year_price * 0.94
  let profit := february_price - C
  profit = C * 0.41 := by sorry

end NUMINAMATH_CALUDE_turtleneck_profit_l386_38633


namespace NUMINAMATH_CALUDE_school_girls_count_l386_38649

theorem school_girls_count (boys : ℕ) (girls : ℝ) : 
  boys = 387 →
  girls = boys + 0.54 * boys →
  ⌊girls + 0.5⌋ = 596 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l386_38649


namespace NUMINAMATH_CALUDE_det_E_l386_38658

/-- A 2×2 matrix representing a dilation centered at the origin with scale factor 9 -/
def E : Matrix (Fin 2) (Fin 2) ℝ := !![9, 0; 0, 9]

/-- The determinant of E is 81 -/
theorem det_E : Matrix.det E = 81 := by sorry

end NUMINAMATH_CALUDE_det_E_l386_38658


namespace NUMINAMATH_CALUDE_at_least_one_subgraph_not_planar_l386_38698

/-- A complete graph with 11 vertices where each edge is colored either red or blue. -/
def CompleteGraph11 : Type := Unit

/-- The red subgraph of the complete graph. -/
def RedSubgraph (G : CompleteGraph11) : Type := Unit

/-- The blue subgraph of the complete graph. -/
def BlueSubgraph (G : CompleteGraph11) : Type := Unit

/-- Predicate to check if a graph is planar. -/
def IsPlanar (G : Type) : Prop := sorry

/-- Theorem stating that at least one of the monochromatic subgraphs is not planar. -/
theorem at_least_one_subgraph_not_planar (G : CompleteGraph11) : 
  ¬(IsPlanar (RedSubgraph G) ∧ IsPlanar (BlueSubgraph G)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_subgraph_not_planar_l386_38698


namespace NUMINAMATH_CALUDE_cindy_wins_prob_l386_38604

-- Define the probability of tossing a five
def prob_five : ℚ := 1/6

-- Define the probability of not tossing a five
def prob_not_five : ℚ := 1 - prob_five

-- Define the probability of Cindy winning in the first cycle
def prob_cindy_first_cycle : ℚ := prob_not_five * prob_five

-- Define the probability of the game continuing after one full cycle
def prob_continue : ℚ := prob_not_five^3

-- Theorem statement
theorem cindy_wins_prob : 
  let a : ℚ := prob_cindy_first_cycle
  let r : ℚ := prob_continue
  (a / (1 - r)) = 30/91 := by
  sorry

end NUMINAMATH_CALUDE_cindy_wins_prob_l386_38604


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l386_38694

/-- A line in the 2D plane represented by its slope-intercept form -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Defines symmetry of two lines with respect to the y-axis -/
def symmetricAboutYAxis (l₁ l₂ : Line) : Prop :=
  ∀ x y, l₁.contains x y ↔ l₂.contains (-x) y

/-- The main theorem -/
theorem symmetric_line_equation (l₁ l₂ : Line) :
  l₁.slope = 3 →
  l₁.contains 1 2 →
  symmetricAboutYAxis l₁ l₂ →
  ∀ x y, l₂.contains x y ↔ 3 * x + y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l386_38694


namespace NUMINAMATH_CALUDE_cuboid_missing_edge_l386_38651

/-- Proves that for a cuboid with given dimensions and volume, the missing edge length is 5 cm -/
theorem cuboid_missing_edge (edge1 edge2 x : ℝ) (volume : ℝ) : 
  edge1 = 2 → edge2 = 8 → volume = 80 → edge1 * x * edge2 = volume → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_missing_edge_l386_38651


namespace NUMINAMATH_CALUDE_car_rental_maximum_profit_l386_38620

/-- Represents the car rental company's profit function --/
def profit_function (n : ℝ) : ℝ :=
  -50 * (n^2 - 100*n + 630000)

/-- Represents the rental fee calculation --/
def rental_fee (n : ℝ) : ℝ :=
  3000 + 50 * n

theorem car_rental_maximum_profit :
  ∃ (n : ℝ),
    profit_function n = 307050 ∧
    rental_fee n = 4050 ∧
    ∀ (m : ℝ), profit_function m ≤ profit_function n :=
by
  sorry

end NUMINAMATH_CALUDE_car_rental_maximum_profit_l386_38620


namespace NUMINAMATH_CALUDE_jim_shopping_cost_l386_38656

theorem jim_shopping_cost (lamp_cost : ℕ) (bulb_cost : ℕ) (num_lamps : ℕ) (num_bulbs : ℕ) 
  (h1 : lamp_cost = 7)
  (h2 : bulb_cost = lamp_cost - 4)
  (h3 : num_lamps = 2)
  (h4 : num_bulbs = 6) :
  num_lamps * lamp_cost + num_bulbs * bulb_cost = 32 := by
sorry

end NUMINAMATH_CALUDE_jim_shopping_cost_l386_38656


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l386_38655

/-- Calculates the total number of heartbeats for an athlete jogging a given distance -/
def total_heartbeats (heart_rate : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * distance * pace

/-- Proves that an athlete jogging 15 miles at 8 minutes per mile with a heart rate of 120 bpm will have 14400 total heartbeats -/
theorem athlete_heartbeats :
  total_heartbeats 120 15 8 = 14400 := by
  sorry

#eval total_heartbeats 120 15 8

end NUMINAMATH_CALUDE_athlete_heartbeats_l386_38655


namespace NUMINAMATH_CALUDE_multiple_condition_l386_38613

theorem multiple_condition (n : ℕ+) : 
  (∃ k : ℕ, 3^n.val + 5^n.val = k * (3^(n.val - 1) + 5^(n.val - 1))) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_condition_l386_38613


namespace NUMINAMATH_CALUDE_probability_at_least_one_man_l386_38632

theorem probability_at_least_one_man (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 10 →
  women = 5 →
  selected = 5 →
  (1 : ℚ) - (Nat.choose women selected : ℚ) / (Nat.choose total selected : ℚ) = 3002 / 3003 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_man_l386_38632


namespace NUMINAMATH_CALUDE_b_share_is_160_l386_38668

/-- Given a total cost and individual usage times, calculates an individual's share of the cost -/
def calculate_share (total_cost : ℚ) (individual_time : ℚ) (time_a : ℚ) (time_b : ℚ) (time_c : ℚ) : ℚ :=
  total_cost * individual_time / (time_a + time_b + time_c)

/-- Proves that B's share of the car hire cost is 160 Rs -/
theorem b_share_is_160 (total_cost : ℚ) (time_a time_b time_c : ℚ)
    (h1 : total_cost = 520)
    (h2 : time_a = 7)
    (h3 : time_b = 8)
    (h4 : time_c = 11) :
  calculate_share total_cost time_b time_a time_b time_c = 160 := by
  sorry

#eval calculate_share 520 8 7 8 11

end NUMINAMATH_CALUDE_b_share_is_160_l386_38668


namespace NUMINAMATH_CALUDE_max_product_is_48_l386_38665

def max_product (x y z : ℕ+) : Prop :=
  (x : ℕ) + y + z = 12 ∧
  x ≤ y ∧ y ≤ z ∧
  z ≤ 3 * x ∧
  x * y * z ≤ 48

theorem max_product_is_48 :
  ∀ x y z : ℕ+, max_product x y z → x * y * z = 48 :=
sorry

end NUMINAMATH_CALUDE_max_product_is_48_l386_38665


namespace NUMINAMATH_CALUDE_solution_implies_m_minus_n_equals_negative_three_l386_38605

theorem solution_implies_m_minus_n_equals_negative_three :
  ∀ m n : ℤ,
  (3 * (-2) + 2 * 1 = m) →
  (n * (-2) - 1 = 1) →
  m - n = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_minus_n_equals_negative_three_l386_38605


namespace NUMINAMATH_CALUDE_student_village_arrangements_l386_38628

theorem student_village_arrangements :
  let num_students : ℕ := 3
  let num_villages : ℕ := 2
  let arrangements : ℕ := (num_students.choose (num_students - 1)) * (num_villages.factorial)
  arrangements = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_village_arrangements_l386_38628


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l386_38614

-- Define the function f(x) = x^3 - 3x - 3
def f (x : ℝ) : ℝ := x^3 - 3*x - 3

-- State the theorem
theorem f_has_root_in_interval : 
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l386_38614


namespace NUMINAMATH_CALUDE_overlap_percentage_l386_38645

theorem overlap_percentage (square_side : ℝ) (rect_width rect_length : ℝ) 
  (overlap_rect_width overlap_rect_length : ℝ) :
  square_side = 12 →
  rect_width = 9 →
  rect_length = 12 →
  overlap_rect_width = 12 →
  overlap_rect_length = 18 →
  (((square_side + rect_width - overlap_rect_length) * rect_width) / 
   (overlap_rect_width * overlap_rect_length)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_percentage_l386_38645


namespace NUMINAMATH_CALUDE_inequalities_for_negative_fractions_l386_38685

theorem inequalities_for_negative_fractions (a b : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) : 
  (1 / a > 1 / b) ∧ 
  (a^2 + b^2 > 2*a*b) ∧ 
  (a + 1/a > b + 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_fractions_l386_38685


namespace NUMINAMATH_CALUDE_smallest_digit_change_l386_38606

def original_sum : ℕ := 753 + 946 + 821
def incorrect_result : ℕ := 2420
def correct_result : ℕ := 2520

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ := 
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∃ (d : ℕ), d < 10 ∧ 
    change_digit 821 2 (d + 1) + 753 + 946 = correct_result ∧
    ∀ (n : ℕ) (p : ℕ) (digit : ℕ), 
      digit < d → 
      change_digit 753 p digit + 946 + 821 ≠ correct_result ∧
      753 + change_digit 946 p digit + 821 ≠ correct_result ∧
      753 + 946 + change_digit 821 p digit ≠ correct_result :=
sorry

#check smallest_digit_change

end NUMINAMATH_CALUDE_smallest_digit_change_l386_38606


namespace NUMINAMATH_CALUDE_increase_amount_is_four_l386_38687

/-- Represents a set of numbers with a known size and average -/
structure NumberSet where
  size : ℕ
  average : ℝ

/-- Calculates the sum of elements in a NumberSet -/
def NumberSet.sum (s : NumberSet) : ℝ := s.size * s.average

/-- The original set of numbers -/
def original_set : NumberSet := { size := 10, average := 6.2 }

/-- The new set of numbers after increasing one element -/
def new_set : NumberSet := { size := 10, average := 6.6 }

/-- The theorem to be proved -/
theorem increase_amount_is_four :
  new_set.sum - original_set.sum = 4 := by sorry

end NUMINAMATH_CALUDE_increase_amount_is_four_l386_38687


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_trajectory_l386_38629

/-- The equation of the trajectory of the midpoint of a chord passing through the focus of the parabola y² = 4x is y² = 2x - 2 -/
theorem parabola_chord_midpoint_trajectory (x y : ℝ) :
  (∀ x₀ y₀, y₀^2 = 4*x₀ → -- Parabola equation
   ∃ a b : ℝ, (y - y₀)^2 = 4*(a^2 + b^2)*(x - x₀) ∧ -- Chord passing through focus
   x = (x₀ + a)/2 ∧ y = (y₀ + b)/2) -- Midpoint of chord
  → y^2 = 2*x - 2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_trajectory_l386_38629
