import Mathlib

namespace NUMINAMATH_CALUDE_negative_expression_l3891_389160

theorem negative_expression : 
  (|(-1)| - |(-7)| < 0) ∧ 
  (|(-7)| + |(-1)| ≥ 0) ∧ 
  (|(-7)| - (-1) ≥ 0) ∧ 
  (|(-1)| - (-7) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_expression_l3891_389160


namespace NUMINAMATH_CALUDE_group_size_after_new_member_l3891_389180

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 = n * 14) →  -- Initial average age is 14
  (n * 14 + 32 = (n + 1) * 15) →  -- New average age is 15 after adding a 32-year-old
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_group_size_after_new_member_l3891_389180


namespace NUMINAMATH_CALUDE_pages_torn_off_l3891_389111

def total_pages : ℕ := 100
def sum_remaining_pages : ℕ := 4949

theorem pages_torn_off : 
  ∃ (torn_pages : Finset ℕ), 
    torn_pages.card = 3 ∧ 
    (Finset.range total_pages.succ).sum id - torn_pages.sum id = sum_remaining_pages :=
sorry

end NUMINAMATH_CALUDE_pages_torn_off_l3891_389111


namespace NUMINAMATH_CALUDE_power_seven_150_mod_12_l3891_389168

theorem power_seven_150_mod_12 : 7^150 ≡ 1 [ZMOD 12] := by sorry

end NUMINAMATH_CALUDE_power_seven_150_mod_12_l3891_389168


namespace NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l3891_389129

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ a / b ∧ a / b ≤ Real.sqrt (n + 1)) ∧
  (∃ f : ℕ → ℕ+, StrictMono f ∧ ∀ n : ℕ, ¬∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt (f n) ∧ Real.sqrt (f n) ≤ a / b ∧ a / b ≤ Real.sqrt (f n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l3891_389129


namespace NUMINAMATH_CALUDE_divisible_by_five_l3891_389170

theorem divisible_by_five (B : Nat) : B < 10 → (647 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3891_389170


namespace NUMINAMATH_CALUDE_midpoint_of_translated_segment_l3891_389108

/-- Given a segment s₁ with endpoints (2, -3) and (10, 7), and segment s₂ obtained by
    translating s₁ by 3 units to the left and 2 units down, prove that the midpoint
    of s₂ is (3, 0). -/
theorem midpoint_of_translated_segment :
  let s₁_start : ℝ × ℝ := (2, -3)
  let s₁_end : ℝ × ℝ := (10, 7)
  let translation : ℝ × ℝ := (-3, -2)
  let s₂_start : ℝ × ℝ := (s₁_start.1 + translation.1, s₁_start.2 + translation.2)
  let s₂_end : ℝ × ℝ := (s₁_end.1 + translation.1, s₁_end.2 + translation.2)
  let s₂_midpoint : ℝ × ℝ := ((s₂_start.1 + s₂_end.1) / 2, (s₂_start.2 + s₂_end.2) / 2)
  s₂_midpoint = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_translated_segment_l3891_389108


namespace NUMINAMATH_CALUDE_whack_a_mole_tickets_value_l3891_389163

/-- The number of tickets Ned won playing 'skee ball' -/
def skee_ball_tickets : ℕ := 19

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 9

/-- The number of candies Ned could buy -/
def candies_bought : ℕ := 5

/-- The number of tickets Ned won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := candy_cost * candies_bought - skee_ball_tickets

theorem whack_a_mole_tickets_value : whack_a_mole_tickets = 26 := by
  sorry

end NUMINAMATH_CALUDE_whack_a_mole_tickets_value_l3891_389163


namespace NUMINAMATH_CALUDE_b_age_is_39_l3891_389186

/-- Represents a person's age --/
structure Age where
  value : ℕ

/-- Represents the ages of three people A, B, and C --/
structure AgeGroup where
  a : Age
  b : Age
  c : Age

/-- Checks if the given ages satisfy the conditions of the problem --/
def satisfiesConditions (ages : AgeGroup) : Prop :=
  (ages.a.value + 10 = 2 * (ages.b.value - 10)) ∧
  (ages.a.value = ages.b.value + 9) ∧
  (ages.c.value = ages.a.value + 4)

/-- Theorem stating that if the conditions are satisfied, B's age is 39 --/
theorem b_age_is_39 (ages : AgeGroup) :
  satisfiesConditions ages → ages.b.value = 39 := by
  sorry

#check b_age_is_39

end NUMINAMATH_CALUDE_b_age_is_39_l3891_389186


namespace NUMINAMATH_CALUDE_congruence_problem_l3891_389162

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) :
  ∃! n : ℤ, 200 ≤ n ∧ n ≤ 251 ∧ a - b ≡ n [ZMOD 60] ∧ n = 248 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3891_389162


namespace NUMINAMATH_CALUDE_max_school_leaders_l3891_389183

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the length of a principal's term in years -/
def principal_term : ℕ := 3

/-- Represents the length of an assistant principal's term in years -/
def assistant_principal_term : ℕ := 2

/-- Calculates the maximum number of individuals serving in a role given the period and term length -/
def max_individuals (period : ℕ) (term : ℕ) : ℕ :=
  (period + term - 1) / term

/-- Theorem stating the maximum number of principals and assistant principals over the given period -/
theorem max_school_leaders :
  max_individuals period principal_term + max_individuals period assistant_principal_term = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_school_leaders_l3891_389183


namespace NUMINAMATH_CALUDE_expected_disease_cases_l3891_389164

theorem expected_disease_cases (total_sample : ℕ) (disease_proportion : ℚ) :
  total_sample = 300 →
  disease_proportion = 1 / 4 →
  (total_sample : ℚ) * disease_proportion = 75 := by
  sorry

end NUMINAMATH_CALUDE_expected_disease_cases_l3891_389164


namespace NUMINAMATH_CALUDE_angle_from_point_l3891_389190

/-- Given a point A with coordinates (sin 23°, -cos 23°) on the terminal side of angle α,
    where 0° < α < 360°, prove that α = 293°. -/
theorem angle_from_point (α : Real) : 
  0 < α ∧ α < 360 ∧ 
  (∃ (A : ℝ × ℝ), A.1 = Real.sin (23 * π / 180) ∧ A.2 = -Real.cos (23 * π / 180) ∧ 
    A.1 = Real.cos α ∧ A.2 = Real.sin α) →
  α = 293 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l3891_389190


namespace NUMINAMATH_CALUDE_calculation_proof_l3891_389161

theorem calculation_proof : 
  |(-1/2 : ℝ)| + (2023 - Real.pi)^0 - (27 : ℝ)^(1/3) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3891_389161


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_iff_l3891_389127

/-- Definition of a fixed point for a function f -/
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

/-- The quadratic function f(x) = ax² + (b+1)x + b - 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

theorem fixed_points_for_specific_values (a b : ℝ) (h1 : a = 1) (h2 : b = 5) :
  is_fixed_point (f a b) (-4) ∧ is_fixed_point (f a b) (-1) :=
sorry

theorem two_distinct_fixed_points_iff (a : ℝ) (h : a ≠ 0) :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) ↔
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_two_distinct_fixed_points_iff_l3891_389127


namespace NUMINAMATH_CALUDE_monotonicity_condition_l3891_389139

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cubic function f(x) = x³ - 2x² - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - m*x + 1

theorem monotonicity_condition (m : ℝ) :
  (m > 4/3 → MonotonicallyIncreasing (f m)) ∧
  (∃ m : ℝ, m ≤ 4/3 ∧ MonotonicallyIncreasing (f m)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l3891_389139


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3891_389131

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (3 / a + 2 / b) ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3891_389131


namespace NUMINAMATH_CALUDE_vacation_cost_l3891_389154

theorem vacation_cost (cost : ℝ) : 
  (cost / 3 - cost / 4 = 60) → cost = 720 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l3891_389154


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3891_389133

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  numTiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration --/
def addTiles (config : TileConfiguration) (newTiles : ℕ) : TileConfiguration :=
  { numTiles := config.numTiles + newTiles
  , perimeter := config.perimeter + 2 * newTiles }

theorem perimeter_after_adding_tiles 
  (initialConfig : TileConfiguration) 
  (tilesAdded : ℕ) :
  initialConfig.numTiles = 9 →
  initialConfig.perimeter = 16 →
  tilesAdded = 3 →
  (addTiles initialConfig tilesAdded).perimeter = 22 := by
  sorry

#check perimeter_after_adding_tiles

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3891_389133


namespace NUMINAMATH_CALUDE_expression_simplification_l3891_389100

theorem expression_simplification (x : ℝ) : (36 + 12*x)^2 - (12^2*x^2 + 36^2) = 864*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3891_389100


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l3891_389146

theorem sqrt_sum_difference : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l3891_389146


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3891_389148

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

-- Theorem statement
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ 5) ∧
  (f 1 = 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3891_389148


namespace NUMINAMATH_CALUDE_small_tile_position_l3891_389152

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents the state of the grid -/
structure GridState where
  smallTilePos : Position
  largeTiles : Finset (Position × Position × Position)

/-- Checks if a position is at the center or adjacent to the border -/
def isCenterOrBorder (pos : Position) : Prop :=
  pos.row = 0 ∨ pos.row = 3 ∨ pos.row = 6 ∨
  pos.col = 0 ∨ pos.col = 3 ∨ pos.col = 6

/-- Main theorem -/
theorem small_tile_position (grid : GridState) 
  (h1 : grid.largeTiles.card = 16) : 
  isCenterOrBorder grid.smallTilePos :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l3891_389152


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3891_389123

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
  Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ (5 : ℝ) / 2 ∧
  ((a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
   Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) = (5 : ℝ) / 2 ↔ 
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3891_389123


namespace NUMINAMATH_CALUDE_triangle_problem_l3891_389130

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A) 
  (h6 : b = 3) 
  (h7 : c = 2) : 
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3891_389130


namespace NUMINAMATH_CALUDE_sum_of_ages_is_105_l3891_389158

/-- Calculates the sum of Riza's and her son's ages given their initial conditions -/
def sumOfAges (rizaAgeAtBirth : ℕ) (sonCurrentAge : ℕ) : ℕ :=
  rizaAgeAtBirth + 2 * sonCurrentAge

/-- Proves that the sum of Riza's and her son's ages is 105 years -/
theorem sum_of_ages_is_105 :
  sumOfAges 25 40 = 105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_is_105_l3891_389158


namespace NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3891_389194

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_6_factorial_squared_l3891_389194


namespace NUMINAMATH_CALUDE_count_m_with_integer_roots_l3891_389125

def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  x^2 - m*x + m + 2006 = 0

def has_integer_roots (m : ℤ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ quadratic_equation m a ∧ quadratic_equation m b

theorem count_m_with_integer_roots :
  ∃! (S : Finset ℤ), (∀ m : ℤ, m ∈ S ↔ has_integer_roots m) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_m_with_integer_roots_l3891_389125


namespace NUMINAMATH_CALUDE_all_rules_correct_l3891_389198

/-- Custom addition operation -/
def oplus (a b : ℝ) : ℝ := a + b + 1

/-- Custom subtraction operation -/
def ominus (a b : ℝ) : ℝ := a - b - 1

/-- Theorem stating the correctness of all three rules -/
theorem all_rules_correct (a b c : ℝ) : 
  (oplus a b = oplus b a) ∧ 
  (oplus a (oplus b c) = oplus (oplus a b) c) ∧ 
  (ominus a (oplus b c) = ominus (ominus a b) c) :=
sorry

end NUMINAMATH_CALUDE_all_rules_correct_l3891_389198


namespace NUMINAMATH_CALUDE_devin_teaching_years_l3891_389103

/-- Represents the number of years Devin taught each subject -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  geometry : ℕ
  discrete_math : ℕ

/-- Calculates the total number of years taught -/
def total_years (years : TeachingYears) : ℕ :=
  years.calculus + years.algebra + years.statistics + years.geometry + years.discrete_math

/-- Theorem stating the total number of years Devin taught -/
theorem devin_teaching_years :
  ∃ (years : TeachingYears),
    years.calculus = 4 ∧
    years.algebra = 2 * years.calculus ∧
    years.statistics = 5 * years.algebra ∧
    years.geometry = 3 * years.statistics ∧
    years.discrete_math = years.geometry / 2 ∧
    total_years years = 232 :=
by sorry

end NUMINAMATH_CALUDE_devin_teaching_years_l3891_389103


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3891_389119

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 + 1 > 2*x ∧ x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → x^2 + 1 > 2*x) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3891_389119


namespace NUMINAMATH_CALUDE_valentines_day_treats_cost_l3891_389171

/-- The cost of Valentine's Day treats for two dogs -/
def total_cost (heart_biscuit_cost puppy_boots_cost : ℕ) : ℕ :=
  let dog_a_cost := 5 * heart_biscuit_cost + puppy_boots_cost
  let dog_b_cost := 7 * heart_biscuit_cost + 2 * puppy_boots_cost
  dog_a_cost + dog_b_cost

/-- Theorem stating the total cost of Valentine's Day treats for two dogs -/
theorem valentines_day_treats_cost :
  total_cost 2 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_treats_cost_l3891_389171


namespace NUMINAMATH_CALUDE_water_volume_is_16_l3891_389176

/-- Represents a cubical water tank -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_ratio : ℝ

/-- Calculates the volume of water in a cubical tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.water_level * tank.side_length * tank.side_length

/-- Theorem: The volume of water in the specified cubical tank is 16 cubic feet -/
theorem water_volume_is_16 (tank : CubicalTank) 
  (h1 : tank.water_level = 1)
  (h2 : tank.capacity_ratio = 0.25)
  (h3 : tank.water_level = tank.capacity_ratio * tank.side_length) :
  water_volume tank = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_is_16_l3891_389176


namespace NUMINAMATH_CALUDE_candy_bar_weight_reduction_l3891_389115

theorem candy_bar_weight_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (new_weight : ℝ) 
  (h1 : original_weight > 0) 
  (h2 : original_price > 0) 
  (h3 : new_weight > 0) 
  (h4 : new_weight < original_weight) 
  (h5 : original_price / new_weight = (1 + 1/3) * (original_price / original_weight)) :
  (original_weight - new_weight) / original_weight = 1/4 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_weight_reduction_l3891_389115


namespace NUMINAMATH_CALUDE_multiplication_of_squares_l3891_389121

theorem multiplication_of_squares (a b : ℝ) : 2 * a^2 * 3 * b^2 = 6 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_squares_l3891_389121


namespace NUMINAMATH_CALUDE_flour_for_two_loaves_l3891_389156

/-- The amount of flour needed for one loaf of bread in cups -/
def flour_per_loaf : ℝ := 2.5

/-- The number of loaves of bread to be baked -/
def num_loaves : ℕ := 2

/-- Theorem: The amount of flour needed for two loaves of bread is 5 cups -/
theorem flour_for_two_loaves : flour_per_loaf * num_loaves = 5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_two_loaves_l3891_389156


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l3891_389189

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l3891_389189


namespace NUMINAMATH_CALUDE_bailey_rawhide_bones_l3891_389107

theorem bailey_rawhide_bones (dog_treats chew_toys credit_cards items_per_charge : ℕ) 
  (h1 : dog_treats = 8)
  (h2 : chew_toys = 2)
  (h3 : credit_cards = 4)
  (h4 : items_per_charge = 5) :
  credit_cards * items_per_charge - (dog_treats + chew_toys) = 10 := by
  sorry

end NUMINAMATH_CALUDE_bailey_rawhide_bones_l3891_389107


namespace NUMINAMATH_CALUDE_complex_number_theorem_l3891_389184

theorem complex_number_theorem (m : ℝ) : 
  let z : ℂ := m + (m^2 - 1) * Complex.I
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l3891_389184


namespace NUMINAMATH_CALUDE_todd_initial_gum_l3891_389182

/-- Todd's initial amount of gum -/
def initial_gum : ℕ := sorry

/-- Amount of gum Steve gave Todd -/
def steve_gum : ℕ := 16

/-- Todd's final amount of gum -/
def final_gum : ℕ := 54

/-- Theorem stating that Todd's initial amount of gum is 38 pieces -/
theorem todd_initial_gum : initial_gum = 54 - 16 := by sorry

end NUMINAMATH_CALUDE_todd_initial_gum_l3891_389182


namespace NUMINAMATH_CALUDE_cube_sum_divided_l3891_389105

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 219 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divided_l3891_389105


namespace NUMINAMATH_CALUDE_base_b_is_five_l3891_389124

/-- The base in which 200 (base 10) is represented with exactly 4 digits -/
def base_b : ℕ := 5

/-- 200 in base 10 -/
def number : ℕ := 200

theorem base_b_is_five :
  ∃! b : ℕ, b > 1 ∧ 
  (b ^ 3 ≤ number) ∧ 
  (number < b ^ 4) ∧
  (∀ d : ℕ, d < b → number ≥ d * b ^ 3) :=
sorry

end NUMINAMATH_CALUDE_base_b_is_five_l3891_389124


namespace NUMINAMATH_CALUDE_point_movement_specific_point_movement_l3891_389138

/-- Given a point P in a 2D Cartesian coordinate system, moving it right and up results in a new point P'. -/
theorem point_movement (x y dx dy : ℝ) :
  let P : ℝ × ℝ := (x, y)
  let P' : ℝ × ℝ := (x + dx, y + dy)
  P' = (x + dx, y + dy) :=
by sorry

/-- The specific case of moving point P(2, -3) right by 2 units and up by 4 units results in P'(4, 1). -/
theorem specific_point_movement :
  let P : ℝ × ℝ := (2, -3)
  let P' : ℝ × ℝ := (2 + 2, -3 + 4)
  P' = (4, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_specific_point_movement_l3891_389138


namespace NUMINAMATH_CALUDE_intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l3891_389159

/-- The line equation y = ax + 1 -/
def line_equation (a x y : ℝ) : Prop := y = a * x + 1

/-- The hyperbola equation 3x^2 - y^2 = 1 -/
def hyperbola_equation (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

/-- Two points A(x₁, y₁) and B(x₂, y₂) are the intersection of the line and hyperbola -/
def intersection_points (a x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation a x₁ y₁ ∧ hyperbola_equation x₁ y₁ ∧
  line_equation a x₂ y₂ ∧ hyperbola_equation x₂ y₂

/-- The circle with diameter AB passes through the origin -/
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- Two points are symmetric about the line y = (1/2)x -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ + y₂) / 2 = (1 / 2) * ((x₁ + x₂) / 2) ∧
  (y₁ - y₂) / (x₁ - x₂) = -2

theorem intersection_circle_origin_implies_a_plusminus_one :
  ∀ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ →
  circle_through_origin x₁ y₁ x₂ y₂ →
  a = 1 ∨ a = -1 :=
sorry

theorem no_symmetric_intersection :
  ¬ ∃ (a x₁ y₁ x₂ y₂ : ℝ),
  intersection_points a x₁ y₁ x₂ y₂ ∧
  symmetric_about_line x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_intersection_circle_origin_implies_a_plusminus_one_no_symmetric_intersection_l3891_389159


namespace NUMINAMATH_CALUDE_range_of_a_l3891_389174

-- Define the conditions
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the set A (solution set for p)
def A : Set ℝ := {x | p x}

-- Define the set B (solution set for q)
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (A ⊆ B a) ∧ (A ≠ B a)) →
  (∃ a_min a_max : ℝ, a_min = 0 ∧ a_max = 1/2 ∧ ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3891_389174


namespace NUMINAMATH_CALUDE_linear_equation_integer_solutions_l3891_389153

theorem linear_equation_integer_solutions :
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0) ∧ 
    (∀ m ∈ S, ∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) ∧
    (∀ m : ℕ, m > 0 → (∃ x : ℕ, x > 0 ∧ m * x + 2 * x - 12 = 0) → m ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_integer_solutions_l3891_389153


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3891_389136

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (3,x), 
    if (a + b) is perpendicular to a, then x = -4. -/
theorem perpendicular_vectors_x_value : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∀ i : Fin 2, (a + b) i * a i = 0) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3891_389136


namespace NUMINAMATH_CALUDE_fuel_cost_calculation_l3891_389140

/-- Calculates the new fuel cost after a price increase and capacity increase -/
def new_fuel_cost (original_cost : ℚ) (price_increase_percent : ℚ) (capacity_multiplier : ℚ) : ℚ :=
  original_cost * (1 + price_increase_percent / 100) * capacity_multiplier

/-- Proves that the new fuel cost is $480 given the specified conditions -/
theorem fuel_cost_calculation :
  new_fuel_cost 200 20 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_fuel_cost_calculation_l3891_389140


namespace NUMINAMATH_CALUDE_vectors_opposite_direction_l3891_389117

/-- Given non-zero vectors a and b satisfying a + 4b = 0, prove that the directions of a and b are opposite. -/
theorem vectors_opposite_direction {n : Type*} [NormedAddCommGroup n] [NormedSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + 4 • b = 0) : 
  ∃ (k : ℝ), k < 0 ∧ a = k • b :=
sorry

end NUMINAMATH_CALUDE_vectors_opposite_direction_l3891_389117


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3891_389157

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Constructs the first line l1 given parameter a -/
def line1 (a : ℝ) : Line :=
  { a := a, b := a + 2, c := 1 }

/-- Constructs the second line l2 given parameter a -/
def line2 (a : ℝ) : Line :=
  { a := 1, b := a, c := 2 }

/-- States that a = -3 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -3 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -3 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3891_389157


namespace NUMINAMATH_CALUDE_abc_product_l3891_389150

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + 1/b = 5) (h2 : b + 1/c = 2) (h3 : (c + 1/a)^2 = 4) :
  a * b * c = (11 + Real.sqrt 117) / 2 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3891_389150


namespace NUMINAMATH_CALUDE_cube_edge_length_from_sphere_volume_l3891_389128

theorem cube_edge_length_from_sphere_volume (V : ℝ) (h : V = 4 * Real.pi / 3) :
  ∃ (a : ℝ), a > 0 ∧ a = 2 * Real.sqrt 3 / 3 ∧
  V = 4 * Real.pi * (3 * a^2 / 4) / 3 :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_sphere_volume_l3891_389128


namespace NUMINAMATH_CALUDE_tailoring_cost_l3891_389192

theorem tailoring_cost (num_shirts num_pants : ℕ) (shirt_time : ℝ) (hourly_rate : ℝ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 1.5 →
  hourly_rate = 30 →
  (num_shirts * shirt_time + num_pants * (2 * shirt_time)) * hourly_rate = 1530 := by
  sorry

end NUMINAMATH_CALUDE_tailoring_cost_l3891_389192


namespace NUMINAMATH_CALUDE_circle_area_through_points_l3891_389169

/-- The area of a circle with center P(5, -2) passing through point Q(-7, 6) is 208π -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (5, -2)
  let Q : ℝ × ℝ := (-7, 6)
  let r := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  π * r^2 = 208 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l3891_389169


namespace NUMINAMATH_CALUDE_circle_area_around_equilateral_triangle_l3891_389110

theorem circle_area_around_equilateral_triangle :
  let side_length : ℝ := 12
  let circumradius : ℝ := side_length / Real.sqrt 3
  let circle_area : ℝ := Real.pi * circumradius^2
  circle_area = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_around_equilateral_triangle_l3891_389110


namespace NUMINAMATH_CALUDE_distinct_collections_eq_110_l3891_389141

def vowels : ℕ := 5
def consonants : ℕ := 4
def indistinguishable_consonants : ℕ := 2
def vowels_to_select : ℕ := 3
def consonants_to_select : ℕ := 4

def distinct_collections : ℕ :=
  (Nat.choose vowels vowels_to_select) *
  (Nat.choose consonants consonants_to_select +
   Nat.choose consonants (consonants_to_select - 1) +
   Nat.choose consonants (consonants_to_select - 2))

theorem distinct_collections_eq_110 :
  distinct_collections = 110 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_eq_110_l3891_389141


namespace NUMINAMATH_CALUDE_max_visible_sum_is_164_l3891_389155

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers used to form each cube --/
def cube_numbers : Finset ℕ := {1, 2, 4, 8, 16, 32}

/-- A cube is valid if it uses exactly the numbers in cube_numbers --/
def valid_cube (c : Cube) : Prop :=
  (Finset.image c.faces (Finset.univ : Finset (Fin 6))) = cube_numbers

/-- The sum of visible faces when a cube is stacked --/
def visible_sum (c : Cube) (top : Bool) : ℕ :=
  if top then
    c.faces 0 + c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4
  else
    c.faces 1 + c.faces 2 + c.faces 3 + c.faces 4

/-- The theorem to be proved --/
theorem max_visible_sum_is_164 :
  ∃ (c1 c2 c3 : Cube),
    valid_cube c1 ∧ valid_cube c2 ∧ valid_cube c3 ∧
    visible_sum c1 false + visible_sum c2 false + visible_sum c3 true = 164 ∧
    ∀ (d1 d2 d3 : Cube),
      valid_cube d1 → valid_cube d2 → valid_cube d3 →
      visible_sum d1 false + visible_sum d2 false + visible_sum d3 true ≤ 164 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_sum_is_164_l3891_389155


namespace NUMINAMATH_CALUDE_triangle_side_length_l3891_389132

/-- Proves that in a triangle ABC with given conditions, the length of side c is 20 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  a = 4 → B = π / 3 → S = 20 * Real.sqrt 3 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3891_389132


namespace NUMINAMATH_CALUDE_abigail_savings_l3891_389179

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end NUMINAMATH_CALUDE_abigail_savings_l3891_389179


namespace NUMINAMATH_CALUDE_sin_difference_product_l3891_389101

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l3891_389101


namespace NUMINAMATH_CALUDE_quadratic_properties_l3891_389199

/-- A quadratic function with vertex at (1, -4) and axis of symmetry at x = 1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  (∀ x, f a b c x = a * (x - 1)^2 - 4) →
  (2 * a + b = 0) ∧
  (f a b c (-1) = 0 ∧ f a b c 3 = 0) ∧
  (∀ m, f a b c (m - 1) < f a b c m → m > 3/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3891_389199


namespace NUMINAMATH_CALUDE_max_single_color_coins_l3891_389193

/-- Represents the state of coins -/
structure CoinState where
  red : Nat
  yellow : Nat
  blue : Nat

/-- Represents a coin exchange -/
inductive Exchange
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an exchange to a coin state -/
def applyExchange (state : CoinState) (exchange : Exchange) : CoinState :=
  match exchange with
  | Exchange.RedYellowToBlue => 
      { red := state.red - 1, yellow := state.yellow - 1, blue := state.blue + 1 }
  | Exchange.RedBlueToYellow => 
      { red := state.red - 1, yellow := state.yellow + 1, blue := state.blue - 1 }
  | Exchange.YellowBlueToRed => 
      { red := state.red + 1, yellow := state.yellow - 1, blue := state.blue - 1 }

/-- Checks if all coins are of the same color -/
def isSingleColor (state : CoinState) : Bool :=
  (state.red = 0 && state.blue = 0) || 
  (state.red = 0 && state.yellow = 0) || 
  (state.yellow = 0 && state.blue = 0)

/-- Counts the total number of coins -/
def totalCoins (state : CoinState) : Nat :=
  state.red + state.yellow + state.blue

/-- The main theorem to prove -/
theorem max_single_color_coins :
  ∃ (finalState : CoinState) (exchanges : List Exchange), 
    let initialState := { red := 3, yellow := 4, blue := 5 : CoinState }
    finalState = exchanges.foldl applyExchange initialState ∧
    isSingleColor finalState ∧
    totalCoins finalState = 7 ∧
    finalState.yellow = 7 ∧
    ∀ (otherState : CoinState) (otherExchanges : List Exchange),
      otherState = otherExchanges.foldl applyExchange initialState →
      isSingleColor otherState →
      totalCoins otherState ≤ totalCoins finalState :=
by
  sorry


end NUMINAMATH_CALUDE_max_single_color_coins_l3891_389193


namespace NUMINAMATH_CALUDE_f_value_at_one_l3891_389113

/-- A quadratic function f(x) with a specific behavior -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The function is increasing on [-2, +∞) -/
def increasing_on_right (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- The function is decreasing on (-∞, -2] -/
def decreasing_on_left (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -2 → f m x > f m y

theorem f_value_at_one (m : ℝ) 
  (h1 : increasing_on_right m) 
  (h2 : decreasing_on_left m) : 
  f m 1 = 25 := by sorry

end NUMINAMATH_CALUDE_f_value_at_one_l3891_389113


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3891_389134

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - x - 2 < 0 ↔ -1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3891_389134


namespace NUMINAMATH_CALUDE_max_value_fraction_l3891_389173

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2*x + y)) + (y / (x + 2*y)) ≤ 2/3 ∧ 
  ((x / (2*x + y)) + (y / (x + 2*y)) = 2/3 ↔ 2*x + y = x + 2*y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3891_389173


namespace NUMINAMATH_CALUDE_intersection_area_is_zero_l3891_389109

/-- The first curve: x^2 + y^2 = 16 -/
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- The second curve: (x-3)^2 + y^2 = 9 -/
def curve2 (x y : ℝ) : Prop := (x-3)^2 + y^2 = 9

/-- The set of intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

/-- The polygon formed by the intersection points -/
def intersection_polygon : Set (ℝ × ℝ) :=
  intersection_points

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is 0 -/
theorem intersection_area_is_zero :
  area intersection_polygon = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_zero_l3891_389109


namespace NUMINAMATH_CALUDE_inequality_proof_l3891_389149

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3891_389149


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3891_389143

/-- A quadratic function f(x) = ax^2 + bx + c with a ≠ 0 and satisfying 5a + b + 2c = 0 has two distinct real roots. -/
theorem quadratic_two_roots (a b c : ℝ) (ha : a ≠ 0) (h_cond : 5 * a + b + 2 * c = 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3891_389143


namespace NUMINAMATH_CALUDE_olivers_shirts_l3891_389135

theorem olivers_shirts (short_sleeve : ℕ) (washed : ℕ) (unwashed : ℕ) :
  short_sleeve = 39 →
  washed = 20 →
  unwashed = 66 →
  ∃ (long_sleeve : ℕ), long_sleeve = 7 ∧ short_sleeve + long_sleeve = washed + unwashed :=
by sorry

end NUMINAMATH_CALUDE_olivers_shirts_l3891_389135


namespace NUMINAMATH_CALUDE_max_swaps_is_19_l3891_389104

/-- A permutation of the numbers 1 to 20 -/
def Permutation := Fin 20 → Fin 20

/-- The identity permutation -/
def id_perm : Permutation := fun i => i

/-- A swap operation on a permutation -/
def swap (p : Permutation) (i j : Fin 20) : Permutation :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- The minimum number of swaps needed to transform a permutation into the identity permutation -/
def min_swaps (p : Permutation) : ℕ := sorry

/-- Theorem: The maximum number of swaps needed for any permutation is 19 -/
theorem max_swaps_is_19 :
  ∃ (p : Permutation), min_swaps p = 19 ∧ 
  ∀ (q : Permutation), min_swaps q ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_swaps_is_19_l3891_389104


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3891_389151

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h_a : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 8 * x
  let focus : ℝ × ℝ := (2, 0)
  hyperbola focus.1 focus.2 →
  let c := Real.sqrt (a^2 + 1)
  c / a = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3891_389151


namespace NUMINAMATH_CALUDE_triangle_area_from_parametric_lines_l3891_389116

/-- The area of a triangle formed by two points on given lines and the origin -/
theorem triangle_area_from_parametric_lines (t s : ℝ) : 
  let l : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 3 + 5*t ∧ p.2 = 2 + 4*t
  let m : ℝ × ℝ → Prop := λ p => ∃ s, p.1 = 2 + 5*s ∧ p.2 = 3 + 4*s
  let C : ℝ × ℝ := (3 + 5*t, 2 + 4*t)
  let D : ℝ × ℝ := (2 + 5*s, 3 + 4*s)
  let O : ℝ × ℝ := (0, 0)
  l C → m D → 
  (1/2 : ℝ) * |5 + 2*s + 7*t| = 
  (1/2 : ℝ) * |C.1 * D.2 - C.2 * D.1| :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_parametric_lines_l3891_389116


namespace NUMINAMATH_CALUDE_sick_animals_count_l3891_389167

/-- The number of chickens at Stacy's farm -/
def num_chickens : ℕ := 26

/-- The number of piglets at Stacy's farm -/
def num_piglets : ℕ := 40

/-- The number of goats at Stacy's farm -/
def num_goats : ℕ := 34

/-- The fraction of animals that get sick -/
def sick_fraction : ℚ := 1/2

/-- The total number of sick animals -/
def total_sick_animals : ℕ := (num_chickens + num_piglets + num_goats) / 2

theorem sick_animals_count : total_sick_animals = 50 := by
  sorry

end NUMINAMATH_CALUDE_sick_animals_count_l3891_389167


namespace NUMINAMATH_CALUDE_sum_of_four_variables_l3891_389181

theorem sum_of_four_variables (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 250)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 3) :
  a + b + c + d = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_variables_l3891_389181


namespace NUMINAMATH_CALUDE_jenna_peeled_potatoes_l3891_389120

/-- The number of potatoes Jenna peeled -/
def jenna_potatoes : ℕ := 24

/-- The total number of potatoes -/
def total_potatoes : ℕ := 60

/-- Homer's peeling rate in potatoes per minute -/
def homer_rate : ℕ := 4

/-- Jenna's peeling rate in potatoes per minute -/
def jenna_rate : ℕ := 6

/-- The time Homer peeled alone in minutes -/
def homer_alone_time : ℕ := 6

/-- The combined peeling rate of Homer and Jenna in potatoes per minute -/
def combined_rate : ℕ := homer_rate + jenna_rate

theorem jenna_peeled_potatoes :
  jenna_potatoes = total_potatoes - (homer_rate * homer_alone_time) :=
by sorry

#check jenna_peeled_potatoes

end NUMINAMATH_CALUDE_jenna_peeled_potatoes_l3891_389120


namespace NUMINAMATH_CALUDE_max_value_ln_minus_x_l3891_389142

open Real

theorem max_value_ln_minus_x :
  ∃ (x : ℝ), 0 < x ∧ x ≤ exp 1 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ exp 1 → log y - y ≤ log x - x) ∧
  log x - x = -1 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ln_minus_x_l3891_389142


namespace NUMINAMATH_CALUDE_component_scrap_probability_l3891_389188

/-- The probability of a component passing the first inspection -/
def p_pass_first : ℝ := 0.8

/-- The probability of a component passing the second inspection, given it failed the first -/
def p_pass_second : ℝ := 0.9

/-- The probability of a component being scrapped -/
def p_scrapped : ℝ := (1 - p_pass_first) * (1 - p_pass_second)

theorem component_scrap_probability :
  p_scrapped = 0.02 :=
sorry

end NUMINAMATH_CALUDE_component_scrap_probability_l3891_389188


namespace NUMINAMATH_CALUDE_neg_f_is_reflection_about_x_axis_l3891_389197

/-- A function representing the original graph -/
def f : ℝ → ℝ := sorry

/-- The negation of function f -/
def neg_f (x : ℝ) : ℝ := -f x

/-- Theorem stating that neg_f is a reflection of f about the x-axis -/
theorem neg_f_is_reflection_about_x_axis :
  ∀ x y : ℝ, f x = y ↔ neg_f x = -y :=
sorry

end NUMINAMATH_CALUDE_neg_f_is_reflection_about_x_axis_l3891_389197


namespace NUMINAMATH_CALUDE_divisors_of_sum_of_primes_l3891_389144

-- Define a prime number p ≥ 5
def p : ℕ := sorry

-- Define q as the smallest prime number greater than p
def q : ℕ := sorry

-- Define n as the number of positive divisors of p + q
def n : ℕ := sorry

-- Axioms based on the problem conditions
axiom p_prime : Nat.Prime p
axiom p_ge_5 : p ≥ 5
axiom q_prime : Nat.Prime q
axiom q_gt_p : q > p
axiom q_smallest : ∀ r, Nat.Prime r → r > p → r ≥ q

-- Theorem to prove
theorem divisors_of_sum_of_primes :
  n ≥ 4 ∧ (∀ m, m ≥ 6 → n ≤ m) := by sorry

end NUMINAMATH_CALUDE_divisors_of_sum_of_primes_l3891_389144


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_sum_l3891_389191

theorem arithmetic_sequence_constant_sum (a₁ d : ℝ) :
  let a : ℕ → ℝ := λ n => a₁ + (n - 1) * d
  let S : ℕ → ℝ := λ n => n * (2 * a₁ + (n - 1) * d) / 2
  (∀ a₁' d', a₁' + (1 + 7 + 10) * d' = a₁ + (1 + 7 + 10) * d) →
  (∀ a₁' d', S 13 = 13 * (2 * a₁' + 12 * d') / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_sum_l3891_389191


namespace NUMINAMATH_CALUDE_shells_weight_calculation_l3891_389187

/-- Given an initial weight of shells and an additional weight of shells,
    calculate the total weight of shells. -/
def total_weight (initial_weight additional_weight : ℕ) : ℕ :=
  initial_weight + additional_weight

/-- Theorem: The total weight of shells is 17 pounds when
    the initial weight is 5 pounds and the additional weight is 12 pounds. -/
theorem shells_weight_calculation :
  total_weight 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_shells_weight_calculation_l3891_389187


namespace NUMINAMATH_CALUDE_exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l3891_389165

/-- The measure of an exterior angle formed by a regular hexagon and a regular octagon sharing a common side -/
theorem exterior_angle_hexagon_octagon : ℝ :=
  let hexagon_interior_angle := (180 * (6 - 2) / 6 : ℝ)
  let octagon_interior_angle := (180 * (8 - 2) / 8 : ℝ)
  360 - (hexagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular hexagon and a regular octagon sharing a common side is 105 degrees -/
theorem exterior_angle_hexagon_octagon_is_105 :
  exterior_angle_hexagon_octagon = 105 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_hexagon_octagon_exterior_angle_hexagon_octagon_is_105_l3891_389165


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l3891_389102

theorem initially_calculated_average_weight
  (n : ℕ)
  (correct_avg : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : n = 20)
  (h2 : correct_avg = 58.9)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 66)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 :=
by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l3891_389102


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3891_389145

theorem power_mod_eleven : 5^2023 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3891_389145


namespace NUMINAMATH_CALUDE_max_sequence_length_l3891_389147

/-- Represents a quadratic equation in the sequence -/
structure QuadraticEquation where
  p : ℝ
  q : ℝ
  h : p < q

/-- Constructs the next quadratic equation in the sequence -/
def nextEquation (eq : QuadraticEquation) : QuadraticEquation :=
  { p := eq.q, q := -eq.p - eq.q, h := sorry }

/-- The sequence of quadratic equations -/
def quadraticSequence (initial : QuadraticEquation) : ℕ → QuadraticEquation
  | 0 => initial
  | n + 1 => nextEquation (quadraticSequence initial n)

/-- The main theorem: the maximum length of the sequence is 5 -/
theorem max_sequence_length (initial : QuadraticEquation) :
  ∃ n : ℕ, n ≤ 5 ∧ ∀ m : ℕ, m > n → ¬ (quadraticSequence initial m).p < (quadraticSequence initial m).q :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_l3891_389147


namespace NUMINAMATH_CALUDE_wax_left_after_detailing_l3891_389195

/-- The amount of wax needed to detail Kellan's car in ounces -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces -/
def bottle_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces -/
def spilled_wax : ℕ := 2

/-- Theorem stating the amount of wax left after detailing both vehicles -/
theorem wax_left_after_detailing : 
  bottle_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_left_after_detailing_l3891_389195


namespace NUMINAMATH_CALUDE_odd_function_inequality_l3891_389137

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_ineq : ∀ x₁ x₂, x₁ < 0 → x₂ < 0 → x₁ ≠ x₂ → 
    (x₂ * f x₁ - x₁ * f x₂) / (x₁ - x₂) > 0) :
  3 * f (1/3) > -5/2 * f (-2/5) ∧ -5/2 * f (-2/5) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l3891_389137


namespace NUMINAMATH_CALUDE_floor_sum_example_l3891_389112

theorem floor_sum_example : ⌊(2.7 : ℝ) + 1.5⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3891_389112


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3891_389126

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℂ, X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * q :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3891_389126


namespace NUMINAMATH_CALUDE_unique_solution_l3891_389196

theorem unique_solution (a m n : ℕ+) (h : Real.sqrt (a^2 - 4 * Real.sqrt 2) = Real.sqrt m - Real.sqrt n) :
  m = 8 ∧ n = 1 ∧ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3891_389196


namespace NUMINAMATH_CALUDE_math_competition_problem_l3891_389122

theorem math_competition_problem :
  ∀ (total students_only_A students_A_and_others students_only_B students_only_C students_B_and_C : ℕ),
    total = 25 →
    total = students_only_A + students_A_and_others + students_only_B + students_only_C + students_B_and_C →
    students_only_B + students_B_and_C = 2 * (students_only_C + students_B_and_C) →
    students_only_A = students_A_and_others + 1 →
    2 * (students_only_B + students_only_C) = students_only_A →
    students_only_B = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l3891_389122


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l3891_389118

theorem arccos_cos_nine : Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l3891_389118


namespace NUMINAMATH_CALUDE_triangle_problem_l3891_389114

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h_area : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15)
  (h_diff : t.b - t.c = 2)
  (h_cosA : Real.cos t.A = -1/4) : 
  t.a = 8 ∧ Real.sin t.C = Real.sqrt 15 / 8 ∧ 
  Real.cos (2 * t.A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3891_389114


namespace NUMINAMATH_CALUDE_expression_value_l3891_389106

theorem expression_value (a b x y c : ℝ) 
  (h1 : a = -b) 
  (h2 : x * y = 1) 
  (h3 : c = 2 ∨ c = -2) : 
  (a + b) / 2 + x * y - (1 / 4) * c = 1 / 2 ∨ (a + b) / 2 + x * y - (1 / 4) * c = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3891_389106


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3891_389177

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 4}

-- Define set N
def N : Set Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equals_set :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3891_389177


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3891_389172

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 →
  p_black = 0.5 →
  p_red + p_black + p_white = 1 →
  p_white = 0.2 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3891_389172


namespace NUMINAMATH_CALUDE_quadratic_function_domain_range_l3891_389166

theorem quadratic_function_domain_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, ∃ y ∈ Set.Icc (-6) (-2), y = x^2 - 4*x - 2) ∧
  (∀ y ∈ Set.Icc (-6) (-2), ∃ x ∈ Set.Icc 0 m, y = x^2 - 4*x - 2) →
  m ∈ Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_domain_range_l3891_389166


namespace NUMINAMATH_CALUDE_largest_value_l3891_389178

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > max (1/2) (max (a^2 + b^2) (2*a*b)) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3891_389178


namespace NUMINAMATH_CALUDE_tims_sleep_hours_l3891_389175

/-- Proves that Tim slept 6 hours each day for the first 2 days given the conditions -/
theorem tims_sleep_hours (x : ℝ) : 
  (2 * x + 2 * 10 = 32) → x = 6 := by sorry

end NUMINAMATH_CALUDE_tims_sleep_hours_l3891_389175


namespace NUMINAMATH_CALUDE_horner_rule_v₃_l3891_389185

/-- Horner's Rule for a polynomial of degree 6 -/
def horner_rule (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  ((((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁) * x + a₀)

/-- The third intermediate value in Horner's Rule calculation -/
def v₃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ x : ℝ) : ℝ :=
  (((a₆ * x + a₅) * x + a₄) * x + a₃)

theorem horner_rule_v₃ :
  v₃ 64 (-192) 240 (-160) 60 (-12) 1 2 = -80 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_v₃_l3891_389185
