import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_24_l2932_293242

/-- An arithmetic expression using integers and basic operators -/
inductive Expr where
  | const : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an arithmetic expression -/
def eval : Expr → Int
  | Expr.const n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Check if an expression uses each of the given numbers exactly once -/
def usesNumbers (e : Expr) (nums : List Int) : Bool := sorry

/-- There exists an arithmetic expression using 1, 4, 7, and 7 that evaluates to 24 -/
theorem expression_equals_24 : ∃ e : Expr, 
  usesNumbers e [1, 4, 7, 7] ∧ eval e = 24 := by sorry

end NUMINAMATH_CALUDE_expression_equals_24_l2932_293242


namespace NUMINAMATH_CALUDE_triangle_problem_l2932_293290

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that
    under certain conditions, angle A is π/4 and the area is 9/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a = 3 →
  b^2 + c^2 - a^2 - Real.sqrt 2 * b * c = 0 →
  Real.sin B^2 + Real.sin C^2 = 2 * Real.sin A^2 →
  A = π / 4 ∧ 
  (1/2) * b * c * Real.sin A = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2932_293290


namespace NUMINAMATH_CALUDE_max_value_theorem_l2932_293274

/-- A function satisfying the given recurrence relation -/
def RecurrenceFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = 1 + Real.sqrt (2 * f x - f x ^ 2)

/-- The theorem stating the maximum value of f(1) + f(2020) -/
theorem max_value_theorem (f : ℝ → ℝ) (h : RecurrenceFunction f) :
    ∃ M : ℝ, M = 2 + Real.sqrt 2 ∧ f 1 + f 2020 ≤ M ∧ 
    ∃ g : ℝ → ℝ, RecurrenceFunction g ∧ g 1 + g 2020 = M :=
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2932_293274


namespace NUMINAMATH_CALUDE_pie_slices_remaining_l2932_293267

theorem pie_slices_remaining (total_slices : ℕ) 
  (joe_fraction darcy_fraction carl_fraction emily_fraction : ℚ) : 
  total_slices = 24 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  total_slices - (total_slices * joe_fraction + total_slices * darcy_fraction + 
    total_slices * carl_fraction + total_slices * emily_fraction) = 3 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_remaining_l2932_293267


namespace NUMINAMATH_CALUDE_john_initial_diamonds_l2932_293250

/-- Represents the number of diamonds each pirate has -/
structure DiamondCount where
  bill : ℕ
  sam : ℕ
  john : ℕ

/-- Represents the average mass of diamonds for each pirate -/
structure AverageMass where
  bill : ℝ
  sam : ℝ
  john : ℝ

/-- The initial distribution of diamonds -/
def initial_distribution : DiamondCount :=
  { bill := 12, sam := 12, john := 9 }

/-- The distribution after the theft events -/
def final_distribution : DiamondCount :=
  { bill := initial_distribution.bill,
    sam := initial_distribution.sam,
    john := initial_distribution.john }

/-- The change in average mass for each pirate -/
def mass_change : AverageMass :=
  { bill := -1, sam := -2, john := 4 }

theorem john_initial_diamonds :
  initial_distribution.john = 9 →
  (initial_distribution.bill * mass_change.bill +
   initial_distribution.sam * mass_change.sam +
   initial_distribution.john * mass_change.john = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_john_initial_diamonds_l2932_293250


namespace NUMINAMATH_CALUDE_investment_problem_l2932_293270

theorem investment_problem (T : ℝ) :
  (0.10 * (T - 700) - 0.08 * 700 = 74) →
  T = 2000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l2932_293270


namespace NUMINAMATH_CALUDE_area_S_eq_four_sqrt_three_thirds_l2932_293231

/-- A rhombus with side length 4 and one angle of 150 degrees -/
structure Rhombus150 where
  side_length : ℝ
  angle_F : ℝ
  side_length_eq : side_length = 4
  angle_F_eq : angle_F = 150 * π / 180

/-- The region S inside the rhombus closer to vertex F than to other vertices -/
def region_S (r : Rhombus150) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region S -/
noncomputable def area_S (r : Rhombus150) : ℝ :=
  sorry

/-- Theorem stating that the area of region S is 4√3/3 -/
theorem area_S_eq_four_sqrt_three_thirds (r : Rhombus150) :
  area_S r = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_area_S_eq_four_sqrt_three_thirds_l2932_293231


namespace NUMINAMATH_CALUDE_negative_two_squared_l2932_293292

theorem negative_two_squared : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_l2932_293292


namespace NUMINAMATH_CALUDE_function_value_at_2_l2932_293245

/-- Given a function f(x) = ax^5 - bx + |x| - 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem function_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x + |x| - 1)
    (h2 : f (-2) = 2) : 
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2_l2932_293245


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l2932_293241

/-- Given two polynomials g and f, where g has three distinct roots that are also roots of f,
    prove that f(1) = -1333 -/
theorem polynomial_root_problem (a b c : ℝ) : 
  let g := fun x : ℝ => x^3 + a*x^2 + x + 8
  let f := fun x : ℝ => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ x : ℝ, g x = 0 → f x = 0) →
  f 1 = -1333 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l2932_293241


namespace NUMINAMATH_CALUDE_ruble_payment_l2932_293265

theorem ruble_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ruble_payment_l2932_293265


namespace NUMINAMATH_CALUDE_area_increase_rect_to_circle_l2932_293236

/-- Increase in area when changing a rectangular field to a circular field -/
theorem area_increase_rect_to_circle (length width : ℝ) (h1 : length = 60) (h2 : width = 20) :
  let rect_area := length * width
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  let circle_area := Real.pi * radius^2
  ∃ ε > 0, abs (circle_area - rect_area - 837.94) < ε :=
by sorry

end NUMINAMATH_CALUDE_area_increase_rect_to_circle_l2932_293236


namespace NUMINAMATH_CALUDE_expression_value_l2932_293219

theorem expression_value (x y : ℝ) (h1 : x = 3 * y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2932_293219


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2932_293238

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2932_293238


namespace NUMINAMATH_CALUDE_lindas_bills_l2932_293201

/-- Represents the number of bills of each denomination -/
structure BillCount where
  fives : ℕ
  tens : ℕ

/-- Calculates the total value of bills -/
def totalValue (bc : BillCount) : ℕ :=
  5 * bc.fives + 10 * bc.tens

/-- Calculates the total number of bills -/
def totalBills (bc : BillCount) : ℕ :=
  bc.fives + bc.tens

theorem lindas_bills :
  ∃ (bc : BillCount), totalValue bc = 80 ∧ totalBills bc = 12 ∧ bc.fives = 8 := by
  sorry

end NUMINAMATH_CALUDE_lindas_bills_l2932_293201


namespace NUMINAMATH_CALUDE_inequality_proof_l2932_293203

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2932_293203


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2932_293246

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver
  | BlueToSilver
  | BothToSilver

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver => 
      { red := tc.red - 4, blue := tc.blue + 1, silver := tc.silver + 2 }
  | ExchangeRule.BlueToSilver => 
      { red := tc.red + 1, blue := tc.blue - 5, silver := tc.silver + 2 }
  | ExchangeRule.BothToSilver => 
      { red := tc.red - 3, blue := tc.blue - 3, silver := tc.silver + 3 }

/-- Checks if an exchange is possible --/
def canExchange (tc : TokenCount) (rule : ExchangeRule) : Prop :=
  match rule with
  | ExchangeRule.RedToSilver => tc.red ≥ 4
  | ExchangeRule.BlueToSilver => tc.blue ≥ 5
  | ExchangeRule.BothToSilver => tc.red ≥ 3 ∧ tc.blue ≥ 3

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (final : TokenCount), 
    (∃ (exchanges : List ExchangeRule), 
      final = exchanges.foldl applyExchange { red := 100, blue := 100, silver := 0 } ∧
      ∀ rule, ¬(canExchange final rule)) ∧
    final.silver = 85 :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l2932_293246


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2932_293271

theorem polynomial_division_theorem (x : ℚ) : 
  let dividend := 10 * x^4 - 3 * x^3 + 2 * x^2 - x + 6
  let divisor := 3 * x + 4
  let quotient := 10/3 * x^3 - 49/9 * x^2 + 427/27 * x - 287/54
  let remainder := 914/27
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2932_293271


namespace NUMINAMATH_CALUDE_log_and_power_equality_l2932_293224

theorem log_and_power_equality : 
  (Real.log 32 - Real.log 4) / Real.log 2 + (27 : ℝ) ^ (2/3) = 12 := by sorry

end NUMINAMATH_CALUDE_log_and_power_equality_l2932_293224


namespace NUMINAMATH_CALUDE_total_interest_compound_linh_investment_interest_l2932_293287

/-- Calculate the total interest earned on an investment with compound interest -/
theorem total_interest_compound (P : ℝ) (r : ℝ) (n : ℕ) :
  let A := P * (1 + r) ^ n
  A - P = P * ((1 + r) ^ n - 1) := by
  sorry

/-- Prove the total interest earned for Linh's investment -/
theorem linh_investment_interest :
  let P : ℝ := 1200  -- Initial investment
  let r : ℝ := 0.08  -- Annual interest rate
  let n : ℕ := 4     -- Number of years
  let A := P * (1 + r) ^ n
  A - P = 1200 * ((1 + 0.08) ^ 4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_total_interest_compound_linh_investment_interest_l2932_293287


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2932_293204

theorem complex_magnitude_fourth_power : Complex.abs ((2 + 3*Complex.I)^4) = 169 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l2932_293204


namespace NUMINAMATH_CALUDE_seed_mixture_problem_l2932_293209

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The problem statement -/
theorem seed_mixture_problem (X Y : SeedMixture) (mixture_weight : ℝ) :
  X.ryegrass = 40 →
  Y.ryegrass = 25 →
  Y.fescue = 75 →
  X.ryegrass + X.bluegrass + X.fescue = 100 →
  Y.ryegrass + Y.bluegrass + Y.fescue = 100 →
  mixture_weight * 30 / 100 = X.ryegrass * (mixture_weight * 100 / 3 / 100) + Y.ryegrass * (mixture_weight * 200 / 3 / 100) →
  X.bluegrass = 60 := by
  sorry


end NUMINAMATH_CALUDE_seed_mixture_problem_l2932_293209


namespace NUMINAMATH_CALUDE_billy_reads_three_books_l2932_293214

theorem billy_reads_three_books 
  (free_time_per_day : ℕ) 
  (weekend_days : ℕ) 
  (video_game_percentage : ℚ) 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (h1 : free_time_per_day = 8)
  (h2 : weekend_days = 2)
  (h3 : video_game_percentage = 3/4)
  (h4 : pages_per_hour = 60)
  (h5 : pages_per_book = 80) :
  (free_time_per_day * weekend_days * (1 - video_game_percentage) * pages_per_hour) / pages_per_book = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_reads_three_books_l2932_293214


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l2932_293276

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = Real.sqrt 65 / 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l2932_293276


namespace NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2932_293289

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def A : ℕ := digit_sum (4444^4444)

def B : ℕ := digit_sum A

theorem sum_of_digits_B_is_seven :
  digit_sum B = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_B_is_seven_l2932_293289


namespace NUMINAMATH_CALUDE_games_not_working_l2932_293266

theorem games_not_working (friend_games garage_games good_games : ℕ) : 
  friend_games = 2 → garage_games = 2 → good_games = 2 →
  friend_games + garage_games - good_games = 2 := by
sorry

end NUMINAMATH_CALUDE_games_not_working_l2932_293266


namespace NUMINAMATH_CALUDE_cube_plus_self_equality_l2932_293239

theorem cube_plus_self_equality (m n : ℤ) : m^3 = n^3 + n → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_self_equality_l2932_293239


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2932_293286

theorem no_solution_for_sock_problem : ¬∃ (n m : ℕ), 
  n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2932_293286


namespace NUMINAMATH_CALUDE_linear_function_above_x_axis_l2932_293235

/-- A linear function y = ax + a + 2 is above the x-axis for -2 ≤ x ≤ 1 if and only if
    -1 < a < 2 and a ≠ 0 -/
theorem linear_function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → a * x + a + 2 > 0) ↔ (-1 < a ∧ a < 2 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_above_x_axis_l2932_293235


namespace NUMINAMATH_CALUDE_expected_bullets_is_1_89_l2932_293218

/-- The expected number of remaining bullets in a shooting scenario -/
def expected_remaining_bullets (total_bullets : ℕ) (hit_probability : ℝ) : ℝ :=
  let miss_probability := 1 - hit_probability
  let p_zero := miss_probability * miss_probability
  let p_one := miss_probability * hit_probability
  let p_two := hit_probability
  1 * p_one + 2 * p_two

/-- The theorem stating that the expected number of remaining bullets is 1.89 -/
theorem expected_bullets_is_1_89 :
  expected_remaining_bullets 3 0.9 = 1.89 := by sorry

end NUMINAMATH_CALUDE_expected_bullets_is_1_89_l2932_293218


namespace NUMINAMATH_CALUDE_quadratic_properties_l2932_293272

def quadratic_function (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_properties (a h k : ℝ) :
  quadratic_function a h k (-2) = 0 →
  quadratic_function a h k 4 = 0 →
  quadratic_function a h k 1 = -9/2 →
  (a = 1/2 ∧ h = 1 ∧ k = -9/2 ∧ ∀ x, quadratic_function a h k x ≥ -9/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2932_293272


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2932_293205

/-- The volume of ice cream in a cone with a spherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let sphere_volume := (4/3) * π * r^3
  cone_volume + sphere_volume = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l2932_293205


namespace NUMINAMATH_CALUDE_exist_valid_subgrid_l2932_293275

/-- Represents a grid of 0s and 1s -/
def Grid := Matrix (Fin 100) (Fin 2018) Bool

/-- A predicate that checks if a grid satisfies the condition of having at least 75 ones in each column -/
def ValidGrid (g : Grid) : Prop :=
  ∀ j : Fin 2018, (Finset.filter (fun i => g i j) Finset.univ).card ≥ 75

/-- A predicate that checks if a 5-row subgrid has at most one all-zero column -/
def ValidSubgrid (g : Grid) (rows : Finset (Fin 100)) : Prop :=
  rows.card = 5 ∧
  (Finset.filter (fun j : Fin 2018 => ∀ i ∈ rows, ¬g i j) Finset.univ).card ≤ 1

/-- The main theorem to be proved -/
theorem exist_valid_subgrid (g : Grid) (h : ValidGrid g) :
  ∃ rows : Finset (Fin 100), ValidSubgrid g rows := by
  sorry

end NUMINAMATH_CALUDE_exist_valid_subgrid_l2932_293275


namespace NUMINAMATH_CALUDE_combination_sum_identity_l2932_293216

theorem combination_sum_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_identity_l2932_293216


namespace NUMINAMATH_CALUDE_root_implies_sum_l2932_293202

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_implies_sum (a b : ℤ) : 
  (∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) →
  b - a = 1 →
  a + b = -3 := by sorry

end NUMINAMATH_CALUDE_root_implies_sum_l2932_293202


namespace NUMINAMATH_CALUDE_chantel_final_bracelet_count_l2932_293228

/-- The number of bracelets Chantel has at the end -/
def final_bracelet_count : ℕ :=
  let first_week_production := 7 * 4
  let after_first_giveaway := first_week_production - 8
  let second_period_production := 10 * 5
  let before_second_giveaway := after_first_giveaway + second_period_production
  before_second_giveaway - 12

/-- Theorem stating that Chantel ends up with 58 bracelets -/
theorem chantel_final_bracelet_count : final_bracelet_count = 58 := by
  sorry

end NUMINAMATH_CALUDE_chantel_final_bracelet_count_l2932_293228


namespace NUMINAMATH_CALUDE_quadratic_roots_l2932_293254

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 0 ∧ x₂ = 4/5) ∧ 
  (∀ x : ℝ, 5 * x^2 = 4 * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2932_293254


namespace NUMINAMATH_CALUDE_new_average_after_modification_l2932_293264

def consecutive_integers (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i)

def modified_sequence (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i - (9 - i))

theorem new_average_after_modification (start : ℤ) :
  (consecutive_integers start).sum / 10 = 20 →
  (modified_sequence start).sum / 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_modification_l2932_293264


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l2932_293212

theorem walnut_trees_planted (trees_before planting : ℕ) (trees_after : ℕ) : 
  trees_before = 22 → trees_after = 55 → planting = trees_after - trees_before :=
by
  sorry

#check walnut_trees_planted 22 33 55

end NUMINAMATH_CALUDE_walnut_trees_planted_l2932_293212


namespace NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2932_293293

/-- A rectangular solid with volume 512 cm³, surface area 384 cm², and dimensions in geometric progression has a sum of edge lengths equal to 96 cm. -/
theorem rectangular_solid_edge_sum : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 512 →
    2 * (a * b + b * c + a * c) = 384 →
    ∃ (r : ℝ), r > 0 ∧ b = a * r ∧ c = b * r →
    4 * (a + b + c) = 96 :=
by sorry


end NUMINAMATH_CALUDE_rectangular_solid_edge_sum_l2932_293293


namespace NUMINAMATH_CALUDE_probability_ABABABBB_proof_l2932_293253

/-- The probability of arranging 5 A tiles and 3 B tiles in the specific order ABABABBB -/
def probability_ABABABBB : ℚ :=
  1 / 56

/-- The total number of ways to arrange 5 A tiles and 3 B tiles in a row -/
def total_arrangements : ℕ :=
  Nat.choose 8 5

theorem probability_ABABABBB_proof :
  probability_ABABABBB = (1 : ℚ) / total_arrangements := by
  sorry

#eval probability_ABABABBB
#eval total_arrangements

end NUMINAMATH_CALUDE_probability_ABABABBB_proof_l2932_293253


namespace NUMINAMATH_CALUDE_squares_in_100th_ring_l2932_293221

/-- The number of squares in the nth ring of a diamond pattern -/
def ring_squares (n : ℕ) : ℕ :=
  4 + 8 * (n - 1)

/-- Theorem stating the number of squares in the 100th ring -/
theorem squares_in_100th_ring :
  ring_squares 100 = 796 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_100th_ring_l2932_293221


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2932_293211

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 0) 
  (h3 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2932_293211


namespace NUMINAMATH_CALUDE_youngsville_population_change_l2932_293251

def initial_population : ℕ := 684
def growth_rate : ℚ := 25 / 100
def decline_rate : ℚ := 40 / 100

theorem youngsville_population_change :
  let increased_population := initial_population + (initial_population * growth_rate).floor
  let final_population := increased_population - (increased_population * decline_rate).floor
  final_population = 513 := by sorry

end NUMINAMATH_CALUDE_youngsville_population_change_l2932_293251


namespace NUMINAMATH_CALUDE_classroom_pencils_l2932_293243

/-- The number of pencils a teacher needs to give out to a classroom of students -/
def pencils_to_give_out (num_students : ℕ) (dozens_per_student : ℕ) : ℕ :=
  num_students * (dozens_per_student * 12)

/-- Theorem: Given 46 children in a classroom, with each child receiving 4 dozen pencils,
    the total number of pencils the teacher needs to give out is 2208 -/
theorem classroom_pencils : pencils_to_give_out 46 4 = 2208 := by
  sorry

end NUMINAMATH_CALUDE_classroom_pencils_l2932_293243


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l2932_293237

-- Define the properties of rectangle R1
def R1_side : ℝ := 3
def R1_area : ℝ := 24

-- Define the diagonal of rectangle R2
def R2_diagonal : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle :
  let R1_other_side := R1_area / R1_side
  let ratio := R1_other_side / R1_side
  let R2_side := (R2_diagonal^2 / (1 + ratio^2))^(1/2)
  R2_side * (ratio * R2_side) = 28800 / 219 :=
by sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l2932_293237


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2932_293249

theorem quadratic_inequality_solution_set (x : ℝ) : 
  -x^2 + 2*x + 3 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2932_293249


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2932_293200

theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  (x - 1)^2 + (8 - 3)^2 = 15^2 → 
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2932_293200


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2932_293299

theorem inequality_system_solution :
  let S := {x : ℝ | 2*x > x + 1 ∧ 4*x - 1 > 7}
  S = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2932_293299


namespace NUMINAMATH_CALUDE_range_of_a_l2932_293220

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) →  -- p is true
  (∃ x : ℝ, x > 0 ∧ 2^x - a ≤ 0) →  -- q is false
  a ∈ Set.Ioo 1 2 :=  -- a is in the open interval (1, 2)
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2932_293220


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2932_293217

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2932_293217


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l2932_293258

theorem consecutive_squares_sum (n : ℤ) : 
  n^2 + (n + 1)^2 = 452 → n + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l2932_293258


namespace NUMINAMATH_CALUDE_root_range_l2932_293262

/-- Given that the equation |x-k| = (√2/2)k√x has two unequal real roots in the interval [k-1, k+1], prove that the range of k is 0 < k ≤ 1. -/
theorem root_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k - 1 ≤ x₁ ∧ x₁ ≤ k + 1 ∧
    k - 1 ≤ x₂ ∧ x₂ ≤ k + 1 ∧
    |x₁ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₁ ∧
    |x₂ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_l2932_293262


namespace NUMINAMATH_CALUDE_product_of_six_integers_square_sum_l2932_293291

theorem product_of_six_integers_square_sum (ints : Finset ℕ) : 
  ints = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  ∃ (A B : Finset ℕ), 
    A ⊆ ints ∧ B ⊆ ints ∧
    A.card = 6 ∧ B.card = 6 ∧
    A ≠ B ∧
    (∃ p : ℕ, (A.prod id : ℕ) = p^2) ∧
    (∃ q : ℕ, (B.prod id : ℕ) = q^2) ∧
    ∃ (p q : ℕ), 
      (A.prod id : ℕ) = p^2 ∧
      (B.prod id : ℕ) = q^2 ∧
      p + q = 108 :=
by sorry

end NUMINAMATH_CALUDE_product_of_six_integers_square_sum_l2932_293291


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2932_293281

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithm equations -/
def cryptarithm (A B C D E F G H J : Digit) : Prop :=
  (A.val * 10 + B.val) * (C.val * 10 + A.val) = D.val * 100 + E.val * 10 + B.val ∧
  F.val * 10 + C.val - (D.val * 10 + G.val) = D.val ∧
  E.val * 10 + G.val + H.val * 10 + J.val = A.val * 100 + A.val * 10 + G.val

/-- All digits are different -/
def all_different (A B C D E F G H J : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J

theorem cryptarithm_solution :
  ∃! (A B C D E F G H J : Digit),
    cryptarithm A B C D E F G H J ∧
    all_different A B C D E F G H J ∧
    A.val = 1 ∧ B.val = 7 ∧ C.val = 2 ∧ D.val = 3 ∧
    E.val = 5 ∧ F.val = 4 ∧ G.val = 9 ∧ H.val = 6 ∧ J.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2932_293281


namespace NUMINAMATH_CALUDE_factorization_equality_minimum_value_expression_minimum_value_at_one_l2932_293229

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  1 - 2 * (x - y) + (x - y)^2 = (1 - x + y)^2 := by sorry

-- Problem 2
theorem minimum_value_expression (n : ℝ) :
  (n^2 - 2*n - 3) * (n^2 - 2*n + 5) + 17 ≥ 1 := by sorry

theorem minimum_value_at_one :
  (1^2 - 2*1 - 3) * (1^2 - 2*1 + 5) + 17 = 1 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_minimum_value_expression_minimum_value_at_one_l2932_293229


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l2932_293277

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 18) 
  (eq2 : 5 * a + 4 * b = 31) : 
  2 * a + b = 11.5 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l2932_293277


namespace NUMINAMATH_CALUDE_solve_for_a_l2932_293215

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2932_293215


namespace NUMINAMATH_CALUDE_polynomial_decomposition_l2932_293223

theorem polynomial_decomposition (x : ℝ) : 
  1 + x^5 + x^10 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_decomposition_l2932_293223


namespace NUMINAMATH_CALUDE_unique_solution_l2932_293225

/-- Represents a three-digit number in the form ABA --/
def ABA (A B : ℕ) : ℕ := 100 * A + 10 * B + A

/-- Represents a four-digit number in the form CCDC --/
def CCDC (C D : ℕ) : ℕ := 1000 * C + 100 * C + 10 * D + C

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    (ABA A B)^2 = CCDC C D ∧
    CCDC C D < 100000 ∧
    A = 2 ∧ B = 1 ∧ C = 4 ∧ D = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2932_293225


namespace NUMINAMATH_CALUDE_expression_value_l2932_293268

theorem expression_value (a b c d m : ℝ) : 
  (a = -b) → (c * d = 1) → (abs m = 2) → 
  (3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = -8 ∨ 
   3 * (a + b - 1) + (-c * d) ^ 2023 - 2 * m = 0) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2932_293268


namespace NUMINAMATH_CALUDE_quincy_age_l2932_293298

/-- Given the ages of several people and their relationships, calculate Quincy's age -/
theorem quincy_age (kiarra bea job figaro quincy : ℝ) : 
  kiarra = 2 * bea →
  job = 3 * bea →
  figaro = job + 7 →
  kiarra = 30 →
  quincy = (job + figaro) / 2 →
  quincy = 48.5 := by
sorry

end NUMINAMATH_CALUDE_quincy_age_l2932_293298


namespace NUMINAMATH_CALUDE_compound_animals_l2932_293248

theorem compound_animals (dogs : ℕ) (cats : ℕ) (frogs : ℕ) : 
  cats = dogs - dogs / 5 →
  frogs = 2 * dogs →
  cats + dogs + frogs = 304 →
  frogs = 160 := by sorry

end NUMINAMATH_CALUDE_compound_animals_l2932_293248


namespace NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l2932_293273

/-- A color type with three possible values -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point -/
def Coloring := Point → Color

/-- Predicate to check if a triangle is right-angled -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = 
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2

/-- Main theorem -/
theorem exists_tricolor_right_triangle (coloring : Coloring) 
  (h1 : ∃ p : Point, coloring p = Color.Red)
  (h2 : ∃ p : Point, coloring p = Color.Green)
  (h3 : ∃ p : Point, coloring p = Color.Blue) :
  ∃ p1 p2 p3 : Point, 
    isRightTriangle p1 p2 p3 ∧ 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 :=
sorry

end NUMINAMATH_CALUDE_exists_tricolor_right_triangle_l2932_293273


namespace NUMINAMATH_CALUDE_mean_of_specific_numbers_l2932_293206

theorem mean_of_specific_numbers :
  let numbers : List ℝ := [12, 14, 16, 18]
  (numbers.sum / numbers.length : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_specific_numbers_l2932_293206


namespace NUMINAMATH_CALUDE_complement_of_A_l2932_293296

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

theorem complement_of_A : 
  (U \ A) = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2932_293296


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2932_293259

/-- A geometric sequence with the given property -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  arithmetic_property : 3 * a 1 + 2 * a 2 = a 3

/-- The main theorem -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  (seq.a 9 + seq.a 10) / (seq.a 7 + seq.a 8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2932_293259


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2932_293213

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron PQRS is 1715/(144√2) -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 3,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 15 / 4 * Real.sqrt 2
  }
  tetrahedronVolume t = 1715 / (144 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2932_293213


namespace NUMINAMATH_CALUDE_donut_theft_ratio_l2932_293260

theorem donut_theft_ratio (initial_donuts : ℕ) (bill_eaten : ℕ) (secretary_taken : ℕ) (final_donuts : ℕ)
  (h1 : initial_donuts = 50)
  (h2 : bill_eaten = 2)
  (h3 : secretary_taken = 4)
  (h4 : final_donuts = 22) :
  (initial_donuts - bill_eaten - secretary_taken - final_donuts) / (initial_donuts - bill_eaten - secretary_taken) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_donut_theft_ratio_l2932_293260


namespace NUMINAMATH_CALUDE_max_min_difference_l2932_293294

theorem max_min_difference (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  let f := fun (x y z : ℝ) => x*y + y*z + z*x
  ∃ (M m : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → f x y z ≤ M) ∧
               (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → m ≤ f x y z) ∧
               M - m = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l2932_293294


namespace NUMINAMATH_CALUDE_super_mindmaster_codes_l2932_293247

theorem super_mindmaster_codes (colors : ℕ) (slots : ℕ) : 
  colors = 9 → slots = 5 → colors ^ slots = 59049 := by
  sorry

end NUMINAMATH_CALUDE_super_mindmaster_codes_l2932_293247


namespace NUMINAMATH_CALUDE_percentage_same_grade_l2932_293283

/-- Represents the grade a student can receive -/
inductive Grade
| A
| B
| C
| D
| E

/-- Represents the grade distribution for a single test -/
structure GradeDistribution :=
  (A : Nat)
  (B : Nat)
  (C : Nat)
  (D : Nat)
  (E : Nat)

/-- The total number of students in the class -/
def totalStudents : Nat := 50

/-- The grade distribution for the first test -/
def firstTestDistribution : GradeDistribution := {
  A := 7,
  B := 12,
  C := 19,
  D := 8,
  E := 4
}

/-- The grade distribution for the second test -/
def secondTestDistribution : GradeDistribution := {
  A := 8,
  B := 16,
  C := 14,
  D := 7,
  E := 5
}

/-- The number of students who received the same grade on both tests -/
def sameGradeCount : Nat := 20

/-- Theorem: The percentage of students who received the same grade on both tests is 40% -/
theorem percentage_same_grade :
  (sameGradeCount : ℚ) / (totalStudents : ℚ) * 100 = 40 := by sorry

end NUMINAMATH_CALUDE_percentage_same_grade_l2932_293283


namespace NUMINAMATH_CALUDE_lcm_of_9_and_14_l2932_293252

theorem lcm_of_9_and_14 : Nat.lcm 9 14 = 126 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_and_14_l2932_293252


namespace NUMINAMATH_CALUDE_lauren_mail_count_l2932_293295

/-- The number of pieces of mail Lauren sent on Monday -/
def monday_mail : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday_mail : ℕ := monday_mail + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday_mail : ℕ := tuesday_mail - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday_mail : ℕ := wednesday_mail + 15

/-- The total number of pieces of mail Lauren sent over the four days -/
def total_mail : ℕ := monday_mail + tuesday_mail + wednesday_mail + thursday_mail

theorem lauren_mail_count : total_mail = 295 := by
  sorry

end NUMINAMATH_CALUDE_lauren_mail_count_l2932_293295


namespace NUMINAMATH_CALUDE_cube_sum_eq_product_squares_l2932_293232

theorem cube_sum_eq_product_squares (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_product_squares_l2932_293232


namespace NUMINAMATH_CALUDE_min_value_condition_l2932_293227

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

theorem min_value_condition (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f b x ≥ 1) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f b x = 1) ↔ 
  b = Real.sqrt 2 ∨ b = -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_condition_l2932_293227


namespace NUMINAMATH_CALUDE_equation_solution_l2932_293257

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2932_293257


namespace NUMINAMATH_CALUDE_union_equals_interval_l2932_293256

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B : Set ℝ := {y : ℝ | y ≥ 1}

-- Define the interval [-1, +∞)
def interval : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem stating that the union of A and B is equal to the interval [-1, +∞)
theorem union_equals_interval : A ∪ B = interval := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l2932_293256


namespace NUMINAMATH_CALUDE_fourteenth_root_unity_l2932_293288

theorem fourteenth_root_unity (n : ℕ) : 
  0 ≤ n ∧ n ≤ 13 → 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 14)) → 
  n = 4 := by sorry

end NUMINAMATH_CALUDE_fourteenth_root_unity_l2932_293288


namespace NUMINAMATH_CALUDE_honey_jar_problem_l2932_293210

/-- The proportion of honey remaining after each extraction -/
def remaining_proportion : ℚ := 75 / 100

/-- The number of times the extraction process is repeated -/
def num_extractions : ℕ := 6

/-- The amount of honey remaining after all extractions (in grams) -/
def final_honey : ℚ := 420

/-- Calculates the initial amount of honey given the final amount and extraction process -/
def initial_honey : ℚ := final_honey / remaining_proportion ^ num_extractions

theorem honey_jar_problem :
  initial_honey * remaining_proportion ^ num_extractions = final_honey :=
sorry

end NUMINAMATH_CALUDE_honey_jar_problem_l2932_293210


namespace NUMINAMATH_CALUDE_total_distance_is_410_l2932_293261

-- Define bird types and their speeds
structure Bird where
  name : String
  speed : ℝ
  flightTime : ℝ

-- Define constants
def headwind : ℝ := 5
def totalBirds : ℕ := 6

-- Define the list of birds
def birds : List Bird := [
  { name := "eagle", speed := 15, flightTime := 2.5 },
  { name := "falcon", speed := 46, flightTime := 2.5 },
  { name := "pelican", speed := 33, flightTime := 2.5 },
  { name := "hummingbird", speed := 30, flightTime := 2.5 },
  { name := "hawk", speed := 45, flightTime := 3 },
  { name := "swallow", speed := 25, flightTime := 1.5 }
]

-- Calculate actual distance traveled by a bird
def actualDistance (bird : Bird) : ℝ :=
  (bird.speed - headwind) * bird.flightTime

-- Calculate total distance traveled by all birds
def totalDistance : ℝ :=
  (birds.map actualDistance).sum

-- Theorem to prove
theorem total_distance_is_410 : totalDistance = 410 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_410_l2932_293261


namespace NUMINAMATH_CALUDE_bus_ticket_probability_l2932_293263

/-- Represents the lottery game with given parameters -/
structure LotteryGame where
  initialAmount : ℝ
  ticketCost : ℝ
  winProbability : ℝ
  prizeAmount : ℝ
  targetAmount : ℝ

/-- Calculates the probability of winning enough money to reach the target amount -/
noncomputable def winProbability (game : LotteryGame) : ℝ :=
  let p := game.winProbability
  let q := 1 - p
  (p^2 * (1 + 2*q)) / (1 - 2*p*q^2)

/-- Theorem stating the probability of winning the bus ticket -/
theorem bus_ticket_probability (game : LotteryGame) 
  (h1 : game.initialAmount = 20)
  (h2 : game.ticketCost = 10)
  (h3 : game.winProbability = 0.1)
  (h4 : game.prizeAmount = 30)
  (h5 : game.targetAmount = 45) :
  ∃ ε > 0, |winProbability game - 0.033| < ε :=
sorry

end NUMINAMATH_CALUDE_bus_ticket_probability_l2932_293263


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2932_293240

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_thirds_l2932_293240


namespace NUMINAMATH_CALUDE_mark_bill_calculation_l2932_293208

def original_bill : ℝ := 500
def first_late_charge_rate : ℝ := 0.02
def second_late_charge_rate : ℝ := 0.03

def final_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem mark_bill_calculation : final_amount = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_mark_bill_calculation_l2932_293208


namespace NUMINAMATH_CALUDE_dereks_current_dogs_l2932_293282

/-- Represents the number of dogs and cars Derek has at different ages -/
structure DereksPets where
  dogs_at_six : ℕ
  cars_at_six : ℕ
  cars_bought : ℕ
  current_dogs : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem dereks_current_dogs (d : DereksPets) 
  (h1 : d.dogs_at_six = 3 * d.cars_at_six)
  (h2 : d.dogs_at_six = 90)
  (h3 : d.cars_bought = 210)
  (h4 : d.cars_at_six + d.cars_bought = 2 * d.current_dogs) :
  d.current_dogs = 120 := by
  sorry

end NUMINAMATH_CALUDE_dereks_current_dogs_l2932_293282


namespace NUMINAMATH_CALUDE_mean_quiz_score_l2932_293244

def quiz_scores : List ℝ := [88, 90, 94, 86, 85, 91]

theorem mean_quiz_score : 
  (quiz_scores.sum / quiz_scores.length : ℝ) = 89 := by sorry

end NUMINAMATH_CALUDE_mean_quiz_score_l2932_293244


namespace NUMINAMATH_CALUDE_equation_solution_l2932_293230

theorem equation_solution :
  ∀ x : ℚ, x ≠ 3 → ((x + 5) / (x - 3) = 4 ↔ x = 17 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2932_293230


namespace NUMINAMATH_CALUDE_complex_division_result_l2932_293297

theorem complex_division_result : (1 + 2*I : ℂ) / I = 2 - I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l2932_293297


namespace NUMINAMATH_CALUDE_f_has_minimum_at_one_point_five_l2932_293284

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem f_has_minimum_at_one_point_five :
  ∃ (y : ℝ), ∀ (x : ℝ), f x ≥ f (3/2) := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_at_one_point_five_l2932_293284


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2932_293269

theorem sqrt_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 ∧ x = 1225 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2932_293269


namespace NUMINAMATH_CALUDE_seeds_per_medium_row_is_twenty_l2932_293279

/-- Represents the garden setup with large and medium beds -/
structure GardenSetup where
  largeBeds : Nat
  mediumBeds : Nat
  largeRowsPerBed : Nat
  mediumRowsPerBed : Nat
  seedsPerLargeRow : Nat
  totalSeeds : Nat

/-- Calculates the number of seeds per row in the medium bed -/
def seedsPerMediumRow (setup : GardenSetup) : Nat :=
  let largeSeeds := setup.largeBeds * setup.largeRowsPerBed * setup.seedsPerLargeRow
  let mediumSeeds := setup.totalSeeds - largeSeeds
  let totalMediumRows := setup.mediumBeds * setup.mediumRowsPerBed
  mediumSeeds / totalMediumRows

/-- Theorem stating that the number of seeds per row in the medium bed is 20 -/
theorem seeds_per_medium_row_is_twenty :
  let setup : GardenSetup := {
    largeBeds := 2,
    mediumBeds := 2,
    largeRowsPerBed := 4,
    mediumRowsPerBed := 3,
    seedsPerLargeRow := 25,
    totalSeeds := 320
  }
  seedsPerMediumRow setup = 20 := by sorry

end NUMINAMATH_CALUDE_seeds_per_medium_row_is_twenty_l2932_293279


namespace NUMINAMATH_CALUDE_arithmetic_sequence_40th_term_l2932_293207

/-- Given an arithmetic sequence where the first term is 3 and the twentieth term is 63,
    prove that the fortieth term is 126. -/
theorem arithmetic_sequence_40th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 3 →                                 -- first term is 3
    a 19 = 63 →                               -- twentieth term is 63
    a 39 = 126 := by                          -- fortieth term is 126
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_40th_term_l2932_293207


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2932_293222

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

/-- The shifted parabola function -/
def shifted_parabola (n : ℝ) (x : ℝ) : ℝ := original_parabola (x - n)

/-- Theorem stating the conditions and the result to be proved -/
theorem parabola_shift_theorem (n : ℝ) (y1 y2 : ℝ) 
  (h1 : n > 0)
  (h2 : shifted_parabola n 2 = y1)
  (h3 : shifted_parabola n 4 = y2)
  (h4 : y1 > y2) :
  n = 6 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2932_293222


namespace NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2932_293233

/-- Given that Nellie needs to sell 45 rolls of gift wrap in total and has already sold some, 
    prove that she needs to sell 28 more rolls. -/
theorem nellie_gift_wrap_sales (total_needed : ℕ) (sold_to_grandmother : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) 
    (h1 : total_needed = 45)
    (h2 : sold_to_grandmother = 1)
    (h3 : sold_to_uncle = 10)
    (h4 : sold_to_neighbor = 6) :
    total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2932_293233


namespace NUMINAMATH_CALUDE_min_fold_length_l2932_293234

/-- Given a rectangle ABCD with AB = 6 and AD = 12, when corner B is folded to edge AD
    creating a fold line MN, this function represents the length of MN (l)
    as a function of t, where t = sin θ and θ = �angle MNB -/
def fold_length (t : ℝ) : ℝ := 6 * t

/-- The theorem states that the minimum value of the fold length is 0 -/
theorem min_fold_length :
  ∃ (t : ℝ), t ≥ 0 ∧ t ≤ 1 ∧ ∀ (s : ℝ), s ≥ 0 → s ≤ 1 → fold_length t ≤ fold_length s :=
by sorry

end NUMINAMATH_CALUDE_min_fold_length_l2932_293234


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2932_293255

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (2 - i)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2932_293255


namespace NUMINAMATH_CALUDE_workshop_workers_l2932_293280

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (total_avg : ℕ) (tech_count : ℕ) (tech_avg : ℕ) (non_tech_avg : ℕ) :
  total_avg = 8000 →
  tech_count = 7 →
  tech_avg = 18000 →
  non_tech_avg = 6000 →
  ∃ (total_workers : ℕ), total_workers = 42 ∧
    total_workers * total_avg = tech_count * tech_avg + (total_workers - tech_count) * non_tech_avg :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2932_293280


namespace NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_in_sphere_l2932_293285

theorem max_surface_area_rectangular_solid_in_sphere :
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 + c^2 = 4) →
  2 * (a * b + a * c + b * c) ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_rectangular_solid_in_sphere_l2932_293285


namespace NUMINAMATH_CALUDE_M_equality_l2932_293278

theorem M_equality : 
  let M := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = (5/2) * Real.sqrt 2 - Real.sqrt 3 + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_M_equality_l2932_293278


namespace NUMINAMATH_CALUDE_zu_chongzhi_complex_theory_incorrect_l2932_293226

-- Define a structure for a scientist-field pairing
structure ScientistFieldPair where
  scientist : String
  field : String

-- Define the list of pairings
def pairings : List ScientistFieldPair := [
  { scientist := "Descartes", field := "Analytic Geometry" },
  { scientist := "Pascal", field := "Probability Theory" },
  { scientist := "Cantor", field := "Set Theory" },
  { scientist := "Zu Chongzhi", field := "Complex Number Theory" }
]

-- Define a function to check if a pairing is correct based on historical contributions
def isCorrectPairing (pair : ScientistFieldPair) : Bool :=
  match pair with
  | { scientist := "Descartes", field := "Analytic Geometry" } => true
  | { scientist := "Pascal", field := "Probability Theory" } => true
  | { scientist := "Cantor", field := "Set Theory" } => true
  | { scientist := "Zu Chongzhi", field := "Complex Number Theory" } => false
  | _ => false

-- Theorem: The pairing of Zu Chongzhi with Complex Number Theory is incorrect
theorem zu_chongzhi_complex_theory_incorrect :
  ∃ pair ∈ pairings, pair.scientist = "Zu Chongzhi" ∧ pair.field = "Complex Number Theory" ∧ ¬(isCorrectPairing pair) :=
by
  sorry

end NUMINAMATH_CALUDE_zu_chongzhi_complex_theory_incorrect_l2932_293226
