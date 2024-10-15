import Mathlib

namespace NUMINAMATH_CALUDE_angle_C_in_similar_triangles_l1612_161218

-- Define the triangles and their properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_C_in_similar_triangles (ABC DEF : Triangle) 
  (h1 : similar ABC DEF) (h2 : ABC.A = 30) (h3 : DEF.B = 30) : ABC.C = 120 := by
  sorry


end NUMINAMATH_CALUDE_angle_C_in_similar_triangles_l1612_161218


namespace NUMINAMATH_CALUDE_double_time_double_discount_l1612_161253

/-- Represents the true discount for a bill over a given time period. -/
structure TrueDiscount where
  bill : ℝ  -- Face value of the bill
  discount : ℝ  -- Amount of discount
  time : ℝ  -- Time period

/-- Calculates the true discount for a doubled time period. -/
def double_time_discount (td : TrueDiscount) : ℝ :=
  2 * td.discount

/-- Theorem stating that doubling the time period doubles the true discount. -/
theorem double_time_double_discount (td : TrueDiscount) 
  (h1 : td.bill = 110) 
  (h2 : td.discount = 10) :
  double_time_discount td = 20 := by
  sorry

#check double_time_double_discount

end NUMINAMATH_CALUDE_double_time_double_discount_l1612_161253


namespace NUMINAMATH_CALUDE_power_calculation_l1612_161289

theorem power_calculation : (16^4 * 8^6) / 4^14 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1612_161289


namespace NUMINAMATH_CALUDE_bicycle_problem_l1612_161205

theorem bicycle_problem (total_distance : ℝ) (walking_speed : ℝ) (cycling_speed : ℝ) 
  (h1 : total_distance = 20)
  (h2 : walking_speed = 4)
  (h3 : cycling_speed = 20) :
  ∃ (x : ℝ) (t : ℝ),
    0 < x ∧ x < total_distance ∧
    (x / cycling_speed + (total_distance - x) / walking_speed = 
     x / walking_speed + (total_distance - x) / cycling_speed) ∧
    x = 10 ∧
    t = 3 ∧
    t = x / cycling_speed + (total_distance - x) / walking_speed :=
by sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1612_161205


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1612_161215

theorem larger_integer_problem (x y : ℤ) 
  (h1 : y = 4 * x) 
  (h2 : (x + 12) / y = 1 / 2) : 
  y = 48 := by sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1612_161215


namespace NUMINAMATH_CALUDE_no_winning_strategy_l1612_161291

-- Define the game board
def GameBoard := Fin 99

-- Define a piece
structure Piece where
  number : Fin 99
  position : GameBoard

-- Define a player
inductive Player
| Jia
| Yi

-- Define the game state
structure GameState where
  board : List Piece
  currentPlayer : Player

-- Define a winning condition
def isWinningState (state : GameState) : Prop :=
  ∃ (i j k : GameBoard),
    (i.val + 1) % 99 = j.val ∧
    (j.val + 1) % 99 = k.val ∧
    ∃ (pi pj pk : Piece),
      pi ∈ state.board ∧ pj ∈ state.board ∧ pk ∈ state.board ∧
      pi.position = i ∧ pj.position = j ∧ pk.position = k ∧
      pj.number.val - pi.number.val = pk.number.val - pj.number.val

-- Define a strategy
def Strategy := GameState → Option Piece

-- Define the theorem
theorem no_winning_strategy :
  ¬∃ (s : Strategy), ∀ (opponent_strategy : Strategy),
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Yi ∧
      isWinningState final_state) ∨
    (∃ (n : ℕ) (final_state : GameState),
      final_state.currentPlayer = Player.Jia ∧
      isWinningState final_state) :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l1612_161291


namespace NUMINAMATH_CALUDE_seven_x_plus_four_is_odd_l1612_161252

theorem seven_x_plus_four_is_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_seven_x_plus_four_is_odd_l1612_161252


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l1612_161284

theorem absolute_value_nonnegative (a : ℝ) : ¬(|a| < 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l1612_161284


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1953_l1612_161264

theorem smallest_prime_factor_of_1953 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1953 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1953 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1953_l1612_161264


namespace NUMINAMATH_CALUDE_mikes_net_spent_l1612_161265

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating the net amount Mike spent at the music store -/
theorem mikes_net_spent :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spent_l1612_161265


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l1612_161244

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 12*x + k = 0 ∧ 
   y^2 + 12*y + k = 0 ∧
   x / y = 3) → k = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l1612_161244


namespace NUMINAMATH_CALUDE_cat_kittens_count_l1612_161260

def animal_shelter_problem (initial_cats : ℕ) (new_cats : ℕ) (adopted_cats : ℕ) (final_cats : ℕ) : ℕ :=
  let total_before_events := initial_cats + new_cats
  let after_adoption := total_before_events - adopted_cats
  final_cats - after_adoption + 1

theorem cat_kittens_count : animal_shelter_problem 6 12 3 19 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cat_kittens_count_l1612_161260


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161290

open Set

def A : Set ℝ := {x | 1 < x^2 ∧ x^2 < 4}
def B : Set ℝ := {x | x - 1 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161290


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1612_161222

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l1612_161222


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1612_161239

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) -- Points in 2D plane
  (is_right_triangle : (Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2) = 0) -- Right angle condition
  (cos_R : ((Q.1 - R.1) * (P.1 - R.1) + (Q.2 - R.2) * (P.2 - R.2)) / 
           (Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) * Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) = 3/5) -- cos R = 3/5
  (RP_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 10) -- RP = 10
  : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 8 := by -- PQ = 8
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1612_161239


namespace NUMINAMATH_CALUDE_max_value_of_f_l1612_161267

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1612_161267


namespace NUMINAMATH_CALUDE_cubic_root_difference_l1612_161233

/-- The cubic equation x³ - px² + (p² - 1)/4x = 0 has a difference of 1 between its largest and smallest roots -/
theorem cubic_root_difference (p : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - p*x^2 + (p^2 - 1)/4*x
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ ∀ c ∈ roots, a ≤ c ∧ c ≤ b ∧ b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l1612_161233


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l1612_161255

def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_of_S_and_T : S ∩ T = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l1612_161255


namespace NUMINAMATH_CALUDE_expression_value_l1612_161204

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1612_161204


namespace NUMINAMATH_CALUDE_star_arrangement_count_l1612_161256

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct shells to be placed -/
def num_shells : ℕ := 12

/-- The number of distinct arrangements of shells on a regular six-pointed star,
    considering rotational and reflectional symmetries -/
def distinct_arrangements : ℕ := Nat.factorial num_shells / star_symmetries

theorem star_arrangement_count :
  distinct_arrangements = 39916800 := by sorry

end NUMINAMATH_CALUDE_star_arrangement_count_l1612_161256


namespace NUMINAMATH_CALUDE_spare_time_is_five_hours_l1612_161236

/-- Calculates the spare time for painting a room given the following conditions:
  * The room has 5 walls
  * Each wall is 2 meters by 3 meters
  * The painter can paint 1 square meter every 10 minutes
  * The painter has 10 hours to paint everything
-/
def spare_time_for_painting : ℕ :=
  let num_walls : ℕ := 5
  let wall_width : ℕ := 2
  let wall_height : ℕ := 3
  let painting_rate : ℕ := 10  -- minutes per square meter
  let total_time : ℕ := 10 * 60  -- total time in minutes

  let wall_area : ℕ := wall_width * wall_height
  let total_area : ℕ := num_walls * wall_area
  let painting_time : ℕ := total_area * painting_rate
  let spare_time_minutes : ℕ := total_time - painting_time
  spare_time_minutes / 60

theorem spare_time_is_five_hours : spare_time_for_painting = 5 := by
  sorry

end NUMINAMATH_CALUDE_spare_time_is_five_hours_l1612_161236


namespace NUMINAMATH_CALUDE_value_calculation_l1612_161299

theorem value_calculation (n : ℝ) (v : ℝ) (h : n = 50) : 0.20 * n - 4 = v → v = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l1612_161299


namespace NUMINAMATH_CALUDE_ratio_subtraction_l1612_161262

theorem ratio_subtraction (x : ℕ) : 
  x = 3 ∧ 
  (6 - x : ℚ) / (7 - x) < 16 / 21 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 → 
  6 = 6 :=
sorry

end NUMINAMATH_CALUDE_ratio_subtraction_l1612_161262


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_decrease_l1612_161231

/-- Proves that the decrease in average expenditure per head is 1 rupee
    given the initial conditions of the hostel mess problem. -/
theorem hostel_mess_expenditure_decrease :
  let initial_students : ℕ := 35
  let new_students : ℕ := 7
  let total_students : ℕ := initial_students + new_students
  let initial_expenditure : ℕ := 420
  let expenditure_increase : ℕ := 42
  let new_expenditure : ℕ := initial_expenditure + expenditure_increase
  let initial_average : ℚ := initial_expenditure / initial_students
  let new_average : ℚ := new_expenditure / total_students
  initial_average - new_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_decrease_l1612_161231


namespace NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1612_161270

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 8

/-- The number of packages Katie could make after Todd ate some cupcakes -/
def packages : ℕ := 5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 2

/-- The initial number of cupcakes Katie baked -/
def initial_cupcakes : ℕ := todd_ate + packages * cupcakes_per_package

theorem katie_baked_18_cupcakes : initial_cupcakes = 18 := by
  sorry

end NUMINAMATH_CALUDE_katie_baked_18_cupcakes_l1612_161270


namespace NUMINAMATH_CALUDE_broken_cone_height_l1612_161278

/-- Theorem: New height of a broken cone -/
theorem broken_cone_height (r : ℝ) (l : ℝ) (l_new : ℝ) (H : ℝ) :
  r = 6 →
  l = 13 →
  l_new = l - 2 →
  H^2 + r^2 = l_new^2 →
  H = Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_broken_cone_height_l1612_161278


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1612_161228

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1612_161228


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_l1612_161209

/-- Given three consecutive even integers and a condition, prove the value of k. -/
theorem consecutive_even_numbers (N₁ N₂ N₃ k : ℤ) : 
  N₃ = 19 →
  N₂ = N₁ + 2 →
  N₃ = N₂ + 2 →
  3 * N₁ = k * N₃ + 7 →
  k = 2 := by
sorry


end NUMINAMATH_CALUDE_consecutive_even_numbers_l1612_161209


namespace NUMINAMATH_CALUDE_preimage_of_one_seven_l1612_161242

/-- The mapping function from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (4, -3) is the preimage of (1, 7) under f -/
theorem preimage_of_one_seven :
  f (4, -3) = (1, 7) ∧ 
  ∀ p : ℝ × ℝ, f p = (1, 7) → p = (4, -3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_seven_l1612_161242


namespace NUMINAMATH_CALUDE_min_value_theorem_l1612_161282

theorem min_value_theorem (a b : ℝ) (hb : b > 0) (h : a + 2*b = 1) :
  (3/b) + (1/a) ≥ 7 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1612_161282


namespace NUMINAMATH_CALUDE_circle_passes_through_focus_l1612_161206

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 8 * p.1

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency of a circle to a line
def tangent_to_line (c : Circle) : Prop :=
  c.radius = |c.center.1 + 2|

-- Main theorem
theorem circle_passes_through_focus :
  ∀ c : Circle,
  parabola c.center →
  tangent_to_line c →
  c.center.1^2 + c.center.2^2 = (2 - c.center.1)^2 + c.center.2^2 :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_focus_l1612_161206


namespace NUMINAMATH_CALUDE_shaded_area_squares_l1612_161250

theorem shaded_area_squares (large_side small_side : ℝ) 
  (h1 : large_side = 14) 
  (h2 : small_side = 10) : 
  (large_side^2 - small_side^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_squares_l1612_161250


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l1612_161276

/-- The sum of positive divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors function -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors :
  let n : ℕ := 450
  let sum_of_divisors : ℕ := sigma n
  num_distinct_prime_factors sum_of_divisors = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l1612_161276


namespace NUMINAMATH_CALUDE_pie_distribution_l1612_161226

/-- Given a pie with 48 slices, prove that after distributing specific fractions, 2 slices remain -/
theorem pie_distribution (total_slices : ℕ) (joe_fraction darcy_fraction carl_fraction emily_fraction frank_percent : ℚ) : 
  total_slices = 48 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  frank_percent = 10/100 →
  total_slices - (total_slices * joe_fraction).floor - (total_slices * darcy_fraction).floor - 
  (total_slices * carl_fraction).floor - (total_slices * emily_fraction).floor - 
  (total_slices * frank_percent).floor = 2 := by
sorry

end NUMINAMATH_CALUDE_pie_distribution_l1612_161226


namespace NUMINAMATH_CALUDE_postcard_problem_l1612_161279

theorem postcard_problem (initial_postcards : ℕ) : 
  (initial_postcards / 2 + (initial_postcards / 2) * 3 = 36) → 
  initial_postcards = 18 := by
  sorry

end NUMINAMATH_CALUDE_postcard_problem_l1612_161279


namespace NUMINAMATH_CALUDE_divisibility_condition_solutions_l1612_161275

theorem divisibility_condition_solutions (n p : ℕ) (h_prime : Nat.Prime p) (h_range : 0 < n ∧ n ≤ 2 * p) :
  n^(p-1) ∣ (p-1)^n + 1 ↔ 
    (n = 1 ∧ p ≥ 2) ∨
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_solutions_l1612_161275


namespace NUMINAMATH_CALUDE_digits_of_product_l1612_161254

theorem digits_of_product : ∃ (n : ℕ), n = 3^4 * 6^8 ∧ (Nat.log 10 n + 1 = 9) := by sorry

end NUMINAMATH_CALUDE_digits_of_product_l1612_161254


namespace NUMINAMATH_CALUDE_yulia_lemonade_expenses_l1612_161232

/-- Represents the financial data for Yulia's earnings --/
structure YuliaFinances where
  net_profit : ℝ
  lemonade_revenue : ℝ
  babysitting_earnings : ℝ

/-- Calculates the expenses for operating the lemonade stand --/
def lemonade_expenses (finances : YuliaFinances) : ℝ :=
  finances.lemonade_revenue + finances.babysitting_earnings - finances.net_profit

/-- Theorem stating that Yulia's lemonade stand expenses are $34 --/
theorem yulia_lemonade_expenses :
  let finances : YuliaFinances := {
    net_profit := 44,
    lemonade_revenue := 47,
    babysitting_earnings := 31
  }
  lemonade_expenses finances = 34 := by
  sorry

end NUMINAMATH_CALUDE_yulia_lemonade_expenses_l1612_161232


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l1612_161217

theorem rectangular_solid_width (length depth surface_area : ℝ) : 
  length = 10 →
  depth = 6 →
  surface_area = 408 →
  surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth →
  width = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l1612_161217


namespace NUMINAMATH_CALUDE_philips_farm_animals_l1612_161258

/-- Represents the number of animals on Philip's farm -/
structure FarmAnimals where
  cows : ℕ
  ducks : ℕ
  horses : ℕ
  pigs : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.ducks + farm.horses + farm.pigs + farm.chickens

/-- Theorem stating the total number of animals on Philip's farm -/
theorem philips_farm_animals :
  ∃ (farm : FarmAnimals),
    farm.cows = 20 ∧
    farm.ducks = farm.cows + farm.cows / 2 ∧
    farm.horses = (farm.cows + farm.ducks) / 5 ∧
    farm.pigs = (farm.cows + farm.ducks + farm.horses) / 5 ∧
    farm.chickens = 3 * (farm.cows - farm.horses) ∧
    total_animals farm = 102 := by
  sorry


end NUMINAMATH_CALUDE_philips_farm_animals_l1612_161258


namespace NUMINAMATH_CALUDE_triangle_area_l1612_161251

theorem triangle_area (base height : ℝ) (h1 : base = 3) (h2 : height = 4) :
  (base * height) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1612_161251


namespace NUMINAMATH_CALUDE_total_project_hours_l1612_161225

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ := 
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 65
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours : 
  ∃ (kate_hours : ℝ), 
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧ 
    pat_hours + kate_hours + mark_hours = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l1612_161225


namespace NUMINAMATH_CALUDE_equation_solution_l1612_161295

theorem equation_solution : 
  ∃! x : ℝ, (2 : ℝ) / (x + 3) + (3 * x) / (x + 3) - (5 : ℝ) / (x + 3) = 4 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1612_161295


namespace NUMINAMATH_CALUDE_min_sum_distances_l1612_161210

/-- Given a parabola and a line, prove the minimum sum of distances -/
theorem min_sum_distances (x y : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8*x}
  let line := {(x, y) : ℝ × ℝ | x - y + 2 = 0}
  let d1 (p : ℝ × ℝ) := |p.1|  -- distance from point to y-axis
  let d2 (p : ℝ × ℝ) := |p.1 - p.2 + 2| / Real.sqrt 2  -- distance from point to line
  ∃ (min : ℝ), ∀ (p : ℝ × ℝ), p ∈ parabola → d1 p + d2 p ≥ min ∧ 
  ∃ (q : ℝ × ℝ), q ∈ parabola ∧ d1 q + d2 q = min ∧ min = 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1612_161210


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1612_161285

/-- Given two vectors a and b in ℝ², where a = (x - 1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1612_161285


namespace NUMINAMATH_CALUDE_circle_center_l1612_161298

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 5) ∧ h = 1 ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1612_161298


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l1612_161200

theorem unique_solution_for_all_y :
  ∃! x : ℝ, ∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l1612_161200


namespace NUMINAMATH_CALUDE_equation_solution_l1612_161261

theorem equation_solution :
  ∃ x : ℝ, 4 * (x - 2) * (x + 5) = (2 * x - 3) * (2 * x + 11) + 11 ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1612_161261


namespace NUMINAMATH_CALUDE_ice_cream_volume_specific_ice_cream_volume_l1612_161286

/-- The volume of ice cream in a right circular cone with a hemisphere on top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

/-- The specific case with h = 12 and r = 4 -/
theorem specific_ice_cream_volume :
  let h : ℝ := 12
  let r : ℝ := 4
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (2/3) * π * r^3
  cone_volume + hemisphere_volume = (320/3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_specific_ice_cream_volume_l1612_161286


namespace NUMINAMATH_CALUDE_library_visits_l1612_161274

/-- Proves that William goes to the library 2 times per week given the conditions -/
theorem library_visits (jason_freq : ℕ) (william_freq : ℕ) (jason_total : ℕ) (weeks : ℕ) :
  jason_freq = 4 * william_freq →
  jason_total = 32 →
  weeks = 4 →
  jason_total = jason_freq * weeks →
  william_freq = 2 := by
  sorry

end NUMINAMATH_CALUDE_library_visits_l1612_161274


namespace NUMINAMATH_CALUDE_stating_non_parallel_necessary_not_sufficient_l1612_161263

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Represents a system of two linear equations -/
structure LinearSystem where
  line1 : Line
  line2 : Line

/-- Checks if a linear system has a unique solution -/
def has_unique_solution (sys : LinearSystem) : Prop :=
  sys.line1.a * sys.line2.b ≠ sys.line1.b * sys.line2.a

/-- 
Theorem stating that non-parallel lines are a necessary but insufficient condition
for a system of two linear equations to have a unique solution
-/
theorem non_parallel_necessary_not_sufficient (sys : LinearSystem) :
  has_unique_solution sys → ¬(are_parallel sys.line1 sys.line2) ∧
  ¬(¬(are_parallel sys.line1 sys.line2) → has_unique_solution sys) :=
by sorry

end NUMINAMATH_CALUDE_stating_non_parallel_necessary_not_sufficient_l1612_161263


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1612_161241

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1612_161241


namespace NUMINAMATH_CALUDE_multiplication_associative_l1612_161281

theorem multiplication_associative (x y z : ℝ) : (x * y) * z = x * (y * z) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_associative_l1612_161281


namespace NUMINAMATH_CALUDE_triangle_property_l1612_161249

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c →
  a = 4 →
  A = π / 3 ∧ (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (Real.sin A + Real.sin B) * (a - b') = (Real.sin C - Real.sin B) * c' →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1612_161249


namespace NUMINAMATH_CALUDE_scientific_notation_120_million_l1612_161243

theorem scientific_notation_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_120_million_l1612_161243


namespace NUMINAMATH_CALUDE_solution_set_equality_l1612_161277

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | x ≠ 0 ∧ 1 / x < 1 / 2}

-- State the theorem
theorem solution_set_equality : solution_set = Set.Ioi 2 ∪ Set.Iio 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l1612_161277


namespace NUMINAMATH_CALUDE_initial_saree_purchase_l1612_161280

/-- The number of sarees in the initial purchase -/
def num_sarees : ℕ := 2

/-- The price of one saree -/
def saree_price : ℕ := 400

/-- The price of one shirt -/
def shirt_price : ℕ := 200

/-- Theorem stating that the number of sarees in the initial purchase is 2 -/
theorem initial_saree_purchase : 
  (∃ (X : ℕ), X * saree_price + 4 * shirt_price = 1600) ∧ 
  (saree_price + 6 * shirt_price = 1600) ∧
  (12 * shirt_price = 2400) →
  num_sarees = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_saree_purchase_l1612_161280


namespace NUMINAMATH_CALUDE_balloon_arrangements_l1612_161246

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedTwice : ℕ) (appearOnce : ℕ) : ℕ :=
  Nat.factorial totalLetters / (2^repeatedTwice * Nat.factorial appearOnce)

/-- Theorem: The number of distinct arrangements of letters in a word with 7 letters,
    where two letters each appear twice and three letters appear once, is equal to 1260 -/
theorem balloon_arrangements :
  distinctArrangements 7 2 3 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l1612_161246


namespace NUMINAMATH_CALUDE_suraj_average_l1612_161213

theorem suraj_average (initial_average : ℝ) (innings : ℕ) (new_score : ℝ) (average_increase : ℝ) : 
  innings = 14 →
  new_score = 140 →
  average_increase = 8 →
  (innings * initial_average + new_score) / (innings + 1) = initial_average + average_increase →
  initial_average + average_increase = 28 :=
by sorry

end NUMINAMATH_CALUDE_suraj_average_l1612_161213


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1612_161216

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 - 3 * Complex.I) :
  z.im = 5/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1612_161216


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1612_161259

/-- Given a school with 2000 students, of which 700 are sophomores,
    and a stratified sample of 100 students, the number of sophomores
    in the sample should be 35. -/
theorem stratified_sampling_sophomores :
  ∀ (total_students sample_size num_sophomores : ℕ),
    total_students = 2000 →
    sample_size = 100 →
    num_sophomores = 700 →
    (num_sophomores * sample_size) / total_students = 35 :=
by
  sorry

#check stratified_sampling_sophomores

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1612_161259


namespace NUMINAMATH_CALUDE_total_kites_sold_l1612_161220

-- Define the sequence
def kite_sequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

-- Define the sum of the sequence
def kite_sum (n : ℕ) : ℕ := 
  n * (kite_sequence 1 + kite_sequence n) / 2

-- Theorem statement
theorem total_kites_sold : kite_sum 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_total_kites_sold_l1612_161220


namespace NUMINAMATH_CALUDE_neighbor_cans_is_46_l1612_161227

/-- Represents the recycling problem Collin faces --/
structure RecyclingProblem where
  /-- The amount earned per aluminum can in dollars --/
  earnings_per_can : ℚ
  /-- The number of cans found at home --/
  cans_at_home : ℕ
  /-- The factor by which the number of cans at grandparents' house exceeds those at home --/
  grandparents_factor : ℕ
  /-- The number of cans brought by dad from the office --/
  cans_from_office : ℕ
  /-- The amount Collin has to put into savings in dollars --/
  savings_amount : ℚ

/-- Calculates the number of cans Collin's neighbor gave him --/
def neighbor_cans (p : RecyclingProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of cans Collin's neighbor gave him is 46 --/
theorem neighbor_cans_is_46 (p : RecyclingProblem)
  (h1 : p.earnings_per_can = 1/4)
  (h2 : p.cans_at_home = 12)
  (h3 : p.grandparents_factor = 3)
  (h4 : p.cans_from_office = 250)
  (h5 : p.savings_amount = 43) :
  neighbor_cans p = 46 :=
  sorry

end NUMINAMATH_CALUDE_neighbor_cans_is_46_l1612_161227


namespace NUMINAMATH_CALUDE_inequality_proof_l1612_161272

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * (a^a * b^b * c^c * d^d) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1612_161272


namespace NUMINAMATH_CALUDE_range_of_ab_l1612_161219

-- Define the polynomial
def P (a b x : ℝ) : ℝ := (x^2 - a*x + 1) * (x^2 - b*x + 1)

-- State the theorem
theorem range_of_ab (a b : ℝ) (q : ℝ) (h_q : q ∈ Set.Icc (1/3) 2) 
  (h_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (P a b r₁ = 0 ∧ P a b r₂ = 0 ∧ P a b r₃ = 0 ∧ P a b r₄ = 0) ∧ 
    (∃ (m : ℝ), r₁ = m ∧ r₂ = m*q ∧ r₃ = m*q^2 ∧ r₄ = m*q^3)) :
  a * b ∈ Set.Icc 4 (112/9) :=
sorry

end NUMINAMATH_CALUDE_range_of_ab_l1612_161219


namespace NUMINAMATH_CALUDE_shirt_pricing_l1612_161266

theorem shirt_pricing (total_shirts : Nat) (price_shirt1 price_shirt2 : ℝ) (min_avg_price_remaining : ℝ) :
  total_shirts = 5 →
  price_shirt1 = 30 →
  price_shirt2 = 20 →
  min_avg_price_remaining = 33.333333333333336 →
  (price_shirt1 + price_shirt2 + (total_shirts - 2) * min_avg_price_remaining) / total_shirts = 30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_pricing_l1612_161266


namespace NUMINAMATH_CALUDE_bobbys_candy_problem_l1612_161203

/-- The problem of Bobby's candy consumption -/
theorem bobbys_candy_problem (initial_candy : ℕ) : 
  initial_candy + 42 = 70 → initial_candy = 28 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_candy_problem_l1612_161203


namespace NUMINAMATH_CALUDE_prob_black_fourth_draw_l1612_161223

structure Box where
  red_balls : ℕ
  black_balls : ℕ

def initial_box : Box := { red_balls := 3, black_balls := 3 }

def total_balls (b : Box) : ℕ := b.red_balls + b.black_balls

def prob_black_first_draw (b : Box) : ℚ :=
  b.black_balls / (total_balls b)

theorem prob_black_fourth_draw (b : Box) :
  prob_black_first_draw b = 1/2 →
  (∃ (p : ℚ), p = prob_black_first_draw b ∧ p = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_prob_black_fourth_draw_l1612_161223


namespace NUMINAMATH_CALUDE_function_periodicity_l1612_161237

variable (a : ℝ)
variable (f : ℝ → ℝ)

theorem function_periodicity
  (h : ∀ x, f (x + a) = (1 + f x) / (1 - f x)) :
  ∀ x, f (x + 4 * a) = f x :=
by sorry

end NUMINAMATH_CALUDE_function_periodicity_l1612_161237


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l1612_161248

theorem propositions_p_and_q : 
  (∃ a b c : ℝ, a < b ∧ a * c^2 ≥ b * c^2) ∧ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l1612_161248


namespace NUMINAMATH_CALUDE_arrangements_with_three_together_eq_36_l1612_161234

/-- The number of different arrangements of five students in a row,
    where three specific students must be together. -/
def arrangements_with_three_together : ℕ :=
  (3 : ℕ).factorial * (3 : ℕ).factorial

theorem arrangements_with_three_together_eq_36 :
  arrangements_with_three_together = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_three_together_eq_36_l1612_161234


namespace NUMINAMATH_CALUDE_prime_extension_l1612_161235

theorem prime_extension (n : ℕ+) (h : ∀ k : ℕ, 0 ≤ k ∧ k < Real.sqrt ((n + 2) / 3) → Nat.Prime (k^2 + k + n + 2)) :
  ∀ k : ℕ, Real.sqrt ((n + 2) / 3) ≤ k ∧ k ≤ n → Nat.Prime (k^2 + k + n + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l1612_161235


namespace NUMINAMATH_CALUDE_imaginary_root_cubic_equation_l1612_161293

theorem imaginary_root_cubic_equation (a b q r : ℝ) :
  b ≠ 0 →
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ x = a + b*Complex.I) →
  q = b^2 - 3*a^2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_root_cubic_equation_l1612_161293


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l1612_161221

def f (x : ℝ) := x^3 + 5*x^2 + 8*x + 4

theorem cubic_inequality_solution :
  {x : ℝ | f x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l1612_161221


namespace NUMINAMATH_CALUDE_dog_barking_problem_l1612_161240

/-- Given the barking patterns of two dogs and the owner's hushing behavior, 
    calculate the number of times the owner said "hush". -/
theorem dog_barking_problem (poodle_barks terrier_barks owner_hushes : ℕ) : 
  poodle_barks = 24 →
  poodle_barks = 2 * terrier_barks →
  owner_hushes * 2 = terrier_barks →
  owner_hushes = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_barking_problem_l1612_161240


namespace NUMINAMATH_CALUDE_no_valid_acute_triangle_l1612_161212

def is_valid_angle (α : ℕ) : Prop :=
  α % 10 = 0 ∧ α ≠ 30 ∧ α ≠ 60 ∧ α > 0 ∧ α < 90

def is_acute_triangle (α β γ : ℕ) : Prop :=
  α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90

theorem no_valid_acute_triangle :
  ¬ ∃ (α β γ : ℕ), is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧
  is_acute_triangle α β γ ∧ α ≠ β ∧ β ≠ γ ∧ α ≠ γ :=
sorry

end NUMINAMATH_CALUDE_no_valid_acute_triangle_l1612_161212


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161245

def A : Set ℝ := {-1, 0, (1/2 : ℝ), 3}
def B : Set ℝ := {x : ℝ | x^2 ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1612_161245


namespace NUMINAMATH_CALUDE_fountain_water_after_25_days_l1612_161292

def fountain_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (rain_interval : ℕ) (rain_amount : ℝ) (days : ℕ) : ℝ :=
  let total_evaporation := evaporation_rate * days
  let rain_events := days / rain_interval
  let total_rain := rain_events * rain_amount
  initial_volume + total_rain - total_evaporation

theorem fountain_water_after_25_days :
  fountain_water_volume 120 0.8 5 5 25 = 125 := by sorry

end NUMINAMATH_CALUDE_fountain_water_after_25_days_l1612_161292


namespace NUMINAMATH_CALUDE_hyperbola_property_l1612_161207

/-- The hyperbola with equation x²/9 - y²/4 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 9) - (p.2^2 / 4) = 1}

/-- Left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the right branch of the hyperbola -/
def A : ℝ × ℝ := sorry

/-- Origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Point P such that 2 * OP = OA + OF₁ -/
def P : ℝ × ℝ := sorry

/-- Point Q such that 2 * OQ = OA + OF₂ -/
def Q : ℝ × ℝ := sorry

theorem hyperbola_property (h₁ : A ∈ Hyperbola)
    (h₂ : 2 • (P - O) = (A - O) + (F₁ - O))
    (h₃ : 2 • (Q - O) = (A - O) + (F₂ - O)) :
  ‖Q - O‖ - ‖P - O‖ = 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_property_l1612_161207


namespace NUMINAMATH_CALUDE_figure_area_bound_l1612_161247

-- Define the unit square
def UnitSquare : Set (Real × Real) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the property of the figure
def ValidFigure (F : Set (Real × Real)) : Prop :=
  F ⊆ UnitSquare ∧
  ∀ p q : Real × Real, p ∈ F → q ∈ F → dist p q ≠ 0.001

-- Define the area of a set
noncomputable def area (S : Set (Real × Real)) : Real :=
  sorry

-- State the theorem
theorem figure_area_bound {F : Set (Real × Real)} (hF : ValidFigure F) :
  area F ≤ 0.34 ∧ area F ≤ 0.287 :=
sorry

end NUMINAMATH_CALUDE_figure_area_bound_l1612_161247


namespace NUMINAMATH_CALUDE_circles_common_chord_l1612_161202

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

end NUMINAMATH_CALUDE_circles_common_chord_l1612_161202


namespace NUMINAMATH_CALUDE_point_coordinates_sum_l1612_161287

theorem point_coordinates_sum (X Y Z : ℝ × ℝ) : 
  (X.1 - Z.1) / (X.1 - Y.1) = 1/2 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/2 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/2 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/2 →
  Y = (2, 5) →
  Z = (1, -3) →
  X.1 + X.2 = -11 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_sum_l1612_161287


namespace NUMINAMATH_CALUDE_star_polygon_interior_angles_sum_l1612_161224

/-- A star polygon with n angles -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 5

/-- The sum of interior angles of a star polygon -/
def sum_interior_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem: The sum of interior angles of a star polygon is 180° * (n - 4) -/
theorem star_polygon_interior_angles_sum (sp : StarPolygon) :
  sum_interior_angles sp = 180 * (sp.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_interior_angles_sum_l1612_161224


namespace NUMINAMATH_CALUDE_union_of_sets_l1612_161211

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1612_161211


namespace NUMINAMATH_CALUDE_selection_theorem_l1612_161288

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls
def num_to_select : ℕ := 4

theorem selection_theorem : 
  (Nat.choose total_people num_to_select) - (Nat.choose num_boys num_to_select) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1612_161288


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l1612_161229

theorem square_inequality_for_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l1612_161229


namespace NUMINAMATH_CALUDE_flood_probability_l1612_161268

theorem flood_probability (p_30 p_40 : ℝ) 
  (h1 : p_30 = 0.8) 
  (h2 : p_40 = 0.85) : 
  (p_40 - p_30) / (1 - p_30) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_flood_probability_l1612_161268


namespace NUMINAMATH_CALUDE_evaluate_expression_l1612_161214

theorem evaluate_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + y^2) = 4 * Q + 10 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1612_161214


namespace NUMINAMATH_CALUDE_symmetric_function_max_value_l1612_161296

/-- Given a function f(x) = (1-x^2)(x^2 + ax + b) that is symmetric about x = -2,
    prove that its maximum value is 16. -/
theorem symmetric_function_max_value
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - x^2) * (x^2 + a*x + b))
  (h_sym : ∀ x, f (x + (-2)) = f ((-2) - x)) :
  ∃ x, f x = 16 ∧ ∀ y, f y ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_max_value_l1612_161296


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1612_161294

/-- The ratio of the volume of a sphere with radius 3q to the volume of a hemisphere with radius q is 54 -/
theorem sphere_hemisphere_volume_ratio (q : ℝ) (q_pos : 0 < q) : 
  (4 / 3 * Real.pi * (3 * q)^3) / ((1 / 2) * (4 / 3 * Real.pi * q^3)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1612_161294


namespace NUMINAMATH_CALUDE_petya_max_win_margin_l1612_161297

theorem petya_max_win_margin :
  ∀ (p1 p2 v1 v2 : ℕ),
    p1 + p2 + v1 + v2 = 27 →
    p1 = v1 + 9 →
    v2 = p2 + 9 →
    p1 + p2 > v1 + v2 →
    p1 + p2 - (v1 + v2) ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_petya_max_win_margin_l1612_161297


namespace NUMINAMATH_CALUDE_equation_simplification_l1612_161230

theorem equation_simplification (Y : ℝ) : ((3.242 * 10 * Y) / 100) = 0.3242 * Y := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l1612_161230


namespace NUMINAMATH_CALUDE_triangle_third_side_validity_l1612_161283

theorem triangle_third_side_validity (a b c : ℝ) : 
  a = 4 → b = 10 → c = 11 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_validity_l1612_161283


namespace NUMINAMATH_CALUDE_article_sale_loss_percent_l1612_161271

/-- Theorem: Given an article with a 35% gain at its original selling price,
    when sold at 2/3 of the original price, the loss percent is 10%. -/
theorem article_sale_loss_percent (cost_price : ℝ) (original_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  let new_price := (2 / 3) * original_price
  let loss := cost_price - new_price
  let loss_percent := (loss / cost_price) * 100
  loss_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_article_sale_loss_percent_l1612_161271


namespace NUMINAMATH_CALUDE_max_product_of_primes_sum_l1612_161238

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

theorem max_product_of_primes_sum (p : List ℕ) (h : p = primes) :
  ∃ (a b c d e f g h : ℕ),
    a ∈ p ∧ b ∈ p ∧ c ∈ p ∧ d ∈ p ∧ e ∈ p ∧ f ∈ p ∧ g ∈ p ∧ h ∈ p ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    (a + b + c + d) * (e + f + g + h) = 1480 ∧
    ∀ (a' b' c' d' e' f' g' h' : ℕ),
      a' ∈ p → b' ∈ p → c' ∈ p → d' ∈ p → e' ∈ p → f' ∈ p → g' ∈ p → h' ∈ p →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' → a' ≠ f' → a' ≠ g' → a' ≠ h' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' → b' ≠ f' → b' ≠ g' → b' ≠ h' →
      c' ≠ d' → c' ≠ e' → c' ≠ f' → c' ≠ g' → c' ≠ h' →
      d' ≠ e' → d' ≠ f' → d' ≠ g' → d' ≠ h' →
      e' ≠ f' → e' ≠ g' → e' ≠ h' →
      f' ≠ g' → f' ≠ h' →
      g' ≠ h' →
      (a' + b' + c' + d') * (e' + f' + g' + h') ≤ 1480 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_primes_sum_l1612_161238


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l1612_161201

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l1612_161201


namespace NUMINAMATH_CALUDE_fraction_equality_l1612_161208

theorem fraction_equality : (1632^2 - 1625^2) / (1645^2 - 1612^2) = 7/33 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1612_161208


namespace NUMINAMATH_CALUDE_a_5_of_1034_is_5_l1612_161273

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Factorial base representation -/
def factorial_base_rep (n : ℕ) : List ℕ :=
  sorry

/-- The 5th coefficient in the factorial base representation -/
def a_5 (n : ℕ) : ℕ :=
  match factorial_base_rep n with
  | b₁ :: b₂ :: b₃ :: b₄ :: b₅ :: _ => b₅
  | _ => 0  -- Default case if the list is too short

/-- Theorem stating that the 5th coefficient of 1034 in factorial base is 5 -/
theorem a_5_of_1034_is_5 : a_5 1034 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_5_of_1034_is_5_l1612_161273


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1612_161257

theorem complex_equation_solution (z : ℂ) : (2 * Complex.I) / z = 1 - Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1612_161257


namespace NUMINAMATH_CALUDE_cosine_function_property_l1612_161269

/-- Given a cosine function with specific properties, prove that its angular frequency is 2. -/
theorem cosine_function_property (f : ℝ → ℝ) (ω φ : ℝ) (h_ω_pos : ω > 0) (h_φ_bound : |φ| ≤ π/2) 
  (h_f_def : ∀ x, f x = Real.sqrt 2 * Real.cos (ω * x + φ)) 
  (h_product : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2)
  (h_min_diff : ∃ x₁ x₂ : ℝ, f x₁ * f x₂ = -2 ∧ |x₁ - x₂| = π/2) : ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_function_property_l1612_161269
