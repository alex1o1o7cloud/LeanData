import Mathlib

namespace NUMINAMATH_CALUDE_solution_check_l2793_279340

theorem solution_check (x : ℝ) : 
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 := by
  sorry

#check solution_check

end NUMINAMATH_CALUDE_solution_check_l2793_279340


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2793_279335

/-- The number of dots in each row and column of the square array -/
def gridSize : ℕ := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : ℕ := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating the number of different rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2793_279335


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2793_279324

theorem complex_magnitude_product : Complex.abs ((7 - 4*Complex.I) * (5 + 3*Complex.I)) = Real.sqrt 2210 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2793_279324


namespace NUMINAMATH_CALUDE_P_equals_59_when_V_is_9_l2793_279313

-- Define the relationship between P, h, and V
def P (h V : ℝ) : ℝ := 3 * h * V + 5

-- State the theorem
theorem P_equals_59_when_V_is_9 : 
  ∃ (h : ℝ), (P h 6 = 41) → (P h 9 = 59) := by
  sorry

end NUMINAMATH_CALUDE_P_equals_59_when_V_is_9_l2793_279313


namespace NUMINAMATH_CALUDE_sum_mod_ten_zero_l2793_279358

theorem sum_mod_ten_zero : (5000 + 5001 + 5002 + 5003 + 5004) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_ten_zero_l2793_279358


namespace NUMINAMATH_CALUDE_heat_required_temperature_dependent_specific_heat_l2793_279392

/-- The amount of heat required to heat a body with temperature-dependent specific heat capacity. -/
theorem heat_required_temperature_dependent_specific_heat
  (m : ℝ) (c₀ : ℝ) (α : ℝ) (t₁ t₂ : ℝ)
  (hm : m = 2)
  (hc₀ : c₀ = 150)
  (hα : α = 0.05)
  (ht₁ : t₁ = 20)
  (ht₂ : t₂ = 100)
  : ∃ Q : ℝ, Q = 96000 ∧ Q = m * (c₀ * (1 + α * t₂) + c₀ * (1 + α * t₁)) / 2 * (t₂ - t₁) :=
by sorry

end NUMINAMATH_CALUDE_heat_required_temperature_dependent_specific_heat_l2793_279392


namespace NUMINAMATH_CALUDE_geometric_sum_456_l2793_279302

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_456 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 9 →
  (a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_456_l2793_279302


namespace NUMINAMATH_CALUDE_weighted_average_is_correct_l2793_279354

def english_score : Rat := 76 / 120
def english_weight : Nat := 2

def math_score : Rat := 65 / 150
def math_weight : Nat := 3

def physics_score : Rat := 82 / 100
def physics_weight : Nat := 2

def chemistry_score : Rat := 67 / 80
def chemistry_weight : Nat := 1

def biology_score : Rat := 85 / 100
def biology_weight : Nat := 2

def history_score : Rat := 92 / 150
def history_weight : Nat := 1

def geography_score : Rat := 58 / 75
def geography_weight : Nat := 1

def total_weight : Nat := english_weight + math_weight + physics_weight + chemistry_weight + biology_weight + history_weight + geography_weight

def weighted_average_score : Rat :=
  (english_score * english_weight +
   math_score * math_weight +
   physics_score * physics_weight +
   chemistry_score * chemistry_weight +
   biology_score * biology_weight +
   history_score * history_weight +
   geography_score * geography_weight) / total_weight

theorem weighted_average_is_correct : weighted_average_score = 67755 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_is_correct_l2793_279354


namespace NUMINAMATH_CALUDE_professor_coffee_meeting_l2793_279308

theorem professor_coffee_meeting (n p q r : ℕ) : 
  (∀ (x : ℕ), x > 1 → x.Prime → r % (x ^ 2) ≠ 0) →  -- r is not divisible by the square of any prime
  (n : ℝ) = p - q * Real.sqrt r →  -- n = p - q√r
  (((120 : ℝ) - n) ^ 2 / 14400 = 1 / 2) →  -- probability of meeting is 50%
  p + q + r = 182 := by
  sorry

end NUMINAMATH_CALUDE_professor_coffee_meeting_l2793_279308


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2793_279339

theorem rectangular_plot_breadth (area length breadth : ℝ) : 
  area = 24 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2793_279339


namespace NUMINAMATH_CALUDE_blocks_used_proof_l2793_279321

/-- The number of blocks Randy used to build a tower -/
def tower_blocks : ℕ := 27

/-- The number of blocks Randy used to build a house -/
def house_blocks : ℕ := 53

/-- The total number of blocks Randy used for both the tower and the house -/
def total_blocks : ℕ := tower_blocks + house_blocks

theorem blocks_used_proof : total_blocks = 80 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_proof_l2793_279321


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1739_l2793_279387

theorem largest_prime_factor_of_1739 :
  (Nat.factors 1739).maximum? = some 47 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1739_l2793_279387


namespace NUMINAMATH_CALUDE_tie_distribution_impossibility_l2793_279327

theorem tie_distribution_impossibility 
  (B : Type) -- Set of boys
  (G : Type) -- Set of girls
  (knows : B → G → Prop) -- Relation representing who knows whom
  (color : B ⊕ G → Fin 99) -- Function assigning colors to people
  : ¬ (
    -- For any boy who knows at least 2015 girls
    (∀ b : B, (∃ (girls : Finset G), girls.card ≥ 2015 ∧ ∀ g ∈ girls, knows b g) →
      -- There are two girls among them with different colored ties
      ∃ g1 g2 : G, g1 ≠ g2 ∧ knows b g1 ∧ knows b g2 ∧ color (Sum.inr g1) ≠ color (Sum.inr g2)) ∧
    -- For any girl who knows at least 2015 boys
    (∀ g : G, (∃ (boys : Finset B), boys.card ≥ 2015 ∧ ∀ b ∈ boys, knows b g) →
      -- There are two boys among them with different colored ties
      ∃ b1 b2 : B, b1 ≠ b2 ∧ knows b1 g ∧ knows b2 g ∧ color (Sum.inl b1) ≠ color (Sum.inl b2))
  ) :=
by sorry

end NUMINAMATH_CALUDE_tie_distribution_impossibility_l2793_279327


namespace NUMINAMATH_CALUDE_race_length_proof_l2793_279375

/-- The race length in meters -/
def race_length : ℕ := 210

/-- Runner A's constant speed in m/s -/
def runner_a_speed : ℕ := 10

/-- Runner B's initial speed in m/s -/
def runner_b_initial_speed : ℕ := 1

/-- Runner B's speed increase per second in m/s -/
def runner_b_speed_increase : ℕ := 1

/-- Time difference between runners at finish in seconds -/
def finish_time_difference : ℕ := 1

/-- Function to calculate the distance covered by Runner B in t seconds -/
def runner_b_distance (t : ℕ) : ℕ := t * (t + 1) / 2

theorem race_length_proof :
  ∃ (t : ℕ), 
    (t * runner_a_speed = race_length) ∧ 
    (runner_b_distance (t - 1) = race_length) ∧ 
    (t > finish_time_difference) :=
by sorry

end NUMINAMATH_CALUDE_race_length_proof_l2793_279375


namespace NUMINAMATH_CALUDE_lamps_remain_lighted_l2793_279394

def toggle_lamps (n : ℕ) : ℕ :=
  n - (n / 2 + n / 3 + n / 5 - n / 6 - n / 10 - n / 15 + n / 30)

theorem lamps_remain_lighted :
  toggle_lamps 2015 = 1006 := by
  sorry

end NUMINAMATH_CALUDE_lamps_remain_lighted_l2793_279394


namespace NUMINAMATH_CALUDE_new_triangle_area_ratio_l2793_279374

/-- Represents a triangle -/
structure Triangle where
  area : ℝ

/-- Represents a point on a side of a triangle -/
structure PointOnSide where
  distance_ratio : ℝ

/-- Creates a new triangle from points on the sides of an original triangle -/
def new_triangle_from_points (original : Triangle) (p1 p2 p3 : PointOnSide) : Triangle :=
  sorry

theorem new_triangle_area_ratio (T : Triangle) :
  let p1 : PointOnSide := { distance_ratio := 1/3 }
  let p2 : PointOnSide := { distance_ratio := 1/3 }
  let p3 : PointOnSide := { distance_ratio := 1/3 }
  let new_T := new_triangle_from_points T p1 p2 p3
  new_T.area = (1/9) * T.area := by
  sorry

end NUMINAMATH_CALUDE_new_triangle_area_ratio_l2793_279374


namespace NUMINAMATH_CALUDE_find_S_l2793_279343

theorem find_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l2793_279343


namespace NUMINAMATH_CALUDE_sandy_clothes_spending_l2793_279373

/-- The amount Sandy spent on clothes -/
def total_spent (shorts_cost shirt_cost jacket_cost : ℚ) : ℚ :=
  shorts_cost + shirt_cost + jacket_cost

/-- Theorem: Sandy's total spending on clothes -/
theorem sandy_clothes_spending :
  total_spent 13.99 12.14 7.43 = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_spending_l2793_279373


namespace NUMINAMATH_CALUDE_johns_total_time_l2793_279367

/-- The total time John spent on his book and exploring is 5 years. -/
theorem johns_total_time (exploring_time note_writing_time book_writing_time : ℝ) :
  exploring_time = 3 →
  note_writing_time = exploring_time / 2 →
  book_writing_time = 0.5 →
  exploring_time + note_writing_time + book_writing_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_johns_total_time_l2793_279367


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2793_279332

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2793_279332


namespace NUMINAMATH_CALUDE_train_passing_bridge_l2793_279312

/-- Time for a train to pass a bridge -/
theorem train_passing_bridge (train_length : Real) (train_speed_kmph : Real) (bridge_length : Real) :
  train_length = 360 ∧ 
  train_speed_kmph = 45 ∧ 
  bridge_length = 140 →
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_bridge_l2793_279312


namespace NUMINAMATH_CALUDE_sqrt_2n_equals_64_l2793_279398

theorem sqrt_2n_equals_64 (n : ℝ) : Real.sqrt (2 * n) = 64 → n = 2048 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2n_equals_64_l2793_279398


namespace NUMINAMATH_CALUDE_annie_village_trick_or_treat_l2793_279310

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end NUMINAMATH_CALUDE_annie_village_trick_or_treat_l2793_279310


namespace NUMINAMATH_CALUDE_g_property_g_2022_l2793_279364

/-- A function g that satisfies the given property for all real x and y -/
def g : ℝ → ℝ := fun x ↦ 2021 * x

/-- The theorem stating that g satisfies the required property -/
theorem g_property : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y) := by sorry

/-- The main theorem proving that g(2022) equals 4086462 -/
theorem g_2022 : g 2022 = 4086462 := by sorry

end NUMINAMATH_CALUDE_g_property_g_2022_l2793_279364


namespace NUMINAMATH_CALUDE_always_true_inequality_l2793_279399

theorem always_true_inequality (a b x y : ℝ) (h1 : x < a) (h2 : y < b) : x * y < a * b := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l2793_279399


namespace NUMINAMATH_CALUDE_karen_cindy_crayon_difference_l2793_279323

theorem karen_cindy_crayon_difference :
  let karen_crayons : ℕ := 639
  let cindy_crayons : ℕ := 504
  karen_crayons - cindy_crayons = 135 :=
by sorry

end NUMINAMATH_CALUDE_karen_cindy_crayon_difference_l2793_279323


namespace NUMINAMATH_CALUDE_sides_in_nth_figure_formula_l2793_279349

/-- The number of sides in the n-th figure of a sequence starting with a hexagon
    and increasing by 5 sides for each subsequent figure. -/
def sides_in_nth_figure (n : ℕ) : ℕ := 5 * n + 1

/-- Theorem stating that the number of sides in the n-th figure is 5n + 1 -/
theorem sides_in_nth_figure_formula (n : ℕ) :
  sides_in_nth_figure n = 5 * n + 1 := by sorry

end NUMINAMATH_CALUDE_sides_in_nth_figure_formula_l2793_279349


namespace NUMINAMATH_CALUDE_range_of_a_l2793_279352

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2793_279352


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2793_279391

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   p * x₁^2 + (p - 1) * x₁ + p + 1 = 0 ∧
   p * x₂^2 + (p - 1) * x₂ + p + 1 = 0 ∧
   x₂ > 2 * x₁) →
  (0 < p ∧ p < 1/7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2793_279391


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2793_279390

theorem inequality_solution_set (p : ℝ) :
  (p ≥ 0 ∧ ∀ q > 0, (5 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q)) / (p + 2 * q) > 3 * p^2 * q) ↔
  0 ≤ p ∧ p < 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2793_279390


namespace NUMINAMATH_CALUDE_fraction_equality_l2793_279320

theorem fraction_equality : (2 - (1/2) * (1 - 1/4)) / (2 - (1 - 1/3)) = 39/32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2793_279320


namespace NUMINAMATH_CALUDE_nearest_year_with_more_zeros_than_ones_l2793_279314

/-- Given a natural number, returns the number of ones in its binary representation. -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Given a natural number, returns the number of zeros in its binary representation. -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Theorem: 2048 is the smallest integer greater than 2017 such that in its binary representation, 
    the number of ones is less than or equal to the number of zeros. -/
theorem nearest_year_with_more_zeros_than_ones : 
  ∀ k : ℕ, k > 2017 → k < 2048 → countOnes k > countZeros k :=
by sorry

end NUMINAMATH_CALUDE_nearest_year_with_more_zeros_than_ones_l2793_279314


namespace NUMINAMATH_CALUDE_gcd_problem_l2793_279303

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = k * 7771) : 
  Nat.gcd (Int.natAbs (8 * a^2 + 57 * a + 132)) (Int.natAbs (2 * a + 9)) = 9 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l2793_279303


namespace NUMINAMATH_CALUDE_correct_height_proof_l2793_279362

/-- Proves the correct height of a boy in a class given certain conditions -/
theorem correct_height_proof (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 184 →
  wrong_height = 166 →
  actual_avg = 182 →
  ∃ (correct_height : ℝ), correct_height = 236 ∧
    n * actual_avg = n * initial_avg - wrong_height + correct_height :=
by sorry

end NUMINAMATH_CALUDE_correct_height_proof_l2793_279362


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l2793_279307

/-- 
Given a restaurant that:
- Made some hamburgers and 4 hot dogs
- Served 3 hamburgers
- Had 6 hamburgers left over

Prove that the initial number of hamburgers was 9.
-/
theorem restaurant_hamburgers (served : ℕ) (leftover : ℕ) : 
  served = 3 → leftover = 6 → served + leftover = 9 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l2793_279307


namespace NUMINAMATH_CALUDE_max_basketballs_min_basketballs_for_profit_max_profit_l2793_279351

/-- Represents the sports equipment store problem -/
structure StoreProblem where
  total_balls : ℕ
  max_payment : ℕ
  basketball_wholesale : ℕ
  volleyball_wholesale : ℕ
  basketball_retail : ℕ
  volleyball_retail : ℕ
  min_profit : ℕ

/-- The specific instance of the store problem -/
def store_instance : StoreProblem :=
  { total_balls := 100
  , max_payment := 11815
  , basketball_wholesale := 130
  , volleyball_wholesale := 100
  , basketball_retail := 160
  , volleyball_retail := 120
  , min_profit := 2580
  }

/-- Calculates the total cost of purchasing basketballs and volleyballs -/
def total_cost (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  p.basketball_wholesale * basketballs + p.volleyball_wholesale * (p.total_balls - basketballs)

/-- Calculates the profit from selling all balls -/
def profit (p : StoreProblem) (basketballs : ℕ) : ℕ :=
  (p.basketball_retail - p.basketball_wholesale) * basketballs +
  (p.volleyball_retail - p.volleyball_wholesale) * (p.total_balls - basketballs)

/-- Theorem stating the maximum number of basketballs that can be purchased -/
theorem max_basketballs (p : StoreProblem) :
  ∃ (max_basketballs : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → b ≤ max_basketballs) ∧
    total_cost p max_basketballs ≤ p.max_payment ∧
    max_basketballs = 60 :=
  sorry

/-- Theorem stating the minimum number of basketballs needed for desired profit -/
theorem min_basketballs_for_profit (p : StoreProblem) :
  ∃ (min_basketballs : ℕ),
    (∀ (b : ℕ), profit p b ≥ p.min_profit → b ≥ min_basketballs) ∧
    profit p min_basketballs ≥ p.min_profit ∧
    min_basketballs = 58 :=
  sorry

/-- Theorem stating the maximum profit achievable -/
theorem max_profit (p : StoreProblem) :
  ∃ (max_profit : ℕ),
    (∀ (b : ℕ), total_cost p b ≤ p.max_payment → profit p b ≤ max_profit) ∧
    (∃ (b : ℕ), total_cost p b ≤ p.max_payment ∧ profit p b = max_profit) ∧
    max_profit = 2600 :=
  sorry

end NUMINAMATH_CALUDE_max_basketballs_min_basketballs_for_profit_max_profit_l2793_279351


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2793_279337

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 15) 
  (h2 : x*y + y*z + z*x = 34) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1845 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2793_279337


namespace NUMINAMATH_CALUDE_xy_equals_three_l2793_279345

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l2793_279345


namespace NUMINAMATH_CALUDE_correct_solution_l2793_279346

/-- The original equation -/
def original_equation (x : ℚ) : Prop :=
  (2 - 2*x) / 3 = (3*x - 3) / 7 + 3

/-- Xiao Jun's incorrect equation -/
def incorrect_equation (x m : ℚ) : Prop :=
  7*(2 - 2*x) = 3*(3*x - m) + 3

/-- Xiao Jun's solution -/
def xiao_jun_solution : ℚ := 14/23

/-- The correct value of m -/
def correct_m : ℚ := 3

theorem correct_solution :
  incorrect_equation xiao_jun_solution correct_m →
  ∃ x : ℚ, x = 2 ∧ original_equation x :=
by sorry

end NUMINAMATH_CALUDE_correct_solution_l2793_279346


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l2793_279360

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l2793_279360


namespace NUMINAMATH_CALUDE_bouquet_cost_45_lilies_l2793_279301

/-- Represents the cost of a bouquet of lilies -/
def bouquet_cost (num_lilies : ℕ) : ℚ :=
  let base_price_per_lily : ℚ := 30 / 15
  let discount_threshold : ℕ := 30
  let discount_rate : ℚ := 1 / 10
  if num_lilies ≤ discount_threshold then
    num_lilies * base_price_per_lily
  else
    num_lilies * (base_price_per_lily * (1 - discount_rate))

theorem bouquet_cost_45_lilies :
  bouquet_cost 45 = 81 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_45_lilies_l2793_279301


namespace NUMINAMATH_CALUDE_tomato_cucumber_ratio_l2793_279383

/-- Given the initial quantities of tomatoes and cucumbers, and the amounts picked,
    prove that the ratio of remaining tomatoes to remaining cucumbers is 7:68. -/
theorem tomato_cucumber_ratio
  (initial_tomatoes : ℕ)
  (initial_cucumbers : ℕ)
  (tomatoes_picked_yesterday : ℕ)
  (tomatoes_picked_today : ℕ)
  (cucumbers_picked_total : ℕ)
  (h1 : initial_tomatoes = 171)
  (h2 : initial_cucumbers = 225)
  (h3 : tomatoes_picked_yesterday = 134)
  (h4 : tomatoes_picked_today = 30)
  (h5 : cucumbers_picked_total = 157)
  : (initial_tomatoes - (tomatoes_picked_yesterday + tomatoes_picked_today)) /
    (initial_cucumbers - cucumbers_picked_total) = 7 / 68 :=
by sorry

end NUMINAMATH_CALUDE_tomato_cucumber_ratio_l2793_279383


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2793_279309

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 1 = 0) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2793_279309


namespace NUMINAMATH_CALUDE_division_simplification_l2793_279304

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2793_279304


namespace NUMINAMATH_CALUDE_cindys_cycling_speed_l2793_279377

/-- Cindy's cycling problem -/
theorem cindys_cycling_speed :
  -- Cindy leaves school at the same time every day
  ∀ (leave_time : ℝ),
  -- Define the distance from school to home
  ∀ (distance : ℝ),
  -- If she cycles at 20 km/h, she arrives home at 4:30 PM
  (distance / 20 = 4.5 - leave_time) →
  -- If she cycles at 10 km/h, she arrives home at 5:15 PM
  (distance / 10 = 5.25 - leave_time) →
  -- Then the speed at which she must cycle to arrive home at 5:00 PM is 12 km/h
  (distance / 12 = 5 - leave_time) :=
by sorry

end NUMINAMATH_CALUDE_cindys_cycling_speed_l2793_279377


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2793_279397

-- Define the circle C
def circle_C (a x y : ℝ) : Prop := (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a x y : ℝ) : Prop :=
  circle_C a x y ∧ (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- Main theorem
theorem circle_intersection_range (a : ℝ) :
  (∃ x y : ℝ, condition_M a x y) → a ∈ Set.Icc 0 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2793_279397


namespace NUMINAMATH_CALUDE_different_color_probability_l2793_279344

theorem different_color_probability : 
  let total_balls : ℕ := 5
  let white_balls : ℕ := 2
  let black_balls : ℕ := 3
  let probability_different_colors : ℚ := 12 / 25
  (white_balls + black_balls = total_balls) →
  (probability_different_colors = 
    (white_balls * black_balls + black_balls * white_balls) / (total_balls * total_balls)) :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l2793_279344


namespace NUMINAMATH_CALUDE_infinite_points_satisfying_condition_l2793_279326

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the condition for points P
def SatisfiesCondition (p : ℝ × ℝ) (endpoints : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a, b) := endpoints
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 = 5

-- Theorem statement
theorem infinite_points_satisfying_condition 
  (center : ℝ × ℝ) : 
  ∃ (s : Set (ℝ × ℝ)), 
    (∀ p ∈ s, p ∈ Circle center 2 ∧ 
              SatisfiesCondition p (DiameterEndpoints center 2)) ∧
    (Set.Infinite s) := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_satisfying_condition_l2793_279326


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l2793_279347

/-- The complex number z = (1+2i)/(1-2i) is in the second quadrant -/
theorem complex_in_second_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - 2*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l2793_279347


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2793_279328

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : BallColor)
  (second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

theorem mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome ∨ exactlyTwoWhite outcome) ∧
  (∃ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∨ exactlyTwoWhite outcome)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2793_279328


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2793_279322

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The value of k for which the points (2, -3), (k, k + 2), and (-3k + 4, 1) are collinear. -/
theorem collinear_points_k_value :
  ∃ k : ℝ, collinear 2 (-3) k (k + 2) (-3 * k + 4) 1 ∧
    (k = (17 + Real.sqrt 505) / (-6) ∨ k = (17 - Real.sqrt 505) / (-6)) := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2793_279322


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_range_l2793_279330

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (3/2) 3 := by sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_range_l2793_279330


namespace NUMINAMATH_CALUDE_number_of_men_l2793_279359

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 20 : ℝ) = W / ((M - 4) * 25 : ℝ)) → M = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l2793_279359


namespace NUMINAMATH_CALUDE_fourth_derivative_y_l2793_279384

noncomputable def y (x : ℝ) : ℝ := (5 * x - 8) * (2 ^ (-x))

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = 2^(-x) * (Real.log 2)^4 * (5*x - 9) := by sorry

end NUMINAMATH_CALUDE_fourth_derivative_y_l2793_279384


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2793_279306

theorem hyperbola_vertices_distance (x y : ℝ) :
  x^2 / 144 - y^2 / 64 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (64 : ℝ) = 1 ∧ 2 * a = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2793_279306


namespace NUMINAMATH_CALUDE_diagonal_sum_property_l2793_279342

/-- A convex regular polygon with 3k sides, where k > 4 is an integer -/
structure RegularPolygon (k : ℕ) :=
  (sides : ℕ)
  (convex : Bool)
  (regular : Bool)
  (k_gt_4 : k > 4)
  (sides_eq_3k : sides = 3 * k)

/-- A diagonal in a polygon -/
structure Diagonal (P : RegularPolygon k) :=
  (length : ℝ)

/-- Theorem: In a convex regular polygon with 3k sides (k > 4), 
    there exist diagonals whose lengths are equal to the sum of 
    the lengths of two shorter diagonals -/
theorem diagonal_sum_property (k : ℕ) (P : RegularPolygon k) :
  ∃ (d1 d2 d3 : Diagonal P), 
    d1.length = d2.length + d3.length ∧ 
    d1.length > d2.length ∧ 
    d1.length > d3.length :=
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_property_l2793_279342


namespace NUMINAMATH_CALUDE_purple_ring_weight_l2793_279395

/-- The weight of the purple ring in Karin's science class experiment -/
theorem purple_ring_weight (orange_weight white_weight total_weight : ℚ)
  (h_orange : orange_weight = 8/100)
  (h_white : white_weight = 42/100)
  (h_total : total_weight = 83/100) :
  total_weight - orange_weight - white_weight = 33/100 := by
  sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l2793_279395


namespace NUMINAMATH_CALUDE_no_inscribable_2010_gon_l2793_279380

theorem no_inscribable_2010_gon : ¬ ∃ (sides : Fin 2010 → ℕ), 
  (∀ i : Fin 2010, 1 ≤ sides i ∧ sides i ≤ 2010) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 2010 → ∃ i : Fin 2010, sides i = n) ∧
  (∃ r : ℝ, r > 0 ∧ ∀ i : Fin 2010, 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = (sides i)^2 ∧ a * b = r * (sides i)) :=
by sorry

end NUMINAMATH_CALUDE_no_inscribable_2010_gon_l2793_279380


namespace NUMINAMATH_CALUDE_diana_total_earnings_l2793_279318

def july_earnings : ℝ := 150

def august_earnings : ℝ := 3 * july_earnings

def september_earnings : ℝ := 2 * august_earnings

def october_earnings : ℝ := september_earnings * 1.1

def november_earnings : ℝ := october_earnings * 0.95

def total_earnings : ℝ := july_earnings + august_earnings + september_earnings + october_earnings + november_earnings

theorem diana_total_earnings : total_earnings = 3430.50 := by
  sorry

end NUMINAMATH_CALUDE_diana_total_earnings_l2793_279318


namespace NUMINAMATH_CALUDE_staircase_steps_l2793_279300

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ :=
  (total_steps + step_size - 1) / step_size

theorem staircase_steps : ∃ (n : ℕ), n > 0 ∧ jumps 3 n - jumps 4 n = 10 ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_staircase_steps_l2793_279300


namespace NUMINAMATH_CALUDE_half_floors_full_capacity_l2793_279385

/-- Represents a building with floors, apartments, and occupants. -/
structure Building where
  total_floors : ℕ
  apartments_per_floor : ℕ
  people_per_apartment : ℕ
  total_people : ℕ

/-- Calculates the number of full-capacity floors in the building. -/
def full_capacity_floors (b : Building) : ℕ :=
  let people_per_full_floor := b.apartments_per_floor * b.people_per_apartment
  let total_full_floor_capacity := b.total_floors * people_per_full_floor
  (2 * b.total_people - total_full_floor_capacity) / people_per_full_floor

/-- Theorem stating that for a building with specific parameters,
    the number of full-capacity floors is half the total floors. -/
theorem half_floors_full_capacity (b : Building)
    (h1 : b.total_floors = 12)
    (h2 : b.apartments_per_floor = 10)
    (h3 : b.people_per_apartment = 4)
    (h4 : b.total_people = 360) :
    full_capacity_floors b = b.total_floors / 2 := by
  sorry

end NUMINAMATH_CALUDE_half_floors_full_capacity_l2793_279385


namespace NUMINAMATH_CALUDE_volume_of_cubes_l2793_279333

/-- Given two cubes where the ratio of their edges is 3:1 and the volume of the smaller cube is 8 units,
    the volume of the larger cube is 216 units. -/
theorem volume_of_cubes (a b : ℝ) (h1 : a / b = 3) (h2 : b^3 = 8) : a^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cubes_l2793_279333


namespace NUMINAMATH_CALUDE_oldest_child_age_l2793_279317

-- Define the problem parameters
def num_children : ℕ := 4
def average_age : ℝ := 8
def younger_ages : List ℝ := [5, 7, 9]

-- State the theorem
theorem oldest_child_age :
  ∀ (oldest_age : ℝ),
  (List.sum younger_ages + oldest_age) / num_children = average_age →
  oldest_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l2793_279317


namespace NUMINAMATH_CALUDE_lcm_1008_672_l2793_279348

theorem lcm_1008_672 : Nat.lcm 1008 672 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1008_672_l2793_279348


namespace NUMINAMATH_CALUDE_watch_loss_percentage_l2793_279341

/-- Proves that the loss percentage is 10% given the conditions of the watch sale problem -/
theorem watch_loss_percentage (cost_price : ℝ) (additional_price : ℝ) (gain_percentage : ℝ) 
  (h1 : cost_price = 1428.57)
  (h2 : additional_price = 200)
  (h3 : gain_percentage = 4) : 
  ∃ (loss_percentage : ℝ), 
    loss_percentage = 10 ∧ 
    cost_price + additional_price = cost_price * (1 + gain_percentage / 100) ∧
    cost_price * (1 - loss_percentage / 100) + additional_price = cost_price * (1 + gain_percentage / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_watch_loss_percentage_l2793_279341


namespace NUMINAMATH_CALUDE_num_event_committees_l2793_279376

/-- The number of teams in the tournament -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the event committee -/
def committee_size : ℕ := 16

/-- Theorem stating the number of possible event committees -/
theorem num_event_committees : 
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 3443073600 := by
  sorry

end NUMINAMATH_CALUDE_num_event_committees_l2793_279376


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_13_l2793_279368

/-- The area of a quadrilateral with vertices (0, 0), (4, 0), (4, 3), and (2, 5) -/
def quadrilateral_area : ℝ :=
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (4, 3)
  let v4 : ℝ × ℝ := (2, 5)
  -- Define the area calculation here
  0 -- placeholder, replace with actual calculation

/-- Theorem: The area of the quadrilateral is 13 -/
theorem quadrilateral_area_is_13 : quadrilateral_area = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_13_l2793_279368


namespace NUMINAMATH_CALUDE_leapYearsIn200Years_l2793_279386

/-- Definition of a leap year in the modified calendar system -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 == 0 && year % 128 ≠ 0

/-- Count of leap years in a given period -/
def countLeapYears (period : ℕ) : ℕ :=
  (List.range period).filter isLeapYear |>.length

/-- Theorem: There are 49 leap years in a 200-year period -/
theorem leapYearsIn200Years : countLeapYears 200 = 49 := by
  sorry

end NUMINAMATH_CALUDE_leapYearsIn200Years_l2793_279386


namespace NUMINAMATH_CALUDE_larger_number_is_588_l2793_279366

/-- Given two positive integers with HCF 42 and LCM factors 12 and 14, the larger number is 588 -/
theorem larger_number_is_588 (a b : ℕ+) (hcf : Nat.gcd a b = 42) 
  (lcm_factors : ∃ (x y : ℕ+), x = 12 ∧ y = 14 ∧ Nat.lcm a b = 42 * x * y) :
  max a b = 588 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_588_l2793_279366


namespace NUMINAMATH_CALUDE_infinite_power_tower_eq_four_l2793_279325

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := Real.log x / Real.log (Real.log x)

/-- Theorem stating that if the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_eq_four (x : ℝ) (h : x > 0) :
  infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_eq_four_l2793_279325


namespace NUMINAMATH_CALUDE_adams_change_l2793_279336

/-- Given that Adam has $5 and an airplane costs $4.28, prove that the change Adam will receive is $0.72. -/
theorem adams_change (adam_money : ℝ) (airplane_cost : ℝ) (change : ℝ) 
  (h1 : adam_money = 5)
  (h2 : airplane_cost = 4.28)
  (h3 : change = adam_money - airplane_cost) :
  change = 0.72 := by
sorry

end NUMINAMATH_CALUDE_adams_change_l2793_279336


namespace NUMINAMATH_CALUDE_toy_production_proof_l2793_279379

/-- A factory produces toys. -/
structure ToyFactory where
  weekly_production : ℕ
  working_days : ℕ
  uniform_production : Bool

/-- Calculate the daily toy production for a given factory. -/
def daily_production (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.working_days

theorem toy_production_proof (factory : ToyFactory) 
  (h1 : factory.weekly_production = 5505)
  (h2 : factory.working_days = 5)
  (h3 : factory.uniform_production = true) :
  daily_production factory = 1101 := by
  sorry

#eval daily_production { weekly_production := 5505, working_days := 5, uniform_production := true }

end NUMINAMATH_CALUDE_toy_production_proof_l2793_279379


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2793_279355

theorem sufficient_not_necessary (a : ℝ) : 
  (a > 1 → 1/a < 1) ∧ (∃ b : ℝ, b ≤ 1 ∧ 1/b < 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2793_279355


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2793_279372

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 5 → b = 5 → c = 2 →
  (a = b) →  -- isosceles condition
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →  -- triangle inequality
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2793_279372


namespace NUMINAMATH_CALUDE_log_equation_solution_l2793_279353

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 →
  x = 2^(27/10) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2793_279353


namespace NUMINAMATH_CALUDE_rational_function_value_at_one_l2793_279331

/-- A structure representing a rational function with specific properties. -/
structure RationalFunction where
  r : ℝ → ℝ  -- Numerator polynomial
  s : ℝ → ℝ  -- Denominator polynomial
  is_quadratic_r : ∃ a b c : ℝ, ∀ x, r x = a * x^2 + b * x + c
  is_quadratic_s : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  hole_at_4 : r 4 = 0 ∧ s 4 = 0
  zero_at_0 : r 0 = 0
  horizontal_asymptote : ∀ ε > 0, ∃ M, ∀ x > M, |r x / s x + 2| < ε
  vertical_asymptote : s 3 = 0 ∧ r 3 ≠ 0

/-- Theorem stating that for a rational function with the given properties, r(1)/s(1) = 1 -/
theorem rational_function_value_at_one (f : RationalFunction) : f.r 1 / f.s 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_one_l2793_279331


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2793_279396

theorem x_minus_y_value (x y : ℝ) 
  (eq1 : 3015 * x + 3020 * y = 3025)
  (eq2 : 3018 * x + 3024 * y = 3030) : 
  x - y = 11.1167 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2793_279396


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2793_279334

theorem fraction_to_decimal :
  (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2793_279334


namespace NUMINAMATH_CALUDE_comic_book_ratio_l2793_279371

/-- Represents the number of comic books Sandy has at different stages -/
structure ComicBooks where
  initial : ℕ
  sold : ℕ
  bought : ℕ
  final : ℕ

/-- The ratio of sold books to initial books is 1:2 -/
theorem comic_book_ratio (s : ComicBooks) 
  (h1 : s.initial = 14)
  (h2 : s.bought = 6)
  (h3 : s.final = 13)
  (h4 : s.initial - s.sold + s.bought = s.final) :
  s.sold / s.initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l2793_279371


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2793_279370

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_6 = 6 and a_9 = 9, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_6 : a 6 = 6) 
    (h_9 : a 9 = 9) : 
  a 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l2793_279370


namespace NUMINAMATH_CALUDE_mount_tai_temp_difference_l2793_279350

/-- The temperature difference between two points is the absolute value of their difference. -/
def temperature_difference (t1 t2 : ℝ) : ℝ := |t1 - t2|

/-- The average temperature at the top of Mount Tai in January (in °C). -/
def temp_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai in January (in °C). -/
def temp_foot : ℝ := -1

/-- The temperature difference between the foot and top of Mount Tai is 8°C. -/
theorem mount_tai_temp_difference : temperature_difference temp_foot temp_top = 8 := by
  sorry

end NUMINAMATH_CALUDE_mount_tai_temp_difference_l2793_279350


namespace NUMINAMATH_CALUDE_factory_output_percentage_l2793_279382

theorem factory_output_percentage (T X Y : ℝ) : 
  T > 0 →  -- Total output is positive
  X > 0 →  -- Machine-x output is positive
  Y > 0 →  -- Machine-y output is positive
  X + Y = T →  -- Total output is sum of both machines
  0.006 * T = 0.009 * X + 0.004 * Y →  -- Defective units equation
  X = 0.4 * T  -- Machine-x produces 40% of total output
  := by sorry

end NUMINAMATH_CALUDE_factory_output_percentage_l2793_279382


namespace NUMINAMATH_CALUDE_log3_one_over_81_l2793_279378

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_over_81 : log3 (1/81) = -4 := by
  sorry

end NUMINAMATH_CALUDE_log3_one_over_81_l2793_279378


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2793_279381

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define monotonically increasing function
def isMonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x ^ 3

-- Theorem statement
theorem cubic_function_properties :
  isPowerFunction f ∧ isMonotonicallyIncreasing f :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2793_279381


namespace NUMINAMATH_CALUDE_marble_collection_total_l2793_279357

/-- Given a collection of orange, purple, and yellow marbles, where:
    - The number of orange marbles is o
    - There are 30% more orange marbles than purple marbles
    - There are 50% more yellow marbles than orange marbles
    Prove that the total number of marbles is 3.269o -/
theorem marble_collection_total (o : ℝ) (o_positive : o > 0) : ∃ (p y : ℝ),
  p > 0 ∧ y > 0 ∧
  o = 1.3 * p ∧
  y = 1.5 * o ∧
  o + p + y = 3.269 * o :=
sorry

end NUMINAMATH_CALUDE_marble_collection_total_l2793_279357


namespace NUMINAMATH_CALUDE_last_digit_sum_l2793_279389

theorem last_digit_sum (x y : ℕ) : 
  (135^x + 31^y + 56^(x+y)) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_l2793_279389


namespace NUMINAMATH_CALUDE_diophantine_approximation_l2793_279315

theorem diophantine_approximation (x : ℝ) : 
  ∀ N : ℕ, ∃ p q : ℤ, q > N ∧ |x - (p : ℝ) / (q : ℝ)| < 1 / (q : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l2793_279315


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l2793_279329

theorem price_reduction_theorem (x : ℝ) : 
  (1 - x / 100) * 1.8 = 1.17 → x = 35 := by sorry

end NUMINAMATH_CALUDE_price_reduction_theorem_l2793_279329


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_of_digits_divisible_l2793_279305

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def satisfies_condition (start : ℕ) (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → k ≤ N → (sum_of_digits (start + k - 1)) % k = 0

theorem largest_consecutive_sum_of_digits_divisible :
  ∃ start : ℕ, satisfies_condition start 21 ∧
  ∀ N : ℕ, N > 21 → ¬∃ start : ℕ, satisfies_condition start N :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_of_digits_divisible_l2793_279305


namespace NUMINAMATH_CALUDE_current_speed_l2793_279393

/-- The speed of the current given a woman's swimming times -/
theorem current_speed (v c : ℝ) 
  (h1 : v + c = 64 / 8)  -- Downstream speed
  (h2 : v - c = 24 / 8)  -- Upstream speed
  : c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l2793_279393


namespace NUMINAMATH_CALUDE_benny_apples_l2793_279319

def dan_apples : ℕ := 9
def total_apples : ℕ := 11

theorem benny_apples : total_apples - dan_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_apples_l2793_279319


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2793_279361

/-- The lateral area of a cone with specific properties -/
theorem cone_lateral_area (base_diameter : ℝ) (slant_height : ℝ) :
  base_diameter = 6 →
  slant_height = 6 →
  (1 / 2) * (2 * Real.pi) * (base_diameter / 2) * slant_height = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2793_279361


namespace NUMINAMATH_CALUDE_cubic_equation_roots_of_unity_l2793_279363

theorem cubic_equation_roots_of_unity :
  ∃ (a b c : ℤ), (1 : ℂ)^3 + a*(1 : ℂ)^2 + b*(1 : ℂ) + c = 0 ∧
                 (-1 : ℂ)^3 + a*(-1 : ℂ)^2 + b*(-1 : ℂ) + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_of_unity_l2793_279363


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2793_279388

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 4*m + 3)
  (z.re = 0 ∧ z.im ≠ 0) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2793_279388


namespace NUMINAMATH_CALUDE_schedule_count_is_576_l2793_279338

/-- Represents a table tennis match between two schools -/
structure TableTennisMatch where
  /-- Number of players in each school -/
  players_per_school : Nat
  /-- Number of opponents each player faces from the other school -/
  opponents_per_player : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- The specific match configuration from the problem -/
def match_config : TableTennisMatch :=
  { players_per_school := 4
  , opponents_per_player := 2
  , total_rounds := 6
  , games_per_round := 4
  }

/-- Calculate the number of ways to schedule the match -/
def schedule_count (m : TableTennisMatch) : Nat :=
  (Nat.factorial m.total_rounds) * (Nat.factorial m.games_per_round)

/-- Theorem stating that the number of ways to schedule the match is 576 -/
theorem schedule_count_is_576 : schedule_count match_config = 576 := by
  sorry


end NUMINAMATH_CALUDE_schedule_count_is_576_l2793_279338


namespace NUMINAMATH_CALUDE_sum_of_counts_l2793_279365

/-- A function that returns the count of four-digit even numbers -/
def count_four_digit_even : ℕ :=
  sorry

/-- A function that returns the count of four-digit numbers divisible by both 5 and 3 -/
def count_four_digit_div_by_5_and_3 : ℕ :=
  sorry

/-- Theorem stating that the sum of four-digit even numbers and four-digit numbers
    divisible by both 5 and 3 is equal to 5100 -/
theorem sum_of_counts : count_four_digit_even + count_four_digit_div_by_5_and_3 = 5100 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_counts_l2793_279365


namespace NUMINAMATH_CALUDE_xf_is_even_l2793_279311

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- Theorem statement
theorem xf_is_even (f : ℝ → ℝ) (h : OddFunction f) :
  EvenFunction (fun x ↦ x * f x) := by
  sorry

end NUMINAMATH_CALUDE_xf_is_even_l2793_279311


namespace NUMINAMATH_CALUDE_cylinder_radius_l2793_279356

structure Cone where
  diameter : ℚ
  altitude : ℚ

structure Cylinder where
  radius : ℚ

def inscribed_cylinder (cone : Cone) (cyl : Cylinder) : Prop :=
  cyl.radius * 2 = cyl.radius * 2 ∧  -- cylinder's diameter equals its height
  cone.diameter = 10 ∧
  cone.altitude = 12 ∧
  -- The axes of the cylinder and cone coincide (implicit in the problem setup)
  true

theorem cylinder_radius (cone : Cone) (cyl : Cylinder) :
  inscribed_cylinder cone cyl → cyl.radius = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l2793_279356


namespace NUMINAMATH_CALUDE_power_product_l2793_279316

theorem power_product (x y : ℝ) (h1 : (10 : ℝ) ^ x = 3) (h2 : (10 : ℝ) ^ y = 4) : 
  (10 : ℝ) ^ (x * y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_product_l2793_279316


namespace NUMINAMATH_CALUDE_scientific_notation_of_1373100000000_l2793_279369

theorem scientific_notation_of_1373100000000 :
  (1373100000000 : ℝ) = 1.3731 * (10 ^ 12) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1373100000000_l2793_279369
