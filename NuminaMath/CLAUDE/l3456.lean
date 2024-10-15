import Mathlib

namespace NUMINAMATH_CALUDE_inverse_sum_modulo_thirteen_l3456_345671

theorem inverse_sum_modulo_thirteen : 
  (((5⁻¹ : ZMod 13) + (7⁻¹ : ZMod 13) + (9⁻¹ : ZMod 13) + (11⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 11 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_modulo_thirteen_l3456_345671


namespace NUMINAMATH_CALUDE_min_sum_for_product_3006_l3456_345611

theorem min_sum_for_product_3006 (a b c : ℕ+) (h : a * b * c = 3006) :
  (∀ x y z : ℕ+, x * y * z = 3006 → a + b + c ≤ x + y + z) ∧ a + b + c = 105 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_for_product_3006_l3456_345611


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_inequality_l3456_345676

/-- Given a tetrahedron ABCD with inscribed sphere radius r and exinscribed sphere radii r_A, r_B, r_C, r_D,
    the sum of the reciprocals of the square roots of the sums of squares minus products of adjacent radii
    is less than or equal to 2/r. -/
theorem tetrahedron_sphere_inequality (r r_A r_B r_C r_D : ℝ) 
  (hr : r > 0) (hr_A : r_A > 0) (hr_B : r_B > 0) (hr_C : r_C > 0) (hr_D : r_D > 0) :
  1 / Real.sqrt (r_A^2 - r_A*r_B + r_B^2) + 
  1 / Real.sqrt (r_B^2 - r_B*r_C + r_C^2) + 
  1 / Real.sqrt (r_C^2 - r_C*r_D + r_D^2) + 
  1 / Real.sqrt (r_D^2 - r_D*r_A + r_A^2) ≤ 2 / r :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_inequality_l3456_345676


namespace NUMINAMATH_CALUDE_basketball_game_result_l3456_345637

/-- Calculates the final score difference after the last quarter of a basketball game -/
def final_score_difference (initial_deficit : ℤ) (liz_free_throws : ℕ) (liz_three_pointers : ℕ) (liz_jump_shots : ℕ) (opponent_points : ℕ) : ℤ :=
  initial_deficit - (liz_free_throws + 3 * liz_three_pointers + 2 * liz_jump_shots - opponent_points)

theorem basketball_game_result :
  final_score_difference 20 5 3 4 10 = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_game_result_l3456_345637


namespace NUMINAMATH_CALUDE_eggs_leftover_l3456_345679

theorem eggs_leftover (abigail_eggs beatrice_eggs carson_eggs : ℕ) 
  (h1 : abigail_eggs = 37)
  (h2 : beatrice_eggs = 49)
  (h3 : carson_eggs = 14) :
  (abigail_eggs + beatrice_eggs + carson_eggs) % 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l3456_345679


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l3456_345636

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (straight_only : ℕ) 
  (h1 : total = 40)
  (h2 : both = 10)
  (h3 : straight_only = 24)
  (h4 : total = both + straight_only + (total - (both + straight_only))) :
  total - (both + straight_only) = 6 := by
sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l3456_345636


namespace NUMINAMATH_CALUDE_max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l3456_345633

/-- Given a right-angled triangle with legs a and b, the maximum area of a rectangle
    that can be cut from it, using the right angle of the triangle as one of the
    rectangle's corners, is (a * b) / 4 -/
theorem max_rectangle_area_in_right_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let triangle_area := a * b / 2
  let max_rectangle_area := triangle_area / 2
  max_rectangle_area = a * b / 4 := by sorry

/-- The maximum area of a rectangle that can be cut from a right-angled triangle
    with legs measuring 40 cm and 60 cm, using the right angle of the triangle as
    one of the rectangle's corners, is 600 cm² -/
theorem max_rectangle_area_40_60 :
  let a : ℝ := 40
  let b : ℝ := 60
  let max_area := a * b / 4
  max_area = 600 := by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l3456_345633


namespace NUMINAMATH_CALUDE_park_short_trees_l3456_345648

def initial_short_trees : ℕ := 3
def short_trees_to_plant : ℕ := 9
def final_short_trees : ℕ := 12

theorem park_short_trees :
  initial_short_trees + short_trees_to_plant = final_short_trees :=
by sorry

end NUMINAMATH_CALUDE_park_short_trees_l3456_345648


namespace NUMINAMATH_CALUDE_complex_sum_equality_l3456_345614

theorem complex_sum_equality : 
  let A : ℂ := 2 + I
  let O : ℂ := -4
  let P : ℂ := -I
  let S : ℂ := 2 + 4*I
  A - O + P + S = 8 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l3456_345614


namespace NUMINAMATH_CALUDE_chord_length_through_focus_l3456_345666

/-- Given a parabola y^2 = 8x, prove that a chord AB passing through the focus
    with endpoints A(x₁, y₁) and B(x₂, y₂) on the parabola, where x₁ + x₂ = 10,
    has length |AB| = 14. -/
theorem chord_length_through_focus (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 8*x₁ →  -- A is on the parabola
  y₂^2 = 8*x₂ →  -- B is on the parabola
  x₁ + x₂ = 10 → -- Given condition
  -- AB passes through the focus (2, 0)
  (y₂ - y₁) * 2 = (x₂ - x₁) * (y₂ + y₁) →
  -- The length of AB is 14
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 14^2 := by
sorry

end NUMINAMATH_CALUDE_chord_length_through_focus_l3456_345666


namespace NUMINAMATH_CALUDE_journey_length_l3456_345689

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total + 24 + (1 / 6 : ℚ) * total = total →
  total = 288 / 7 := by
sorry

end NUMINAMATH_CALUDE_journey_length_l3456_345689


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3456_345695

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_roots_range (a b : ℝ) :
  (∃ x y, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  a^2 - 2*b ∈ Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3456_345695


namespace NUMINAMATH_CALUDE_isabel_piggy_bank_l3456_345664

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  initial_amount / 2 / 2 = 51 → initial_amount = 204 := by
  sorry

end NUMINAMATH_CALUDE_isabel_piggy_bank_l3456_345664


namespace NUMINAMATH_CALUDE_sum_A_B_eq_x_squared_sum_A_2B_eq_24_when_x_neg_2_l3456_345699

-- Define variables
variable (x : ℝ)

-- Define B as a function of x
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 6

-- Define A as a function of x, given A-B
def A (x : ℝ) : ℝ := (-7 * x^2 + 10 * x + 12) + B x

-- Theorem 1: A+B = x^2
theorem sum_A_B_eq_x_squared (x : ℝ) : A x + B x = x^2 := by sorry

-- Theorem 2: A+2B = 24 when x=-2
theorem sum_A_2B_eq_24_when_x_neg_2 : A (-2) + 2 * B (-2) = 24 := by sorry

end NUMINAMATH_CALUDE_sum_A_B_eq_x_squared_sum_A_2B_eq_24_when_x_neg_2_l3456_345699


namespace NUMINAMATH_CALUDE_c_value_for_four_distinct_roots_l3456_345672

/-- The polynomial P(x) -/
def P (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 5) * (x^2 - c*x + 2) * (x^2 - 5*x + 10)

/-- The theorem stating the relationship between c and the number of distinct roots of P(x) -/
theorem c_value_for_four_distinct_roots (c : ℂ) : 
  (∃ (S : Finset ℂ), S.card = 4 ∧ (∀ x ∈ S, P c x = 0) ∧ (∀ x, P c x = 0 → x ∈ S)) →
  Complex.abs c = Real.sqrt (22.5 - Real.sqrt 165) := by
  sorry

end NUMINAMATH_CALUDE_c_value_for_four_distinct_roots_l3456_345672


namespace NUMINAMATH_CALUDE_blanche_eggs_l3456_345661

theorem blanche_eggs (gertrude nancy martha blanche total_eggs : ℕ) : 
  gertrude = 4 →
  nancy = 2 →
  martha = 2 →
  total_eggs = gertrude + nancy + martha + blanche →
  total_eggs - 2 = 9 →
  blanche = 3 := by
sorry

end NUMINAMATH_CALUDE_blanche_eggs_l3456_345661


namespace NUMINAMATH_CALUDE_k_range_l3456_345627

theorem k_range (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k^2 - 1 ≤ 0) ↔ k ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_k_range_l3456_345627


namespace NUMINAMATH_CALUDE_typing_time_together_l3456_345653

-- Define the typing rates for Randy and Candy
def randy_rate : ℚ := 1 / 30
def candy_rate : ℚ := 1 / 45

-- Define the combined typing rate
def combined_rate : ℚ := randy_rate + candy_rate

-- Theorem to prove
theorem typing_time_together : (1 : ℚ) / combined_rate = 18 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_together_l3456_345653


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_bound_l3456_345655

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a line passing through the right focus with a slope angle of 60° 
    intersects the right branch of the hyperbola at exactly one point,
    then the eccentricity e of the hyperbola satisfies e ≥ 2. -/
theorem hyperbola_eccentricity_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let slope := Real.tan (π / 3)
  (b / a ≥ slope) → e ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_bound_l3456_345655


namespace NUMINAMATH_CALUDE_gym_treadmills_l3456_345685

def gym_problem (num_gyms : ℕ) (bikes_per_gym : ℕ) (ellipticals_per_gym : ℕ) 
  (bike_cost : ℚ) (total_cost : ℚ) : Prop :=
  let treadmill_cost : ℚ := bike_cost * (3/2)
  let elliptical_cost : ℚ := treadmill_cost * 2
  let total_bike_cost : ℚ := num_gyms * bikes_per_gym * bike_cost
  let total_elliptical_cost : ℚ := num_gyms * ellipticals_per_gym * elliptical_cost
  let treadmill_cost_per_gym : ℚ := (total_cost - total_bike_cost - total_elliptical_cost) / num_gyms
  let treadmills_per_gym : ℚ := treadmill_cost_per_gym / treadmill_cost
  treadmills_per_gym = 5

theorem gym_treadmills : 
  gym_problem 20 10 5 700 455000 := by
  sorry

end NUMINAMATH_CALUDE_gym_treadmills_l3456_345685


namespace NUMINAMATH_CALUDE_hilt_share_money_l3456_345663

/-- The number of people Mrs. Hilt will share the money with -/
def number_of_people (total_amount : ℚ) (amount_per_person : ℚ) : ℚ :=
  total_amount / amount_per_person

/-- Theorem stating that Mrs. Hilt will share the money with 3 people -/
theorem hilt_share_money : 
  number_of_people (3.75 : ℚ) (1.25 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_hilt_share_money_l3456_345663


namespace NUMINAMATH_CALUDE_machine_production_theorem_l3456_345675

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days. -/
theorem machine_production_theorem 
  (x : ℝ) -- x is the number of units produced by 4 machines in 6 days
  (h1 : x > 0) -- x is positive
  : 
  let rate := x / (4 * 6) -- rate of production per machine per day
  16 * rate * 3 = 2 * x := by
sorry

end NUMINAMATH_CALUDE_machine_production_theorem_l3456_345675


namespace NUMINAMATH_CALUDE_richard_cleaning_time_l3456_345692

/-- Richard's room cleaning time in minutes -/
def richard_time : ℕ := 45

/-- Cory's room cleaning time in minutes -/
def cory_time (r : ℕ) : ℕ := r + 3

/-- Blake's room cleaning time in minutes -/
def blake_time (r : ℕ) : ℕ := cory_time r - 4

/-- Total cleaning time for all three people in minutes -/
def total_time : ℕ := 136

theorem richard_cleaning_time :
  richard_time + cory_time richard_time + blake_time richard_time = total_time :=
sorry

end NUMINAMATH_CALUDE_richard_cleaning_time_l3456_345692


namespace NUMINAMATH_CALUDE_final_points_count_l3456_345677

/-- The number of points after performing the insertion operation n times -/
def points_after_operations (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * points_after_operations initial_points k - 1

theorem final_points_count : points_after_operations 2010 3 = 16073 := by
  sorry

end NUMINAMATH_CALUDE_final_points_count_l3456_345677


namespace NUMINAMATH_CALUDE_ab_equals_six_l3456_345634

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_six_l3456_345634


namespace NUMINAMATH_CALUDE_roots_of_equation_l3456_345678

theorem roots_of_equation :
  let f : ℝ → ℝ := λ x => x * (x + 2) + x + 2
  (f (-2) = 0) ∧ (f (-1) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -2 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3456_345678


namespace NUMINAMATH_CALUDE_equation_solution_l3456_345600

theorem equation_solution : ∃ t : ℝ, t = 9/4 ∧ Real.sqrt (3 * Real.sqrt (t - 1)) = (t + 9) ^ (1/4) :=
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3456_345600


namespace NUMINAMATH_CALUDE_nell_card_difference_l3456_345686

/-- Represents Nell's card collection --/
structure CardCollection where
  initial_baseball : Nat
  initial_ace : Nat
  current_baseball : Nat
  current_ace : Nat

/-- Calculates the difference between baseball and Ace cards --/
def card_difference (c : CardCollection) : Int :=
  c.current_baseball - c.current_ace

/-- Theorem stating the difference between Nell's baseball and Ace cards --/
theorem nell_card_difference (nell : CardCollection)
  (h1 : nell.initial_baseball = 438)
  (h2 : nell.initial_ace = 18)
  (h3 : nell.current_baseball = 178)
  (h4 : nell.current_ace = 55) :
  card_difference nell = 123 := by
  sorry

end NUMINAMATH_CALUDE_nell_card_difference_l3456_345686


namespace NUMINAMATH_CALUDE_triangle_toothpicks_l3456_345693

/-- Calculates the number of toothpicks needed for a large equilateral triangle
    with a given base length and border. -/
def toothpicks_for_triangle (base : ℕ) (border : ℕ) : ℕ :=
  let interior_triangles := base * (base + 1) / 2
  let interior_toothpicks := 3 * interior_triangles / 2
  let boundary_toothpicks := 3 * base
  let border_toothpicks := 2 * border + 2
  interior_toothpicks + boundary_toothpicks + border_toothpicks

/-- Theorem stating that a triangle with base 100 and border 100 requires 8077 toothpicks -/
theorem triangle_toothpicks :
  toothpicks_for_triangle 100 100 = 8077 := by
  sorry

end NUMINAMATH_CALUDE_triangle_toothpicks_l3456_345693


namespace NUMINAMATH_CALUDE_green_eyed_students_l3456_345668

theorem green_eyed_students (total : ℕ) (both : ℕ) (neither : ℕ) (green : ℕ) :
  total = 40 →
  3 * green = total - green - both - neither →
  both = 9 →
  neither = 4 →
  green = 9 := by
sorry

end NUMINAMATH_CALUDE_green_eyed_students_l3456_345668


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l3456_345607

/-- The area of a circle with diameter 8 meters is 16π square meters. -/
theorem circle_area_with_diameter_8 :
  ∃ (A : ℝ), A = π * 16 ∧ A = (π * (8 / 2)^2) := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l3456_345607


namespace NUMINAMATH_CALUDE_arc_length_sector_l3456_345615

/-- The arc length of a sector with central angle 2π/3 and radius 3 is 2π. -/
theorem arc_length_sector (α : Real) (r : Real) (l : Real) : 
  α = 2 * Real.pi / 3 → r = 3 → l = α * r → l = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arc_length_sector_l3456_345615


namespace NUMINAMATH_CALUDE_complex_power_sum_l3456_345602

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1 / z^100 = 2 * Real.cos (140 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3456_345602


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l3456_345642

theorem cosine_sum_identity (α : ℝ) : 
  Real.cos (π/4 - α) * Real.cos (α + π/12) - Real.sin (π/4 - α) * Real.sin (α + π/12) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l3456_345642


namespace NUMINAMATH_CALUDE_road_repaving_l3456_345629

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l3456_345629


namespace NUMINAMATH_CALUDE_math_interest_group_problem_l3456_345674

theorem math_interest_group_problem (m n : ℕ) : 
  m * (m - 1) / 2 + m * n + n = 51 → m = 6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_interest_group_problem_l3456_345674


namespace NUMINAMATH_CALUDE_exp_sum_lt_four_l3456_345626

noncomputable def f (x : ℝ) := Real.exp x - x^2 - x

theorem exp_sum_lt_four (x₁ x₂ : ℝ) 
  (h1 : x₁ < Real.log 2) 
  (h2 : x₂ > Real.log 2) 
  (h3 : deriv f x₁ = deriv f x₂) : 
  Real.exp (x₁ + x₂) < 4 := by sorry

end NUMINAMATH_CALUDE_exp_sum_lt_four_l3456_345626


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3456_345662

def total_players : ℕ := 15
def guaranteed_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3456_345662


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l3456_345647

theorem pizza_slices_per_person
  (coworkers : ℕ)
  (pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (h1 : coworkers = 18)
  (h2 : pizzas = 4)
  (h3 : slices_per_pizza = 10)
  : (pizzas * slices_per_pizza) / coworkers = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l3456_345647


namespace NUMINAMATH_CALUDE_min_value_theorem_l3456_345601

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2*x*y) :
  3*x + 4*y ≥ 5 + 2*Real.sqrt 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3456_345601


namespace NUMINAMATH_CALUDE_correct_males_in_orchestra_l3456_345617

/-- The number of males in the orchestra -/
def males_in_orchestra : ℕ := 11

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

/-- The number of musicians in the orchestra -/
def musicians_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

/-- The number of musicians in the band -/
def musicians_in_band : ℕ := 2 * musicians_in_orchestra

/-- The number of musicians in the choir -/
def musicians_in_choir : ℕ := 12 + 17

/-- The total number of musicians in all three groups -/
def total_musicians : ℕ := 98

theorem correct_males_in_orchestra :
  musicians_in_orchestra + musicians_in_band + musicians_in_choir = total_musicians :=
sorry

end NUMINAMATH_CALUDE_correct_males_in_orchestra_l3456_345617


namespace NUMINAMATH_CALUDE_three_fourths_to_sixth_power_l3456_345603

theorem three_fourths_to_sixth_power : (3 / 4 : ℚ) ^ 6 = 729 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_sixth_power_l3456_345603


namespace NUMINAMATH_CALUDE_millet_majority_on_day_three_l3456_345622

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  other : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  let remainingMillet := state.millet * 0.8
  let newMillet := if state.day = 1 then 0.5 else 0.4
  { day := state.day + 1,
    millet := remainingMillet + newMillet,
    other := 0.6 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.4, other := 0.6 }

/-- Theorem stating that on Day 3, more than half of the seeds are millet -/
theorem millet_majority_on_day_three :
  let day3State := nextDay (nextDay initialState)
  day3State.millet / (day3State.millet + day3State.other) > 0.5 := by
  sorry


end NUMINAMATH_CALUDE_millet_majority_on_day_three_l3456_345622


namespace NUMINAMATH_CALUDE_rest_stop_location_l3456_345631

/-- The location of the rest stop between two towns -/
theorem rest_stop_location (town_a town_b rest_stop_fraction : ℚ) : 
  town_a = 30 → 
  town_b = 210 → 
  rest_stop_fraction = 4/5 → 
  town_a + rest_stop_fraction * (town_b - town_a) = 174 := by
sorry

end NUMINAMATH_CALUDE_rest_stop_location_l3456_345631


namespace NUMINAMATH_CALUDE_max_area_difference_l3456_345639

-- Define a rectangle with integer dimensions
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Theorem statement
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 180 ∧ 
    perimeter r2 = 180 ∧ 
    (∀ (r : Rectangle), perimeter r = 180 → 
      area r1 - area r2 ≥ area r1 - area r ∧ 
      area r1 - area r2 ≥ area r - area r2) ∧
    area r1 - area r2 = 1936 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l3456_345639


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l3456_345657

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of 101 -/
def binary_101 : List Bool := [true, false, true]

theorem binary_101_equals_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l3456_345657


namespace NUMINAMATH_CALUDE_parking_lot_cars_l3456_345640

/-- Given a parking lot with observed wheels and wheels per car, calculate the number of cars -/
def number_of_cars (total_wheels : ℕ) (wheels_per_car : ℕ) : ℕ :=
  total_wheels / wheels_per_car

/-- Theorem: In a parking lot with 48 observed wheels and 4 wheels per car, there are 12 cars -/
theorem parking_lot_cars : number_of_cars 48 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l3456_345640


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l3456_345681

/-- Circle type with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles -/
def intersection_line_equation (c1 c2 : Circle) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = -2

/-- Theorem stating that the line passing through the intersection points of the given circles has the equation x + y = -2 -/
theorem intersection_line_of_circles :
  let c1 : Circle := { center := (-4, -10), radius := 15 }
  let c2 : Circle := { center := (8, 6), radius := Real.sqrt 104 }
  ∀ p, p ∈ { p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 } ∩
           { p | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 } →
  intersection_line_equation c1 c2 p :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l3456_345681


namespace NUMINAMATH_CALUDE_correct_tax_distribution_l3456_345660

/-- Represents the tax calculation for three individuals based on their yields -/
def tax_calculation (total_tax : ℚ) (yield1 yield2 yield3 : ℕ) : Prop :=
  let total_yield := yield1 + yield2 + yield3
  let tax1 := total_tax * (yield1 : ℚ) / total_yield
  let tax2 := total_tax * (yield2 : ℚ) / total_yield
  let tax3 := total_tax * (yield3 : ℚ) / total_yield
  (tax1 = 1 + 3/32) ∧ (tax2 = 1 + 1/4) ∧ (tax3 = 1 + 13/32)

/-- Theorem stating the correct tax distribution for the given problem -/
theorem correct_tax_distribution :
  tax_calculation (15/4) 7 8 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_tax_distribution_l3456_345660


namespace NUMINAMATH_CALUDE_triangle_area_l3456_345621

/-- Given a triangle with perimeter 24 cm and inradius 2.5 cm, its area is 30 cm². -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 24 → r = 2.5 → A = r * (P / 2) → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3456_345621


namespace NUMINAMATH_CALUDE_sales_solution_l3456_345605

def sales_problem (m1 m3 m4 m5 m6 avg : ℕ) : Prop :=
  ∃ m2 : ℕ, 
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg ∧
    m2 = 5744

theorem sales_solution :
  sales_problem 5266 5864 6122 6588 4916 5750 :=
by sorry

end NUMINAMATH_CALUDE_sales_solution_l3456_345605


namespace NUMINAMATH_CALUDE_simplify_expression_l3456_345644

theorem simplify_expression (z : ℝ) : (5 - 2*z) - (4 + 5*z) = 1 - 7*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3456_345644


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l3456_345612

/-- Given a group of families, some with children and some without, 
    calculate the average number of children in families that have children. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  (h4 : childless_families < total_families) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l3456_345612


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l3456_345658

theorem abs_sum_inequality (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ x ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l3456_345658


namespace NUMINAMATH_CALUDE_sum_of_digits_3n_l3456_345697

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If n has sum of digits 100 and 44n has sum of digits 800, then 3n has sum of digits 300 -/
theorem sum_of_digits_3n 
  (n : ℕ) 
  (h1 : sum_of_digits n = 100) 
  (h2 : sum_of_digits (44 * n) = 800) : 
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3n_l3456_345697


namespace NUMINAMATH_CALUDE_point_on_135_degree_angle_l3456_345609

/-- Given a point (√4, a) on the terminal side of the angle 135°, prove that a = 2 -/
theorem point_on_135_degree_angle (a : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 4 ∧ y = a ∧ 
   x = 2 * Real.cos (135 * π / 180) ∧ 
   y = 2 * Real.sin (135 * π / 180)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_135_degree_angle_l3456_345609


namespace NUMINAMATH_CALUDE_opposite_of_three_abs_l3456_345608

theorem opposite_of_three_abs (x : ℝ) : x = -3 → |x + 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_abs_l3456_345608


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3456_345632

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 1/45) : x^2 - y^2 = 8/675 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3456_345632


namespace NUMINAMATH_CALUDE_sum_of_squares_l3456_345650

theorem sum_of_squares (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 9)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 405 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3456_345650


namespace NUMINAMATH_CALUDE_ram_money_l3456_345635

/-- Given the ratio of money between Ram and Gopal, and between Gopal and Krishan,
    prove that Ram has 637 rupees when Krishan has 3757 rupees. -/
theorem ram_money (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  krishan = 3757 →
  ram = 637 := by
  sorry

end NUMINAMATH_CALUDE_ram_money_l3456_345635


namespace NUMINAMATH_CALUDE_range_of_a_l3456_345667

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ∈ Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3456_345667


namespace NUMINAMATH_CALUDE_final_amount_calculation_l3456_345643

/-- Calculates the final amount after two years of compound interest with different rates for each year -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that the final amount after two years is approximately 5518.80 Rs -/
theorem final_amount_calculation :
  ∃ ε > 0, |final_amount 5253 0.02 0.03 - 5518.80| < ε :=
by
  sorry

#eval final_amount 5253 0.02 0.03

end NUMINAMATH_CALUDE_final_amount_calculation_l3456_345643


namespace NUMINAMATH_CALUDE_june_production_l3456_345618

/-- Represents a restaurant's daily pizza and hot dog production. -/
structure RestaurantProduction where
  hotDogs : ℕ
  pizzaDifference : ℕ

/-- Calculates the total number of pizzas and hot dogs made in June. -/
def totalInJune (r : RestaurantProduction) : ℕ :=
  30 * (r.hotDogs + (r.hotDogs + r.pizzaDifference))

/-- Theorem stating the total production in June for a specific restaurant. -/
theorem june_production (r : RestaurantProduction) 
  (h1 : r.hotDogs = 60) 
  (h2 : r.pizzaDifference = 40) : 
  totalInJune r = 4800 := by
  sorry

#eval totalInJune ⟨60, 40⟩

end NUMINAMATH_CALUDE_june_production_l3456_345618


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3456_345665

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers :
  i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 1 + i :=
by
  sorry

-- Define the property i^4 = 1
axiom i_fourth_power : i^4 = 1

-- Define i as the imaginary unit
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_sum_of_i_powers_l3456_345665


namespace NUMINAMATH_CALUDE_problem_statements_l3456_345659

theorem problem_statements :
  -- Statement 1
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  -- Statement 2
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d ∧
  -- Statement 3
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3456_345659


namespace NUMINAMATH_CALUDE_max_type_A_books_l3456_345623

/-- Represents the unit price of type A books -/
def price_A : ℝ := 20

/-- Represents the unit price of type B books -/
def price_B : ℝ := 15

/-- Represents the total number of books to be purchased -/
def total_books : ℕ := 300

/-- Represents the discount factor for type A books -/
def discount_A : ℝ := 0.9

/-- Represents the maximum total cost -/
def max_cost : ℝ := 5100

/-- Theorem stating the maximum number of type A books that can be purchased -/
theorem max_type_A_books : 
  ∃ (n : ℕ), n ≤ total_books ∧ 
  discount_A * price_A * n + price_B * (total_books - n) ≤ max_cost ∧
  ∀ (m : ℕ), m > n → discount_A * price_A * m + price_B * (total_books - m) > max_cost :=
sorry

end NUMINAMATH_CALUDE_max_type_A_books_l3456_345623


namespace NUMINAMATH_CALUDE_unique_solution_k_squared_minus_2016_equals_3_to_n_l3456_345670

theorem unique_solution_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n := by sorry

end NUMINAMATH_CALUDE_unique_solution_k_squared_minus_2016_equals_3_to_n_l3456_345670


namespace NUMINAMATH_CALUDE_lucy_popsicle_purchase_l3456_345645

theorem lucy_popsicle_purchase (lucy_money : ℕ) (popsicle_cost : ℕ) : 
  lucy_money = 2540 → popsicle_cost = 175 → (lucy_money / popsicle_cost : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_popsicle_purchase_l3456_345645


namespace NUMINAMATH_CALUDE_point_on_bisector_l3456_345619

/-- 
Given a point (a, 2) in the second quadrant and on the angle bisector of the coordinate axes,
prove that a = -2.
-/
theorem point_on_bisector (a : ℝ) :
  (a < 0) →  -- Point is in the second quadrant
  (a = -2) →  -- Point is on the angle bisector
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_point_on_bisector_l3456_345619


namespace NUMINAMATH_CALUDE_hotel_bubble_bath_amount_l3456_345696

/-- Calculates the total amount of bubble bath needed for a hotel --/
def total_bubble_bath_needed (luxury_suites rooms_for_couples single_rooms family_rooms : ℕ)
  (luxury_capacity couple_capacity single_capacity family_capacity : ℕ)
  (adult_bath_ml child_bath_ml : ℕ) : ℕ :=
  let total_guests := 
    luxury_suites * luxury_capacity + 
    rooms_for_couples * couple_capacity + 
    single_rooms * single_capacity + 
    family_rooms * family_capacity
  let adults := (2 * total_guests) / 3
  let children := total_guests - adults
  adults * adult_bath_ml + children * child_bath_ml

/-- The amount of bubble bath needed for the given hotel configuration --/
theorem hotel_bubble_bath_amount : 
  total_bubble_bath_needed 6 12 15 4 5 2 1 7 20 15 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bubble_bath_amount_l3456_345696


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3456_345651

theorem simplify_complex_fraction :
  1 / ((1 / (Real.sqrt 2 + 2)) + (3 / (2 * Real.sqrt 3 - 1))) =
  1 / (25 - 11 * Real.sqrt 2 + 6 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3456_345651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3456_345683

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3456_345683


namespace NUMINAMATH_CALUDE_extra_sodas_l3456_345649

/-- Given that Robin bought 11 sodas and drank 3 sodas, prove that the number of extra sodas is 8. -/
theorem extra_sodas (total : ℕ) (drank : ℕ) (h1 : total = 11) (h2 : drank = 3) :
  total - drank = 8 := by
  sorry

end NUMINAMATH_CALUDE_extra_sodas_l3456_345649


namespace NUMINAMATH_CALUDE_johnson_class_activity_c_contribution_l3456_345628

/-- Calculates the individual student contribution for an activity -/
def individualContribution (totalCost classFunds numStudents : ℚ) : ℚ :=
  (totalCost - classFunds) / numStudents

/-- Proves that Mrs. Johnson's class individual contribution for Activity C is $3.60 -/
theorem johnson_class_activity_c_contribution :
  let totalCost : ℚ := 150
  let classFunds : ℚ := 60
  let numStudents : ℚ := 25
  individualContribution totalCost classFunds numStudents = 3.60 := by
sorry

end NUMINAMATH_CALUDE_johnson_class_activity_c_contribution_l3456_345628


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3456_345669

theorem largest_four_digit_congruent_to_17_mod_26 :
  (∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧
    ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) →
  (∃ x : ℕ, x = 9972 ∧ x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧
    ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3456_345669


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3456_345698

/-- An arithmetic sequence with non-zero common difference -/
def arithmetic_seq (a : ℕ → ℝ) := ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_seq (b : ℕ → ℝ) := ∃ r ≠ 0, ∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_seq a)
  (h_geom : geometric_seq b)
  (h_relation : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l3456_345698


namespace NUMINAMATH_CALUDE_sara_pumpkins_l3456_345656

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := 20

/-- The original number of pumpkins Sara grew -/
def original_pumpkins : ℕ := pumpkins_eaten + pumpkins_left

theorem sara_pumpkins : original_pumpkins = 43 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l3456_345656


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_k_range_l3456_345641

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k + 3}

-- State the theorem
theorem intersection_nonempty_implies_k_range (k : ℝ) :
  (M ∩ N k).Nonempty → k ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_k_range_l3456_345641


namespace NUMINAMATH_CALUDE_art_price_increase_theorem_l3456_345620

/-- Calculates the price increase of an art piece given its initial price and a multiplier for its future price. -/
def art_price_increase (initial_price : ℕ) (price_multiplier : ℕ) : ℕ :=
  (price_multiplier * initial_price) - initial_price

/-- Theorem stating that for an art piece with an initial price of $4000 and a future price 3 times the initial price, the price increase is $8000. -/
theorem art_price_increase_theorem :
  art_price_increase 4000 3 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_art_price_increase_theorem_l3456_345620


namespace NUMINAMATH_CALUDE_domain_transformation_l3456_345616

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc 0 1

-- Define the domain of f(√(2x-1))
def domain_f_sqrt : Set ℝ := Set.Icc 1 (5/2)

-- State the theorem
theorem domain_transformation (h : ∀ x ∈ domain_f_shifted, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_sqrt, f (Real.sqrt (2 * x - 1)) = f (Real.sqrt (2 * x - 1)) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l3456_345616


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3456_345613

theorem inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x + b > c ↔ x < 4) →
  (∀ x : ℝ, a * (x - 3) + b > c ↔ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3456_345613


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3456_345630

theorem coefficient_x_squared_in_binomial_expansion :
  let n : ℕ := 8
  let k : ℕ := 3
  let coeff : ℤ := (-1)^k * 2^k * Nat.choose n k
  coeff = -448 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l3456_345630


namespace NUMINAMATH_CALUDE_langsley_commute_time_l3456_345684

theorem langsley_commute_time :
  let first_bus_time : ℕ := 40
  let first_bus_delay : ℕ := 10
  let first_wait_time : ℕ := 10
  let second_bus_time : ℕ := 50
  let second_bus_delay : ℕ := 5
  let second_wait_time : ℕ := 15
  let third_bus_time : ℕ := 95
  let third_bus_delay : ℕ := 15
  first_bus_time + first_bus_delay + first_wait_time +
  second_bus_time + second_bus_delay + second_wait_time +
  third_bus_time + third_bus_delay = 240 := by
sorry

end NUMINAMATH_CALUDE_langsley_commute_time_l3456_345684


namespace NUMINAMATH_CALUDE_count_special_numbers_l3456_345652

/-- Represents a relation where two integers have the same digits (possibly rearranged) -/
def digit_rearrangement (a b : ℕ) : Prop := sorry

/-- Counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is an 11-digit positive integer -/
def is_11_digit_positive (n : ℕ) : Prop :=
  digit_count n = 11 ∧ n > 0

/-- The main theorem stating the count of numbers satisfying the given conditions -/
theorem count_special_numbers :
  ∃ (S : Finset ℕ),
    (∀ K ∈ S, is_11_digit_positive K ∧ 2 ∣ K ∧ 3 ∣ K ∧ 5 ∣ K ∧
      ∃ K', digit_rearrangement K K' ∧
        7 ∣ K' ∧ 11 ∣ K' ∧ 13 ∣ K' ∧ 17 ∣ K' ∧ 101 ∣ K' ∧ 9901 ∣ K') ∧
    S.card = 3628800 :=
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l3456_345652


namespace NUMINAMATH_CALUDE_angle_cde_is_85_l3456_345654

-- Define the points
variable (A B C D E : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the conditions
variable (h1 : angle A B C = 90)
variable (h2 : angle B C D = 90)
variable (h3 : angle C D A = 90)
variable (h4 : angle A E B = 50)
variable (h5 : angle B E D = angle B D E)

-- State the theorem
theorem angle_cde_is_85 : angle C D E = 85 := by sorry

end NUMINAMATH_CALUDE_angle_cde_is_85_l3456_345654


namespace NUMINAMATH_CALUDE_circle_transformation_l3456_345624

/-- Given a circle and a transformation, prove the equation of the resulting shape -/
theorem circle_transformation (x y x' y' : ℝ) : 
  (x^2 + y^2 = 4) → (x' = 2*x ∧ y' = 3*y) → ((x'^2 / 16) + (y'^2 / 36) = 1) := by
sorry

end NUMINAMATH_CALUDE_circle_transformation_l3456_345624


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l3456_345680

/-- Given a car traveling for two hours with a speed of 80 km/h in the first hour
    and an average speed of 60 km/h over the two hours,
    prove that the speed in the second hour must be 40 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 80)
  (h2 : average_speed = 60) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 40 := by
sorry


end NUMINAMATH_CALUDE_car_speed_second_hour_l3456_345680


namespace NUMINAMATH_CALUDE_parabola_slope_AF_l3456_345604

-- Define the parabola
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem parabola_slope_AF (C : Parabola) (A F : Point) :
  A.x = -2 ∧ A.y = 3 ∧  -- A is (-2, 3)
  A.x = -C.p/2 ∧        -- A is on the directrix
  F.x = C.p/2 ∧ F.y = 0 -- F is the focus
  →
  (F.y - A.y) / (F.x - A.x) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_AF_l3456_345604


namespace NUMINAMATH_CALUDE_estimate_correct_l3456_345625

/-- Represents the sample data of homework times in minutes -/
def sample_data : List Nat := [75, 80, 85, 65, 95, 100, 70, 55, 65, 75, 85, 110, 120, 80, 85, 80, 75, 90, 90, 95, 70, 60, 60, 75, 90, 95, 65, 75, 80, 80]

/-- The total number of students in the school -/
def total_students : Nat := 2100

/-- The size of the sample -/
def sample_size : Nat := 30

/-- The threshold time in minutes -/
def threshold : Nat := 90

/-- Counts the number of elements in the list that are greater than or equal to the threshold -/
def count_above_threshold (data : List Nat) (threshold : Nat) : Nat :=
  data.filter (λ x => x ≥ threshold) |>.length

/-- Estimates the number of students in the entire school population who spend at least the threshold time on homework -/
def estimate_students_above_threshold : Nat :=
  let count := count_above_threshold sample_data threshold
  (count * total_students) / sample_size

theorem estimate_correct : estimate_students_above_threshold = 630 := by
  sorry

end NUMINAMATH_CALUDE_estimate_correct_l3456_345625


namespace NUMINAMATH_CALUDE_distance_traveled_l3456_345688

/-- Given a person traveling at 40 km/hr for 6 hours, prove that the distance traveled is 240 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 40) (h2 : time = 6) :
  speed * time = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3456_345688


namespace NUMINAMATH_CALUDE_investment_problem_l3456_345691

/-- Given two investors P and Q, where P invested 40000 and their profit ratio is 2:3,
    prove that Q's investment is 60000. -/
theorem investment_problem (P Q : ℕ) (h1 : P = 40000) (h2 : 2 * Q = 3 * P) : Q = 60000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3456_345691


namespace NUMINAMATH_CALUDE_police_can_see_bandit_l3456_345690

/-- Represents a point in the city grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a policeman -/
structure Policeman where
  position : Point
  canSeeInfinitely : Bool

/-- Represents the bandit -/
structure Bandit where
  position : Point

/-- Represents the city -/
structure City where
  grid : Set Point
  police : Set Policeman
  bandit : Bandit

/-- Represents the initial configuration of the city -/
def initialCity : City :=
  { grid := Set.univ,
    police := { p | ∃ k : ℤ, p.position = ⟨100 * k, 0⟩ ∧ p.canSeeInfinitely = true },
    bandit := ⟨⟨0, 0⟩⟩ }  -- Arbitrary initial position for the bandit

/-- Represents a strategy for the police -/
def PoliceStrategy := City → City

/-- Theorem: There exists a police strategy that guarantees seeing the bandit -/
theorem police_can_see_bandit :
  ∃ (strategy : PoliceStrategy), ∀ (c : City),
    ∃ (t : ℕ), ∃ (p : Policeman),
      p ∈ (strategy^[t] c).police ∧
      (strategy^[t] c).bandit.position.x = p.position.x ∨
      (strategy^[t] c).bandit.position.y = p.position.y :=
sorry

end NUMINAMATH_CALUDE_police_can_see_bandit_l3456_345690


namespace NUMINAMATH_CALUDE_sector_central_angle_l3456_345694

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 8) (h2 : s.area = 4) :
  ∃ (r l θ : ℝ), r > 0 ∧ l > 0 ∧ θ > 0 ∧ 
  2 * r + l = s.perimeter ∧
  1 / 2 * l * r = s.area ∧
  θ = l / r ∧
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3456_345694


namespace NUMINAMATH_CALUDE_token_passing_game_termination_l3456_345638

/-- Represents the state of the token-passing game -/
structure GameState where
  tokens : Fin 1994 → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def make_move (state : GameState) (i : Fin 1994) : GameState :=
  sorry

/-- Predicate to check if the game has terminated -/
def is_terminated (state : GameState) : Prop :=
  ∀ i : Fin 1994, state.tokens i ≤ 1

/-- The main theorem about the token-passing game -/
theorem token_passing_game_termination 
  (n : ℕ) (initial_state : GameState) 
  (h_initial : ∃ i : Fin 1994, initial_state.tokens i = n ∧ 
               ∀ j : Fin 1994, j ≠ i → initial_state.tokens j = 0) 
  (h_total : initial_state.total_tokens = n) :
  (n < 1994 → ∃ (final_state : GameState), is_terminated final_state) ∧
  (n = 1994 → ∀ (state : GameState), ¬is_terminated state) :=
sorry

end NUMINAMATH_CALUDE_token_passing_game_termination_l3456_345638


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3456_345673

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  (Real.sin (12 * π / 180) * (4 * (Real.cos (12 * π / 180))^2 - 2)) = 
  -4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3456_345673


namespace NUMINAMATH_CALUDE_apple_bag_theorem_l3456_345610

/-- Represents the number of apples in a bag -/
inductive BagSize
  | small : BagSize  -- 6 apples
  | large : BagSize  -- 12 apples

/-- The total number of apples from all bags -/
def totalApples (bags : List BagSize) : Nat :=
  bags.foldl (fun sum bag => sum + match bag with
    | BagSize.small => 6
    | BagSize.large => 12) 0

/-- Theorem stating the possible total numbers of apples -/
theorem apple_bag_theorem (bags : List BagSize) :
  (totalApples bags ≥ 70 ∧ totalApples bags ≤ 80) →
  (totalApples bags = 72 ∨ totalApples bags = 78) := by
  sorry

end NUMINAMATH_CALUDE_apple_bag_theorem_l3456_345610


namespace NUMINAMATH_CALUDE_f_extrema_l3456_345606

-- Define the function f
def f (x y : ℝ) : ℝ := x^3 + y^3 + 6*x*y

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | -3 ≤ p.1 ∧ p.1 ≤ 1 ∧ -3 ≤ p.2 ∧ p.2 ≤ 2}

theorem f_extrema :
  ∃ (min_point max_point : ℝ × ℝ),
    min_point ∈ rectangle ∧
    max_point ∈ rectangle ∧
    (∀ p ∈ rectangle, f min_point.1 min_point.2 ≤ f p.1 p.2) ∧
    (∀ p ∈ rectangle, f p.1 p.2 ≤ f max_point.1 max_point.2) ∧
    min_point = (-3, 2) ∧
    max_point = (1, 2) ∧
    f min_point.1 min_point.2 = -55 ∧
    f max_point.1 max_point.2 = 21 :=
  sorry


end NUMINAMATH_CALUDE_f_extrema_l3456_345606


namespace NUMINAMATH_CALUDE_gina_remaining_money_l3456_345682

def initial_amount : ℚ := 400

def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def charity_fraction : ℚ := 1/5

def remaining_amount : ℚ := initial_amount * (1 - mom_fraction - clothes_fraction - charity_fraction)

theorem gina_remaining_money :
  remaining_amount = 170 := by sorry

end NUMINAMATH_CALUDE_gina_remaining_money_l3456_345682


namespace NUMINAMATH_CALUDE_expand_expression_l3456_345687

theorem expand_expression (m n : ℝ) : (2*m + n - 1) * (2*m - n + 1) = 4*m^2 - n^2 + 2*n - 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3456_345687


namespace NUMINAMATH_CALUDE_marias_carrots_l3456_345646

theorem marias_carrots (initial thrown_out picked_more final : ℕ) : 
  thrown_out = 11 →
  picked_more = 15 →
  final = 52 →
  initial - thrown_out + picked_more = final →
  initial = 48 := by
sorry

end NUMINAMATH_CALUDE_marias_carrots_l3456_345646
