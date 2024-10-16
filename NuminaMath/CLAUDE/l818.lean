import Mathlib

namespace NUMINAMATH_CALUDE_river_road_cars_l818_81851

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →
  buses = cars - 40 →
  cars = 60 := by
sorry

end NUMINAMATH_CALUDE_river_road_cars_l818_81851


namespace NUMINAMATH_CALUDE_square_root_of_four_l818_81848

theorem square_root_of_four : 
  {x : ℝ | x ^ 2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l818_81848


namespace NUMINAMATH_CALUDE_cylinder_radius_equals_8_l818_81808

/-- Given a cylinder and a cone with equal volumes, prove that the cylinder's radius is 8 cm -/
theorem cylinder_radius_equals_8 (h_cyl : ℝ) (r_cyl : ℝ) (h_cone : ℝ) (r_cone : ℝ)
  (h_cyl_val : h_cyl = 2)
  (h_cone_val : h_cone = 6)
  (r_cone_val : r_cone = 8)
  (volume_equal : π * r_cyl^2 * h_cyl = (1/3) * π * r_cone^2 * h_cone) :
  r_cyl = 8 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_equals_8_l818_81808


namespace NUMINAMATH_CALUDE_square_dissection_interior_rectangle_l818_81870

-- Define a rectangle type
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

-- Define the square dissection
def SquareDissection (n : ℕ) (rectangles : Finset Rectangle) : Prop :=
  n > 1 ∧
  rectangles.card = n ∧
  (∀ r ∈ rectangles, r.x ≥ 0 ∧ r.y ≥ 0 ∧ r.x + r.width ≤ 1 ∧ r.y + r.height ≤ 1) ∧
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
    ∃ r ∈ rectangles, r.x < x ∧ x < r.x + r.width ∧ r.y < y ∧ y < r.y + r.height)

-- Define an interior rectangle
def InteriorRectangle (r : Rectangle) : Prop :=
  r.x > 0 ∧ r.y > 0 ∧ r.x + r.width < 1 ∧ r.y + r.height < 1

-- The theorem to be proved
theorem square_dissection_interior_rectangle
  (n : ℕ) (rectangles : Finset Rectangle) (h : SquareDissection n rectangles) :
  ∃ r ∈ rectangles, InteriorRectangle r := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_interior_rectangle_l818_81870


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l818_81819

/-- The weight of a bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of a canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating that one bowling ball weighs 14 pounds -/
theorem bowling_ball_weight_is_14 : bowling_ball_weight = 14 := by
  have h1 : 8 * bowling_ball_weight = 4 * canoe_weight := sorry
  have h2 : 3 * canoe_weight = 84 := sorry
  sorry


end NUMINAMATH_CALUDE_bowling_ball_weight_is_14_l818_81819


namespace NUMINAMATH_CALUDE_digit_101_of_7_over_26_l818_81867

theorem digit_101_of_7_over_26 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a : ℕ → ℕ), 
    (∀ n, a n < 10) ∧ 
    (∀ n, (7 * 10^(n+1)) / 26 % 10 = a n) ∧ 
    a 100 = d) := by
  sorry

end NUMINAMATH_CALUDE_digit_101_of_7_over_26_l818_81867


namespace NUMINAMATH_CALUDE_min_value_sum_l818_81828

theorem min_value_sum (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4) : 
  x₁ / (1 - x₁^2) + x₂ / (1 - x₂^2) + x₃ / (1 - x₃^2) + x₄ / (1 - x₄^2) ≥ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l818_81828


namespace NUMINAMATH_CALUDE_diff_of_squares_equals_fifth_power_l818_81864

theorem diff_of_squares_equals_fifth_power (a : ℤ) :
  ∃ x y : ℤ, x^2 - y^2 = a^5 := by
sorry

end NUMINAMATH_CALUDE_diff_of_squares_equals_fifth_power_l818_81864


namespace NUMINAMATH_CALUDE_division_problem_l818_81850

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 271 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 30 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l818_81850


namespace NUMINAMATH_CALUDE_school_population_theorem_l818_81814

/-- Represents the school population statistics -/
structure SchoolPopulation where
  y : ℕ  -- Total number of students
  x : ℚ  -- Percentage of boys that 162 students represent
  z : ℚ  -- Percentage of girls in the school

/-- The conditions given in the problem -/
def school_conditions (pop : SchoolPopulation) : Prop :=
  (162 : ℚ) = pop.x / 100 * (1/2 : ℚ) * pop.y ∧ 
  pop.z = 100 - 50

/-- The theorem to be proved -/
theorem school_population_theorem (pop : SchoolPopulation) 
  (h : school_conditions pop) : 
  pop.z = 50 ∧ pop.x = 32400 / pop.y := by
  sorry


end NUMINAMATH_CALUDE_school_population_theorem_l818_81814


namespace NUMINAMATH_CALUDE_least_days_to_double_l818_81825

/-- Given a loan of 50 dollars with a 10% simple interest rate per day,
    this function calculates the total amount to be repaid after a given number of days. -/
def totalAmount (days : ℕ) : ℚ :=
  50 + 50 * (10 : ℚ) / 100 * days

/-- This theorem states that 10 is the least integer number of days
    after which the total amount to be repaid is at least twice the borrowed amount. -/
theorem least_days_to_double : 
  (∀ d : ℕ, d < 10 → totalAmount d < 100) ∧ 
  totalAmount 10 ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_least_days_to_double_l818_81825


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l818_81805

/-- A geometric sequence -/
def geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = r * α n

/-- The theorem stating that if α_4 · α_5 · α_6 = 27 in a geometric sequence, then α_5 = 3 -/
theorem geometric_sequence_property (α : ℕ → ℝ) :
  geometric_sequence α → α 4 * α 5 * α 6 = 27 → α 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l818_81805


namespace NUMINAMATH_CALUDE_soccer_team_uniform_numbers_l818_81817

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The uniform numbers of Emily, Fiona, and Grace -/
structure UniformNumbers where
  emily : ℕ
  fiona : ℕ
  grace : ℕ

/-- The conditions of the soccer team uniform numbers problem -/
structure SoccerTeamConditions (u : UniformNumbers) : Prop where
  emily_prime : isTwoDigitPrime u.emily
  fiona_prime : isTwoDigitPrime u.fiona
  grace_prime : isTwoDigitPrime u.grace
  emily_fiona_sum : u.emily + u.fiona = 23
  emily_grace_sum : u.emily + u.grace = 31

theorem soccer_team_uniform_numbers (u : UniformNumbers) 
  (h : SoccerTeamConditions u) : u.grace = 19 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_uniform_numbers_l818_81817


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l818_81816

/-- The bowtie operation defined as a ⋈ b = a + √(b + √(b + √(b + ...))) -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem: If 3 ⋈ z = 9, then z = 30 -/
theorem bowtie_equation_solution :
  ∃ z : ℝ, bowtie 3 z = 9 ∧ z = 30 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l818_81816


namespace NUMINAMATH_CALUDE_working_hours_growth_equation_l818_81893

-- Define the initial and final average working hours
def initial_hours : ℝ := 40
def final_hours : ℝ := 48.4

-- Define the growth rate variable
variable (x : ℝ)

-- State the theorem
theorem working_hours_growth_equation :
  initial_hours * (1 + x)^2 = final_hours := by
  sorry

end NUMINAMATH_CALUDE_working_hours_growth_equation_l818_81893


namespace NUMINAMATH_CALUDE_homework_group_existence_l818_81801

theorem homework_group_existence :
  ∀ (S : Finset ℕ) (f : Finset ℕ → Finset ℕ → Prop),
    S.card = 21 →
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
      (f {a, b, c} {0} ∨ f {a, b, c} {1}) ∧
      ¬(f {a, b, c} {0} ∧ f {a, b, c} {1})) →
    ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 4 ∧
      (∀ a b c : ℕ, a ∈ T → b ∈ T → c ∈ T → a ≠ b → b ≠ c → a ≠ c →
        (f {a, b, c} {0} ∨ f {a, b, c} {1})) :=
by sorry


end NUMINAMATH_CALUDE_homework_group_existence_l818_81801


namespace NUMINAMATH_CALUDE_max_value_xyz_l818_81822

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + 2*y + 3*z = 1) :
  x^3 * y^2 * z ≤ 2048 / 11^6 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^3 * y₀^2 * z₀ = 2048 / 11^6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_l818_81822


namespace NUMINAMATH_CALUDE_f_2011_value_l818_81840

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_def : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_value_l818_81840


namespace NUMINAMATH_CALUDE_cube_root_inequality_l818_81860

theorem cube_root_inequality (x : ℝ) : 
  (x ^ (1/3) : ℝ) - 3 / ((x ^ (1/3) : ℝ) + 4) ≤ 0 ↔ -27 < x ∧ x < -1 := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l818_81860


namespace NUMINAMATH_CALUDE_probability_higher_first_lower_second_l818_81859

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes (s : Finset ℕ) : Finset (ℕ × ℕ) :=
  s.product s |>.filter (fun (a, b) => a > b)

theorem probability_higher_first_lower_second :
  (favorable_outcomes card_set).card / (card_set.card * card_set.card : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_higher_first_lower_second_l818_81859


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l818_81882

/-- Calculate the overall gain percentage for three items -/
theorem overall_gain_percentage
  (cycle_cp cycle_sp scooter_cp scooter_sp skateboard_cp skateboard_sp : ℚ)
  (h_cycle_cp : cycle_cp = 900)
  (h_cycle_sp : cycle_sp = 1170)
  (h_scooter_cp : scooter_cp = 15000)
  (h_scooter_sp : scooter_sp = 18000)
  (h_skateboard_cp : skateboard_cp = 2000)
  (h_skateboard_sp : skateboard_sp = 2400) :
  let total_cp := cycle_cp + scooter_cp + skateboard_cp
  let total_sp := cycle_sp + scooter_sp + skateboard_sp
  let gain_percentage := (total_sp - total_cp) / total_cp * 100
  ∃ (ε : ℚ), abs (gain_percentage - 20.50) < ε ∧ ε > 0 ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l818_81882


namespace NUMINAMATH_CALUDE_roses_picked_later_l818_81826

/-- Given a florist's rose inventory, prove the number of roses picked later. -/
theorem roses_picked_later
  (initial_roses : ℕ)
  (roses_sold : ℕ)
  (final_roses : ℕ)
  (h1 : initial_roses = 5)
  (h2 : roses_sold = 3)
  (h3 : final_roses = 36)
  : final_roses - (initial_roses - roses_sold) = 34 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_later_l818_81826


namespace NUMINAMATH_CALUDE_infinite_divisibility_sequence_l818_81861

theorem infinite_divisibility_sequence : 
  ∃ (a : ℕ → ℕ), ∀ n, (a n)^2 ∣ (2^(a n) + 3^(a n)) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisibility_sequence_l818_81861


namespace NUMINAMATH_CALUDE_no_extrema_on_open_interval_l818_81810

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem no_extrema_on_open_interval : 
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≤ f x)) ∧
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≥ f x)) :=
by sorry

end NUMINAMATH_CALUDE_no_extrema_on_open_interval_l818_81810


namespace NUMINAMATH_CALUDE_three_Z_five_equals_two_l818_81898

-- Define the Z operation
def Z (a b : ℝ) : ℝ := b + 5*a - 2*a^2

-- Theorem statement
theorem three_Z_five_equals_two : Z 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_five_equals_two_l818_81898


namespace NUMINAMATH_CALUDE_white_tile_count_in_specific_arrangement_l818_81835

/-- Represents the tiling arrangement of a large square --/
structure TilingArrangement where
  side_length : ℕ
  black_tile_count : ℕ
  black_tile_size : ℕ
  red_tile_size : ℕ
  white_tile_width : ℕ
  white_tile_length : ℕ

/-- Calculates the number of white tiles in the tiling arrangement --/
def count_white_tiles (t : TilingArrangement) : ℕ :=
  sorry

/-- Theorem stating the number of white tiles in the specific arrangement --/
theorem white_tile_count_in_specific_arrangement :
  ∀ t : TilingArrangement,
    t.side_length = 81 ∧
    t.black_tile_count = 81 ∧
    t.black_tile_size = 1 ∧
    t.red_tile_size = 2 ∧
    t.white_tile_width = 1 ∧
    t.white_tile_length = 2 →
    count_white_tiles t = 2932 :=
  sorry

end NUMINAMATH_CALUDE_white_tile_count_in_specific_arrangement_l818_81835


namespace NUMINAMATH_CALUDE_sum_of_satisfying_numbers_is_34_l818_81847

def satisfies_condition (n : ℕ) : Prop :=
  1.5 * (n : ℝ) - 5.5 > 4.5

def sum_of_satisfying_numbers : ℕ :=
  (Finset.range 4).sum (fun i => i + 7)

theorem sum_of_satisfying_numbers_is_34 :
  sum_of_satisfying_numbers = 34 ∧
  ∀ n, 7 ≤ n → n ≤ 10 → satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_sum_of_satisfying_numbers_is_34_l818_81847


namespace NUMINAMATH_CALUDE_sum_of_squares_is_45_l818_81834

/-- Represents the ages of Alice, Bob, and Charlie -/
structure Ages where
  alice : ℕ
  bob : ℕ
  charlie : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  (3 * ages.alice + 2 * ages.bob = 4 * ages.charlie) ∧
  (3 * ages.charlie^2 = 4 * ages.alice^2 + 2 * ages.bob^2) ∧
  (Nat.gcd ages.alice ages.bob = 1) ∧
  (Nat.gcd ages.alice ages.charlie = 1) ∧
  (Nat.gcd ages.bob ages.charlie = 1)

/-- The theorem to be proved -/
theorem sum_of_squares_is_45 (ages : Ages) :
  satisfies_conditions ages →
  ages.alice^2 + ages.bob^2 + ages.charlie^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_45_l818_81834


namespace NUMINAMATH_CALUDE_bowling_team_score_l818_81873

theorem bowling_team_score (total_score : ℕ) (bowler1 bowler2 bowler3 : ℕ) : 
  total_score = 810 →
  bowler1 = bowler2 / 3 →
  bowler2 = 3 * bowler3 →
  bowler1 + bowler2 + bowler3 = total_score →
  bowler3 = 162 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_score_l818_81873


namespace NUMINAMATH_CALUDE_smallest_perimeter_l818_81821

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

def DOnAC (t : Triangle) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.C

def BDPerpAC (t : Triangle) : Prop :=
  (t.B.1 - t.D.1) * (t.A.1 - t.C.1) + (t.B.2 - t.D.2) * (t.A.2 - t.C.2) = 0

def ACCDEven (t : Triangle) : Prop :=
  ∃ m n : ℕ, ‖t.A - t.C‖ = 2 * m ∧ ‖t.C - t.D‖ = 2 * n

def BDSquared36 (t : Triangle) : Prop :=
  ‖t.B - t.D‖^2 = 36

def perimeter (t : Triangle) : ℝ :=
  ‖t.A - t.B‖ + ‖t.B - t.C‖ + ‖t.C - t.A‖

theorem smallest_perimeter (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : DOnAC t) 
  (h3 : BDPerpAC t) 
  (h4 : ACCDEven t) 
  (h5 : BDSquared36 t) : 
  ∀ t' : Triangle, 
    isIsosceles t' → DOnAC t' → BDPerpAC t' → ACCDEven t' → BDSquared36 t' → 
    perimeter t ≤ perimeter t' ∧ perimeter t = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l818_81821


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_four_neg_three_l818_81855

theorem cos_alpha_for_point_four_neg_three (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_four_neg_three_l818_81855


namespace NUMINAMATH_CALUDE_percentage_problem_l818_81832

theorem percentage_problem (x : ℝ) : 120 = 2.4 * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l818_81832


namespace NUMINAMATH_CALUDE_fish_for_white_duck_l818_81812

/-- The number of fish for each white duck -/
def fish_per_white_duck : ℕ := sorry

/-- The number of fish for each black duck -/
def fish_per_black_duck : ℕ := 10

/-- The number of fish for each multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- The number of white ducks -/
def white_ducks : ℕ := 3

/-- The number of black ducks -/
def black_ducks : ℕ := 7

/-- The number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- The total number of fish in the lake -/
def total_fish : ℕ := 157

theorem fish_for_white_duck :
  fish_per_white_duck * white_ducks +
  fish_per_black_duck * black_ducks +
  fish_per_multicolor_duck * multicolor_ducks = total_fish ∧
  fish_per_white_duck = 5 := by sorry

end NUMINAMATH_CALUDE_fish_for_white_duck_l818_81812


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l818_81886

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq_1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l818_81886


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l818_81818

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l818_81818


namespace NUMINAMATH_CALUDE_correct_num_recipes_l818_81845

/-- The number of recipes to be made for a chocolate chip cookie bake sale. -/
def num_recipes : ℕ := 23

/-- The number of cups of chocolate chips required for one recipe. -/
def cups_per_recipe : ℕ := 2

/-- The total number of cups of chocolate chips needed for all recipes. -/
def total_cups_needed : ℕ := 46

/-- Theorem stating that the number of recipes is correct given the conditions. -/
theorem correct_num_recipes : 
  num_recipes * cups_per_recipe = total_cups_needed :=
by sorry

end NUMINAMATH_CALUDE_correct_num_recipes_l818_81845


namespace NUMINAMATH_CALUDE_circle_properties_l818_81815

/-- For a circle with area 4π, prove its diameter is 4 and circumference is 4π -/
theorem circle_properties (r : ℝ) (h : r^2 * π = 4 * π) : 
  2 * r = 4 ∧ 2 * π * r = 4 * π :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l818_81815


namespace NUMINAMATH_CALUDE_sqrt_inequality_l818_81896

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < 1) :
  let f : ℝ → ℝ := fun x ↦ Real.sqrt x
  f a < f b ∧ f b < f (1/b) ∧ f (1/b) < f (1/a) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l818_81896


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l818_81846

theorem number_with_specific_remainders (n : ℕ) :
  ∃ (x : ℕ+), 
    x > 1 ∧ 
    n % x = 2 ∧ 
    (2 * n) % x = 4 → 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l818_81846


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l818_81879

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l818_81879


namespace NUMINAMATH_CALUDE_single_digit_square_5929_l818_81853

theorem single_digit_square_5929 :
  ∃! (A : ℕ), A < 10 ∧ (10 * A + A)^2 = 5929 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_single_digit_square_5929_l818_81853


namespace NUMINAMATH_CALUDE_connie_tickets_connie_redeemed_fifty_tickets_l818_81837

theorem connie_tickets : ℕ → Prop := fun total =>
  let koala := total / 2
  let earbuds := 10
  let bracelets := 15
  (koala + earbuds + bracelets = total) → total = 50

-- Proof
theorem connie_redeemed_fifty_tickets : ∃ total, connie_tickets total :=
  sorry

end NUMINAMATH_CALUDE_connie_tickets_connie_redeemed_fifty_tickets_l818_81837


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l818_81836

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im (i / (i - 1)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l818_81836


namespace NUMINAMATH_CALUDE_a_bounds_l818_81839

theorem a_bounds (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (square_sum_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l818_81839


namespace NUMINAMATH_CALUDE_circle_equation_with_given_conditions_l818_81807

/-- A circle with center (h, k) and radius r has the standard equation (x - h)² + (y - k)² = r² -/
def is_standard_circle_equation (h k r : ℝ) (f : ℝ → ℝ → Prop) :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on the line 2x - y = 3 -/
def lies_on_line (x y : ℝ) : Prop := 2*x - y = 3

/-- A circle is tangent to the x-axis if its distance to the x-axis equals its radius -/
def tangent_to_x_axis (h k r : ℝ) : Prop := |k| = r

/-- A circle is tangent to the y-axis if its distance to the y-axis equals its radius -/
def tangent_to_y_axis (h k r : ℝ) : Prop := |h| = r

theorem circle_equation_with_given_conditions :
  ∃ f : ℝ → ℝ → Prop,
    (∃ h k r : ℝ, 
      is_standard_circle_equation h k r f ∧
      lies_on_line h k ∧
      tangent_to_x_axis h k r ∧
      tangent_to_y_axis h k r) →
    (∀ x y, f x y ↔ ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_conditions_l818_81807


namespace NUMINAMATH_CALUDE_jackson_decorations_to_friend_l818_81863

/-- Represents the number of Christmas decorations Mrs. Jackson gives to her friend. -/
def decorations_to_friend (total_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ) (given_to_neighbor : ℕ) : ℕ :=
  total_boxes * decorations_per_box - used_decorations - given_to_neighbor

/-- Proves that Mrs. Jackson gives 17 decorations to her friend under the given conditions. -/
theorem jackson_decorations_to_friend :
  decorations_to_friend 6 25 58 75 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_to_friend_l818_81863


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l818_81843

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > -2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l818_81843


namespace NUMINAMATH_CALUDE_four_last_in_hundreds_of_fib_l818_81885

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Function to get the hundreds digit of a natural number -/
def hundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

/-- Predicate to check if a digit has appeared in the hundreds position of any Fibonacci number up to the nth term -/
def digitAppearedInHundreds (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ hundredsDigit (fib k) = d

/-- The main theorem: 4 is the last digit to appear in the hundreds position of a Fibonacci number -/
theorem four_last_in_hundreds_of_fib :
  ∃ N, digitAppearedInHundreds 4 N ∧
    ∀ d, d ≠ 4 → ∃ n, n < N ∧ digitAppearedInHundreds d n :=
  sorry

end NUMINAMATH_CALUDE_four_last_in_hundreds_of_fib_l818_81885


namespace NUMINAMATH_CALUDE_inequality_solution_set_l818_81809

theorem inequality_solution_set (x : ℝ) : 
  ((x + 2) * (1 - x) > 0) ↔ (-2 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l818_81809


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l818_81883

theorem sqrt_expression_equals_six :
  Real.sqrt ((16^10 / 16^9)^2 * 6^2) / 2^4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l818_81883


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l818_81887

theorem square_root_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (25 - x) = 9 →
  (10 + x) * (25 - x) = 529 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l818_81887


namespace NUMINAMATH_CALUDE_integral_2x_over_half_pi_l818_81830

theorem integral_2x_over_half_pi : ∫ x in (0)..(π/2), 2*x = π^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_2x_over_half_pi_l818_81830


namespace NUMINAMATH_CALUDE_bus_cost_relationship_l818_81862

/-- The functional relationship between the number of large buses purchased and the total cost -/
theorem bus_cost_relationship (x : ℝ) (y : ℝ) : y = 22 * x + 800 ↔ 
  y = 62 * x + 40 * (20 - x) := by sorry

end NUMINAMATH_CALUDE_bus_cost_relationship_l818_81862


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l818_81890

/-- Proves that given a loan with 4% annual simple interest over 8 years,
    if the interest is Rs. 306 less than the principal,
    then the principal must be Rs. 450. -/
theorem loan_principal_calculation (P : ℚ) : 
  (P * (4 : ℚ) * (8 : ℚ) / (100 : ℚ) = P - (306 : ℚ)) → P = (450 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l818_81890


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l818_81894

theorem complex_fraction_simplification :
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 6 - 3*I
  z₁ / z₂ = -9/15 + 8/15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l818_81894


namespace NUMINAMATH_CALUDE_clothing_cost_problem_l818_81844

theorem clothing_cost_problem (total_spent : ℕ) (num_pieces : ℕ) (piece1_cost : ℕ) (piece2_cost : ℕ) (same_cost_piece : ℕ) :
  total_spent = 610 →
  num_pieces = 7 →
  piece1_cost = 49 →
  same_cost_piece = 96 →
  total_spent = piece1_cost + piece2_cost + 5 * same_cost_piece →
  piece2_cost = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_clothing_cost_problem_l818_81844


namespace NUMINAMATH_CALUDE_log_weight_when_cut_l818_81852

/-- Given a log of length 20 feet that weighs 150 pounds per linear foot,
    prove that when cut in half, each piece weighs 1500 pounds. -/
theorem log_weight_when_cut (log_length : ℝ) (weight_per_foot : ℝ) :
  log_length = 20 →
  weight_per_foot = 150 →
  (log_length / 2) * weight_per_foot = 1500 := by
  sorry

end NUMINAMATH_CALUDE_log_weight_when_cut_l818_81852


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l818_81833

/-- Given two rectangles of equal area, where one rectangle has dimensions 6 inches by 50 inches,
    and the other has a width of 20 inches, prove that the length of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (area : ℝ) (length_jordan width_jordan width_carol : ℝ) :
  area = length_jordan * width_jordan →
  length_jordan = 6 →
  width_jordan = 50 →
  width_carol = 20 →
  ∃ length_carol : ℝ, area = length_carol * width_carol ∧ length_carol = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l818_81833


namespace NUMINAMATH_CALUDE_number_sequence_count_l818_81800

/-- The total number of numbers in the sequence -/
def n : ℕ := 8

/-- The average of all numbers -/
def total_average : ℚ := 25

/-- The average of the first two numbers -/
def first_two_average : ℚ := 20

/-- The average of the next three numbers -/
def next_three_average : ℚ := 26

/-- The sixth number in the sequence -/
def sixth_number : ℚ := 14

/-- The last (eighth) number in the sequence -/
def last_number : ℚ := 30

theorem number_sequence_count :
  (2 * first_two_average + 3 * next_three_average + sixth_number + 
   (sixth_number + 4) + (sixth_number + 6) + last_number) / n = total_average := by
  sorry

#check number_sequence_count

end NUMINAMATH_CALUDE_number_sequence_count_l818_81800


namespace NUMINAMATH_CALUDE_calvin_chips_weeks_l818_81888

/-- Calculates the number of weeks Calvin has been buying chips -/
def weeks_buying_chips (cost_per_pack : ℚ) (days_per_week : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / (cost_per_pack * days_per_week)

/-- Theorem stating that Calvin has been buying chips for 4 weeks -/
theorem calvin_chips_weeks :
  let cost_per_pack : ℚ := 1/2  -- $0.50 represented as a rational number
  let days_per_week : ℕ := 5
  let total_spent : ℚ := 10
  weeks_buying_chips cost_per_pack days_per_week total_spent = 4 := by
  sorry

end NUMINAMATH_CALUDE_calvin_chips_weeks_l818_81888


namespace NUMINAMATH_CALUDE_seven_boys_handshakes_l818_81803

/-- The number of handshakes between n boys, where each boy shakes hands exactly once with each of the others -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 7 boys is 21 -/
theorem seven_boys_handshakes : num_handshakes 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_handshakes_l818_81803


namespace NUMINAMATH_CALUDE_bert_sandwiches_l818_81874

def sandwiches_problem (initial_sandwiches : ℕ) : ℕ :=
  let day1_remaining := initial_sandwiches / 2
  let day2_remaining := day1_remaining - (2 * day1_remaining / 3)
  let day3_eaten := (2 * day1_remaining / 3) - 2
  day2_remaining - min day2_remaining day3_eaten

theorem bert_sandwiches :
  sandwiches_problem 36 = 0 := by
  sorry

end NUMINAMATH_CALUDE_bert_sandwiches_l818_81874


namespace NUMINAMATH_CALUDE_expression_evaluation_l818_81881

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  5 * (a^2 + b) - 2 * (b + 2 * a^2) + 2 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l818_81881


namespace NUMINAMATH_CALUDE_area_remaining_after_iterations_l818_81824

/-- The fraction of area that remains after each iteration -/
def remaining_fraction : ℚ := 3 / 4

/-- The number of iterations -/
def num_iterations : ℕ := 5

/-- The final fraction of the original area remaining -/
def final_fraction : ℚ := 243 / 1024

theorem area_remaining_after_iterations :
  remaining_fraction ^ num_iterations = final_fraction := by
  sorry

end NUMINAMATH_CALUDE_area_remaining_after_iterations_l818_81824


namespace NUMINAMATH_CALUDE_book_sales_theorem_l818_81858

def monday_sales : ℕ := 15

def tuesday_sales : ℕ := 2 * monday_sales

def wednesday_sales : ℕ := tuesday_sales + (tuesday_sales / 2)

def thursday_sales : ℕ := wednesday_sales + (wednesday_sales / 2)

def friday_expected_sales : ℕ := thursday_sales + (thursday_sales / 2)

def friday_actual_sales : ℕ := friday_expected_sales + (friday_expected_sales / 4)

def saturday_sales : ℕ := (friday_expected_sales * 7) / 10

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_actual_sales + saturday_sales

theorem book_sales_theorem : total_sales = 357 := by
  sorry

end NUMINAMATH_CALUDE_book_sales_theorem_l818_81858


namespace NUMINAMATH_CALUDE_principal_amount_l818_81872

/-- Proves that given the specified conditions, the principal amount is 2600 --/
theorem principal_amount (rate : ℚ) (time : ℕ) (interest_difference : ℚ) : 
  rate = 4/100 → 
  time = 5 → 
  interest_difference = 2080 → 
  (∃ (principal : ℚ), 
    principal * rate * time = principal - interest_difference ∧ 
    principal = 2600) := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l818_81872


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l818_81871

theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 60 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 288 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l818_81871


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_9_15_45_l818_81892

theorem gcf_lcm_sum_9_15_45 : ∃ (C D : ℕ),
  (C = Nat.gcd 9 (Nat.gcd 15 45)) ∧
  (D = Nat.lcm 9 (Nat.lcm 15 45)) ∧
  (C + D = 60) := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_9_15_45_l818_81892


namespace NUMINAMATH_CALUDE_divisible_difference_exists_l818_81841

theorem divisible_difference_exists (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_divisible_difference_exists_l818_81841


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l818_81854

/-- Given a geometric sequence {aₙ} with common ratio q = √2 and a₁ · a₃ · a₅ = 4,
    prove that a₄ · a₅ · a₆ = 32. -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- geometric sequence with ratio √2
  a 1 * a 3 * a 5 = 4 →                   -- given condition
  a 4 * a 5 * a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l818_81854


namespace NUMINAMATH_CALUDE_range_of_x_l818_81806

theorem range_of_x (a b c x : ℝ) : 
  a^2 + 2*b^2 + 3*c^2 = 6 →
  a + 2*b + 3*c > |x + 1| →
  -7 < x ∧ x < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l818_81806


namespace NUMINAMATH_CALUDE_ernie_circles_l818_81811

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_l818_81811


namespace NUMINAMATH_CALUDE_point_on_circle_range_l818_81823

/-- Given two points A(a,0) and B(-a,0) where a > 0, and a circle with center (2√3, 2) and radius 3,
    if there exists a point P on the circle such that ∠APB = 90°, then 1 ≤ a ≤ 7. -/
theorem point_on_circle_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ P : ℝ × ℝ, (P.1 - 2 * Real.sqrt 3)^2 + (P.2 - 2)^2 = 9 ∧ 
   (P.1 - a)^2 + P.2^2 + (P.1 + a)^2 + P.2^2 = ((P.1 - a)^2 + P.2^2) + ((P.1 + a)^2 + P.2^2)) →
  1 ≤ a ∧ a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_point_on_circle_range_l818_81823


namespace NUMINAMATH_CALUDE_solve_for_y_l818_81827

theorem solve_for_y (x y : ℤ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l818_81827


namespace NUMINAMATH_CALUDE_power_of_two_divides_factorial_l818_81880

theorem power_of_two_divides_factorial (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_divides_factorial_l818_81880


namespace NUMINAMATH_CALUDE_number_of_signups_l818_81869

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of sports competitions --/
def num_competitions : ℕ := 3

/-- Theorem: The number of ways for students to sign up for competitions --/
theorem number_of_signups : (num_competitions ^ num_students) = 243 := by
  sorry

end NUMINAMATH_CALUDE_number_of_signups_l818_81869


namespace NUMINAMATH_CALUDE_probability_is_four_ninths_l818_81831

/-- A cube that has been cut into smaller cubes -/
structure CutCube where
  /-- The number of smaller cubes the original cube is cut into -/
  total_cubes : ℕ
  /-- The number of smaller cubes with exactly two faces painted -/
  two_faced_cubes : ℕ
  /-- The total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27
  /-- The number of two-faced cubes is 12 -/
  two_faced_is_12 : two_faced_cubes = 12

/-- The probability of selecting a small cube with exactly two faces painted -/
def probability_two_faced (c : CutCube) : ℚ :=
  c.two_faced_cubes / c.total_cubes

theorem probability_is_four_ninths (c : CutCube) :
  probability_two_faced c = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_four_ninths_l818_81831


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l818_81897

/-- Represents a parallelepiped with integer dimensions --/
structure Parallelepiped where
  width : ℕ
  depth : ℕ
  height : ℕ

/-- Represents a square with an integer side length --/
structure Square where
  side : ℕ

/-- Checks if a set of squares can cover a parallelepiped without gaps or overlaps --/
def can_cover (p : Parallelepiped) (squares : List Square) : Prop :=
  let surface_area := 2 * (p.width * p.depth + p.width * p.height + p.depth * p.height)
  let squares_area := squares.map (λ s => s.side * s.side) |>.sum
  surface_area = squares_area

theorem parallelepiped_coverage : 
  let p := Parallelepiped.mk 1 1 4
  let squares := [Square.mk 4, Square.mk 1, Square.mk 1]
  can_cover p squares := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l818_81897


namespace NUMINAMATH_CALUDE_fraction_problem_l818_81878

theorem fraction_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 6) :
  d / a = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l818_81878


namespace NUMINAMATH_CALUDE_modulo_six_equivalence_l818_81804

theorem modulo_six_equivalence : 47^1987 - 22^1987 ≡ 1 [ZMOD 6] := by sorry

end NUMINAMATH_CALUDE_modulo_six_equivalence_l818_81804


namespace NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l818_81849

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 14th term of an arithmetic sequence given its 5th and 8th terms -/
theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_5 : a 5 = 6)
  (h_8 : a 8 = 15) :
  a 14 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_14th_term_l818_81849


namespace NUMINAMATH_CALUDE_florist_roses_l818_81868

/-- The number of roses a florist has after selling some and picking more. -/
def roses_after_selling_and_picking (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Theorem: Given the specific numbers from the problem, 
    the florist ends up with 40 roses. -/
theorem florist_roses : roses_after_selling_and_picking 37 16 19 = 40 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l818_81868


namespace NUMINAMATH_CALUDE_license_plate_count_l818_81884

/-- The number of different license plate combinations with three unique letters 
    followed by a dash and three digits, where exactly one digit is repeated exactly once. -/
def license_plate_combinations : ℕ :=
  let letter_combinations := 26 * 25 * 24
  let digit_combinations := 10 * 3 * 9
  letter_combinations * digit_combinations

/-- Theorem stating that the number of license plate combinations is 4,212,000 -/
theorem license_plate_count :
  license_plate_combinations = 4212000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l818_81884


namespace NUMINAMATH_CALUDE_falcons_win_percentage_l818_81856

theorem falcons_win_percentage (initial_games : ℕ) (initial_falcon_wins : ℕ) (win_percentage : ℚ) :
  let additional_games : ℕ := 42
  initial_games = 8 ∧ 
  initial_falcon_wins = 3 ∧ 
  win_percentage = 9/10 →
  (initial_falcon_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ win_percentage ∧
  ∀ n : ℕ, n < additional_games → 
    (initial_falcon_wins + n : ℚ) / (initial_games + n) < win_percentage :=
by sorry

end NUMINAMATH_CALUDE_falcons_win_percentage_l818_81856


namespace NUMINAMATH_CALUDE_function_properties_l818_81829

open Real

-- Define the function and its derivative
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the interval
variable (a b : ℝ)

-- State the theorem
theorem function_properties
  (hf : Continuous f)
  (hf' : Continuous f')
  (hderiv : ∀ x, HasDerivAt f (f' x) x)
  (hab : a < b)
  (hf'a : f' a > 0)
  (hf'b : f' b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b > f' x₀ * (a - b)) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ = 0 → False) ∧
  ¬(∀ x₀ ∈ Set.Icc a b, f x₀ > f a → False) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l818_81829


namespace NUMINAMATH_CALUDE_investment_problem_l818_81891

theorem investment_problem (x y T : ℝ) : 
  x + y = T →
  y = 800 →
  0.1 * x - 0.08 * y = 56 →
  T = 2000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l818_81891


namespace NUMINAMATH_CALUDE_water_depth_multiple_l818_81895

/-- Given Dean's height and the water depth, prove that the multiple of Dean's height
    representing the water depth is 10. -/
theorem water_depth_multiple (dean_height water_depth : ℝ) 
  (h1 : dean_height = 6)
  (h2 : water_depth = 60) :
  water_depth / dean_height = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_multiple_l818_81895


namespace NUMINAMATH_CALUDE_basketball_success_rate_l818_81876

theorem basketball_success_rate (p : ℝ) 
  (h : 1 - p^2 = 16/25) : p = 3/5 := by sorry

end NUMINAMATH_CALUDE_basketball_success_rate_l818_81876


namespace NUMINAMATH_CALUDE_afternoon_email_count_l818_81866

/-- Represents the number of emails Jack received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- The theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_email_count (e : EmailCount) 
  (h1 : e.morning = 10)
  (h2 : e.evening = 17)
  (h3 : e.morning = e.afternoon + 3) :
  e.afternoon = 7 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_email_count_l818_81866


namespace NUMINAMATH_CALUDE_equation_solution_l818_81865

theorem equation_solution : ∃ x : ℝ, (3 / (x^2 - 9) + x / (x - 3) = 1) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l818_81865


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l818_81838

def f (x : ℝ) := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l818_81838


namespace NUMINAMATH_CALUDE_cylinder_properties_l818_81857

/-- Properties of a cylinder with height 15 and radius 5 -/
theorem cylinder_properties :
  ∀ (h r : ℝ),
  h = 15 →
  r = 5 →
  (2 * π * r^2 + 2 * π * r * h = 200 * π) ∧
  (π * r^2 * h = 375 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_properties_l818_81857


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l818_81875

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's actual work schedule -/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 360 }

/-- Theorem: Sheila's hourly wage is $10 -/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l818_81875


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l818_81820

theorem smallest_sum_of_reciprocals (x y : ℕ) : 
  x ≠ y → 
  x > 0 → 
  y > 0 → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24 → 
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 → 
  x + y ≤ a + b →
  x + y = 98 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l818_81820


namespace NUMINAMATH_CALUDE_h_equals_one_l818_81899

-- Define a group G
variable {G : Type*} [Group G]

-- Define the elements g and h
variable (g h : G)

-- Define n as a natural number
variable (n : ℕ)

-- State the theorem
theorem h_equals_one
  (rel : g * h * g = h * g^2 * h)
  (g_cube : g^3 = 1)
  (h_power : h^n = 1)
  (n_odd : Odd n) :
  h = 1 :=
sorry

end NUMINAMATH_CALUDE_h_equals_one_l818_81899


namespace NUMINAMATH_CALUDE_florist_roses_problem_l818_81802

theorem florist_roses_problem (x : ℕ) : 
  x - 2 + 32 = 41 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l818_81802


namespace NUMINAMATH_CALUDE_li_zhi_assignment_l818_81877

-- Define the universities
inductive University
| Tongji
| ShanghaiJiaoTong
| ShanghaiNormal

-- Define the volunteer roles
inductive VolunteerRole
| Translator
| City
| Social

-- Define the students
inductive Student
| LiZhi
| WenWen
| LiuBing

-- Define the assignment function
def assignment (s : Student) : University × VolunteerRole :=
  sorry

-- State the theorem
theorem li_zhi_assignment :
  (∀ s, s = Student.LiZhi → (assignment s).1 ≠ University.Tongji) →
  (∀ s, s = Student.WenWen → (assignment s).1 ≠ University.ShanghaiJiaoTong) →
  (∀ s, (assignment s).1 = University.Tongji → (assignment s).2 ≠ VolunteerRole.Translator) →
  (∀ s, (assignment s).1 = University.ShanghaiJiaoTong → (assignment s).2 = VolunteerRole.City) →
  (∀ s, s = Student.WenWen → (assignment s).2 ≠ VolunteerRole.Social) →
  (assignment Student.LiZhi).1 = University.ShanghaiJiaoTong ∧ 
  (assignment Student.LiZhi).2 = VolunteerRole.City :=
by
  sorry

end NUMINAMATH_CALUDE_li_zhi_assignment_l818_81877


namespace NUMINAMATH_CALUDE_product_of_fraction_parts_l818_81889

/-- Represents a repeating decimal with a 4-digit repeating sequence -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d) / 9999

/-- The fraction representation of 0.0012 (repeating) -/
def fraction : ℚ := RepeatingDecimal 0 0 1 2

theorem product_of_fraction_parts : ∃ (n d : ℕ), fraction = n / d ∧ Nat.gcd n d = 1 ∧ n * d = 13332 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fraction_parts_l818_81889


namespace NUMINAMATH_CALUDE_lee_cookies_l818_81842

/-- Given an initial ratio of flour to cookies and a new amount of flour, 
    calculate the number of cookies that can be made. -/
def cookies_from_flour (initial_flour : ℚ) (initial_cookies : ℚ) (new_flour : ℚ) : ℚ :=
  (initial_cookies / initial_flour) * new_flour

/-- Theorem stating that with the given initial ratio and remaining flour, 
    Lee can make 36 cookies. -/
theorem lee_cookies : 
  let initial_flour : ℚ := 2
  let initial_cookies : ℚ := 18
  let initial_available : ℚ := 5
  let spilled : ℚ := 1
  let remaining_flour : ℚ := initial_available - spilled
  cookies_from_flour initial_flour initial_cookies remaining_flour = 36 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l818_81842


namespace NUMINAMATH_CALUDE_polyhedron_face_edges_divisible_by_three_l818_81813

-- Define a polyhedron
structure Polyhedron where
  faces : Set Face
  edges : Set Edge
  vertices : Set Vertex

-- Define a face
structure Face where
  edges : Set Edge

-- Define an edge
structure Edge where
  vertices : Fin 2 → Vertex

-- Define a vertex
structure Vertex where

-- Define a color
inductive Color
  | White
  | Black

-- Define a coloring function
def coloring (p : Polyhedron) : Face → Color := sorry

-- Define the number of edges for a face
def numEdges (f : Face) : Nat := sorry

-- Define adjacency for faces
def adjacent (f1 f2 : Face) : Prop := sorry

-- Theorem statement
theorem polyhedron_face_edges_divisible_by_three 
  (p : Polyhedron) 
  (h1 : ∀ f1 f2 : Face, f1 ∈ p.faces → f2 ∈ p.faces → adjacent f1 f2 → coloring p f1 ≠ coloring p f2)
  (h2 : ∃ f : Face, f ∈ p.faces ∧ ∀ f' : Face, f' ∈ p.faces → f' ≠ f → (numEdges f') % 3 = 0) :
  ∀ f : Face, f ∈ p.faces → (numEdges f) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_face_edges_divisible_by_three_l818_81813
