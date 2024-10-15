import Mathlib

namespace NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l139_13997

/-- Represents the types of algorithm statements -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem: The set of basic algorithm statements is exactly 
    {INPUT, PRINT, IF-THEN, DO, WHILE} -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = 
    {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN, 
     AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l139_13997


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l139_13936

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
def strawberry_harvest (length width planting_density yield_per_plant : ℕ) : ℕ :=
  length * width * planting_density * yield_per_plant

/-- Proves that Carrie's garden will yield 7200 strawberries. -/
theorem carries_strawberry_harvest :
  strawberry_harvest 10 12 5 12 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l139_13936


namespace NUMINAMATH_CALUDE_leah_savings_days_l139_13932

/-- Proves that Leah saved for 20 days given the conditions of the problem -/
theorem leah_savings_days : ℕ :=
  let josiah_daily_savings : ℚ := 25 / 100
  let josiah_days : ℕ := 24
  let leah_daily_savings : ℚ := 1 / 2
  let megan_days : ℕ := 12
  let total_savings : ℚ := 28
  let leah_days : ℕ := 20

  have josiah_total : ℚ := josiah_daily_savings * josiah_days
  have megan_total : ℚ := 2 * leah_daily_savings * megan_days
  have leah_total : ℚ := leah_daily_savings * leah_days

  have savings_equation : josiah_total + leah_total + megan_total = total_savings := by sorry

  leah_days


end NUMINAMATH_CALUDE_leah_savings_days_l139_13932


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l139_13999

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  z.im = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l139_13999


namespace NUMINAMATH_CALUDE_cuboids_painted_count_l139_13925

/-- The number of outer faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of painted faces -/
def total_painted_faces : ℕ := 60

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem cuboids_painted_count : num_cuboids = 10 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_painted_count_l139_13925


namespace NUMINAMATH_CALUDE_min_fruits_in_platter_l139_13976

/-- Represents the types of fruits --/
inductive Fruit
  | GreenApple
  | RedApple
  | YellowApple
  | RedOrange
  | YellowOrange
  | GreenKiwi
  | PurpleGrape
  | GreenGrape

/-- Represents the fruit platter --/
structure FruitPlatter :=
  (greenApples : ℕ)
  (redApples : ℕ)
  (yellowApples : ℕ)
  (redOranges : ℕ)
  (yellowOranges : ℕ)
  (greenKiwis : ℕ)
  (purpleGrapes : ℕ)
  (greenGrapes : ℕ)

/-- Checks if the platter satisfies all constraints --/
def isValidPlatter (p : FruitPlatter) : Prop :=
  p.greenApples + p.redApples + p.yellowApples ≥ 5 ∧
  p.redOranges + p.yellowOranges ≤ 5 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≥ 8 ∧
  p.greenKiwis + p.purpleGrapes + p.greenGrapes ≤ 12 ∧
  p.greenGrapes ≥ 1 ∧
  p.purpleGrapes ≥ 1 ∧
  p.greenApples * 2 = p.redApples ∧
  p.greenApples * 3 = p.yellowApples * 2 ∧
  p.redOranges = 1 ∧
  p.yellowOranges = 2 ∧
  p.greenKiwis = p.purpleGrapes

/-- Calculates the total number of fruits in the platter --/
def totalFruits (p : FruitPlatter) : ℕ :=
  p.greenApples + p.redApples + p.yellowApples +
  p.redOranges + p.yellowOranges +
  p.greenKiwis + p.purpleGrapes + p.greenGrapes

/-- Theorem stating that the minimum number of fruits in a valid platter is 30 --/
theorem min_fruits_in_platter :
  ∀ p : FruitPlatter, isValidPlatter p → totalFruits p ≥ 30 :=
sorry

end NUMINAMATH_CALUDE_min_fruits_in_platter_l139_13976


namespace NUMINAMATH_CALUDE_no_real_solutions_l139_13926

theorem no_real_solutions : ¬∃ (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = (1:ℝ)/3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l139_13926


namespace NUMINAMATH_CALUDE_min_value_f_when_a_is_1_range_of_a_when_solution_exists_l139_13998

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f_when_a_is_1 :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Theorem 2: Range of a when solution set of f(x) ≤ 3 is non-empty
theorem range_of_a_when_solution_exists :
  (∃ x : ℝ, f a x ≤ 3) → |3 - a| ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_when_a_is_1_range_of_a_when_solution_exists_l139_13998


namespace NUMINAMATH_CALUDE_sum_of_integers_l139_13927

theorem sum_of_integers (x y : ℕ+) 
  (h_diff : x - y = 18) 
  (h_prod : x * y = 72) : 
  x + y = 2 * Real.sqrt 153 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l139_13927


namespace NUMINAMATH_CALUDE_geometric_series_sum_l139_13978

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 4) : ∑' n, a / (a + b)^n = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l139_13978


namespace NUMINAMATH_CALUDE_sqrt_inequality_l139_13990

theorem sqrt_inequality (a : ℝ) (h : a > 1) : Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l139_13990


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l139_13905

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π / 4) = 1 / 3) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l139_13905


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l139_13975

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 →  -- x and y are two-digit positive integers
  (x + y) / 2 = 55 →                     -- mean of x and y is 55
  ∃ (z : ℕ), x * y = z ^ 2 →             -- product xy is a square number
  ∀ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
    (a + b) / 2 = 55 ∧
    (∃ (w : ℕ), a * b = w ^ 2) →
    x / y ≥ a / b →
  x / y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_integers_l139_13975


namespace NUMINAMATH_CALUDE_simplified_expression_l139_13931

theorem simplified_expression : -1^2008 + 3*(-1)^2007 + 1^2008 - 2*(-1)^2009 = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_l139_13931


namespace NUMINAMATH_CALUDE_f_properties_l139_13924

open Real

noncomputable def f (x : ℝ) := x * log x

theorem f_properties :
  ∀ (m : ℝ), m > 0 →
  (∀ (x : ℝ), x > 0 →
    (∃ (min_value : ℝ),
      (∀ (y : ℝ), y ∈ Set.Icc m (m + 2) → f y ≥ min_value) ∧
      ((0 < m ∧ m < exp (-1)) → min_value = -(exp (-1))) ∧
      (m ≥ exp (-1) → min_value = f m))) ∧
  (∀ (x : ℝ), x > 0 → f x > x / (exp x) - 2 / exp 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l139_13924


namespace NUMINAMATH_CALUDE_friends_who_ate_bread_l139_13967

theorem friends_who_ate_bread (loaves : ℕ) (slices_per_loaf : ℕ) (slices_per_friend : ℕ) :
  loaves = 4 →
  slices_per_loaf = 15 →
  slices_per_friend = 6 →
  (loaves * slices_per_loaf) % slices_per_friend = 0 →
  (loaves * slices_per_loaf) / slices_per_friend = 10 := by
  sorry

end NUMINAMATH_CALUDE_friends_who_ate_bread_l139_13967


namespace NUMINAMATH_CALUDE_first_year_payment_is_twenty_l139_13909

/-- Calculates the first year payment given the total payment and yearly increases -/
def firstYearPayment (totalPayment : ℚ) (secondYearIncrease thirdYearIncrease fourthYearIncrease : ℚ) : ℚ :=
  (totalPayment - (secondYearIncrease + (secondYearIncrease + thirdYearIncrease) + 
   (secondYearIncrease + thirdYearIncrease + fourthYearIncrease))) / 4

/-- Theorem stating that the first year payment is 20.00 given the problem conditions -/
theorem first_year_payment_is_twenty :
  firstYearPayment 96 2 3 4 = 20 := by
  sorry

#eval firstYearPayment 96 2 3 4

end NUMINAMATH_CALUDE_first_year_payment_is_twenty_l139_13909


namespace NUMINAMATH_CALUDE_simplify_expression_l139_13917

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l139_13917


namespace NUMINAMATH_CALUDE_product_one_cube_sum_inequality_l139_13922

theorem product_one_cube_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^3 + b^3 + c^3 + d^3 ≥ max (a + b + c + d) (1/a + 1/b + 1/c + 1/d) := by
  sorry

end NUMINAMATH_CALUDE_product_one_cube_sum_inequality_l139_13922


namespace NUMINAMATH_CALUDE_place_mat_length_l139_13937

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) : 
  r = 4 →
  n = 6 →
  w = 1 →
  (x + 2 * Real.sqrt 3 - 1/2)^2 = 63/4 →
  x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_place_mat_length_l139_13937


namespace NUMINAMATH_CALUDE_smallest_number_properties_l139_13929

/-- The number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The smallest natural number divisible by 35 with exactly 75 divisors -/
def smallestNumber : ℕ := 490000

theorem smallest_number_properties :
  (35 ∣ smallestNumber) ∧
  (numDivisors smallestNumber = 75) ∧
  ∀ n : ℕ, n < smallestNumber → ¬((35 ∣ n) ∧ (numDivisors n = 75)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_properties_l139_13929


namespace NUMINAMATH_CALUDE_equation_solution_l139_13934

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / (x - 1) = 1 / (x - 2) + 1 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l139_13934


namespace NUMINAMATH_CALUDE_zachary_crunches_l139_13920

/-- Proves that Zachary did 58 crunches given the problem conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_crunches david_pushups david_crunches : ℕ),
  zachary_pushups = 46 →
  david_pushups = zachary_pushups + 38 →
  david_crunches = zachary_crunches - 62 →
  zachary_crunches = zachary_pushups + 12 →
  zachary_crunches = 58 := by
  sorry

end NUMINAMATH_CALUDE_zachary_crunches_l139_13920


namespace NUMINAMATH_CALUDE_one_point_one_billion_scientific_notation_l139_13974

/-- Expresses 1.1 billion in scientific notation -/
theorem one_point_one_billion_scientific_notation :
  (1.1 * 10^9 : ℝ) = 1100000000 := by
  sorry

end NUMINAMATH_CALUDE_one_point_one_billion_scientific_notation_l139_13974


namespace NUMINAMATH_CALUDE_triangular_frame_is_stable_bicycle_frame_triangle_stability_l139_13969

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides. -/
inductive Triangle : Type

/-- Stability is a property that can be possessed by structures. -/
class Stable (α : Type) where
  is_stable : α → Prop

/-- A bicycle frame made in the shape of a triangle -/
def triangular_frame : BicycleFrame := { shape := Triangle }

/-- The theorem stating that a triangular bicycle frame is stable -/
theorem triangular_frame_is_stable :
  Stable Triangle → Stable (triangular_frame.shape) :=
by
  sorry

/-- The main theorem proving that a bicycle frame made in the shape of a triangle is stable -/
theorem bicycle_frame_triangle_stability :
  Stable (triangular_frame.shape) :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_frame_is_stable_bicycle_frame_triangle_stability_l139_13969


namespace NUMINAMATH_CALUDE_repair_charge_is_30_l139_13996

/-- Represents the services and pricing at Cid's mechanic shop --/
structure MechanicShop where
  oil_change_price : ℕ
  car_wash_price : ℕ
  oil_changes : ℕ
  repairs : ℕ
  car_washes : ℕ
  total_earnings : ℕ

/-- Theorem stating that the repair charge is $30 --/
theorem repair_charge_is_30 (shop : MechanicShop) 
  (h1 : shop.oil_change_price = 20)
  (h2 : shop.car_wash_price = 5)
  (h3 : shop.oil_changes = 5)
  (h4 : shop.repairs = 10)
  (h5 : shop.car_washes = 15)
  (h6 : shop.total_earnings = 475) :
  (shop.total_earnings - (shop.oil_changes * shop.oil_change_price + shop.car_washes * shop.car_wash_price)) / shop.repairs = 30 :=
by
  sorry

#check repair_charge_is_30

end NUMINAMATH_CALUDE_repair_charge_is_30_l139_13996


namespace NUMINAMATH_CALUDE_sequence_representation_l139_13947

theorem sequence_representation (q : ℕ → ℕ) 
  (h_increasing : ∀ n, q n < q (n + 1))
  (h_bound : ∀ n, q n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, q k = m ∨ q l - q k = m :=
sorry

end NUMINAMATH_CALUDE_sequence_representation_l139_13947


namespace NUMINAMATH_CALUDE_log_condition_l139_13923

theorem log_condition (m : ℝ) (m_pos : m > 0) (m_neq_1 : m ≠ 1) :
  (∃ a b : ℝ, 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 → Real.log b / Real.log a > 0) ∧
  (∃ a b : ℝ, Real.log b / Real.log a > 0 ∧ ¬(0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1)) :=
by sorry

end NUMINAMATH_CALUDE_log_condition_l139_13923


namespace NUMINAMATH_CALUDE_household_survey_l139_13941

/-- Proves that the total number of households surveyed is 240 given the specified conditions -/
theorem household_survey (neither_brand : ℕ) (only_A : ℕ) (both_brands : ℕ)
  (h1 : neither_brand = 80)
  (h2 : only_A = 60)
  (h3 : both_brands = 25) :
  neither_brand + only_A + 3 * both_brands + both_brands = 240 := by
sorry

end NUMINAMATH_CALUDE_household_survey_l139_13941


namespace NUMINAMATH_CALUDE_pet_shop_legs_l139_13980

/-- The total number of legs in a pet shop with birds, dogs, snakes, and spiders -/
def total_legs (num_birds num_dogs num_snakes num_spiders : ℕ) 
               (bird_legs dog_legs snake_legs spider_legs : ℕ) : ℕ :=
  num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs + num_spiders * spider_legs

/-- Theorem stating that the total number of legs in the given pet shop scenario is 34 -/
theorem pet_shop_legs : 
  total_legs 3 5 4 1 2 4 0 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_legs_l139_13980


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l139_13985

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_product : a 2 * a 4 = 16) :
  a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l139_13985


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l139_13960

theorem perpendicular_lines_intersection (a b c d : ℝ) : 
  (∀ x y, a * x - 2 * y = d) →  -- First line equation
  (∀ x y, 2 * x + b * y = c) →  -- Second line equation
  (a * 2 - 2 * (-3) = d) →      -- Lines intersect at (2, -3)
  (2 * 2 + b * (-3) = c) →      -- Lines intersect at (2, -3)
  (a * b = -4) →                -- Perpendicular lines condition
  (d = 12) :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l139_13960


namespace NUMINAMATH_CALUDE_positive_integer_solutions_5x_plus_y_11_l139_13988

theorem positive_integer_solutions_5x_plus_y_11 :
  {(x, y) : ℕ × ℕ | 5 * x + y = 11 ∧ x > 0 ∧ y > 0} = {(1, 6), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_5x_plus_y_11_l139_13988


namespace NUMINAMATH_CALUDE_inscribed_square_area_l139_13919

/-- The area of a square inscribed in a quadrant of a circle with radius 10 is equal to 40 -/
theorem inscribed_square_area (r : ℝ) (s : ℝ) (h1 : r = 10) (h2 : s > 0) 
  (h3 : s^2 + (3/2 * s)^2 = r^2) : s^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l139_13919


namespace NUMINAMATH_CALUDE_cricket_team_captain_age_l139_13986

theorem cricket_team_captain_age (team_size : Nat) (captain_age wicket_keeper_age : Nat) 
  (remaining_players_avg_age team_avg_age : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 5 →
  remaining_players_avg_age = team_avg_age - 1 →
  team_avg_age = 24 →
  (team_size - 2 : ℚ) * remaining_players_avg_age + captain_age + wicket_keeper_age = 
    team_size * team_avg_age →
  captain_age = 26 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_captain_age_l139_13986


namespace NUMINAMATH_CALUDE_uncommon_card_cost_is_half_dollar_l139_13910

/-- The cost of an uncommon card in Tom's deck -/
def uncommon_card_cost : ℚ :=
  let rare_cards : ℕ := 19
  let uncommon_cards : ℕ := 11
  let common_cards : ℕ := 30
  let rare_card_cost : ℚ := 1
  let common_card_cost : ℚ := 1/4
  let total_deck_cost : ℚ := 32
  (total_deck_cost - rare_cards * rare_card_cost - common_cards * common_card_cost) / uncommon_cards

theorem uncommon_card_cost_is_half_dollar : uncommon_card_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_card_cost_is_half_dollar_l139_13910


namespace NUMINAMATH_CALUDE_science_club_enrollment_l139_13973

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120) 
  (h2 : math = 80) 
  (h3 : physics = 50) 
  (h4 : both = 15) : 
  total - (math + physics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l139_13973


namespace NUMINAMATH_CALUDE_existence_of_numbers_l139_13942

theorem existence_of_numbers : ∃ (a b c d : ℕ), 
  (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (c : ℚ) / b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_numbers_l139_13942


namespace NUMINAMATH_CALUDE_translation_theorem_l139_13907

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translate a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem translation_theorem (m n : ℝ) :
  let p := Point.mk m n
  let p' := translateVertical (translateHorizontal p 2) 1
  p'.x = m + 2 ∧ p'.y = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l139_13907


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l139_13994

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l139_13994


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l139_13944

/-- The number of years until the ratio of Mike's age to Sam's age is 3:2 -/
def years_until_ratio (m s : ℕ) : ℕ :=
  9

theorem age_ratio_theorem (m s : ℕ) 
  (h1 : m - 5 = 2 * (s - 5))
  (h2 : m - 12 = 3 * (s - 12)) :
  (m + years_until_ratio m s) / (s + years_until_ratio m s) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l139_13944


namespace NUMINAMATH_CALUDE_line_symmetry_l139_13972

-- Define the lines
def l1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def l2 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Define the symmetry relation
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2)/2) ((y1 + y2)/2) ∧ x1 + y2 = 0 ∧ y1 + x2 = 0

-- Theorem statement
theorem line_symmetry :
  (∀ x y : ℝ, l1 x y ↔ l2 (-y) (-x)) →
  (∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 ∧ l2 x2 y2 → symmetric_points x1 y1 x2 y2) →
  ∀ x y : ℝ, l2 x y ↔ 2*x - y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l139_13972


namespace NUMINAMATH_CALUDE_no_integer_solution_l139_13954

theorem no_integer_solution (n : ℕ) (hn : n ≥ 11) :
  ¬ ∃ m : ℤ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l139_13954


namespace NUMINAMATH_CALUDE_parabola_distances_arithmetic_l139_13981

/-- A parabola with focus F and three points A, B, C on it. -/
structure Parabola where
  p : ℝ
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  h_p_pos : 0 < p
  h_on_parabola_1 : y₁^2 = 2 * p * x₁
  h_on_parabola_2 : y₂^2 = 2 * p * x₂
  h_on_parabola_3 : y₃^2 = 2 * p * x₃
  h_arithmetic : ∃ d : ℝ, 
    (x₂ : ℝ) - x₁ = d ∧ 
    (x₃ : ℝ) - x₂ = d

/-- If the distances from A, B, C to the focus form an arithmetic sequence,
    then x₁, x₂, x₃ form an arithmetic sequence. -/
theorem parabola_distances_arithmetic (par : Parabola) :
  ∃ d : ℝ, (par.x₂ - par.x₁ = d) ∧ (par.x₃ - par.x₂ = d) := by
  sorry

end NUMINAMATH_CALUDE_parabola_distances_arithmetic_l139_13981


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l139_13989

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

-- Define the points
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (4, 0)
def point_C : ℝ × ℝ := (-1, 1)

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l139_13989


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l139_13979

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ (3 * x : ℤ) - 5 < 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l139_13979


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l139_13966

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l139_13966


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l139_13913

def vector_a (t : ℝ) : ℝ × ℝ := (1, t)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 9)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel (vector_a t) (vector_b t) → t = 3 ∨ t = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l139_13913


namespace NUMINAMATH_CALUDE_shirts_sold_l139_13955

def commission_rate : ℚ := 15 / 100
def suit_price : ℚ := 700
def suit_quantity : ℕ := 2
def shirt_price : ℚ := 50
def loafer_price : ℚ := 150
def loafer_quantity : ℕ := 2
def total_commission : ℚ := 300

theorem shirts_sold (shirt_quantity : ℕ) : 
  commission_rate * (suit_price * suit_quantity + shirt_price * shirt_quantity + loafer_price * loafer_quantity) = total_commission →
  shirt_quantity = 6 := by sorry

end NUMINAMATH_CALUDE_shirts_sold_l139_13955


namespace NUMINAMATH_CALUDE_button_probability_l139_13987

/-- Given a jar with red and blue buttons, prove the probability of selecting two red buttons after a specific removal process. -/
theorem button_probability (initial_red initial_blue : ℕ) 
  (h1 : initial_red = 6)
  (h2 : initial_blue = 10)
  (h3 : ∃ (removed : ℕ), 
    removed ≤ initial_red ∧ 
    removed ≤ initial_blue ∧ 
    initial_red + initial_blue - 2 * removed = (3 / 4) * (initial_red + initial_blue)) :
  let total_initial := initial_red + initial_blue
  let removed := (total_initial - (3 / 4) * total_initial) / 2
  let red_a := initial_red - removed
  let total_a := (3 / 4) * total_initial
  let prob_red_a := red_a / total_a
  let prob_red_b := removed / (2 * removed)
  prob_red_a * prob_red_b = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l139_13987


namespace NUMINAMATH_CALUDE_range_of_a_l139_13943

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (ha : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l139_13943


namespace NUMINAMATH_CALUDE_binomial_and_factorial_l139_13912

theorem binomial_and_factorial : 
  (Nat.choose 10 5 = 252) ∧ (Nat.factorial (Nat.choose 10 5 - 5) = Nat.factorial 247) := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_factorial_l139_13912


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l139_13900

/-- A line with slope 6 passing through (4, -3) and intersecting y = -x + 1 has m + b = -21 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 6 ∧ 
  -3 = 6 * 4 + b ∧ 
  ∃ x y : ℝ, y = 6 * x + b ∧ y = -x + 1 →
  m + b = -21 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l139_13900


namespace NUMINAMATH_CALUDE_difference_of_percentages_l139_13965

-- Define the percentage
def percentage : ℚ := 25 / 100

-- Define the two amounts in pence (to avoid floating-point issues)
def amount1 : ℕ := 3700  -- £37 in pence
def amount2 : ℕ := 1700  -- £17 in pence

-- Theorem statement
theorem difference_of_percentages :
  (percentage * amount1 - percentage * amount2 : ℚ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l139_13965


namespace NUMINAMATH_CALUDE_first_digit_value_l139_13956

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem first_digit_value (x y : ℕ) : 
  x < 10 → 
  y < 10 → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 2 → 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_value_l139_13956


namespace NUMINAMATH_CALUDE_expected_ones_value_l139_13959

/-- The number of magnets --/
def n : ℕ := 50

/-- The probability of a difference of 1 between two randomly chosen numbers --/
def p : ℚ := 49 / 1225

/-- The number of pairs of consecutive magnets --/
def num_pairs : ℕ := n - 1

/-- The expected number of times the difference 1 occurs --/
def expected_ones : ℚ := num_pairs * p

theorem expected_ones_value : expected_ones = 49 / 25 := by sorry

end NUMINAMATH_CALUDE_expected_ones_value_l139_13959


namespace NUMINAMATH_CALUDE_field_ratio_l139_13906

theorem field_ratio (l w : ℝ) (h1 : ∃ k : ℕ, l = k * w) 
  (h2 : l = 36) (h3 : 81 = (1/8) * (l * w)) : l / w = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l139_13906


namespace NUMINAMATH_CALUDE_no_real_solutions_l139_13915

theorem no_real_solutions :
  ¬∃ y : ℝ, (8 * y^2 + 47 * y + 5) / (4 * y + 15) = 4 * y + 2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l139_13915


namespace NUMINAMATH_CALUDE_farm_hens_count_l139_13968

/-- Given a farm with roosters and hens, where the number of hens is 5 less than 9 times
    the number of roosters, and the total number of chickens is 75, prove that there are 67 hens. -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by
sorry

end NUMINAMATH_CALUDE_farm_hens_count_l139_13968


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l139_13950

/-- A line passing through (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  ((-1, 2) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = -1 + 3*t ∧ y = 2 - 2*t) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x + 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l139_13950


namespace NUMINAMATH_CALUDE_subset_implies_complement_subset_l139_13992

theorem subset_implies_complement_subset (P Q : Set α) 
  (h_nonempty_P : P.Nonempty) (h_nonempty_Q : Q.Nonempty) 
  (h_intersection : P ∩ Q = P) : 
  ∀ x, x ∉ Q → x ∉ P := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_complement_subset_l139_13992


namespace NUMINAMATH_CALUDE_min_value_of_function_l139_13930

/-- The function f(x) = 4/(x-2) + x has a minimum value of 6 for x > 2 -/
theorem min_value_of_function (x : ℝ) (h : x > 2) : 
  (4 / (x - 2) + x) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l139_13930


namespace NUMINAMATH_CALUDE_nitin_rank_last_l139_13911

def class_size : ℕ := 58
def nitin_rank_start : ℕ := 24

theorem nitin_rank_last : class_size - nitin_rank_start + 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_nitin_rank_last_l139_13911


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l139_13921

/-- Given two cylinders A and B, where A's radius is r and height is h,
    B's height is r and radius is h, and A's volume is twice B's volume,
    prove that A's volume can be expressed as 4π h^3. -/
theorem cylinder_volume_relation (r h : ℝ) (h_pos : h > 0) :
  let volume_A := π * r^2 * h
  let volume_B := π * h^2 * r
  volume_A = 2 * volume_B → r = 2 * h → volume_A = 4 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l139_13921


namespace NUMINAMATH_CALUDE_sequence_properties_l139_13933

def a (n : ℕ+) : ℚ := (3 * n - 2) / (3 * n + 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (a 3 = 7 / 10) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l139_13933


namespace NUMINAMATH_CALUDE_painting_price_increase_l139_13946

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 15 / 100) = 93.5 / 100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l139_13946


namespace NUMINAMATH_CALUDE_choir_size_after_new_members_l139_13916

theorem choir_size_after_new_members (original : Nat) (new : Nat) : 
  original = 36 → new = 9 → original + new = 45 := by
  sorry

end NUMINAMATH_CALUDE_choir_size_after_new_members_l139_13916


namespace NUMINAMATH_CALUDE_max_value_problem_l139_13982

theorem max_value_problem (x y : ℝ) (h : y^2 + x - 2 = 0) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (x' y' : ℝ), y'^2 + x' - 2 = 0 → y'^2 - x'^2 + x' + 5 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l139_13982


namespace NUMINAMATH_CALUDE_xiaoxiao_reading_plan_l139_13908

/-- Given a book with a total number of pages, pages already read, and days to finish,
    calculate the average number of pages to read per day. -/
def averagePagesPerDay (totalPages pagesRead daysToFinish : ℕ) : ℚ :=
  (totalPages - pagesRead : ℚ) / daysToFinish

/-- Theorem stating that for a book with 160 pages, 60 pages read, and 5 days to finish,
    the average number of pages to read per day is 20. -/
theorem xiaoxiao_reading_plan :
  averagePagesPerDay 160 60 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_xiaoxiao_reading_plan_l139_13908


namespace NUMINAMATH_CALUDE_extreme_value_implies_f_2_l139_13991

/-- A function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- Theorem: If f(x) has an extreme value of 10 at x = 1, then f(2) = 18 -/
theorem extreme_value_implies_f_2 (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b 1 ≥ f a b x) ∧
  f a b 1 = 10 →
  f a b 2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_f_2_l139_13991


namespace NUMINAMATH_CALUDE_max_stores_visited_is_three_l139_13948

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ

/-- The given shopping scenario -/
def given_scenario : ShoppingScenario :=
  { num_stores := 8
  , total_visits := 22
  , num_shoppers := 12
  , two_store_visitors := 8 }

/-- The maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : ℕ :=
  3

/-- Theorem stating that the maximum number of stores visited by any single person is 3 -/
theorem max_stores_visited_is_three (scenario : ShoppingScenario) 
  (h1 : scenario.num_stores = given_scenario.num_stores)
  (h2 : scenario.total_visits = given_scenario.total_visits)
  (h3 : scenario.num_shoppers = given_scenario.num_shoppers)
  (h4 : scenario.two_store_visitors = given_scenario.two_store_visitors)
  (h5 : scenario.two_store_visitors * 2 + (scenario.num_shoppers - scenario.two_store_visitors) ≤ scenario.total_visits)
  : max_stores_visited scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_stores_visited_is_three_l139_13948


namespace NUMINAMATH_CALUDE_theatre_audience_girls_fraction_l139_13958

theorem theatre_audience_girls_fraction 
  (total : ℝ) 
  (adults : ℝ) 
  (children : ℝ) 
  (boys : ℝ) 
  (girls : ℝ) 
  (h1 : adults = (1 / 6) * total) 
  (h2 : children = total - adults) 
  (h3 : boys = (2 / 5) * children) 
  (h4 : girls = children - boys) : 
  girls = (1 / 2) * total := by
sorry

end NUMINAMATH_CALUDE_theatre_audience_girls_fraction_l139_13958


namespace NUMINAMATH_CALUDE_special_quadratic_property_l139_13971

/-- A quadratic function f(x) = x^2 + ax + b satisfying specific conditions -/
def special_quadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- Theorem: If f(f(0)) = f(f(1)) = 0 and f(0) ≠ f(1), then f(2) = 3 -/
theorem special_quadratic_property (a b : ℝ) :
  let f := special_quadratic a b
  (f (f 0) = 0) → (f (f 1) = 0) → (f 0 ≠ f 1) → (f 2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_property_l139_13971


namespace NUMINAMATH_CALUDE_inverse_of_A_l139_13918

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; 2, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/14, 1/14; -1/7, 2/7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l139_13918


namespace NUMINAMATH_CALUDE_V_upper_bound_l139_13940

/-- V(n; b) is the number of decompositions of n into a product of one or more positive integers greater than b -/
def V (n b : ℕ+) : ℕ := sorry

/-- For all positive integers n and b, V(n; b) < n/b -/
theorem V_upper_bound (n b : ℕ+) : V n b < (n : ℚ) / b := by sorry

end NUMINAMATH_CALUDE_V_upper_bound_l139_13940


namespace NUMINAMATH_CALUDE_mixture_volume_l139_13962

/-- Given a mixture of milk and water, prove that the initial volume is 145 liters -/
theorem mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 3 / 2 →
  initial_milk / (initial_water + 58) = 3 / 4 →
  initial_milk + initial_water = 145 := by
sorry

end NUMINAMATH_CALUDE_mixture_volume_l139_13962


namespace NUMINAMATH_CALUDE_jasmine_bouquet_cost_l139_13957

/-- The cost of a bouquet of jasmines after discount -/
def bouquet_cost (n : ℕ) (base_cost : ℚ) (base_num : ℕ) (discount : ℚ) : ℚ :=
  (base_cost * n / base_num) * (1 - discount)

/-- The theorem statement -/
theorem jasmine_bouquet_cost :
  bouquet_cost 50 24 8 (1/10) = 135 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_bouquet_cost_l139_13957


namespace NUMINAMATH_CALUDE_min_value_problem_l139_13945

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_sum : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = -8 * Real.sqrt 2 ∧ ∀ z : ℝ, z = x * y - 4 * (x + y) - 2 → z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l139_13945


namespace NUMINAMATH_CALUDE_inscribed_cube_side_length_l139_13903

/-- A cone with a circular base of radius 1 and height 3 --/
structure Cone :=
  (base_radius : ℝ := 1)
  (height : ℝ := 3)

/-- A cube inscribed in a cone such that four vertices lie on the base and four on the sloping sides --/
structure InscribedCube :=
  (cone : Cone)
  (side_length : ℝ)
  (four_vertices_on_base : Prop)
  (four_vertices_on_slope : Prop)

/-- The side length of the inscribed cube is 3√2 / (3 + √2) --/
theorem inscribed_cube_side_length (cube : InscribedCube) :
  cube.side_length = 3 * Real.sqrt 2 / (3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_side_length_l139_13903


namespace NUMINAMATH_CALUDE_only_expr4_is_equation_l139_13953

-- Define the four expressions
def expr1 : ℝ → Prop := λ x ↦ 3 + x < 1
def expr2 : ℝ → ℝ := λ x ↦ x - 67 + 63
def expr3 : ℝ → ℝ := λ x ↦ 4.8 + x
def expr4 : ℝ → Prop := λ x ↦ x + 0.7 = 12

-- Theorem stating that only expr4 is an equation
theorem only_expr4_is_equation :
  (∃ (x : ℝ), expr4 x) ∧
  (¬∃ (x : ℝ), expr1 x = (3 + x < 1)) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr2 x = y) ∧
  (∀ (x : ℝ), ¬∃ (y : ℝ), expr3 x = y) :=
sorry

end NUMINAMATH_CALUDE_only_expr4_is_equation_l139_13953


namespace NUMINAMATH_CALUDE_expression_equals_sixteen_ten_to_five_hundred_l139_13935

theorem expression_equals_sixteen_ten_to_five_hundred :
  (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_sixteen_ten_to_five_hundred_l139_13935


namespace NUMINAMATH_CALUDE_soldier_hit_target_l139_13914

theorem soldier_hit_target (p q : Prop) : 
  (p ∨ q) ↔ (∃ shot : Fin 2, shot.val = 0 ∧ p ∨ shot.val = 1 ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_soldier_hit_target_l139_13914


namespace NUMINAMATH_CALUDE_gasoline_canister_detonation_probability_l139_13938

/-- The probability of detonating a gasoline canister -/
theorem gasoline_canister_detonation_probability :
  let n : ℕ := 5  -- number of available shots
  let p : ℚ := 2/3  -- probability of hitting the target
  let q : ℚ := 1 - p  -- probability of missing the target
  -- Assumption: shots are independent (implied by using binomial probability)
  -- Assumption: first successful hit causes a leak, second causes detonation (implied by the problem setup)
  232/243 = 1 - (q^n + n * q^(n-1) * p) :=
by sorry

end NUMINAMATH_CALUDE_gasoline_canister_detonation_probability_l139_13938


namespace NUMINAMATH_CALUDE_magnitude_a_minus_b_equals_5_l139_13949

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b_equals_5 :
  Real.sqrt ((vector_a.1 - vector_b.1)^2 + (vector_a.2 - vector_b.2)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_b_equals_5_l139_13949


namespace NUMINAMATH_CALUDE_lucky_lila_problem_l139_13963

theorem lucky_lila_problem (a b c d e : ℤ) : 
  a = 5 → b = 3 → c = 2 → d = 6 →
  (a - b + c * d - e = a - (b + (c * (d - e)))) →
  e = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lila_problem_l139_13963


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_137_l139_13970

theorem first_nonzero_digit_of_1_over_137 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 137 = n / 10^(n.succ) + r ∧ 0 < r ∧ r < 1 / 10^n ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_137_l139_13970


namespace NUMINAMATH_CALUDE_max_integer_value_of_fraction_l139_13977

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4 * x^2 + 12 * x + 20) / (4 * x^2 + 12 * x + 8) < 12002 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (4 * y^2 + 12 * y + 20) / (4 * y^2 + 12 * y + 8) > 12001 - ε :=
by sorry

#check max_integer_value_of_fraction

end NUMINAMATH_CALUDE_max_integer_value_of_fraction_l139_13977


namespace NUMINAMATH_CALUDE_circle_properties_l139_13961

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (0, 1)

theorem circle_properties :
  ∀ x y : ℝ,
    (line1 x y ∧ line2 x y → (x, y) = center) ∧
    circle_equation 1 0 ∧
    (∀ a b : ℝ, (a - center.1)^2 + (b - center.2)^2 = 2 ↔ circle_equation a b) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l139_13961


namespace NUMINAMATH_CALUDE_zoo_trip_vans_l139_13995

def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

theorem zoo_trip_vans : vans_needed 4 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_vans_l139_13995


namespace NUMINAMATH_CALUDE_math_team_selection_count_l139_13983

theorem math_team_selection_count :
  let total_boys : ℕ := 7
  let total_girls : ℕ := 10
  let boys_needed : ℕ := 2
  let girls_needed : ℕ := 3
  (Nat.choose total_boys boys_needed) * (Nat.choose total_girls girls_needed) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l139_13983


namespace NUMINAMATH_CALUDE_fraction_addition_l139_13904

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l139_13904


namespace NUMINAMATH_CALUDE_diamond_eight_five_l139_13993

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem diamond_eight_five : diamond 8 5 = 160 := by sorry

end NUMINAMATH_CALUDE_diamond_eight_five_l139_13993


namespace NUMINAMATH_CALUDE_hilt_pencil_cost_l139_13902

/-- The cost of a pencil given total money and number of pencils that can be bought --/
def pencil_cost (total_money : ℚ) (num_pencils : ℕ) : ℚ :=
  total_money / num_pencils

theorem hilt_pencil_cost :
  pencil_cost 50 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hilt_pencil_cost_l139_13902


namespace NUMINAMATH_CALUDE_valerie_light_bulbs_l139_13964

theorem valerie_light_bulbs :
  let total_budget : ℕ := 60
  let small_bulb_cost : ℕ := 8
  let large_bulb_cost : ℕ := 12
  let small_bulb_count : ℕ := 3
  let remaining_money : ℕ := 24
  let large_bulb_count : ℕ := (total_budget - remaining_money - small_bulb_cost * small_bulb_count) / large_bulb_cost
  large_bulb_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_valerie_light_bulbs_l139_13964


namespace NUMINAMATH_CALUDE_equation_solver_l139_13901

theorem equation_solver (m : ℕ) (p : ℝ) 
  (h1 : ((1^m) / (5^m)) * ((1^16) / (4^16)) = 1 / (2*(p^31)))
  (h2 : m = 31) : 
  p = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solver_l139_13901


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l139_13951

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | (x^2 - y^2)^2 = 16*y + 1} =
  {(1, 0), (-1, 0), (4, 3), (-4, 3), (4, 5), (-4, 5)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l139_13951


namespace NUMINAMATH_CALUDE_green_packs_count_l139_13928

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The total number of bouncy balls bought -/
def total_balls : ℕ := 160

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

theorem green_packs_count : green_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_packs_count_l139_13928


namespace NUMINAMATH_CALUDE_domain_fxPlus2_l139_13984

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x-3)
def domain_f2xMinus3 : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem domain_fxPlus2 (h : ∀ x ∈ domain_f2xMinus3, f (2*x - 3) = f (2*x - 3)) :
  {x : ℝ | f (x + 2) = f (x + 2)} = {x : ℝ | -9 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_domain_fxPlus2_l139_13984


namespace NUMINAMATH_CALUDE_oranges_from_second_tree_l139_13939

theorem oranges_from_second_tree :
  ∀ (first_tree second_tree third_tree total : ℕ),
  first_tree = 80 →
  third_tree = 120 →
  total = 260 →
  total = first_tree + second_tree + third_tree →
  second_tree = 60 := by
sorry

end NUMINAMATH_CALUDE_oranges_from_second_tree_l139_13939


namespace NUMINAMATH_CALUDE_difference_of_squares_l139_13952

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l139_13952
