import Mathlib

namespace NUMINAMATH_CALUDE_max_intersections_theorem_l3412_341247

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ

/-- The configuration of two convex polygons where one is contained within the other -/
structure PolygonConfiguration where
  outer : ConvexPolygon
  inner : ConvexPolygon
  inner_contained : inner.sides ≤ outer.sides
  no_coincident_sides : Bool

/-- The maximum number of intersection points between the sides of two polygons in the given configuration -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.inner.sides * config.outer.sides

/-- Theorem stating that the maximum number of intersections is the product of the number of sides -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  max_intersections config = config.inner.sides * config.outer.sides :=
sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l3412_341247


namespace NUMINAMATH_CALUDE_train_length_l3412_341299

/-- The length of a train given its speed and time to cross a fixed point. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 72 →
  time_s = 5.999520038396929 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 119.99040076793858 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3412_341299


namespace NUMINAMATH_CALUDE_can_capacity_is_30_liters_l3412_341269

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 30

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 10

/-- Checks if the given contents match the initial ratio of 4:3 -/
def isInitialRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 4 / 3

/-- Checks if the given contents match the final ratio of 5:2 -/
def isFinalRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 5 / 2

/-- Theorem stating that given the conditions, the can capacity is 30 liters -/
theorem can_capacity_is_30_liters 
  (initialContents : CanContents) 
  (hInitialRatio : isInitialRatio initialContents)
  (hFinalRatio : isFinalRatio { milk := initialContents.milk + milkAdded, water := initialContents.water })
  (hFull : initialContents.milk + initialContents.water + milkAdded = canCapacity) : 
  canCapacity = 30 := by
  sorry


end NUMINAMATH_CALUDE_can_capacity_is_30_liters_l3412_341269


namespace NUMINAMATH_CALUDE_inverse_sum_product_l3412_341239

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2 * x + y / 2 ≠ 0) :
  (2 * x + y / 2)⁻¹ * ((2 * x)⁻¹ + (y / 2)⁻¹) = (x * y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l3412_341239


namespace NUMINAMATH_CALUDE_three_letter_initials_count_l3412_341245

theorem three_letter_initials_count (n : ℕ) (h : n = 10) : n ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_initials_count_l3412_341245


namespace NUMINAMATH_CALUDE_boys_transferred_l3412_341259

theorem boys_transferred (initial_boys : ℕ) (initial_ratio_boys : ℕ) (initial_ratio_girls : ℕ)
  (final_ratio_boys : ℕ) (final_ratio_girls : ℕ) :
  initial_boys = 120 →
  initial_ratio_boys = 3 →
  initial_ratio_girls = 4 →
  final_ratio_boys = 4 →
  final_ratio_girls = 5 →
  ∃ (transferred_boys : ℕ),
    transferred_boys = 13 ∧
    ∃ (initial_girls : ℕ),
      initial_girls * initial_ratio_boys = initial_boys * initial_ratio_girls ∧
      (initial_boys - transferred_boys) * final_ratio_girls = 
      (initial_girls - 2 * transferred_boys) * final_ratio_boys :=
by sorry

end NUMINAMATH_CALUDE_boys_transferred_l3412_341259


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l3412_341250

theorem least_positive_integer_multiple_of_43 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬((3*y)^2 + 3*29*3*y + 29^2) % 43 = 0) ∧
  ((3*x)^2 + 3*29*3*x + 29^2) % 43 = 0 ∧
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_43_l3412_341250


namespace NUMINAMATH_CALUDE_tournament_properties_l3412_341274

structure Tournament :=
  (teams : ℕ)
  (scores : List ℕ)
  (win_points : ℕ)
  (draw_points : ℕ)
  (loss_points : ℕ)

def round_robin (t : Tournament) : Prop :=
  t.teams = 10 ∧ t.scores.length = 10 ∧ t.win_points = 3 ∧ t.draw_points = 1 ∧ t.loss_points = 0

theorem tournament_properties (t : Tournament) (h : round_robin t) :
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 1)).length = 2 * k) ∧
  (∃ k : ℕ, (t.scores.filter (λ x => x % 2 = 0)).length = 2 * k) ∧
  ¬(∃ a b c : ℕ, a < b ∧ b < c ∧ c < t.scores.length ∧ t.scores[a]! = 0 ∧ t.scores[b]! = 0 ∧ t.scores[c]! = 0) ∧
  (∃ scores : List ℕ, scores.length = 10 ∧ scores.sum < 135 ∧ round_robin ⟨10, scores, 3, 1, 0⟩) ∧
  (∃ m : ℕ, m ≥ 15 ∧ m ∈ t.scores) :=
by sorry

end NUMINAMATH_CALUDE_tournament_properties_l3412_341274


namespace NUMINAMATH_CALUDE_stream_speed_l3412_341227

/-- Proves that given a man rowing 84 km downstream and 60 km upstream, each taking 4 hours, the speed of the stream is 3 km/h. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 84)
  (h2 : upstream_distance = 60)
  (h3 : time = 4) :
  let boat_speed := (downstream_distance + upstream_distance) / (2 * time)
  let stream_speed := (downstream_distance - upstream_distance) / (4 * time)
  stream_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3412_341227


namespace NUMINAMATH_CALUDE_class_artworks_l3412_341203

theorem class_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : 
  num_students = 10 →
  artworks_group1 = 3 →
  artworks_group2 = 4 →
  (num_students / 2 : ℕ) * artworks_group1 + (num_students / 2 : ℕ) * artworks_group2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_artworks_l3412_341203


namespace NUMINAMATH_CALUDE_zain_coin_count_l3412_341207

/-- Represents the number of coins Emerie has -/
structure EmerieCoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has -/
def zainTotalCoins (emerie : EmerieCoinCount) : Nat :=
  (emerie.quarters + 10) + (emerie.dimes + 10) + (emerie.nickels + 10)

/-- Theorem: Given Emerie's coin counts, prove that Zain has 48 coins -/
theorem zain_coin_count (emerie : EmerieCoinCount)
  (hq : emerie.quarters = 6)
  (hd : emerie.dimes = 7)
  (hn : emerie.nickels = 5) :
  zainTotalCoins emerie = 48 := by
  sorry


end NUMINAMATH_CALUDE_zain_coin_count_l3412_341207


namespace NUMINAMATH_CALUDE_g_zero_at_three_l3412_341242

-- Define the polynomial g(x)
def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

-- Theorem statement
theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -867 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l3412_341242


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3412_341248

theorem arithmetic_evaluation : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3412_341248


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3412_341275

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3412_341275


namespace NUMINAMATH_CALUDE_power_relations_l3412_341265

theorem power_relations (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m = 4) (h4 : a^n = 3) :
  a^(-m/2) = 1/2 ∧ a^(2*m-n) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_power_relations_l3412_341265


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l3412_341212

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_part1 (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_part2 (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a x + |x - 1| ≥ 1) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l3412_341212


namespace NUMINAMATH_CALUDE_total_lives_theorem_l3412_341292

def cat_lives : ℕ := 9

def dog_lives : ℕ := cat_lives - 3

def mouse_lives : ℕ := dog_lives + 7

def elephant_lives : ℕ := 2 * cat_lives - 5

def fish_lives : ℕ := min (dog_lives + mouse_lives) (elephant_lives / 2)

theorem total_lives_theorem :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_theorem_l3412_341292


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3412_341264

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8*t - 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3412_341264


namespace NUMINAMATH_CALUDE_complex_difference_on_unit_circle_l3412_341216

theorem complex_difference_on_unit_circle (z₁ z₂ : ℂ) : 
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≤ Complex.abs (z₁ + 1 + Complex.I)) →
  (∀ z : ℂ, Complex.abs z = 1 → Complex.abs (z + 1 + Complex.I) ≥ Complex.abs (z₂ + 1 + Complex.I)) →
  Complex.abs z₁ = 1 →
  Complex.abs z₂ = 1 →
  z₁ - z₂ = Complex.mk (Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_difference_on_unit_circle_l3412_341216


namespace NUMINAMATH_CALUDE_expression_equals_one_l3412_341222

theorem expression_equals_one (x : ℝ) (h1 : x^3 ≠ 2) (h2 : x^3 ≠ -2) :
  ((x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3)^3 * ((x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3412_341222


namespace NUMINAMATH_CALUDE_difference_of_squares_103_97_l3412_341267

theorem difference_of_squares_103_97 : 
  |((103 : ℚ) / 2)^2 - ((97 : ℚ) / 2)^2| = 300 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_103_97_l3412_341267


namespace NUMINAMATH_CALUDE_special_polynomial_remainder_l3412_341254

/-- A polynomial with specific division properties -/
def SpecialPolynomial (Q : ℝ → ℝ) : Prop :=
  (∃ R₁ : ℝ → ℝ, ∀ x, Q x = (x - 15) * (R₁ x) + 7) ∧
  (∃ R₂ : ℝ → ℝ, ∀ x, Q x = (x - 10) * (R₂ x) + 2)

/-- The main theorem about the remainder of the special polynomial -/
theorem special_polynomial_remainder (Q : ℝ → ℝ) (h : SpecialPolynomial Q) :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 10) * (x - 15) * (R x) + (x - 8) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_remainder_l3412_341254


namespace NUMINAMATH_CALUDE_undetermined_zeros_l3412_341235

theorem undetermined_zeros (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : f a * f b < 0) :
  ∃ (n : ℕ), n ≥ 0 ∧ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0) ∧
  ¬ (∀ (m : ℕ), m ≠ n → ¬ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0 ∧
    (∃ (y : ℝ), y ≠ x ∧ y ∈ Set.Ioo a b ∧ f y = 0))) :=
sorry

end NUMINAMATH_CALUDE_undetermined_zeros_l3412_341235


namespace NUMINAMATH_CALUDE_adam_has_ten_apples_l3412_341288

def apples_problem (jackie_apples adam_more_apples : ℕ) : Prop :=
  let adam_apples := jackie_apples + adam_more_apples
  adam_apples = 10

theorem adam_has_ten_apples :
  apples_problem 2 8 := by sorry

end NUMINAMATH_CALUDE_adam_has_ten_apples_l3412_341288


namespace NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_2_l3412_341285

theorem max_gcd_14n_plus_5_9n_plus_2 :
  (∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) ≤ k) ∧
  (∃ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) = 4) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_2_l3412_341285


namespace NUMINAMATH_CALUDE_square_field_area_l3412_341270

/-- Given a square field where a horse takes 4 hours to run around it at 20 km/h, 
    prove that the area of the field is 400 km². -/
theorem square_field_area (s : ℝ) (h : s > 0) : 
  (4 * s = 20 * 4) → s^2 = 400 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3412_341270


namespace NUMINAMATH_CALUDE_contest_prize_money_l3412_341279

/-- The total prize money for a novel contest -/
def total_prize_money (
  total_novels : ℕ
  ) (first_prize second_prize third_prize remaining_prize : ℕ
  ) : ℕ :=
  first_prize + second_prize + third_prize + (total_novels - 3) * remaining_prize

/-- Theorem stating that the total prize money for the given contest is $800 -/
theorem contest_prize_money :
  total_prize_money 18 200 150 120 22 = 800 := by
  sorry

end NUMINAMATH_CALUDE_contest_prize_money_l3412_341279


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_append_l3412_341255

theorem smallest_three_digit_square_append (a : ℕ) : 
  (100 ≤ a ∧ a ≤ 999) →  -- a is a three-digit number
  (∃ n : ℕ, 1001 * a + 1 = n^2) →  -- appending a+1 to a results in a perfect square
  a ≥ 183 :=  -- 183 is the smallest such number
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_append_l3412_341255


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3412_341286

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : a + b + c = 35) 
  (diagonal : a^2 + b^2 + c^2 = 21^2) : 
  2 * (a*b + b*c + c*a) = 784 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3412_341286


namespace NUMINAMATH_CALUDE_square_sum_geq_product_l3412_341223

theorem square_sum_geq_product (x y z : ℝ) (h : x + y + z ≥ x * y * z) : x^2 + y^2 + z^2 ≥ x * y * z := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_l3412_341223


namespace NUMINAMATH_CALUDE_smallest_value_l3412_341213

theorem smallest_value : 
  let a := -((-3 - 2)^2)
  let b := (-3) * (-2)
  let c := (-3)^2 / (-2)^2
  let d := (-3)^2 / (-2)
  (a ≤ b) ∧ (a ≤ c) ∧ (a ≤ d) := by sorry

end NUMINAMATH_CALUDE_smallest_value_l3412_341213


namespace NUMINAMATH_CALUDE_diary_ratio_proof_l3412_341202

def diary_problem (initial_diaries : ℕ) (current_diaries : ℕ) : Prop :=
  let bought_diaries := 2 * initial_diaries
  let total_after_buying := initial_diaries + bought_diaries
  let lost_diaries := total_after_buying - current_diaries
  (lost_diaries : ℚ) / total_after_buying = 1 / 4

theorem diary_ratio_proof :
  diary_problem 8 18 := by
  sorry

end NUMINAMATH_CALUDE_diary_ratio_proof_l3412_341202


namespace NUMINAMATH_CALUDE_divisible_by_three_l3412_341220

theorem divisible_by_three (x y : ℤ) (h : 3 ∣ (x^2 + y^2)) : 3 ∣ x ∧ 3 ∣ y := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l3412_341220


namespace NUMINAMATH_CALUDE_quadratic_solution_l3412_341256

theorem quadratic_solution (b : ℝ) : (5^2 + b*5 - 35 = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3412_341256


namespace NUMINAMATH_CALUDE_outer_digits_swap_l3412_341240

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  tens_range : tens ≥ 0 ∧ tens ≤ 9
  units_range : units ≥ 0 ∧ units ≤ 9

/-- Convert a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

theorem outer_digits_swap (n : ThreeDigitNumber) 
  (h1 : n.toNum + 45 = 100 * n.hundreds + 10 * n.units + n.tens)
  (h2 : n.toNum = 100 * n.tens + 10 * n.hundreds + n.units + 270) :
  100 * n.units + 10 * n.tens + n.hundreds = n.toNum + 198 := by
  sorry

#check outer_digits_swap

end NUMINAMATH_CALUDE_outer_digits_swap_l3412_341240


namespace NUMINAMATH_CALUDE_age_squares_sum_l3412_341226

theorem age_squares_sum (T J A : ℕ) 
  (sum_TJ : T + J = 23)
  (sum_JA : J + A = 24)
  (sum_TA : T + A = 25) :
  T^2 + J^2 + A^2 = 434 := by
sorry

end NUMINAMATH_CALUDE_age_squares_sum_l3412_341226


namespace NUMINAMATH_CALUDE_sugar_amount_l3412_341293

/-- Represents the quantities of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ
  eggs : ℝ
  chocolateChips : ℝ

/-- Represents the ratios between ingredients in the bakery storage room. -/
def BakeryRatios (s : BakeryStorage) : Prop :=
  s.sugar / s.flour = 5 / 2 ∧
  s.flour / s.bakingSoda = 10 / 1 ∧
  s.eggs / s.sugar = 3 / 4 ∧
  s.chocolateChips / s.flour = 3 / 5

/-- Represents the new ratios after adding more baking soda and chocolate chips. -/
def NewRatios (s : BakeryStorage) : Prop :=
  s.flour / (s.bakingSoda + 60) = 8 / 1 ∧
  s.eggs / s.sugar = 5 / 6

/-- Theorem stating that given the conditions, the amount of sugar is 6000 pounds. -/
theorem sugar_amount (s : BakeryStorage) 
  (h1 : BakeryRatios s) (h2 : NewRatios s) : s.sugar = 6000 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_l3412_341293


namespace NUMINAMATH_CALUDE_max_got_more_candy_l3412_341284

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := 92

/-- The difference in candy pieces between Max and Frankie -/
def candy_difference : ℕ := maxs_candy - frankies_candy

theorem max_got_more_candy : candy_difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_got_more_candy_l3412_341284


namespace NUMINAMATH_CALUDE_f_monotonicity_and_properties_l3412_341231

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a + a / x

theorem f_monotonicity_and_properties :
  ∀ a : ℝ, ∀ x : ℝ, x > 0 →
  (a > 0 → (
    (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → f a x₁ > f a x₂)
  )) ∧
  (a < 0 → (
    (∀ x₁ x₂, -a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -a → f a x₁ < f a x₂)
  )) ∧
  (a = 1/2 → (
    ∀ x₀, x₀ > 0 →
    (2 - 1 / (2 * x₀^2) = 3/2 →
      ∀ x y, 3*x - 2*y + 2 = 0 ↔ y - f (1/2) x₀ = 3/2 * (x - x₀))
  )) ∧
  (a = 1/2 → ∀ x, x > 0 → f (1/2) x > Real.log x + x/2) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_properties_l3412_341231


namespace NUMINAMATH_CALUDE_systematic_sampling_distance_l3412_341237

/-- Calculates the sampling distance for systematic sampling -/
def sampling_distance (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

theorem systematic_sampling_distance :
  let population : ℕ := 1200
  let sample_size : ℕ := 30
  sampling_distance population sample_size = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_distance_l3412_341237


namespace NUMINAMATH_CALUDE_circular_sequence_three_elements_l3412_341273

/-- A circular sequence of distinct elements -/
structure CircularSequence (α : Type*) where
  elements : List α
  distinct : elements.Nodup
  circular : elements ≠ []

/-- Predicate to check if a CircularSequence contains zero -/
def containsZero (s : CircularSequence ℤ) : Prop :=
  0 ∈ s.elements

/-- Predicate to check if a CircularSequence has an odd number of elements -/
def hasOddElements (s : CircularSequence ℤ) : Prop :=
  s.elements.length % 2 = 1

/-- The main theorem -/
theorem circular_sequence_three_elements
  (s : CircularSequence ℤ)
  (zero_in_s : containsZero s)
  (odd_elements : hasOddElements s) :
  s.elements.length = 3 :=
sorry

end NUMINAMATH_CALUDE_circular_sequence_three_elements_l3412_341273


namespace NUMINAMATH_CALUDE_petya_wins_l3412_341238

/-- Represents the state of the game -/
structure GameState :=
  (contacts : Nat)
  (wires : Nat)
  (player_turn : Bool)

/-- The initial game state -/
def initial_state : GameState :=
  { contacts := 2000
  , wires := 2000 * 1999 / 2
  , player_turn := true }

/-- Represents a move in the game -/
inductive Move
  | cut_one
  | cut_three

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.cut_one => 
      { state with 
        wires := state.wires - 1
        player_turn := ¬state.player_turn }
  | Move.cut_three => 
      { state with 
        wires := state.wires - 3
        player_turn := ¬state.player_turn }

/-- Checks if the game is over -/
def is_game_over (state : GameState) : Bool :=
  state.wires < state.contacts

/-- Theorem: Player 2 (Petya) has a winning strategy -/
theorem petya_wins : 
  ∃ (strategy : GameState → Move), 
    ∀ (game : List Move), 
      is_game_over (List.foldl apply_move initial_state game) → 
        (List.length game % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_l3412_341238


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l3412_341230

theorem least_five_digit_square_cube : 
  (∀ n : ℕ, n < 15625 → (n < 10000 ∨ ¬∃ a b : ℕ, n = a^2 ∧ n = b^3)) ∧ 
  15625 ≥ 10000 ∧ 
  ∃ a b : ℕ, 15625 = a^2 ∧ 15625 = b^3 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l3412_341230


namespace NUMINAMATH_CALUDE_alchemerion_age_ratio_l3412_341217

/-- Represents the ages of Alchemerion, his son, and his father -/
structure WizardFamily where
  alchemerion : ℕ
  son : ℕ
  father : ℕ

/-- Defines the properties of the Wizard family's ages -/
def is_valid_wizard_family (f : WizardFamily) : Prop :=
  f.alchemerion = 360 ∧
  f.father = 2 * f.alchemerion + 40 ∧
  f.alchemerion + f.son + f.father = 1240 ∧
  ∃ k : ℕ, f.alchemerion = k * f.son

/-- Theorem stating that Alchemerion is 3 times older than his son -/
theorem alchemerion_age_ratio (f : WizardFamily) 
  (h : is_valid_wizard_family f) : 
  f.alchemerion = 3 * f.son :=
sorry

end NUMINAMATH_CALUDE_alchemerion_age_ratio_l3412_341217


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3412_341277

/-- An isosceles triangle with integer side lengths and perimeter 10 has a base length of 2 or 4 -/
theorem isosceles_triangle_base_length : 
  ∀ x y : ℕ, 
  x > 0 → y > 0 →
  x + x + y = 10 → 
  y = 2 ∨ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3412_341277


namespace NUMINAMATH_CALUDE_thompson_purchase_cost_l3412_341225

/-- The total cost of chickens and potatoes -/
def total_cost (num_chickens : ℕ) (chicken_price : ℝ) (potato_price : ℝ) : ℝ :=
  (num_chickens : ℝ) * chicken_price + potato_price

/-- Theorem: The total cost of 3 chickens at $3 each and a bag of potatoes at $6 is $15 -/
theorem thompson_purchase_cost : total_cost 3 3 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_thompson_purchase_cost_l3412_341225


namespace NUMINAMATH_CALUDE_exists_unreachable_positive_configuration_l3412_341224

/-- Represents a cell in the grid -/
inductive Cell
| Plus
| Minus

/-- Represents an 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Cell

/-- Represents the allowed operations -/
inductive Operation
| Flip3x3 (row col : Fin 6)  -- Top-left corner of 3x3 square
| Flip4x4 (row col : Fin 5)  -- Top-left corner of 4x4 square

/-- Applies an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Checks if a grid is all positive -/
def isAllPositive (g : Grid) : Prop :=
  ∀ i j, g i j = Cell.Plus

/-- Theorem: There exists an initial grid configuration that cannot be transformed to all positive -/
theorem exists_unreachable_positive_configuration :
  ∃ (initial : Grid), ¬∃ (ops : List Operation), isAllPositive (ops.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unreachable_positive_configuration_l3412_341224


namespace NUMINAMATH_CALUDE_no_k_exists_for_not_in_second_quadrant_l3412_341280

/-- A linear function that does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x y : ℝ, y = (k - 1) * x + k → (x < 0 → y ≤ 0)

/-- Theorem stating that there is no k for which the linear function y=(k-1)x+k does not pass through the second quadrant -/
theorem no_k_exists_for_not_in_second_quadrant :
  ¬ ∃ k : ℝ, not_in_second_quadrant k :=
sorry

end NUMINAMATH_CALUDE_no_k_exists_for_not_in_second_quadrant_l3412_341280


namespace NUMINAMATH_CALUDE_problem_solution_l3412_341262

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2-x}

theorem problem_solution (x : ℝ) (C : Set ℝ) 
  (h1 : B x ⊆ A x) 
  (h2 : B x ∪ C = A x) : 
  x = -2 ∧ C = {3} := by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_problem_solution_l3412_341262


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3412_341276

/-- The distance between the foci of a hyperbola with equation xy = 4 is 8 -/
theorem hyperbola_foci_distance :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (x y : ℝ), x * y = 4 →
    ∃ (d : ℝ), d > 0 ∧
    ∀ (P : ℝ × ℝ), P.1 * P.2 = 4 →
      Real.sqrt ((P.1 + t)^2 + (P.2 + t)^2) - Real.sqrt ((P.1 - t)^2 + (P.2 - t)^2) = d) →
  Real.sqrt ((t + t)^2 + (t + t)^2) = 8 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3412_341276


namespace NUMINAMATH_CALUDE_fixed_point_on_line_unique_intersection_l3412_341296

-- Define the lines
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem 1: Line l passes through a fixed point
theorem fixed_point_on_line : ∀ k : ℝ, line k (-2) 1 := by sorry

-- Theorem 2: Unique intersection point when k = -3
theorem unique_intersection :
  ∃! k : ℝ, ∃! x y : ℝ, line k x y ∧ line1 x y ∧ line2 x y ∧ k = -3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_unique_intersection_l3412_341296


namespace NUMINAMATH_CALUDE_farmer_apples_l3412_341211

/-- The number of apples the farmer has after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: The farmer has 4337 apples after giving away 3588 from his initial 7925 apples -/
theorem farmer_apples : remaining_apples 7925 3588 = 4337 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3412_341211


namespace NUMINAMATH_CALUDE_paper_towel_savings_l3412_341291

theorem paper_towel_savings (package_price : ℚ) (individual_price : ℚ) (rolls : ℕ) : 
  package_price = 9 → individual_price = 1 → rolls = 12 →
  (1 - package_price / (individual_price * rolls)) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_paper_towel_savings_l3412_341291


namespace NUMINAMATH_CALUDE_number_division_proof_l3412_341205

theorem number_division_proof (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Ensure all parts are positive
  a / 5 = b / 7 ∧ a / 5 = c / 4 ∧ a / 5 = d / 8 →  -- Parts are proportional
  c = 60 →  -- Smallest part is 60
  a + b + c + d = 360 :=  -- Total number is 360
by sorry

end NUMINAMATH_CALUDE_number_division_proof_l3412_341205


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3412_341268

theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 30 / 100) :
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / 
  (seeds_plot1 + seeds_plot2) = 27 / 100 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3412_341268


namespace NUMINAMATH_CALUDE_nonagon_trapezium_existence_l3412_341287

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex : Type
  | A | B | C | D | E | F | G | H | I

/-- A function to determine if four vertices form a trapezium -/
def is_trapezium (v1 v2 v3 v4 : Vertex) : Prop :=
  sorry -- The actual implementation would depend on the geometry of the nonagon

/-- Main theorem: Given any five vertices of a regular nonagon, 
    there always exists a subset of four vertices among them that form a trapezium -/
theorem nonagon_trapezium_existence 
  (chosen : Finset Vertex) 
  (h : chosen.card = 5) : 
  ∃ (v1 v2 v3 v4 : Vertex), v1 ∈ chosen ∧ v2 ∈ chosen ∧ v3 ∈ chosen ∧ v4 ∈ chosen ∧
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧
    is_trapezium v1 v2 v3 v4 :=
  sorry


end NUMINAMATH_CALUDE_nonagon_trapezium_existence_l3412_341287


namespace NUMINAMATH_CALUDE_youngest_child_age_l3412_341246

def total_bill : ℚ := 12.25
def mother_meal : ℚ := 3.75
def cost_per_year : ℚ := 0.5

structure Family :=
  (triplet_age : ℕ)
  (youngest_age : ℕ)

def valid_family (f : Family) : Prop :=
  f.youngest_age < f.triplet_age ∧
  mother_meal + cost_per_year * (3 * f.triplet_age + f.youngest_age) = total_bill

theorem youngest_child_age :
  ∃ (f₁ f₂ : Family), valid_family f₁ ∧ valid_family f₂ ∧
    f₁.youngest_age = 2 ∧ f₂.youngest_age = 5 ∧
    ∀ (f : Family), valid_family f → f.youngest_age = 2 ∨ f.youngest_age = 5 :=
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3412_341246


namespace NUMINAMATH_CALUDE_second_number_in_set_l3412_341258

theorem second_number_in_set (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + x + 13) / 3 + 9 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_set_l3412_341258


namespace NUMINAMATH_CALUDE_inequality_proof_l3412_341281

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3412_341281


namespace NUMINAMATH_CALUDE_angle_value_for_given_function_l3412_341263

/-- Given a function f(x) = sin x + √3 * cos x, prove that if there exists an acute angle θ
    such that f(θ) = 2, then θ = π/6 -/
theorem angle_value_for_given_function (θ : Real) :
  (∃ f : Real → Real, f = λ x => Real.sin x + Real.sqrt 3 * Real.cos x) →
  (0 < θ ∧ θ < π / 2) →
  (∃ f : Real → Real, f θ = 2) →
  θ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_for_given_function_l3412_341263


namespace NUMINAMATH_CALUDE_product_xyz_equals_one_l3412_341272

theorem product_xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 2) 
  (eq2 : y + 1/z = 2) 
  (eq3 : z + 1/x = 2) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_one_l3412_341272


namespace NUMINAMATH_CALUDE_f_x1_gt_f_x2_l3412_341214

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Define the theorem
theorem f_x1_gt_f_x2 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_pos f) 
  (x₁ x₂ : ℝ) 
  (h_x1_neg : x₁ < 0) 
  (h_x2_pos : x₂ > 0) 
  (h_abs : abs x₁ > abs x₂) : 
  f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_x1_gt_f_x2_l3412_341214


namespace NUMINAMATH_CALUDE_four_square_figure_perimeter_l3412_341244

/-- A figure consisting of four identical squares -/
structure FourSquareFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a four-square figure is 60 cm -/
theorem four_square_figure_perimeter (fig : FourSquareFigure) : 
  10 * fig.side_length = 60 := by
  sorry

#check four_square_figure_perimeter

end NUMINAMATH_CALUDE_four_square_figure_perimeter_l3412_341244


namespace NUMINAMATH_CALUDE_factor_w6_minus_81_l3412_341298

theorem factor_w6_minus_81 (w : ℝ) : 
  w^6 - 81 = (w - 3) * (w^2 + 3*w + 9) * (w^3 + 9) := by sorry

end NUMINAMATH_CALUDE_factor_w6_minus_81_l3412_341298


namespace NUMINAMATH_CALUDE_china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l3412_341219

-- Define the data types
structure GDPData where
  year_code : ℕ
  gdp : ℝ

-- Define the given data
def china_gdp_data : List GDPData := [
  ⟨1, 8.5⟩, ⟨2, 9.6⟩, ⟨3, 10.4⟩, ⟨4, 11⟩, ⟨5, 11.1⟩, ⟨6, 12.1⟩, ⟨7, 13.6⟩
]

-- Define the sums given in the problem
def sum_y : ℝ := 76.3
def sum_xy : ℝ := 326.2

-- Define the US GDP in 2018
def us_gdp_2018 : ℝ := 20.5

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ := 0.75 * x + 7.9

-- Theorem statement
theorem china_gdp_surpass_us (n : ℕ) :
  (linear_regression (n + 7 : ℝ) ≥ us_gdp_2018) ↔ (n + 2021 ≥ 2028) := by
  sorry

-- Prove that the linear regression equation is correct
theorem correct_regression_equation :
  ∀ x, linear_regression x = 0.75 * x + 7.9 := by
  sorry

-- Prove that China's GDP will surpass US 2018 GDP in 2028
theorem china_gdp_surpass_us_in_2028 :
  ∃ n : ℕ, n + 2021 = 2028 ∧ linear_regression (n + 7 : ℝ) ≥ us_gdp_2018 := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_surpass_us_correct_regression_equation_china_gdp_surpass_us_in_2028_l3412_341219


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3412_341266

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  /-- The number of faces of a rectangular prism -/
  faces : ℕ
  /-- The number of edges of a rectangular prism -/
  edges : ℕ
  /-- The number of vertices of a rectangular prism -/
  vertices : ℕ
  /-- A rectangular prism has 6 faces -/
  face_count : faces = 6
  /-- A rectangular prism has 12 edges -/
  edge_count : edges = 12
  /-- A rectangular prism has 8 vertices -/
  vertex_count : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l3412_341266


namespace NUMINAMATH_CALUDE_rotated_A_coordinates_l3412_341236

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)

-- Define the properties of the triangle
structure Triangle where
  A : ℝ × ℝ
  first_quadrant : A.1 > 0 ∧ A.2 > 0
  right_angle : (A.1 - B.1) * (A.1 - O.1) + (A.2 - B.2) * (A.2 - O.2) = 0
  angle_AOB : Real.arctan ((A.2 - O.2) / (A.1 - O.1)) = π / 4

-- Function to rotate a point 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- Theorem statement
theorem rotated_A_coordinates (t : Triangle) : 
  rotate90 t.A = (-8, 8) := by sorry

end NUMINAMATH_CALUDE_rotated_A_coordinates_l3412_341236


namespace NUMINAMATH_CALUDE_at_op_difference_l3412_341253

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem at_op_difference : at_op 5 9 - at_op 9 5 = 16 := by sorry

end NUMINAMATH_CALUDE_at_op_difference_l3412_341253


namespace NUMINAMATH_CALUDE_simplify_expression_a_l3412_341261

theorem simplify_expression_a (x a b : ℝ) :
  (3 * x^2 * (a^2 + b^2) - 3 * a^2 * b^2 + 3 * (x^2 + (a + b) * x + a * b) * (x * (x - a) - b * (x - a))) / x^2 = 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_a_l3412_341261


namespace NUMINAMATH_CALUDE_f_satisfies_points_l3412_341257

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := 200 - 15 * x - 15 * x^2

/-- The set of points that the function should satisfy --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 170), (2, 120), (3, 50), (4, 0)]

/-- Theorem stating that the function satisfies all given points --/
theorem f_satisfies_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

#check f_satisfies_points

end NUMINAMATH_CALUDE_f_satisfies_points_l3412_341257


namespace NUMINAMATH_CALUDE_problem_solution_l3412_341278

def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m + 3}

def B : Set ℝ := {x | -x^2 + 2*x + 8 > 0}

theorem problem_solution :
  (∀ m : ℝ, 
    (m = 2 → A m ∪ B = {x | -2 < x ∧ x ≤ 7}) ∧
    (m = 2 → (Set.univ \ A m) ∩ B = {x | -2 < x ∧ x < 1})) ∧
  (∀ m : ℝ, A m ∩ B = A m ↔ m < -4 ∨ (-1 < m ∧ m < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3412_341278


namespace NUMINAMATH_CALUDE_probability_face_then_number_standard_deck_l3412_341294

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- The probability of drawing a face card first and a number card second from a standard deck -/
def probability_face_then_number (d : Deck) : ℚ :=
  let total_face_cards := d.face_cards_per_suit * d.suits
  let total_number_cards := d.number_cards_per_suit * d.suits
  (total_face_cards * total_number_cards : ℚ) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing a face card first and a number card second from a standard deck -/
theorem probability_face_then_number_standard_deck :
  let d : Deck := {
    total_cards := 52,
    ranks := 13,
    suits := 4,
    face_cards_per_suit := 3,
    number_cards_per_suit := 9
  }
  probability_face_then_number d = 8 / 49 := by sorry

end NUMINAMATH_CALUDE_probability_face_then_number_standard_deck_l3412_341294


namespace NUMINAMATH_CALUDE_circle_rotation_invariance_l3412_341206

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a rotation
def rotate (θ : ℝ) (O : ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_rotation_invariance (S : Circle) (θ : ℝ) (O : ℝ × ℝ) :
  ∃ (S' : Circle), S'.radius = S.radius ∧
    (∀ (p : ℝ × ℝ), (p.1 - S.center.1)^2 + (p.2 - S.center.2)^2 = S.radius^2 →
      let p' := rotate θ O p
      (p'.1 - S'.center.1)^2 + (p'.2 - S'.center.2)^2 = S'.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_rotation_invariance_l3412_341206


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3412_341221

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  a + b + c = Real.sqrt 2 + 1 →  -- perimeter condition
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →  -- sine condition
  (1/2) * b * c * Real.sin A = (1/6) * Real.sin C →  -- area condition
  C = π / 3 :=  -- 60° in radians
by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3412_341221


namespace NUMINAMATH_CALUDE_x_leq_y_l3412_341282

theorem x_leq_y (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  Real.sqrt ((a - b) * (b - c)) ≤ (a - c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_leq_y_l3412_341282


namespace NUMINAMATH_CALUDE_engineering_pass_percentage_approx_l3412_341229

/-- Represents the statistics of an acoustics class -/
structure AcousticsClass where
  male_students : ℕ
  female_students : ℕ
  international_students : ℕ
  students_with_disabilities : ℕ
  male_eng_percentage : ℚ
  female_eng_percentage : ℚ
  international_eng_percentage : ℚ
  disabilities_eng_percentage : ℚ
  male_eng_pass_rate : ℚ
  female_eng_pass_rate : ℚ
  international_eng_pass_rate : ℚ
  disabilities_eng_pass_rate : ℚ

/-- Calculates the percentage of engineering students who passed the exam -/
def engineering_pass_percentage (c : AcousticsClass) : ℚ :=
  let male_eng := c.male_students * c.male_eng_percentage
  let female_eng := c.female_students * c.female_eng_percentage
  let international_eng := c.international_students * c.international_eng_percentage
  let disabilities_eng := c.students_with_disabilities * c.disabilities_eng_percentage
  let total_eng := male_eng + female_eng + international_eng + disabilities_eng
  let male_pass := male_eng * c.male_eng_pass_rate
  let female_pass := female_eng * c.female_eng_pass_rate
  let international_pass := international_eng * c.international_eng_pass_rate
  let disabilities_pass := disabilities_eng * c.disabilities_eng_pass_rate
  let total_pass := male_pass + female_pass + international_pass + disabilities_pass
  (total_pass / total_eng) * 100

/-- Theorem stating that the percentage of engineering students who passed is approximately 23.44% -/
theorem engineering_pass_percentage_approx (c : AcousticsClass) 
  (h1 : c.male_students = 120)
  (h2 : c.female_students = 100)
  (h3 : c.international_students = 70)
  (h4 : c.students_with_disabilities = 30)
  (h5 : c.male_eng_percentage = 25/100)
  (h6 : c.female_eng_percentage = 20/100)
  (h7 : c.international_eng_percentage = 15/100)
  (h8 : c.disabilities_eng_percentage = 10/100)
  (h9 : c.male_eng_pass_rate = 20/100)
  (h10 : c.female_eng_pass_rate = 25/100)
  (h11 : c.international_eng_pass_rate = 30/100)
  (h12 : c.disabilities_eng_pass_rate = 35/100) :
  ∃ ε > 0, |engineering_pass_percentage c - 2344/100| < ε :=
sorry

end NUMINAMATH_CALUDE_engineering_pass_percentage_approx_l3412_341229


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l3412_341243

theorem simplify_and_ratio (m : ℚ) : 
  let expr := (6 * m + 18) / 6
  let simplified := m + 3
  expr = simplified ∧ 
  (∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 3) := by
sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l3412_341243


namespace NUMINAMATH_CALUDE_streamers_for_confetti_l3412_341200

/-- The price relationship between streamers and confetti packages -/
def price_relationship (p q : ℝ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x

/-- The theorem stating the number of streamer packages that can be bought for 10 confetti packages -/
theorem streamers_for_confetti (p q : ℝ) :
  price_relationship p q →
  |p - q| = 90 →
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x * (1 + p / 100) = y ∧
  y * (1 - q / 100) = x ∧
  10 * x = 4 * y :=
by sorry

end NUMINAMATH_CALUDE_streamers_for_confetti_l3412_341200


namespace NUMINAMATH_CALUDE_problem_statement_l3412_341234

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) : 
  (x - 2*y)^y = 1/121 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3412_341234


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3412_341252

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ t : ℝ, m * t^2 - m * t + 4 > 0) ↔ (0 ≤ m ∧ m < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3412_341252


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l3412_341297

theorem quadratic_single_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x : ℝ, b * x^2 + 16 * x + 5 = 0) →
  (∃ x : ℝ, b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l3412_341297


namespace NUMINAMATH_CALUDE_min_value_theorem_l3412_341201

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  1/(x-1) + 3/(y-1) ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3412_341201


namespace NUMINAMATH_CALUDE_division_result_l3412_341210

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by sorry

end NUMINAMATH_CALUDE_division_result_l3412_341210


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3412_341232

def is_geometric_sequence_with_ratio_2 (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

theorem condition_necessary_not_sufficient :
  (∀ a : ℕ → ℝ, is_geometric_sequence_with_ratio_2 a → condition a) ∧
  (∃ a : ℕ → ℝ, condition a ∧ ¬is_geometric_sequence_with_ratio_2 a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3412_341232


namespace NUMINAMATH_CALUDE_tv_price_reduction_l3412_341251

/-- Proves that the price reduction percentage is 10% given the conditions of the problem -/
theorem tv_price_reduction (x : ℝ) : 
  (1 - x / 100) * 1.85 = 1.665 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_reduction_l3412_341251


namespace NUMINAMATH_CALUDE_dany_sheep_count_l3412_341208

/-- Represents the number of bushels eaten by sheep and chickens on Dany's farm -/
def farm_bushels (num_sheep : ℕ) : ℕ :=
  2 * num_sheep + 3

/-- Theorem stating that Dany has 16 sheep on his farm -/
theorem dany_sheep_count : ∃ (num_sheep : ℕ), farm_bushels num_sheep = 35 ∧ num_sheep = 16 := by
  sorry

end NUMINAMATH_CALUDE_dany_sheep_count_l3412_341208


namespace NUMINAMATH_CALUDE_fence_pole_count_l3412_341290

/-- Calculates the number of fence poles needed for a path with a bridge --/
def fence_poles (total_length : ℕ) (bridge_length : ℕ) (pole_spacing : ℕ) : ℕ :=
  2 * ((total_length - bridge_length) / pole_spacing)

/-- Theorem statement for the fence pole problem --/
theorem fence_pole_count : 
  fence_poles 900 42 6 = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_pole_count_l3412_341290


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3412_341249

/-- For a constant m, x^2 + 2x + m is a perfect square trinomial if and only if m = 1 -/
theorem perfect_square_trinomial (m : ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, x^2 + 2*x + m = (x + a)^2) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3412_341249


namespace NUMINAMATH_CALUDE_right_triangle_exists_l3412_341283

/-- Checks if three line segments can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_exists :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧
  (a = 3 ∧ b = 4 ∧ c = 5) ∧
  ¬(is_right_triangle 2 3 4) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_exists_l3412_341283


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_l3412_341218

/-- The distance between a point in polar coordinates and a line given by a polar equation -/
theorem distance_point_to_line_polar (ρ_A θ_A : ℝ) :
  let r := 2 * ρ_A * Real.sin (θ_A - π / 4) - Real.sqrt 2
  let x := ρ_A * Real.cos θ_A
  let y := ρ_A * Real.sin θ_A
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 →
  (r^2 / (1 + 1)) = (5 * Real.sqrt 2 / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_distance_point_to_line_polar_l3412_341218


namespace NUMINAMATH_CALUDE_arun_weight_average_l3412_341271

def weight_range (w : ℝ) : Prop :=
  66 < w ∧ w ≤ 69 ∧ 60 < w ∧ w < 70

theorem arun_weight_average : 
  ∃ (w₁ w₂ w₃ : ℝ), 
    weight_range w₁ ∧ 
    weight_range w₂ ∧ 
    weight_range w₃ ∧ 
    w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₂ ≠ w₃ ∧
    (w₁ + w₂ + w₃) / 3 = 68 := by
  sorry

end NUMINAMATH_CALUDE_arun_weight_average_l3412_341271


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3412_341228

theorem decimal_to_fraction : (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3412_341228


namespace NUMINAMATH_CALUDE_green_area_percentage_l3412_341209

/-- Represents a square flag with a symmetric pattern -/
structure SymmetricFlag where
  side_length : ℝ
  cross_area_percentage : ℝ
  green_area_percentage : ℝ

/-- The flag satisfies the problem conditions -/
def valid_flag (flag : SymmetricFlag) : Prop :=
  flag.cross_area_percentage = 25 ∧
  flag.green_area_percentage > 0 ∧
  flag.green_area_percentage < flag.cross_area_percentage

/-- The theorem to be proved -/
theorem green_area_percentage (flag : SymmetricFlag) :
  valid_flag flag → flag.green_area_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_area_percentage_l3412_341209


namespace NUMINAMATH_CALUDE_koi_fish_count_l3412_341233

/-- Calculates the number of koi fish after three weeks given the initial conditions and final number of goldfish --/
theorem koi_fish_count (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : 
  initial_total = 280 →
  days = 21 →
  koi_added_per_day = 2 →
  goldfish_added_per_day = 5 →
  final_goldfish = 200 →
  initial_total + days * (koi_added_per_day + goldfish_added_per_day) - final_goldfish = 227 :=
by
  sorry

#check koi_fish_count

end NUMINAMATH_CALUDE_koi_fish_count_l3412_341233


namespace NUMINAMATH_CALUDE_quadratic_solution_existence_l3412_341241

/-- A quadratic function f(x) = ax^2 + bx + c, where a ≠ 0 and a, b, c are constants. -/
def QuadraticFunction (a b c : ℝ) (h : a ≠ 0) := fun (x : ℝ) ↦ a * x^2 + b * x + c

theorem quadratic_solution_existence (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c h
  (f 6.17 = -0.03) →
  (f 6.18 = -0.01) →
  (f 6.19 = 0.02) →
  (f 6.20 = 0.04) →
  ∃ x : ℝ, (f x = 0) ∧ (6.18 < x) ∧ (x < 6.19) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_existence_l3412_341241


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_equals_neg_two_fifths_l3412_341260

/-- Given a point P(-3,4) on the terminal side of angle θ, prove that sin θ + 2cos θ = -2/5 -/
theorem sin_plus_two_cos_equals_neg_two_fifths (θ : ℝ) (P : ℝ × ℝ) :
  P = (-3, 4) →
  (∃ r : ℝ, r > 0 ∧ r * (Real.cos θ) = -3 ∧ r * (Real.sin θ) = 4) →
  Real.sin θ + 2 * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_equals_neg_two_fifths_l3412_341260


namespace NUMINAMATH_CALUDE_circle_equation_implies_a_eq_neg_one_l3412_341215

/-- A circle equation in the form x^2 + by^2 + cx + d = 0 --/
structure CircleEquation where
  b : ℝ
  c : ℝ
  d : ℝ

/-- Condition for an equation to represent a circle --/
def is_circle (eq : CircleEquation) : Prop :=
  eq.b = 1 ∧ eq.b ≠ 0

/-- The given equation x^2 + (a+2)y^2 + 2ax + a = 0 --/
def given_equation (a : ℝ) : CircleEquation :=
  { b := a + 2
  , c := 2 * a
  , d := a }

theorem circle_equation_implies_a_eq_neg_one :
  ∀ a : ℝ, is_circle (given_equation a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_a_eq_neg_one_l3412_341215


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3412_341204

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3412_341204


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_3_l3412_341295

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := 3 * x + (a - 1) * y = a - 7

-- Define parallelism
def parallel (a : ℝ) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ 
  ∀ (x y : ℝ), line1 a x y ↔ line2 a (x + k) y

-- State the theorem
theorem lines_parallel_iff_a_eq_3 : 
  ∀ (a : ℝ), parallel a ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_3_l3412_341295


namespace NUMINAMATH_CALUDE_quadratic_equation_factorization_l3412_341289

theorem quadratic_equation_factorization (n : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x^2 - 4*x + 1 = n ↔ (x - m)^2 = 5) →
  n = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_factorization_l3412_341289
