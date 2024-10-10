import Mathlib

namespace shirt_price_calculation_l1377_137709

theorem shirt_price_calculation (num_shirts : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_shirts = 6 →
  discount_rate = 1/5 →
  total_paid = 240 →
  ∃ (regular_price : ℚ), regular_price = 50 ∧ 
    num_shirts * (regular_price * (1 - discount_rate)) = total_paid := by
  sorry

end shirt_price_calculation_l1377_137709


namespace lottery_probability_l1377_137783

def sharpBallCount : ℕ := 30
def prizeBallCount : ℕ := 50
def prizeBallsDrawn : ℕ := 6

theorem lottery_probability :
  (1 : ℚ) / sharpBallCount * (1 : ℚ) / (Nat.choose prizeBallCount prizeBallsDrawn) = 1 / 476721000 := by
  sorry

end lottery_probability_l1377_137783


namespace elevator_distribution_ways_l1377_137782

/-- The number of elevators available --/
def num_elevators : ℕ := 4

/-- The number of people taking elevators --/
def num_people : ℕ := 3

/-- The number of people taking the same elevator --/
def same_elevator : ℕ := 2

/-- The number of ways to distribute people among elevators --/
def distribute_ways : ℕ := 36

/-- Theorem stating that the number of ways to distribute people among elevators is 36 --/
theorem elevator_distribution_ways :
  (num_elevators = 4) →
  (num_people = 3) →
  (same_elevator = 2) →
  (distribute_ways = 36) := by
sorry

end elevator_distribution_ways_l1377_137782


namespace inscribed_hexagon_side_length_l1377_137713

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BAC : ℝ

/-- Regular hexagon UVWXYZ inscribed in triangle ABC -/
structure InscribedHexagon where
  triangle : Triangle
  sideLength : ℝ

/-- Theorem stating the side length of the inscribed hexagon -/
theorem inscribed_hexagon_side_length (t : Triangle) (h : InscribedHexagon) 
  (h1 : t.AB = 5)
  (h2 : t.AC = 8)
  (h3 : t.BAC = π / 3)
  (h4 : h.triangle = t)
  (h5 : ∃ (U V W X Z : ℝ × ℝ), 
    U.1 + V.1 = t.AB ∧ 
    W.2 + X.2 = t.AC ∧ 
    Z.1^2 + Z.2^2 = t.AB^2 + t.AC^2 - 2 * t.AB * t.AC * Real.cos t.BAC) :
  h.sideLength = 35 / 19 := by
  sorry

end inscribed_hexagon_side_length_l1377_137713


namespace special_multiples_count_l1377_137780

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def count_special_multiples (n : ℕ) : ℕ :=
  count_multiples n 3 + count_multiples n 4 - count_multiples n 6

theorem special_multiples_count :
  count_special_multiples 3000 = 1250 := by sorry

end special_multiples_count_l1377_137780


namespace f_lower_bound_a_range_when_f2_less_than_4_l1377_137741

noncomputable section

variable (a : ℝ)
variable (h : a > 0)

def f (x : ℝ) : ℝ := |x + 1/a| + |x - a|

theorem f_lower_bound : ∀ x : ℝ, f a x ≥ 2 :=
sorry

theorem a_range_when_f2_less_than_4 : 
  f a 2 < 4 → 1 < a ∧ a < 2 + Real.sqrt 3 :=
sorry

end f_lower_bound_a_range_when_f2_less_than_4_l1377_137741


namespace fuel_cost_difference_l1377_137744

-- Define the parameters
def num_vans : ℝ := 6.0
def num_buses : ℝ := 8
def people_per_van : ℝ := 6
def people_per_bus : ℝ := 18
def van_distance : ℝ := 120
def bus_distance : ℝ := 150
def van_efficiency : ℝ := 20
def bus_efficiency : ℝ := 6
def van_fuel_cost : ℝ := 2.5
def bus_fuel_cost : ℝ := 3

-- Define the theorem
theorem fuel_cost_difference : 
  let van_total_distance := num_vans * van_distance
  let bus_total_distance := num_buses * bus_distance
  let van_fuel_consumed := van_total_distance / van_efficiency
  let bus_fuel_consumed := bus_total_distance / bus_efficiency
  let van_total_cost := van_fuel_consumed * van_fuel_cost
  let bus_total_cost := bus_fuel_consumed * bus_fuel_cost
  bus_total_cost - van_total_cost = 510 := by
  sorry

end fuel_cost_difference_l1377_137744


namespace product_not_divisible_by_72_l1377_137766

def S : Finset Nat := {4, 8, 18, 28, 36, 49, 56}

theorem product_not_divisible_by_72 (a b : Nat) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ¬(72 ∣ a * b) := by
  sorry

#check product_not_divisible_by_72

end product_not_divisible_by_72_l1377_137766


namespace set_of_naturals_less_than_three_l1377_137796

theorem set_of_naturals_less_than_three :
  {x : ℕ | x < 3} = {0, 1, 2} := by sorry

end set_of_naturals_less_than_three_l1377_137796


namespace shaded_area_comparison_l1377_137731

/-- Represents a square divided into smaller squares -/
structure DividedSquare where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares described in the problem -/
def square_I : DividedSquare := { total_divisions := 16, shaded_divisions := 4 }
def square_II : DividedSquare := { total_divisions := 64, shaded_divisions := 16 }
def square_III : DividedSquare := { total_divisions := 16, shaded_divisions := 8 }

/-- Calculates the shaded area ratio of a divided square -/
def shaded_area_ratio (s : DividedSquare) : ℚ :=
  s.shaded_divisions / s.total_divisions

/-- Theorem stating the equality of shaded areas for squares I and II, and the difference for square III -/
theorem shaded_area_comparison :
  shaded_area_ratio square_I = shaded_area_ratio square_II ∧
  shaded_area_ratio square_I ≠ shaded_area_ratio square_III ∧
  shaded_area_ratio square_II ≠ shaded_area_ratio square_III := by
  sorry

#eval shaded_area_ratio square_I
#eval shaded_area_ratio square_II
#eval shaded_area_ratio square_III

end shaded_area_comparison_l1377_137731


namespace train_speed_l1377_137720

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 441) (h2 : time = 21) :
  length / time = 21 := by
  sorry

end train_speed_l1377_137720


namespace square_difference_l1377_137792

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end square_difference_l1377_137792


namespace box_volume_correct_l1377_137738

/-- The volume of an open box created from a rectangular sheet -/
def boxVolume (L W S : ℝ) : ℝ := (L - 2*S) * (W - 2*S) * S

/-- Theorem stating that the boxVolume function correctly calculates the volume of the open box -/
theorem box_volume_correct (L W S : ℝ) (hL : L > 0) (hW : W > 0) (hS : 0 < S ∧ S < L/2 ∧ S < W/2) : 
  boxVolume L W S = (L - 2*S) * (W - 2*S) * S :=
sorry

end box_volume_correct_l1377_137738


namespace negative_fraction_greater_than_negative_decimal_l1377_137727

theorem negative_fraction_greater_than_negative_decimal : -3/4 > -0.8 := by
  sorry

end negative_fraction_greater_than_negative_decimal_l1377_137727


namespace jeff_cabinet_count_l1377_137759

/-- The number of cabinets Jeff has after installation and removal -/
def total_cabinets : ℕ :=
  let initial_cabinets := 3
  let counters_with_double := 4
  let cabinets_per_double_counter := 2 * initial_cabinets
  let additional_cabinets := [3, 5, 7]
  let cabinets_to_remove := 2

  initial_cabinets + 
  counters_with_double * cabinets_per_double_counter + 
  additional_cabinets.sum - 
  cabinets_to_remove

theorem jeff_cabinet_count : total_cabinets = 37 := by
  sorry

end jeff_cabinet_count_l1377_137759


namespace sample_variance_estimates_stability_l1377_137714

-- Define the type for sample statistics
inductive SampleStatistic
  | Mean
  | Median
  | Variance
  | Maximum

-- Define a function that determines if a statistic estimates population stability
def estimatesStability (stat : SampleStatistic) : Prop :=
  match stat with
  | SampleStatistic.Variance => True
  | _ => False

-- Theorem statement
theorem sample_variance_estimates_stability :
  ∃ (stat : SampleStatistic), estimatesStability stat ∧
  (stat = SampleStatistic.Mean ∨
   stat = SampleStatistic.Median ∨
   stat = SampleStatistic.Variance ∨
   stat = SampleStatistic.Maximum) :=
by
  sorry

end sample_variance_estimates_stability_l1377_137714


namespace complex_problem_l1377_137735

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := Complex.mk 1 2
def z₂ : ℂ := Complex.mk (-1) 1

-- Define the equation for z
def equation (a : ℝ) (z : ℂ) : Prop :=
  2 * z^2 + a * z + 10 = 0

-- Define the relationship between z and z₁z₂
def z_condition (z : ℂ) : Prop :=
  z.re = (z₁ * z₂).im

-- Main theorem
theorem complex_problem :
  ∃ (a : ℝ) (z : ℂ),
    Complex.abs (z₁ - z₂) = Real.sqrt 5 ∧
    a = 4 ∧
    (z = Complex.mk (-1) 2 ∨ z = Complex.mk (-1) (-2)) ∧
    equation a z ∧
    z_condition z := by sorry

end complex_problem_l1377_137735


namespace min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l1377_137728

def grid_size : ℕ := 2021

theorem min_colors_for_grid (n : ℕ) : 
  (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
    i < j → j < k → k < l →
    (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
      (∀ (col' : Fin grid_size), col' > col → 
        ∃ (color' : Fin n), color' ≠ color))) →
  n ≥ 506 :=
sorry

theorem colors_506_sufficient : 
  ∃ (coloring : Fin grid_size → Fin grid_size → Fin 506),
    ∀ (col : Fin grid_size) (color : Fin 506) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin 506), color' ≠ color)) :=
sorry

theorem min_colors_is_506 : 
  (∃ (n : ℕ), 
    (∀ (col : Fin grid_size) (color : Fin n) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin n), color' ≠ color))) ∧
    (∀ m < n, ¬(∀ (col : Fin grid_size) (color : Fin m) (i j k l : Fin grid_size),
      i < j → j < k → k < l →
      (∀ (row : Fin grid_size), row ≤ i ∨ l ≤ row → 
        (∀ (col' : Fin grid_size), col' > col → 
          ∃ (color' : Fin m), color' ≠ color))))) →
  n = 506 :=
sorry

end min_colors_for_grid_colors_506_sufficient_min_colors_is_506_l1377_137728


namespace min_value_sqrt_expression_l1377_137777

theorem min_value_sqrt_expression (x : ℝ) (hx : x > 0) :
  2 * Real.sqrt x + 3 / Real.sqrt x ≥ 2 * Real.sqrt 6 ∧
  (2 * Real.sqrt x + 3 / Real.sqrt x = 2 * Real.sqrt 6 ↔ x = 3/2) :=
by sorry

end min_value_sqrt_expression_l1377_137777


namespace base_4_addition_l1377_137751

/-- Convert a base 10 number to base 4 --/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Convert a base 4 number (represented as a list of digits) to base 10 --/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

/-- Add two base 4 numbers (represented as lists of digits) --/
def addBase4 (a b : List ℕ) : List ℕ :=
  sorry

theorem base_4_addition :
  addBase4 (toBase4 45) (toBase4 28) = [1, 0, 2, 1] ∧ fromBase4 [1, 0, 2, 1] = 45 + 28 := by
  sorry

end base_4_addition_l1377_137751


namespace sister_share_is_49_50_l1377_137776

/-- Calculates the amount each sister receives after Gina's spending and investments --/
def sister_share (initial_amount : ℚ) : ℚ :=
  let mom_share := initial_amount * (1 / 4)
  let clothes_share := initial_amount * (1 / 8)
  let charity_share := initial_amount * (1 / 5)
  let groceries_share := initial_amount * (15 / 100)
  let remaining_before_stocks := initial_amount - mom_share - clothes_share - charity_share - groceries_share
  let stocks_investment := remaining_before_stocks * (1 / 10)
  let final_remaining := remaining_before_stocks - stocks_investment
  final_remaining / 2

/-- Theorem stating that each sister receives $49.50 --/
theorem sister_share_is_49_50 :
  sister_share 400 = 49.50 := by sorry

end sister_share_is_49_50_l1377_137776


namespace ellipse_fixed_point_l1377_137739

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the upper vertex M of the ellipse
def upper_vertex (M : ℝ × ℝ) : Prop := 
  M.1 = 0 ∧ M.2 = 1

-- Define points A and B on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := 
  ellipse_C P.1 P.2

-- Define the slopes of lines MA and MB
def slopes_sum_2 (k₁ k₂ : ℝ) : Prop := 
  k₁ + k₂ = 2

-- Theorem statement
theorem ellipse_fixed_point 
  (M A B : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (hM : upper_vertex M) 
  (hA : on_ellipse A) 
  (hB : on_ellipse B) 
  (hk : slopes_sum_2 k₁ k₂) :
  ∃ (t : ℝ), A.1 * t + A.2 * (1 - t) = -1 ∧ 
             B.1 * t + B.2 * (1 - t) = -1 :=
sorry

end ellipse_fixed_point_l1377_137739


namespace probability_two_females_l1377_137710

/-- The probability of selecting two female contestants out of 7 total contestants 
    (4 female, 3 male) when choosing 2 contestants at random -/
theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) 
    (h1 : total = 7) 
    (h2 : females = 4) 
    (h3 : chosen = 2) : 
    (Nat.choose females chosen : Rat) / (Nat.choose total chosen : Rat) = 2 / 7 := by
  sorry

end probability_two_females_l1377_137710


namespace mango_seller_profit_l1377_137726

/-- Proves that given a fruit seller who loses 15% when selling mangoes at Rs. 6 per kg,
    if they want to sell at Rs. 7.411764705882353 per kg, their desired profit percentage is 5%. -/
theorem mango_seller_profit (loss_price : ℝ) (loss_percentage : ℝ) (desired_price : ℝ) :
  loss_price = 6 →
  loss_percentage = 15 →
  desired_price = 7.411764705882353 →
  let cost_price := loss_price / (1 - loss_percentage / 100)
  let profit_percentage := (desired_price / cost_price - 1) * 100
  profit_percentage = 5 := by
  sorry

end mango_seller_profit_l1377_137726


namespace water_displaced_squared_volume_l1377_137779

/-- The volume of water displaced by a cube in a cylindrical tank -/
def water_displaced (cube_side : ℝ) (tank_radius : ℝ) (tank_height : ℝ) : ℝ :=
  -- Definition left abstract
  sorry

/-- The main theorem stating the squared volume of water displaced -/
theorem water_displaced_squared_volume :
  let cube_side : ℝ := 10
  let tank_radius : ℝ := 5
  let tank_height : ℝ := 12
  (water_displaced cube_side tank_radius tank_height) ^ 2 = 79156.25 := by
  sorry

end water_displaced_squared_volume_l1377_137779


namespace sequence_range_l1377_137705

/-- Given a sequence {a_n} with the following properties:
  1) a_1 = a > 0
  2) a_(n+1) = -a_n^2 + t*a_n for n ∈ ℕ*
  3) There exists a real number t that makes {a_n} monotonically increasing
  Then the range of a is (0,1) -/
theorem sequence_range (a : ℝ) (t : ℝ) (a_n : ℕ → ℝ) :
  a > 0 →
  (∀ n : ℕ, n > 0 → a_n (n + 1) = -a_n n ^ 2 + t * a_n n) →
  (∃ t : ℝ, ∀ n : ℕ, n > 0 → a_n (n + 1) > a_n n) →
  a_n 1 = a →
  0 < a ∧ a < 1 :=
by sorry

end sequence_range_l1377_137705


namespace half_angle_formulas_l1377_137701

/-- For a triangle with sides a, b, c, angle α opposite side a, and semi-perimeter p = (a + b + c) / 2,
    we prove the half-angle formulas for cos and sin. -/
theorem half_angle_formulas (a b c : ℝ) (α : Real) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a < b + c ∧ b < a + c ∧ c < a + b) 
    (h_angle : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) : 
  let p := (a + b + c) / 2
  (Real.cos (α / 2))^2 = p * (p - a) / (b * c) ∧ 
  (Real.sin (α / 2))^2 = (p - b) * (p - c) / (b * c) := by
sorry

end half_angle_formulas_l1377_137701


namespace sum_in_B_l1377_137799

def A : Set ℤ := {x | ∃ k, x = 2 * k}
def B : Set ℤ := {x | ∃ k, x = 2 * k + 1}
def C : Set ℤ := {x | ∃ k, x = 4 * k + 1}

theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_in_B_l1377_137799


namespace pool_visitors_l1377_137765

theorem pool_visitors (total_earned : ℚ) (cost_per_person : ℚ) (amount_left : ℚ) 
  (h1 : total_earned = 30)
  (h2 : cost_per_person = 5/2)
  (h3 : amount_left = 5) :
  (total_earned - amount_left) / cost_per_person = 10 := by
  sorry

end pool_visitors_l1377_137765


namespace total_distance_is_11500_l1377_137767

/-- A right-angled triangle with sides XY, YZ, and ZX -/
structure RightTriangle where
  XY : ℝ
  ZX : ℝ
  YZ : ℝ
  right_angle : YZ^2 + ZX^2 = XY^2

/-- The total distance traveled in the triangle -/
def total_distance (t : RightTriangle) : ℝ :=
  t.XY + t.YZ + t.ZX

/-- Theorem: The total distance traveled in the given triangle is 11500 km -/
theorem total_distance_is_11500 :
  ∃ t : RightTriangle, t.XY = 5000 ∧ t.ZX = 4000 ∧ total_distance t = 11500 := by
  sorry

end total_distance_is_11500_l1377_137767


namespace sallys_nickels_l1377_137781

/-- The number of nickels Sally has after receiving some from her parents -/
def total_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Sally's total nickels equals the sum of her initial nickels and those received from parents -/
theorem sallys_nickels (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) :
  total_nickels initial from_dad from_mom = initial + from_dad + from_mom := by
  sorry

end sallys_nickels_l1377_137781


namespace annas_initial_candies_l1377_137775

/-- Given that Anna receives some candies from Larry and ends up with a total number of candies,
    this theorem proves how many candies Anna started with. -/
theorem annas_initial_candies
  (candies_from_larry : ℕ)
  (total_candies : ℕ)
  (h1 : candies_from_larry = 86)
  (h2 : total_candies = 91)
  : total_candies - candies_from_larry = 5 := by
  sorry

end annas_initial_candies_l1377_137775


namespace smallest_sum_is_337_dice_sum_theorem_l1377_137791

/-- Represents a set of symmetrical dice --/
structure DiceSet where
  num_dice : ℕ
  max_sum : ℕ
  min_sum : ℕ

/-- The property that the dice set can achieve a sum of 2022 --/
def can_sum_2022 (d : DiceSet) : Prop :=
  d.max_sum = 2022

/-- The property that each die is symmetrical (6-sided) --/
def symmetrical_dice (d : DiceSet) : Prop :=
  d.max_sum = 6 * d.num_dice ∧ d.min_sum = d.num_dice

/-- The theorem stating that the smallest possible sum is 337 --/
theorem smallest_sum_is_337 (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  d.min_sum = 337 := by
  sorry

/-- The main theorem combining all conditions --/
theorem dice_sum_theorem (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  ∃ (p : ℝ), p > 0 ∧ 
    (∃ (sum : ℕ), sum = 2022 ∧ sum ≤ d.max_sum) ∧
    (∃ (min_sum : ℕ), min_sum = 337 ∧ min_sum = d.min_sum) := by
  sorry

end smallest_sum_is_337_dice_sum_theorem_l1377_137791


namespace solution_to_system_of_equations_l1377_137707

theorem solution_to_system_of_equations :
  ∃ x y : ℚ, x + 2*y = 3 ∧ 9*x - 8*y = 5 ∧ x = 17/13 ∧ y = 11/13 := by
  sorry

end solution_to_system_of_equations_l1377_137707


namespace hyperbola_asymptotes_l1377_137773

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptotic line equations of a hyperbola -/
def asymptoticLines (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- Theorem stating that the given standard equation implies the asymptotic lines,
    but not necessarily vice versa -/
theorem hyperbola_asymptotes (h : Hyperbola) :
  (h.a = 4 ∧ h.b = 3 → ∀ x y, standardEquation h x y → asymptoticLines h x y) ∧
  ¬(∀ h : Hyperbola, (∀ x y, asymptoticLines h x y → standardEquation h x y)) :=
sorry

end hyperbola_asymptotes_l1377_137773


namespace quadratic_function_unique_form_l1377_137754

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

-- Define the conditions
def symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ k : ℝ, f (-1 + k) = f (-1 - k)

def y_intercept_at_one (f : ℝ → ℝ) : Prop :=
  f 0 = 1

def x_axis_intercept_length (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 2 * Real.sqrt 2

-- Main theorem
theorem quadratic_function_unique_form (f : ℝ → ℝ) :
  quadratic_function f →
  symmetric_about_negative_one f →
  y_intercept_at_one f →
  x_axis_intercept_length f →
  ∀ x, f x = -x^2 - 2*x + 1 :=
by sorry

end quadratic_function_unique_form_l1377_137754


namespace andrew_age_proof_l1377_137789

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  (grandfather_age - andrew_age = 60) ∧ (grandfather_age = 15 * andrew_age) → andrew_age = 30 / 7 := by
  sorry

end andrew_age_proof_l1377_137789


namespace trigonometric_sum_l1377_137762

theorem trigonometric_sum (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 = (1 + (Real.cos (2 * φ))^2) / 2 := by
  sorry

end trigonometric_sum_l1377_137762


namespace simplify_trig_fraction_l1377_137786

theorem simplify_trig_fraction (x : ℝ) : 
  (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = -Real.tan (x / 2) :=
by sorry

end simplify_trig_fraction_l1377_137786


namespace right_handed_players_count_l1377_137730

theorem right_handed_players_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 34 →
  throwers ≤ total_players →
  (total_players - throwers) % 3 = 0 →
  (∃ (right_handed : ℕ), 
    right_handed = throwers + 2 * ((total_players - throwers) / 3) ∧
    right_handed = 58) := by
  sorry

end right_handed_players_count_l1377_137730


namespace series_sum_is_36118_l1377_137753

/-- The sign of a term in the series based on its position -/
def sign (n : ℕ) : ℤ :=
  if n ≤ 8 then 1
  else if n ≤ 35 then -1
  else if n ≤ 80 then 1
  else if n ≤ 143 then -1
  -- Continue this pattern up to 10003
  else if n ≤ 9801 then -1
  else 1

/-- The nth term of the series -/
def term (n : ℕ) : ℤ := sign n * n

/-- The sum of the series from 1 to 10003 -/
def seriesSum : ℤ := (List.range 10003).map term |>.sum

theorem series_sum_is_36118 : seriesSum = 36118 := by
  sorry

#eval seriesSum

end series_sum_is_36118_l1377_137753


namespace inequality_proof_l1377_137758

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ + x₃*y₃ - 1)^2 ≥ (x₁^2 + x₂^2 + x₃^2 - 1)*(y₁^2 + y₂^2 + y₃^2 - 1) := by
  sorry

end inequality_proof_l1377_137758


namespace coin_flip_probability_l1377_137794

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure CoinSet :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (half_dollar : CoinFlip)
  (dollar : CoinFlip)

/-- The condition for a successful outcome -/
def successful_outcome (cs : CoinSet) : Prop :=
  (cs.penny = cs.nickel) ∧
  (cs.dime = cs.quarter) ∧ (cs.quarter = cs.half_dollar)

/-- The total number of possible outcomes -/
def total_outcomes : Nat := 64

/-- The number of successful outcomes -/
def successful_outcomes : Nat := 16

/-- The theorem to be proved -/
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 4 := by sorry

end coin_flip_probability_l1377_137794


namespace closest_to_sqrt_difference_l1377_137729

theorem closest_to_sqrt_difference : 
  let diff := Real.sqrt 145 - Real.sqrt 141
  ∀ x ∈ ({0.19, 0.20, 0.21, 0.22} : Set ℝ), 
    |diff - 0.18| < |diff - x| := by
  sorry

end closest_to_sqrt_difference_l1377_137729


namespace residue_mod_35_l1377_137740

theorem residue_mod_35 : ∃ r : ℤ, 0 ≤ r ∧ r < 35 ∧ (-963 + 100) ≡ r [ZMOD 35] ∧ r = 12 := by
  sorry

end residue_mod_35_l1377_137740


namespace mba_committee_size_l1377_137790

/-- Represents the number of second-year MBAs -/
def total_mbas : ℕ := 6

/-- Represents the number of committees -/
def num_committees : ℕ := 2

/-- Represents the probability that Jane and Albert are on the same committee -/
def same_committee_prob : ℚ := 2/5

/-- Represents the number of members in each committee -/
def committee_size : ℕ := total_mbas / num_committees

theorem mba_committee_size :
  (committee_size = 3) ∧
  (same_committee_prob = (committee_size - 1 : ℚ) / (total_mbas - 1 : ℚ)) :=
sorry

end mba_committee_size_l1377_137790


namespace fraction_inequality_l1377_137760

theorem fraction_inequality (x y : ℝ) (h : x / y = 3 / 4) :
  (2 * x + y) / y ≠ 11 / 4 := by
  sorry

end fraction_inequality_l1377_137760


namespace wedge_volume_l1377_137725

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : ℝ) : 
  d = 16 → θ = 30 → h = d → (π * d^2 * h) / 8 = 512 * π := by
  sorry

end wedge_volume_l1377_137725


namespace work_time_ratio_l1377_137757

-- Define the time taken by A to finish the work
def time_A : ℝ := 4

-- Define the combined work rate of A and B
def combined_work_rate : ℝ := 0.75

-- Define the time taken by B to finish the work
def time_B : ℝ := 2

-- Theorem statement
theorem work_time_ratio :
  (1 / time_A + 1 / time_B = combined_work_rate) →
  (time_B / time_A = 1 / 2) :=
by sorry

end work_time_ratio_l1377_137757


namespace product_of_roots_product_of_roots_specific_equation_l1377_137706

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let p := (- b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  let q := (- b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a)
  p * q = c / a :=
by sorry

theorem product_of_roots_specific_equation :
  let p := (9 + Real.sqrt (81 + 4 * 36)) / 2
  let q := (9 - Real.sqrt (81 + 4 * 36)) / 2
  p * q = -36 :=
by sorry

end product_of_roots_product_of_roots_specific_equation_l1377_137706


namespace shaded_fraction_of_specific_rectangle_l1377_137747

/-- Represents a rectangle divided into equal squares -/
structure DividedRectangle where
  total_squares : ℕ
  shaded_half_squares : ℕ

/-- Calculates the fraction of a divided rectangle that is shaded -/
def shaded_fraction (rect : DividedRectangle) : ℚ :=
  rect.shaded_half_squares / (2 * rect.total_squares)

theorem shaded_fraction_of_specific_rectangle : 
  ∀ (rect : DividedRectangle), 
    rect.total_squares = 6 → 
    rect.shaded_half_squares = 5 → 
    shaded_fraction rect = 5 / 12 := by
  sorry

end shaded_fraction_of_specific_rectangle_l1377_137747


namespace sqrt_product_simplification_l1377_137723

theorem sqrt_product_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (3 * y) * Real.sqrt (50 * y) = 30 * y * Real.sqrt (2 * y) :=
by sorry

end sqrt_product_simplification_l1377_137723


namespace thirty_first_never_sunday_l1377_137787

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- The number of days in each month -/
def daysInMonth (m : Month) (isLeapYear : Bool) : Nat :=
  match m with
  | Month.February => if isLeapYear then 29 else 28
  | Month.April | Month.June | Month.September | Month.November => 30
  | _ => 31

/-- The theorem stating that 31 is the only date that can never be a Sunday -/
theorem thirty_first_never_sunday :
  ∃! (date : Nat), date > 0 ∧ date ≤ 31 ∧
  ∀ (year : Nat) (m : Month),
    daysInMonth m (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ≥ date →
    ∃ (dow : DayOfWeek), dow ≠ DayOfWeek.Sunday :=
by
  sorry

end thirty_first_never_sunday_l1377_137787


namespace force_system_ratio_l1377_137716

/-- Two forces acting on a material point at a right angle -/
structure ForceSystem where
  f1 : ℝ
  f2 : ℝ
  resultant : ℝ

/-- The magnitudes form an arithmetic progression -/
def is_arithmetic_progression (fs : ForceSystem) : Prop :=
  ∃ (d : ℝ), fs.f2 = fs.f1 + d ∧ fs.resultant = fs.f1 + 2*d

/-- The forces act at a right angle -/
def forces_at_right_angle (fs : ForceSystem) : Prop :=
  fs.resultant^2 = fs.f1^2 + fs.f2^2

/-- The ratio of the magnitudes of the forces is 3:4 -/
def force_ratio_is_3_to_4 (fs : ForceSystem) : Prop :=
  3 * fs.f2 = 4 * fs.f1

theorem force_system_ratio (fs : ForceSystem) 
  (h1 : is_arithmetic_progression fs) 
  (h2 : forces_at_right_angle fs) : 
  force_ratio_is_3_to_4 fs :=
sorry

end force_system_ratio_l1377_137716


namespace fraction_sum_equals_decimal_l1377_137769

theorem fraction_sum_equals_decimal : 
  (1 : ℚ) / 10 + 9 / 100 + 9 / 1000 + 7 / 10000 = 0.1997 := by
  sorry

end fraction_sum_equals_decimal_l1377_137769


namespace correct_people_left_l1377_137733

/-- Calculates the number of people left on a train after two stops -/
def peopleLeftOnTrain (initialPeople : ℕ) (peopleGotOff : ℕ) (peopleGotOn : ℕ) : ℕ :=
  initialPeople - peopleGotOff + peopleGotOn

theorem correct_people_left : peopleLeftOnTrain 123 58 37 = 102 := by
  sorry

end correct_people_left_l1377_137733


namespace basketball_probability_l1377_137704

/-- A sequence of basketball shots where the probability of hitting each shot
    after the first two is equal to the proportion of shots hit so far. -/
def BasketballSequence (n : ℕ) : Type :=
  Fin n → Bool

/-- The probability of hitting exactly k shots out of n in a BasketballSequence. -/
def hitProbability (n k : ℕ) : ℚ :=
  if k = 0 ∨ k = n then 0
  else if k = 1 ∧ n = 2 then 1
  else 1 / (n - 1)

/-- The theorem stating the probability of hitting exactly k shots out of n. -/
theorem basketball_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
    hitProbability n k = 1 / (n - 1) := by
  sorry

#eval hitProbability 100 50

end basketball_probability_l1377_137704


namespace circle_equation_l1377_137756

-- Define the circle ⊙C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

def center_on_line (c : Circle) : Prop :=
  3 * c.center.1 = c.center.2

def intersects_line (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2

def triangle_area (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    1/2 * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
    (|c.center.1 - c.center.2| / Real.sqrt 2) = Real.sqrt 14

-- Theorem statement
theorem circle_equation (c : Circle) :
  tangent_to_x_axis c →
  center_on_line c →
  intersects_line c →
  triangle_area c →
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -1 ∧ c.center.2 = -3 ∧ c.radius = 3)) :=
by sorry

end circle_equation_l1377_137756


namespace equilateral_roots_ratio_l1377_137750

theorem equilateral_roots_ratio (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₂ = (Complex.exp (2*Real.pi*Complex.I/3)) * z₁ → 
  a^2 / b = 0 := by
  sorry

end equilateral_roots_ratio_l1377_137750


namespace crosswalk_distance_l1377_137718

/-- Given a parallelogram with one side of length 20 feet, height of 60 feet,
    and another side of length 80 feet, the distance between the 20-foot side
    and the 80-foot side is 15 feet. -/
theorem crosswalk_distance (side1 side2 height : ℝ) : 
  side1 = 20 → side2 = 80 → height = 60 → 
  (side1 * height) / side2 = 15 := by sorry

end crosswalk_distance_l1377_137718


namespace pyramid_surface_area_l1377_137772

/-- Represents the length of an edge in the pyramid --/
inductive EdgeLength
| ten : EdgeLength
| twenty : EdgeLength
| twentyFive : EdgeLength

/-- Represents a triangular face of the pyramid --/
structure TriangularFace where
  edge1 : EdgeLength
  edge2 : EdgeLength
  edge3 : EdgeLength

/-- Represents the pyramid WXYZ --/
structure Pyramid where
  faces : List TriangularFace
  edge_length_condition : ∀ f ∈ faces, f.edge1 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge2 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive] ∧
                                       f.edge3 ∈ [EdgeLength.ten, EdgeLength.twenty, EdgeLength.twentyFive]
  not_equilateral : ∀ f ∈ faces, f.edge1 ≠ f.edge2 ∨ f.edge1 ≠ f.edge3 ∨ f.edge2 ≠ f.edge3

/-- The surface area of a pyramid --/
def surfaceArea (p : Pyramid) : ℝ := sorry

/-- Theorem stating the surface area of the pyramid WXYZ --/
theorem pyramid_surface_area (p : Pyramid) : surfaceArea p = 100 * Real.sqrt 15 := by sorry

end pyramid_surface_area_l1377_137772


namespace sector_central_angle_l1377_137722

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 10) (h2 : area = 100) :
  (2 * area) / (r^2) = 2 := by
  sorry

end sector_central_angle_l1377_137722


namespace lowest_price_after_discounts_l1377_137721

/-- Calculates the lowest possible price of a product after applying regular and sale discounts -/
theorem lowest_price_after_discounts 
  (msrp : ℝ)
  (max_regular_discount : ℝ)
  (sale_discount : ℝ)
  (h1 : msrp = 40)
  (h2 : max_regular_discount = 0.3)
  (h3 : sale_discount = 0.2)
  : ∃ (lowest_price : ℝ), lowest_price = 22.4 :=
by
  sorry

#check lowest_price_after_discounts

end lowest_price_after_discounts_l1377_137721


namespace sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1377_137703

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_of_roots_specific :
  let a : ℝ := 10
  let b : ℝ := 15
  let c : ℝ := -20
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 25/4 :=
by sorry

end sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l1377_137703


namespace annie_milkshakes_l1377_137793

/-- The number of milkshakes Annie bought -/
def milkshakes : ℕ := sorry

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- Annie's initial amount of money in dollars -/
def initial_money : ℕ := 132

/-- Annie's remaining money after purchases in dollars -/
def remaining_money : ℕ := 70

theorem annie_milkshakes :
  milkshakes = 6 ∧
  initial_money = remaining_money + hamburgers_bought * hamburger_cost + milkshakes * milkshake_cost :=
by sorry

end annie_milkshakes_l1377_137793


namespace min_sum_of_product_72_l1377_137711

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (a + b = -17) :=
sorry

end min_sum_of_product_72_l1377_137711


namespace factor_implies_k_equals_five_l1377_137736

theorem factor_implies_k_equals_five (m k : ℤ) : 
  (∃ (A B : ℤ), m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * (A*m + B)) → k = 5 := by
  sorry

end factor_implies_k_equals_five_l1377_137736


namespace factorization_proof_l1377_137745

theorem factorization_proof (a b : ℝ) : 4 * a^2 * (a - b) - (a - b) = (a - b) * (2*a + 1) * (2*a - 1) := by
  sorry

end factorization_proof_l1377_137745


namespace least_jumps_to_19999_l1377_137732

/-- Represents the total distance jumped after n jumps -/
def totalDistance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distance of the nth jump -/
def nthJump (n : ℕ) : ℕ := n

theorem least_jumps_to_19999 :
  ∀ k : ℕ, (totalDistance k ≥ 19999 → k ≥ 201) ∧
  (∃ (adjustedJump : ℤ), 
    totalDistance 201 + nthJump 201 + adjustedJump = 19999 ∧ 
    adjustedJump.natAbs < nthJump 201) := by
  sorry

end least_jumps_to_19999_l1377_137732


namespace max_value_of_expression_l1377_137743

theorem max_value_of_expression (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 4 :=
sorry

end max_value_of_expression_l1377_137743


namespace incorrect_average_calculation_l1377_137761

theorem incorrect_average_calculation (n : ℕ) (incorrect_num correct_num : ℝ) (correct_avg : ℝ) :
  n = 10 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 65 ∧ 
  correct_avg = 50 →
  ∃ (other_sum : ℝ),
    (other_sum + correct_num) / n = correct_avg ∧
    (other_sum + incorrect_num) / n = 46 :=
by sorry

end incorrect_average_calculation_l1377_137761


namespace triangles_in_decagon_l1377_137795

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def numTrianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def numVerticesInDecagon : ℕ := 10

theorem triangles_in_decagon :
  numTrianglesInDecagon = (numVerticesInDecagon.choose 3) := by
  sorry

end triangles_in_decagon_l1377_137795


namespace min_value_reciprocal_sum_l1377_137768

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = (3 / 2 + Real.sqrt 2) ∧ ∀ z w, z > 0 → w > 0 → 2 * z + w = 2 → 1 / z + 1 / w ≥ min :=
by sorry

end min_value_reciprocal_sum_l1377_137768


namespace union_complement_problem_l1377_137717

theorem union_complement_problem (U A B : Set Char) : 
  U = {'a', 'b', 'c', 'd', 'e'} →
  A = {'b', 'c', 'd'} →
  B = {'b', 'e'} →
  B ∪ (U \ A) = {'a', 'b', 'e'} := by
sorry

end union_complement_problem_l1377_137717


namespace car_impact_suitable_for_sampling_l1377_137724

/-- Characteristics of a suitable sampling survey scenario -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_survey : Bool
  representative_sample_possible : Bool

/-- Options for the survey scenario -/
inductive SurveyOption
  | A  -- Understanding the height of students in Class 7(1)
  | B  -- Companies recruiting and interviewing job applicants
  | C  -- Investigating the impact resistance of a batch of cars
  | D  -- Selecting the fastest runner in our school for competition

/-- Determine if an option is suitable for sampling survey -/
def is_suitable_for_sampling (option : SurveyOption) : Bool :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Characteristics of the car impact resistance scenario -/
def car_impact_scenario : SamplingSurveyCharacteristics :=
  { large_population := true,
    impractical_full_survey := true,
    representative_sample_possible := true }

/-- Theorem stating that investigating car impact resistance is suitable for sampling survey -/
theorem car_impact_suitable_for_sampling :
  is_suitable_for_sampling SurveyOption.C ∧
  car_impact_scenario.large_population ∧
  car_impact_scenario.impractical_full_survey ∧
  car_impact_scenario.representative_sample_possible :=
sorry

end car_impact_suitable_for_sampling_l1377_137724


namespace power_of_negative_two_a_cubed_l1377_137755

theorem power_of_negative_two_a_cubed (a : ℝ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end power_of_negative_two_a_cubed_l1377_137755


namespace number_ratio_l1377_137778

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 117) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l1377_137778


namespace barbi_weight_loss_duration_l1377_137784

/-- Proves that Barbi lost weight for 12 months given the conditions -/
theorem barbi_weight_loss_duration :
  let barbi_monthly_loss : ℝ := 1.5
  let luca_total_loss : ℝ := 9 * 11
  let loss_difference : ℝ := 81
  
  ∃ (months : ℝ), 
    months * barbi_monthly_loss = luca_total_loss - loss_difference ∧ 
    months = 12 := by
  sorry

end barbi_weight_loss_duration_l1377_137784


namespace complement_intersection_theorem_l1377_137770

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4} := by sorry

end complement_intersection_theorem_l1377_137770


namespace dishonest_dealer_profit_percentage_l1377_137774

/-- Calculates the profit percentage of a dishonest dealer. -/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_positive : 0 < claimed_weight)
  (h_weights : actual_weight = 575 ∧ claimed_weight = 1000) :
  (claimed_weight - actual_weight) / claimed_weight * 100 = 42.5 :=
by
  sorry

#check dishonest_dealer_profit_percentage

end dishonest_dealer_profit_percentage_l1377_137774


namespace stock_price_change_l1377_137719

theorem stock_price_change (P1 P2 D : ℝ) (h1 : D = 0.18 * P1) (h2 : D = 0.12 * P2) :
  P2 = 1.5 * P1 := by
  sorry

end stock_price_change_l1377_137719


namespace min_reciprocal_sum_l1377_137798

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), (2*a*x - b*y + 2 = 0 ∧ 
                 x^2 + y^2 + 2*x - 4*y + 1 = 0) ∧
   (∃ (x1 y1 x2 y2 : ℝ), 
      (2*a*x1 - b*y1 + 2 = 0 ∧ x1^2 + y1^2 + 2*x1 - 4*y1 + 1 = 0) ∧
      (2*a*x2 - b*y2 + 2 = 0 ∧ x2^2 + y2^2 + 2*x2 - 4*y2 + 1 = 0) ∧
      ((x1 - x2)^2 + (y1 - y2)^2 = 16))) →
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 1/a + 1/b) →
  1/a + 1/b = 2 :=
by sorry

end min_reciprocal_sum_l1377_137798


namespace fresh_fruit_water_content_l1377_137785

theorem fresh_fruit_water_content
  (dried_water_content : ℝ)
  (dried_weight : ℝ)
  (fresh_weight : ℝ)
  (h1 : dried_water_content = 0.15)
  (h2 : dried_weight = 12)
  (h3 : fresh_weight = 101.99999999999999) :
  (fresh_weight - dried_weight * (1 - dried_water_content)) / fresh_weight = 0.9 := by
sorry

end fresh_fruit_water_content_l1377_137785


namespace world_grain_supply_l1377_137746

/-- World grain supply problem -/
theorem world_grain_supply :
  let world_grain_demand : ℝ := 2400000
  let supply_ratio : ℝ := 0.75
  let world_grain_supply : ℝ := supply_ratio * world_grain_demand
  world_grain_supply = 1800000 := by
  sorry

end world_grain_supply_l1377_137746


namespace angle_sum_in_circle_l1377_137700

/-- Given a circle with four angles around its center measured as 7x°, 3x°, 4x°, and x°,
    prove that x = 24°. -/
theorem angle_sum_in_circle (x : ℝ) : 
  (7 * x + 3 * x + 4 * x + x : ℝ) = 360 → x = 24 := by
  sorry

end angle_sum_in_circle_l1377_137700


namespace range_of_a_l1377_137788

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, p x → q x a) →
  (∃ x : ℝ, q x a ∧ ¬p x) →
  (0 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l1377_137788


namespace triangle_perimeter_l1377_137734

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 12 = 0 →
  x > 0 →
  4 + x > 7 ∧ 7 + x > 4 ∧ 4 + 7 > x →
  4 + 7 + x = 17 :=
by
  sorry

end triangle_perimeter_l1377_137734


namespace conference_handshakes_l1377_137771

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group_a : ℕ)
  (group_b : ℕ)
  (h_total : total = group_a + group_b)

/-- Calculates the number of handshakes in a conference -/
def handshakes (c : Conference) : ℕ :=
  c.group_a * c.group_b + (c.group_b.choose 2)

/-- Theorem stating the number of handshakes in the specific conference -/
theorem conference_handshakes :
  ∃ (c : Conference), c.total = 40 ∧ c.group_a = 25 ∧ c.group_b = 15 ∧ handshakes c = 480 :=
by sorry

end conference_handshakes_l1377_137771


namespace sister_watermelons_count_l1377_137764

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of slices Danny's sister cuts each watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := 45

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

theorem sister_watermelons_count : sister_watermelons = 
  (total_slices - danny_watermelons * danny_slices_per_watermelon) / sister_slices_per_watermelon := by
  sorry

end sister_watermelons_count_l1377_137764


namespace power_division_rule_l1377_137712

theorem power_division_rule (x : ℝ) : x^6 / x^3 = x^3 := by
  sorry

end power_division_rule_l1377_137712


namespace age_difference_l1377_137749

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A = C + 17 := by
  sorry

end age_difference_l1377_137749


namespace profit_percentage_calculation_l1377_137742

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 290)
  (h2 : cost_price = 241.67) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end profit_percentage_calculation_l1377_137742


namespace common_divisors_count_l1377_137702

def a : Nat := 12600
def b : Nat := 14400

theorem common_divisors_count : (Nat.divisors (Nat.gcd a b)).card = 45 := by
  sorry

end common_divisors_count_l1377_137702


namespace max_probability_two_color_balls_l1377_137797

def p (n : ℕ+) : ℚ :=
  (10 * n) / ((n + 5) * (n + 4))

theorem max_probability_two_color_balls :
  ∀ n : ℕ+, p n ≤ 5/9 :=
by sorry

end max_probability_two_color_balls_l1377_137797


namespace solve_equation_l1377_137748

theorem solve_equation (x : ℝ) : 3 * x + 36 = 48 → x = 4 := by
  sorry

end solve_equation_l1377_137748


namespace cubic_expansion_coefficient_relation_l1377_137708

theorem cubic_expansion_coefficient_relation :
  ∀ (a₀ a₁ a₂ a₃ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -1 := by
  sorry

end cubic_expansion_coefficient_relation_l1377_137708


namespace library_books_count_l1377_137752

theorem library_books_count :
  ∃ (n : ℕ), 
    500 < n ∧ n < 650 ∧ 
    ∃ (r : ℕ), n = 12 * r + 7 ∧
    ∃ (l : ℕ), n = 25 * l - 5 ∧
    n = 595 := by
  sorry

end library_books_count_l1377_137752


namespace rectangle_arrangement_perimeters_l1377_137715

/- Define the properties of the rectangles -/
def identical_rectangles (l w : ℝ) : Prop := l = 2 * w

/- Define the first arrangement's perimeter -/
def first_arrangement_perimeter (l w : ℝ) : ℝ := 3 * l + 4 * w

/- Define the second arrangement's perimeter -/
def second_arrangement_perimeter (l w : ℝ) : ℝ := 6 * l + 2 * w

/- Theorem statement -/
theorem rectangle_arrangement_perimeters (l w : ℝ) :
  identical_rectangles l w →
  first_arrangement_perimeter l w = 20 →
  second_arrangement_perimeter l w = 28 := by
  sorry

end rectangle_arrangement_perimeters_l1377_137715


namespace quadratic_roots_relations_l1377_137763

/-- Given complex numbers a, b, c satisfying certain conditions, prove specific algebraic relations -/
theorem quadratic_roots_relations (a b c : ℂ) 
  (h1 : a + b ≠ 0)
  (h2 : b + c ≠ 0)
  (h3 : c + a ≠ 0)
  (h4 : ∀ (x : ℂ), (x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) → 
    ∃ (y : ℂ), y^2 + a*y + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = -y)
  (h5 : ∀ (x : ℂ), (x^2 + b*x + c = 0 ∧ x^2 + c*x + a = 0) → 
    ∃ (y : ℂ), y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0 ∧ x = -y)
  (h6 : ∀ (x : ℂ), (x^2 + c*x + a = 0 ∧ x^2 + a*x + b = 0) → 
    ∃ (y : ℂ), y^2 + c*y + a = 0 ∧ y^2 + a*y + b = 0 ∧ x = -y) :
  a^2 + b^2 + c^2 = 18 ∧ 
  a^2*b + b^2*c + c^2*a = 27 ∧ 
  a^3*b^2 + b^3*c^2 + c^3*a^2 = -162 := by
sorry

end quadratic_roots_relations_l1377_137763


namespace solve_equation_l1377_137737

theorem solve_equation (x : ℝ) : 144 / 0.144 = 14.4 / x → x = 0.0144 := by
  sorry

end solve_equation_l1377_137737
