import Mathlib

namespace heather_blocks_l1893_189399

/-- Given that Heather starts with 86 blocks and shares 41 blocks,
    prove that she ends up with 45 blocks. -/
theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (final_blocks : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_blocks = 41)
  (h3 : final_blocks = initial_blocks - shared_blocks) :
  final_blocks = 45 := by
  sorry

end heather_blocks_l1893_189399


namespace fill_with_corners_l1893_189326

/-- A type representing a box with integer dimensions -/
structure Box where
  m : ℕ
  n : ℕ
  k : ℕ
  m_gt_one : m > 1
  n_gt_one : n > 1
  k_gt_one : k > 1

/-- A type representing a 1 × 1 × 3 bar -/
structure Bar

/-- A type representing a corner made from three 1 × 1 × 1 cubes -/
structure Corner

/-- A function that checks if a box can be filled with bars and corners -/
def canFillWithBarsAndCorners (b : Box) : Prop :=
  ∃ (bars : ℕ) (corners : ℕ), bars * 3 + corners * 3 = b.m * b.n * b.k

/-- A function that checks if a box can be filled with only corners -/
def canFillWithOnlyCorners (b : Box) : Prop :=
  ∃ (corners : ℕ), corners * 3 = b.m * b.n * b.k

/-- The main theorem to be proved -/
theorem fill_with_corners (b : Box) :
  canFillWithBarsAndCorners b → canFillWithOnlyCorners b :=
by sorry

end fill_with_corners_l1893_189326


namespace equation_roots_imply_sum_l1893_189348

/-- Given two equations with constants a and b, prove that 100a + b = 156 -/
theorem equation_roots_imply_sum (a b : ℝ) : 
  (∃! x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x + a) * (x + b) * (x + 12) = 0 ∧
    (y + a) * (y + b) * (y + 12) = 0 ∧
    (z + a) * (z + b) * (z + 12) = 0 ∧
    x ≠ -3 ∧ y ≠ -3 ∧ z ≠ -3) →
  (∃! w, (w + 2*a) * (w + 3) * (w + 6) = 0 ∧ 
    w + b ≠ 0 ∧ w + 12 ≠ 0) →
  100 * a + b = 156 := by
sorry

end equation_roots_imply_sum_l1893_189348


namespace box_side_length_l1893_189311

/-- Proves that the length of one side of a cubic box is approximately 18.17 inches
    given the cost per box, total volume needed, and total cost. -/
theorem box_side_length (cost_per_box : ℝ) (total_volume : ℝ) (total_cost : ℝ)
  (h1 : cost_per_box = 1.30)
  (h2 : total_volume = 3.06 * 1000000)
  (h3 : total_cost = 663)
  : ∃ (side_length : ℝ), abs (side_length - 18.17) < 0.01 := by
  sorry


end box_side_length_l1893_189311


namespace birds_on_fence_l1893_189325

/-- Given an initial number of birds and a final number of birds on a fence,
    calculate the number of additional birds that joined. -/
def additional_birds (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that given 6 initial birds and 10 final birds on a fence,
    the number of additional birds that joined is 4. -/
theorem birds_on_fence : additional_birds 6 10 = 4 := by
  sorry

end birds_on_fence_l1893_189325


namespace line_slope_is_two_l1893_189373

/-- The slope of a line given by the equation 3y - 6x = 9 is 2 -/
theorem line_slope_is_two : 
  ∀ (x y : ℝ), 3 * y - 6 * x = 9 → (∃ b : ℝ, y = 2 * x + b) :=
by sorry

end line_slope_is_two_l1893_189373


namespace discriminant_of_5x2_minus_3x_plus_4_l1893_189359

/-- The discriminant of a quadratic equation ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The discriminant of the quadratic equation 5x² - 3x + 4 is -71 -/
theorem discriminant_of_5x2_minus_3x_plus_4 :
  discriminant 5 (-3) 4 = -71 := by sorry

end discriminant_of_5x2_minus_3x_plus_4_l1893_189359


namespace three_tangent_lines_imply_a_8_symmetry_of_circle_C_l1893_189304

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x + x + 2*y - 1 + m = 0

-- Define the curve (another circle)
def curve (a x y : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Three common tangent lines imply a = 8
theorem three_tangent_lines_imply_a_8 :
  (∃ (a : ℝ), (∀ x y : ℝ, curve a x y) ∧ 
  (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop), 
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → (curve a x y ∨ circle_C x y)) ∧
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → 
      (∃ ε > 0, ∀ x' y' : ℝ, ((x' - x)^2 + (y' - y)^2 < ε^2) → 
        ¬(curve a x' y' ∧ circle_C x' y'))))) →
  a = 8 :=
sorry

-- Theorem 2: Symmetry of circle C with respect to line l when m = 1
theorem symmetry_of_circle_C :
  ∀ x y : ℝ, line_l 1 x y → 
  (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x + x')/2 = x ∧ (y + y')/2 = y) ∧ 
    (x^2 + (y-2)^2 = 4)) :=
sorry

end three_tangent_lines_imply_a_8_symmetry_of_circle_C_l1893_189304


namespace max_squared_ratio_is_four_thirds_l1893_189381

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 →
      (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3

theorem max_squared_ratio_is_four_thirds (a b : ℝ) :
  max_squared_ratio a b :=
sorry

end max_squared_ratio_is_four_thirds_l1893_189381


namespace f_bounded_iff_alpha_in_unit_interval_l1893_189388

/-- The function f defined on pairs of nonnegative integers -/
noncomputable def f (α : ℝ) : ℕ → ℕ → ℝ
| 0, 0 => 1
| m, 0 => 0
| 0, n => 0
| (m+1), (n+1) => α * f α m (n+1) + (1 - α) * f α m n

/-- The theorem statement -/
theorem f_bounded_iff_alpha_in_unit_interval (α : ℝ) :
  (∀ m n : ℕ, |f α m n| < 1989) ↔ 0 < α ∧ α < 1 := by
  sorry

end f_bounded_iff_alpha_in_unit_interval_l1893_189388


namespace hemisphere_surface_area_l1893_189356

/-- The total surface area of a hemisphere with radius 9 cm, including its circular base, is 243π cm². -/
theorem hemisphere_surface_area :
  let r : ℝ := 9
  let base_area : ℝ := π * r^2
  let curved_area : ℝ := 2 * π * r^2
  let total_area : ℝ := base_area + curved_area
  total_area = 243 * π := by sorry

end hemisphere_surface_area_l1893_189356


namespace koala_fiber_intake_l1893_189337

/-- Represents the amount of fiber a koala eats and absorbs in a day. -/
structure KoalaFiber where
  eaten : ℝ
  absorbed : ℝ
  absorption_rate : ℝ
  absorption_equation : absorbed = absorption_rate * eaten

/-- Theorem: If a koala absorbs 20% of the fiber it eats and absorbed 12 ounces
    of fiber in one day, then it ate 60 ounces of fiber that day. -/
theorem koala_fiber_intake (k : KoalaFiber) 
    (h1 : k.absorption_rate = 0.20)
    (h2 : k.absorbed = 12) : 
    k.eaten = 60 := by
  sorry

end koala_fiber_intake_l1893_189337


namespace slope_of_line_l1893_189378

/-- The slope of a line given by the equation y/4 - x/5 = 2 is 4/5 -/
theorem slope_of_line (x y : ℝ) :
  y / 4 - x / 5 = 2 → (∃ b : ℝ, y = 4 / 5 * x + b) :=
by sorry

end slope_of_line_l1893_189378


namespace night_rides_calculation_wills_ferris_wheel_rides_l1893_189322

/-- Calculates the number of night rides on a Ferris wheel -/
def night_rides (total_rides day_rides : ℕ) : ℕ :=
  total_rides - day_rides

/-- Theorem: The number of night rides is equal to the total rides minus the day rides -/
theorem night_rides_calculation (total_rides day_rides : ℕ) 
  (h : day_rides ≤ total_rides) : 
  night_rides total_rides day_rides = total_rides - day_rides := by
  sorry

/-- Given Will's specific scenario -/
theorem wills_ferris_wheel_rides : 
  night_rides 13 7 = 6 := by
  sorry

end night_rides_calculation_wills_ferris_wheel_rides_l1893_189322


namespace absolute_value_sqrt_square_expression_l1893_189305

theorem absolute_value_sqrt_square_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end absolute_value_sqrt_square_expression_l1893_189305


namespace computer_price_increase_l1893_189344

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 →
  ((364 - d) / d) * 100 = 30 := by
sorry

end computer_price_increase_l1893_189344


namespace perpendicular_bisector_correct_parallel_line_correct_l1893_189375

-- Define points A and B
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem for the perpendicular bisector
theorem perpendicular_bisector_correct : 
  perp_bisector = λ x y => (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line_correct : 
  ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ parallel_line x y ∧
  ∃ k : ℝ, ∀ x' y' : ℝ, parallel_line x' y' ↔ line3 x' y' ∧ (y' - y = k * (x' - x)) :=
sorry

end perpendicular_bisector_correct_parallel_line_correct_l1893_189375


namespace smallest_shift_l1893_189330

-- Define a periodic function g with period 25
def g (x : ℝ) : ℝ := sorry

-- Define the period of g
def period : ℝ := 25

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x - period) = g x

-- Define the property we want to prove
def property (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 4) = g (x / 4)

-- State the theorem
theorem smallest_shift :
  (∃ a > 0, property a) ∧ (∀ a > 0, property a → a ≥ 100) :=
sorry

end smallest_shift_l1893_189330


namespace third_trial_point_l1893_189366

/-- The 0.618 method for optimization --/
def golden_ratio : ℝ := 0.618

/-- The lower bound of the initial range --/
def lower_bound : ℝ := 100

/-- The upper bound of the initial range --/
def upper_bound : ℝ := 1100

/-- Calculate the first trial point --/
def x₁ : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

/-- Calculate the second trial point --/
def x₂ : ℝ := lower_bound + (upper_bound - x₁)

/-- Calculate the third trial point --/
def x₃ : ℝ := lower_bound + golden_ratio * (x₂ - lower_bound)

/-- The theorem to be proved --/
theorem third_trial_point : ⌊x₃⌋ = 336 := by sorry

end third_trial_point_l1893_189366


namespace students_playing_both_sports_l1893_189301

/-- Given a class of students, calculate the number of students who play both football and long tennis. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 35) 
  (h2 : football = 26) 
  (h3 : tennis = 20) 
  (h4 : neither = 6) : 
  football + tennis - (total - neither) = 17 := by
  sorry

end students_playing_both_sports_l1893_189301


namespace extended_pattern_ratio_l1893_189335

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (black : ℕ)
  (white : ℕ)

/-- Represents the extended pattern with a black border -/
def extendPattern (p : TilePattern) : TilePattern :=
  let side := Nat.sqrt (p.black + p.white)
  let newBlack := p.black + 4 * side + 4
  { black := newBlack, white := p.white }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) :
  p.black = 13 ∧ p.white = 23 →
  let ep := extendPattern p
  (ep.black : ℚ) / ep.white = 41 / 23 := by
  sorry


end extended_pattern_ratio_l1893_189335


namespace square_difference_cube_and_sixth_power_equation_l1893_189309

theorem square_difference_cube_and_sixth_power_equation :
  (∀ m : ℕ, m > 1 → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3) ∧
  (∀ x y : ℕ, x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 → x = 4 ∧ y = 63) :=
by
  sorry

#check square_difference_cube_and_sixth_power_equation

end square_difference_cube_and_sixth_power_equation_l1893_189309


namespace new_person_weight_l1893_189314

/-- Given a group of 6 people, if replacing one person weighing 75 kg with a new person
    increases the average weight by 4.5 kg, then the weight of the new person is 102 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 6 →
  weight_increase = 4.5 →
  replaced_weight = 75 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 102 :=
by sorry

end new_person_weight_l1893_189314


namespace golden_ratio_problem_l1893_189390

theorem golden_ratio_problem (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * Real.cos (27 * π / 180)^2 - 1) = 2 := by
sorry

end golden_ratio_problem_l1893_189390


namespace breakfast_calories_proof_l1893_189329

/-- Calculates the breakfast calories given the daily calorie limit, remaining calories, dinner calories, and lunch calories. -/
def breakfast_calories (daily_limit : ℕ) (remaining : ℕ) (dinner : ℕ) (lunch : ℕ) : ℕ :=
  daily_limit - remaining - (dinner + lunch)

/-- Proves that given the specific calorie values, the breakfast calories are 560. -/
theorem breakfast_calories_proof :
  breakfast_calories 2500 525 635 780 = 560 := by
  sorry

end breakfast_calories_proof_l1893_189329


namespace xy_equals_four_l1893_189318

theorem xy_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdistinct : x ≠ y) 
  (h_eq : x + 4 / x = y + 4 / y) : x * y = 4 := by
  sorry

end xy_equals_four_l1893_189318


namespace internal_diagonal_cubes_l1893_189308

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: In a 200 × 325 × 376 rectangular solid, an internal diagonal passes through 868 unit cubes -/
theorem internal_diagonal_cubes : cubes_passed 200 325 376 = 868 := by
  sorry

end internal_diagonal_cubes_l1893_189308


namespace century_park_weed_removal_l1893_189351

/-- Represents the weed growth and removal scenario in Century Park --/
structure WeedScenario where
  weed_growth_rate : ℝ
  worker_removal_rate : ℝ
  day1_duration : ℕ
  day2_workers : ℕ
  day2_duration : ℕ

/-- Calculates the finish time for day 3 given a WeedScenario --/
def day3_finish_time (scenario : WeedScenario) : ℕ :=
  sorry

/-- The theorem states that given the specific scenario, 8 workers will finish at 8:38 AM on day 3 --/
theorem century_park_weed_removal 
  (scenario : WeedScenario)
  (h1 : scenario.day1_duration = 60)
  (h2 : scenario.day2_workers = 10)
  (h3 : scenario.day2_duration = 30) :
  day3_finish_time scenario = 38 :=
sorry

end century_park_weed_removal_l1893_189351


namespace intersection_product_l1893_189370

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- Line l in polar coordinates -/
def line_l (k : ℝ) (θ : ℝ) : Prop :=
  k > 0 ∧ θ ∈ Set.Ioo 0 (Real.pi / 2)

/-- Intersection points of curve C and line l -/
def intersection_points (ρ₁ ρ₂ θ : ℝ) : Prop :=
  curve_C ρ₁ θ ∧ curve_C ρ₂ θ ∧ ∃ k, line_l k θ

theorem intersection_product (ρ₁ ρ₂ θ : ℝ) :
  intersection_points ρ₁ ρ₂ θ → |ρ₁ * ρ₂| = 3 := by
  sorry

end intersection_product_l1893_189370


namespace rectangular_solid_surface_area_l1893_189333

/-- A rectangular solid with prime edge lengths and volume 1001 has surface area 622 -/
theorem rectangular_solid_surface_area :
  ∀ (a b c : ℕ),
  Prime a → Prime b → Prime c →
  a * b * c = 1001 →
  2 * (a * b + b * c + c * a) = 622 := by
sorry

end rectangular_solid_surface_area_l1893_189333


namespace binomial_8_choose_5_l1893_189368

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l1893_189368


namespace investment_average_rate_l1893_189389

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) (x : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.07 →
  rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.042 :=
by sorry

end investment_average_rate_l1893_189389


namespace gina_rose_cups_per_hour_l1893_189392

theorem gina_rose_cups_per_hour :
  let lily_cups_per_hour : ℕ := 7
  let order_rose_cups : ℕ := 6
  let order_lily_cups : ℕ := 14
  let total_payment : ℕ := 90
  let hourly_rate : ℕ := 30
  let rose_cups_per_hour : ℕ := order_rose_cups / (total_payment / hourly_rate - order_lily_cups / lily_cups_per_hour)
  rose_cups_per_hour = 6 := by
sorry

end gina_rose_cups_per_hour_l1893_189392


namespace degree_to_radian_conversion_l1893_189316

theorem degree_to_radian_conversion (π : Real) :
  (180 : Real) = π → (300 : Real) * π / 180 = 5 * π / 3 := by sorry

end degree_to_radian_conversion_l1893_189316


namespace adam_figurines_l1893_189364

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end adam_figurines_l1893_189364


namespace strawberry_jelly_sales_l1893_189362

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales based on the given conditions -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry * 3 = s.grape ∧
  s.plum = 6

/-- Theorem stating that given the conditions, 18 jars of strawberry jelly were sold -/
theorem strawberry_jelly_sales (s : JellySales) (h : valid_jelly_sales s) : s.strawberry = 18 := by
  sorry


end strawberry_jelly_sales_l1893_189362


namespace num_women_is_sixteen_l1893_189360

-- Define the number of men
def num_men : ℕ := 24

-- Define the daily wage of a man
def man_wage : ℕ := 350

-- Define the total daily wage
def total_wage : ℕ := 11600

-- Define the number of women in the second condition
def women_in_second_condition : ℕ := 37

-- Define the function to calculate the number of women
def calculate_women : ℕ := 16

-- Theorem statement
theorem num_women_is_sixteen :
  ∃ (women_wage : ℕ),
    -- Condition 1: Total wage equation
    num_men * man_wage + calculate_women * women_wage = total_wage ∧
    -- Condition 2: Half men and 37 women earn the same as all men and all women
    (num_men / 2) * man_wage + women_in_second_condition * women_wage = num_men * man_wage + calculate_women * women_wage :=
by
  sorry


end num_women_is_sixteen_l1893_189360


namespace quadratic_equation_roots_ratio_l1893_189379

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = 5 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 - 5*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 6 := by
sorry

end quadratic_equation_roots_ratio_l1893_189379


namespace no_solution_iff_m_geq_two_l1893_189342

theorem no_solution_iff_m_geq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x < m + 1 ∧ x > 2*m - 1)) ↔ m ≥ 2 := by
  sorry

end no_solution_iff_m_geq_two_l1893_189342


namespace existence_of_xy_l1893_189353

theorem existence_of_xy : ∃ x y : ℕ+, 
  (x.val < 30 ∧ y.val < 30) ∧ 
  (x.val + y.val + x.val * y.val = 119) ∧
  (x.val + y.val = 20) := by
  sorry

end existence_of_xy_l1893_189353


namespace factor_w4_minus_81_factors_are_monic_real_polynomials_l1893_189395

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by sorry

theorem factors_are_monic_real_polynomials :
  ∀ w : ℝ, 
    (∃ a b c : ℝ, (w - 3) = w + a ∧ (w + 3) = w + b ∧ (w^2 + 9) = w^2 + c) := by sorry

end factor_w4_minus_81_factors_are_monic_real_polynomials_l1893_189395


namespace negative_of_negative_five_greater_than_negative_five_l1893_189357

theorem negative_of_negative_five_greater_than_negative_five : -(-5) > -5 := by
  sorry

end negative_of_negative_five_greater_than_negative_five_l1893_189357


namespace taco_truck_lunch_rush_earnings_l1893_189336

/-- Calculates the total earnings of a taco truck during lunch rush -/
def taco_truck_earnings (soft_taco_price : ℕ) (hard_taco_price : ℕ) 
  (family_hard_tacos : ℕ) (family_soft_tacos : ℕ) 
  (other_customers : ℕ) (tacos_per_customer : ℕ) : ℕ :=
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price) + 
  (other_customers * tacos_per_customer * soft_taco_price)

/-- The taco truck's earnings during lunch rush is $66 -/
theorem taco_truck_lunch_rush_earnings : 
  taco_truck_earnings 2 5 4 3 10 2 = 66 := by
  sorry

end taco_truck_lunch_rush_earnings_l1893_189336


namespace rayden_extra_birds_l1893_189377

def lily_ducks : ℕ := 20
def lily_geese : ℕ := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def rayden_geese : ℕ := 4 * lily_geese

theorem rayden_extra_birds : 
  (rayden_ducks + rayden_geese) - (lily_ducks + lily_geese) = 70 := by
  sorry

end rayden_extra_birds_l1893_189377


namespace triangle_area_l1893_189363

/-- Given a triangle ABC with angles A, B, C forming an arithmetic sequence,
    side b = √3, and f(x) = 2√3 sin²x + 2sin x cos x - √3 reaching its maximum at x = A,
    prove that the area of triangle ABC is (3 + √3) / 4 -/
theorem triangle_area (A B C : Real) (b : Real) (f : Real → Real) :
  (∃ d : Real, B = A - d ∧ C = A + d) →  -- Angles form arithmetic sequence
  b = Real.sqrt 3 →  -- Side b equals √3
  (∀ x, f x = 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3) →  -- Definition of f
  (∀ x, f x ≤ f A) →  -- f reaches maximum at A
  A + B + C = π →  -- Angle sum in triangle
  (∃ a c : Real, a * Real.sin B = b * Real.sin A ∧ c * Real.sin A = b * Real.sin C) →  -- Sine law
  1 / 2 * b * Real.sin A * Real.sin C / Real.sin B = (3 + Real.sqrt 3) / 4 :=  -- Area formula
by sorry

end triangle_area_l1893_189363


namespace weight_problem_l1893_189340

theorem weight_problem (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 84) →
  ((a + b + c + d) / 4 = 80) →
  (e = d + 3) →
  ((b + c + d + e) / 4 = 79) →
  a = 75 := by
sorry

end weight_problem_l1893_189340


namespace bobby_paycheck_l1893_189300

/-- Calculates the final paycheck amount given the salary and deductions --/
def final_paycheck_amount (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (salary * federal_tax_rate + salary * state_tax_rate + 
    health_insurance + life_insurance + parking_fee)

/-- Theorem stating that Bobby's final paycheck amount is $184 --/
theorem bobby_paycheck :
  final_paycheck_amount 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end bobby_paycheck_l1893_189300


namespace eric_erasers_friends_l1893_189339

theorem eric_erasers_friends (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 9306) (h2 : erasers_per_friend = 94) :
  total_erasers / erasers_per_friend = 99 := by
sorry

end eric_erasers_friends_l1893_189339


namespace smallest_angle_WYZ_l1893_189371

-- Define the angles
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem to prove
theorem smallest_angle_WYZ : 
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 21 := by sorry

end smallest_angle_WYZ_l1893_189371


namespace complex_square_plus_self_l1893_189350

theorem complex_square_plus_self (z : ℂ) (h : z = -1/2 + (Real.sqrt 3)/2 * Complex.I) : z^2 + z = -1 := by
  sorry

end complex_square_plus_self_l1893_189350


namespace locus_of_equidistant_points_l1893_189319

-- Define the oblique coordinate system
structure ObliqueCoordSystem where
  angle : ℝ
  e₁ : ℝ × ℝ
  e₂ : ℝ × ℝ

-- Define a point in the oblique coordinate system
structure ObliquePoint where
  x : ℝ
  y : ℝ

-- Define the locus equation
def locusEquation (p : ObliquePoint) : Prop :=
  Real.sqrt 2 * p.x + p.y = 0

-- State the theorem
theorem locus_of_equidistant_points
  (sys : ObliqueCoordSystem)
  (F₁ F₂ M : ObliquePoint)
  (h_angle : sys.angle = Real.pi / 4)
  (h_F₁ : F₁ = ⟨-1, 0⟩)
  (h_F₂ : F₂ = ⟨1, 0⟩)
  (h_equidistant : ‖(M.x - F₁.x, M.y - F₁.y)‖ = ‖(M.x - F₂.x, M.y - F₂.y)‖) :
  locusEquation M :=
sorry

end locus_of_equidistant_points_l1893_189319


namespace vector_equality_transitivity_l1893_189393

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by sorry

end vector_equality_transitivity_l1893_189393


namespace coprime_iterations_exists_coprime_polynomial_l1893_189332

/-- The polynomial f(x) = x^2007 - x^2006 + 1 -/
def f (x : ℤ) : ℤ := x^2007 - x^2006 + 1

/-- The m-th iteration of f -/
def f_iter (m : ℕ) (x : ℤ) : ℤ :=
  match m with
  | 0 => x
  | m+1 => f (f_iter m x)

theorem coprime_iterations (n : ℤ) (m : ℕ) : Int.gcd n (f_iter m n) = 1 := by
  sorry

/-- The main theorem stating that the polynomial f satisfies the required property -/
theorem exists_coprime_polynomial :
  ∃ (f : ℤ → ℤ), (∀ (x : ℤ), ∃ (a b c : ℤ), f x = a * x^2007 + b * x^2006 + c) ∧
                 (∀ (n : ℤ) (m : ℕ), Int.gcd n (f_iter m n) = 1) := by
  sorry

end coprime_iterations_exists_coprime_polynomial_l1893_189332


namespace circle_configuration_theorem_l1893_189358

/-- Represents a configuration of three circles tangent to each other and a line -/
structure CircleConfiguration where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : 0 < c
  h2 : c < b
  h3 : b < a

/-- The relation between radii of mutually tangent circles according to Descartes' theorem -/
def descartes_relation (config : CircleConfiguration) : Prop :=
  ((1 / config.a + 1 / config.b + 1 / config.c) ^ 2 : ℝ) = 
  2 * ((1 / config.a ^ 2 + 1 / config.b ^ 2 + 1 / config.c ^ 2) : ℝ)

/-- A configuration is nice if all radii are integers -/
def is_nice (config : CircleConfiguration) : Prop :=
  ∃ (i j k : ℕ), (config.a = i) ∧ (config.b = j) ∧ (config.c = k)

theorem circle_configuration_theorem :
  ∀ (config : CircleConfiguration),
  descartes_relation config →
  (∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    (config.a = 16 ∧ config.b = 4 → |config.c - 40| < ε)) ∧
  (∀ (nice_config : CircleConfiguration),
    is_nice nice_config → descartes_relation nice_config → 
    nice_config.c ≥ 2) ∧
  (∃ (nice_config : CircleConfiguration),
    is_nice nice_config ∧ descartes_relation nice_config ∧ nice_config.c = 2) :=
by sorry

end circle_configuration_theorem_l1893_189358


namespace symmetric_points_a_value_l1893_189328

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- Given that point A(a,1) is symmetric to point B(-3,-1) with respect to the origin, prove that a = 3 -/
theorem symmetric_points_a_value :
  ∀ a : ℝ, symmetric_wrt_origin (a, 1) (-3, -1) → a = 3 := by
  sorry

end symmetric_points_a_value_l1893_189328


namespace percent_difference_z_w_l1893_189365

theorem percent_difference_z_w (y x w z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
  sorry

end percent_difference_z_w_l1893_189365


namespace quadratic_max_condition_l1893_189312

/-- Quadratic function f(x) = x^2 + (2-a)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2-a)*x + 5

theorem quadratic_max_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ f a 1) →
  a ≥ 6 := by
  sorry

end quadratic_max_condition_l1893_189312


namespace cylinder_height_l1893_189385

/-- Given a cylinder with the following properties:
  * AB is the diameter of the lower base
  * A₁B₁ is a chord of the upper base, parallel to AB
  * The plane passing through AB and A₁B₁ forms an acute angle α with the lower base
  * The line AB₁ forms an angle β with the lower base
  * R is the radius of the base of the cylinder
  * A and A₁ lie on the same side of the line passing through the midpoints of AB and A₁B₁

  Prove that the height of the cylinder is equal to the given expression. -/
theorem cylinder_height (R α β : ℝ) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) :
  ∃ (height : ℝ), height = 2 * R * Real.tan β * (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) / (Real.sin α * Real.cos β) :=
by sorry

end cylinder_height_l1893_189385


namespace reciprocal_product_theorem_l1893_189307

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 := by
  sorry

end reciprocal_product_theorem_l1893_189307


namespace no_solution_equation_l1893_189383

theorem no_solution_equation :
  ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by sorry

end no_solution_equation_l1893_189383


namespace power_of_product_equals_product_of_powers_l1893_189387

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end power_of_product_equals_product_of_powers_l1893_189387


namespace egg_production_increase_l1893_189302

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 :=
by sorry

end egg_production_increase_l1893_189302


namespace ratio_problem_l1893_189303

theorem ratio_problem (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) : 
  e / a = 8 / 35 := by
  sorry

end ratio_problem_l1893_189303


namespace min_ships_proof_l1893_189374

/-- The number of passengers to accommodate -/
def total_passengers : ℕ := 792

/-- The maximum capacity of each cruise ship -/
def ship_capacity : ℕ := 55

/-- The minimum number of cruise ships required -/
def min_ships : ℕ := (total_passengers + ship_capacity - 1) / ship_capacity

theorem min_ships_proof : min_ships = 15 := by
  sorry

end min_ships_proof_l1893_189374


namespace line_through_circle_center_l1893_189354

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- 
If the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0,
then a = 1
-/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end line_through_circle_center_l1893_189354


namespace terminal_side_quadrant_l1893_189331

/-- The quadrant in which an angle falls -/
inductive Quadrant
| I
| II
| III
| IV

/-- Determines the quadrant of an angle in degrees -/
def angle_quadrant (angle : Int) : Quadrant :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle && normalized_angle < 90 then Quadrant.I
  else if 90 ≤ normalized_angle && normalized_angle < 180 then Quadrant.II
  else if 180 ≤ normalized_angle && normalized_angle < 270 then Quadrant.III
  else Quadrant.IV

theorem terminal_side_quadrant :
  angle_quadrant (-1060) = Quadrant.I :=
sorry

end terminal_side_quadrant_l1893_189331


namespace ratio_of_21_to_reversed_l1893_189310

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem ratio_of_21_to_reversed : 
  let original := 21
  let reversed := reverse_digits original
  (original : ℚ) / reversed = 7 / 4 := by
sorry

end ratio_of_21_to_reversed_l1893_189310


namespace brians_math_quiz_l1893_189376

theorem brians_math_quiz (x : ℝ) : (x - 11) / 5 = 31 → (x - 5) / 11 = 15 := by
  sorry

end brians_math_quiz_l1893_189376


namespace batting_highest_score_l1893_189361

-- Define the given conditions
def total_innings : ℕ := 46
def overall_average : ℚ := 60
def score_difference : ℕ := 180
def average_excluding_extremes : ℚ := 58
def min_half_centuries : ℕ := 15
def min_centuries : ℕ := 10

-- Define the function to calculate the highest score
def highest_score : ℕ := 194

-- Theorem statement
theorem batting_highest_score :
  (total_innings : ℚ) * overall_average = 
    (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + (highest_score - score_difference) ∧
  highest_score ≥ 100 ∧
  min_half_centuries + min_centuries ≤ total_innings - 2 :=
by sorry

end batting_highest_score_l1893_189361


namespace first_complete_shading_l1893_189343

def board_width : ℕ := 10

def shaded_square (n : ℕ) : ℕ := n * n

def is_shaded (square : ℕ) : Prop :=
  ∃ n : ℕ, shaded_square n = square

def column_of_square (square : ℕ) : ℕ :=
  (square - 1) % board_width + 1

theorem first_complete_shading :
  (∀ col : ℕ, col ≤ board_width → 
    ∃ square : ℕ, is_shaded square ∧ column_of_square square = col) ∧
  (∀ smaller : ℕ, smaller < 100 → 
    ¬(∀ col : ℕ, col ≤ board_width → 
      ∃ square : ℕ, square ≤ smaller ∧ is_shaded square ∧ column_of_square square = col)) :=
by sorry

end first_complete_shading_l1893_189343


namespace equation_system_solution_l1893_189341

theorem equation_system_solution (x y : ℝ) :
  (2 * x^2 + 6 * x + 4 * y + 2 = 0) →
  (3 * x + y + 4 = 0) →
  (y^2 + 17 * y - 11 = 0) := by
sorry

end equation_system_solution_l1893_189341


namespace quadratic_factorization_l1893_189338

theorem quadratic_factorization (t : ℝ) : t^2 - 10*t + 25 = (t - 5)^2 := by
  sorry

end quadratic_factorization_l1893_189338


namespace probability_one_common_course_is_two_thirds_l1893_189372

def total_courses : ℕ := 4
def courses_per_person : ℕ := 2

def probability_one_common_course : ℚ :=
  let total_selections := Nat.choose total_courses courses_per_person * Nat.choose total_courses courses_per_person
  let no_common_courses := Nat.choose total_courses courses_per_person
  let all_common_courses := Nat.choose total_courses courses_per_person
  let one_common_course := total_selections - no_common_courses - all_common_courses
  ↑one_common_course / ↑total_selections

theorem probability_one_common_course_is_two_thirds :
  probability_one_common_course = 2 / 3 := by
  sorry

end probability_one_common_course_is_two_thirds_l1893_189372


namespace trailing_zeros_30_factorial_l1893_189347

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

theorem trailing_zeros_30_factorial :
  trailingZeros (factorial 30) = 7 := by
  sorry

end trailing_zeros_30_factorial_l1893_189347


namespace projection_result_l1893_189306

/-- Given two vectors a and b, if both are projected onto the same vector v
    resulting in the same vector p, then p is equal to (15/58, 35/58). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, -1) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (15/58, 35/58) :=
sorry

end projection_result_l1893_189306


namespace distribute_spots_correct_l1893_189313

/-- The number of ways to distribute 8 spots among 6 classes with at least one spot per class -/
def distribute_spots : ℕ := 21

/-- The number of senior classes -/
def num_classes : ℕ := 6

/-- The total number of spots to be distributed -/
def total_spots : ℕ := 8

/-- The minimum number of spots per class -/
def min_spots_per_class : ℕ := 1

theorem distribute_spots_correct :
  distribute_spots = 
    (num_classes.choose 2) + num_classes ∧
  num_classes * min_spots_per_class ≤ total_spots ∧
  total_spots - num_classes * min_spots_per_class = 2 := by
  sorry

end distribute_spots_correct_l1893_189313


namespace scientific_notation_equivalence_l1893_189323

theorem scientific_notation_equivalence :
  686530000 = 6.8653 * (10 ^ 8) := by sorry

end scientific_notation_equivalence_l1893_189323


namespace hyperbola_properties_l1893_189369

/-- Properties of a hyperbola M with equation x²/4 - y²/2 = 1 -/
theorem hyperbola_properties :
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 2 = 1}
  ∃ (a b c : ℝ) (e : ℝ),
    a = 2 ∧
    b = Real.sqrt 2 ∧
    c = Real.sqrt 6 ∧
    e = Real.sqrt 6 / 2 ∧
    (2 * a = 4) ∧  -- Length of real axis
    (2 * b = 2 * Real.sqrt 2) ∧  -- Length of imaginary axis
    (2 * c = 2 * Real.sqrt 6) ∧  -- Focal distance
    (e = Real.sqrt 6 / 2)  -- Eccentricity
  := by sorry

end hyperbola_properties_l1893_189369


namespace abs_difference_of_roots_l1893_189317

theorem abs_difference_of_roots (α β : ℝ) (h1 : α + β = 17) (h2 : α * β = 70) : 
  |α - β| = 3 := by
sorry

end abs_difference_of_roots_l1893_189317


namespace pentagon_star_area_theorem_l1893_189355

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- The star formed by connecting every second vertex of the pentagon -/
def star (p : RegularPentagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set of points in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersect (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The quadrilateral APQD -/
def quadrilateral_APQD (p : RegularPentagon) : Set (ℝ × ℝ) :=
  let A := p.vertices 0
  let B := p.vertices 1
  let C := p.vertices 2
  let D := p.vertices 3
  let E := p.vertices 4
  let P := intersect A C B E
  let Q := intersect B D C E
  sorry

theorem pentagon_star_area_theorem (p : RegularPentagon) 
  (h : area (star p) = 1) : 
  area (quadrilateral_APQD p) = 1/2 := by
  sorry

end pentagon_star_area_theorem_l1893_189355


namespace kelly_found_games_l1893_189394

def initial_games : ℕ := 80
def games_to_give_away : ℕ := 105
def games_left : ℕ := 6

theorem kelly_found_games : 
  ∃ (found_games : ℕ), 
    initial_games + found_games = games_to_give_away + games_left ∧ 
    found_games = 31 :=
by sorry

end kelly_found_games_l1893_189394


namespace vector_problem_l1893_189382

/-- Given two planar vectors a and b, where a is orthogonal to b and their sum with a third vector c is zero, 
    prove that the first component of a is 2 and the magnitude of c is 5. -/
theorem vector_problem (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (2, 4)
  let c : ℝ × ℝ := (-a.1 - b.1, -a.2 - b.2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b
  (m = 2 ∧ Real.sqrt ((c.1 ^ 2) + (c.2 ^ 2)) = 5) := by
sorry

end vector_problem_l1893_189382


namespace arithmetic_sequence_property_l1893_189334

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_property_l1893_189334


namespace meeting_point_distance_l1893_189320

/-- 
Given two people walking towards each other from a distance of 50 km, 
with one person walking at 4 km/h and the other at 6 km/h, 
the distance traveled by the slower person when they meet is 20 km.
-/
theorem meeting_point_distance 
  (total_distance : ℝ) 
  (speed_a : ℝ) 
  (speed_b : ℝ) 
  (h1 : total_distance = 50) 
  (h2 : speed_a = 4) 
  (h3 : speed_b = 6) : 
  (total_distance * speed_a) / (speed_a + speed_b) = 20 := by
sorry

end meeting_point_distance_l1893_189320


namespace officer_selection_with_past_officer_l1893_189396

/- Given conditions -/
def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 10

/- Theorem to prove -/
theorem officer_selection_with_past_officer :
  (Nat.choose total_candidates positions_available) - 
  (Nat.choose (total_candidates - past_officers) positions_available) = 184690 := by
  sorry

end officer_selection_with_past_officer_l1893_189396


namespace similar_triangles_side_length_l1893_189345

-- Define the triangles and their properties
structure Triangle :=
  (X Y Z : ℝ × ℝ)

def XYZ : Triangle := sorry
def PQR : Triangle := sorry

-- Define the lengths of the sides
def XY : ℝ := 9
def YZ : ℝ := 21
def XZ : ℝ := 15
def PQ : ℝ := 3
def QR : ℝ := 7

-- Define the angles
def angle_XYZ : ℝ := sorry
def angle_PQR : ℝ := sorry

-- State the theorem
theorem similar_triangles_side_length :
  angle_XYZ = angle_PQR →
  XY = 9 →
  XZ = 15 →
  PQ = 3 →
  ∃ (PR : ℝ), PR = 5 := by sorry

end similar_triangles_side_length_l1893_189345


namespace women_average_age_l1893_189349

theorem women_average_age 
  (n : Nat) 
  (initial_avg : ℝ) 
  (age_increase : ℝ) 
  (man1_age : ℝ) 
  (man2_age : ℝ) 
  (h1 : n = 7) 
  (h2 : age_increase = 4) 
  (h3 : man1_age = 26) 
  (h4 : man2_age = 30) 
  (h5 : n * (initial_avg + age_increase) = n * initial_avg - man1_age - man2_age + (women_avg * 2)) : 
  women_avg = 42 := by
  sorry

#check women_average_age

end women_average_age_l1893_189349


namespace inequality_proof_l1893_189384

theorem inequality_proof (x y : ℝ) :
  (x + y) / 2 * (x^2 + y^2) / 2 * (x^3 + y^3) / 2 ≤ (x^6 + y^6) / 2 := by
  sorry

end inequality_proof_l1893_189384


namespace sports_club_overlap_l1893_189391

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 28 → badminton = 17 → tennis = 19 → neither = 2 →
  badminton + tennis - total + neither = 10 := by
sorry

end sports_club_overlap_l1893_189391


namespace epipen_insurance_coverage_l1893_189398

/-- Calculates the insurance coverage percentage for EpiPens -/
theorem epipen_insurance_coverage 
  (frequency : ℕ) -- Number of EpiPens per year
  (cost : ℝ) -- Cost of each EpiPen in dollars
  (annual_payment : ℝ) -- John's annual payment in dollars
  (h1 : frequency = 2) -- John gets 2 EpiPens per year
  (h2 : cost = 500) -- Each EpiPen costs $500
  (h3 : annual_payment = 250) -- John pays $250 per year
  : (1 - annual_payment / (frequency * cost)) * 100 = 75 := by
  sorry


end epipen_insurance_coverage_l1893_189398


namespace square_area_with_circles_l1893_189380

/-- The area of a square containing six circles arranged in two rows and three columns, 
    where each circle has a radius of 3 units. -/
theorem square_area_with_circles (radius : ℝ) (h : radius = 3) : 
  (3 * (2 * radius))^2 = 324 := by
  sorry

end square_area_with_circles_l1893_189380


namespace equation_solutions_l1893_189327

theorem equation_solutions :
  (∀ x, x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) ∧
  (∀ x, 3*x*(x - 1) = 2 - 2*x ↔ x = 1 ∨ x = -2/3) := by
sorry

end equation_solutions_l1893_189327


namespace alice_wrong_questions_l1893_189346

/-- Represents the number of questions a person got wrong in the test. -/
structure TestResult where
  wrong : ℕ

/-- Represents the test results for Alice, Beth, Charlie, Daniel, and Ellen. -/
structure TestResults where
  alice : TestResult
  beth : TestResult
  charlie : TestResult
  daniel : TestResult
  ellen : TestResult

/-- The theorem stating that Alice got 9 questions wrong given the conditions. -/
theorem alice_wrong_questions (results : TestResults) : results.alice.wrong = 9 :=
  by
  have h1 : results.alice.wrong + results.beth.wrong = results.charlie.wrong + results.daniel.wrong + results.ellen.wrong :=
    sorry
  have h2 : results.alice.wrong + results.daniel.wrong = results.beth.wrong + results.charlie.wrong + 3 :=
    sorry
  have h3 : results.charlie.wrong = 6 :=
    sorry
  have h4 : results.daniel.wrong = 8 :=
    sorry
  sorry

end alice_wrong_questions_l1893_189346


namespace two_numbers_problem_l1893_189315

theorem two_numbers_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 := by
  sorry

end two_numbers_problem_l1893_189315


namespace exponent_of_five_in_forty_factorial_l1893_189397

theorem exponent_of_five_in_forty_factorial :
  ∃ k : ℕ, (40 : ℕ).factorial = 5^10 * k ∧ ¬(5 ∣ k) := by
  sorry

end exponent_of_five_in_forty_factorial_l1893_189397


namespace sum_and_count_equals_1271_l1893_189367

/-- The sum of integers from 50 to 70, inclusive -/
def x : ℕ := (List.range 21).map (· + 50) |>.sum

/-- The number of even integers from 50 to 70, inclusive -/
def y : ℕ := (List.range 21).map (· + 50) |>.filter (· % 2 = 0) |>.length

/-- The theorem stating that x + y equals 1271 -/
theorem sum_and_count_equals_1271 : x + y = 1271 := by
  sorry

end sum_and_count_equals_1271_l1893_189367


namespace log_319_approximation_l1893_189321

-- Define the logarithm values for 0.317 and 0.318
def log_317 : ℝ := 0.33320
def log_318 : ℝ := 0.3364

-- Define the approximation function for log 0.319
def approx_log_319 : ℝ := log_318 + (log_318 - log_317)

-- Theorem statement
theorem log_319_approximation : 
  abs (approx_log_319 - 0.3396) < 0.0001 := by
  sorry

end log_319_approximation_l1893_189321


namespace probability_of_no_three_consecutive_ones_l1893_189386

/-- Represents the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element sequence not containing three consecutive 1s -/
def probability : ℚ := b 12 / 2^12

theorem probability_of_no_three_consecutive_ones : probability = 281 / 1024 := by
  sorry

end probability_of_no_three_consecutive_ones_l1893_189386


namespace count_multiples_eq_16_l1893_189324

/-- The number of positive multiples of 3 less than 150 with units digit 3 or 9 -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 10 = 3 ∨ n % 10 = 9)
    (Finset.filter (fun n => n % 3 = 0) (Finset.range 150))).card

theorem count_multiples_eq_16 : count_multiples = 16 := by
  sorry

end count_multiples_eq_16_l1893_189324


namespace jane_mean_score_l1893_189352

def jane_scores : List ℝ := [98, 97, 92, 85, 93, 88, 82]

theorem jane_mean_score : 
  (jane_scores.sum / jane_scores.length : ℝ) = 90.71428571428571 := by
sorry

end jane_mean_score_l1893_189352
