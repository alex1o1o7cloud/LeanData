import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l1954_195494

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = (3 + Real.sqrt 89) / 40 ∧ 
     x₂ = (3 - Real.sqrt 89) / 40) ∧ 
    (∀ x y : ℝ, y = 3 * x → 
      (4 * y^2 + y + 5 = 2 * (8 * x^2 + y + 3) ↔ 
       (x = x₁ ∨ x = x₂))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1954_195494


namespace NUMINAMATH_CALUDE_average_monthly_balance_l1954_195459

def monthly_balances : List ℕ := [100, 200, 150, 150, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 156 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l1954_195459


namespace NUMINAMATH_CALUDE_X_equals_Y_l1954_195426

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem X_equals_Y : X = Y := by sorry

end NUMINAMATH_CALUDE_X_equals_Y_l1954_195426


namespace NUMINAMATH_CALUDE_steves_commute_l1954_195489

/-- The distance from Steve's house to work -/
def distance : ℝ := sorry

/-- Steve's speed on the way to work -/
def speed_to_work : ℝ := sorry

/-- Steve's speed on the way back from work -/
def speed_from_work : ℝ := 10

/-- Total time Steve spends on the road daily -/
def total_time : ℝ := 6

theorem steves_commute :
  (speed_from_work = 2 * speed_to_work) →
  (distance / speed_to_work + distance / speed_from_work = total_time) →
  distance = 20 := by sorry

end NUMINAMATH_CALUDE_steves_commute_l1954_195489


namespace NUMINAMATH_CALUDE_sarahs_brother_apples_l1954_195456

theorem sarahs_brother_apples (sarah_apples : ℕ) (ratio : ℕ) (brother_apples : ℕ) : 
  sarah_apples = 45 → 
  ratio = 5 → 
  sarah_apples = ratio * brother_apples → 
  brother_apples = 9 := by
sorry

end NUMINAMATH_CALUDE_sarahs_brother_apples_l1954_195456


namespace NUMINAMATH_CALUDE_brick_surface_area_l1954_195475

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1954_195475


namespace NUMINAMATH_CALUDE_fraction_equality_l1954_195415

theorem fraction_equality (a : ℕ+) : (a : ℚ) / (a + 35 : ℚ) = 875 / 1000 → a = 245 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1954_195415


namespace NUMINAMATH_CALUDE_eddy_spider_plant_production_l1954_195458

/-- A spider plant that produces baby plants -/
structure SpiderPlant where
  /-- Number of baby plants produced each time -/
  baby_per_time : ℕ
  /-- Total number of baby plants produced -/
  total_babies : ℕ
  /-- Number of years -/
  years : ℕ

/-- The number of times per year a spider plant produces baby plants -/
def times_per_year (plant : SpiderPlant) : ℚ :=
  (plant.total_babies : ℚ) / (plant.years * plant.baby_per_time : ℚ)

/-- Theorem stating that Eddy's spider plant produces baby plants 2 times per year -/
theorem eddy_spider_plant_production :
  ∃ (plant : SpiderPlant),
    plant.baby_per_time = 2 ∧
    plant.total_babies = 16 ∧
    plant.years = 4 ∧
    times_per_year plant = 2 := by
  sorry

end NUMINAMATH_CALUDE_eddy_spider_plant_production_l1954_195458


namespace NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l1954_195405

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ f x) ↔ (∀ x : ℝ, x > 0 → ¬ f x) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_cubic_inequality_negation_l1954_195405


namespace NUMINAMATH_CALUDE_neighbor_purchase_theorem_l1954_195403

/-- Proves that given the conditions of the problem, the total amount spent is 168 shillings -/
theorem neighbor_purchase_theorem (x : ℝ) 
  (h1 : x > 0)  -- Quantity purchased is positive
  (h2 : 2*x + 1.5*x = 3.5*x)  -- Total cost equation
  (h3 : (3.5*x/2)/2 + (3.5*x/2)/1.5 = 2*x + 2)  -- Equal division condition
  : 3.5*x = 168 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_purchase_theorem_l1954_195403


namespace NUMINAMATH_CALUDE_binomial_100_3_l1954_195484

theorem binomial_100_3 : Nat.choose 100 3 = 161700 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_3_l1954_195484


namespace NUMINAMATH_CALUDE_wedge_volume_approximation_l1954_195419

/-- The volume of a wedge cut from a cylinder --/
theorem wedge_volume_approximation (r h : ℝ) (h_r : r = 6) (h_h : h = 6) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := cylinder_volume / 2
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |wedge_volume - 339.12| < ε :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_approximation_l1954_195419


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l1954_195474

/-- Represents the money redistribution game with three players -/
structure MoneyGame where
  amy_initial : ℝ
  bob_initial : ℝ
  cal_initial : ℝ
  cal_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  ∃ (amy_mid bob_mid cal_mid : ℝ),
    -- Amy's redistribution
    amy_mid + bob_mid + cal_mid = game.amy_initial + game.bob_initial + game.cal_initial ∧
    bob_mid = 2 * game.bob_initial ∧
    cal_mid = 2 * game.cal_initial ∧
    -- Bob's redistribution
    ∃ (amy_mid2 bob_mid2 cal_mid2 : ℝ),
      amy_mid2 + bob_mid2 + cal_mid2 = amy_mid + bob_mid + cal_mid ∧
      amy_mid2 = 2 * amy_mid ∧
      cal_mid2 = 2 * cal_mid ∧
      -- Cal's redistribution
      ∃ (amy_final bob_final : ℝ),
        amy_final + bob_final + game.cal_final = amy_mid2 + bob_mid2 + cal_mid2 ∧
        amy_final = 2 * amy_mid2 ∧
        bob_final = 2 * bob_mid2

/-- The theorem stating the total money after redistribution -/
theorem total_money_after_redistribution (game : MoneyGame)
    (h1 : game.cal_initial = 50)
    (h2 : game.cal_final = 100)
    (h3 : redistribute game) :
    game.amy_initial + game.bob_initial + game.cal_initial = 300 :=
  sorry

end NUMINAMATH_CALUDE_total_money_after_redistribution_l1954_195474


namespace NUMINAMATH_CALUDE_square_area_is_400_l1954_195482

-- Define the radius of the circles
def circle_radius : ℝ := 5

-- Define the side length of the square
def square_side_length : ℝ := 2 * (2 * circle_radius)

-- Theorem: The area of the square is 400 square inches
theorem square_area_is_400 : square_side_length ^ 2 = 400 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_400_l1954_195482


namespace NUMINAMATH_CALUDE_permutation_problem_l1954_195462

-- Define permutation function
def permutation (n : ℕ) (r : ℕ) : ℕ :=
  if r ≤ n then (n - r + 1).factorial / (n - r).factorial else 0

-- Theorem statement
theorem permutation_problem : 
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * Nat.factorial 0 = 4 :=
by sorry

end NUMINAMATH_CALUDE_permutation_problem_l1954_195462


namespace NUMINAMATH_CALUDE_complex_power_four_l1954_195483

theorem complex_power_four : (1 + 2 * Complex.I) ^ 4 = -7 - 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l1954_195483


namespace NUMINAMATH_CALUDE_annual_increase_rate_proof_l1954_195457

/-- Proves that given an initial value of 32000 and a value of 40500 after two years,
    the annual increase rate is 0.125. -/
theorem annual_increase_rate_proof (initial_value final_value : ℝ) 
  (h1 : initial_value = 32000)
  (h2 : final_value = 40500)
  (h3 : final_value = initial_value * (1 + 0.125)^2) : 
  ∃ (r : ℝ), r = 0.125 ∧ final_value = initial_value * (1 + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_annual_increase_rate_proof_l1954_195457


namespace NUMINAMATH_CALUDE_arrival_time_difference_l1954_195488

-- Define the constants
def distance : ℝ := 2
def jenna_speed : ℝ := 12
def jamie_speed : ℝ := 6

-- Define the theorem
theorem arrival_time_difference : 
  (distance / jenna_speed * 60 - distance / jamie_speed * 60) = 10 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l1954_195488


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l1954_195480

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - 2^x) / (2^(x+1) + a)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ t, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - 1) < 0 ↔ t > 1 ∨ t < -1/3) :=
sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l1954_195480


namespace NUMINAMATH_CALUDE_derivative_at_two_l1954_195438

theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 3 * (deriv f 2) * x) : 
  deriv f 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l1954_195438


namespace NUMINAMATH_CALUDE_min_sum_y_intersections_l1954_195471

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through a given point with a slope -/
structure Line where
  point : Point
  slope : ℝ

/-- Represents a parabola of the form x^2 = 2y -/
def Parabola : Type := Unit

/-- Returns the y-coordinate of a point on the given line -/
def lineY (l : Line) (x : ℝ) : ℝ :=
  l.point.y + l.slope * (x - l.point.x)

/-- Returns true if the given point lies on the parabola -/
def onParabola (p : Point) : Prop :=
  p.x^2 = 2 * p.y

/-- Returns true if the given point lies on the given line -/
def onLine (l : Line) (p : Point) : Prop :=
  p.y = lineY l p.x

/-- Theorem stating that the minimum sum of y-coordinates of intersection points is 2 -/
theorem min_sum_y_intersections (p : Parabola) :
  ∀ l : Line,
    l.point = Point.mk 0 1 →
    ∃ A B : Point,
      onParabola A ∧ onLine l A ∧
      onParabola B ∧ onLine l B ∧
      A ≠ B →
      (∀ C D : Point,
        onParabola C ∧ onLine l C ∧
        onParabola D ∧ onLine l D ∧
        C ≠ D →
        A.y + B.y ≤ C.y + D.y) →
      A.y + B.y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_y_intersections_l1954_195471


namespace NUMINAMATH_CALUDE_sam_lee_rates_sum_of_squares_l1954_195434

theorem sam_lee_rates_sum_of_squares : 
  ∃ (r c k : ℕ+), 
    (4 * r + 5 * c + 3 * k = 120) ∧ 
    (5 * r + 3 * c + 4 * k = 138) ∧ 
    (r ^ 2 + c ^ 2 + k ^ 2 = 436) := by
  sorry

end NUMINAMATH_CALUDE_sam_lee_rates_sum_of_squares_l1954_195434


namespace NUMINAMATH_CALUDE_sherlock_lock_combination_l1954_195497

def is_valid_solution (d : ℕ) (S E N D R : ℕ) : Prop :=
  S < d ∧ E < d ∧ N < d ∧ D < d ∧ R < d ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ R ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ R ∧
  N ≠ D ∧ N ≠ R ∧
  D ≠ R ∧
  (S * d^3 + E * d^2 + N * d + D) +
  (E * d^2 + N * d + D) +
  (R * d^2 + E * d + D) =
  (D * d^3 + E * d^2 + E * d + R)

theorem sherlock_lock_combination :
  ∃ (d : ℕ), ∃ (S E N D R : ℕ),
    is_valid_solution d S E N D R ∧
    R * d^2 + E * d + D = 879 :=
sorry

end NUMINAMATH_CALUDE_sherlock_lock_combination_l1954_195497


namespace NUMINAMATH_CALUDE_bolt_width_calculation_l1954_195412

/-- The width of a bolt of fabric given specific cuts and remaining area --/
theorem bolt_width_calculation (living_room_length living_room_width bedroom_length bedroom_width bolt_length remaining_fabric : ℝ) 
  (h1 : living_room_length = 4)
  (h2 : living_room_width = 6)
  (h3 : bedroom_length = 2)
  (h4 : bedroom_width = 4)
  (h5 : bolt_length = 12)
  (h6 : remaining_fabric = 160) :
  (remaining_fabric + living_room_length * living_room_width + bedroom_length * bedroom_width) / bolt_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_bolt_width_calculation_l1954_195412


namespace NUMINAMATH_CALUDE_account_balance_difference_l1954_195450

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between Angela's and Bob's account balances after 15 years -/
theorem account_balance_difference : 
  let angela_balance := compound_interest 9000 0.05 15
  let bob_balance := simple_interest 11000 0.06 15
  ⌊|bob_balance - angela_balance|⌋ = 2189 := by
sorry

end NUMINAMATH_CALUDE_account_balance_difference_l1954_195450


namespace NUMINAMATH_CALUDE_sum_a_d_l1954_195407

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 5) 
  (h2 : b + c = 6) 
  (h3 : c + d = 3) : 
  a + d = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l1954_195407


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1954_195406

/-- The equation of the tangent line to y = xe^(2x-1) at (1, e) is 3ex - y - 2e = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (2 * x - 1)) → -- Given curve equation
  (3 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0) ↔ -- Tangent line equation
  (y - Real.exp 1 = 3 * Real.exp 1 * (x - 1)) -- Point-slope form at (1, e)
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1954_195406


namespace NUMINAMATH_CALUDE_press_conference_arrangement_l1954_195449

/-- Number of reporters in each station -/
def n : ℕ := 5

/-- Total number of reporters to be selected -/
def k : ℕ := 4

/-- Number of ways to arrange questioning when selecting 1 from A and 3 from B -/
def case1 : ℕ := Nat.choose n 1 * Nat.choose n 3 * Nat.choose k 1 * (Nat.factorial 3)

/-- Number of ways to arrange questioning when selecting 2 from A and 2 from B -/
def case2 : ℕ := Nat.choose n 2 * Nat.choose n 2 * (2 * (Nat.factorial 2) * (Nat.factorial 2) + (Nat.factorial 2) * (Nat.factorial 2))

/-- Total number of ways to arrange the questioning -/
def total_ways : ℕ := case1 + case2

theorem press_conference_arrangement :
  total_ways = 2400 := by sorry

end NUMINAMATH_CALUDE_press_conference_arrangement_l1954_195449


namespace NUMINAMATH_CALUDE_parallelogram_sticks_l1954_195421

/-- A parallelogram formed by four sticks -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  is_parallelogram : side1 = side3 ∧ side2 = side4

/-- The theorem stating that if four sticks of lengths 5, 5, 7, and a can form a parallelogram, then a = 7 -/
theorem parallelogram_sticks (a : ℝ) :
  (∃ p : Parallelogram, p.side1 = 5 ∧ p.side2 = 5 ∧ p.side3 = 7 ∧ p.side4 = a) →
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sticks_l1954_195421


namespace NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l1954_195447

/-- The daily diet of a polar bear at Richmond's zoo -/
structure PolarBearDiet where
  trout : ℝ
  salmon : ℝ
  total_fish : ℝ

/-- Properties of the polar bear's diet -/
def is_valid_diet (d : PolarBearDiet) : Prop :=
  d.trout = 0.2 ∧ d.total_fish = 0.6 ∧ d.total_fish = d.trout + d.salmon

theorem polar_bear_salmon_consumption (d : PolarBearDiet) 
  (h : is_valid_diet d) : d.salmon = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l1954_195447


namespace NUMINAMATH_CALUDE_tan_sum_diff_implies_sin_2alpha_cos_2beta_l1954_195414

theorem tan_sum_diff_implies_sin_2alpha_cos_2beta
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 1)
  (h2 : Real.tan (α - β) = 2) :
  (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_diff_implies_sin_2alpha_cos_2beta_l1954_195414


namespace NUMINAMATH_CALUDE_tank_filling_time_l1954_195432

theorem tank_filling_time (fill_rate_1 fill_rate_2 remaining_time : ℝ) 
  (h1 : fill_rate_1 = 1 / 20)
  (h2 : fill_rate_2 = 1 / 60)
  (h3 : remaining_time = 20.000000000000004)
  : ∃ t : ℝ, t * (fill_rate_1 + fill_rate_2) + remaining_time * fill_rate_2 = 1 ∧ t = 10 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1954_195432


namespace NUMINAMATH_CALUDE_right_triangle_area_l1954_195446

/-- Given a right triangle ABC with legs a and b, and hypotenuse c,
    if a + b = 21 and c = 15, then the area of triangle ABC is 54. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b = 21 → 
  c = 15 → 
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1954_195446


namespace NUMINAMATH_CALUDE_average_apples_sold_example_l1954_195416

/-- Calculates the average number of kg of apples sold per hour given the sales in two hours -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / 2

theorem average_apples_sold_example : average_apples_sold 10 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_sold_example_l1954_195416


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1954_195452

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1954_195452


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1954_195444

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (eggs_in_deficient : ℕ) : 
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  eggs_in_deficient = 13 →
  container_capacity * total_containers - 
    deficient_containers * (container_capacity - eggs_in_deficient) = 159 ∧
  159 > 150 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1954_195444


namespace NUMINAMATH_CALUDE_least_absolute_prime_l1954_195431

theorem least_absolute_prime (n : ℤ) : 
  Nat.Prime n.natAbs → 101 * n^2 ≤ 3600 → (∀ m : ℤ, Nat.Prime m.natAbs → 101 * m^2 ≤ 3600 → n.natAbs ≤ m.natAbs) → n.natAbs = 2 :=
by sorry

end NUMINAMATH_CALUDE_least_absolute_prime_l1954_195431


namespace NUMINAMATH_CALUDE_complex_modulus_l1954_195423

theorem complex_modulus (a b : ℝ) (h : b^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1954_195423


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l1954_195401

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l1954_195401


namespace NUMINAMATH_CALUDE_hash_four_negative_three_l1954_195408

-- Define the # operation
def hash (x y : Int) : Int := x * (y - 1) + x * y

-- Theorem statement
theorem hash_four_negative_three : hash 4 (-3) = -28 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_negative_three_l1954_195408


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l1954_195495

def monomial : ℤ × ℤ × ℤ → ℤ
| (a, m, n) => a * m * n^3

theorem coefficient_of_monomial :
  ∃ (m n : ℤ), monomial (-5, m, n) = -5 * m * n^3 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l1954_195495


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l1954_195461

/-- The set of possible slopes for a line with y-intercept (0, -3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def PossibleSlopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def LineEquation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def EllipseEquation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ PossibleSlopes ↔
  ∃ x : ℝ, EllipseEquation x (LineEquation m x) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l1954_195461


namespace NUMINAMATH_CALUDE_system_solution_l1954_195499

theorem system_solution : 
  ∃ (x y : ℝ), (6 / (x^2 + y^2) + x^2 * y^2 = 10) ∧ 
               (x^4 + y^4 + 7 * x^2 * y^2 = 81) ∧
               ((x = Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
                (x = -Real.sqrt 3 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1954_195499


namespace NUMINAMATH_CALUDE_green_shells_count_l1954_195490

/-- Proves that the number of green shells is 49 given the total number of shells,
    the number of red shells, and the number of shells that are not red or green. -/
theorem green_shells_count
  (total_shells : ℕ)
  (red_shells : ℕ)
  (not_red_or_green_shells : ℕ)
  (h1 : total_shells = 291)
  (h2 : red_shells = 76)
  (h3 : not_red_or_green_shells = 166) :
  total_shells - not_red_or_green_shells - red_shells = 49 :=
by sorry

end NUMINAMATH_CALUDE_green_shells_count_l1954_195490


namespace NUMINAMATH_CALUDE_complexity_bound_power_of_two_no_complexity_less_than_n_l1954_195417

/-- Complexity of an integer is the number of factors in its prime factorization -/
def complexity (n : ℕ) : ℕ := sorry

/-- For n = 2^k, all numbers between n and 2n have complexity no greater than n -/
theorem complexity_bound_power_of_two (k : ℕ) :
  ∀ m, 2^k ≤ m → m < 2^(k+1) → complexity m ≤ k := by sorry

/-- There exists no n such that all numbers between n and 2n have complexity less than n -/
theorem no_complexity_less_than_n :
  ¬ ∃ n : ℕ, ∀ m, n < m → m ≤ 2*n → complexity m < complexity n := by sorry

end NUMINAMATH_CALUDE_complexity_bound_power_of_two_no_complexity_less_than_n_l1954_195417


namespace NUMINAMATH_CALUDE_cupboard_cost_price_proof_l1954_195404

/-- The cost price of a cupboard satisfying given conditions -/
def cupboard_cost_price : ℝ := 6250

/-- The selling price of the cupboard -/
def selling_price (cost : ℝ) : ℝ := cost * (1 - 0.12)

/-- The selling price that would result in a 12% profit -/
def profit_selling_price (cost : ℝ) : ℝ := cost * (1 + 0.12)

theorem cupboard_cost_price_proof :
  selling_price cupboard_cost_price + 1500 = profit_selling_price cupboard_cost_price :=
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_proof_l1954_195404


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l1954_195402

/-- The probability of exactly k successes in n trials of a Bernoulli experiment -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of exactly 7 tails in 10 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomialProbability 10 7 (2/3) = 512/6561 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l1954_195402


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l1954_195453

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) : 
  x + y ≤ 31 / 11 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l1954_195453


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1954_195472

theorem right_triangle_segment_ratio (x y z u v : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ v > 0 →
  x^2 + y^2 = z^2 →
  x / y = 2 / 5 →
  u * z = x^2 →
  v * z = y^2 →
  u + v = z →
  u / v = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l1954_195472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1954_195493

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (a + (a + 29 * d)) = 600) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * ((a + 30 * d) + (a + 59 * d)) = 2100) -- Sum of next 30 terms
  : a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1954_195493


namespace NUMINAMATH_CALUDE_larger_number_problem_l1954_195486

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : 
  max x y = 25 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1954_195486


namespace NUMINAMATH_CALUDE_min_sum_squares_l1954_195469

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1954_195469


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1954_195442

theorem arithmetic_expression_evaluation : 2 + (4 * 3 - 2) / 2 * 3 + 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l1954_195442


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1954_195422

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2023) ↔ x ≥ 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1954_195422


namespace NUMINAMATH_CALUDE_proof_by_contradiction_step_l1954_195460

theorem proof_by_contradiction_step (a : ℝ) (h : a > 1) :
  (∀ P : Prop, (¬P → False) → P) →
  (¬(a^2 > 1) ↔ a^2 ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_step_l1954_195460


namespace NUMINAMATH_CALUDE_second_product_of_98_l1954_195448

def second_digit_product (n : ℕ) : ℕ :=
  let first_product := (n / 10) * (n % 10)
  (first_product / 10) * (first_product % 10)

theorem second_product_of_98 :
  second_digit_product 98 = 14 := by
  sorry

end NUMINAMATH_CALUDE_second_product_of_98_l1954_195448


namespace NUMINAMATH_CALUDE_unsold_bars_unsold_bars_correct_l1954_195481

/-- Proves the number of unsold chocolate bars given the total number of bars,
    the cost per bar, and the total money made from sold bars. -/
theorem unsold_bars (total_bars : ℕ) (cost_per_bar : ℕ) (money_made : ℕ) : ℕ :=
  total_bars - (money_made / cost_per_bar)

#check unsold_bars 7 3 9 = 4

theorem unsold_bars_correct :
  unsold_bars 7 3 9 = 4 := by sorry

end NUMINAMATH_CALUDE_unsold_bars_unsold_bars_correct_l1954_195481


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l1954_195467

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 5 * x + 18 = 3 * x + 55

-- Define the solutions of the quadratic equation
noncomputable def solution1 : ℝ := 2 + (3 * Real.sqrt 10) / 2
noncomputable def solution2 : ℝ := 2 - (3 * Real.sqrt 10) / 2

-- Theorem statement
theorem quadratic_solution_difference :
  quadratic_equation solution1 ∧ 
  quadratic_equation solution2 ∧ 
  |solution1 - solution2| = 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l1954_195467


namespace NUMINAMATH_CALUDE_log_product_sum_logs_l1954_195428

theorem log_product_sum_logs (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (Real.log a / Real.log 10)^2 - 4 * (Real.log a / Real.log 10) + 1 = 0) →
  (2 * (Real.log b / Real.log 10)^2 - 4 * (Real.log b / Real.log 10) + 1 = 0) →
  (Real.log (a * b) / Real.log 10) * ((Real.log b / Real.log a) + (Real.log a / Real.log b)) = 12 := by
  sorry


end NUMINAMATH_CALUDE_log_product_sum_logs_l1954_195428


namespace NUMINAMATH_CALUDE_equation_solutions_l1954_195470

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁ = 1 ∧ x₂ = 4) ∧ 
    (∀ x : ℝ, (x - 1)^2 = 3*(x - 1) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ = 2 + Real.sqrt 3 ∧ y₂ = 2 - Real.sqrt 3) ∧ 
    (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1954_195470


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l1954_195473

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 5/3) : 
  Real.cos (A - B) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l1954_195473


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1954_195400

theorem cubic_root_sum (p q s : ℝ) : 
  10 * p^3 - 25 * p^2 + 8 * p - 1 = 0 →
  10 * q^3 - 25 * q^2 + 8 * q - 1 = 0 →
  10 * s^3 - 25 * s^2 + 8 * s - 1 = 0 →
  0 < p → p < 1 →
  0 < q → q < 1 →
  0 < s → s < 1 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - s) = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1954_195400


namespace NUMINAMATH_CALUDE_evaluate_expression_l1954_195420

theorem evaluate_expression : (2^2020 + 2^2016) / (2^2020 - 2^2016) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1954_195420


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1954_195455

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_fifth : a 5 = 16)
  (h_eleventh : a 11 = 4) :
  a 7 = 4 * Real.rpow 4 (1/3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1954_195455


namespace NUMINAMATH_CALUDE_parallelogram_vertex_C_l1954_195410

/-- Represents a parallelogram in the complex plane -/
structure ComplexParallelogram where
  O : ℂ
  A : ℂ
  B : ℂ
  C : ℂ
  is_origin : O = 0
  is_parallelogram : C - O = B - A

/-- The complex number corresponding to vertex C in the given parallelogram -/
def vertex_C (p : ComplexParallelogram) : ℂ := p.B + p.A

/-- Theorem stating that for the given parallelogram, vertex C corresponds to 3+5i -/
theorem parallelogram_vertex_C :
  ∀ (p : ComplexParallelogram),
    p.O = 0 ∧ p.A = 1 - 3*I ∧ p.B = 4 + 2*I →
    vertex_C p = 3 + 5*I := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_C_l1954_195410


namespace NUMINAMATH_CALUDE_triangle_sine_value_l1954_195496

theorem triangle_sine_value (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  (0 < A) ∧ (A < π) ∧
  (0 < B) ∧ (B < π) ∧
  (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  -- Given conditions
  (C = π / 6) ∧
  (a = 3) ∧
  (c = 4) →
  Real.sin A = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_value_l1954_195496


namespace NUMINAMATH_CALUDE_pizza_size_increase_l1954_195498

theorem pizza_size_increase (r : ℝ) (hr : r > 0) : 
  let medium_area := π * r^2
  let large_radius := 1.1 * r
  let large_area := π * large_radius^2
  (large_area - medium_area) / medium_area = 0.21
  := by sorry

end NUMINAMATH_CALUDE_pizza_size_increase_l1954_195498


namespace NUMINAMATH_CALUDE_work_completion_time_l1954_195479

theorem work_completion_time 
  (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1/18)         -- b alone finishes in 18 days
  (h3 : c = 1/6)          -- c alone finishes in 6 days
  : a = 1/36 :=           -- a alone finishes in 36 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1954_195479


namespace NUMINAMATH_CALUDE_f_g_deriv_neg_l1954_195424

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_neg (x : ℝ) (h : x < 0) : deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_deriv_neg_l1954_195424


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l1954_195433

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l1954_195433


namespace NUMINAMATH_CALUDE_intersection_M_N_l1954_195477

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}

-- Define set N
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1954_195477


namespace NUMINAMATH_CALUDE_smallest_positive_integer_properties_l1954_195436

theorem smallest_positive_integer_properties : ∃ a : ℕ, 
  (∀ n : ℕ, n > 0 → a ≤ n) ∧ 
  (a^3 + 1 = 2) ∧ 
  ((a + 1) * (a^2 - a + 1) = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_properties_l1954_195436


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l1954_195464

def P : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_complement_theorem : P ∩ (Set.univ \ N) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l1954_195464


namespace NUMINAMATH_CALUDE_probability_walking_200_or_less_l1954_195411

/-- Number of gates in the airport --/
def num_gates : ℕ := 20

/-- Distance between adjacent gates in feet --/
def gate_distance : ℕ := 50

/-- Maximum walking distance in feet --/
def max_distance : ℕ := 200

/-- Calculate the number of favorable outcomes --/
def favorable_outcomes : ℕ := sorry

/-- Calculate the total number of possible outcomes --/
def total_outcomes : ℕ := num_gates * (num_gates - 1)

/-- The probability of walking 200 feet or less --/
theorem probability_walking_200_or_less :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 19 := by sorry

end NUMINAMATH_CALUDE_probability_walking_200_or_less_l1954_195411


namespace NUMINAMATH_CALUDE_correlation_coefficient_properties_l1954_195413

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define positive correlation
def positively_correlated (x y : ℝ → ℝ) : Prop := 
  ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂

-- Define the strength of linear correlation
def linear_correlation_strength (x y : ℝ → ℝ) : ℝ := sorry

-- Define perfect linear relationship
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ t, y t = a * x t + b

-- Theorem statement
theorem correlation_coefficient_properties 
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positively_correlated x y) ∧
  (∀ ε > 0, |r| > 1 - ε → linear_correlation_strength x y > 1 - ε) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_properties_l1954_195413


namespace NUMINAMATH_CALUDE_triangle_inequalities_l1954_195485

/-- For any triangle ABC with exradii r_a, r_b, r_c, inradius r, and circumradius R -/
theorem triangle_inequalities (r_a r_b r_c r R : ℝ) (h_positive : r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r > 0 ∧ R > 0) :
  r_a^2 + r_b^2 + r_c^2 ≥ 27 * r^2 ∧ 4 * R < r_a + r_b + r_c ∧ r_a + r_b + r_c ≤ 9/2 * R := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l1954_195485


namespace NUMINAMATH_CALUDE_smallest_n_for_grape_contest_l1954_195425

theorem smallest_n_for_grape_contest : ∃ (c : ℕ+), 
  (c : ℕ) * (89 - c + 1) = 2009 ∧ 
  89 ≥ 2 * (c - 1) ∧
  ∀ (n : ℕ), n < 89 → ¬(∃ (d : ℕ+), (d : ℕ) * (n - d + 1) = 2009 ∧ n ≥ 2 * (d - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_grape_contest_l1954_195425


namespace NUMINAMATH_CALUDE_frequency_distribution_forms_l1954_195454

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable

/-- Represents a frequency distribution histogram -/
structure FrequencyDistributionHistogram

/-- Represents a set of data -/
structure DataSet

/-- A frequency distribution form for a set of data -/
class FrequencyDistributionForm (α : Type) where
  represents : α → DataSet → Prop

/-- Accuracy property for frequency distribution forms -/
class Accurate (α : Type) where
  is_accurate : α → Prop

/-- Intuitiveness property for frequency distribution forms -/
class Intuitive (α : Type) where
  is_intuitive : α → Prop

instance : FrequencyDistributionForm FrequencyDistributionTable where
  represents := sorry

instance : FrequencyDistributionForm FrequencyDistributionHistogram where
  represents := sorry

instance : Accurate FrequencyDistributionTable where
  is_accurate := sorry

instance : Intuitive FrequencyDistributionHistogram where
  is_intuitive := sorry

/-- Theorem stating that frequency distribution tables and histograms are two forms of frequency distribution for a set of data, with tables being accurate and histograms being intuitive -/
theorem frequency_distribution_forms :
  (∃ (t : FrequencyDistributionTable) (h : FrequencyDistributionHistogram) (d : DataSet),
    FrequencyDistributionForm.represents t d ∧
    FrequencyDistributionForm.represents h d) ∧
  (∀ (t : FrequencyDistributionTable), Accurate.is_accurate t) ∧
  (∀ (h : FrequencyDistributionHistogram), Intuitive.is_intuitive h) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_forms_l1954_195454


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l1954_195418

def moon_distance : ℝ := 384000

theorem moon_distance_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), moon_distance = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l1954_195418


namespace NUMINAMATH_CALUDE_simplify_expression_l1954_195463

theorem simplify_expression : (1024 : ℝ) ^ (1/5 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1954_195463


namespace NUMINAMATH_CALUDE_nora_oranges_count_l1954_195427

/-- The number of oranges Nora picked from the first tree -/
def oranges_tree1 : ℕ := 80

/-- The number of oranges Nora picked from the second tree -/
def oranges_tree2 : ℕ := 60

/-- The number of oranges Nora picked from the third tree -/
def oranges_tree3 : ℕ := 120

/-- The total number of oranges Nora picked -/
def total_oranges : ℕ := oranges_tree1 + oranges_tree2 + oranges_tree3

theorem nora_oranges_count : total_oranges = 260 := by
  sorry

end NUMINAMATH_CALUDE_nora_oranges_count_l1954_195427


namespace NUMINAMATH_CALUDE_carnival_tickets_used_l1954_195439

theorem carnival_tickets_used 
  (ferris_rides : ℕ) 
  (bumper_rides : ℕ) 
  (ticket_cost_per_ride : ℕ) 
  (h1 : ferris_rides = 5)
  (h2 : bumper_rides = 4)
  (h3 : ticket_cost_per_ride = 7) :
  (ferris_rides + bumper_rides) * ticket_cost_per_ride = 63 :=
by sorry

end NUMINAMATH_CALUDE_carnival_tickets_used_l1954_195439


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1954_195435

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 2 + a 7 = 18)
  (h_fourth : a 4 = 3) :
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1954_195435


namespace NUMINAMATH_CALUDE_unique_solution_l1954_195465

/-- Represents the arithmetic operations and equality --/
inductive Operation
| Add
| Sub
| Mul
| Div
| Eq

/-- The set of equations given in the problem --/
def Equations (A B C D E : Operation) : Prop :=
  (4 / 2 = 2) ∧
  (8 = 4 * 2) ∧
  (2 + 3 = 5) ∧
  (4 = 5 - 1) ∧
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧
  (C ≠ D) ∧ (C ≠ E) ∧
  (D ≠ E)

/-- The theorem stating the unique solution to the problem --/
theorem unique_solution :
  ∃! (A B C D E : Operation),
    Equations A B C D E ∧
    A = Operation.Div ∧
    B = Operation.Eq ∧
    C = Operation.Mul ∧
    D = Operation.Add ∧
    E = Operation.Sub := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1954_195465


namespace NUMINAMATH_CALUDE_max_player_salary_l1954_195451

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 18 →
  min_salary = 20000 →
  total_cap = 600000 →
  (∃ (salaries : Fin n → ℕ), 
    (∀ i, salaries i ≥ min_salary) ∧ 
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∀ i, salaries i ≤ 260000) ∧
    (∃ j, salaries j = 260000)) ∧
  ¬(∃ (salaries : Fin n → ℕ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ total_cap) ∧
    (∃ j, salaries j > 260000)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l1954_195451


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_l1954_195437

theorem consecutive_integers_product_812 (x : ℕ) :
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_l1954_195437


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_quadratic_function_solution_l1954_195468

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  x^2 - 6*x + 3 = 0 ↔ x = 3 + Real.sqrt 6 ∨ x = 3 - Real.sqrt 6 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  x*(x+2) = 3*(x+2) ↔ x = -2 ∨ x = 3 :=
sorry

-- Equation 3
def quadratic_function (x : ℝ) : ℝ := 4*x^2 + 5*x

theorem quadratic_function_solution :
  (quadratic_function 0 = 0) ∧ 
  (quadratic_function (-1) = -1) ∧ 
  (quadratic_function 1 = 9) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_quadratic_function_solution_l1954_195468


namespace NUMINAMATH_CALUDE_base5_product_121_11_l1954_195430

/-- Represents a number in base 5 --/
def Base5 := ℕ

/-- Multiplication in base 5 --/
def mul_base5 (a b : Base5) : Base5 := sorry

/-- Addition in base 5 --/
def add_base5 (a b : Base5) : Base5 := sorry

/-- Convert a natural number to its base 5 representation --/
def to_base5 (n : ℕ) : Base5 := sorry

/-- Theorem: The product of 121₅ and 11₅ in base 5 is 1331₅ --/
theorem base5_product_121_11 :
  mul_base5 (to_base5 121) (to_base5 11) = to_base5 1331 := by sorry

end NUMINAMATH_CALUDE_base5_product_121_11_l1954_195430


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l1954_195466

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m : ℝ) : Prop :=
  (-(1 : ℝ) / (1 + m) = -(m / 2))

/-- The first line equation -/
def line1 (m x y : ℝ) : Prop :=
  x + (1 + m) * y = 2 - m

/-- The second line equation -/
def line2 (m x y : ℝ) : Prop :=
  m * x + 2 * y + 8 = 0

/-- The theorem stating that the lines are parallel if and only if m = 1 -/
theorem lines_parallel_iff_m_eq_one :
  ∀ m : ℝ, (∃ x y : ℝ, line1 m x y ∧ line2 m x y) →
    (parallel_lines m ↔ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_l1954_195466


namespace NUMINAMATH_CALUDE_seventh_observation_l1954_195440

theorem seventh_observation (initial_count : Nat) (initial_avg : ℝ) (new_avg : ℝ) :
  initial_count = 6 →
  initial_avg = 16 →
  new_avg = 15 →
  (initial_count * initial_avg + (initial_count + 1) * new_avg - initial_count * initial_avg) / 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_l1954_195440


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1954_195487

theorem quadratic_inequality_solution (a : ℝ) :
  (a > 0 → ∀ x : ℝ, x^2 - a*x - 2*a^2 < 0 ↔ -a < x ∧ x < 2*a) ∧
  (a < 0 → ∀ x : ℝ, x^2 - a*x - 2*a^2 < 0 ↔ 2*a < x ∧ x < -a) ∧
  (a = 0 → ¬∃ x : ℝ, x^2 - a*x - 2*a^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1954_195487


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1954_195429

/-- The side length of an equilateral triangle with the same perimeter as a regular pentagon -/
theorem equilateral_triangle_side_length (pentagon_side : ℝ) (triangle_side : ℝ) : 
  pentagon_side = 5 → 
  5 * pentagon_side = 3 * triangle_side → 
  triangle_side = 25 / 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1954_195429


namespace NUMINAMATH_CALUDE_distance_from_point_on_number_line_l1954_195476

theorem distance_from_point_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_distance_from_point_on_number_line_l1954_195476


namespace NUMINAMATH_CALUDE_line_chart_for_weekly_temperature_l1954_195441

/-- A type representing different chart types -/
inductive ChartType
  | Bar
  | Line
  | Pie
  | Scatter

/-- A structure representing data over time -/
structure TimeSeriesData where
  time_period : String
  has_continuous_change : Bool

/-- A function to determine the most appropriate chart type for a given data set -/
def most_appropriate_chart (data : TimeSeriesData) : ChartType :=
  if data.has_continuous_change then ChartType.Line else ChartType.Bar

/-- Theorem stating that a line chart is most appropriate for weekly temperature data -/
theorem line_chart_for_weekly_temperature :
  let weekly_temp_data : TimeSeriesData := { time_period := "Week", has_continuous_change := true }
  most_appropriate_chart weekly_temp_data = ChartType.Line :=
by
  sorry


end NUMINAMATH_CALUDE_line_chart_for_weekly_temperature_l1954_195441


namespace NUMINAMATH_CALUDE_coffee_lasts_13_days_l1954_195478

def coffee_problem (coffee_weight : ℕ) (cups_per_pound : ℕ) 
  (angie_cups : ℕ) (bob_cups : ℕ) (carol_cups : ℕ) : ℕ :=
  let total_cups := coffee_weight * cups_per_pound
  let daily_consumption := angie_cups + bob_cups + carol_cups
  total_cups / daily_consumption

theorem coffee_lasts_13_days :
  coffee_problem 3 40 3 2 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_coffee_lasts_13_days_l1954_195478


namespace NUMINAMATH_CALUDE_first_year_payment_is_20_l1954_195409

/-- Represents the payment structure over four years -/
structure PaymentStructure where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Defines the conditions of the payment structure -/
def valid_payment_structure (p : PaymentStructure) : Prop :=
  p.second_year = p.first_year + 2 ∧
  p.third_year = p.second_year + 3 ∧
  p.fourth_year = p.third_year + 4 ∧
  p.first_year + p.second_year + p.third_year + p.fourth_year = 96

/-- Theorem stating that the first year's payment is 20 rupees -/
theorem first_year_payment_is_20 :
  ∀ (p : PaymentStructure), valid_payment_structure p → p.first_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_year_payment_is_20_l1954_195409


namespace NUMINAMATH_CALUDE_xiao_ming_current_age_l1954_195443

/-- Xiao Ming's age this year -/
def xiao_ming_age : ℕ := sorry

/-- Xiao Ming's mother's age this year -/
def mother_age : ℕ := sorry

/-- Xiao Ming's age three years from now -/
def xiao_ming_age_future : ℕ := sorry

/-- Xiao Ming's mother's age three years from now -/
def mother_age_future : ℕ := sorry

/-- The theorem stating Xiao Ming's age this year -/
theorem xiao_ming_current_age :
  (mother_age = 3 * xiao_ming_age) ∧
  (mother_age_future = 2 * xiao_ming_age_future + 10) ∧
  (xiao_ming_age_future = xiao_ming_age + 3) ∧
  (mother_age_future = mother_age + 3) →
  xiao_ming_age = 13 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_current_age_l1954_195443


namespace NUMINAMATH_CALUDE_largest_integer_solution_l1954_195445

theorem largest_integer_solution : 
  (∀ x : ℤ, 10 - 3*x > 25 → x ≤ -6) ∧ (10 - 3*(-6) > 25) := by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l1954_195445


namespace NUMINAMATH_CALUDE_work_scaling_l1954_195492

/-- Given that 3 people can do 3 times of a particular work in 3 days,
    prove that 6 people can do 6 times of that work in the same number of days. -/
theorem work_scaling (work : ℕ → ℕ → ℕ → Prop) : 
  work 3 3 3 → work 6 6 3 :=
by sorry

end NUMINAMATH_CALUDE_work_scaling_l1954_195492


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1954_195491

theorem squirrel_acorns (total_acorns : ℕ) (winter_months : ℕ) (spring_acorns : ℕ) 
  (h1 : total_acorns = 210)
  (h2 : winter_months = 3)
  (h3 : spring_acorns = 30) :
  (total_acorns / winter_months) - (spring_acorns / winter_months) = 60 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1954_195491
