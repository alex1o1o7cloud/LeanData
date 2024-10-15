import Mathlib

namespace NUMINAMATH_CALUDE_max_students_distribution_l2583_258357

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1340) (h_pencils : pencils = 1280) : 
  Nat.gcd pens pencils = 20 := by
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l2583_258357


namespace NUMINAMATH_CALUDE_magnitude_of_one_minus_i_l2583_258372

theorem magnitude_of_one_minus_i :
  let z : ℂ := 1 - Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_one_minus_i_l2583_258372


namespace NUMINAMATH_CALUDE_original_prices_calculation_l2583_258309

/-- Given the price increases and final prices of three items, prove their original prices. -/
theorem original_prices_calculation (computer_increase : ℝ) (tv_increase : ℝ) (fridge_increase : ℝ)
  (computer_final : ℝ) (tv_final : ℝ) (fridge_final : ℝ)
  (h1 : computer_increase = 0.30)
  (h2 : tv_increase = 0.20)
  (h3 : fridge_increase = 0.15)
  (h4 : computer_final = 377)
  (h5 : tv_final = 720)
  (h6 : fridge_final = 1150) :
  ∃ (computer_original tv_original fridge_original : ℝ),
    computer_original = 290 ∧
    tv_original = 600 ∧
    fridge_original = 1000 ∧
    computer_final = computer_original * (1 + computer_increase) ∧
    tv_final = tv_original * (1 + tv_increase) ∧
    fridge_final = fridge_original * (1 + fridge_increase) :=
by sorry

end NUMINAMATH_CALUDE_original_prices_calculation_l2583_258309


namespace NUMINAMATH_CALUDE_remainder_of_n_mod_11_l2583_258396

def A : ℕ := (10^20069 - 1) / 9
def B : ℕ := 7 * (10^20066 - 1) / 9

def n : ℤ := A^2 - B

theorem remainder_of_n_mod_11 : n % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_n_mod_11_l2583_258396


namespace NUMINAMATH_CALUDE_light_flash_duration_l2583_258376

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The interval between flashes in seconds -/
def flash_interval : ℕ := 20

/-- The number of flashes -/
def num_flashes : ℕ := 180

/-- Theorem: The time it takes for 180 flashes of a light that flashes every 20 seconds is equal to 1 hour -/
theorem light_flash_duration : 
  (flash_interval * num_flashes) = seconds_per_hour := by sorry

end NUMINAMATH_CALUDE_light_flash_duration_l2583_258376


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2583_258311

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| - |x - 1| > m) → m < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2583_258311


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2583_258342

-- Define the initial amount
def initial_amount : ℚ := 64000

-- Define the annual increase rate
def annual_rate : ℚ := 1 / 8

-- Define the time period in years
def years : ℕ := 2

-- Define the function to calculate the amount after n years
def amount_after_years (initial : ℚ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

-- Theorem statement
theorem amount_after_two_years :
  amount_after_years initial_amount annual_rate years = 81000 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2583_258342


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l2583_258369

def z : ℂ := 3 * Complex.I * (Complex.I + 1)

theorem complex_parts_of_z :
  Complex.re z = -3 ∧ Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l2583_258369


namespace NUMINAMATH_CALUDE_k_range_l2583_258370

def p (k : ℝ) : Prop := ∀ x y : ℝ, x < y → k * x + 1 < k * y + 1

def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

theorem k_range (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → 
  (k ≤ 0 ∨ (1/2 < k ∧ k < 5/2)) :=
sorry

end NUMINAMATH_CALUDE_k_range_l2583_258370


namespace NUMINAMATH_CALUDE_no_simultaneous_extrema_l2583_258355

/-- A partition of rational numbers -/
structure RationalPartition where
  M : Set ℚ
  N : Set ℚ
  M_nonempty : M.Nonempty
  N_nonempty : N.Nonempty
  union_eq_rat : M ∪ N = Set.univ
  intersection_empty : M ∩ N = ∅
  M_lt_N : ∀ m ∈ M, ∀ n ∈ N, m < n

/-- Theorem stating that in a partition of rationals, M cannot have a maximum and N cannot have a minimum simultaneously -/
theorem no_simultaneous_extrema (p : RationalPartition) :
  ¬(∃ (max_M : ℚ), max_M ∈ p.M ∧ ∀ m ∈ p.M, m ≤ max_M) ∨
  ¬(∃ (min_N : ℚ), min_N ∈ p.N ∧ ∀ n ∈ p.N, min_N ≤ n) :=
sorry

end NUMINAMATH_CALUDE_no_simultaneous_extrema_l2583_258355


namespace NUMINAMATH_CALUDE_f_zero_implies_a_bound_l2583_258366

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_implies_a_bound (a : ℝ) :
  (∃ x > 0, f x a = 0) →
  a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_implies_a_bound_l2583_258366


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2583_258388

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 64 * π → 2 * π * r^2 + π * r^2 = 192 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2583_258388


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_l2583_258306

theorem smallest_three_digit_multiple_of_9 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 9 ∣ n → n ≥ 108 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_l2583_258306


namespace NUMINAMATH_CALUDE_nine_cakes_l2583_258381

/-- Represents the arrangement of cakes on a round table. -/
def CakeArrangement (n : ℕ) := Fin n

/-- Represents the action of eating every third cake. -/
def eatEveryThird (n : ℕ) (i : Fin n) : Fin n :=
  ⟨(i + 3) % n, by sorry⟩

/-- Represents the number of laps needed to eat all cakes. -/
def lapsToEatAll (n : ℕ) : ℕ := 7

/-- The last cake eaten is the same as the first one encountered. -/
def lastIsFirst (n : ℕ) : Prop :=
  ∃ (i : Fin n), (lapsToEatAll n).iterate (eatEveryThird n) i = i

/-- The main theorem stating that there are 9 cakes on the table. -/
theorem nine_cakes :
  ∃ (n : ℕ), n = 9 ∧ 
  lapsToEatAll n = 7 ∧
  lastIsFirst n :=
sorry

end NUMINAMATH_CALUDE_nine_cakes_l2583_258381


namespace NUMINAMATH_CALUDE_buy_three_items_count_l2583_258302

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def ways_to_buy_three (inventory : StoreInventory) : Nat :=
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards +
  inventory.headphones * inventory.mice * inventory.keyboards

/-- The theorem stating that there are 646 ways to buy three items -/
theorem buy_three_items_count (inventory : StoreInventory) 
  (h1 : inventory.headphones = 9)
  (h2 : inventory.mice = 13)
  (h3 : inventory.keyboards = 5)
  (h4 : inventory.keyboard_mouse_sets = 4)
  (h5 : inventory.headphone_mouse_sets = 5) :
  ways_to_buy_three inventory = 646 := by
  sorry

#eval ways_to_buy_three { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end NUMINAMATH_CALUDE_buy_three_items_count_l2583_258302


namespace NUMINAMATH_CALUDE_expression_equality_l2583_258322

theorem expression_equality (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) = 
  6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2583_258322


namespace NUMINAMATH_CALUDE_sunglasses_wearers_l2583_258362

theorem sunglasses_wearers (total_adults : ℕ) (women_percentage : ℚ) (men_percentage : ℚ) : 
  total_adults = 1800 → 
  women_percentage = 25 / 100 →
  men_percentage = 10 / 100 →
  (total_adults / 2 * women_percentage + total_adults / 2 * men_percentage : ℚ) = 315 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_wearers_l2583_258362


namespace NUMINAMATH_CALUDE_certain_event_draw_two_white_l2583_258333

/-- A box containing only white balls -/
structure WhiteBallBox where
  num_balls : ℕ

/-- The probability of drawing two white balls from a box -/
def prob_draw_two_white (box : WhiteBallBox) : ℚ :=
  if box.num_balls ≥ 2 then 1 else 0

/-- Theorem: Drawing 2 white balls from a box with 5 white balls is a certain event -/
theorem certain_event_draw_two_white :
  prob_draw_two_white ⟨5⟩ = 1 := by sorry

end NUMINAMATH_CALUDE_certain_event_draw_two_white_l2583_258333


namespace NUMINAMATH_CALUDE_S_max_at_n_max_l2583_258308

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n.val^2 + 8*n.val

/-- n_max is the value of n at which S_n attains its maximum value -/
def n_max : ℕ+ := 4

theorem S_max_at_n_max :
  ∀ n : ℕ+, S n ≤ S n_max :=
sorry

end NUMINAMATH_CALUDE_S_max_at_n_max_l2583_258308


namespace NUMINAMATH_CALUDE_fox_distribution_l2583_258303

/-- The fox distribution problem -/
theorem fox_distribution
  (m : ℕ) (a : ℝ) (x y : ℝ)
  (h_positive : m > 1 ∧ a > 0)
  (h_distribution : ∀ (n : ℕ), n > 0 → n * a + (x - (n - 1) * y - n * a) / m = y) :
  x = (m - 1)^2 * a ∧ y = (m - 1) * a ∧ (m - 1 : ℝ) = x / y :=
by sorry

end NUMINAMATH_CALUDE_fox_distribution_l2583_258303


namespace NUMINAMATH_CALUDE_sequence_properties_l2583_258373

def S (n : ℕ) : ℝ := 3 * n^2 - 2 * n

def a : ℕ → ℝ := λ n => 6 * n - 5

theorem sequence_properties :
  (∀ n, S n = 3 * n^2 - 2 * n) →
  (∀ n, a n = 6 * n - 5) ∧
  (a 1 = 1) ∧
  (∀ n, n ≥ 2 → a n - a (n-1) = 6) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2583_258373


namespace NUMINAMATH_CALUDE_trig_identity_30_degrees_l2583_258305

theorem trig_identity_30_degrees :
  let tan30 : ℝ := 1 / Real.sqrt 3
  let sin30 : ℝ := 1 / 2
  (tan30^2 - sin30^2) / (tan30^2 * sin30^2) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_30_degrees_l2583_258305


namespace NUMINAMATH_CALUDE_parametric_to_equation_l2583_258356

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  constant : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.B) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.C) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The main theorem to prove -/
theorem parametric_to_equation (plane : ParametricPlane) :
  ∃ (eq : PlaneEquation),
    validCoefficients eq ∧
    (∀ (s t : ℝ),
      let p : Point3D := {
        x := plane.constant.x + s * plane.direction1.x + t * plane.direction2.x,
        y := plane.constant.y + s * plane.direction1.y + t * plane.direction2.y,
        z := plane.constant.z + s * plane.direction1.z + t * plane.direction2.z
      }
      satisfiesEquation p eq) ∧
    eq.A = 2 ∧ eq.B = -5 ∧ eq.C = 2 ∧ eq.D = -7 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_equation_l2583_258356


namespace NUMINAMATH_CALUDE_cosine_half_angle_in_interval_l2583_258350

theorem cosine_half_angle_in_interval (θ m : Real) 
  (h1 : 5/2 * Real.pi < θ) 
  (h2 : θ < 3 * Real.pi) 
  (h3 : |Real.cos θ| = m) : 
  Real.cos (θ/2) = -Real.sqrt ((1 - m)/2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_half_angle_in_interval_l2583_258350


namespace NUMINAMATH_CALUDE_ingrids_tax_rate_l2583_258325

theorem ingrids_tax_rate 
  (john_tax_rate : ℝ)
  (john_income : ℝ)
  (ingrid_income : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : john_tax_rate = 0.30)
  (h2 : john_income = 58000)
  (h3 : ingrid_income = 72000)
  (h4 : combined_tax_rate = 0.3554)
  : (combined_tax_rate * (john_income + ingrid_income) - john_tax_rate * john_income) / ingrid_income = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_ingrids_tax_rate_l2583_258325


namespace NUMINAMATH_CALUDE_calculation_result_l2583_258379

theorem calculation_result : 1 + 0.1 - 0.1 + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2583_258379


namespace NUMINAMATH_CALUDE_tea_containers_needed_l2583_258330

/-- The volume of tea in milliliters that each container can hold -/
def container_volume : ℕ := 500

/-- The minimum volume of tea in liters needed for the event -/
def required_volume : ℕ := 5

/-- Conversion factor from liters to milliliters -/
def liter_to_ml : ℕ := 1000

/-- The minimum number of containers needed to hold at least the required volume of tea -/
def min_containers : ℕ := 10

theorem tea_containers_needed :
  min_containers = 
    (required_volume * liter_to_ml + container_volume - 1) / container_volume :=
by sorry

end NUMINAMATH_CALUDE_tea_containers_needed_l2583_258330


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2583_258399

theorem sum_of_fractions : 
  (4 : ℚ) / 3 + 8 / 9 + 18 / 27 + 40 / 81 + 88 / 243 - 5 = -305 / 243 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2583_258399


namespace NUMINAMATH_CALUDE_balanced_allocation_exists_l2583_258361

/-- Represents the daily production capacity of a worker for each part type -/
structure ProductionRate where
  typeA : ℕ
  typeB : ℕ

/-- Represents the composition of a set in terms of part types -/
structure SetComposition where
  typeA : ℕ
  typeB : ℕ

/-- Represents the allocation of workers to different part types -/
structure WorkerAllocation where
  typeA : ℕ
  typeB : ℕ

/-- Checks if the worker allocation is valid and balanced -/
def isBalancedAllocation (totalWorkers : ℕ) (rate : ProductionRate) (composition : SetComposition) (allocation : WorkerAllocation) : Prop :=
  allocation.typeA + allocation.typeB = totalWorkers ∧
  rate.typeA * allocation.typeA * composition.typeB = rate.typeB * allocation.typeB * composition.typeA

theorem balanced_allocation_exists (totalWorkers : ℕ) (rate : ProductionRate) (composition : SetComposition) 
    (h_total : totalWorkers = 85)
    (h_rate : rate = { typeA := 10, typeB := 16 })
    (h_composition : composition = { typeA := 3, typeB := 2 }) :
  ∃ (allocation : WorkerAllocation), isBalancedAllocation totalWorkers rate composition allocation ∧ 
    allocation.typeA = 60 ∧ allocation.typeB = 25 := by
  sorry

end NUMINAMATH_CALUDE_balanced_allocation_exists_l2583_258361


namespace NUMINAMATH_CALUDE_value_of_y_l2583_258363

theorem value_of_y : ∃ y : ℝ, (3 * y - 9) / 3 = 18 ∧ y = 21 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2583_258363


namespace NUMINAMATH_CALUDE_rectangle_area_l2583_258393

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (a b c d : ℝ × ℝ) : 
  a = (-3, 1) → b = (1, 1) → c = (1, -2) → d = (-3, -2) →
  (b.1 - a.1) * (a.2 - d.2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2583_258393


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l2583_258398

theorem smallest_n_for_fraction_inequality : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ (m : ℤ), 0 < m → m < 2004 → 
    ∃ (k : ℤ), (m : ℚ) / 2004 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m + 1) : ℚ) / 2005) ∧
  (∀ (n' : ℕ), 0 < n' → n' < n → 
    ∃ (m : ℤ), 0 < m ∧ m < 2004 ∧
      ∀ (k : ℤ), ¬((m : ℚ) / 2004 < (k : ℚ) / n' ∧ (k : ℚ) / n' < ((m + 1) : ℚ) / 2005)) ∧
  n = 4009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_inequality_l2583_258398


namespace NUMINAMATH_CALUDE_age_ratio_l2583_258394

def sachin_age : ℕ := 14
def age_difference : ℕ := 7

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l2583_258394


namespace NUMINAMATH_CALUDE_unique_solution_floor_product_l2583_258323

theorem unique_solution_floor_product : 
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 45 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_product_l2583_258323


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l2583_258378

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

-- Theorem statement
theorem fixed_point_on_circle :
  ∀ m : ℝ, circle_equation 1 1 m ∨ circle_equation (1/5) (7/5) m :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l2583_258378


namespace NUMINAMATH_CALUDE_min_value_of_f_l2583_258340

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2583_258340


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2583_258315

-- Problem 1
theorem problem_1 : Real.sqrt 27 - (1/3) * Real.sqrt 18 - Real.sqrt 12 = Real.sqrt 3 - Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 48 + Real.sqrt 30 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 * Real.sqrt 3 + Real.sqrt 30 + Real.sqrt 6 := by sorry

-- Problem 3
theorem problem_3 : (2 - Real.sqrt 5) * (2 + Real.sqrt 5) - (2 - Real.sqrt 2)^2 = 4 * Real.sqrt 2 - 7 := by sorry

-- Problem 4
theorem problem_4 : (27 : Real)^(1/3) - (Real.sqrt 2 * Real.sqrt 6) / Real.sqrt 3 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2583_258315


namespace NUMINAMATH_CALUDE_savings_difference_is_75_cents_l2583_258352

/-- The price of the book in dollars -/
def book_price : ℚ := 30

/-- The fixed discount amount in dollars -/
def fixed_discount : ℚ := 5

/-- The percentage discount as a decimal -/
def percent_discount : ℚ := 0.15

/-- The cost after applying the fixed discount first, then the percentage discount -/
def cost_fixed_first : ℚ := (book_price - fixed_discount) * (1 - percent_discount)

/-- The cost after applying the percentage discount first, then the fixed discount -/
def cost_percent_first : ℚ := book_price * (1 - percent_discount) - fixed_discount

/-- The difference in savings between the two discount sequences, in cents -/
def savings_difference : ℚ := (cost_fixed_first - cost_percent_first) * 100

theorem savings_difference_is_75_cents : savings_difference = 75 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_is_75_cents_l2583_258352


namespace NUMINAMATH_CALUDE_shortest_dividing_line_l2583_258327

-- Define a circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a broken line
def BrokenLine := List (ℝ × ℝ)

-- Function to calculate the length of a broken line
def length (bl : BrokenLine) : ℝ := sorry

-- Function to check if a broken line divides the circle into two equal parts
def divides_equally (bl : BrokenLine) (c : Circle) : Prop := sorry

-- Define the diameter of a circle
def diameter (c : Circle) : ℝ := 2

-- Theorem statement
theorem shortest_dividing_line (c : Circle) (bl : BrokenLine) :
  divides_equally bl c → length bl ≥ diameter c ∧
  (length bl = diameter c ↔ ∃ a b : ℝ × ℝ, bl = [a, b] ∧ a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ (a.1 + b.1 = 0 ∧ a.2 + b.2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_shortest_dividing_line_l2583_258327


namespace NUMINAMATH_CALUDE_kimberly_skittles_l2583_258390

/-- Given that Kimberly buys 7 more Skittles and ends up with 12 Skittles in total,
    prove that she initially had 5 Skittles. -/
theorem kimberly_skittles (bought : ℕ) (total : ℕ) (initial : ℕ) : 
  bought = 7 → total = 12 → initial + bought = total → initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_skittles_l2583_258390


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_x_range_of_x_for_inequality_l2583_258382

-- Define the function f
def f (x : ℝ) := |2*x - 1| - |x + 1|

-- Theorem for part I
theorem solution_set_f_greater_than_x :
  {x : ℝ | f x > x} = {x : ℝ | x < 0} := by sorry

-- Theorem for part II
theorem range_of_x_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, (1/a + 4/b) ≥ f x) →
  (∀ x : ℝ, f x ≤ 9) →
  (∀ x : ℝ, -7 ≤ x ∧ x ≤ 11) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_x_range_of_x_for_inequality_l2583_258382


namespace NUMINAMATH_CALUDE_max_value_operation_l2583_258335

theorem max_value_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 4 * (300 - n) ≤ 1160 :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l2583_258335


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2583_258329

theorem gcd_of_specific_numbers : Nat.gcd 333333 888888888 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l2583_258329


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l2583_258334

-- Define the sample space (faces of the die)
def Die : Type := Fin 6

-- Define event B
def eventB (x : Die) : Prop := x.val + 1 ≤ 3

-- Define event C
def eventC (x : Die) : Prop := x.val + 1 ≥ 4

-- Theorem statement
theorem events_B_C_complementary :
  ∀ (x : Die), (eventB x ∧ ¬eventC x) ∨ (¬eventB x ∧ eventC x) :=
by sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l2583_258334


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l2583_258314

theorem smallest_perfect_square_divisible_by_2_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  (∃ m : ℕ, n = m ^ 2) ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m ^ 2) → k % 2 = 0 → k % 5 = 0 → k ≥ n) ∧
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l2583_258314


namespace NUMINAMATH_CALUDE_project_distribution_count_l2583_258326

/-- The number of ways to distribute 8 distinct projects among 4 companies -/
def distribute_projects : ℕ :=
  Nat.choose 8 3 * Nat.choose 5 1 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l2583_258326


namespace NUMINAMATH_CALUDE_juice_bar_spending_l2583_258337

theorem juice_bar_spending (mango_price pineapple_price pineapple_total group_size : ℕ) 
  (h1 : mango_price = 5)
  (h2 : pineapple_price = 6)
  (h3 : pineapple_total = 54)
  (h4 : group_size = 17) :
  ∃ (mango_glasses pineapple_glasses : ℕ),
    mango_glasses + pineapple_glasses = group_size ∧
    mango_glasses * mango_price + pineapple_glasses * pineapple_price = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_juice_bar_spending_l2583_258337


namespace NUMINAMATH_CALUDE_association_member_condition_l2583_258324

/-- Represents a member of the association -/
structure Member where
  number : Nat
  country : Fin 6

/-- The set of all members in the association -/
def Association := Fin 1978 → Member

/-- Predicate to check if a member's number satisfies the condition -/
def SatisfiesCondition (assoc : Association) (m : Member) : Prop :=
  ∃ (a b : Member),
    a.country = m.country ∧ b.country = m.country ∧
    ((a.number + b.number = m.number) ∨ (2 * a.number = m.number))

/-- Main theorem -/
theorem association_member_condition (assoc : Association) :
  ∃ (m : Member), m ∈ Set.range assoc ∧ SatisfiesCondition assoc m := by
  sorry


end NUMINAMATH_CALUDE_association_member_condition_l2583_258324


namespace NUMINAMATH_CALUDE_expression_equality_l2583_258359

theorem expression_equality : -2^2 + (1 / (Real.sqrt 2 - 1))^0 - abs (2 * Real.sqrt 2 - 3) + Real.cos (π / 3) = -5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2583_258359


namespace NUMINAMATH_CALUDE_unemployment_rate_calculation_l2583_258341

theorem unemployment_rate_calculation (previous_employment_rate previous_unemployment_rate : ℝ)
  (h1 : previous_employment_rate + previous_unemployment_rate = 100)
  (h2 : previous_employment_rate > 0)
  (h3 : previous_unemployment_rate > 0) :
  let new_employment_rate := 0.85 * previous_employment_rate
  let new_unemployment_rate := 1.1 * previous_unemployment_rate
  new_unemployment_rate = 66 :=
by
  sorry

#check unemployment_rate_calculation

end NUMINAMATH_CALUDE_unemployment_rate_calculation_l2583_258341


namespace NUMINAMATH_CALUDE_dime_difference_is_243_l2583_258328

/-- Represents the types of coins in the piggy bank --/
inductive Coin
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : Coin → Nat
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A configuration of coins in the piggy bank --/
structure CoinConfiguration where
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a configuration --/
def totalCoins (c : CoinConfiguration) : Nat :=
  c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of coins in a configuration in cents --/
def totalValue (c : CoinConfiguration) : Nat :=
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter +
  c.halfDollars * coinValue Coin.HalfDollar

/-- Predicate to check if a configuration is valid --/
def isValidConfiguration (c : CoinConfiguration) : Prop :=
  totalCoins c = 150 ∧ totalValue c = 2000

/-- The maximum number of dimes possible in a valid configuration --/
def maxDimes : Nat :=
  250

/-- The minimum number of dimes possible in a valid configuration --/
def minDimes : Nat :=
  7

theorem dime_difference_is_243 :
  ∃ (cMax cMin : CoinConfiguration),
    isValidConfiguration cMax ∧
    isValidConfiguration cMin ∧
    cMax.dimes = maxDimes ∧
    cMin.dimes = minDimes ∧
    maxDimes - minDimes = 243 :=
  sorry

end NUMINAMATH_CALUDE_dime_difference_is_243_l2583_258328


namespace NUMINAMATH_CALUDE_square_difference_l2583_258320

theorem square_difference (x y z : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15) :
  x^2 - z^2 = -115 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2583_258320


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2583_258385

/-- An arithmetic sequence and its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ  -- The sequence
  d : ℚ       -- Common difference
  sum : ℕ+ → ℚ -- Sum function
  sum_def : ∀ n : ℕ+, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The sequence b_n defined as S_n / n -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  seq.sum n / n

/-- Main theorem about the properties of sequence b_n -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.sum 7 = 7)
    (h2 : seq.sum 15 = 75) :
  (∀ n m : ℕ+, b seq (n + m) - b seq n = b seq (m + 1) - b seq 1) ∧
  (∀ n : ℕ+, b seq n = (n - 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2583_258385


namespace NUMINAMATH_CALUDE_no_prime_10101_base_n_l2583_258354

theorem no_prime_10101_base_n : ¬ ∃ (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_10101_base_n_l2583_258354


namespace NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_l2583_258374

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 0

-- State the theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∃ (m b : ℝ), ∀ x y, domain x → (y = f a x) →
  (x = 2 → y = f a 2 → x - 4*y + 4*Real.log 2 - 4 = 0) :=
sorry

-- Define the interval (0, e]
def interval (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

-- State the theorem for the minimum value
theorem minimum_value (a : ℝ) :
  (a ≤ 0 → ¬∃ m, ∀ x, interval x → f a x ≥ m) ∧
  (0 < a → a < Real.exp 1 → ∃ m, m = Real.log a ∧ ∀ x, interval x → f a x ≥ m) ∧
  (a ≥ Real.exp 1 → ∃ m, m = a / Real.exp 1 ∧ ∀ x, interval x → f a x ≥ m) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_minimum_value_l2583_258374


namespace NUMINAMATH_CALUDE_cookies_left_for_monica_l2583_258395

/-- The number of cookies Monica made for herself and her family. -/
def total_cookies : ℕ := 30

/-- The number of cookies Monica's father ate. -/
def father_cookies : ℕ := 10

/-- The number of cookies Monica's mother ate. -/
def mother_cookies : ℕ := father_cookies / 2

/-- The number of cookies Monica's brother ate. -/
def brother_cookies : ℕ := mother_cookies + 2

/-- Theorem stating the number of cookies left for Monica. -/
theorem cookies_left_for_monica : 
  total_cookies - father_cookies - mother_cookies - brother_cookies = 8 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_for_monica_l2583_258395


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l2583_258348

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l2583_258348


namespace NUMINAMATH_CALUDE_largest_power_divisor_l2583_258343

theorem largest_power_divisor (m n : ℕ) (h1 : m = 1991^1992) (h2 : n = 1991^1990) :
  ∃ k : ℕ, k = 1991^1990 ∧ 
  k ∣ (1990*m + 1992*n) ∧ 
  ∀ l : ℕ, l > k → l = 1991^(1990 + (l.log 1991 - 1990)) → ¬(l ∣ (1990*m + 1992*n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_divisor_l2583_258343


namespace NUMINAMATH_CALUDE_pitcher_problem_l2583_258391

theorem pitcher_problem (pitcher_capacity : ℝ) (h_positive : pitcher_capacity > 0) :
  let juice_amount : ℝ := (2/3) * pitcher_capacity
  let num_cups : ℕ := 6
  let juice_per_cup : ℝ := juice_amount / num_cups
  (juice_per_cup / pitcher_capacity) * 100 = 11.1111111111 :=
by sorry

end NUMINAMATH_CALUDE_pitcher_problem_l2583_258391


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2583_258318

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 ≥ 3) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2583_258318


namespace NUMINAMATH_CALUDE_function_equal_to_parabola_l2583_258383

-- Define a property for functions that have the same intersection behavior as x^2
def HasSameIntersections (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ), (∀ x : ℝ, (a * x + b = f x) ↔ (a * x + b = x^2))

-- State the theorem
theorem function_equal_to_parabola (f : ℝ → ℝ) :
  HasSameIntersections f → (∀ x : ℝ, f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equal_to_parabola_l2583_258383


namespace NUMINAMATH_CALUDE_monotonic_increasing_intervals_of_f_l2583_258392

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

-- State the theorem about monotonic increasing intervals
theorem monotonic_increasing_intervals_of_f :
  ∃ (a b : ℝ), 
    (∀ x y, x < y ∧ x < a → f x < f y) ∧
    (∀ x y, x < y ∧ b < x → f x < f y) ∧
    (∀ x, a ≤ x ∧ x ≤ b → ∃ y, x < y ∧ f x ≥ f y) ∧
    a = -1 ∧ b = 11 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_intervals_of_f_l2583_258392


namespace NUMINAMATH_CALUDE_ammonium_nitrate_formation_l2583_258336

-- Define the chemical species
def Ammonia : Type := Unit
def NitricAcid : Type := Unit
def AmmoniumNitrate : Type := Unit

-- Define the reaction
def reaction (nh3 : ℕ) (hno3 : ℕ) : ℕ :=
  min nh3 hno3

-- State the theorem
theorem ammonium_nitrate_formation 
  (nh3 : ℕ) -- Some moles of Ammonia
  (hno3 : ℕ) -- Moles of Nitric acid
  (h1 : hno3 = 3) -- 3 moles of Nitric acid are used
  (h2 : reaction nh3 hno3 = 3) -- Total moles of Ammonium nitrate formed are 3
  : reaction nh3 hno3 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ammonium_nitrate_formation_l2583_258336


namespace NUMINAMATH_CALUDE_bathing_suits_for_men_l2583_258387

theorem bathing_suits_for_men (total : ℕ) (women : ℕ) (men : ℕ) : 
  total = 19766 → women = 4969 → men = total - women → men = 14797 :=
by sorry

end NUMINAMATH_CALUDE_bathing_suits_for_men_l2583_258387


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2583_258346

def set_A : Set ℝ := {x | Real.cos x = 0}
def set_B : Set ℝ := {x | x^2 - 5*x ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {Real.pi/2, 3*Real.pi/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2583_258346


namespace NUMINAMATH_CALUDE_max_x2_plus_y2_l2583_258371

theorem max_x2_plus_y2 (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3*x + y| ≤ 6) : x^2 + y^2 ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_x2_plus_y2_l2583_258371


namespace NUMINAMATH_CALUDE_quinns_reading_challenge_l2583_258312

/-- Proves the number of weeks Quinn needs to participate in the reading challenge -/
theorem quinns_reading_challenge
  (books_per_donut : ℕ)
  (books_per_week : ℕ)
  (target_donuts : ℕ)
  (h1 : books_per_donut = 5)
  (h2 : books_per_week = 2)
  (h3 : target_donuts = 4) :
  (target_donuts * books_per_donut) / books_per_week = 10 :=
by sorry

end NUMINAMATH_CALUDE_quinns_reading_challenge_l2583_258312


namespace NUMINAMATH_CALUDE_fraction_simplification_l2583_258389

theorem fraction_simplification (b y : ℝ) :
  (Real.sqrt (b^2 + y^2) - (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 + 2*y^2) = 
  2*b^2 / (b^2 + 2*y^2)^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2583_258389


namespace NUMINAMATH_CALUDE_investment_rate_correct_l2583_258319

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := sorry

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 20000

/-- The amount withdrawn after the first year in yuan -/
def withdrawal : ℝ := 10000

/-- The final amount received after two years in yuan -/
def final_amount : ℝ := 13200

/-- Theorem stating that the annual interest rate satisfies the investment conditions -/
theorem investment_rate_correct : 
  (initial_investment * (1 + annual_interest_rate) - withdrawal) * (1 + annual_interest_rate) = final_amount ∧ 
  annual_interest_rate = 0.1 := by sorry

end NUMINAMATH_CALUDE_investment_rate_correct_l2583_258319


namespace NUMINAMATH_CALUDE_max_salary_soccer_team_l2583_258353

/-- Represents the maximum salary problem for a soccer team -/
theorem max_salary_soccer_team 
  (num_players : ℕ) 
  (min_salary : ℕ) 
  (max_total_salary : ℕ) 
  (h1 : num_players = 25)
  (h2 : min_salary = 20000)
  (h3 : max_total_salary = 900000) :
  ∃ (max_single_salary : ℕ),
    max_single_salary = 420000 ∧
    max_single_salary + (num_players - 1) * min_salary ≤ max_total_salary ∧
    ∀ (salary : ℕ), 
      salary > max_single_salary → 
      salary + (num_players - 1) * min_salary > max_total_salary :=
by sorry

end NUMINAMATH_CALUDE_max_salary_soccer_team_l2583_258353


namespace NUMINAMATH_CALUDE_f_max_value_l2583_258300

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l2583_258300


namespace NUMINAMATH_CALUDE_committee_rearrangements_count_l2583_258364

/-- The number of distinguishable rearrangements of the letters in "COMMITTEE" with all vowels at the beginning of the sequence -/
def committee_rearrangements : ℕ := sorry

/-- The number of vowels in "COMMITTEE" -/
def num_vowels : ℕ := 4

/-- The number of consonants in "COMMITTEE" -/
def num_consonants : ℕ := 5

/-- The number of repeated vowels (E) in "COMMITTEE" -/
def num_repeated_vowels : ℕ := 2

/-- The number of repeated consonants (M and T) in "COMMITTEE" -/
def num_repeated_consonants : ℕ := 2

theorem committee_rearrangements_count :
  committee_rearrangements = (Nat.factorial num_vowels / Nat.factorial num_repeated_vowels) *
                             (Nat.factorial num_consonants / (Nat.factorial num_repeated_consonants * Nat.factorial num_repeated_consonants)) :=
by sorry

end NUMINAMATH_CALUDE_committee_rearrangements_count_l2583_258364


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2583_258377

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

theorem complement_A_intersect_B : (Aᶜ ∩ B) = {7, 9} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2583_258377


namespace NUMINAMATH_CALUDE_marble_drawing_probability_l2583_258304

/-- Represents the total number of marbles in the bag. -/
def total_marbles : ℕ := 800

/-- Represents the number of different colors of marbles. -/
def num_colors : ℕ := 100

/-- Represents the number of marbles of each color. -/
def marbles_per_color : ℕ := 8

/-- Represents the number of marbles drawn so far. -/
def marbles_drawn : ℕ := 699

/-- Represents the target number of marbles of the same color to stop drawing. -/
def target_same_color : ℕ := 8

/-- Represents the probability of stopping after drawing the 700th marble. -/
def stop_probability : ℚ := 99 / 101

theorem marble_drawing_probability :
  total_marbles = num_colors * marbles_per_color ∧
  marbles_drawn < total_marbles ∧
  marbles_drawn ≥ (num_colors - 1) * (target_same_color - 1) + (target_same_color - 2) →
  stop_probability = 99 / 101 :=
by sorry

end NUMINAMATH_CALUDE_marble_drawing_probability_l2583_258304


namespace NUMINAMATH_CALUDE_suits_sold_is_two_l2583_258316

/-- The number of suits sold given the commission rate, shirt sales, loafer sales, and total commission earned. -/
def suits_sold (commission_rate : ℚ) (num_shirts : ℕ) (shirt_price : ℚ) (num_loafers : ℕ) (loafer_price : ℚ) (suit_price : ℚ) (total_commission : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of suits sold is 2 under the given conditions. -/
theorem suits_sold_is_two :
  suits_sold (15 / 100) 6 50 2 150 700 300 = 2 := by
  sorry

end NUMINAMATH_CALUDE_suits_sold_is_two_l2583_258316


namespace NUMINAMATH_CALUDE_right_triangle_area_15_degree_l2583_258397

theorem right_triangle_area_15_degree (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_acute : Real.cos (15 * π / 180) = b / c) : a * b / 2 = c^2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_15_degree_l2583_258397


namespace NUMINAMATH_CALUDE_max_x2_plus_y2_l2583_258301

theorem max_x2_plus_y2 (x y a : ℝ) (h1 : x + y = a + 1) (h2 : x * y = a^2 - 7*a + 16) :
  ∃ (max : ℝ), max = 32 ∧ ∀ (x' y' a' : ℝ), x' + y' = a' + 1 → x' * y' = a'^2 - 7*a' + 16 → x'^2 + y'^2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x2_plus_y2_l2583_258301


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2583_258310

theorem quadratic_equation_root_zero (k : ℝ) : 
  (k + 3 ≠ 0) →
  (∀ x, (k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0 ↔ x = 0 ∨ x ≠ 0) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l2583_258310


namespace NUMINAMATH_CALUDE_propositions_correctness_l2583_258349

theorem propositions_correctness : 
  -- Proposition ②
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  -- Proposition ③
  (∀ a b : ℝ, a > |b| → a > b) ∧
  -- Proposition ① (negation)
  (∃ a b : ℝ, a > b ∧ (1 / a ≥ 1 / b)) ∧
  -- Proposition ④ (negation)
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry


end NUMINAMATH_CALUDE_propositions_correctness_l2583_258349


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2583_258332

theorem fraction_evaluation : (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2583_258332


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2583_258331

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2583_258331


namespace NUMINAMATH_CALUDE_sqrt_25000_simplified_l2583_258384

theorem sqrt_25000_simplified : Real.sqrt 25000 = 50 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_25000_simplified_l2583_258384


namespace NUMINAMATH_CALUDE_work_remaining_fraction_l2583_258313

theorem work_remaining_fraction 
  (days_a : ℝ) (days_b : ℝ) (days_c : ℝ) (work_days : ℝ) 
  (h1 : days_a = 10) 
  (h2 : days_b = 20) 
  (h3 : days_c = 30) 
  (h4 : work_days = 5) : 
  1 - work_days * (1 / days_a + 1 / days_b + 1 / days_c) = 5 / 60 := by
  sorry

end NUMINAMATH_CALUDE_work_remaining_fraction_l2583_258313


namespace NUMINAMATH_CALUDE_expression_equals_eight_l2583_258351

theorem expression_equals_eight (a : ℝ) (h : a = 2) : 
  (a^3 + (3*a)^3) / (a^2 - a*(3*a) + (3*a)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l2583_258351


namespace NUMINAMATH_CALUDE_number_problem_l2583_258338

theorem number_problem (N : ℝ) :
  (4 / 5) * N = (N / (4 / 5)) - 27 → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2583_258338


namespace NUMINAMATH_CALUDE_horner_method_v3_l2583_258367

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^3 - 2x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 2*x^5 + 3*x^3 - 2*x^2 + x - 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [2, 0, 3, -2, 1, -1]

theorem horner_method_v3 :
  let v₃ := horner (f_coeffs.take 4) 2
  v₃ = 20 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2583_258367


namespace NUMINAMATH_CALUDE_divisiblity_condition_l2583_258380

def recursive_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => (recursive_sequence (n + 1))^2 + recursive_sequence (n + 1) + 1 / recursive_sequence n

theorem divisiblity_condition (a b : ℕ) :
  a > 0 ∧ b > 0 →
  a ∣ (b^2 + b + 1) →
  b ∣ (a^2 + a + 1) →
  ((a = 1 ∧ b = 3) ∨ 
   (∃ n : ℕ, a = recursive_sequence n ∧ b = recursive_sequence (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisiblity_condition_l2583_258380


namespace NUMINAMATH_CALUDE_union_complement_equal_set_l2583_258386

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 5}

theorem union_complement_equal_set : N ∪ (U \ M) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_equal_set_l2583_258386


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l2583_258358

theorem probability_of_two_in_three_elevenths : 
  let decimal_rep := (3 : ℚ) / 11
  let period := 2
  let count_of_two := 1
  (count_of_two : ℚ) / period = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l2583_258358


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2583_258347

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/14) (h2 : x - y = 3/14) : x^2 - y^2 = 27/196 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2583_258347


namespace NUMINAMATH_CALUDE_a_range_l2583_258307

-- Define the function f(x,a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem a_range :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f x a ≥ 0) → a ∈ Set.Icc (-6) (-2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l2583_258307


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l2583_258368

/-- The Euler family children's ages after one year -/
def euler_family_ages : List ℕ := [9, 9, 9, 9, 11, 13, 13]

/-- The number of children in the Euler family -/
def num_children : ℕ := 7

/-- The sum of the Euler family children's ages after one year -/
def sum_ages : ℕ := euler_family_ages.sum

theorem euler_family_mean_age :
  (sum_ages : ℚ) / num_children = 73 / 7 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l2583_258368


namespace NUMINAMATH_CALUDE_stevens_weight_l2583_258375

/-- Given that Danny weighs 40 kg and Steven weighs 20% more than Danny, 
    prove that Steven's weight is 48 kg. -/
theorem stevens_weight (danny_weight : ℝ) (steven_weight : ℝ) 
    (h1 : danny_weight = 40)
    (h2 : steven_weight = danny_weight * 1.2) : 
  steven_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_stevens_weight_l2583_258375


namespace NUMINAMATH_CALUDE_committee_count_l2583_258360

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of students in the group -/
def total_students : ℕ := 8

/-- The number of students in each committee -/
def committee_size : ℕ := 3

/-- The number of different committees that can be formed -/
def num_committees : ℕ := binomial total_students committee_size

theorem committee_count : num_committees = 56 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l2583_258360


namespace NUMINAMATH_CALUDE_consecutive_squares_remainder_l2583_258321

theorem consecutive_squares_remainder (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 ≡ 2 [MOD 3] :=
by sorry

end NUMINAMATH_CALUDE_consecutive_squares_remainder_l2583_258321


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2583_258317

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2583_258317


namespace NUMINAMATH_CALUDE_initial_pencils_l2583_258344

/-- Given that a person:
  - starts with an initial number of pencils
  - gives away 18 pencils
  - buys 22 more pencils
  - ends up with 43 pencils
  This theorem proves that the initial number of pencils was 39. -/
theorem initial_pencils (initial : ℕ) : 
  initial - 18 + 22 = 43 → initial = 39 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_l2583_258344


namespace NUMINAMATH_CALUDE_number_equation_and_interval_l2583_258365

theorem number_equation_and_interval : ∃ (x : ℝ), 
  x = (1 / x) * x^2 + 3 ∧ x = 4 ∧ 3 < x ∧ x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_and_interval_l2583_258365


namespace NUMINAMATH_CALUDE_marbles_in_bag_l2583_258345

theorem marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) : 
  red_marbles = 12 →
  ((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = 9 / 16 →
  total_marbles = 48 := by
sorry

end NUMINAMATH_CALUDE_marbles_in_bag_l2583_258345


namespace NUMINAMATH_CALUDE_no_k_exists_product_minus_one_is_power_l2583_258339

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k - 1 = a^n :=
sorry

end NUMINAMATH_CALUDE_no_k_exists_product_minus_one_is_power_l2583_258339
