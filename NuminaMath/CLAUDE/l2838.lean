import Mathlib

namespace simplify_sqrt_450_l2838_283859

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l2838_283859


namespace president_and_committee_choices_l2838_283866

/-- The number of ways to choose a president and committee from a group --/
def choose_president_and_committee (total_group : ℕ) (senior_members : ℕ) (committee_size : ℕ) : ℕ :=
  let non_senior_members := total_group - senior_members
  let president_choices := non_senior_members
  let remaining_for_committee := total_group - 1
  president_choices * (Nat.choose remaining_for_committee committee_size)

/-- Theorem stating the number of ways to choose a president and committee --/
theorem president_and_committee_choices :
  choose_president_and_committee 10 4 3 = 504 := by
  sorry

end president_and_committee_choices_l2838_283866


namespace intersecting_planes_parallel_line_l2838_283854

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection relation for planes
variable (intersect : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Theorem statement
theorem intersecting_planes_parallel_line 
  (α β : Plane) 
  (h_intersect : intersect α β) :
  ∃ l : Line, parallel l α ∧ parallel l β := by
  sorry

end intersecting_planes_parallel_line_l2838_283854


namespace female_officers_count_l2838_283836

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 200 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 1/10 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_ratio = 1000 := by
  sorry

end female_officers_count_l2838_283836


namespace base_conversion_l2838_283823

/-- Given a base r where 175 in base r equals 125 in base 10, 
    prove that 76 in base r equals 62 in base 10 -/
theorem base_conversion (r : ℕ) (hr : r > 1) : 
  (1 * r^2 + 7 * r + 5 = 125) → (7 * r + 6 = 62) :=
by
  sorry

end base_conversion_l2838_283823


namespace vectors_not_coplanar_l2838_283863

/-- Given three vectors in R³, prove that they are not coplanar. -/
theorem vectors_not_coplanar : 
  let a : Fin 3 → ℝ := ![3, 3, 1]
  let b : Fin 3 → ℝ := ![1, -2, 1]
  let c : Fin 3 → ℝ := ![1, 1, 1]
  ¬ (∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end vectors_not_coplanar_l2838_283863


namespace simplify_complex_square_l2838_283889

theorem simplify_complex_square : 
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end simplify_complex_square_l2838_283889


namespace additional_barking_dogs_l2838_283801

theorem additional_barking_dogs (initial_dogs final_dogs : ℕ) 
  (h1 : initial_dogs = 30)
  (h2 : final_dogs = 40)
  (h3 : initial_dogs < final_dogs) : 
  final_dogs - initial_dogs = 10 := by
sorry

end additional_barking_dogs_l2838_283801


namespace exists_nonprime_between_primes_l2838_283858

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

/-- There exists a natural number n such that n is not prime, but both n-2 and n+2 are prime. -/
theorem exists_nonprime_between_primes : ∃ n : ℕ, 
  ¬ isPrime n ∧ isPrime (n - 2) ∧ isPrime (n + 2) :=
sorry

end exists_nonprime_between_primes_l2838_283858


namespace det2022_2023_2021_2022_solve_det_eq_16_l2838_283877

-- Definition of second-order determinant
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1
theorem det2022_2023_2021_2022 : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2
theorem solve_det_eq_16 (m : ℤ) : det2 (m + 2) (m - 2) (m - 2) (m + 2) = 16 → m = 2 := by sorry

end det2022_2023_2021_2022_solve_det_eq_16_l2838_283877


namespace alternating_series_sum_l2838_283811

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 = 0 then x else -x)) 0

theorem alternating_series_sum :
  let series := arithmetic_series 3 4 18
  alternating_sum series = -36 := by sorry

end alternating_series_sum_l2838_283811


namespace workshop_equation_system_l2838_283887

/-- Represents the production capabilities of workers for desks and chairs -/
structure ProductionRate where
  desk : ℕ
  chair : ℕ

/-- Represents the composition of a set of furniture -/
structure FurnitureSet where
  desk : ℕ
  chair : ℕ

/-- The problem setup for the furniture workshop -/
structure WorkshopSetup where
  totalWorkers : ℕ
  productionRate : ProductionRate
  furnitureSet : FurnitureSet

/-- Theorem stating the correct system of equations for the workshop problem -/
theorem workshop_equation_system 
  (setup : WorkshopSetup)
  (h_setup : setup.totalWorkers = 32 ∧ 
             setup.productionRate = { desk := 5, chair := 6 } ∧
             setup.furnitureSet = { desk := 1, chair := 2 }) :
  ∃ (x y : ℕ), 
    x + y = setup.totalWorkers ∧ 
    2 * (setup.productionRate.desk * x) = setup.productionRate.chair * y :=
sorry

end workshop_equation_system_l2838_283887


namespace joan_lost_balloons_l2838_283886

theorem joan_lost_balloons (initial_balloons current_balloons : ℕ) 
  (h1 : initial_balloons = 9)
  (h2 : current_balloons = 7) : 
  initial_balloons - current_balloons = 2 := by
  sorry

end joan_lost_balloons_l2838_283886


namespace system_solution_unique_l2838_283833

theorem system_solution_unique (x y : ℝ) : 
  (5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15) ↔ (x = 5 ∧ y = 0) := by
  sorry

end system_solution_unique_l2838_283833


namespace tank_capacity_l2838_283876

theorem tank_capacity : ∀ (initial_fraction final_fraction added_volume capacity : ℚ),
  initial_fraction = 1 / 4 →
  final_fraction = 3 / 4 →
  added_volume = 160 →
  (final_fraction - initial_fraction) * capacity = added_volume →
  capacity = 320 := by
  sorry

end tank_capacity_l2838_283876


namespace profit_decrease_l2838_283865

theorem profit_decrease (march_profit : ℝ) (april_may_decrease : ℝ) : 
  (1 + 0.35) * (1 - april_may_decrease / 100) * (1 + 0.5) = 1.62000000000000014 →
  april_may_decrease = 20 := by
sorry

end profit_decrease_l2838_283865


namespace rotated_semicircle_area_l2838_283844

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45° -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let rotation_angle : ℝ := 45 * π / 180
  let shaded_area := (2 * R)^2 * rotation_angle / 2
  shaded_area = π * R^2 / 2 := by
  sorry


end rotated_semicircle_area_l2838_283844


namespace fish_in_large_aquarium_l2838_283884

def fish_redistribution (initial_fish : ℕ) (additional_fish : ℕ) (small_aquarium_capacity : ℕ) : ℕ :=
  let total_fish := initial_fish + additional_fish
  total_fish - small_aquarium_capacity

theorem fish_in_large_aquarium :
  fish_redistribution 125 250 150 = 225 :=
by sorry

end fish_in_large_aquarium_l2838_283884


namespace quadratic_roots_theorem_l2838_283867

theorem quadratic_roots_theorem (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (p + 3 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (p + 3 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  (q + 6 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (q + 6 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  p = 9.5 ∧ q = 6.5 := by
sorry


end quadratic_roots_theorem_l2838_283867


namespace calculate_loss_percentage_l2838_283864

/-- Calculates the percentage of loss given the selling prices and profit percentage --/
theorem calculate_loss_percentage
  (sp_profit : ℝ)       -- Selling price with profit
  (profit_percent : ℝ)  -- Profit percentage
  (sp_loss : ℝ)         -- Selling price with loss
  (h1 : sp_profit = 800)
  (h2 : profit_percent = 25)
  (h3 : sp_loss = 512)
  : (sp_profit * (100 / (100 + profit_percent)) - sp_loss) / 
    (sp_profit * (100 / (100 + profit_percent))) * 100 = 20 := by
  sorry

end calculate_loss_percentage_l2838_283864


namespace virginia_april_rainfall_l2838_283819

/-- Calculates the rainfall in April given the rainfall in other months and the average -/
def april_rainfall (march may june july average : ℝ) : ℝ :=
  5 * average - (march + may + june + july)

/-- Theorem stating that given the specified rainfall amounts and average, April's rainfall was 4.5 inches -/
theorem virginia_april_rainfall :
  let march : ℝ := 3.79
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  april_rainfall march may june july average = 4.5 := by
  sorry

end virginia_april_rainfall_l2838_283819


namespace cement_mixture_percentage_l2838_283814

/-- Proves that in a concrete mixture, given specific conditions, the remaining mixture must be 20% cement. -/
theorem cement_mixture_percentage
  (total_concrete : ℝ)
  (final_cement_percentage : ℝ)
  (high_cement_mixture_amount : ℝ)
  (high_cement_percentage : ℝ)
  (h1 : total_concrete = 10)
  (h2 : final_cement_percentage = 0.62)
  (h3 : high_cement_mixture_amount = 7)
  (h4 : high_cement_percentage = 0.8)
  : (total_concrete * final_cement_percentage - high_cement_mixture_amount * high_cement_percentage) / (total_concrete - high_cement_mixture_amount) = 0.2 := by
  sorry

end cement_mixture_percentage_l2838_283814


namespace circle_equation_from_center_and_point_specific_circle_equation_l2838_283839

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of a circle given its center and a point on the circle -/
theorem circle_equation_from_center_and_point 
  (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  ∃ (c : Circle), 
    c.center = center ∧ 
    lies_on_circle c point ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ (x - center.1)^2 + (y - center.2)^2 = c.radius^2 := by
  sorry

/-- The specific circle equation for the given problem -/
theorem specific_circle_equation : 
  ∃ (c : Circle), 
    c.center = (0, 4) ∧ 
    lies_on_circle c (3, 0) ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ x^2 + (y - 4)^2 = 25 := by
  sorry

end circle_equation_from_center_and_point_specific_circle_equation_l2838_283839


namespace equivalent_operation_l2838_283809

theorem equivalent_operation (x : ℚ) : x * (4/5) / (4/7) = x * (7/5) := by
  sorry

end equivalent_operation_l2838_283809


namespace sum_product_difference_l2838_283829

theorem sum_product_difference (x y : ℝ) : 
  x + y = 24 → x * y = 23 → |x - y| = 22 := by
sorry

end sum_product_difference_l2838_283829


namespace crude_oil_temperature_l2838_283807

-- Define the function f(x) = x^2 - 7x + 15 on the interval [0, 8]
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) : 
  -- The derivative of f at x = 4 is 1
  (deriv f) 4 = 1 ∧ 
  -- The function is increasing at x = 4
  (deriv f) 4 > 0 := by
  sorry

end crude_oil_temperature_l2838_283807


namespace max_product_is_18000_l2838_283861

def numbers : List ℕ := [10, 15, 20, 30, 40, 60]

def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 6 ∧ 
  arrangement.toFinset = numbers.toFinset ∧
  ∃ (product : ℕ), 
    (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
    (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
    (arrangement[2]! * arrangement[4]! * arrangement[5]! = product)

theorem max_product_is_18000 :
  ∀ (arrangement : List ℕ), is_valid_arrangement arrangement →
    ∃ (product : ℕ), 
      (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
      (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
      (arrangement[2]! * arrangement[4]! * arrangement[5]! = product) ∧
      product ≤ 18000 :=
by sorry

end max_product_is_18000_l2838_283861


namespace parallel_lines_k_value_l2838_283843

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l2838_283843


namespace custom_mult_equation_solution_l2838_283855

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * (3x - 1) = 27 under the custom multiplication, then x = 7/3 -/
theorem custom_mult_equation_solution :
  ∀ x : ℝ, custom_mult 3 (3 * x - 1) = 27 → x = 7/3 := by
  sorry

end custom_mult_equation_solution_l2838_283855


namespace kara_water_consumption_l2838_283888

/-- The amount of water Kara drinks with each dose of medication -/
def water_per_dose : ℕ := 4

/-- The number of doses Kara takes per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Kara followed her medication routine -/
def total_weeks : ℕ := 2

/-- The number of doses Kara forgot in the second week -/
def forgotten_doses : ℕ := 2

/-- Calculates the total amount of water Kara drank with her medication over two weeks -/
def total_water_consumption : ℕ :=
  water_per_dose * doses_per_day * days_per_week * total_weeks - water_per_dose * forgotten_doses

theorem kara_water_consumption :
  total_water_consumption = 160 := by
  sorry

end kara_water_consumption_l2838_283888


namespace base2_to_base4_conversion_l2838_283896

/-- Converts a natural number from base 2 to base 10 --/
def base2ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a natural number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number --/
def base2Number : ℕ := 101101100

/-- The expected base 4 representation of the number --/
def expectedBase4Number : ℕ := 23110

theorem base2_to_base4_conversion :
  base10ToBase4 (base2ToBase10 base2Number) = expectedBase4Number := by sorry

end base2_to_base4_conversion_l2838_283896


namespace z_range_l2838_283802

-- Define the region (as we don't have specific inequalities, we'll use a general set)
variable (R : Set (ℝ × ℝ))

-- Define the function z = x - y
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- State the theorem
theorem z_range (h : Set.Nonempty R) :
  Set.Icc (-1 : ℝ) 2 = {t | ∃ p ∈ R, z p = t} := by sorry

end z_range_l2838_283802


namespace remainder_sum_modulo_l2838_283890

theorem remainder_sum_modulo (x y : ℤ) 
  (hx : x % 126 = 37) 
  (hy : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
sorry

end remainder_sum_modulo_l2838_283890


namespace molecular_weight_AlBr3_10_moles_value_l2838_283885

/-- The molecular weight of 10 moles of AlBr3 -/
def molecular_weight_AlBr3_10_moles : ℝ :=
  let atomic_weight_Al : ℝ := 26.98
  let atomic_weight_Br : ℝ := 79.90
  let molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br
  10 * molecular_weight_AlBr3

/-- Theorem stating that the molecular weight of 10 moles of AlBr3 is 2666.8 grams -/
theorem molecular_weight_AlBr3_10_moles_value :
  molecular_weight_AlBr3_10_moles = 2666.8 := by sorry

end molecular_weight_AlBr3_10_moles_value_l2838_283885


namespace cake_sharing_l2838_283822

theorem cake_sharing (n : ℕ) : 
  (∃ (shares : Fin n → ℚ), 
    (∀ i, 0 < shares i) ∧ 
    (∃ j, shares j = 1/11) ∧
    (∃ k, shares k = 1/14) ∧
    (∀ i, 1/14 ≤ shares i ∧ shares i ≤ 1/11) ∧
    (Finset.sum Finset.univ shares = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
by sorry


end cake_sharing_l2838_283822


namespace john_got_36_rolls_l2838_283808

/-- The number of rolls John got given the price and amount spent -/
def rolls_bought (price_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / price_per_dozen) * 12

/-- Theorem: John got 36 rolls -/
theorem john_got_36_rolls :
  rolls_bought 5 15 = 36 := by
  sorry

end john_got_36_rolls_l2838_283808


namespace bucket_capacity_reduction_l2838_283848

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets = 63 ∧ 
    (↑new_buckets : ℚ) * capacity_ratio ≥ ↑original_buckets ∧
    (↑new_buckets - 1 : ℚ) * capacity_ratio < ↑original_buckets :=
by sorry

end bucket_capacity_reduction_l2838_283848


namespace geometric_sequence_ratio_l2838_283883

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 6 / a 3)^(1/3)) →  -- Geometric sequence property
  a 3 = 2 →                                     -- Given condition
  a 6 = 16 →                                    -- Given condition
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 2 := by
sorry

end geometric_sequence_ratio_l2838_283883


namespace last_two_digits_of_7_power_l2838_283869

def last_two_digits (n : ℕ) : ℕ := n % 100

def periodic_sequence (a : ℕ → ℕ) (period : ℕ) :=
  ∀ n : ℕ, a (n + period) = a n

theorem last_two_digits_of_7_power (n : ℕ) (h : n ≥ 2) :
  periodic_sequence (λ k => last_two_digits (7^k)) 4 →
  last_two_digits (7^2017) = last_two_digits (7^5) :=
by
  sorry

#eval last_two_digits (7^5)  -- Should output 7

end last_two_digits_of_7_power_l2838_283869


namespace rectangle_perimeter_l2838_283841

/-- The perimeter of a rectangle with longer sides 28cm and shorter sides 22cm is 100cm -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 28, 22 => 100
  | _, _ => 0

#check rectangle_perimeter

end rectangle_perimeter_l2838_283841


namespace oil_measurement_l2838_283874

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) :
  initial_oil = 17/100 → added_oil = 67/100 → initial_oil + added_oil = 84/100 := by
  sorry

end oil_measurement_l2838_283874


namespace max_gcd_sum_1025_l2838_283821

theorem max_gcd_sum_1025 : 
  ∃ (max : ℕ), max > 0 ∧ 
  (∀ a b : ℕ, a > 0 → b > 0 → a + b = 1025 → Nat.gcd a b ≤ max) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b = 1025 ∧ Nat.gcd a b = max) ∧
  max = 205 :=
by sorry

end max_gcd_sum_1025_l2838_283821


namespace perfect_square_function_characterization_l2838_283825

theorem perfect_square_function_characterization 
  (g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end perfect_square_function_characterization_l2838_283825


namespace total_wheels_in_parking_lot_l2838_283820

/-- Calculates the total number of wheels in a parking lot with cars and motorcycles -/
theorem total_wheels_in_parking_lot 
  (num_cars : ℕ) 
  (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) 
  (wheels_per_motorcycle : ℕ) 
  (h1 : num_cars = 19) 
  (h2 : num_motorcycles = 11) 
  (h3 : wheels_per_car = 5) 
  (h4 : wheels_per_motorcycle = 2) : 
  num_cars * wheels_per_car + num_motorcycles * wheels_per_motorcycle = 117 := by
sorry

end total_wheels_in_parking_lot_l2838_283820


namespace no_8002_integers_divisibility_property_l2838_283852

theorem no_8002_integers_divisibility_property (P : ℕ → ℕ) 
  (h_P : ∀ x : ℕ, P x = x^2000 - x^1000 + 1) : 
  ¬ ∃ (a : Fin 8002 → ℕ), 
    (∀ i j : Fin 8002, i ≠ j → a i ≠ a j) ∧ 
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → 
      (a i * a j * a k) ∣ (P (a i) * P (a j) * P (a k))) :=
sorry

end no_8002_integers_divisibility_property_l2838_283852


namespace trig_problem_l2838_283803

theorem trig_problem (x : ℝ) (h : Real.sin (x + π/6) = 1/3) :
  Real.sin (5*π/6 - x) - (Real.sin (π/3 - x))^2 = -5/9 := by
  sorry

end trig_problem_l2838_283803


namespace no_real_roots_l2838_283845

theorem no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

end no_real_roots_l2838_283845


namespace solve_for_P_l2838_283857

theorem solve_for_P : ∃ P : ℝ, (P^4)^(1/3) = 9 * 81^(1/9) → P = 3^(11/6) := by
  sorry

end solve_for_P_l2838_283857


namespace probability_4H_before_3T_value_l2838_283850

/-- The probability of getting 4 heads before 3 tails in repeated fair coin flips -/
def probability_4H_before_3T : ℚ :=
  13 / 17

/-- Theorem stating that the probability of getting 4 heads before 3 tails
    in repeated fair coin flips is 13/17 -/
theorem probability_4H_before_3T_value :
  probability_4H_before_3T = 13 / 17 := by
  sorry

#eval Nat.gcd 13 17  -- To verify that 13 and 17 are coprime

end probability_4H_before_3T_value_l2838_283850


namespace card_exchange_probability_l2838_283846

def number_of_people : ℕ := 4

def probability_B_drew_A_given_A_drew_B : ℚ :=
  1 / 3

theorem card_exchange_probability :
  ∀ (A B : Fin number_of_people),
  A ≠ B →
  (probability_B_drew_A_given_A_drew_B : ℚ) =
    (1 : ℚ) / (number_of_people - 1 : ℚ) :=
by sorry

end card_exchange_probability_l2838_283846


namespace a_minus_b_equals_four_l2838_283895

theorem a_minus_b_equals_four :
  ∀ (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) →  -- A is a two-digit number
    (B ≥ 10 ∧ B ≤ 99) →  -- B is a two-digit number
    A = 23 - 8 →         -- A is 8 less than 23
    B + 7 = 18 →         -- The number that is 7 greater than B is 18
    A - B = 4 :=
by
  sorry

end a_minus_b_equals_four_l2838_283895


namespace algebraic_expressions_simplification_l2838_283891

theorem algebraic_expressions_simplification (x y m a b c : ℝ) :
  (4 * y * (-2 * x * y^2) = -8 * x * y^3) ∧
  ((-5/2 * x^2) * (-4 * x) = 10 * x^3) ∧
  ((3 * m^2) * (-2 * m^3)^2 = 12 * m^8) ∧
  ((-a * b^2 * c^3)^2 * (-a^2 * b)^3 = -a^8 * b^7 * c^6) := by
sorry


end algebraic_expressions_simplification_l2838_283891


namespace ducks_in_marsh_l2838_283828

/-- Given a marsh with geese and ducks, calculate the number of ducks -/
theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l2838_283828


namespace nancy_carrots_count_l2838_283892

/-- The number of carrots Nancy's mother picked -/
def mother_carrots : ℕ := 47

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy picked -/
def nancy_carrots : ℕ := 38

theorem nancy_carrots_count : 
  nancy_carrots = (good_carrots + bad_carrots) - mother_carrots := by
  sorry

end nancy_carrots_count_l2838_283892


namespace no_common_elements_except_one_l2838_283868

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + 2 * sequence_a n

def sequence_b : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * sequence_b (n + 1) + 3 * sequence_b n

theorem no_common_elements_except_one :
  ∀ n : ℕ, n > 0 → sequence_a n ≠ sequence_b n :=
by sorry

end no_common_elements_except_one_l2838_283868


namespace geometric_arithmetic_sequence_sum_l2838_283878

theorem geometric_arithmetic_sequence_sum : 
  ∃ (x y : ℝ), 3 < x ∧ x < y ∧ y < 9 ∧ 
  (x^2 = 3*y) ∧ (2*y = x + 9) ∧ 
  (x + y = 11.25) := by
  sorry

end geometric_arithmetic_sequence_sum_l2838_283878


namespace halloween_candy_problem_l2838_283824

/-- Given the initial candy counts for Katie and her sister, and the number of pieces eaten,
    calculate the remaining candy count. -/
def remaining_candy (katie_candy : ℕ) (sister_candy : ℕ) (eaten : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten

/-- Theorem stating that for the given problem, the remaining candy count is 7. -/
theorem halloween_candy_problem :
  remaining_candy 10 6 9 = 7 := by
  sorry

end halloween_candy_problem_l2838_283824


namespace min_value_theorem_l2838_283812

/-- Given real numbers a, b, c, d satisfying the equation,
    the minimum value of the expression is 8 -/
theorem min_value_theorem (a b c d : ℝ) 
  (h : (b - 2*a^2 + 3*Real.log a)^2 + (c - d - 3)^2 = 0) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y z w : ℝ), 
    (y - 2*x^2 + 3*Real.log x)^2 + (z - w - 3)^2 = 0 → 
    (x - z)^2 + (y - w)^2 ≥ m :=
by sorry

end min_value_theorem_l2838_283812


namespace tickets_to_buy_l2838_283817

/-- The number of additional tickets Zach needs to buy for three rides -/
theorem tickets_to_buy (ferris_wheel_cost roller_coaster_cost log_ride_cost current_tickets : ℕ) 
  (h1 : ferris_wheel_cost = 2)
  (h2 : roller_coaster_cost = 7)
  (h3 : log_ride_cost = 1)
  (h4 : current_tickets = 1) :
  ferris_wheel_cost + roller_coaster_cost + log_ride_cost - current_tickets = 9 := by
  sorry

end tickets_to_buy_l2838_283817


namespace fraction_cube_equality_l2838_283835

theorem fraction_cube_equality : 
  (81000 : ℝ)^3 / (27000 : ℝ)^3 = 27 :=
by
  have h : (81000 : ℝ) = 3 * 27000 := by norm_num
  sorry

end fraction_cube_equality_l2838_283835


namespace no_solution_absolute_value_equation_l2838_283826

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 3)^2 ≠ 0 → False) ∧
  (∀ x : ℝ, |2*x| + 4 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (3*x) - 1 ≠ 0 → False) ∧
  (∀ x : ℝ, x ≤ 0 → Real.sqrt (-3*x) - 3 ≠ 0 → False) ∧
  (∀ x : ℝ, |5*x| - 6 ≠ 0 → False) := by
  sorry

#check no_solution_absolute_value_equation

end no_solution_absolute_value_equation_l2838_283826


namespace ratio_proof_l2838_283870

theorem ratio_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (hx : x = 1.25 * a) (hm : m = 0.2 * b) (hm_x : m / x = 0.2) : 
  a / b = 1.25 := by
  sorry

end ratio_proof_l2838_283870


namespace f_composite_negative_two_l2838_283818

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composite_negative_two :
  f (f (-2)) = 1/2 := by
  sorry

end f_composite_negative_two_l2838_283818


namespace line_circle_intersection_l2838_283853

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), (y = k * x + 1 ∧ x^2 + y^2 = 2) ∧ 
  ¬(∃ (x y : ℝ), y = k * x + 1 ∧ x = 0 ∧ y = 0) := by
  sorry

end line_circle_intersection_l2838_283853


namespace equation_solution_l2838_283897

theorem equation_solution (x : ℝ) : 
  (x^2 + x - 2)^3 + (2*x^2 - x - 1)^3 = 27*(x^2 - 1)^3 ↔ 
  x = 1 ∨ x = -1 ∨ x = -2 ∨ x = -1/2 :=
by sorry

end equation_solution_l2838_283897


namespace largest_integer_satisfying_inequality_l2838_283813

theorem largest_integer_satisfying_inequality :
  ∃ (x : ℤ), (3 * |2 * x + 1| - 5 < 22) ∧
  (∀ (y : ℤ), y > x → ¬(3 * |2 * y + 1| - 5 < 22)) ∧
  x = 3 := by
  sorry

end largest_integer_satisfying_inequality_l2838_283813


namespace safari_leopards_l2838_283838

theorem safari_leopards (total_animals : ℕ) 
  (saturday_lions sunday_buffaloes monday_rhinos : ℕ)
  (saturday_elephants monday_warthogs : ℕ) :
  total_animals = 20 →
  saturday_lions = 3 →
  saturday_elephants = 2 →
  sunday_buffaloes = 2 →
  monday_rhinos = 5 →
  monday_warthogs = 3 →
  ∃ (sunday_leopards : ℕ),
    total_animals = 
      saturday_lions + saturday_elephants + 
      sunday_buffaloes + sunday_leopards +
      monday_rhinos + monday_warthogs ∧
    sunday_leopards = 5 := by
  sorry

end safari_leopards_l2838_283838


namespace lcm_from_product_and_hcf_l2838_283882

theorem lcm_from_product_and_hcf (a b : ℕ+) : 
  a * b = 18000 → Nat.gcd a b = 30 → Nat.lcm a b = 600 := by
  sorry

end lcm_from_product_and_hcf_l2838_283882


namespace proposition_truth_l2838_283806

theorem proposition_truth (p q : Prop) (hp : ¬p) (hq : ¬q) :
  (p ∨ ¬q) ∧ ¬(p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∧ q) := by
  sorry

end proposition_truth_l2838_283806


namespace zero_score_students_l2838_283800

theorem zero_score_students (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (rest_average : ℚ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  high_score = 95 →
  rest_average = 45 →
  class_average = 47.32142857142857 →
  ∃ (zero_scorers : ℕ),
    zero_scorers = 3 ∧
    (high_scorers * high_score + zero_scorers * 0 + 
     (total_students - high_scorers - zero_scorers) * rest_average) / total_students = class_average :=
by sorry

end zero_score_students_l2838_283800


namespace staircase_step_difference_l2838_283873

/-- Theorem: Difference in steps between second and third staircases --/
theorem staircase_step_difference :
  ∀ (steps1 steps2 steps3 : ℕ) (step_height : ℚ) (total_height : ℚ),
    steps1 = 20 →
    steps2 = 2 * steps1 →
    step_height = 1/2 →
    total_height = 45 →
    (steps1 + steps2 + steps3 : ℚ) * step_height = total_height →
    steps2 - steps3 = 10 :=
by
  sorry

end staircase_step_difference_l2838_283873


namespace solution_a_solution_b_l2838_283832

-- Part (a)
theorem solution_a (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a ≠ b) :
  let x := (2 * a * b) / (a + b)
  (x + a) / (x - a) + (x + b) / (x - b) = 2 :=
sorry

-- Part (b)
theorem solution_b (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) (h5 : a * b + c ≠ 0) :
  let x := (a * b * c) / d
  c * (d / (a * b) - a * b / x) + d = c^2 / x :=
sorry

end solution_a_solution_b_l2838_283832


namespace find_k_l2838_283894

theorem find_k (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end find_k_l2838_283894


namespace square_pentagon_alignment_l2838_283830

/-- The number of sides in a square -/
def squareSides : ℕ := 4

/-- The number of sides in a regular pentagon -/
def pentagonSides : ℕ := 5

/-- The least common multiple of the number of sides of a square and a regular pentagon -/
def lcmSides : ℕ := Nat.lcm squareSides pentagonSides

/-- The minimum number of full rotations required for a square to align with a regular pentagon -/
def minRotations : ℕ := lcmSides / squareSides

theorem square_pentagon_alignment :
  minRotations = 5 := by
  sorry

end square_pentagon_alignment_l2838_283830


namespace f_extrema_on_interval_l2838_283893

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 27

-- State the theorem
theorem f_extrema_on_interval :
  (∀ x ∈ Set.Icc 0 3, f x ≤ 54) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 54) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ 27/4) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 27/4) :=
sorry

end f_extrema_on_interval_l2838_283893


namespace henrys_shells_l2838_283810

theorem henrys_shells (perfect_shells : ℕ) (non_spiral_perfect : ℕ) (broken_spiral_diff : ℕ) :
  perfect_shells = 17 →
  non_spiral_perfect = 12 →
  broken_spiral_diff = 21 →
  (perfect_shells - non_spiral_perfect + broken_spiral_diff) * 2 = 52 := by
sorry

end henrys_shells_l2838_283810


namespace product_of_roots_l2838_283840

theorem product_of_roots : (32 : ℝ) ^ (1/5 : ℝ) * (128 : ℝ) ^ (1/7 : ℝ) = 4 := by
  sorry

end product_of_roots_l2838_283840


namespace apple_delivery_problem_l2838_283860

theorem apple_delivery_problem (first_grade_value second_grade_value : ℝ)
  (price_difference : ℝ) (quantity_difference : ℝ) :
  first_grade_value = 228 →
  second_grade_value = 180 →
  price_difference = 0.9 →
  quantity_difference = 5 →
  ∃ x : ℝ,
    x > 0 ∧
    x + quantity_difference > 0 ∧
    (first_grade_value / x - price_difference) * (2 * x + quantity_difference) =
      first_grade_value + second_grade_value ∧
    2 * x + quantity_difference = 85 :=
by sorry

end apple_delivery_problem_l2838_283860


namespace power_of_64_l2838_283871

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 :=
by
  have h : 64 = 2^6 := by sorry
  sorry

end power_of_64_l2838_283871


namespace pencils_per_pack_l2838_283880

/-- Given information about Faye's pencils, prove the number of pencils in each pack -/
theorem pencils_per_pack 
  (total_packs : ℕ) 
  (pencils_per_row : ℕ) 
  (total_rows : ℕ) 
  (h1 : total_packs = 28) 
  (h2 : pencils_per_row = 16) 
  (h3 : total_rows = 42) : 
  (total_rows * pencils_per_row) / total_packs = 24 := by
  sorry

end pencils_per_pack_l2838_283880


namespace meaningful_fraction_l2838_283827

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 := by
sorry

end meaningful_fraction_l2838_283827


namespace tribal_leadership_theorem_l2838_283842

def tribal_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2

theorem tribal_leadership_theorem :
  tribal_leadership_arrangements 13 = 18604800 := by
  sorry

end tribal_leadership_theorem_l2838_283842


namespace exists_number_divisible_by_2_100_without_zero_l2838_283856

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ k : ℕ, (n / (10^k)) % 10 = 0

/-- Theorem stating that there exists an integer divisible by 2^100 that does not contain the digit 0 -/
theorem exists_number_divisible_by_2_100_without_zero :
  ∃ n : ℕ, (n % (2^100) = 0) ∧ ¬(containsZero n) := by
  sorry

end exists_number_divisible_by_2_100_without_zero_l2838_283856


namespace arithmetic_series_sum_l2838_283875

theorem arithmetic_series_sum (t : ℝ) : 
  let first_term := t^2 + 3
  let num_terms := 3*t + 2
  let common_difference := 1
  let last_term := first_term + (num_terms - 1) * common_difference
  (num_terms / 2) * (first_term + last_term) = (3*t + 2) * (t^2 + 1.5*t + 3.5) :=
by sorry

end arithmetic_series_sum_l2838_283875


namespace quadratic_system_sum_l2838_283862

theorem quadratic_system_sum (x y r₁ s₁ r₂ s₂ : ℝ) : 
  (9 * x^2 - 27 * x - 54 = 0) →
  (4 * y^2 + 28 * y + 49 = 0) →
  ((x - r₁)^2 = s₁) →
  ((y - r₂)^2 = s₂) →
  (r₁ + s₁ + r₂ + s₂ = -11/4) := by
sorry

end quadratic_system_sum_l2838_283862


namespace fifth_term_is_sixteen_l2838_283898

/-- A geometric sequence with first term 1 and a_2 * a_4 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 2 * a 4 = 16

theorem fifth_term_is_sixteen 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 5 = 16 := by
sorry

end fifth_term_is_sixteen_l2838_283898


namespace floor_sqrt_120_l2838_283879

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end floor_sqrt_120_l2838_283879


namespace equation_solution_l2838_283851

theorem equation_solution : ∃ x : ℝ, (2*x - 1)^2 - (1 - 3*x)^2 = 5*(1 - x)*(x + 1) ∧ x = 5/2 := by
  sorry

end equation_solution_l2838_283851


namespace mason_car_nuts_l2838_283872

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nuts_in_car (busy_squirrels : ℕ) (busy_nuts_per_day : ℕ) (sleepy_squirrels : ℕ) (sleepy_nuts_per_day : ℕ) (days : ℕ) : ℕ :=
  (busy_squirrels * busy_nuts_per_day + sleepy_squirrels * sleepy_nuts_per_day) * days

/-- Theorem stating the number of nuts in Mason's car -/
theorem mason_car_nuts :
  nuts_in_car 2 30 1 20 40 = 3200 :=
by sorry

end mason_car_nuts_l2838_283872


namespace ian_money_left_l2838_283849

def hourly_rate : ℝ := 18
def hours_worked : ℝ := 8
def spending_ratio : ℝ := 0.5

def total_earnings : ℝ := hourly_rate * hours_worked
def amount_spent : ℝ := total_earnings * spending_ratio
def amount_left : ℝ := total_earnings - amount_spent

theorem ian_money_left : amount_left = 72 := by
  sorry

end ian_money_left_l2838_283849


namespace dodecagon_diagonal_intersection_probability_l2838_283805

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- Add any necessary properties here

/-- Represents a diagonal in the dodecagon -/
structure Diagonal where
  -- Add any necessary properties here

/-- The probability that two randomly chosen diagonals intersect inside a regular dodecagon -/
def intersection_probability (d : RegularDodecagon) : ℚ :=
  495 / 1431

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a regular dodecagon is 495/1431 -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersection_probability d = 495 / 1431 := by
  sorry


end dodecagon_diagonal_intersection_probability_l2838_283805


namespace incorrect_roots_correct_roots_l2838_283881

-- Define the original quadratic equation
def original_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_eq x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := y = x^2 ∧ y = 3*x - 2
def pair_B (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 0
def pair_C (x y : ℝ) : Prop := y = x ∧ y = Real.sqrt (x + 2)
def pair_D (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 2
def pair_E (x y : ℝ) : Prop := y = Real.sin x ∧ y = 3*x - 4

-- Theorem stating that (C), (D), and (E) do not yield the correct roots
theorem incorrect_roots :
  (∃ x y : ℝ, pair_C x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_D x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_E x y ∧ ¬(is_root x)) :=
sorry

-- Theorem stating that (A) and (B) yield the correct roots
theorem correct_roots :
  (∀ x y : ℝ, pair_A x y → is_root x) ∧
  (∀ x y : ℝ, pair_B x y → is_root x) :=
sorry

end incorrect_roots_correct_roots_l2838_283881


namespace inequality_interval_length_l2838_283834

theorem inequality_interval_length (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) ∧ 
  ((d - 5) / 3 - (c - 5) / 3 = 12) → 
  d - c = 36 := by
sorry

end inequality_interval_length_l2838_283834


namespace game_result_depends_only_on_blue_parity_l2838_283815

/-- Represents the color of a sprite -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the game -/
structure GameState :=
  (red : Nat)   -- Number of red sprites
  (blue : Nat)  -- Number of blue sprites

/-- Represents the result of the game -/
def gameResult (initialState : GameState) : Color :=
  if initialState.blue % 2 = 1 then Color.Blue else Color.Red

/-- The main theorem stating that the game result only depends on the initial number of blue sprites -/
theorem game_result_depends_only_on_blue_parity (m n : Nat) :
  gameResult { red := m, blue := n } = 
  if n % 2 = 1 then Color.Blue else Color.Red :=
by sorry

end game_result_depends_only_on_blue_parity_l2838_283815


namespace color_paint_can_size_is_one_gallon_l2838_283816

/-- Represents the paint job for a house --/
structure PaintJob where
  bedrooms : Nat
  otherRooms : Nat
  gallonsPerRoom : Nat
  whitePaintCanSize : Nat
  totalCans : Nat

/-- Calculates the size of each can of color paint --/
def colorPaintCanSize (job : PaintJob) : Rat :=
  let totalRooms := job.bedrooms + job.otherRooms
  let totalPaint := totalRooms * job.gallonsPerRoom
  let whitePaint := job.otherRooms * job.gallonsPerRoom
  let whiteCans := whitePaint / job.whitePaintCanSize
  let colorCans := job.totalCans - whiteCans
  let colorPaint := job.bedrooms * job.gallonsPerRoom
  colorPaint / colorCans

/-- Theorem stating that the size of each can of color paint is 1 gallon --/
theorem color_paint_can_size_is_one_gallon (job : PaintJob)
  (h1 : job.bedrooms = 3)
  (h2 : job.otherRooms = 2 * job.bedrooms)
  (h3 : job.gallonsPerRoom = 2)
  (h4 : job.whitePaintCanSize = 3)
  (h5 : job.totalCans = 10) :
  colorPaintCanSize job = 1 := by
  sorry

#eval colorPaintCanSize { bedrooms := 3, otherRooms := 6, gallonsPerRoom := 2, whitePaintCanSize := 3, totalCans := 10 }

end color_paint_can_size_is_one_gallon_l2838_283816


namespace election_winner_votes_l2838_283847

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 65 / 100 →
  vote_difference = 300 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 650 :=
by
  sorry

end election_winner_votes_l2838_283847


namespace house_sale_price_l2838_283831

theorem house_sale_price (initial_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  initial_price = 100000 ∧ profit_percent = 10 ∧ loss_percent = 10 →
  initial_price * (1 + profit_percent / 100) * (1 - loss_percent / 100) = 99000 := by
sorry

end house_sale_price_l2838_283831


namespace sum_of_a_and_b_is_seven_l2838_283837

theorem sum_of_a_and_b_is_seven (A B : Set ℕ) (a b : ℕ) : 
  A = {1, 2} →
  B = {2, a, b} →
  A ∪ B = {1, 2, 3, 4} →
  a + b = 7 := by
sorry

end sum_of_a_and_b_is_seven_l2838_283837


namespace two_digit_number_relationship_l2838_283804

theorem two_digit_number_relationship :
  ∀ (tens units : ℕ),
    tens * 10 + units = 16 →
    tens + units = 7 →
    ∃ (k : ℕ), units = k * tens →
    units = 6 * tens :=
by sorry

end two_digit_number_relationship_l2838_283804


namespace partial_fraction_decomposition_l2838_283899

theorem partial_fraction_decomposition :
  let P : ℚ := 17/7
  let Q : ℚ := 4/7
  ∀ x : ℚ, x ≠ 10 → x ≠ -4 →
    (3*x + 4) / (x^2 - 6*x - 40) = P / (x - 10) + Q / (x + 4) :=
by sorry

end partial_fraction_decomposition_l2838_283899
