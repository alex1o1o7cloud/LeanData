import Mathlib

namespace NUMINAMATH_CALUDE_jamies_flyer_delivery_l416_41627

/-- Jamie's flyer delivery problem -/
theorem jamies_flyer_delivery 
  (hourly_rate : ℝ) 
  (hours_per_delivery : ℝ) 
  (total_weeks : ℕ) 
  (total_earnings : ℝ) 
  (h1 : hourly_rate = 10)
  (h2 : hours_per_delivery = 3)
  (h3 : total_weeks = 6)
  (h4 : total_earnings = 360) : 
  (total_earnings / hourly_rate / total_weeks / hours_per_delivery : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jamies_flyer_delivery_l416_41627


namespace NUMINAMATH_CALUDE_floor_times_self_equals_45_l416_41608

theorem floor_times_self_equals_45 (y : ℝ) (h1 : y > 0) (h2 : ⌊y⌋ * y = 45) : y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_45_l416_41608


namespace NUMINAMATH_CALUDE_min_savings_theorem_l416_41652

/-- Represents Kathleen's savings and spending -/
structure KathleenFinances where
  june_savings : ℕ
  july_savings : ℕ
  august_savings : ℕ
  school_supplies_cost : ℕ
  clothes_cost : ℕ
  amount_left : ℕ

/-- The minimum amount Kathleen needs to save to get $25 from her aunt -/
def min_savings_for_bonus (k : KathleenFinances) : ℕ :=
  k.amount_left

theorem min_savings_theorem (k : KathleenFinances) 
  (h1 : k.june_savings = 21)
  (h2 : k.july_savings = 46)
  (h3 : k.august_savings = 45)
  (h4 : k.school_supplies_cost = 12)
  (h5 : k.clothes_cost = 54)
  (h6 : k.amount_left = 46)
  (h7 : k.june_savings + k.july_savings + k.august_savings - k.school_supplies_cost - k.clothes_cost = k.amount_left) :
  min_savings_for_bonus k = k.amount_left :=
by sorry

end NUMINAMATH_CALUDE_min_savings_theorem_l416_41652


namespace NUMINAMATH_CALUDE_man_speed_opposite_train_man_speed_specific_case_l416_41646

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed_opposite_train (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let man_speed_mps := train_length / passing_time - train_speed_mps
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of a man running opposite to a train is approximately 5.99 kmph, given:
    - The train is 550 meters long
    - The train's speed is 60 kmph
    - The train passes the man in 30 seconds -/
theorem man_speed_specific_case : 
  abs (man_speed_opposite_train 550 60 30 - 5.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_opposite_train_man_speed_specific_case_l416_41646


namespace NUMINAMATH_CALUDE_pizzeria_sales_l416_41641

/-- Calculates the total sales of a pizzeria given the prices and quantities of small and large pizzas sold. -/
theorem pizzeria_sales
  (small_price : ℕ)
  (large_price : ℕ)
  (small_quantity : ℕ)
  (large_quantity : ℕ)
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : small_quantity = 8)
  (h4 : large_quantity = 3) :
  small_price * small_quantity + large_price * large_quantity = 40 :=
by sorry

#check pizzeria_sales

end NUMINAMATH_CALUDE_pizzeria_sales_l416_41641


namespace NUMINAMATH_CALUDE_yogurt_cost_yogurt_cost_is_one_l416_41618

/-- The cost of yogurt given Seth's purchase information -/
theorem yogurt_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream_cost := ice_cream_quantity * ice_cream_cost
  let yogurt_cost := (total_ice_cream_cost - extra_spent) / yogurt_quantity
  yogurt_cost

/-- Proof that the cost of each carton of yogurt is $1 -/
theorem yogurt_cost_is_one :
  yogurt_cost 20 2 6 118 = 1 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_cost_yogurt_cost_is_one_l416_41618


namespace NUMINAMATH_CALUDE_rectangle_length_l416_41693

/-- The length of a rectangle with given area and width -/
theorem rectangle_length (area width : ℝ) (h_area : area = 36.48) (h_width : width = 6.08) :
  area / width = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l416_41693


namespace NUMINAMATH_CALUDE_f_max_value_l416_41623

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (16 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l416_41623


namespace NUMINAMATH_CALUDE_no_real_roots_l416_41645

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (2 * x + 8) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l416_41645


namespace NUMINAMATH_CALUDE_equation_solution_l416_41655

theorem equation_solution :
  let f (x : ℝ) := 5 * (x^2)^2 + 3 * x^2 + 2 - 4 * (4 * x^2 + x^2 + 1)
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt ((17 + Real.sqrt 329) / 10) ∨ x = -Real.sqrt ((17 + Real.sqrt 329) / 10) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solution_l416_41655


namespace NUMINAMATH_CALUDE_coin_problem_l416_41610

/-- Given a total sum in paise, the number of 20 paise coins, and that the remaining sum is made up of 25 paise coins, calculate the total number of coins. -/
def total_coins (total_sum : ℕ) (coins_20 : ℕ) : ℕ :=
  let sum_20 := coins_20 * 20
  let sum_25 := total_sum - sum_20
  let coins_25 := sum_25 / 25
  coins_20 + coins_25

/-- Theorem stating that given the specific conditions, the total number of coins is 334. -/
theorem coin_problem : total_coins 7100 250 = 334 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l416_41610


namespace NUMINAMATH_CALUDE_tan_double_alpha_l416_41649

theorem tan_double_alpha (α : ℝ) (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) :
  Real.tan (2 * α) = -8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_alpha_l416_41649


namespace NUMINAMATH_CALUDE_min_f_1998_l416_41604

theorem min_f_1998 (f : ℕ → ℕ) 
  (h : ∀ s t : ℕ, f (t^2 * f s) = s * (f t)^2) : 
  f 1998 ≥ 1998 := by
  sorry

end NUMINAMATH_CALUDE_min_f_1998_l416_41604


namespace NUMINAMATH_CALUDE_complex_equation_solution_l416_41636

theorem complex_equation_solution :
  ∃ (x : ℂ), 5 - 2 * I * x = 4 - 5 * I * x ∧ x = I / 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l416_41636


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l416_41609

/-- Given a curve y = 3x^2 + 2x with a tangent line at (1, 5) parallel to 2ax - y - 6 = 0, 
    the value of a is 4. -/
theorem tangent_line_parallel (a : ℝ) : 
  (∃ (f : ℝ → ℝ) (line : ℝ → ℝ), 
    (∀ x, f x = 3 * x^2 + 2 * x) ∧ 
    (∀ x, line x = 2 * a * x - 6) ∧
    (∃ (tangent : ℝ → ℝ), 
      (tangent 1 = f 1) ∧
      (∀ h : ℝ, h ≠ 0 → (tangent (1 + h) - tangent 1) / h = (line (1 + h) - line 1) / h))) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l416_41609


namespace NUMINAMATH_CALUDE_darcy_shirts_count_darcy_shirts_proof_l416_41601

theorem darcy_shirts_count : ℕ :=
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11

  -- Define a function to calculate the total number of clothing items
  let total_clothing (shirts : ℕ) : ℕ := shirts + total_shorts

  -- Define a function to calculate the number of folded items
  let folded_items : ℕ := folded_shirts + folded_shorts

  -- The number of shirts that satisfies the conditions
  20

theorem darcy_shirts_proof (shirts : ℕ) : 
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11
  let total_clothing := shirts + total_shorts
  let folded_items := folded_shirts + folded_shorts

  shirts = 20 ↔ 
    total_clothing - folded_items = remaining_to_fold :=
by sorry

end NUMINAMATH_CALUDE_darcy_shirts_count_darcy_shirts_proof_l416_41601


namespace NUMINAMATH_CALUDE_ozone_experiment_properties_l416_41662

/-- Represents the experimental setup and data for the ozone effect study on mice. -/
structure OzoneExperiment where
  total_mice : Nat
  control_group : Nat
  experimental_group : Nat
  weight_increases : List ℝ
  k_squared_threshold : ℝ

/-- Represents the distribution of X, where X is the number of specified two mice
    assigned to the control group. -/
def distribution_X (exp : OzoneExperiment) : Fin 3 → ℝ := sorry

/-- Calculates the expected value of X. -/
def expected_value_X (exp : OzoneExperiment) : ℝ := sorry

/-- Calculates the median of all mice weight increases. -/
def median_weight_increase (exp : OzoneExperiment) : ℝ := sorry

/-- Represents the contingency table based on the median. -/
structure ContingencyTable where
  control_less : Nat
  control_greater_equal : Nat
  experimental_less : Nat
  experimental_greater_equal : Nat

/-- Constructs the contingency table based on the median. -/
def create_contingency_table (exp : OzoneExperiment) : ContingencyTable := sorry

/-- Calculates the K^2 value based on the contingency table. -/
def calculate_k_squared (table : ContingencyTable) : ℝ := sorry

/-- The main theorem stating the properties of the ozone experiment. -/
theorem ozone_experiment_properties (exp : OzoneExperiment) :
  exp.total_mice = 40 ∧
  exp.control_group = 20 ∧
  exp.experimental_group = 20 ∧
  distribution_X exp 0 = 19/78 ∧
  distribution_X exp 1 = 20/39 ∧
  distribution_X exp 2 = 19/78 ∧
  expected_value_X exp = 1 ∧
  median_weight_increase exp = 23.4 ∧
  let table := create_contingency_table exp
  table.control_less = 6 ∧
  table.control_greater_equal = 14 ∧
  table.experimental_less = 14 ∧
  table.experimental_greater_equal = 6 ∧
  calculate_k_squared table = 6.4 ∧
  calculate_k_squared table > exp.k_squared_threshold := by
  sorry

end NUMINAMATH_CALUDE_ozone_experiment_properties_l416_41662


namespace NUMINAMATH_CALUDE_garage_wheels_count_l416_41667

/-- The number of wheels on a bicycle -/
def bicycle_wheels : Nat := 2

/-- The number of wheels on a car -/
def car_wheels : Nat := 4

/-- The number of wheels on a motorcycle -/
def motorcycle_wheels : Nat := 2

/-- The number of bicycles in the garage -/
def num_bicycles : Nat := 20

/-- The number of cars in the garage -/
def num_cars : Nat := 10

/-- The number of motorcycles in the garage -/
def num_motorcycles : Nat := 5

/-- The total number of wheels in the garage -/
def total_wheels : Nat := bicycle_wheels * num_bicycles + car_wheels * num_cars + motorcycle_wheels * num_motorcycles

theorem garage_wheels_count : total_wheels = 90 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l416_41667


namespace NUMINAMATH_CALUDE_clock_chime_theorem_l416_41672

/-- Represents the number of chimes at a given time -/
def num_chimes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 0 then hour % 12
  else if minute = 30 then 1
  else 0

/-- Represents a sequence of four consecutive chimes -/
def chime_sequence (start_hour : ℕ) (start_minute : ℕ) : Prop :=
  num_chimes start_hour start_minute = 1 ∧
  num_chimes ((start_hour + (start_minute + 30) / 60) % 24) ((start_minute + 30) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 60) / 60) % 24) ((start_minute + 60) % 60) = 1 ∧
  num_chimes ((start_hour + (start_minute + 90) / 60) % 24) ((start_minute + 90) % 60) = 1

theorem clock_chime_theorem :
  ∀ (start_hour : ℕ) (start_minute : ℕ),
    chime_sequence start_hour start_minute →
    start_hour = 12 ∧ start_minute = 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_chime_theorem_l416_41672


namespace NUMINAMATH_CALUDE_dog_tail_length_l416_41624

/-- Represents a dog with specific proportions and measurements -/
structure Dog where
  body_length : ℝ
  tail_length : ℝ
  head_length : ℝ
  height : ℝ
  width : ℝ
  weight : ℝ

/-- The dog satisfies the given proportions and measurements -/
def is_valid_dog (d : Dog) : Prop :=
  d.tail_length = d.body_length / 2 ∧
  d.head_length = d.body_length / 6 ∧
  d.height = 1.5 * d.width ∧
  d.weight = 36 ∧
  d.body_length + d.tail_length + d.head_length = 30 ∧
  d.width = 12

/-- The theorem stating that a valid dog's tail length is 10 inches -/
theorem dog_tail_length (d : Dog) (h : is_valid_dog d) : d.tail_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_tail_length_l416_41624


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l416_41670

def rotation_angle (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2 ∧ Real.cos α = 4 / 5

def overlapping_area (α : Real) : Real :=
  -- Definition of the overlapping area function
  sorry

theorem overlapping_squares_area (α : Real) 
  (h : rotation_angle α) : overlapping_area α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l416_41670


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l416_41633

theorem exactly_one_greater_than_one
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (ineq : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l416_41633


namespace NUMINAMATH_CALUDE_car_speed_increase_l416_41678

theorem car_speed_increase (original_speed : ℝ) (supercharge_percent : ℝ) (weight_reduction_increase : ℝ) : 
  original_speed = 150 → 
  supercharge_percent = 30 → 
  weight_reduction_increase = 10 → 
  original_speed * (1 + supercharge_percent / 100) + weight_reduction_increase = 205 :=
by
  sorry

#check car_speed_increase

end NUMINAMATH_CALUDE_car_speed_increase_l416_41678


namespace NUMINAMATH_CALUDE_root_sum_product_l416_41659

def complex_plane : Type := ℂ

def coordinates (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem root_sum_product (z : ℂ) (p q : ℝ) :
  coordinates z = (-1, 3) →
  (z^2 + p*z + q = 0) →
  p + q = 12 := by sorry

end NUMINAMATH_CALUDE_root_sum_product_l416_41659


namespace NUMINAMATH_CALUDE_pattern_and_application_l416_41616

theorem pattern_and_application (n : ℕ) (a b : ℝ) :
  n > 1 →
  (n : ℝ) * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) ∧
  (a * Real.sqrt (7 / b) = Real.sqrt (a + 7 / b) → a + b = 55) :=
by sorry

end NUMINAMATH_CALUDE_pattern_and_application_l416_41616


namespace NUMINAMATH_CALUDE_negation_of_existence_l416_41695

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x, x^2 - 1 < 0) ↔ (∀ x, x^2 - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l416_41695


namespace NUMINAMATH_CALUDE_triangle_side_less_than_half_perimeter_l416_41679

theorem triangle_side_less_than_half_perimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_less_than_half_perimeter_l416_41679


namespace NUMINAMATH_CALUDE_handshakes_in_social_event_l416_41686

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  totalPeople : Nat
  group1Size : Nat
  group2Size : Nat
  knownInGroup1 : Nat
  knownInGroup2 : Nat

/-- Calculates the number of handshakes in a social event -/
def calculateHandshakes (event : SocialEvent) : Nat :=
  let group1Handshakes := event.group1Size * (event.totalPeople - event.group1Size + event.knownInGroup1)
  let group2Handshakes := event.group2Size * (event.totalPeople - event.group2Size + event.knownInGroup2)
  (group1Handshakes + group2Handshakes) / 2

/-- Theorem stating that the number of handshakes in the given social event is 630 -/
theorem handshakes_in_social_event :
  let event : SocialEvent := {
    totalPeople := 40,
    group1Size := 25,
    group2Size := 15,
    knownInGroup1 := 18,
    knownInGroup2 := 4
  }
  calculateHandshakes event = 630 := by
  sorry


end NUMINAMATH_CALUDE_handshakes_in_social_event_l416_41686


namespace NUMINAMATH_CALUDE_sum_of_series_l416_41692

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series : ∑' k, (k : ℝ) / 3^k = 3/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_l416_41692


namespace NUMINAMATH_CALUDE_integer_part_inequality_l416_41642

theorem integer_part_inequality (m n : ℕ+) : 
  (∀ (α β : ℝ), ⌊(m+n)*α⌋ + ⌊(m+n)*β⌋ ≥ ⌊m*α⌋ + ⌊m*β⌋ + ⌊n*(α+β)⌋) ↔ m = n :=
sorry

end NUMINAMATH_CALUDE_integer_part_inequality_l416_41642


namespace NUMINAMATH_CALUDE_symmetry_xoy_plane_l416_41625

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_xoy (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_xoy_plane :
  let P : Point3D := ⟨1, 3, -5⟩
  symmetric_xoy P = ⟨1, 3, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetry_xoy_plane_l416_41625


namespace NUMINAMATH_CALUDE_keith_picked_no_pears_l416_41614

-- Define the number of apples picked by each person
def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def keith_apples : ℕ := 6

-- Define the total number of apples picked
def total_apples : ℕ := 16

-- Define Keith's pears as a variable
def keith_pears : ℕ := sorry

-- Theorem to prove
theorem keith_picked_no_pears : keith_pears = 0 := by
  sorry

end NUMINAMATH_CALUDE_keith_picked_no_pears_l416_41614


namespace NUMINAMATH_CALUDE_log_equation_solution_l416_41685

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : 2 * x ≠ 1) (h3 : 4 * x ≠ 1) :
  (Real.log (4 * x) / Real.log (2 * x)) + (Real.log (16 * x) / Real.log (4 * x)) = 4 ↔ 
  x = 1 ∨ x = 1 / (2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l416_41685


namespace NUMINAMATH_CALUDE_existence_of_integers_l416_41665

theorem existence_of_integers (a₁ a₂ a₃ : ℕ) (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) :
  ∃ x₁ x₂ x₃ : ℤ,
    (abs x₁ + abs x₂ + abs x₃ > 0) ∧
    (a₁ * x₁ + a₂ * x₂ + a₃ * x₃ = 0) ∧
    (max (abs x₁) (max (abs x₂) (abs x₃)) < (2 / Real.sqrt 3) * Real.sqrt a₃ + 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_integers_l416_41665


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l416_41634

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (81^2 - (m + 3) * 81 + m + 2 = 0) → m = 79 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l416_41634


namespace NUMINAMATH_CALUDE_sin_cos_sum_17_13_l416_41681

theorem sin_cos_sum_17_13 : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.cos (17 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_17_13_l416_41681


namespace NUMINAMATH_CALUDE_beach_house_rental_l416_41657

theorem beach_house_rental (individual_payment : ℕ) (total_payment : ℕ) 
  (h1 : individual_payment = 70)
  (h2 : total_payment = 490) :
  total_payment / individual_payment = 7 :=
by sorry

end NUMINAMATH_CALUDE_beach_house_rental_l416_41657


namespace NUMINAMATH_CALUDE_function_value_at_three_l416_41650

/-- Given a continuous and differentiable function f satisfying
    f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2,
    prove that f(3) = 11. -/
theorem function_value_at_three
  (f : ℝ → ℝ)
  (hcont : Continuous f)
  (hdiff : Differentiable ℝ f)
  (hfunc : ∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1)
  (hf0 : f 0 = 2) :
  f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l416_41650


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l416_41647

/-- The center of the circle (x-1)^2 + (y+1)^2 = 2 -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The slope of the given line 2x + y = 0 -/
def given_line_slope : ℝ := -2

/-- The perpendicular line passing through the circle center -/
def perpendicular_line (x y : ℝ) : Prop :=
  x - given_line_slope * y - (circle_center.1 - given_line_slope * circle_center.2) = 0

theorem perpendicular_line_equation :
  perpendicular_line = fun x y ↦ x - 2 * y - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l416_41647


namespace NUMINAMATH_CALUDE_tim_stored_bales_l416_41654

theorem tim_stored_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28) 
  (h2 : final_bales = 54) : 
  final_bales - initial_bales = 26 := by
  sorry

end NUMINAMATH_CALUDE_tim_stored_bales_l416_41654


namespace NUMINAMATH_CALUDE_base_4_last_digit_379_l416_41611

def base_4_last_digit (n : ℕ) : ℕ :=
  n % 4

theorem base_4_last_digit_379 : base_4_last_digit 379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_4_last_digit_379_l416_41611


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l416_41656

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (21/16, 9/8)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 4 * x - 3

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l416_41656


namespace NUMINAMATH_CALUDE_line_passes_through_center_l416_41637

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2

-- Theorem statement
theorem line_passes_through_center :
  ∃ h k : ℝ, is_center h k ∧ line_equation h k :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_center_l416_41637


namespace NUMINAMATH_CALUDE_x_greater_than_x_squared_only_half_satisfies_l416_41663

theorem x_greater_than_x_squared (x : ℝ) : x > x^2 ↔ x ∈ (Set.Ioo 0 1) := by sorry

theorem only_half_satisfies :
  ∀ x ∈ ({-2, -(1/2), 0, 1/2, 2} : Set ℝ), x > x^2 ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_x_greater_than_x_squared_only_half_satisfies_l416_41663


namespace NUMINAMATH_CALUDE_least_common_multiple_4_5_6_9_l416_41682

theorem least_common_multiple_4_5_6_9 : ∃ (n : ℕ), n > 0 ∧ 
  4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 → 4 ∣ m → 5 ∣ m → 6 ∣ m → 9 ∣ m → n ≤ m :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_4_5_6_9_l416_41682


namespace NUMINAMATH_CALUDE_earthquake_victims_scientific_notation_l416_41602

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The problem statement -/
theorem earthquake_victims_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 153000 a n ∧ a = 1.53 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_victims_scientific_notation_l416_41602


namespace NUMINAMATH_CALUDE_flagpole_breaking_point_l416_41684

theorem flagpole_breaking_point (h : ℝ) (b : ℝ) (t : ℝ) :
  h = 12 ∧ t = 2 ∧ b > 0 →
  b^2 + (h - t)^2 = h^2 →
  b = 2 * Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_breaking_point_l416_41684


namespace NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l416_41660

theorem percentage_of_boys_studying_science (total_boys : ℕ) (boys_from_A : ℕ) (boys_A_not_science : ℕ) :
  total_boys = 150 →
  boys_from_A = (20 : ℕ) * total_boys / 100 →
  boys_A_not_science = 21 →
  (boys_from_A - boys_A_not_science) * 100 / boys_from_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l416_41660


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l416_41651

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l416_41651


namespace NUMINAMATH_CALUDE_quadratic_inequality_sufficient_not_necessary_l416_41643

theorem quadratic_inequality_sufficient_not_necessary :
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ ¬(x^2 - 2*x < 0)) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sufficient_not_necessary_l416_41643


namespace NUMINAMATH_CALUDE_characterize_functions_l416_41628

def is_valid_function (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → ⌊(f (m * n) : ℚ) / n⌋ = f m

theorem characterize_functions :
  ∀ f : ℕ → ℤ, is_valid_function f →
    ∃ r : ℝ, (∀ n : ℕ, n > 0 → f n = ⌊n * r⌋) ∨
              (∀ n : ℕ, n > 0 → f n = ⌈n * r⌉ - 1) :=
sorry

end NUMINAMATH_CALUDE_characterize_functions_l416_41628


namespace NUMINAMATH_CALUDE_equation_satisfied_l416_41694

theorem equation_satisfied (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_satisfied_l416_41694


namespace NUMINAMATH_CALUDE_two_x_value_l416_41690

theorem two_x_value (x : ℚ) (h : 4 * x + 14 = 8 * x - 48) : 2 * x = 31 := by
  sorry

end NUMINAMATH_CALUDE_two_x_value_l416_41690


namespace NUMINAMATH_CALUDE_special_matrix_sum_l416_41600

/-- Represents a 3x3 matrix with the given structure -/
structure SpecialMatrix where
  v : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  sum_equality : ℝ
  sum_row_1 : v + 50 + w = sum_equality
  sum_row_2 : 196 + x + y = sum_equality
  sum_row_3 : 269 + z + 123 = sum_equality
  sum_col_1 : v + 196 + 269 = sum_equality
  sum_col_2 : 50 + x + z = sum_equality
  sum_col_3 : w + y + 123 = sum_equality
  sum_diag_1 : v + x + 123 = sum_equality
  sum_diag_2 : w + x + 269 = sum_equality

/-- Theorem: In the SpecialMatrix, y + z = 196 -/
theorem special_matrix_sum (m : SpecialMatrix) : m.y + m.z = 196 := by
  sorry

end NUMINAMATH_CALUDE_special_matrix_sum_l416_41600


namespace NUMINAMATH_CALUDE_problem_solution_l416_41674

def sum_of_integers (a b : ℕ) : ℕ :=
  ((b - a + 1) * (a + b)) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem problem_solution :
  let x := sum_of_integers 40 60
  let y := count_even_integers 40 60
  x + y = 1061 → y = 11 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l416_41674


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l416_41673

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 7 * x^2 - 4 * x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l416_41673


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l416_41640

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  (2 * b - Real.sqrt 3 * c) * Real.cos A = Real.sqrt 3 * a * Real.cos C →
  B = π / 6 →
  Real.sqrt ((M.1 - (b + c) / 2)^2 + (M.2)^2) = Real.sqrt 7 →
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l416_41640


namespace NUMINAMATH_CALUDE_unique_solutions_l416_41626

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_solutions (x y n : ℕ) : 
  (factorial x + factorial y) / factorial n = 3^n ↔ 
  ((x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l416_41626


namespace NUMINAMATH_CALUDE_stamp_denominations_l416_41658

/-- Given stamps of denominations 7, n, and n+2 cents, 
    if 120 cents is the greatest postage that cannot be formed, then n = 22 -/
theorem stamp_denominations (n : ℕ) : 
  (∀ k > 120, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c) ∧
  (¬ ∃ a b c : ℕ, 120 = 7 * a + n * b + (n + 2) * c) →
  n = 22 := by sorry

end NUMINAMATH_CALUDE_stamp_denominations_l416_41658


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l416_41671

theorem absolute_value_simplification : |(-4^2 + 6)| = 10 := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l416_41671


namespace NUMINAMATH_CALUDE_cans_needed_for_35_rooms_l416_41612

/-- Represents the number of rooms that can be painted with the available paint -/
def initial_rooms : ℕ := 45

/-- Represents the number of paint cans lost -/
def lost_cans : ℕ := 5

/-- Represents the number of rooms that can be painted after losing some paint cans -/
def remaining_rooms : ℕ := 35

/-- Represents that each can must be used entirely (no partial cans) -/
def whole_cans_only : Prop := True

/-- Theorem stating that 18 cans are needed to paint 35 rooms given the conditions -/
theorem cans_needed_for_35_rooms : 
  ∃ (cans_per_room : ℚ),
    cans_per_room * (initial_rooms - remaining_rooms) = lost_cans ∧
    ∃ (cans_needed : ℕ),
      cans_needed = ⌈(remaining_rooms : ℚ) / cans_per_room⌉ ∧
      cans_needed = 18 :=
sorry

end NUMINAMATH_CALUDE_cans_needed_for_35_rooms_l416_41612


namespace NUMINAMATH_CALUDE_puppy_adoption_ratio_l416_41653

theorem puppy_adoption_ratio :
  let first_week : ℕ := 20
  let second_week : ℕ := (2 * first_week) / 5
  let fourth_week : ℕ := first_week + 10
  let total_puppies : ℕ := 74
  let third_week : ℕ := total_puppies - (first_week + second_week + fourth_week)
  (third_week : ℚ) / second_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_adoption_ratio_l416_41653


namespace NUMINAMATH_CALUDE_max_value_L_in_triangle_l416_41676

/-- The function L(x, y) = -2x + y -/
def L (x y : ℝ) : ℝ := -2*x + y

/-- The triangle ABC with vertices A(-2, -1), B(0, 1), and C(2, -1) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
       p.1 = -2*a + 0*b + 2*c ∧
       p.2 = -1*a + 1*b - 1*c}

theorem max_value_L_in_triangle :
  ∃ (max : ℝ), max = 3 ∧ 
  ∀ (x y : ℝ), (x, y) ∈ triangle_ABC → L x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_L_in_triangle_l416_41676


namespace NUMINAMATH_CALUDE_loss_percentage_is_twenty_percent_l416_41639

-- Define the given conditions
def articles_sold_gain : ℕ := 20
def selling_price_gain : ℚ := 60
def gain_percentage : ℚ := 20 / 100

def articles_sold_loss : ℚ := 24.999996875000388
def selling_price_loss : ℚ := 50

-- Theorem to prove
theorem loss_percentage_is_twenty_percent :
  let cost_price := selling_price_gain / (1 + gain_percentage)
  let cost_per_article := cost_price / articles_sold_gain
  let cost_price_loss := cost_per_article * articles_sold_loss
  let loss := cost_price_loss - selling_price_loss
  loss / cost_price_loss = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_twenty_percent_l416_41639


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l416_41699

-- Define the arithmetic progression
def arithmeticProgression (n : ℕ) : ℕ := 36 * n + 3

-- Define the property of not being a sum of two squares
def notSumOfTwoSquares (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^2 + b^2

-- Define the property of not being a sum of two cubes
def notSumOfTwoCubes (k : ℕ) : Prop := ∀ a b : ℕ, k ≠ a^3 + b^3

theorem arithmetic_progression_properties :
  (∀ n : ℕ, arithmeticProgression n > 0) ∧  -- Positive integers
  (∀ n m : ℕ, n ≠ m → arithmeticProgression n ≠ arithmeticProgression m) ∧  -- Non-constant
  (∀ n : ℕ, notSumOfTwoSquares (arithmeticProgression n)) ∧  -- Not sum of two squares
  (∀ n : ℕ, notSumOfTwoCubes (arithmeticProgression n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l416_41699


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l416_41620

/-- Proves that the percentage of passed candidates is 32% given the conditions of the examination --/
theorem exam_pass_percentage : 
  ∀ (total_candidates : ℕ) 
    (num_girls : ℕ) 
    (num_boys : ℕ) 
    (fail_percentage : ℝ) 
    (pass_percentage : ℝ),
  total_candidates = 2000 →
  num_girls = 900 →
  num_boys = total_candidates - num_girls →
  fail_percentage = 68 →
  pass_percentage = 100 - fail_percentage →
  pass_percentage = 32 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l416_41620


namespace NUMINAMATH_CALUDE_intersection_theorem_l416_41630

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x + 1 > 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_inter_B = Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l416_41630


namespace NUMINAMATH_CALUDE_remainder_of_741147_div_6_l416_41629

theorem remainder_of_741147_div_6 : 741147 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_741147_div_6_l416_41629


namespace NUMINAMATH_CALUDE_a_formula_l416_41698

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℤ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_formula (n : ℕ+) : a n = 2*n - 2 := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l416_41698


namespace NUMINAMATH_CALUDE_selection_methods_count_l416_41687

def total_students : ℕ := 9
def selected_students : ℕ := 4
def specific_students : ℕ := 3

def selection_methods : ℕ := 
  Nat.choose specific_students 2 * Nat.choose (total_students - specific_students) 2 +
  Nat.choose specific_students 3 * Nat.choose (total_students - specific_students) 1

theorem selection_methods_count : selection_methods = 51 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l416_41687


namespace NUMINAMATH_CALUDE_fourth_term_is_nine_l416_41613

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that the 4th term of the arithmetic sequence is 9 -/
theorem fourth_term_is_nine (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 3)
    (sum_three : seq.S 3 = 15) : 
  seq.a 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_nine_l416_41613


namespace NUMINAMATH_CALUDE_tan_alpha_value_l416_41619

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l416_41619


namespace NUMINAMATH_CALUDE_largest_number_proof_l416_41621

theorem largest_number_proof (a b c : ℕ) 
  (h1 : c - a = 6) 
  (h2 : b = (a + c) / 2) 
  (h3 : a * b * c = 46332) : 
  c = 39 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_proof_l416_41621


namespace NUMINAMATH_CALUDE_choose_4_from_10_l416_41668

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l416_41668


namespace NUMINAMATH_CALUDE_x_value_l416_41691

theorem x_value (x : ℝ) : x = 40 * (1 + 0.2) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l416_41691


namespace NUMINAMATH_CALUDE_rectangle_area_l416_41635

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 →
  ratio = 3 →
  (2 * r * ratio) * (2 * r) = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l416_41635


namespace NUMINAMATH_CALUDE_eggs_remaining_l416_41617

theorem eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_taken = 5 → eggs_left = initial_eggs - eggs_taken → eggs_left = 42 :=
by sorry

end NUMINAMATH_CALUDE_eggs_remaining_l416_41617


namespace NUMINAMATH_CALUDE_fundraiser_results_l416_41606

/-- Represents the sales data for Markeesha's fundraiser --/
structure SalesData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total profit given the sales data --/
def totalProfit (data : SalesData) : ℚ :=
  (4 * data.monday + 205) * (4.5 : ℚ)

/-- Calculates the total number of boxes sold given the sales data --/
def totalBoxesSold (data : SalesData) : ℕ :=
  4 * data.monday + 205

/-- Determines the most profitable day given the sales data --/
def mostProfitableDay (data : SalesData) : String :=
  if data.saturday ≥ data.monday ∧ data.saturday ≥ data.tuesday ∧ 
     data.saturday ≥ data.wednesday ∧ data.saturday ≥ data.thursday ∧ 
     data.saturday ≥ data.friday ∧ data.saturday ≥ data.sunday
  then "Saturday"
  else "Other"

theorem fundraiser_results (M : ℕ) :
  let data : SalesData := {
    monday := M,
    tuesday := M + 10,
    wednesday := M + 20,
    thursday := M + 30,
    friday := 30,
    saturday := 60,
    sunday := 45
  }
  totalProfit data = (4 * M + 205 : ℕ) * (4.5 : ℚ) ∧
  totalBoxesSold data = 4 * M + 205 ∧
  mostProfitableDay data = "Saturday" :=
by sorry

#check fundraiser_results

end NUMINAMATH_CALUDE_fundraiser_results_l416_41606


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l416_41632

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l416_41632


namespace NUMINAMATH_CALUDE_problem_statement_l416_41669

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The statement to be proved -/
theorem problem_statement (m : ℝ) : 
  (∀ x : ℝ, f m x > 0) ∧ 
  (∃ x : ℝ, x^2 < 9 - m^2) ↔ 
  (m > -3 ∧ m ≤ -2) ∨ (m ≥ 2 ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l416_41669


namespace NUMINAMATH_CALUDE_lowest_score_proof_l416_41689

theorem lowest_score_proof (scores : List ℝ) (highest lowest : ℝ) : 
  scores.length = 12 →
  scores.sum / scores.length = 82 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 98 →
  (scores.filter (λ x => x ≠ highest ∧ x ≠ lowest)).sum / 10 = 84 →
  lowest = 46 := by
sorry

end NUMINAMATH_CALUDE_lowest_score_proof_l416_41689


namespace NUMINAMATH_CALUDE_sum_x_y_equals_six_l416_41696

theorem sum_x_y_equals_six (x y : ℝ) : 
  (|x| + x + y = 16) → (x + |y| - y = 18) → (x + y = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_six_l416_41696


namespace NUMINAMATH_CALUDE_field_area_l416_41677

/-- Given a rectangular field with one side of 34 feet and three sides fenced with a total of 74 feet of fencing, the area of the field is 680 square feet. -/
theorem field_area (L W : ℝ) (h1 : L = 34) (h2 : 2 * W + L = 74) : L * W = 680 := by
  sorry

end NUMINAMATH_CALUDE_field_area_l416_41677


namespace NUMINAMATH_CALUDE_equation_satisfied_l416_41607

theorem equation_satisfied (a b c : ℤ) (h1 : a = c - 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l416_41607


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l416_41697

theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →  -- angles are positive
    a + b + c = 180 →        -- sum of angles is 180°
    b = 3 * a →              -- ratio condition
    c = 4 * a →              -- ratio condition
    c = 90 :=                -- largest angle is 90°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l416_41697


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l416_41688

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The theorem stating that 6n^2 is the largest divisor of n^4 - n^2 for all composite n -/
theorem largest_divisor_of_n4_minus_n2 (n : ℕ) (h : IsComposite n) :
  (∃ (k : ℕ), (n^4 - n^2) % (6 * n^2) = 0 ∧
    ∀ (m : ℕ), (n^4 - n^2) % m = 0 → m ≤ 6 * n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l416_41688


namespace NUMINAMATH_CALUDE_chip_cost_is_fifty_cents_l416_41666

/-- The cost of a bag of chips given the conditions in the problem -/
def chip_cost : ℚ :=
  let candy_cost : ℚ := 2
  let student_count : ℕ := 5
  let total_cost : ℚ := 15
  let candy_per_student : ℕ := 1
  let chips_per_student : ℕ := 2
  (total_cost - student_count * candy_cost) / (student_count * chips_per_student)

/-- Theorem stating that the cost of a bag of chips is $0.50 -/
theorem chip_cost_is_fifty_cents : chip_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_is_fifty_cents_l416_41666


namespace NUMINAMATH_CALUDE_count_cubic_functions_l416_41622

-- Define the structure of our cubic function
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property we're interested in
def satisfiesProperty (f : CubicFunction) : Prop :=
  ∀ x : ℝ, (f.a * x^3 + f.b * x^2 + f.c * x + f.d) *
            ((-f.a) * x^3 + f.b * x^2 + (-f.c) * x + f.d) =
            f.a * x^6 + f.b * x^4 + f.c * x^2 + f.d

-- State the theorem
theorem count_cubic_functions :
  ∃! (s : Finset CubicFunction),
    (∀ f ∈ s, satisfiesProperty f) ∧ s.card = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_cubic_functions_l416_41622


namespace NUMINAMATH_CALUDE_cos_double_angle_with_tan_l416_41680

theorem cos_double_angle_with_tan (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (2 * θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_with_tan_l416_41680


namespace NUMINAMATH_CALUDE_sin_C_equals_half_l416_41661

theorem sin_C_equals_half (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- b = 2c * sin(B)
  (b = 2 * c * Real.sin B) →
  -- Then sin(C) = 1/2
  Real.sin C = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_C_equals_half_l416_41661


namespace NUMINAMATH_CALUDE_min_x_plus_y_l416_41631

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = x₀*y₀ ∧ x₀ + y₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l416_41631


namespace NUMINAMATH_CALUDE_max_2x2_squares_5x7_grid_l416_41615

/-- Represents the dimensions of the grid -/
structure GridDimensions where
  rows : ℕ
  cols : ℕ

/-- Represents the different types of pieces that can be cut from the grid -/
inductive PieceType
  | Square2x2
  | LShape
  | Strip1x3

/-- Represents a configuration of pieces cut from the grid -/
structure Configuration where
  square2x2Count : ℕ
  lShapeCount : ℕ
  strip1x3Count : ℕ

/-- Checks if a configuration is valid for the given grid dimensions -/
def isValidConfiguration (grid : GridDimensions) (config : Configuration) : Prop :=
  4 * config.square2x2Count + 3 * config.lShapeCount + 3 * config.strip1x3Count = grid.rows * grid.cols

/-- Theorem: The maximum number of 2x2 squares in a valid configuration for a 5x7 grid is 5 -/
theorem max_2x2_squares_5x7_grid :
  ∃ (maxSquares : ℕ),
    maxSquares = 5 ∧
    (∃ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config ∧
      config.square2x2Count = maxSquares) ∧
    (∀ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config →
      config.square2x2Count ≤ maxSquares) :=
by
  sorry

end NUMINAMATH_CALUDE_max_2x2_squares_5x7_grid_l416_41615


namespace NUMINAMATH_CALUDE_problem_statement_l416_41664

theorem problem_statement (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) :
  (40/100) * N = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l416_41664


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l416_41605

/-- The line kx + 3y + k - 9 = 0 passes through the point (-1, 3) for all values of k -/
theorem line_passes_through_fixed_point (k : ℝ) : k * (-1) + 3 * 3 + k - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l416_41605


namespace NUMINAMATH_CALUDE_min_sum_at_five_l416_41683

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The conditions given in the problem -/
axiom condition1 : S 10 = 0
axiom condition2 : S 15 = 25

/-- The theorem to prove -/
theorem min_sum_at_five :
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), S m ≥ S n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_five_l416_41683


namespace NUMINAMATH_CALUDE_multiply_fractions_of_numbers_l416_41648

theorem multiply_fractions_of_numbers : 
  (1/4 : ℚ) * 15 * ((1/3 : ℚ) * 10) = 25/2 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_of_numbers_l416_41648


namespace NUMINAMATH_CALUDE_mother_age_twice_lisa_l416_41603

/-- Lisa's birth year -/
def lisa_birth_year : ℕ := 1994

/-- The year of Lisa's 10th birthday -/
def reference_year : ℕ := 2004

/-- Lisa's age in the reference year -/
def lisa_age_reference : ℕ := 10

/-- Lisa's mother's age multiplier in the reference year -/
def mother_age_multiplier_reference : ℕ := 5

/-- The year when Lisa's mother's age is twice Lisa's age -/
def target_year : ℕ := 2034

theorem mother_age_twice_lisa (y : ℕ) :
  (y - lisa_birth_year) * 2 = (y - lisa_birth_year + mother_age_multiplier_reference * lisa_age_reference - lisa_age_reference) →
  y = target_year :=
by sorry

end NUMINAMATH_CALUDE_mother_age_twice_lisa_l416_41603


namespace NUMINAMATH_CALUDE_calvin_chips_weeks_l416_41638

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

end NUMINAMATH_CALUDE_calvin_chips_weeks_l416_41638


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l416_41675

theorem sqrt_fraction_equality : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
  (-2 * Real.sqrt 7 + Real.sqrt 3 + Real.sqrt 5) / 59 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l416_41675


namespace NUMINAMATH_CALUDE_prime_sum_30_l416_41644

theorem prime_sum_30 (A B C : ℕ) : 
  Prime A ∧ Prime B ∧ Prime C ∧
  A < 20 ∧ B < 20 ∧ C < 20 ∧
  A + B + C = 30 →
  (A = 2 ∧ B = 11 ∧ C = 17) ∨
  (A = 2 ∧ B = 17 ∧ C = 11) ∨
  (A = 11 ∧ B = 2 ∧ C = 17) ∨
  (A = 11 ∧ B = 17 ∧ C = 2) ∨
  (A = 17 ∧ B = 2 ∧ C = 11) ∨
  (A = 17 ∧ B = 11 ∧ C = 2) := by
sorry

end NUMINAMATH_CALUDE_prime_sum_30_l416_41644
