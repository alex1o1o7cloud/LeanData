import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1353_135368

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h1 : a 1 > 0) 
  (h2 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1353_135368


namespace NUMINAMATH_CALUDE_square_and_product_l1353_135304

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : x = 42 ∧ (x + 2) * (x - 2) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_square_and_product_l1353_135304


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1353_135300

/-- The trajectory of the midpoint Q between a point P on the unit circle and a fixed point M -/
theorem midpoint_trajectory (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1 = (P.1 + 2) / 2 ∧ Q.2 = P.2 / 2) →  -- Q is the midpoint of PM where M is (2, 0)
  (Q.1 - 1)^2 + Q.2^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1353_135300


namespace NUMINAMATH_CALUDE_max_value_theorem_l1353_135331

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 3) :
  (1/2 : ℝ)*x + y ≤ Real.sqrt 6 / 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*y₀^2 = 3 ∧ (1/2 : ℝ)*x₀ + y₀ = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1353_135331


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1353_135353

/-- A line passing through a fixed point for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1353_135353


namespace NUMINAMATH_CALUDE_restaurant_revenue_l1353_135362

/-- Calculates the total revenue from meals sold at a restaurant --/
theorem restaurant_revenue 
  (x y z : ℝ) -- Costs of kids, adult, and seniors' meals respectively
  (ratio_kids : ℕ) (ratio_adult : ℕ) (ratio_senior : ℕ) -- Ratio of meals sold
  (kids_meals_sold : ℕ) -- Number of kids meals sold
  (h_ratio : ratio_kids = 3 ∧ ratio_adult = 2 ∧ ratio_senior = 1) -- Given ratio
  (h_kids_sold : kids_meals_sold = 12) -- Given number of kids meals sold
  : 
  ∃ (total_revenue : ℝ),
    total_revenue = kids_meals_sold * x + 
      (kids_meals_sold * ratio_adult / ratio_kids) * y + 
      (kids_meals_sold * ratio_senior / ratio_kids) * z ∧
    total_revenue = 12 * x + 8 * y + 4 * z :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_revenue_l1353_135362


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1353_135387

/-- Given a line with equation y - 7 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 88/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 7 = -3 * (x - 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 7 = -3 * (x_int - 5)) ∧ 
    (0 - 7 = -3 * (x_int - 5)) ∧ 
    (y_int - 7 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 88 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1353_135387


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1353_135345

/-- Given a sequence a_n where a_2 = 2, a_6 = 0, and {1 / (a_n + 1)} is an arithmetic sequence,
    prove that a_4 = 1/2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 2 = 2)
  (h2 : a 6 = 0)
  (h3 : ∃ d : ℚ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d) :
  a 4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1353_135345


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l1353_135309

theorem total_books_on_shelves (x : ℕ) : 
  (x / 2 + 5 = 2 * (x / 2 - 5)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l1353_135309


namespace NUMINAMATH_CALUDE_tina_total_time_l1353_135392

/-- Calculates the total time for Tina to clean keys, let them dry, take breaks, and complete her assignment -/
def total_time (total_keys : ℕ) (keys_to_clean : ℕ) (clean_time_per_key : ℕ) (dry_time_per_key : ℕ) (break_interval : ℕ) (break_duration : ℕ) (assignment_time : ℕ) : ℕ :=
  let cleaning_time := keys_to_clean * clean_time_per_key
  let drying_time := total_keys * dry_time_per_key
  let break_count := total_keys / break_interval
  let break_time := break_count * break_duration
  cleaning_time + drying_time + break_time + assignment_time

/-- Proves that given the conditions in the problem, the total time is 541 minutes -/
theorem tina_total_time :
  total_time 30 29 7 10 5 3 20 = 541 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l1353_135392


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1353_135302

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + 8 * y₀ - x₀ * y₀ = 0 ∧ x₀ * y₀ = 64) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' * y' ≥ 64) ∧
  (∃ (x₁ y₁ : ℝ), x₁ > 0 ∧ y₁ > 0 ∧ 2 * x₁ + 8 * y₁ - x₁ * y₁ = 0 ∧ x₁ + y₁ = 18) ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 2 * x' + 8 * y' - x' * y' = 0 → x' + y' ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_y_l1353_135302


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l1353_135346

/-- A right-angled triangle with its inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The length of one leg of the right triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- All lengths are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- 
In a right-angled triangle, the sum of the lengths of the two legs 
is equal to the sum of the diameters of the inscribed and circumscribed circles
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum (t : RightTriangle) :
  t.leg1 + t.leg2 = 2 * t.inradius + 2 * t.circumradius := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l1353_135346


namespace NUMINAMATH_CALUDE_max_value_f_l1353_135323

def f (x : ℝ) := x * (1 - x)

theorem max_value_f :
  ∃ (m : ℝ), ∀ (x : ℝ), 0 < x ∧ x < 1 → f x ≤ m ∧ (∃ (y : ℝ), 0 < y ∧ y < 1 ∧ f y = m) ∧ m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_f_l1353_135323


namespace NUMINAMATH_CALUDE_midpoint_polar_coordinates_l1353_135336

/-- The polar coordinates of the midpoint of the chord intercepted by two curves -/
theorem midpoint_polar_coordinates (ρ θ : ℝ) :
  (ρ * (Real.cos θ - Real.sin θ) + 2 = 0) →  -- Curve C₁
  (ρ = 2) →  -- Curve C₂
  ∃ (r θ' : ℝ), (r = Real.sqrt 2 ∧ θ' = 3 * Real.pi / 4 ∧
    r * Real.cos θ' = -1 ∧ r * Real.sin θ' = 1) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_polar_coordinates_l1353_135336


namespace NUMINAMATH_CALUDE_total_figures_is_44_l1353_135305

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 11

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures is 44. -/
theorem total_figures_is_44 : total_figures = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_figures_is_44_l1353_135305


namespace NUMINAMATH_CALUDE_range_of_product_l1353_135337

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_product (a b : ℝ) 
  (h1 : a < b) (h2 : b < -1) (h3 : f a = f b) :
  ∃ y, y ∈ Set.Ioo 0 2 ∧ y = (a + 1) * (b + 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_product_l1353_135337


namespace NUMINAMATH_CALUDE_train_passing_platform_time_l1353_135384

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 720)
  (h2 : train_speed_kmh = 72)
  (h3 : platform_length = 280) : 
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 50 :=
sorry

end NUMINAMATH_CALUDE_train_passing_platform_time_l1353_135384


namespace NUMINAMATH_CALUDE_even_rows_in_pascal_triangle_l1353_135361

/-- Pascal's triangle row -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Check if a row (excluding endpoints) consists of only even numbers -/
def is_even_row (row : List ℕ) : Bool := sorry

/-- Count of even rows in first n rows of Pascal's triangle (excluding row 0 and 1) -/
def count_even_rows (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 4 even rows in the first 30 rows of Pascal's triangle (excluding row 0 and 1) -/
theorem even_rows_in_pascal_triangle : count_even_rows 30 = 4 := by sorry

end NUMINAMATH_CALUDE_even_rows_in_pascal_triangle_l1353_135361


namespace NUMINAMATH_CALUDE_smallest_result_is_24_l1353_135321

def S : Finset ℕ := {2, 3, 5, 7, 11, 13}

def isConsecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

def validTriple (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ¬isConsecutive a b ∧ ¬isConsecutive b c ∧ ¬isConsecutive a c

def process (a b c : ℕ) : Finset ℕ :=
  {a * (b + c), b * (a + c), c * (a + b)}

theorem smallest_result_is_24 :
  ∀ a b c : ℕ, validTriple a b c →
    ∃ x ∈ process a b c, x ≥ 24 ∧ ∀ y ∈ process a b c, y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_is_24_l1353_135321


namespace NUMINAMATH_CALUDE_percent_equality_l1353_135357

theorem percent_equality (x : ℝ) : (70 / 100 * 600 = 40 / 100 * x) → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l1353_135357


namespace NUMINAMATH_CALUDE_three_fifths_of_five_times_nine_l1353_135328

theorem three_fifths_of_five_times_nine : (3 : ℚ) / 5 * (5 * 9) = 27 := by sorry

end NUMINAMATH_CALUDE_three_fifths_of_five_times_nine_l1353_135328


namespace NUMINAMATH_CALUDE_corner_difference_divisible_by_six_l1353_135351

/-- A 9x9 table filled with numbers from 1 to 81 -/
def Table := Fin 9 → Fin 9 → Fin 81

/-- Check if two cells are adjacent -/
def adjacent (i j k l : Fin 9) : Prop :=
  (i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ (j = l ∧ (i = k + 1 ∨ i + 1 = k))

/-- Check if a number is in a corner cell -/
def isCorner (i j : Fin 9) : Prop :=
  (i = 0 ∨ i = 8) ∧ (j = 0 ∨ j = 8)

/-- The main theorem -/
theorem corner_difference_divisible_by_six (t : Table) 
  (h1 : ∀ i j k l, adjacent i j k l → (t i j : ℕ) + 3 = t k l ∨ (t i j : ℕ) = (t k l : ℕ) + 3)
  (h2 : ∀ i j k l, i ≠ k ∨ j ≠ l → t i j ≠ t k l) :
  ∃ i j k l, isCorner i j ∧ isCorner k l ∧ 
    (∃ m : ℕ, (t i j : ℕ) - (t k l : ℕ) = 6 * m ∨ (t k l : ℕ) - (t i j : ℕ) = 6 * m) :=
sorry

end NUMINAMATH_CALUDE_corner_difference_divisible_by_six_l1353_135351


namespace NUMINAMATH_CALUDE_complex_division_proof_l1353_135378

theorem complex_division_proof : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_proof_l1353_135378


namespace NUMINAMATH_CALUDE_smallest_equal_probability_sum_l1353_135317

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The sum we want to compare with -/
def target_sum : ℕ := 1504

/-- The function to calculate the transformed sum -/
def transformed_sum (n : ℕ) : ℕ := 9 * n - target_sum

/-- The proposition that S is the smallest possible value satisfying the conditions -/
theorem smallest_equal_probability_sum : 
  ∃ (n : ℕ), n * sides ≥ target_sum ∧ 
  ∀ (m : ℕ), m < transformed_sum n → 
  ¬(∃ (k : ℕ), k * sides ≥ target_sum ∧ 
    transformed_sum k = m) :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_probability_sum_l1353_135317


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1353_135325

/-- 
Given a man and his son, where:
- The man is 28 years older than his son
- The son's present age is 26 years
Prove that the ratio of their ages in two years will be 2:1
-/
theorem man_son_age_ratio : 
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 28 →
  son_age = 26 →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1353_135325


namespace NUMINAMATH_CALUDE_root_in_interval_l1353_135393

noncomputable section

variables (a b : ℝ) (h : b ≥ 2*a) (h' : a > 0)

def f (x : ℝ) := 2*(a^x) - b^x

theorem root_in_interval :
  ∃ x, x ∈ Set.Ioo 0 1 ∧ f a b x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_root_in_interval_l1353_135393


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1353_135397

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1353_135397


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l1353_135318

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) : 
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧ 
    adult_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l1353_135318


namespace NUMINAMATH_CALUDE_ivanov_family_net_worth_l1353_135329

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The remaining mortgage balance of the Ivanov family in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The remaining car loan balance of the Ivanov family in rubles -/
def car_loan_balance : ℕ := 500000

/-- The debt the Ivanov family owes to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- The total assets of the Ivanov family -/
def total_assets : ℕ := apartment_value + car_value + bank_deposit + securities_value + liquid_cash

/-- The total liabilities of the Ivanov family -/
def total_liabilities : ℕ := mortgage_balance + car_loan_balance + debt_to_relatives

theorem ivanov_family_net_worth :
  ivanov_net_worth = total_assets - total_liabilities := by sorry

end NUMINAMATH_CALUDE_ivanov_family_net_worth_l1353_135329


namespace NUMINAMATH_CALUDE_angle_bisectors_may_not_form_triangle_l1353_135386

/-- Given a triangle with sides a = 2, b = 3, and c < 5, 
    prove that its angle bisectors may not satisfy the triangle inequality -/
theorem angle_bisectors_may_not_form_triangle :
  ∃ (c : ℝ), c < 5 ∧ 
  ∃ (ℓa ℓb ℓc : ℝ),
    (ℓa + ℓb ≤ ℓc ∨ ℓa + ℓc ≤ ℓb ∨ ℓb + ℓc ≤ ℓa) ∧
    ℓa = 3 / (1 + 2 / 7 * 3) ∧
    ℓb = 2 / (2 + 3 / 8 * 2) ∧
    ℓc = 0 := by
  sorry


end NUMINAMATH_CALUDE_angle_bisectors_may_not_form_triangle_l1353_135386


namespace NUMINAMATH_CALUDE_greatest_x_value_l1353_135350

theorem greatest_x_value (a b c d : ℤ) (x : ℝ) :
  x = (a + b * Real.sqrt c) / d →
  (7 * x) / 9 + 1 = 3 / x →
  (∀ y : ℝ, (7 * y) / 9 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -4158 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1353_135350


namespace NUMINAMATH_CALUDE_minimum_economic_loss_l1353_135383

def repair_times : List ℕ := [12, 17, 8, 18, 23, 30, 14]
def num_workers : ℕ := 3
def loss_per_minute : ℕ := 2

def distribute_work (times : List ℕ) (workers : ℕ) : List ℕ :=
  sorry

def calculate_waiting_time (distribution : List ℕ) : ℕ :=
  sorry

def economic_loss (waiting_time : ℕ) (loss_per_minute : ℕ) : ℕ :=
  sorry

theorem minimum_economic_loss :
  economic_loss (calculate_waiting_time (distribute_work repair_times num_workers)) loss_per_minute = 364 := by
  sorry

end NUMINAMATH_CALUDE_minimum_economic_loss_l1353_135383


namespace NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l1353_135313

/-- A positive integer whose decimal representation contains only ones -/
def AllOnesNumber (n : ℕ+) : Prop :=
  ∃ k : ℕ+, n.val = (10^k.val - 1) / 9

theorem divisibility_of_all_ones_number (N : ℕ+) 
  (h_all_ones : AllOnesNumber N) 
  (h_div_7 : 7 ∣ N.val) : 
  13 ∣ N.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_all_ones_number_l1353_135313


namespace NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1353_135358

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (4/7) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks needed to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (7/4) * kunks

theorem lunks_needed_for_twenty_apples : 
  lunks_for_apples 20 = 21 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_twenty_apples_l1353_135358


namespace NUMINAMATH_CALUDE_exists_fixed_point_l1353_135375

variable {α : Type*} [Finite α]

def IsIncreasing (f : Set α → Set α) : Prop :=
  ∀ X Y : Set α, X ⊆ Y → f X ⊆ f Y

theorem exists_fixed_point (f : Set α → Set α) (hf : IsIncreasing f) :
    ∃ H₀ : Set α, f H₀ = H₀ := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l1353_135375


namespace NUMINAMATH_CALUDE_expression_equals_four_l1353_135335

theorem expression_equals_four (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = y^2) :
  (x + 1/x) * (y - 1/y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l1353_135335


namespace NUMINAMATH_CALUDE_first_candidate_percentage_is_70_percent_l1353_135398

/-- The percentage of votes the first candidate received in an election with two candidates -/
def first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) : ℚ :=
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that the first candidate received 70% of the votes -/
theorem first_candidate_percentage_is_70_percent :
  first_candidate_percentage 800 240 = 70 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_is_70_percent_l1353_135398


namespace NUMINAMATH_CALUDE_eleventhNumberWithSumOfDigits12Is156_l1353_135306

-- Define a function to check if the sum of digits of a number is 12
def sumOfDigitsIs12 (n : ℕ) : Prop := sorry

-- Define a function to get the nth number in the sequence
def nthNumberWithSumOfDigits12 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem eleventhNumberWithSumOfDigits12Is156 : 
  nthNumberWithSumOfDigits12 11 = 156 := by sorry

end NUMINAMATH_CALUDE_eleventhNumberWithSumOfDigits12Is156_l1353_135306


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1353_135301

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l1353_135301


namespace NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l1353_135327

/-- Given two planar vectors a and b, prove that 3a + b equals the expected result. -/
theorem vector_sum_scalar_multiple (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (1, 0) → (3 • a) + b = (-2, 6) := by sorry

end NUMINAMATH_CALUDE_vector_sum_scalar_multiple_l1353_135327


namespace NUMINAMATH_CALUDE_pencil_distribution_l1353_135373

theorem pencil_distribution (n : Nat) (k : Nat) : 
  n = 6 → k = 3 → (Nat.choose (n - k + k - 1) (k - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1353_135373


namespace NUMINAMATH_CALUDE_complex_inequality_l1353_135333

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l1353_135333


namespace NUMINAMATH_CALUDE_H_surjective_l1353_135315

def H (x : ℝ) : ℝ := |3 * x + 1| - |x - 2|

theorem H_surjective : Function.Surjective H := by sorry

end NUMINAMATH_CALUDE_H_surjective_l1353_135315


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1353_135381

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 2) = |a (n + 5)| - a (n + 4)) : 
  ∃ N : ℕ, ∀ n ≥ N, a n = a (n + 9) :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1353_135381


namespace NUMINAMATH_CALUDE_race_distance_proof_l1353_135360

/-- The length of the race track in feet -/
def track_length : ℕ := 5000

/-- The distance Alex and Max run evenly at the start -/
def even_start : ℕ := 200

/-- The distance Alex gets ahead after the even start -/
def alex_first_lead : ℕ := 300

/-- The distance Alex gets ahead at the end -/
def alex_final_lead : ℕ := 440

/-- The distance left for Max to catch up at the end -/
def max_remaining : ℕ := 3890

/-- The unknown distance Max gets ahead of Alex -/
def max_lead : ℕ := 170

theorem race_distance_proof :
  even_start + alex_first_lead + max_lead + alex_final_lead = track_length - max_remaining :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1353_135360


namespace NUMINAMATH_CALUDE_article_cost_price_l1353_135310

theorem article_cost_price (C : ℝ) : C = 400 :=
  by
  have h1 : 1.05 * C - 2 = 0.95 * C * 1.10 := by sorry
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l1353_135310


namespace NUMINAMATH_CALUDE_notebook_distribution_ratio_l1353_135312

/-- Given a class where notebooks are distributed equally among children,
    prove that the ratio of notebooks per child to the number of children is 1/8 -/
theorem notebook_distribution_ratio 
  (C : ℕ) -- number of children
  (N : ℕ) -- number of notebooks per child
  (h1 : C * N = 512) -- total notebooks distributed
  (h2 : (C / 2) * 16 = 512) -- if children halved, each gets 16
  : N / C = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_ratio_l1353_135312


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l1353_135355

/-- The function f(x) = kx - 2ln(x) is monotonically increasing on [1, +∞) iff k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x ≥ 1, Monotone (λ x => k * x - 2 * Real.log x)) ↔ k ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l1353_135355


namespace NUMINAMATH_CALUDE_greg_additional_rotations_l1353_135340

/-- Represents the number of wheel rotations per block on flat ground. -/
def flatRotations : ℕ := 200

/-- Represents the number of wheel rotations per block uphill. -/
def uphillRotations : ℕ := 250

/-- Represents the number of blocks Greg has already ridden on flat ground. -/
def flatBlocksRidden : ℕ := 2

/-- Represents the number of blocks Greg has already ridden uphill. -/
def uphillBlocksRidden : ℕ := 1

/-- Represents the total number of wheel rotations Greg has already completed. -/
def rotationsCompleted : ℕ := 600

/-- Represents the number of additional uphill blocks Greg plans to ride. -/
def additionalUphillBlocks : ℕ := 3

/-- Represents the number of additional flat blocks Greg plans to ride. -/
def additionalFlatBlocks : ℕ := 2

/-- Represents the minimum total number of blocks Greg wants to ride. -/
def minTotalBlocks : ℕ := 8

/-- Theorem stating that Greg needs 550 more wheel rotations to reach his goal. -/
theorem greg_additional_rotations :
  let totalPlannedBlocks := flatBlocksRidden + uphillBlocksRidden + additionalFlatBlocks + additionalUphillBlocks
  let totalPlannedRotations := flatBlocksRidden * flatRotations + uphillBlocksRidden * uphillRotations +
                               additionalFlatBlocks * flatRotations + additionalUphillBlocks * uphillRotations
  totalPlannedBlocks ≥ minTotalBlocks ∧
  totalPlannedRotations - rotationsCompleted = 550 := by
  sorry


end NUMINAMATH_CALUDE_greg_additional_rotations_l1353_135340


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1353_135339

theorem vector_subtraction_scalar_multiplication :
  (⟨3, -7⟩ : ℝ × ℝ) - 3 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-3, 5⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1353_135339


namespace NUMINAMATH_CALUDE_cold_production_time_proof_l1353_135319

/-- The time (in minutes) it takes to produce each pot when the machine is cold. -/
def cold_production_time : ℝ := 6

/-- The time (in minutes) it takes to produce each pot when the machine is warm. -/
def warm_production_time : ℝ := 5

/-- The number of additional pots produced in the last hour compared to the first. -/
def additional_pots : ℕ := 2

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

theorem cold_production_time_proof :
  cold_production_time = 6 ∧
  warm_production_time = 5 ∧
  additional_pots = 2 ∧
  minutes_per_hour / cold_production_time + additional_pots = minutes_per_hour / warm_production_time :=
by sorry

end NUMINAMATH_CALUDE_cold_production_time_proof_l1353_135319


namespace NUMINAMATH_CALUDE_quadratic_root_difference_sum_l1353_135359

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 7 * x - 10 = 0

-- Define the condition for m (positive integer not divisible by the square of any prime)
def is_squarefree (m : ℕ) : Prop :=
  m > 0 ∧ ∀ p : ℕ, Prime p → (p^2 ∣ m → False)

-- Main theorem
theorem quadratic_root_difference_sum (m n : ℤ) : 
  (∃ r₁ r₂ : ℝ, quadratic_equation r₁ ∧ quadratic_equation r₂ ∧ |r₁ - r₂| = (Real.sqrt (m : ℝ)) / (n : ℝ)) →
  is_squarefree (m.natAbs) →
  m + n = 254 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_root_difference_sum_l1353_135359


namespace NUMINAMATH_CALUDE_integer_expression_l1353_135394

theorem integer_expression (n : ℕ) : ∃ k : ℤ, (n^5 : ℚ) / 5 + (n^3 : ℚ) / 3 + (7 * n : ℚ) / 15 = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l1353_135394


namespace NUMINAMATH_CALUDE_polynomial_root_not_all_real_l1353_135316

theorem polynomial_root_not_all_real (a b c d e : ℝ) :
  2 * a^2 < 5 * b →
  ∃ z : ℂ, z^5 + a*z^4 + b*z^3 + c*z^2 + d*z + e = 0 ∧ z.im ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_not_all_real_l1353_135316


namespace NUMINAMATH_CALUDE_square_root_of_25_l1353_135370

theorem square_root_of_25 : (Real.sqrt 25) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_25_l1353_135370


namespace NUMINAMATH_CALUDE_sine_shift_right_l1353_135332

/-- Shifting a sine function to the right by π/6 units -/
theorem sine_shift_right (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t + π / 6)
  let g (t : ℝ) := f (t - π / 6)
  g x = Real.sin (2 * x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_right_l1353_135332


namespace NUMINAMATH_CALUDE_complex_fraction_theorem_l1353_135395

theorem complex_fraction_theorem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = -2.871 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_theorem_l1353_135395


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l1353_135365

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 125) ∧ is_valid 125 :=
by sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l1353_135365


namespace NUMINAMATH_CALUDE_stop_signs_per_mile_l1353_135363

-- Define the distance traveled
def distance : ℝ := 5 + 2

-- Define the number of stop signs encountered
def stop_signs : ℕ := 17 - 3

-- Theorem to prove
theorem stop_signs_per_mile : (stop_signs : ℝ) / distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_stop_signs_per_mile_l1353_135363


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1353_135314

/-- Quadratic function -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of N -/
def N (a b c : ℝ) : ℝ := |a + b + c| + |2*a - b|

/-- Definition of M -/
def M (a b c : ℝ) : ℝ := |a - b + c| + |2*a + b|

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : -b/(2*a) > 1) 
  (h3 : f a b c 0 = c) 
  (h4 : ∃ x, f a b c x > 0) : 
  M a b c < N a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1353_135314


namespace NUMINAMATH_CALUDE_track_laying_equation_l1353_135385

theorem track_laying_equation (x : ℝ) (h : x > 0) :
  (6000 / x - 6000 / (x + 20) = 15) ↔
  (∃ (original_days revised_days : ℝ),
    original_days > 0 ∧
    revised_days > 0 ∧
    original_days = 6000 / x ∧
    revised_days = 6000 / (x + 20) ∧
    original_days - revised_days = 15) :=
by sorry

end NUMINAMATH_CALUDE_track_laying_equation_l1353_135385


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1353_135330

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 15 ∧ y = 56 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1353_135330


namespace NUMINAMATH_CALUDE_base8_145_equals_101_in_base10_l1353_135344

-- Define a function to convert a base-8 number to base-10
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Define the base-8 number 145
def base8Number : List Nat := [5, 4, 1]

-- State the theorem
theorem base8_145_equals_101_in_base10 :
  base8ToBase10 base8Number = 101 := by
  sorry

end NUMINAMATH_CALUDE_base8_145_equals_101_in_base10_l1353_135344


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1353_135341

/-- Given a hyperbola with equation x²/m - y²/3 = 1 where m > 0,
    if one of its asymptotic lines is y = (1/2)x, then m = 12 -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ (x y : ℝ), x^2 / m - y^2 / 3 = 1 ∧ y = (1/2) * x) → m = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1353_135341


namespace NUMINAMATH_CALUDE_system_solution_l1353_135369

theorem system_solution (a : ℚ) :
  (∃! x y : ℚ, 2*x + 3*y = 5 ∧ x - y = 2 ∧ x + 4*y = a) ↔ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1353_135369


namespace NUMINAMATH_CALUDE_not_perfect_square_l1353_135376

theorem not_perfect_square (n : ℤ) (h : n > 11) :
  ¬ ∃ m : ℤ, n^2 - 19*n + 89 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1353_135376


namespace NUMINAMATH_CALUDE_other_integer_17_or_21_l1353_135347

/-- Two consecutive odd integers with a sum of at least 36, one being 19 -/
structure ConsecutiveOddIntegers where
  n : ℤ
  sum_at_least_36 : n + (n + 2) ≥ 36
  one_is_19 : n = 19 ∨ n + 2 = 19

/-- The other integer is either 17 or 21 -/
theorem other_integer_17_or_21 (x : ConsecutiveOddIntegers) : 
  x.n = 21 ∨ x.n = 17 := by
  sorry


end NUMINAMATH_CALUDE_other_integer_17_or_21_l1353_135347


namespace NUMINAMATH_CALUDE_parallelogram_sides_l1353_135367

theorem parallelogram_sides (x y : ℝ) : 
  (2*x + 3 = 9) ∧ (8*y - 1 = 7) → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sides_l1353_135367


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1353_135343

theorem quadratic_equation_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*m*x₁ - m - 1 = 0) ∧ 
  (x₂^2 - 2*m*x₂ - m - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1353_135343


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l1353_135303

/-- The total cost in billions of dollars to send a person to Mars -/
def total_cost : ℝ := 30

/-- The number of people in millions sharing the cost -/
def number_of_people : ℝ := 300

/-- Each person's share of the cost in dollars -/
def cost_per_person : ℝ := 100

/-- Theorem stating that if the total cost in billions of dollars is shared equally among the given number of people in millions, each person's share is the specified amount in dollars -/
theorem mars_mission_cost_share : 
  (total_cost * 1000) / number_of_people = cost_per_person := by
  sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l1353_135303


namespace NUMINAMATH_CALUDE_field_width_calculation_l1353_135396

/-- Proves that the width of each field is 250 meters -/
theorem field_width_calculation (num_fields : ℕ) (field_length : ℝ) (total_area_km2 : ℝ) :
  num_fields = 8 →
  field_length = 300 →
  total_area_km2 = 0.6 →
  ∃ (width : ℝ), width = 250 ∧ 
    (num_fields * field_length * width = total_area_km2 * 1000000) :=
by sorry

end NUMINAMATH_CALUDE_field_width_calculation_l1353_135396


namespace NUMINAMATH_CALUDE_probability_divisible_by_5_l1353_135399

/-- A three-digit number is an integer between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers divisible by 5. -/
def CountDivisibleBy5 : ℕ := 180

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being divisible by 5 is 1/5. -/
theorem probability_divisible_by_5 :
  (CountDivisibleBy5 : ℚ) / TotalThreeDigitNumbers = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_divisible_by_5_l1353_135399


namespace NUMINAMATH_CALUDE_complex_number_solution_l1353_135366

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_solution (z : ℂ) 
  (h1 : is_purely_imaginary (z - 1))
  (h2 : is_purely_imaginary ((z + 1)^2 - 8*I)) :
  z = 1 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l1353_135366


namespace NUMINAMATH_CALUDE_average_of_middle_two_l1353_135352

theorem average_of_middle_two (numbers : Fin 6 → ℝ) 
  (h_total_avg : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 6.40)
  (h_first_two_avg : (numbers 0 + numbers 1) / 2 = 6.2)
  (h_last_two_avg : (numbers 4 + numbers 5) / 2 = 6.9) :
  (numbers 2 + numbers 3) / 2 = 6.1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_middle_two_l1353_135352


namespace NUMINAMATH_CALUDE_cubic_unit_circle_roots_l1353_135326

theorem cubic_unit_circle_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∃ w₁ w₂ w₃ : ℂ, 
    (w₁^3 + Complex.abs a * w₁^2 + Complex.abs b * w₁ + Complex.abs c = 0) ∧
    (w₂^3 + Complex.abs a * w₂^2 + Complex.abs b * w₂ + Complex.abs c = 0) ∧
    (w₃^3 + Complex.abs a * w₃^2 + Complex.abs b * w₃ + Complex.abs c = 0) ∧
    Complex.abs w₁ = 1 ∧ Complex.abs w₂ = 1 ∧ Complex.abs w₃ = 1) :=
by sorry


end NUMINAMATH_CALUDE_cubic_unit_circle_roots_l1353_135326


namespace NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1353_135349

theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 8
  let base_radius : ℝ := r * sector_fraction / 2
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = 4.6875 * Real.pi * Real.sqrt 21.9375 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_from_circle_sector_l1353_135349


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1353_135307

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1353_135307


namespace NUMINAMATH_CALUDE_factorization_valid_l1353_135311

theorem factorization_valid (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l1353_135311


namespace NUMINAMATH_CALUDE_haunted_mansion_paths_l1353_135324

theorem haunted_mansion_paths (n : ℕ) (h : n = 8) : n * (n - 1) * (n - 2) = 336 := by
  sorry

end NUMINAMATH_CALUDE_haunted_mansion_paths_l1353_135324


namespace NUMINAMATH_CALUDE_quadratic_roots_range_quadratic_roots_value_l1353_135379

/-- The quadratic equation x^2 + 3x + k - 2 = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 + 3*x + k - 2 = 0

/-- The equation has real roots -/
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic x k

/-- The roots of the equation satisfy (x_1 + 1)(x_2 + 1) = -1 -/
def roots_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic x₁ k ∧ quadratic x₂ k ∧ (x₁ + 1) * (x₂ + 1) = -1

theorem quadratic_roots_range (k : ℝ) :
  has_real_roots k → k ≤ 17/4 :=
sorry

theorem quadratic_roots_value (k : ℝ) :
  has_real_roots k → roots_condition k → k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_quadratic_roots_value_l1353_135379


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1353_135364

theorem complex_number_in_first_quadrant (z : ℂ) : 
  z / (z - Complex.I) = Complex.I → 
  (z.re > 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1353_135364


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l1353_135356

theorem clothing_tax_rate (total_spent : ℝ) (clothing_spent : ℝ) (food_spent : ℝ) (other_spent : ℝ)
  (clothing_tax : ℝ) (other_tax : ℝ) (total_tax : ℝ) :
  clothing_spent = 0.4 * total_spent →
  food_spent = 0.3 * total_spent →
  other_spent = 0.3 * total_spent →
  other_tax = 0.08 * other_spent →
  total_tax = 0.04 * total_spent →
  total_tax = clothing_tax + other_tax →
  clothing_tax / clothing_spent = 0.04 :=
by sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l1353_135356


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_l1353_135372

/-- Represents a three-dimensional shape --/
structure Shape3D where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- A cube --/
def cube : Shape3D :=
  { edges := 12, corners := 8, faces := 6 }

/-- A square pyramid --/
def square_pyramid : Shape3D :=
  { edges := 8, corners := 5, faces := 5 }

/-- The shape formed by placing a square pyramid on one face of a cube --/
def cube_with_pyramid : Shape3D :=
  { edges := cube.edges + 4, -- 4 new edges from pyramid apex
    corners := cube.corners + 1, -- 1 new corner (pyramid apex)
    faces := cube.faces + square_pyramid.faces - 1 } -- -1 for shared base

/-- The sum of edges, corners, and faces of the combined shape --/
def combined_sum (s : Shape3D) : ℕ :=
  s.edges + s.corners + s.faces

/-- Theorem stating that the sum of edges, corners, and faces of the combined shape is 34 --/
theorem cube_pyramid_sum :
  combined_sum cube_with_pyramid = 34 := by
  sorry


end NUMINAMATH_CALUDE_cube_pyramid_sum_l1353_135372


namespace NUMINAMATH_CALUDE_first_meeting_turns_l1353_135338

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The relative clockwise movement per turn -/
def relative_move : ℕ := alice_move - (n - bob_move)

theorem first_meeting_turns : 
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0) → 
  (∃ m : ℕ, m > 0 ∧ (m * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → l ≥ m) →
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) →
  (∀ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) → k = 5 := by
  sorry

#eval relative_move -- Should output 3

end NUMINAMATH_CALUDE_first_meeting_turns_l1353_135338


namespace NUMINAMATH_CALUDE_triangle_properties_l1353_135374

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  (Real.sin t.B = 4/5) ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → t.b^2 / (x^2 + y^2) ≥ 2/5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1353_135374


namespace NUMINAMATH_CALUDE_abs_difference_over_sum_equals_sqrt_three_sevenths_l1353_135388

theorem abs_difference_over_sum_equals_sqrt_three_sevenths
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (3/7) :=
by sorry

end NUMINAMATH_CALUDE_abs_difference_over_sum_equals_sqrt_three_sevenths_l1353_135388


namespace NUMINAMATH_CALUDE_frustum_smaller_base_radius_l1353_135380

/-- A frustum with the given properties has a smaller base radius of 7 -/
theorem frustum_smaller_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem context)
  (2 * π * r) * 3 = 2 * π * (3 * r) → -- one circumference is three times the other
  3 = 3 → -- slant height is 3
  π * (r + 3 * r) * 3 = 84 * π → -- lateral surface area formula
  r = 7 := by
sorry


end NUMINAMATH_CALUDE_frustum_smaller_base_radius_l1353_135380


namespace NUMINAMATH_CALUDE_married_men_fraction_l1353_135354

-- Define the structure of the gathering
structure Gathering where
  single_women : ℕ
  married_couples : ℕ

-- Define the probability of a woman being single
def prob_single_woman (g : Gathering) : ℚ :=
  g.single_women / (g.single_women + g.married_couples)

-- Define the fraction of married men in the gathering
def fraction_married_men (g : Gathering) : ℚ :=
  g.married_couples / (g.single_women + 2 * g.married_couples)

-- Theorem statement
theorem married_men_fraction (g : Gathering) :
  prob_single_woman g = 1/3 → fraction_married_men g = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1353_135354


namespace NUMINAMATH_CALUDE_angle_measure_in_pentagon_l1353_135371

structure Pentagon where
  F : ℝ
  G : ℝ
  H : ℝ
  I : ℝ
  J : ℝ

def is_convex_pentagon (p : Pentagon) : Prop :=
  p.F + p.G + p.H + p.I + p.J = 540

theorem angle_measure_in_pentagon (p : Pentagon) 
  (convex : is_convex_pentagon p)
  (fgh_congruent : p.F = p.G ∧ p.G = p.H)
  (ij_congruent : p.I = p.J)
  (f_less_than_i : p.F + 80 = p.I) :
  p.I = 156 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_pentagon_l1353_135371


namespace NUMINAMATH_CALUDE_arithmetic_sum_modulo_15_l1353_135389

/-- The sum of an arithmetic sequence modulo m -/
def arithmetic_sum_mod (a₁ aₙ d n m : ℕ) : ℕ :=
  ((n * (a₁ + aₙ)) / 2) % m

/-- The number of terms in an arithmetic sequence -/
def arithmetic_terms (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sum_modulo_15 :
  let a₁ := 2  -- First term
  let aₙ := 102  -- Last term
  let d := 5   -- Common difference
  let m := 15  -- Modulus
  let n := arithmetic_terms a₁ aₙ d
  arithmetic_sum_mod a₁ aₙ d n m = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_modulo_15_l1353_135389


namespace NUMINAMATH_CALUDE_union_and_intersection_of_A_and_B_l1353_135391

variable (a : ℝ)

def A : Set ℝ := {x | (x - 3) * (x - a) = 0}
def B : Set ℝ := {x | (x - 4) * (x - 1) = 0}

theorem union_and_intersection_of_A_and_B :
  (a = 3 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = ∅)) ∧
  (a = 1 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {1})) ∧
  (a = 4 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {4})) ∧
  (a ≠ 1 ∧ a ≠ 3 ∧ a ≠ 4 → (A a ∪ B = {1, 3, 4, a} ∧ A a ∩ B = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_union_and_intersection_of_A_and_B_l1353_135391


namespace NUMINAMATH_CALUDE_simplify_expression_l1353_135382

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (2*x + 6)*(3*x - 2) = -3*x^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1353_135382


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l1353_135348

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_five_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_4th : a 4 = 11)
  (h_5th : a 5 = 15)
  (h_6th : a 6 = 19) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 35 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l1353_135348


namespace NUMINAMATH_CALUDE_range_of_slopes_tangent_line_equation_l1353_135377

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Theorem for the range of slopes
theorem range_of_slopes :
  ∀ x ∈ Set.Icc (-2) 1, -3 ≤ f' x ∧ f' x ≤ 9 :=
sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ a b : ℝ,
    f' a = -3 ∧
    f a = b ∧
    (∀ x y : ℝ, line_l x y → (3*x + y + 6 = 0 → x = a ∧ y = b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_slopes_tangent_line_equation_l1353_135377


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_by_five_l1353_135322

theorem least_addition_for_divisibility_by_five (n : ℕ) (h : n = 821562) :
  ∃ k : ℕ, k = 3 ∧ (n + k) % 5 = 0 ∧ ∀ m : ℕ, m < k → (n + m) % 5 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_by_five_l1353_135322


namespace NUMINAMATH_CALUDE_junior_score_l1353_135334

theorem junior_score (n : ℝ) (h_n : n > 0) : 
  let junior_percent : ℝ := 0.2
  let senior_percent : ℝ := 0.8
  let total_average : ℝ := 85
  let senior_average : ℝ := 84
  let junior_score := (total_average * n - senior_average * senior_percent * n) / (junior_percent * n)
  junior_score = 89 := by sorry

end NUMINAMATH_CALUDE_junior_score_l1353_135334


namespace NUMINAMATH_CALUDE_total_books_l1353_135308

theorem total_books (stu_books : ℝ) (albert_ratio : ℝ) : 
  stu_books = 9 → 
  albert_ratio = 4.5 → 
  stu_books + albert_ratio * stu_books = 49.5 := by
sorry

end NUMINAMATH_CALUDE_total_books_l1353_135308


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l1353_135342

/-- Given a circular piece of tissue and an electron microscope, 
    calculates the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem stating that for the given conditions, 
    the magnified diameter is 2 centimeters. -/
theorem magnified_tissue_diameter :
  let actual_diameter : ℝ := 0.002
  let magnification_factor : ℝ := 1000
  magnified_diameter actual_diameter magnification_factor = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l1353_135342


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l1353_135320

theorem mean_equality_implies_y_value : ℝ → Prop :=
  fun y =>
    (((6 : ℝ) + 10 + 14 + 22) / 4 = (15 + y) / 2) → y = 11

-- The proof is omitted
theorem mean_equality_implies_y_value_proof : mean_equality_implies_y_value 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_mean_equality_implies_y_value_proof_l1353_135320


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1353_135390

theorem min_value_of_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 30 ∧ ∃ y > 10, y^2 / (y - 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1353_135390
