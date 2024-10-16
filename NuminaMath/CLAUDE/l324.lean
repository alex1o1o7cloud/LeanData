import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_x_l324_32430

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l324_32430


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l324_32482

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Proves that a 90m by 60m plot with poles 5m apart needs 60 poles -/
theorem rectangular_plot_poles : fence_poles 90 60 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l324_32482


namespace NUMINAMATH_CALUDE_inequality_proof_l324_32420

theorem inequality_proof (x y : ℝ) : 
  (∀ x, |x| + |x - 3| < x + 6 ↔ -1 < x ∧ x < 9) →
  x > 0 →
  y > 0 →
  9*x + y - 1 = 0 →
  x + y ≥ 16*x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l324_32420


namespace NUMINAMATH_CALUDE_angle_terminal_side_l324_32479

/-- Given an angle α whose terminal side passes through the point (m, -3) 
    and whose cosine is -4/5, prove that m = -4. -/
theorem angle_terminal_side (α : Real) (m : Real) : 
  (∃ (x y : Real), x = m ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = -4/5 →
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l324_32479


namespace NUMINAMATH_CALUDE_symmetry_of_A_and_D_l324_32427

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - 2 * y - 4 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define P as the midpoint of AB
def P_midpoint (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1/2

-- Define Q on line l
def Q_on_l : Prop := line_l 4 0

-- Define A between B and Q
def A_between_B_Q (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A.1 = t * B.1 + (1 - t) * 4 ∧ A.2 = t * B.2

-- Define the right focus F
def right_focus (F : ℝ × ℝ) : Prop := F.1 = 1 ∧ F.2 = 0

-- Define D as the intersection of BF and C
def D_intersection (B D F : ℝ × ℝ) : Prop :=
  ellipse_C D.1 D.2 ∧ ∃ t : ℝ, D.1 = B.1 + t * (F.1 - B.1) ∧ D.2 = B.2 + t * (F.2 - B.2)

-- Define symmetry with respect to x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop := A.1 = D.1 ∧ A.2 = -D.2

-- Main theorem
theorem symmetry_of_A_and_D (A B D F : ℝ × ℝ) :
  intersection_points A B →
  P_midpoint A B →
  Q_on_l →
  A_between_B_Q A B →
  right_focus F →
  D_intersection B D F →
  symmetric_x_axis A D :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_A_and_D_l324_32427


namespace NUMINAMATH_CALUDE_problem_statement_l324_32407

theorem problem_statement (x y : ℝ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : Real.log x / Real.log 4 ^ 3 + Real.log y / Real.log 5 ^ 3 + 9 = 
        12 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) : 
  x^2 + y^2 = 64 + 25 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l324_32407


namespace NUMINAMATH_CALUDE_max_cuts_length_30x30_225pieces_l324_32435

/-- Represents a square board with cuts along grid lines -/
structure Board where
  size : ℕ
  pieces : ℕ
  cuts_length : ℕ

/-- The maximum possible total length of cuts for a given board configuration -/
def max_cuts_length (b : Board) : ℕ :=
  (b.pieces * 10 - 4 * b.size) / 2

/-- Theorem stating the maximum possible total length of cuts for the given board -/
theorem max_cuts_length_30x30_225pieces :
  ∃ (b : Board), b.size = 30 ∧ b.pieces = 225 ∧ max_cuts_length b = 1065 := by
  sorry

end NUMINAMATH_CALUDE_max_cuts_length_30x30_225pieces_l324_32435


namespace NUMINAMATH_CALUDE_retail_price_calculation_l324_32480

theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  wholesale_price = 90 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ retail_price : ℝ, 
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l324_32480


namespace NUMINAMATH_CALUDE_square_side_length_l324_32485

/-- Given a rectangle composed of rectangles R1 and R2, and squares S1, S2, and S3,
    this theorem proves that the side length of S2 is 875 units. -/
theorem square_side_length (total_width total_height : ℕ) (s2 s3 : ℕ) :
  total_width = 4020 →
  total_height = 2160 →
  s3 = s2 + 110 →
  ∃ (r : ℕ), 
    2 * r + s2 = total_height ∧
    2 * r + 3 * s2 + 110 = total_width →
  s2 = 875 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l324_32485


namespace NUMINAMATH_CALUDE_positive_root_of_cubic_equation_l324_32459

theorem positive_root_of_cubic_equation :
  ∃ x : ℝ, x > 0 ∧ x^3 - 5*x^2 + 2*x - Real.sqrt 3 = 0 :=
by
  use 3 + Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_positive_root_of_cubic_equation_l324_32459


namespace NUMINAMATH_CALUDE_finite_difference_polynomial_l324_32447

/-- The finite difference operator -/
def finite_difference (f : ℕ → ℚ) : ℕ → ℚ := λ x => f (x + 1) - f x

/-- The n-th finite difference -/
def nth_finite_difference (n : ℕ) (f : ℕ → ℚ) : ℕ → ℚ :=
  match n with
  | 0 => f
  | n + 1 => finite_difference (nth_finite_difference n f)

/-- Polynomial of degree m -/
def polynomial_degree_m (m : ℕ) (coeffs : Fin (m + 1) → ℚ) : ℕ → ℚ :=
  λ x => (Finset.range (m + 1)).sum (λ i => coeffs i * x^i)

theorem finite_difference_polynomial (m n : ℕ) (coeffs : Fin (m + 1) → ℚ) :
  (m < n → ∀ x, nth_finite_difference n (polynomial_degree_m m coeffs) x = 0) ∧
  (∀ x, nth_finite_difference m (polynomial_degree_m m coeffs) x = m.factorial * coeffs m) :=
sorry

end NUMINAMATH_CALUDE_finite_difference_polynomial_l324_32447


namespace NUMINAMATH_CALUDE_triangle_area_l324_32402

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3 under the following conditions:
    - (2b - √3c) / (√3a) = cos(C) / cos(A)
    - B = π/6
    - The median AM on side BC has length √7 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (M : ℝ) : 
  (2 * b - Real.sqrt 3 * c) / (Real.sqrt 3 * a) = Real.cos C / Real.cos A →
  B = π / 6 →
  M = Real.sqrt 7 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l324_32402


namespace NUMINAMATH_CALUDE_negation_of_positive_product_l324_32408

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_positive_product_l324_32408


namespace NUMINAMATH_CALUDE_superhero_speed_in_mph_l324_32432

-- Define the superhero's speed in kilometers per minute
def speed_km_per_min : ℝ := 1000

-- Define the conversion factor from km to miles
def km_to_miles : ℝ := 0.6

-- Define the number of minutes in an hour
def minutes_per_hour : ℝ := 60

-- Theorem statement
theorem superhero_speed_in_mph : 
  speed_km_per_min * km_to_miles * minutes_per_hour = 36000 := by
  sorry

end NUMINAMATH_CALUDE_superhero_speed_in_mph_l324_32432


namespace NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_27_squared_l324_32411

theorem units_digit_of_42_cubed_plus_27_squared : ∃ n : ℕ, 42^3 + 27^2 = 10 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_27_squared_l324_32411


namespace NUMINAMATH_CALUDE_function_characterization_l324_32439

def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b c : ℤ, a + b + c = 0 →
    f a ^ 2 + f b ^ 2 + f c ^ 2 = 2 * f a * f b + 2 * f b * f c + 2 * f c * f a

def IsZeroFunction (f : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, f x = 0

def IsQuadraticFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, f x = k * x ^ 2

def IsEvenOddFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ, 
    (Even x → f x = 0) ∧ 
    (Odd x → f x = k)

def IsModFourFunction (f : ℤ → ℤ) : Prop :=
  ∃ k : ℤ, ∀ x : ℤ,
    (x % 4 = 0 → f x = 0) ∧
    (x % 4 = 1 → f x = k) ∧
    (x % 4 = 2 → f x = 4 * k)

theorem function_characterization (f : ℤ → ℤ) : 
  SatisfiesEquation f → 
    IsZeroFunction f ∨ 
    IsQuadraticFunction f ∨ 
    IsEvenOddFunction f ∨ 
    IsModFourFunction f := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l324_32439


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l324_32483

/-- The rate of drawing barbed wire per meter given the conditions of the problem -/
theorem barbed_wire_rate (field_area : ℝ) (wire_extension : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ)
  (h1 : field_area = 3136)
  (h2 : wire_extension = 3)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : total_cost = 732.6) :
  (total_cost / (4 * Real.sqrt field_area + wire_extension - num_gates * gate_width)) = 3.256 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l324_32483


namespace NUMINAMATH_CALUDE_arcsin_negative_half_l324_32442

theorem arcsin_negative_half : Real.arcsin (-1/2) = -π/6 := by sorry

end NUMINAMATH_CALUDE_arcsin_negative_half_l324_32442


namespace NUMINAMATH_CALUDE_unhappy_redheads_ratio_l324_32493

theorem unhappy_redheads_ratio 
  (x y z : ℕ) -- x: happy subjects, y: redheads, z: total subjects
  (h1 : (40 : ℚ) / 100 * x = (60 : ℚ) / 100 * y) -- Condition 1
  (h2 : z = x + (40 : ℚ) / 100 * y) -- Condition 2
  : (y - ((40 : ℚ) / 100 * y).floor) / z = 4 / 19 := by
  sorry


end NUMINAMATH_CALUDE_unhappy_redheads_ratio_l324_32493


namespace NUMINAMATH_CALUDE_magnitude_relationship_l324_32490

-- Define the equations for a, b, and c
def equation_a (x : ℝ) : Prop := 2^x + x = 1
def equation_b (x : ℝ) : Prop := 2^x + x = 2
def equation_c (x : ℝ) : Prop := 3^x + x = 2

-- State the theorem
theorem magnitude_relationship (a b c : ℝ) 
  (ha : equation_a a) (hb : equation_b b) (hc : equation_c c) : 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l324_32490


namespace NUMINAMATH_CALUDE_canteen_leakage_rate_l324_32444

/-- Calculates the rate of water leakage from a canteen during a hike. -/
theorem canteen_leakage_rate 
  (total_distance : ℝ) 
  (initial_water : ℝ) 
  (time_taken : ℝ) 
  (remaining_water : ℝ) 
  (last_mile_consumption : ℝ) 
  (first_miles_rate : ℝ) 
  (h1 : total_distance = 7) 
  (h2 : initial_water = 11) 
  (h3 : time_taken = 3) 
  (h4 : remaining_water = 2) 
  (h5 : last_mile_consumption = 3) 
  (h6 : first_miles_rate = 0.5) :
  (initial_water - remaining_water - (first_miles_rate * (total_distance - 1) + last_mile_consumption)) / time_taken = 1 := by
  sorry


end NUMINAMATH_CALUDE_canteen_leakage_rate_l324_32444


namespace NUMINAMATH_CALUDE_other_number_proof_l324_32460

theorem other_number_proof (x : Float) : 
  (0.5 : Float) = x + 0.33333333333333337 → x = 0.16666666666666663 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l324_32460


namespace NUMINAMATH_CALUDE_choose_two_from_four_l324_32469

theorem choose_two_from_four : Nat.choose 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_choose_two_from_four_l324_32469


namespace NUMINAMATH_CALUDE_gardener_roses_order_l324_32410

/-- The number of roses ordered by the gardener -/
def roses : ℕ := 320

/-- The number of tulips ordered -/
def tulips : ℕ := 250

/-- The number of carnations ordered -/
def carnations : ℕ := 375

/-- The cost of each flower in euros -/
def flower_cost : ℕ := 2

/-- The total expenses in euros -/
def total_expenses : ℕ := 1890

theorem gardener_roses_order :
  roses = (total_expenses - (tulips + carnations) * flower_cost) / flower_cost := by
  sorry

end NUMINAMATH_CALUDE_gardener_roses_order_l324_32410


namespace NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l324_32472

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : d > 0) :
  let frac := (72 : ℚ) / ((2^4 * 3^6) : ℚ)
  ∃ (a b c : ℕ) (r : ℚ),
    frac = (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + r ∧
    0 < a ∧ a < 10 ∧
    0 < b ∧ b < 10 ∧
    0 < c ∧ c < 10 ∧
    0 ≤ r ∧ r < 1/1000 :=
by sorry

end NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l324_32472


namespace NUMINAMATH_CALUDE_sector_angle_measure_l324_32477

theorem sector_angle_measure (r : ℝ) (l : ℝ) :
  (2 * r + l = 12) →  -- circumference condition
  (1 / 2 * l * r = 8) →  -- area condition
  (l / r = 1 ∨ l / r = 4) :=  -- radian measure of central angle
by sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l324_32477


namespace NUMINAMATH_CALUDE_exist_positive_integers_with_nonzero_integer_roots_l324_32449

theorem exist_positive_integers_with_nonzero_integer_roots :
  ∃ (a b c : ℕ+), 
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 + (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 + (b:ℤ) * y - (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x + (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y + (c:ℤ) = 0) ∧
    (∀ (x : ℤ), (x ≠ 0 ∧ (a:ℤ) * x^2 - (b:ℤ) * x - (c:ℤ) = 0) → 
      ∃ (y : ℤ), y ≠ 0 ∧ y ≠ x ∧ (a:ℤ) * y^2 - (b:ℤ) * y - (c:ℤ) = 0) :=
by sorry

end NUMINAMATH_CALUDE_exist_positive_integers_with_nonzero_integer_roots_l324_32449


namespace NUMINAMATH_CALUDE_burglars_money_min_burglars_money_l324_32489

def x (a n : ℕ) : ℚ := (a / 4 : ℚ) * (1 - (1 / 3 : ℚ) ^ n)

theorem burglars_money (a : ℕ) : 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) →
  a ≥ 4 * 3^2012 :=
sorry

theorem min_burglars_money : 
  ∃ a : ℕ, a = 4 * 3^2012 ∧ 
  (∀ n : ℕ, n ≤ 2012 → (x a n).num % (x a n).den = 0 ∧ ((a : ℚ) - x a n).num % ((a : ℚ) - x a n).den = 0) ∧
  (∀ b : ℕ, b < a → ∃ n : ℕ, n ≤ 2012 ∧ ((x b n).num % (x b n).den ≠ 0 ∨ ((b : ℚ) - x b n).num % ((b : ℚ) - x b n).den ≠ 0)) :=
sorry

end NUMINAMATH_CALUDE_burglars_money_min_burglars_money_l324_32489


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l324_32491

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l324_32491


namespace NUMINAMATH_CALUDE_max_money_is_twelve_dollars_l324_32431

/-- Represents the recycling scenario with given rates and collected items -/
structure RecyclingScenario where
  can_rate : Rat -- Money received for 12 cans
  newspaper_rate : Rat -- Money received for 5 kg of newspapers
  bottle_rate : Rat -- Money received for 3 glass bottles
  weight_limit : Rat -- Weight limit in kg
  cans_collected : Nat -- Number of cans collected
  can_weight : Rat -- Weight of each can in kg
  newspapers_collected : Rat -- Weight of newspapers collected in kg
  bottles_collected : Nat -- Number of bottles collected
  bottle_weight : Rat -- Weight of each bottle in kg

/-- Calculates the maximum money received from recycling -/
noncomputable def max_money_received (scenario : RecyclingScenario) : Rat :=
  sorry

/-- Theorem stating that the maximum money received is $12.00 -/
theorem max_money_is_twelve_dollars (scenario : RecyclingScenario) 
  (h1 : scenario.can_rate = 1/2)
  (h2 : scenario.newspaper_rate = 3/2)
  (h3 : scenario.bottle_rate = 9/10)
  (h4 : scenario.weight_limit = 25)
  (h5 : scenario.cans_collected = 144)
  (h6 : scenario.can_weight = 3/100)
  (h7 : scenario.newspapers_collected = 20)
  (h8 : scenario.bottles_collected = 30)
  (h9 : scenario.bottle_weight = 1/2) :
  max_money_received scenario = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_money_is_twelve_dollars_l324_32431


namespace NUMINAMATH_CALUDE_solve_system_l324_32436

theorem solve_system (a b : ℝ) 
  (eq1 : 2020*a + 2030*b = 2050)
  (eq2 : 2030*a + 2040*b = 2060) : 
  a - b = -5 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l324_32436


namespace NUMINAMATH_CALUDE_total_crosswalk_lines_l324_32448

/-- Given 5 intersections, 4 crosswalks per intersection, and 20 lines per crosswalk,
    the total number of lines in all crosswalks is 400. -/
theorem total_crosswalk_lines
  (num_intersections : ℕ)
  (crosswalks_per_intersection : ℕ)
  (lines_per_crosswalk : ℕ)
  (h1 : num_intersections = 5)
  (h2 : crosswalks_per_intersection = 4)
  (h3 : lines_per_crosswalk = 20) :
  num_intersections * crosswalks_per_intersection * lines_per_crosswalk = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_crosswalk_lines_l324_32448


namespace NUMINAMATH_CALUDE_intersection_A_B_l324_32466

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x < 0}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l324_32466


namespace NUMINAMATH_CALUDE_f_bound_and_g_monotonicity_l324_32424

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

theorem f_bound_and_g_monotonicity :
  (∃ c : ℝ, c = -1 ∧ ∀ x > 0, f x ≤ 2 * x + c) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioo 0 a)) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioi a)) :=
sorry

end NUMINAMATH_CALUDE_f_bound_and_g_monotonicity_l324_32424


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_theorem_l324_32425

/-- A symmetric trapezoid inscribed in a circle -/
structure SymmetricTrapezoid (R : ℝ) where
  x : ℝ
  h_x_range : 0 ≤ x ∧ x ≤ 2*R

/-- The function y for a symmetric trapezoid -/
def y (R : ℝ) (t : SymmetricTrapezoid R) : ℝ :=
  (t.x - R)^2 + 3*R^2

theorem symmetric_trapezoid_theorem (R : ℝ) (h_R : R > 0) :
  ∀ (t : SymmetricTrapezoid R),
    y R t = (t.x - R)^2 + 3*R^2 ∧
    ∀ (a : ℝ), y R t = a^2 → 3*R^2 ≤ a^2 ∧ a^2 ≤ 4*R^2 := by
  sorry

#check symmetric_trapezoid_theorem

end NUMINAMATH_CALUDE_symmetric_trapezoid_theorem_l324_32425


namespace NUMINAMATH_CALUDE_mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l324_32478

theorem mrs_hilt_apple_pies : ℝ → Prop :=
  fun apple_pies =>
    let pecan_pies : ℝ := 16.0
    let total_pies : ℝ := pecan_pies + apple_pies
    let new_total : ℝ := 150.0
    (5.0 * total_pies = new_total) → apple_pies = 14.0

-- The proof is omitted
theorem mrs_hilt_apple_pies_proof : mrs_hilt_apple_pies 14.0 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apple_pies_mrs_hilt_apple_pies_proof_l324_32478


namespace NUMINAMATH_CALUDE_roberts_extra_chocolates_l324_32461

/-- Given that Robert ate 12 chocolates and Nickel ate 3 chocolates,
    prove that Robert ate 9 more chocolates than Nickel. -/
theorem roberts_extra_chocolates (robert : Nat) (nickel : Nat)
    (h1 : robert = 12) (h2 : nickel = 3) :
    robert - nickel = 9 := by
  sorry

end NUMINAMATH_CALUDE_roberts_extra_chocolates_l324_32461


namespace NUMINAMATH_CALUDE_bisecting_plane_intersects_sixteen_cubes_l324_32421

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side_length : ℕ

/-- Represents a plane that bisects a face diagonal of a cube -/
structure BisectingPlane where
  cube : UnitCube

/-- Counts the number of unit cubes intersected by a bisecting plane -/
def count_intersected_cubes (plane : BisectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane bisecting a face diagonal of a 4x4x4 cube intersects 16 unit cubes -/
theorem bisecting_plane_intersects_sixteen_cubes 
  (cube : UnitCube) 
  (plane : BisectingPlane) 
  (h1 : cube.side_length = 4) 
  (h2 : plane.cube = cube) : 
  count_intersected_cubes plane = 16 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_plane_intersects_sixteen_cubes_l324_32421


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l324_32450

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 3 * w) : 
  (l * w) / (Real.pi * r^2) = 3 * Real.pi / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l324_32450


namespace NUMINAMATH_CALUDE_interest_rate_cut_l324_32452

theorem interest_rate_cut (x : ℝ) : 
  (2.25 / 100 : ℝ) * (1 - x)^2 = (1.98 / 100 : ℝ) → 
  (∃ (initial_rate final_rate : ℝ), 
    initial_rate = 2.25 / 100 ∧ 
    final_rate = 1.98 / 100 ∧ 
    final_rate = initial_rate * (1 - x)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_cut_l324_32452


namespace NUMINAMATH_CALUDE_intersection_chord_length_l324_32414

-- Define the circles in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = 2

def circle_O₂ (ρ θ : ℝ) : Prop := ρ^2 - 2*Real.sqrt 2*ρ*(Real.cos (θ - Real.pi/4)) = 2

-- Define the circles in rectangular coordinates
def rect_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

def rect_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (rect_O₁ A.1 A.2 ∧ rect_O₁ B.1 B.2) →
  (rect_O₂ A.1 A.2 ∧ rect_O₂ B.1 B.2) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l324_32414


namespace NUMINAMATH_CALUDE_n_pow_half_n_eq_eight_l324_32454

theorem n_pow_half_n_eq_eight (n : ℝ) : n = 2^Real.sqrt 6 → n^(n/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_n_pow_half_n_eq_eight_l324_32454


namespace NUMINAMATH_CALUDE_odd_function_property_l324_32417

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def has_max_on (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

def has_min_on (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x → x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  is_odd f →
  increasing_on f 3 6 →
  has_max_on f 3 6 2 →
  has_min_on f 3 6 (-1) →
  2 * f (-6) + f (-3) = -3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_property_l324_32417


namespace NUMINAMATH_CALUDE_circle_equation_specific_l324_32495

/-- The equation of a circle with center (h, k) and radius r -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The equation of a circle with center (2, -3) and radius 4 -/
theorem circle_equation_specific : 
  CircleEquation 2 (-3) 4 x y ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_specific_l324_32495


namespace NUMINAMATH_CALUDE_green_balls_count_l324_32413

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob : ℚ) :
  total = 100 ∧ 
  white = 50 ∧ 
  yellow = 8 ∧ 
  red = 9 ∧ 
  purple = 3 ∧ 
  prob = 88/100 ∧
  prob = (white + yellow + (total - white - yellow - red - purple : ℕ)) / total →
  total - white - yellow - red - purple = 30 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l324_32413


namespace NUMINAMATH_CALUDE_sum_product_inequality_l324_32476

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l324_32476


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l324_32426

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 3*a-1, a^2+1}
  A ∩ B = {-3} → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l324_32426


namespace NUMINAMATH_CALUDE_no_right_triangle_with_perimeter_5_times_inradius_l324_32473

theorem no_right_triangle_with_perimeter_5_times_inradius :
  ¬∃ (a b c : ℕ+), 
    (a.val^2 + b.val^2 = c.val^2) ∧  -- right triangle condition
    ((a.val + b.val + c.val : ℚ) = 5 * (a.val * b.val : ℚ) / (a.val + b.val + c.val : ℚ)) 
    -- perimeter = 5 * in-radius condition
  := by sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_perimeter_5_times_inradius_l324_32473


namespace NUMINAMATH_CALUDE_f_negative_ten_equals_three_l324_32412

/-- Given a function f and a real number a, prove that f(-10) = 3 -/
theorem f_negative_ten_equals_three (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = |x| * (Real.exp (a * x) - Real.exp (-a * x)) + 2)
  (h2 : f 10 = 1) : 
  f (-10) = 3 := by sorry

end NUMINAMATH_CALUDE_f_negative_ten_equals_three_l324_32412


namespace NUMINAMATH_CALUDE_evaluate_expression_l324_32494

theorem evaluate_expression (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π + 2)) = Real.sin 2 ^ 2 - Real.cos 2 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l324_32494


namespace NUMINAMATH_CALUDE_angle_parallel_lines_l324_32419

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between two lines
variable (angle_between : Line → Line → Angle)

-- Define equality for angles
variable (angle_eq : Angle → Angle → Prop)

-- Theorem statement
theorem angle_parallel_lines 
  (a b c : Line) (θ : Angle)
  (h1 : parallel a b)
  (h2 : angle_eq (angle_between a c) θ) :
  angle_eq (angle_between b c) θ :=
sorry

end NUMINAMATH_CALUDE_angle_parallel_lines_l324_32419


namespace NUMINAMATH_CALUDE_nail_decoration_time_l324_32437

theorem nail_decoration_time (total_time : ℕ) (num_coats : ℕ) (time_per_coat : ℕ) : 
  total_time = 120 →
  num_coats = 3 →
  total_time = num_coats * 2 * time_per_coat →
  time_per_coat = 20 := by
sorry

end NUMINAMATH_CALUDE_nail_decoration_time_l324_32437


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l324_32484

theorem simplify_trig_expression (α : ℝ) :
  3 - 4 * Real.cos (4 * α) + Real.cos (8 * α) - 8 * (Real.cos (2 * α))^4 = -8 * Real.cos (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l324_32484


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l324_32422

/-- Given a geometric sequence {a_n} with a_1 = 1, prove that a_2 = 4 is sufficient but not necessary for a_3 = 16 -/
theorem geometric_sequence_condition (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 1 →                            -- First term is 1
  (a 2 = 4 → a 3 = 16) ∧               -- Sufficient condition
  ¬(a 3 = 16 → a 2 = 4)                -- Not necessary condition
  := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l324_32422


namespace NUMINAMATH_CALUDE_square_completion_l324_32462

theorem square_completion (x : ℝ) : x^2 + 6*x - 5 = 0 ↔ (x + 3)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l324_32462


namespace NUMINAMATH_CALUDE_complex_arithmetic_problem_l324_32488

theorem complex_arithmetic_problem : 
  (Complex.mk 2 5 + Complex.mk (-1) (-3)) * Complex.mk 3 1 = Complex.mk 1 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_problem_l324_32488


namespace NUMINAMATH_CALUDE_rita_swimming_months_l324_32467

def swimming_months (total_required : ℕ) (completed : ℕ) (monthly_practice : ℕ) : ℕ :=
  (total_required - completed + monthly_practice - 1) / monthly_practice

theorem rita_swimming_months :
  swimming_months 2500 300 300 = 8 := by sorry

end NUMINAMATH_CALUDE_rita_swimming_months_l324_32467


namespace NUMINAMATH_CALUDE_monkey_climb_distance_l324_32404

/-- Represents the climbing behavior of a monkey -/
structure MonkeyClimb where
  climb_distance : ℝ  -- Distance the monkey climbs in one minute
  slip_distance : ℝ   -- Distance the monkey slips in the next minute
  total_time : ℕ      -- Total time taken to reach the top
  total_height : ℝ    -- Total height reached

/-- Theorem stating that given the monkey's climbing behavior, 
    if it takes 37 minutes to reach 60 meters, then it climbs 6 meters per minute -/
theorem monkey_climb_distance 
  (m : MonkeyClimb) 
  (h1 : m.slip_distance = 3) 
  (h2 : m.total_time = 37) 
  (h3 : m.total_height = 60) : 
  m.climb_distance = 6 := by
  sorry

#check monkey_climb_distance

end NUMINAMATH_CALUDE_monkey_climb_distance_l324_32404


namespace NUMINAMATH_CALUDE_kangaroo_distance_l324_32445

/-- The distance traveled after a given number of hops, where each hop is 1/4 of the remaining distance -/
def distance_after_hops (target : ℚ) (num_hops : ℕ) : ℚ :=
  target * (1 - (3/4)^num_hops)

/-- The theorem stating that after 4 hops, the distance traveled is 175/128 of the target distance -/
theorem kangaroo_distance : distance_after_hops 2 4 = 175/128 := by
  sorry

#eval distance_after_hops 2 4

end NUMINAMATH_CALUDE_kangaroo_distance_l324_32445


namespace NUMINAMATH_CALUDE_madeline_homework_hours_l324_32434

/-- Calculates the number of hours Madeline spends on homework per day -/
theorem madeline_homework_hours (class_hours_per_week : ℕ) 
                                 (sleep_hours_per_day : ℕ) 
                                 (work_hours_per_week : ℕ) 
                                 (leftover_hours : ℕ) 
                                 (days_per_week : ℕ) 
                                 (hours_per_day : ℕ) :
  class_hours_per_week = 18 →
  sleep_hours_per_day = 8 →
  work_hours_per_week = 20 →
  leftover_hours = 46 →
  days_per_week = 7 →
  hours_per_day = 24 →
  (hours_per_day * days_per_week - 
   (class_hours_per_week + sleep_hours_per_day * days_per_week + 
    work_hours_per_week + leftover_hours)) / days_per_week = 4 := by
  sorry

end NUMINAMATH_CALUDE_madeline_homework_hours_l324_32434


namespace NUMINAMATH_CALUDE_meetings_percentage_of_workday_l324_32405

def workday_hours : ℝ := 10
def first_meeting_minutes : ℝ := 30
def second_meeting_minutes : ℝ := 3 * first_meeting_minutes
def third_meeting_minutes : ℝ := 2 * second_meeting_minutes

def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes
def workday_minutes : ℝ := workday_hours * 60

theorem meetings_percentage_of_workday :
  (total_meeting_minutes / workday_minutes) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_workday_l324_32405


namespace NUMINAMATH_CALUDE_production_scaling_l324_32441

/-- Given that x men working x hours a day for x days produce x^2 articles,
    prove that z men working z hours a day for z days produce z^3/x articles. -/
theorem production_scaling (x z : ℝ) (hx : x > 0) :
  (x * x * x * x^2 = x^3 * x^2) →
  (z * z * z * (z^3 / x) = z^3 * (z^3 / x)) :=
by sorry

end NUMINAMATH_CALUDE_production_scaling_l324_32441


namespace NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_acute_l324_32416

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define the statement we want to prove false
def sum_of_acute_angles_always_acute : Prop :=
  ∀ (a b : ℝ), is_acute_angle a → is_acute_angle b → is_acute_angle (a + b)

-- Theorem stating that the above statement is false
theorem sum_of_acute_angles_not_always_acute :
  ¬ sum_of_acute_angles_always_acute :=
sorry

end NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_acute_l324_32416


namespace NUMINAMATH_CALUDE_solution_to_system_l324_32475

theorem solution_to_system (x y : ℝ) : 
  (1 / (x^2 + y^2) + x^2 * y^2 = 5/4) ∧ 
  (2 * x^4 + 2 * y^4 + 5 * x^2 * y^2 = 9/4) ↔ 
  ((x = 1/Real.sqrt 2 ∧ (y = 1/Real.sqrt 2 ∨ y = -1/Real.sqrt 2)) ∨
   (x = -1/Real.sqrt 2 ∧ (y = 1/Real.sqrt 2 ∨ y = -1/Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_solution_to_system_l324_32475


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l324_32409

/-- Given three square regions A, B, and C with perimeters 16, 32, and 20 units respectively,
    prove that the ratio of the area of region B to the area of region C is 64/25. -/
theorem area_ratio_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (perim_a : 4 * a = 16) (perim_b : 4 * b = 32) (perim_c : 4 * c = 20) :
  (b * b) / (c * c) = 64 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l324_32409


namespace NUMINAMATH_CALUDE_no_triangle_solution_l324_32423

theorem no_triangle_solution (A B C : Real) (a b c : Real) : 
  A = Real.pi / 3 →  -- 60 degrees in radians
  b = 4 → 
  a = 2 → 
  ¬ (∃ (B C : Real), 
      0 < B ∧ 0 < C ∧ 
      A + B + C = Real.pi ∧ 
      a / Real.sin A = b / Real.sin B ∧ 
      b / Real.sin B = c / Real.sin C) :=
by
  sorry


end NUMINAMATH_CALUDE_no_triangle_solution_l324_32423


namespace NUMINAMATH_CALUDE_negation_equivalence_l324_32451

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (CarefulInvestor : U → Prop)
variable (RecklessInvestor : U → Prop)

-- Define the statements
def AllChildrenAreCareful : Prop := ∀ x, Child x → CarefulInvestor x
def AtLeastOneChildIsReckless : Prop := ∃ x, Child x ∧ RecklessInvestor x

-- The theorem to prove
theorem negation_equivalence : 
  AtLeastOneChildIsReckless U Child RecklessInvestor ↔ 
  ¬(AllChildrenAreCareful U Child CarefulInvestor) :=
sorry

-- Additional assumption: being reckless is the opposite of being careful
axiom reckless_careful_opposite : 
  ∀ x, RecklessInvestor x ↔ ¬(CarefulInvestor x)

end NUMINAMATH_CALUDE_negation_equivalence_l324_32451


namespace NUMINAMATH_CALUDE_smallest_value_expression_l324_32471

theorem smallest_value_expression (x : ℝ) (h : x = -3) :
  let a := x^2 - 3
  let b := (x - 3)^2
  let c := x^2
  let d := (x + 3)^2
  let e := x^2 + 3
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_expression_l324_32471


namespace NUMINAMATH_CALUDE_geometry_theorem_l324_32468

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the perpendicular relation between a line and a plane
def perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relation between lines
def parallelLines (l1 l2 : Line) : Prop := sorry

-- Define skew relation between lines
def skewLines (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between planes
def parallelPlanes (p1 p2 : Plane) : Prop := sorry

-- Define intersection relation between planes
def planesIntersect (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between lines
def perpendicularLines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (h1 : perpendicularToPlane m α) 
  (h2 : perpendicularToPlane n β) :
  (parallelLines m n → parallelPlanes α β) ∧ 
  (skewLines m n → planesIntersect α β) ∧
  (perpendicularPlanes α β → perpendicularLines m n) := by
  sorry

end NUMINAMATH_CALUDE_geometry_theorem_l324_32468


namespace NUMINAMATH_CALUDE_not_exist_prime_power_of_six_plus_nineteen_l324_32406

theorem not_exist_prime_power_of_six_plus_nineteen :
  ∀ n : ℕ, ¬ Nat.Prime (6^n + 19) := by
  sorry

end NUMINAMATH_CALUDE_not_exist_prime_power_of_six_plus_nineteen_l324_32406


namespace NUMINAMATH_CALUDE_only_solution_is_one_l324_32498

theorem only_solution_is_one : 
  ∀ n : ℕ, (2 * n - 1 : ℚ) / (n^5 : ℚ) = 3 - 2 / (n : ℚ) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_one_l324_32498


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l324_32400

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^(2*x) * (1000 : ℝ)^x = (10 : ℝ)^15 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l324_32400


namespace NUMINAMATH_CALUDE_optimal_arrangement_l324_32465

-- Define the harvester types
inductive HarvesterType
| A
| B

-- Define the properties of harvesters
def harvest_rate (t : HarvesterType) : ℕ :=
  match t with
  | HarvesterType.A => 5
  | HarvesterType.B => 3

def fee_per_hectare (t : HarvesterType) : ℕ :=
  match t with
  | HarvesterType.A => 50
  | HarvesterType.B => 45

-- Define the problem constraints
def total_harvesters : ℕ := 12
def min_hectares_per_day : ℕ := 50

-- Define the optimization problem
def is_valid_arrangement (num_A : ℕ) : Prop :=
  num_A ≤ total_harvesters ∧
  num_A * harvest_rate HarvesterType.A + (total_harvesters - num_A) * harvest_rate HarvesterType.B ≥ min_hectares_per_day

def total_cost (num_A : ℕ) : ℕ :=
  num_A * harvest_rate HarvesterType.A * fee_per_hectare HarvesterType.A +
  (total_harvesters - num_A) * harvest_rate HarvesterType.B * fee_per_hectare HarvesterType.B

-- State the theorem
theorem optimal_arrangement :
  ∃ (num_A : ℕ), is_valid_arrangement num_A ∧
  (∀ (m : ℕ), is_valid_arrangement m → total_cost num_A ≤ total_cost m) ∧
  num_A = 7 ∧
  total_cost num_A = 2425 := by sorry

end NUMINAMATH_CALUDE_optimal_arrangement_l324_32465


namespace NUMINAMATH_CALUDE_rectangle_division_l324_32438

theorem rectangle_division (a b : ℝ) (h1 : a + b = 50) (h2 : 7 * b + 10 * a = 434) :
  2 * (a / 8 + b / 11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l324_32438


namespace NUMINAMATH_CALUDE_total_discount_calculation_l324_32496

theorem total_discount_calculation (tshirt_price jeans_price : ℝ)
  (tshirt_discount jeans_discount : ℝ) :
  tshirt_price = 25 →
  jeans_price = 75 →
  tshirt_discount = 0.3 →
  jeans_discount = 0.1 →
  tshirt_price * tshirt_discount + jeans_price * jeans_discount = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l324_32496


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l324_32453

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) : 
  ∃ (x : ℝ), -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 ∧ 
  ∀ (y : ℝ), -9/2 < y ∧ y < 13/2 → ∃ (a' b' : ℝ), 
    -1 < a' + b' ∧ a' + b' < 3 ∧ 
    2 < a' - b' ∧ a' - b' < 4 ∧ 
    2*a' + 3*b' = y :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l324_32453


namespace NUMINAMATH_CALUDE_shopping_cost_l324_32470

/-- The cost of items in a shopping mall with discount --/
theorem shopping_cost (tshirt_cost pants_cost shoe_cost : ℝ) 
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : (4 * tshirt_cost + 3 * pants_cost + 2 * shoe_cost) * 0.9 = 558) :
  shoe_cost = 150 := by
sorry

end NUMINAMATH_CALUDE_shopping_cost_l324_32470


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l324_32455

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4,7) and (-8,9) is -16. -/
theorem midpoint_coordinate_product : 
  let a : ℝ × ℝ := (4, 7)
  let b : ℝ × ℝ := (-8, 9)
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  (midpoint.1 * midpoint.2 : ℝ) = -16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l324_32455


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l324_32429

theorem negative_fraction_comparison : -1/3 > -1/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l324_32429


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l324_32457

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 850 →
  crossing_time = 67.66125376636536 →
  ∃ (speed : ℝ), (abs (speed - 54.018) < 0.001 ∧ 
    speed * crossing_time / 3.6 = train_length + bridge_length) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l324_32457


namespace NUMINAMATH_CALUDE_craig_total_distance_l324_32497

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := 0.2

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_home : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := distance_school_to_david + distance_david_to_home

/-- Theorem stating that the total distance Craig walked is 0.9 miles -/
theorem craig_total_distance : total_distance = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_craig_total_distance_l324_32497


namespace NUMINAMATH_CALUDE_problem_solution_l324_32474

theorem problem_solution (a b : ℤ) 
  (h1 : 4010 * a + 4014 * b = 4020) 
  (h2 : 4012 * a + 4016 * b = 4024) : 
  a - b = 2002 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l324_32474


namespace NUMINAMATH_CALUDE_b_investment_l324_32446

/-- Represents the investment and profit share of a person in the business. -/
structure Participant where
  investment : ℝ
  profitShare : ℝ

/-- Proves that given the conditions of the problem, b's investment is 10000. -/
theorem b_investment (a b c : Participant)
  (h1 : b.profitShare = 3500)
  (h2 : c.profitShare - a.profitShare = 1399.9999999999998)
  (h3 : a.investment = 8000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare / a.investment = b.profitShare / b.investment)
  (h6 : c.profitShare / c.investment = b.profitShare / b.investment) :
  b.investment = 10000 := by
  sorry


end NUMINAMATH_CALUDE_b_investment_l324_32446


namespace NUMINAMATH_CALUDE_enough_paint_l324_32401

/-- Represents the dimensions of the gym --/
structure GymDimensions where
  length : ℝ
  width : ℝ

/-- Represents the paint requirements and availability --/
structure PaintInfo where
  cans : ℕ
  weight_per_can : ℝ
  paint_per_sqm : ℝ

/-- Theorem stating that there is enough paint for the gym floor --/
theorem enough_paint (gym : GymDimensions) (paint : PaintInfo) : 
  gym.length = 65 ∧ 
  gym.width = 32 ∧ 
  paint.cans = 23 ∧ 
  paint.weight_per_can = 25 ∧ 
  paint.paint_per_sqm = 0.25 → 
  (paint.cans : ℝ) * paint.weight_per_can > gym.length * gym.width * paint.paint_per_sqm := by
  sorry

#check enough_paint

end NUMINAMATH_CALUDE_enough_paint_l324_32401


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l324_32499

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) ↔ (m = 8 * Real.sqrt 3 ∨ m = -8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l324_32499


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l324_32464

theorem polynomial_value_theorem (m n : ℝ) 
  (h1 : 2*m + n + 2 = m + 2*n) 
  (h2 : m - n + 2 ≠ 0) : 
  let x := 3*(m + n + 1)
  (x^2 + 4*x + 6 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l324_32464


namespace NUMINAMATH_CALUDE_fraction_value_theorem_l324_32456

theorem fraction_value_theorem (x : ℝ) :
  2 / (x - 3) = 2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_theorem_l324_32456


namespace NUMINAMATH_CALUDE_combined_completion_time_l324_32486

/-- Given the time taken by X, Y, and Z to complete a task individually,
    calculate the time taken when they work together. -/
theorem combined_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 15)
  (hy : y_time = 30)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 20 / 3 := by
  sorry

#check combined_completion_time

end NUMINAMATH_CALUDE_combined_completion_time_l324_32486


namespace NUMINAMATH_CALUDE_logan_max_rent_l324_32415

def current_income : ℕ := 65000
def grocery_expenses : ℕ := 5000
def gas_expenses : ℕ := 8000
def desired_savings : ℕ := 42000
def income_increase : ℕ := 10000

def max_rent : ℕ := 20000

theorem logan_max_rent :
  max_rent = current_income + income_increase - desired_savings - grocery_expenses - gas_expenses :=
by sorry

end NUMINAMATH_CALUDE_logan_max_rent_l324_32415


namespace NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l324_32403

theorem unique_real_sqrt_negative_square : 
  ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(x + 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l324_32403


namespace NUMINAMATH_CALUDE_circle_area_tripled_l324_32443

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l324_32443


namespace NUMINAMATH_CALUDE_expression_evaluation_l324_32487

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l324_32487


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l324_32458

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l324_32458


namespace NUMINAMATH_CALUDE_card_sum_theorem_l324_32433

theorem card_sum_theorem (n : ℕ) (m : ℕ) (h1 : n ≥ 3) (h2 : m = n * (n - 1) / 2) (h3 : Odd m) :
  ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_card_sum_theorem_l324_32433


namespace NUMINAMATH_CALUDE_find_third_number_l324_32463

def third_number (a b n : ℕ) : Prop :=
  (Nat.gcd a (Nat.gcd b n) = 8) ∧
  (Nat.lcm a (Nat.lcm b n) = 2^4 * 3^2 * 17 * 7)

theorem find_third_number :
  third_number 136 144 7 :=
by sorry

end NUMINAMATH_CALUDE_find_third_number_l324_32463


namespace NUMINAMATH_CALUDE_parallelogram_count_l324_32418

-- Define the parallelogram structure
structure Parallelogram where
  b : ℕ
  d : ℕ
  area_eq : b * d = 1728000
  b_positive : b > 0
  d_positive : d > 0

-- Define the count function
def count_parallelograms : ℕ := sorry

-- Theorem statement
theorem parallelogram_count : count_parallelograms = 56 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l324_32418


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_equals_circumference_l324_32428

/-- Represents a circle with a given circumference -/
structure Circle where
  circumference : ℝ

/-- Represents a configuration of four identical circles arranged in a straight line -/
structure CircleConfiguration where
  circle : Circle
  num_circles : Nat
  are_tangent : Bool
  are_identical : Bool
  are_in_line : Bool

/-- Calculates the perimeter of the shaded region between the first and last circle -/
def shaded_region_perimeter (config : CircleConfiguration) : ℝ :=
  config.circle.circumference

/-- Theorem stating that the perimeter of the shaded region is equal to the circumference of one circle -/
theorem shaded_region_perimeter_equals_circumference 
  (config : CircleConfiguration) 
  (h1 : config.num_circles = 4) 
  (h2 : config.are_tangent) 
  (h3 : config.are_identical) 
  (h4 : config.are_in_line) 
  (h5 : config.circle.circumference = 24) :
  shaded_region_perimeter config = 24 := by
  sorry

#check shaded_region_perimeter_equals_circumference

end NUMINAMATH_CALUDE_shaded_region_perimeter_equals_circumference_l324_32428


namespace NUMINAMATH_CALUDE_add_negative_two_l324_32481

theorem add_negative_two : 3 + (-2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_two_l324_32481


namespace NUMINAMATH_CALUDE_complex_equation_solution_l324_32440

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l324_32440


namespace NUMINAMATH_CALUDE_eight_balls_three_boxes_l324_32492

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 45 ways to distribute 8 indistinguishable balls into 3 distinguishable boxes -/
theorem eight_balls_three_boxes : distribute_balls 8 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_eight_balls_three_boxes_l324_32492
