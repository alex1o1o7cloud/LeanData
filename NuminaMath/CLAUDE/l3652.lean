import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_f_l3652_365206

def f (x : ℝ) : ℝ := x^3 - 3*x + 9

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3652_365206


namespace NUMINAMATH_CALUDE_ratio_of_15th_terms_l3652_365257

/-- Two arithmetic sequences with sums P_n and Q_n for the first n terms -/
def arithmetic_sequences (P Q : ℕ → ℚ) : Prop :=
  ∃ (a d b e : ℚ), ∀ n : ℕ,
    P n = n / 2 * (2 * a + (n - 1) * d) ∧
    Q n = n / 2 * (2 * b + (n - 1) * e)

/-- The ratio of P_n to Q_n is (5n+3)/(3n+11) for all n -/
def ratio_condition (P Q : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, P n / Q n = (5 * n + 3) / (3 * n + 11)

/-- The 15th term of an arithmetic sequence -/
def term_15 (a d : ℚ) : ℚ := a + 14 * d

theorem ratio_of_15th_terms (P Q : ℕ → ℚ) 
  (h1 : arithmetic_sequences P Q) 
  (h2 : ratio_condition P Q) : 
  ∃ (a d b e : ℚ), 
    term_15 a d / term_15 b e = 71 / 52 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_15th_terms_l3652_365257


namespace NUMINAMATH_CALUDE_pears_picked_by_keith_l3652_365255

/-- The number of pears Keith picked -/
def keiths_pears : ℝ := 0

theorem pears_picked_by_keith :
  let mikes_apples : ℝ := 7.0
  let nancys_eaten_apples : ℝ := 3.0
  let keiths_apples : ℝ := 6.0
  let apples_left : ℝ := 10.0
  keiths_pears = 0 := by sorry

end NUMINAMATH_CALUDE_pears_picked_by_keith_l3652_365255


namespace NUMINAMATH_CALUDE_envelope_is_hyperbola_l3652_365228

/-- A family of straight lines forming right-angled triangles with area a^2 / 2 -/
def LineFamily (a : ℝ) := {l : Set (ℝ × ℝ) | ∃ α : ℝ, α > 0 ∧ l = {(x, y) | x + α^2 * y = α * a}}

/-- The envelope of the family of lines -/
def Envelope (a : ℝ) := {(x, y) : ℝ × ℝ | x * y = a^2 / 4}

/-- Theorem stating that the envelope of the line family is the hyperbola xy = a^2 / 4 -/
theorem envelope_is_hyperbola (a : ℝ) (h : a > 0) :
  Envelope a = {p : ℝ × ℝ | ∃ l ∈ LineFamily a, p ∈ l ∧ 
    ∀ l' ∈ LineFamily a, l ≠ l' → (∃ q ∈ l ∩ l', ∀ r ∈ l ∩ l', dist p q ≤ dist p r)} :=
sorry

end NUMINAMATH_CALUDE_envelope_is_hyperbola_l3652_365228


namespace NUMINAMATH_CALUDE_maria_fish_removal_l3652_365293

/-- The number of fish Maria took out of her tank -/
def fish_taken_out (initial_fish current_fish : ℕ) : ℕ :=
  initial_fish - current_fish

/-- Theorem: Maria took out 16 fish from her tank -/
theorem maria_fish_removal :
  fish_taken_out 19 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_maria_fish_removal_l3652_365293


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3652_365259

theorem muffin_banana_price_ratio : 
  ∀ (muffin_price banana_price : ℝ),
  (5 * muffin_price + 4 * banana_price > 0) →
  (3 * (5 * muffin_price + 4 * banana_price) = 3 * muffin_price + 20 * banana_price) →
  (muffin_price / banana_price = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3652_365259


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l3652_365296

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 160

/-- Theorem stating the equivalent cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l3652_365296


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3652_365219

-- Define the function f(x) = x^3 - 3x + 2
def f (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Define the closed interval [0, 3]
def interval : Set ℝ := Set.Icc 0 3

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 20 ∧ min = 0 ∧
  (∀ x ∈ interval, f x ≤ max) ∧
  (∃ x ∈ interval, f x = max) ∧
  (∀ x ∈ interval, min ≤ f x) ∧
  (∃ x ∈ interval, f x = min) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3652_365219


namespace NUMINAMATH_CALUDE_smallest_terminating_with_two_l3652_365276

/-- A function that checks if a positive integer contains the digit 2 -/
def containsDigitTwo (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := sorry

/-- Theorem stating that 2 is the smallest positive integer n such that 1/n is a terminating decimal and n contains the digit 2 -/
theorem smallest_terminating_with_two :
  (∀ m : ℕ+, m < 2 → ¬(isTerminatingDecimal m ∧ containsDigitTwo m)) ∧
  (isTerminatingDecimal 2 ∧ containsDigitTwo 2) :=
sorry

end NUMINAMATH_CALUDE_smallest_terminating_with_two_l3652_365276


namespace NUMINAMATH_CALUDE_carpet_width_l3652_365262

/-- Proves that a rectangular carpet covering 30% of a 120 square feet floor with a length of 9 feet has a width of 4 feet. -/
theorem carpet_width (floor_area : ℝ) (carpet_coverage : ℝ) (carpet_length : ℝ) :
  floor_area = 120 →
  carpet_coverage = 0.3 →
  carpet_length = 9 →
  (floor_area * carpet_coverage) / carpet_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l3652_365262


namespace NUMINAMATH_CALUDE_quadratic_sum_l3652_365234

/-- Given a quadratic function f(x) = 8x^2 - 48x - 288, when expressed in the form a(x+b)^2 + c,
    the sum of a, b, and c is -355. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8 * x^2 - 48 * x - 288) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = -355 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3652_365234


namespace NUMINAMATH_CALUDE_paul_filled_six_bags_saturday_l3652_365221

/-- The number of bags Paul filled on Saturday -/
def bags_saturday : ℕ := sorry

/-- The number of bags Paul filled on Sunday -/
def bags_sunday : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 8

/-- The total number of cans collected -/
def total_cans : ℕ := 72

/-- Theorem stating that Paul filled 6 bags on Saturday -/
theorem paul_filled_six_bags_saturday : 
  bags_saturday = 6 := by sorry

end NUMINAMATH_CALUDE_paul_filled_six_bags_saturday_l3652_365221


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l3652_365278

theorem positive_numbers_inequalities (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≥ 1 ∧
  a^2/(b+c) + b^2/(a+c) + c^2/(a+b) ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l3652_365278


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3652_365237

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3) 
  (h2 : Real.pi/2 ≤ α) 
  (h3 : α ≤ Real.pi) : 
  Real.cos α = -2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3652_365237


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3652_365215

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3652_365215


namespace NUMINAMATH_CALUDE_problem_statement_l3652_365282

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 1

theorem problem_statement :
  (∀ x : ℝ, f x > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ∧
  (¬(∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0)) ∧
  (¬(∃ x₀ : ℝ, x₀ > 0 ∧ g x₀ = 0) ↔ (∀ x : ℝ, x > 0 → g x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3652_365282


namespace NUMINAMATH_CALUDE_toy_robot_shipment_l3652_365235

theorem toy_robot_shipment (displayed_percentage : ℚ) (stored : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored = 140 →
  (1 - displayed_percentage) * 200 = stored :=
by sorry

end NUMINAMATH_CALUDE_toy_robot_shipment_l3652_365235


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_l3652_365240

-- Define the natural logarithm function
noncomputable def ln (x : ℝ) : ℝ := Real.log x

-- Theorem statement
theorem x_negative_necessary_not_sufficient :
  (∀ x : ℝ, ln (x + 1) < 0 → x < 0) ∧
  ¬(∀ x : ℝ, x < 0 → ln (x + 1) < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_l3652_365240


namespace NUMINAMATH_CALUDE_matrix_commutation_fraction_l3652_365232

/-- Given two matrices A and B, where A is fixed and B has variable entries,
    prove that if A * B = B * A and 3b ≠ c, then (a - d) / (c - 3b) = 1. -/
theorem matrix_commutation_fraction (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (3 * b ≠ c) → ((a - d) / (c - 3 * b) = 1) := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutation_fraction_l3652_365232


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3652_365216

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width, fp.painting_height + 4 * fp.side_frame_width)

/-- Calculates the area of the frame -/
def frameArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h - fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of dimensions for a specific framed painting -/
theorem framed_painting_ratio :
  ∃ (fp : FramedPainting),
    fp.painting_width = 15 ∧
    fp.painting_height = 30 ∧
    frameArea fp = fp.painting_width * fp.painting_height ∧
    let (w, h) := framedDimensions fp
    w / h = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3652_365216


namespace NUMINAMATH_CALUDE_opposite_pairs_l3652_365218

-- Define the pairs of numbers
def pair_A : ℚ × ℚ := (-5, 1/5)
def pair_B : ℤ × ℤ := (8, 8)
def pair_C : ℤ × ℤ := (-3, 3)
def pair_D : ℚ × ℚ := (7/2, 7/2)

-- Define what it means for two numbers to be opposite
def are_opposite (a b : ℚ) : Prop := a = -b

-- Theorem stating that pair C contains opposite numbers, while others do not
theorem opposite_pairs :
  (¬ are_opposite pair_A.1 pair_A.2) ∧
  (¬ are_opposite pair_B.1 pair_B.2) ∧
  (are_opposite pair_C.1 pair_C.2) ∧
  (¬ are_opposite pair_D.1 pair_D.2) :=
sorry

end NUMINAMATH_CALUDE_opposite_pairs_l3652_365218


namespace NUMINAMATH_CALUDE_leftover_coin_value_l3652_365280

def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def half_dollars_per_roll : ℕ := 20

def james_quarters : ℕ := 120
def james_dimes : ℕ := 200
def james_half_dollars : ℕ := 90

def lindsay_quarters : ℕ := 150
def lindsay_dimes : ℕ := 310
def lindsay_half_dollars : ℕ := 160

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes
def total_half_dollars : ℕ := james_half_dollars + lindsay_half_dollars

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll
def leftover_half_dollars : ℕ := total_half_dollars % half_dollars_per_roll

def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.1
def half_dollar_value : ℚ := 0.5

theorem leftover_coin_value :
  (leftover_quarters : ℚ) * quarter_value +
  (leftover_dimes : ℚ) * dime_value +
  (leftover_half_dollars : ℚ) * half_dollar_value = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_leftover_coin_value_l3652_365280


namespace NUMINAMATH_CALUDE_birdhouse_volume_difference_l3652_365283

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem: The difference in volume between Sara's and Jake's birdhouses is 1152 cubic inches -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_volume_difference_l3652_365283


namespace NUMINAMATH_CALUDE_circle_tangency_l3652_365210

theorem circle_tangency (a : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 1 ∧ (p.1 + 4)^2 + (p.2 - a)^2 = 25) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_l3652_365210


namespace NUMINAMATH_CALUDE_half_dollar_percentage_l3652_365209

def nickel_value : ℚ := 5
def half_dollar_value : ℚ := 50
def num_nickels : ℕ := 75
def num_half_dollars : ℕ := 30

theorem half_dollar_percentage :
  (num_half_dollars * half_dollar_value) / 
  (num_nickels * nickel_value + num_half_dollars * half_dollar_value) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_half_dollar_percentage_l3652_365209


namespace NUMINAMATH_CALUDE_inequality_multiplication_l3652_365211

theorem inequality_multiplication (x y : ℝ) : y > x → 2 * y > 2 * x := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l3652_365211


namespace NUMINAMATH_CALUDE_cameron_donation_ratio_l3652_365224

theorem cameron_donation_ratio :
  let boris_initial : ℕ := 24
  let boris_donation_fraction : ℚ := 1/4
  let cameron_initial : ℕ := 30
  let total_after_donation : ℕ := 38
  let boris_after := boris_initial - boris_initial * boris_donation_fraction
  let cameron_after := total_after_donation - boris_after
  let cameron_donated := cameron_initial - cameron_after
  cameron_donated / cameron_initial = 1/3 := by
sorry

end NUMINAMATH_CALUDE_cameron_donation_ratio_l3652_365224


namespace NUMINAMATH_CALUDE_gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l3652_365274

/-- Represents the cost function for a t-shirt company -/
structure TShirtCompany where
  setupFee : ℕ
  costPerShirt : ℕ

/-- Calculates the total cost for a given number of shirts -/
def totalCost (company : TShirtCompany) (shirts : ℕ) : ℕ :=
  company.setupFee + company.costPerShirt * shirts

/-- The Acme T-Shirt Company -/
def acme : TShirtCompany := ⟨40, 10⟩

/-- The Beta T-Shirt Company -/
def beta : TShirtCompany := ⟨0, 15⟩

/-- The Gamma T-Shirt Company -/
def gamma : TShirtCompany := ⟨20, 12⟩

theorem gamma_cheaper_at_11 :
  totalCost gamma 11 < totalCost acme 11 ∧
  totalCost gamma 11 < totalCost beta 11 :=
sorry

theorem gamma_not_cheaper_at_10 :
  ¬(totalCost gamma 10 < totalCost acme 10 ∧
    totalCost gamma 10 < totalCost beta 10) :=
sorry

theorem min_shirts_for_gamma_cheaper : ℕ :=
  11

end NUMINAMATH_CALUDE_gamma_cheaper_at_11_gamma_not_cheaper_at_10_min_shirts_for_gamma_cheaper_l3652_365274


namespace NUMINAMATH_CALUDE_sum_seven_times_difference_l3652_365281

theorem sum_seven_times_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x - y = 3 → x + y = 7 * (x - y) → x + y = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_times_difference_l3652_365281


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l3652_365298

theorem ones_digit_of_large_power : ∃ n : ℕ, n > 0 ∧ 37^(37*(28^28)) ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l3652_365298


namespace NUMINAMATH_CALUDE_geometric_mean_sqrt3_plus_minus_one_l3652_365246

theorem geometric_mean_sqrt3_plus_minus_one : 
  ∃ (x : ℝ), x^2 = (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_sqrt3_plus_minus_one_l3652_365246


namespace NUMINAMATH_CALUDE_cindy_envelopes_l3652_365203

theorem cindy_envelopes (initial : ℕ) (friend1 friend2 friend3 friend4 friend5 : ℕ) :
  initial = 137 →
  friend1 = 4 →
  friend2 = 7 →
  friend3 = 5 →
  friend4 = 10 →
  friend5 = 3 →
  initial - (friend1 + friend2 + friend3 + friend4 + friend5) = 108 :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l3652_365203


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3652_365231

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponentiation operation
def pow (base : ℕ) (exp : ℕ) : ℕ := base ^ exp

-- Theorem statement
theorem units_digit_sum_of_powers : 
  unitsDigit (pow 3 2014 + pow 4 2015 + pow 5 2016) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3652_365231


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l3652_365268

/-- Sum of reciprocals of non-zero digits from 1 to m -/
def sum_reciprocals (m : ℕ) : ℚ := sorry

/-- S_n as defined in the problem -/
def S (n : ℕ) : ℚ := sum_reciprocals (8^n)

/-- Predicate to check if a number is an integer -/
def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_S :
  (∀ k < 105, ¬ is_integer (S k)) ∧ is_integer (S 105) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l3652_365268


namespace NUMINAMATH_CALUDE_movie_screening_guests_l3652_365277

theorem movie_screening_guests :
  ∀ G : ℕ,
  G / 2 + 15 + (G - (G / 2 + 15)) = G →  -- Total guests = women + men + children
  G - (15 / 5 + 4) = 43 →                -- Guests who stayed
  G = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_screening_guests_l3652_365277


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l3652_365275

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (∃ b : ℝ, (a + Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l3652_365275


namespace NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l3652_365291

/-- A complex number z is on the negative y-axis if its real part is 0 and its imaginary part is negative -/
def on_negative_y_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

/-- The main theorem -/
theorem complex_square_on_negative_y_axis (a : ℝ) :
  on_negative_y_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l3652_365291


namespace NUMINAMATH_CALUDE_odd_function_expression_l3652_365242

noncomputable section

variable (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_expression 
  (h_odd : is_odd f)
  (h_neg : ∀ x < 0, f x = -x * Real.log (2 - x)) :
  ∀ x, f x = -x * Real.log (2 + |x|) := by sorry

end NUMINAMATH_CALUDE_odd_function_expression_l3652_365242


namespace NUMINAMATH_CALUDE_product_increase_theorem_l3652_365285

theorem product_increase_theorem :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ),
    (a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) * (a₆ - 3) * (a₇ - 3) =
    13 * (a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇) :=
by
  sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l3652_365285


namespace NUMINAMATH_CALUDE_switch_pairs_bound_l3652_365208

/-- Represents a row in Pascal's Triangle --/
def PascalRow (n : ℕ) := List ℕ

/-- Counts the number of odd entries in a Pascal's Triangle row --/
def countOddEntries (row : PascalRow n) : ℕ := sorry

/-- Counts the number of switch pairs in a Pascal's Triangle row --/
def countSwitchPairs (row : PascalRow n) : ℕ := sorry

/-- Theorem: The number of switch pairs in a Pascal's Triangle row is at most twice the number of odd entries --/
theorem switch_pairs_bound (n : ℕ) (row : PascalRow n) :
  countSwitchPairs row ≤ 2 * countOddEntries row := by sorry

end NUMINAMATH_CALUDE_switch_pairs_bound_l3652_365208


namespace NUMINAMATH_CALUDE_evaluate_expression_l3652_365225

theorem evaluate_expression : 9^6 * 3^4 / 27^5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3652_365225


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l3652_365213

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_five :
  opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l3652_365213


namespace NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3652_365226

theorem modulo_congruence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 27514 [MOD 16] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_unique_solution_l3652_365226


namespace NUMINAMATH_CALUDE_trailing_zeroes_of_sum_factorials_l3652_365284

/-- The number of trailing zeroes in a natural number -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of trailing zeroes in 70! + 140! is 16 -/
theorem trailing_zeroes_of_sum_factorials :
  trailingZeroes (factorial 70 + factorial 140) = 16 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_of_sum_factorials_l3652_365284


namespace NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_fifteen_l3652_365260

/-- The constant term in the expansion of (x - 1/x^2)^6 -/
theorem constant_term_expansion : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 2
  Nat.choose n k

/-- The constant term in the expansion of (x - 1/x^2)^6 is 15 -/
theorem constant_term_is_fifteen : constant_term_expansion = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_constant_term_is_fifteen_l3652_365260


namespace NUMINAMATH_CALUDE_inradius_value_l3652_365243

/-- Given a triangle with perimeter p and area A, its inradius r satisfies A = r * p / 2 -/
axiom inradius_formula (p A r : ℝ) : A = r * p / 2

/-- The perimeter of the triangle -/
def p : ℝ := 42

/-- The area of the triangle -/
def A : ℝ := 105

/-- The inradius of the triangle -/
def r : ℝ := 5

theorem inradius_value : r = 5 := by sorry

end NUMINAMATH_CALUDE_inradius_value_l3652_365243


namespace NUMINAMATH_CALUDE_petya_win_probability_l3652_365250

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  maxTake : Nat
  minTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The strategy used by a player -/
inductive Strategy
  | Random
  | Optimal

/-- The result of the game -/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The probability of Petya winning the game -/
def winProbability (game : HeapOfStones) (firstPlayer : Player) 
    (petyaStrategy : Strategy) (computerStrategy : Strategy) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 4, 1⟩
  winProbability game Player.Petya Strategy.Random Strategy.Optimal = 1 / 256 :=
sorry

end NUMINAMATH_CALUDE_petya_win_probability_l3652_365250


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l3652_365264

/-- Given parametric equations x = √t, y = 2√(1-t), prove they are equivalent to x² + y²/4 = 1, where 0 ≤ x ≤ 1 and 0 ≤ y ≤ 2 -/
theorem parametric_to_standard_equation (t : ℝ) (x y : ℝ) 
    (hx : x = Real.sqrt t) (hy : y = 2 * Real.sqrt (1 - t)) :
    x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l3652_365264


namespace NUMINAMATH_CALUDE_no_prime_sum_47_l3652_365294

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem no_prime_sum_47 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 47 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_47_l3652_365294


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3652_365202

theorem min_value_of_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3652_365202


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3652_365267

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |2*x - 3| < 1 → x*(x - 3) < 0) ∧
  (∃ x : ℝ, x*(x - 3) < 0 ∧ ¬(|2*x - 3| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3652_365267


namespace NUMINAMATH_CALUDE_min_ab_is_one_l3652_365258

theorem min_ab_is_one (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b)
  (h4 : ∀ x y, 0 < x → x < y → a^x + b^x < a^y + b^y) :
  ∀ ε > 0, ab ≥ 1 - ε :=
sorry

end NUMINAMATH_CALUDE_min_ab_is_one_l3652_365258


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3652_365223

theorem cow_chicken_problem (C H : ℕ) : 
  4 * C + 2 * H = 2 * (C + H) + 12 → C = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3652_365223


namespace NUMINAMATH_CALUDE_absolute_value_calculation_system_of_inequalities_l3652_365227

-- Part 1
theorem absolute_value_calculation : |(-2 : ℝ)| + Real.sqrt 4 - 2^(0 : ℕ) = 3 := by sorry

-- Part 2
theorem system_of_inequalities (x : ℝ) : 
  (2 * x < 6 ∧ 3 * x > -2 * x + 5) ↔ (1 < x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_absolute_value_calculation_system_of_inequalities_l3652_365227


namespace NUMINAMATH_CALUDE_shower_water_reduction_l3652_365220

theorem shower_water_reduction 
  (original_time original_rate : ℝ) 
  (new_time : ℝ := 3/4 * original_time) 
  (new_rate : ℝ := 3/4 * original_rate) : 
  1 - (new_time * new_rate) / (original_time * original_rate) = 7/16 := by
sorry

end NUMINAMATH_CALUDE_shower_water_reduction_l3652_365220


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3652_365287

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3652_365287


namespace NUMINAMATH_CALUDE_expression_evaluation_l3652_365295

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^4 + 1) / x) * ((y^4 + 1) / y) + ((x^4 - 1) / y) * ((y^4 - 1) / x) = 2 * x^3 * y^3 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3652_365295


namespace NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l3652_365200

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem twelfth_term_of_specific_arithmetic_sequence :
  let a := (1 : ℚ) / 2
  let a2 := (5 : ℚ) / 6
  let d := a2 - a
  arithmeticSequence a d 12 = (25 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_specific_arithmetic_sequence_l3652_365200


namespace NUMINAMATH_CALUDE_zeros_of_f_l3652_365288

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3652_365288


namespace NUMINAMATH_CALUDE_doll_problem_l3652_365201

theorem doll_problem (S : ℕ+) (D : ℕ) 
  (h1 : 4 * S + 3 = D) 
  (h2 : 5 * S = D + 6) : 
  D = 39 := by
sorry

end NUMINAMATH_CALUDE_doll_problem_l3652_365201


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3652_365229

theorem intersection_complement_equals_set (U A B : Set Nat) : 
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 3, 4, 5} →
  B = {2, 3, 6, 7} →
  B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3652_365229


namespace NUMINAMATH_CALUDE_problem_statement_l3652_365238

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3652_365238


namespace NUMINAMATH_CALUDE_factorization_equality_l3652_365239

theorem factorization_equality (x : ℝ) : 6 * x^2 + 5 * x - 1 = (6 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3652_365239


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3652_365266

theorem min_value_of_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  ∃ (m : ℝ), m = 3/4 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3652_365266


namespace NUMINAMATH_CALUDE_other_divisor_proof_l3652_365271

theorem other_divisor_proof (x : ℕ) : x = 5 ↔ 
  x ≠ 11 ∧ 
  x > 0 ∧
  (386 % x = 1 ∧ 386 % 11 = 1) ∧
  ∀ y : ℕ, y < x → y ≠ 11 → y > 0 → (386 % y = 1 ∧ 386 % 11 = 1) → False :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_proof_l3652_365271


namespace NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l3652_365205

theorem derivative_inequality_implies_function_inequality
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x < 2 * f x) : 
  Real.exp 2 * f 0 > f 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l3652_365205


namespace NUMINAMATH_CALUDE_word_reduction_l3652_365222

-- Define the alphabet
inductive Letter
| A
| B
| C

-- Define a word as a list of letters
def Word := List Letter

-- Define the equivalence relation
def equivalent : Word → Word → Prop := sorry

-- Define the duplication operation
def duplicate : Word → Word := sorry

-- Define the removal operation
def remove : Word → Word := sorry

-- Main theorem
theorem word_reduction (w : Word) : 
  ∃ (w' : Word), equivalent w w' ∧ w'.length ≤ 8 := by sorry

end NUMINAMATH_CALUDE_word_reduction_l3652_365222


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3652_365249

/-- A regular polygon with side length 8 and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (n : ℝ) * exterior_angle = 360 →
  n * side_length = 32 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3652_365249


namespace NUMINAMATH_CALUDE_replacement_process_terminates_l3652_365270

/-- A finite sequence of zeros and ones -/
def BinarySequence := List Bool

/-- The operation of replacing "01" with "1000" in a binary sequence -/
def replace01With1000 (seq : BinarySequence) : BinarySequence :=
  match seq with
  | [] => []
  | [x] => [x]
  | false :: true :: xs => true :: false :: false :: false :: xs
  | x :: xs => x :: replace01With1000 xs

/-- The weight of a binary sequence -/
def weight (seq : BinarySequence) : Nat :=
  seq.foldl (λ acc x => if x then 4 * acc else acc + 1) 0

/-- Theorem: The replacement process will eventually terminate -/
theorem replacement_process_terminates (seq : BinarySequence) :
  ∃ n : Nat, ∀ m : Nat, m ≥ n → replace01With1000^[m] seq = replace01With1000^[n] seq :=
sorry

end NUMINAMATH_CALUDE_replacement_process_terminates_l3652_365270


namespace NUMINAMATH_CALUDE_greenfield_basketball_league_players_l3652_365244

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 7

/-- The total expenditure on socks and T-shirts for all players in dollars -/
def total_expenditure : ℕ := 4092

/-- The number of pairs of socks required per player -/
def socks_per_player : ℕ := 2

/-- The number of T-shirts required per player -/
def tshirts_per_player : ℕ := 2

/-- The cost of equipment (socks and T-shirts) for one player in dollars -/
def cost_per_player : ℕ := 
  socks_per_player * sock_cost + tshirts_per_player * (sock_cost + tshirt_additional_cost)

/-- The number of players in the Greenfield Basketball League -/
def num_players : ℕ := total_expenditure / cost_per_player

theorem greenfield_basketball_league_players : num_players = 108 := by
  sorry

end NUMINAMATH_CALUDE_greenfield_basketball_league_players_l3652_365244


namespace NUMINAMATH_CALUDE_max_triangle_side_l3652_365263

theorem max_triangle_side (a b c : ℕ) : 
  a < b → b < c →  -- Ensure different side lengths
  a + b + c = 24 →  -- Perimeter condition
  a + b > c →  -- Triangle inequality
  a + c > b →  -- Triangle inequality
  b + c > a →  -- Triangle inequality
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_l3652_365263


namespace NUMINAMATH_CALUDE_rectangles_4x4_grid_l3652_365212

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles_4x4 : ℕ :=
  let horizontal_lines := 5
  let vertical_lines := 5
  (horizontal_lines.choose 2) * (vertical_lines.choose 2)

/-- Theorem: The number of rectangles on a 4x4 grid is 100 -/
theorem rectangles_4x4_grid :
  num_rectangles_4x4 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_4x4_grid_l3652_365212


namespace NUMINAMATH_CALUDE_tara_ice_cream_purchase_l3652_365233

/-- The number of cartons of ice cream Tara bought -/
def ice_cream_cartons : ℕ := sorry

/-- The number of cartons of yoghurt Tara bought -/
def yoghurt_cartons : ℕ := 4

/-- The cost of one carton of ice cream in dollars -/
def ice_cream_cost : ℕ := 7

/-- The cost of one carton of yoghurt in dollars -/
def yoghurt_cost : ℕ := 1

/-- Theorem stating that Tara bought 19 cartons of ice cream -/
theorem tara_ice_cream_purchase :
  ice_cream_cartons = 19 ∧
  ice_cream_cartons * ice_cream_cost = yoghurt_cartons * yoghurt_cost + 129 :=
by sorry

end NUMINAMATH_CALUDE_tara_ice_cream_purchase_l3652_365233


namespace NUMINAMATH_CALUDE_fraction_equality_l3652_365204

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 2) :
  (a - b) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3652_365204


namespace NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l3652_365245

/-- A regular nonagon -/
structure RegularNonagon where
  /-- Side length of the nonagon -/
  side_length : ℝ
  /-- Length of the shortest diagonal -/
  shortest_diagonal : ℝ
  /-- Length of the longest diagonal -/
  longest_diagonal : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length

/-- 
In a regular nonagon, the length of the longest diagonal 
is equal to the sum of the side length and the shortest diagonal 
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.side_length + n.shortest_diagonal = n.longest_diagonal := by
  sorry


end NUMINAMATH_CALUDE_regular_nonagon_diagonal_sum_l3652_365245


namespace NUMINAMATH_CALUDE_cross_area_l3652_365251

-- Define the grid size
def gridSize : Nat := 6

-- Define the center point of the cross
def centerPoint : (Nat × Nat) := (3, 3)

-- Define the arm length of the cross
def armLength : Nat := 1

-- Define the boundary points of the cross
def boundaryPoints : List (Nat × Nat) := [(3, 1), (1, 3), (3, 3), (3, 5), (5, 3)]

-- Define the interior points of the cross
def interiorPoints : List (Nat × Nat) := [(3, 2), (2, 3), (4, 3), (3, 4)]

-- Theorem: The area of the cross is 6 square units
theorem cross_area : Nat := by
  sorry

end NUMINAMATH_CALUDE_cross_area_l3652_365251


namespace NUMINAMATH_CALUDE_lowest_class_size_l3652_365236

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 12 ∣ n ∧ 24 ∣ n → n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_lowest_class_size_l3652_365236


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3652_365261

theorem inserted_numbers_sum : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = 4 * r ∧ y = 4 * r^2) ∧
  (∃ d : ℝ, y = x + d ∧ 64 = y + d) ∧
  x + y = 131 + 3 * Real.sqrt 129 :=
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3652_365261


namespace NUMINAMATH_CALUDE_dartboard_region_angle_l3652_365230

/-- Given a circular dartboard with a region where the probability of a dart landing is 1/4,
    prove that the central angle of this region is 90°. -/
theorem dartboard_region_angle (probability : ℝ) (angle : ℝ) :
  probability = 1/4 →
  angle = probability * 360 →
  angle = 90 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_region_angle_l3652_365230


namespace NUMINAMATH_CALUDE_equal_group_formations_20_people_l3652_365252

/-- The number of ways to form a group with an equal number of boys and girls -/
def equalGroupFormations (totalPeople boys girls : ℕ) : ℕ :=
  Nat.choose totalPeople boys

/-- Theorem stating that the number of ways to form a group with an equal number
    of boys and girls from 20 people (10 boys and 10 girls) is equal to C(20,10) -/
theorem equal_group_formations_20_people :
  equalGroupFormations 20 10 10 = Nat.choose 20 10 := by
  sorry

#eval equalGroupFormations 20 10 10

end NUMINAMATH_CALUDE_equal_group_formations_20_people_l3652_365252


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_for_ellipse_l3652_365269

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem mn_positive_necessary_not_sufficient_for_ellipse :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_for_ellipse_l3652_365269


namespace NUMINAMATH_CALUDE_system_solution_l3652_365273

theorem system_solution :
  ∃! (x y : ℝ), (x + 3 * y = 7) ∧ (x + 4 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l3652_365273


namespace NUMINAMATH_CALUDE_average_weight_ab_is_40_l3652_365254

def average_weight_abc : ℝ := 42
def average_weight_bc : ℝ := 43
def weight_b : ℝ := 40

theorem average_weight_ab_is_40 :
  let weight_c := 2 * average_weight_bc - weight_b
  let weight_a := 3 * average_weight_abc - weight_b - weight_c
  (weight_a + weight_b) / 2 = 40 := by sorry

end NUMINAMATH_CALUDE_average_weight_ab_is_40_l3652_365254


namespace NUMINAMATH_CALUDE_original_quadratic_equation_l3652_365253

/-- The original quadratic equation given Xiaoming and Xiaohua's mistakes -/
theorem original_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (x y : ℝ), x * y = -6 ∧ x + y = 2 - (-3)) →  -- Xiaoming's roots condition
  (∃ (u v : ℝ), u + v = -2 + 5) →                 -- Xiaohua's roots condition
  a = 1 →                                         -- Coefficient of x^2 is 1
  (a * X^2 + b * X + c = 0 ↔ X^2 - 3 * X - 6 = 0) -- The original equation
  := by sorry

end NUMINAMATH_CALUDE_original_quadratic_equation_l3652_365253


namespace NUMINAMATH_CALUDE_vector_problem_solution_l3652_365292

def vector_problem (a b : ℝ × ℝ) : Prop :=
  a ≠ (0, 0) ∧ b ≠ (0, 0) ∧
  a + b = (-3, 6) ∧ a - b = (-3, 2) →
  a.1^2 + a.2^2 - (b.1^2 + b.2^2) = 21

theorem vector_problem_solution :
  ∀ a b : ℝ × ℝ, vector_problem a b :=
by
  sorry

end NUMINAMATH_CALUDE_vector_problem_solution_l3652_365292


namespace NUMINAMATH_CALUDE_no_fast_connectivity_algorithm_l3652_365297

/-- A graph with 64 vertices -/
def Graph := Fin 64 → Fin 64 → Bool

/-- Number of queries required -/
def required_queries : ℕ := 2016

/-- An algorithm that determines graph connectivity -/
def ConnectivityAlgorithm := Graph → Bool

/-- The number of queries an algorithm makes -/
def num_queries (alg : ConnectivityAlgorithm) : ℕ := sorry

/-- A graph is connected -/
def is_connected (g : Graph) : Prop := sorry

/-- Theorem: No algorithm can determine connectivity in fewer than 2016 queries -/
theorem no_fast_connectivity_algorithm :
  ¬∃ (alg : ConnectivityAlgorithm),
    (∀ g : Graph, alg g = is_connected g) ∧
    (num_queries alg < required_queries) :=
sorry

end NUMINAMATH_CALUDE_no_fast_connectivity_algorithm_l3652_365297


namespace NUMINAMATH_CALUDE_louises_initial_toys_l3652_365207

/-- Proves that Louise initially had 28 toys in her cart -/
theorem louises_initial_toys (initial_toy_cost : ℕ) (teddy_bear_count : ℕ) (teddy_bear_cost : ℕ) (total_cost : ℕ) :
  initial_toy_cost = 10 →
  teddy_bear_count = 20 →
  teddy_bear_cost = 15 →
  total_cost = 580 →
  ∃ (initial_toy_count : ℕ), initial_toy_count * initial_toy_cost + teddy_bear_count * teddy_bear_cost = total_cost ∧ initial_toy_count = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_louises_initial_toys_l3652_365207


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l3652_365289

/-- The distance between the centers of two circular pulleys with an uncrossed belt -/
theorem pulley_centers_distance (r₁ r₂ d : ℝ) (hr₁ : r₁ = 10) (hr₂ : r₂ = 6) (hd : d = 30) :
  Real.sqrt ((r₁ - r₂)^2 + d^2) = 2 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l3652_365289


namespace NUMINAMATH_CALUDE_sum_factorials_mod_12_l3652_365256

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_12 :
  sum_factorials 7 % 12 = (factorial 1 + factorial 2 + factorial 3) % 12 :=
sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_12_l3652_365256


namespace NUMINAMATH_CALUDE_count_valid_concatenations_eq_825957_l3652_365217

def is_valid_integer (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 99

def concatenate (a b c : ℕ) : ℕ := sorry

def count_valid_concatenations : ℕ := sorry

theorem count_valid_concatenations_eq_825957 :
  count_valid_concatenations = 825957 := by sorry

end NUMINAMATH_CALUDE_count_valid_concatenations_eq_825957_l3652_365217


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l3652_365248

theorem sufficient_condition_for_quadratic_inequality :
  ∀ x : ℝ, x ≥ 3 → x^2 - 2*x - 3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l3652_365248


namespace NUMINAMATH_CALUDE_train_arrangement_count_l3652_365247

/-- Represents the number of trains -/
def total_trains : ℕ := 8

/-- Represents the number of trains in each group -/
def trains_per_group : ℕ := 4

/-- Calculates the number of ways to arrange the trains according to the given conditions -/
def train_arrangements : ℕ := sorry

/-- Theorem stating that the number of train arrangements is 720 -/
theorem train_arrangement_count : train_arrangements = 720 := by sorry

end NUMINAMATH_CALUDE_train_arrangement_count_l3652_365247


namespace NUMINAMATH_CALUDE_cucumbers_for_twenty_apples_l3652_365214

/-- The number of cucumbers that can be bought for the price of 20 apples,
    given the cost equivalences between apples, bananas, and cucumbers. -/
theorem cucumbers_for_twenty_apples :
  -- Condition 1: Ten apples cost the same as five bananas
  ∀ (apple_cost banana_cost : ℝ),
  10 * apple_cost = 5 * banana_cost →
  -- Condition 2: Three bananas cost the same as four cucumbers
  ∀ (cucumber_cost : ℝ),
  3 * banana_cost = 4 * cucumber_cost →
  -- Conclusion: 20 apples are equivalent in cost to 13 cucumbers
  20 * apple_cost = 13 * cucumber_cost :=
by
  sorry

end NUMINAMATH_CALUDE_cucumbers_for_twenty_apples_l3652_365214


namespace NUMINAMATH_CALUDE_series_convergence_l3652_365290

/-- The infinite sum of the given series converges to 2 -/
theorem series_convergence : 
  ∑' k : ℕ, (8 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l3652_365290


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3652_365241

/-- Given an ellipse with equation x^2 + ky^2 = 2, if its focus lies on the y-axis
    and its focal length is 4, then k = 1/3 -/
theorem ellipse_focal_length (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*y^2 = 2) →  -- ellipse equation
  (∃ c : ℝ, c > 0 ∧ (0, c) ∈ {p : ℝ × ℝ | p.1^2 + k*p.2^2 = 2}) →  -- focus on y-axis
  (∃ f : ℝ, f = 4 ∧ f^2 = 2/k - 2) →  -- focal length is 4
  k = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l3652_365241


namespace NUMINAMATH_CALUDE_eldest_child_age_l3652_365279

theorem eldest_child_age 
  (n : ℕ) 
  (d : ℕ) 
  (sum : ℕ) 
  (h1 : n = 5) 
  (h2 : d = 2) 
  (h3 : sum = 50) : 
  (sum - (n * (n - 1) / 2) * d) / n + (n - 1) * d = 14 := by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l3652_365279


namespace NUMINAMATH_CALUDE_product_of_complex_in_polar_form_specific_complex_product_l3652_365272

/-- 
Given two complex numbers in polar form, prove that their product 
is equal to the product of their magnitudes and the sum of their angles.
-/
theorem product_of_complex_in_polar_form 
  (z₁ : ℂ) (z₂ : ℂ) (r₁ r₂ θ₁ θ₂ : ℝ) :
  z₁ = r₁ * Complex.exp (θ₁ * Complex.I) →
  z₂ = r₂ * Complex.exp (θ₂ * Complex.I) →
  r₁ > 0 →
  r₂ > 0 →
  z₁ * z₂ = (r₁ * r₂) * Complex.exp ((θ₁ + θ₂) * Complex.I) :=
by sorry

/-- 
Prove that the product of 5cis(25°) and 4cis(48°) is equal to 20cis(73°).
-/
theorem specific_complex_product :
  let z₁ : ℂ := 5 * Complex.exp (25 * π / 180 * Complex.I)
  let z₂ : ℂ := 4 * Complex.exp (48 * π / 180 * Complex.I)
  z₁ * z₂ = 20 * Complex.exp (73 * π / 180 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_product_of_complex_in_polar_form_specific_complex_product_l3652_365272


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_rectangular_solid_l3652_365265

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem sphere_surface_area_of_rectangular_solid (l w h : ℝ) (S : ℝ) :
  l = 2 →
  w = 2 →
  h = 1 →
  S = 4 * Real.pi * ((l^2 + w^2 + h^2) / 4) →
  S = 9 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_rectangular_solid_l3652_365265


namespace NUMINAMATH_CALUDE_dependent_variable_influences_l3652_365299

/-- Represents a linear regression model --/
structure LinearRegressionModel where
  y : ℝ  -- dependent variable
  x : ℝ  -- independent variable
  b : ℝ  -- slope
  a : ℝ  -- intercept
  e : ℝ  -- random error term

/-- The linear regression equation --/
def linear_regression_equation (model : LinearRegressionModel) : ℝ :=
  model.b * model.x + model.a + model.e

/-- Theorem stating that the dependent variable is influenced by both the independent variable and other factors --/
theorem dependent_variable_influences (model : LinearRegressionModel) :
  ∃ (other_factors : ℝ), model.y = linear_regression_equation model ∧ model.e ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_dependent_variable_influences_l3652_365299


namespace NUMINAMATH_CALUDE_tyson_basketball_scores_l3652_365286

/-- Represents the number of times Tyson scored points in each category -/
structure BasketballScores where
  threePointers : Nat
  twoPointers : Nat
  onePointers : Nat

/-- Calculates the total points scored given a BasketballScores structure -/
def totalPoints (scores : BasketballScores) : Nat :=
  3 * scores.threePointers + 2 * scores.twoPointers + scores.onePointers

theorem tyson_basketball_scores :
  ∃ (scores : BasketballScores),
    scores.threePointers = 15 ∧
    scores.twoPointers = 12 ∧
    scores.onePointers % 2 = 0 ∧
    totalPoints scores = 75 ∧
    scores.onePointers = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_scores_l3652_365286
