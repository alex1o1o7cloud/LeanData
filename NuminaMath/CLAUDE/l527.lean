import Mathlib

namespace NUMINAMATH_CALUDE_notebook_increase_l527_52739

theorem notebook_increase (initial_count mother_bought father_bought : ℕ) :
  initial_count = 33 →
  mother_bought = 7 →
  father_bought = 14 →
  (initial_count + mother_bought + father_bought) - initial_count = 21 := by
  sorry

end NUMINAMATH_CALUDE_notebook_increase_l527_52739


namespace NUMINAMATH_CALUDE_complex_equation_solution_l527_52716

theorem complex_equation_solution (z : ℂ) : 3 + 2 * Complex.I * z = 7 - 4 * Complex.I * z ↔ z = -2 * Complex.I / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l527_52716


namespace NUMINAMATH_CALUDE_evaluate_expression_l527_52788

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l527_52788


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l527_52749

theorem diophantine_equation_implication 
  (a b : ℤ) 
  (ha : ¬∃ (n : ℤ), a = n^2) 
  (hb : ¬∃ (n : ℤ), b = n^2) 
  (h : ∃ (x y z w : ℤ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0 ∧ x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0) :
  ∃ (X Y Z : ℤ), X ≠ 0 ∨ Y ≠ 0 ∨ Z ≠ 0 ∧ X^2 - a*Y^2 - b*Z^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l527_52749


namespace NUMINAMATH_CALUDE_negative_odd_number_representation_l527_52763

theorem negative_odd_number_representation (x : ℤ) :
  (x < 0 ∧ x % 2 = 1) → ∃ n : ℕ+, x = -2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_odd_number_representation_l527_52763


namespace NUMINAMATH_CALUDE_student_calculation_error_l527_52773

/-- Represents a repeating decimal of the form 1.̅cd̅ where c and d are single digits -/
def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

/-- The difference between the correct calculation and the student's miscalculation -/
def calculation_difference (c d : ℕ) : ℚ :=
  84 * (repeating_decimal c d - (1 + (c : ℚ) / 10 + (d : ℚ) / 100))

theorem student_calculation_error :
  ∃ (c d : ℕ), c < 10 ∧ d < 10 ∧ calculation_difference c d = 0.6 ∧ c * 10 + d = 71 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l527_52773


namespace NUMINAMATH_CALUDE_binomial_square_constant_l527_52738

/-- If 9x^2 - 27x + a is the square of a binomial, then a = 20.25 -/
theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 27*x + a = (3*x + b)^2) → a = 20.25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l527_52738


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_l527_52754

-- Define the hyperbola equation
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define that the hyperbola and ellipse share the same foci
def same_foci (m : ℝ) : Prop := ∃ (a b : ℝ), (hyperbola m a b ∧ ellipse a b)

-- Theorem statement
theorem hyperbola_ellipse_foci (m : ℝ) (h : same_foci m) : m = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_l527_52754


namespace NUMINAMATH_CALUDE_inverse_composition_l527_52732

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the given condition
axiom inverse_relation (x : ℝ) : (f⁻¹ ∘ g) x = 4 * x - 1

-- State the theorem to be proved
theorem inverse_composition : g⁻¹ (f 5) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l527_52732


namespace NUMINAMATH_CALUDE_complex_number_properties_l527_52720

theorem complex_number_properties (z : ℂ) (h : Complex.abs z ^ 2 + 2 * z - Complex.I * 2 = 0) :
  z = -1 + Complex.I ∧ Complex.abs z + Complex.abs (z + 3 * Complex.I) > Complex.abs (2 * z + 3 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l527_52720


namespace NUMINAMATH_CALUDE_tangent_and_inequality_l527_52719

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1) - a

theorem tangent_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ = 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  (∀ x t : ℝ, x > t ∧ t ≥ 0 → Real.exp (x - t) + Real.log (t + 1) > Real.log (x + 1) + 1) ∧
  (f a = f 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_l527_52719


namespace NUMINAMATH_CALUDE_f_monotone_increasing_interval_l527_52700

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) := x^2 + 2*x + 1

/-- The monotonically increasing interval of f(x) is [-1, +∞) -/
theorem f_monotone_increasing_interval :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_interval_l527_52700


namespace NUMINAMATH_CALUDE_johns_allowance_l527_52755

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (A > 0) →                                           -- Allowance is positive
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 96 / 100 = A) →  -- Total spending equals allowance
  (A = 36 / 10) :=                                    -- Allowance is $3.60
by sorry

end NUMINAMATH_CALUDE_johns_allowance_l527_52755


namespace NUMINAMATH_CALUDE_no_real_roots_l527_52743

theorem no_real_roots : ∀ x : ℝ, x^2 + 3*x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l527_52743


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l527_52768

open Set
open Function

theorem function_inequality_solution_set 
  (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x, deriv f x < (1/2)) :
  {x | f x < x/2 + 1/2} = {x | x > 1} := by
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l527_52768


namespace NUMINAMATH_CALUDE_division_equality_l527_52777

theorem division_equality : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l527_52777


namespace NUMINAMATH_CALUDE_fraction_subtraction_l527_52709

theorem fraction_subtraction : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l527_52709


namespace NUMINAMATH_CALUDE_water_height_in_aquarium_l527_52725

/-- Proves that the height of water in an aquarium with given dimensions and volume of water is 10 cm. -/
theorem water_height_in_aquarium :
  let aquarium_length : ℝ := 50
  let aquarium_breadth : ℝ := 20
  let aquarium_height : ℝ := 40
  let water_volume : ℝ := 10000  -- 10 litres * 1000 cm³/litre
  let water_height : ℝ := water_volume / (aquarium_length * aquarium_breadth)
  water_height = 10 := by sorry

end NUMINAMATH_CALUDE_water_height_in_aquarium_l527_52725


namespace NUMINAMATH_CALUDE_zoo_visitors_l527_52712

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price : ℕ) (total_sales : ℕ) 
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults kids : ℕ), 
    adults + kids = total_people ∧
    adults * adult_price + kids * kid_price = total_sales ∧
    kids = 202 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l527_52712


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l527_52729

theorem min_distance_between_curves (a b c d : ℝ) :
  (a - 2*Real.exp a)/b = (1 - c)/(d - 1) ∧ (a - 2*Real.exp a)/b = 1 →
  (∀ x y z w : ℝ, (x - 2*Real.exp x)/y = (1 - z)/(w - 1) ∧ (x - 2*Real.exp x)/y = 1 →
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l527_52729


namespace NUMINAMATH_CALUDE_candle_burning_l527_52728

/-- Candle burning problem -/
theorem candle_burning (h₀ : ℕ) (burn_rate : ℕ → ℝ) (T : ℝ) : 
  (h₀ = 150) →
  (∀ k, burn_rate k = 15 * k) →
  (T = (15 : ℝ) * (h₀ * (h₀ + 1) / 2)) →
  (∃ m : ℕ, 
    (7.5 * m * (m + 1) ≤ T / 2) ∧ 
    (T / 2 < 7.5 * (m + 1) * (m + 2)) ∧
    (h₀ - m = 45)) :=
by sorry

end NUMINAMATH_CALUDE_candle_burning_l527_52728


namespace NUMINAMATH_CALUDE_fair_entrance_fee_l527_52796

/-- Represents the entrance fee structure and ride costs at a fair --/
structure FairPrices where
  under18Fee : ℝ
  over18Fee : ℝ
  rideCost : ℝ
  under18Fee_pos : 0 < under18Fee
  over18Fee_eq : over18Fee = 1.2 * under18Fee
  rideCost_eq : rideCost = 0.5

/-- Calculates the total cost for a group at the fair --/
def totalCost (prices : FairPrices) (numUnder18 : ℕ) (numOver18 : ℕ) (totalRides : ℕ) : ℝ :=
  numUnder18 * prices.under18Fee + numOver18 * prices.over18Fee + totalRides * prices.rideCost

/-- The main theorem stating the entrance fee for persons under 18 --/
theorem fair_entrance_fee :
  ∃ (prices : FairPrices), totalCost prices 2 1 9 = 20.5 ∧ prices.under18Fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_fair_entrance_fee_l527_52796


namespace NUMINAMATH_CALUDE_range_of_m_l527_52793

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (25 - m) + y^2 / (m - 7) = 1 ∧ 
  (25 - m > 0) ∧ (m - 7 > 0) ∧ (25 - m > m - 7)

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ (e : ℝ), (∃ (x y : ℝ), y^2 / 5 - x^2 / m = 1) ∧ 
  1 < e ∧ e < 2 ∧ e^2 = (5 + m) / 5

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(¬(p m) ∨ ¬(q m))) → (7 < m ∧ m < 15) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l527_52793


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l527_52703

/-- Calculates the total percentage loss due to thefts for a shopkeeper --/
theorem shopkeeper_theft_loss (X : ℝ) (h : X > 0) : 
  let remaining_after_first_theft := 0.7 * X
  let remaining_after_first_sale := 0.75 * remaining_after_first_theft
  let remaining_after_second_theft := 0.6 * remaining_after_first_sale
  let remaining_after_second_sale := 0.7 * remaining_after_second_theft
  let final_remaining := 0.8 * remaining_after_second_sale
  (X - final_remaining) / X * 100 = 82.36 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l527_52703


namespace NUMINAMATH_CALUDE_exists_small_triangle_area_l527_52786

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a point to be within the given bounds
def withinBounds (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define the condition for three points to be non-collinear
def nonCollinear (p q r : LatticePoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

-- Calculate the area of a triangle formed by three points
def triangleArea (p q r : LatticePoint) : ℚ :=
  let a := (q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)
  (abs a : ℚ) / 2

-- Main theorem
theorem exists_small_triangle_area 
  (P : Fin 6 → LatticePoint)
  (h_bounds : ∀ i, withinBounds (P i))
  (h_noncollinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (P i) (P j) (P k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (P i) (P j) (P k) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_small_triangle_area_l527_52786


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l527_52787

theorem right_triangle_squares_area (XY YZ XZ : ℝ) :
  XY = 5 →
  XZ = 13 →
  XY^2 + YZ^2 = XZ^2 →
  XY^2 + YZ^2 = 169 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l527_52787


namespace NUMINAMATH_CALUDE_pencil_distribution_l527_52791

theorem pencil_distribution (boxes : Real) (pencils_per_box : Real) (students : Nat) :
  boxes = 4.0 →
  pencils_per_box = 648.0 →
  students = 36 →
  (boxes * pencils_per_box) / students = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l527_52791


namespace NUMINAMATH_CALUDE_product_zero_l527_52766

theorem product_zero (b : ℤ) (h : b = 4) : 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l527_52766


namespace NUMINAMATH_CALUDE_inequality_proof_l527_52764

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1+a)*(1+b)*(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l527_52764


namespace NUMINAMATH_CALUDE_selection_theorem_l527_52781

/-- The number of candidates --/
def n : ℕ := 5

/-- The number of languages --/
def k : ℕ := 3

/-- The number of candidates unwilling to study Hebrew --/
def m : ℕ := 2

/-- The number of ways to select students for the languages --/
def selection_methods : ℕ := (n - m) * (Nat.choose (n - 1) (k - 1)) * 2

theorem selection_theorem : selection_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l527_52781


namespace NUMINAMATH_CALUDE_exponent_property_l527_52702

theorem exponent_property (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_property_l527_52702


namespace NUMINAMATH_CALUDE_roots_product_l527_52778

theorem roots_product (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) → 
  (3 * e^2 + 4 * e - 7 = 0) → 
  (d - 1) * (e - 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_product_l527_52778


namespace NUMINAMATH_CALUDE_mrs_petersons_change_l527_52701

theorem mrs_petersons_change (number_of_tumblers : ℕ) (price_per_tumbler : ℕ) (number_of_bills : ℕ) (bill_value : ℕ) : 
  number_of_tumblers = 10 →
  price_per_tumbler = 45 →
  number_of_bills = 5 →
  bill_value = 100 →
  (number_of_bills * bill_value) - (number_of_tumblers * price_per_tumbler) = 50 :=
by
  sorry

#check mrs_petersons_change

end NUMINAMATH_CALUDE_mrs_petersons_change_l527_52701


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l527_52759

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 < |1 - x| ∧ |1 - x| ≤ 2} = Set.Icc (-1) 0 ∪ Set.Ioc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l527_52759


namespace NUMINAMATH_CALUDE_inverse_proportion_min_value_l527_52734

/-- Given an inverse proportion function y = k/x, prove that if the maximum value of y is 4
    when -2 ≤ x ≤ -1, then the minimum value of y is -1/2 when x ≥ 8 -/
theorem inverse_proportion_min_value (k : ℝ) :
  (∀ x, -2 ≤ x → x ≤ -1 → k / x ≤ 4) →
  (∃ x, -2 ≤ x ∧ x ≤ -1 ∧ k / x = 4) →
  (∀ x, x ≥ 8 → k / x ≥ -1/2) ∧
  (∃ x, x ≥ 8 ∧ k / x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_min_value_l527_52734


namespace NUMINAMATH_CALUDE_rectangle_length_l527_52741

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l527_52741


namespace NUMINAMATH_CALUDE_rosys_age_l527_52771

theorem rosys_age (rosy_age : ℕ) : 
  (rosy_age + 12 + 4 = 2 * (rosy_age + 4)) → rosy_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_rosys_age_l527_52771


namespace NUMINAMATH_CALUDE_ab_multiplier_l527_52710

theorem ab_multiplier (a b m : ℝ) : 
  4 * a = 30 ∧ 5 * b = 30 ∧ m * (a * b) = 1800 → m = 40 := by
  sorry

end NUMINAMATH_CALUDE_ab_multiplier_l527_52710


namespace NUMINAMATH_CALUDE_largest_number_l527_52742

def a : ℚ := 24680 + 1 / 1357
def b : ℚ := 24680 - 1 / 1357
def c : ℚ := 24680 * (1 / 1357)
def d : ℚ := 24680 / (1 / 1357)
def e : ℚ := 24680.1357

theorem largest_number : 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l527_52742


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l527_52792

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

/-- Theorem stating that 110101₂ equals 53 in decimal -/
theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l527_52792


namespace NUMINAMATH_CALUDE_no_x_satisfies_arccos_lt_arcsin_l527_52730

theorem no_x_satisfies_arccos_lt_arcsin : ¬∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ Real.arccos x < Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_x_satisfies_arccos_lt_arcsin_l527_52730


namespace NUMINAMATH_CALUDE_largest_last_digit_l527_52758

/-- A string of digits satisfying the problem conditions -/
def ValidString : Type := 
  {s : List Nat // s.length = 2007 ∧ s.head! = 2 ∧ 
    ∀ i, i < 2006 → (s.get! i * 10 + s.get! (i+1)) % 23 = 0 ∨ 
                     (s.get! i * 10 + s.get! (i+1)) % 37 = 0}

/-- The theorem stating the largest possible last digit -/
theorem largest_last_digit (s : ValidString) : s.val.getLast! ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_l527_52758


namespace NUMINAMATH_CALUDE_no_stop_probability_theorem_l527_52762

/-- Represents the probability of a green light at a traffic point -/
def greenLightProbability (duration : ℕ) : ℚ := duration / 60

/-- The probability that a car doesn't stop at all three points -/
def noStopProbability (durationA durationB durationC : ℕ) : ℚ :=
  (greenLightProbability durationA) * (greenLightProbability durationB) * (greenLightProbability durationC)

theorem no_stop_probability_theorem (durationA durationB durationC : ℕ) 
  (hA : durationA = 25) (hB : durationB = 35) (hC : durationC = 45) :
  noStopProbability durationA durationB durationC = 35 / 192 := by
  sorry

end NUMINAMATH_CALUDE_no_stop_probability_theorem_l527_52762


namespace NUMINAMATH_CALUDE_seaplane_speed_l527_52780

theorem seaplane_speed (v : ℝ) (h1 : v > 0) : 
  (2 : ℝ) / ((1 / v) + (1 / 72)) = 91 → v = 6552 / 53 := by
  sorry

end NUMINAMATH_CALUDE_seaplane_speed_l527_52780


namespace NUMINAMATH_CALUDE_set_A_determination_l527_52779

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem set_A_determination (A : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : A ∩ {1, 2, 3} = {2})
  (h3 : A ∪ {1, 2, 3} = U) :
  A = {2, 4} := by
sorry


end NUMINAMATH_CALUDE_set_A_determination_l527_52779


namespace NUMINAMATH_CALUDE_polynomial_identity_l527_52726

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) → 
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l527_52726


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l527_52790

/-- The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die -/
theorem probability_four_ones_in_five_rolls : ℚ :=
  25 / 7776

/-- A fair six-sided die -/
def fair_six_sided_die : Finset ℕ := Finset.range 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of desired ones -/
def desired_ones : ℕ := 4

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

/-- The main theorem: The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die is 25/7776 -/
theorem prob_four_ones_in_five_rolls :
  (Nat.choose num_rolls desired_ones) *
  (prob_single_roll 1) ^ desired_ones *
  (1 - prob_single_roll 1) ^ (num_rolls - desired_ones) =
  probability_four_ones_in_five_rolls := by sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l527_52790


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l527_52744

theorem sarahs_bowling_score (s g : ℕ) : 
  s = g + 50 ∧ (s + g) / 2 = 105 → s = 130 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l527_52744


namespace NUMINAMATH_CALUDE_smallest_k_for_product_sign_change_l527_52772

def sequence_a (n : ℕ) : ℚ :=
  15 - 2/3 * (n - 1)

theorem smallest_k_for_product_sign_change :
  let a := sequence_a
  (∀ n : ℕ, n ≥ 1 → 3 * a (n + 1) = 3 * a n - 2) →
  (∃ k : ℕ, k > 0 ∧ a k * a (k + 1) < 0) →
  (∀ j : ℕ, 0 < j → j < 23 → a j * a (j + 1) ≥ 0) →
  a 23 * a 24 < 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_product_sign_change_l527_52772


namespace NUMINAMATH_CALUDE_money_difference_proof_l527_52705

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters as a function of q -/
def charles_quarters (q : ℤ) : ℤ := 7 * q + 2

/-- Richard's quarters as a function of q -/
def richard_quarters (q : ℤ) : ℤ := 3 * q + 8

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℤ) : ℤ :=
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℤ) :
  money_difference_in_nickels q = 20 * q - 30 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_proof_l527_52705


namespace NUMINAMATH_CALUDE_sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l527_52753

-- Define the first and second quadrants
def first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def second_quadrant (θ : ℝ) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem sin_inequality_implies_angle_inequality (α β : ℝ) :
  Real.sin α ≠ Real.sin β → α ≠ β :=
sorry

theorem sin_positive_in_first_and_second_quadrant (θ : ℝ) :
  (first_quadrant θ ∨ second_quadrant θ) → Real.sin θ > 0 :=
sorry

end NUMINAMATH_CALUDE_sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l527_52753


namespace NUMINAMATH_CALUDE_danai_decorations_l527_52724

/-- The number of decorations Danai will put up in total -/
def total_decorations (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + pumpkins + cauldron + budget_left + left_to_put_up

/-- Theorem stating the total number of decorations Danai will put up -/
theorem danai_decorations :
  ∀ (skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up : ℕ),
    skulls = 12 →
    broomsticks = 4 →
    spiderwebs = 12 →
    pumpkins = 2 * spiderwebs →
    cauldron = 1 →
    budget_left = 20 →
    left_to_put_up = 10 →
    total_decorations skulls broomsticks spiderwebs pumpkins cauldron budget_left left_to_put_up = 83 :=
by sorry

end NUMINAMATH_CALUDE_danai_decorations_l527_52724


namespace NUMINAMATH_CALUDE_dividend_problem_l527_52740

theorem dividend_problem (M D Q : ℕ) (h1 : M = 6 * D) (h2 : D = 4 * Q) : M = 144 := by
  sorry

end NUMINAMATH_CALUDE_dividend_problem_l527_52740


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l527_52784

def point_P : ℝ × ℝ := (5, -12)

theorem distance_to_x_axis :
  ‖point_P.2‖ = 12 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l527_52784


namespace NUMINAMATH_CALUDE_divides_two_pow_minus_one_l527_52783

theorem divides_two_pow_minus_one (n : ℕ) : n > 0 → (n ∣ 2^n - 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divides_two_pow_minus_one_l527_52783


namespace NUMINAMATH_CALUDE_odd_power_of_seven_plus_one_divisible_by_eight_l527_52789

theorem odd_power_of_seven_plus_one_divisible_by_eight (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, (7^n : ℤ) + 1 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_power_of_seven_plus_one_divisible_by_eight_l527_52789


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l527_52717

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (x + 3) * (x - 1) = 2 * x - 4 ↔ x^2 + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l527_52717


namespace NUMINAMATH_CALUDE_angle_PQT_measure_l527_52799

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle PQT in a regular octagon -/
def angle_PQT (octagon : RegularOctagon) : ℝ :=
  22.5

theorem angle_PQT_measure (octagon : RegularOctagon) :
  angle_PQT octagon = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_PQT_measure_l527_52799


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l527_52737

/-- A geometric sequence with its sum function -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1  -- Geometric sequence property
  sum_property : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))  -- Sum formula

/-- Theorem: For a geometric sequence with S_5 = 3 and S_10 = 9, S_15 = 21 -/
theorem geometric_sequence_sum (seq : GeometricSequence) 
  (h1 : seq.S 5 = 3) 
  (h2 : seq.S 10 = 9) : 
  seq.S 15 = 21 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l527_52737


namespace NUMINAMATH_CALUDE_x_range_given_quadratic_inequality_l527_52704

theorem x_range_given_quadratic_inequality (x : ℝ) :
  4 - x^2 ≤ 0 → x ≤ -2 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_given_quadratic_inequality_l527_52704


namespace NUMINAMATH_CALUDE_kristy_baked_69_cookies_l527_52795

def cookie_problem (C : ℕ) : Prop :=
  let remaining_after_kristy := C - 3
  let remaining_after_brother := remaining_after_kristy / 2
  let remaining_after_friend1 := remaining_after_brother - 4
  let friend2_took := 2 * 4
  let friend2_returned := friend2_took / 4
  let remaining_after_friend2 := remaining_after_friend1 - (friend2_took - friend2_returned)
  let remaining_after_friend3 := remaining_after_friend2 - 8
  let remaining_after_friend4 := remaining_after_friend3 - 3
  let final_remaining := remaining_after_friend4 - 7
  2 * final_remaining = 10

theorem kristy_baked_69_cookies : ∃ C : ℕ, cookie_problem C ∧ C = 69 := by
  sorry

end NUMINAMATH_CALUDE_kristy_baked_69_cookies_l527_52795


namespace NUMINAMATH_CALUDE_mechanics_total_charge_l527_52722

/-- Calculates the total amount charged by two mechanics working on a car. -/
theorem mechanics_total_charge
  (hours1 : ℕ)  -- Hours worked by the first mechanic
  (hours2 : ℕ)  -- Hours worked by the second mechanic
  (rate : ℕ)    -- Combined hourly rate in dollars
  (h1 : hours1 = 10)  -- First mechanic worked for 10 hours
  (h2 : hours2 = 5)   -- Second mechanic worked for 5 hours
  (h3 : rate = 160)   -- Combined hourly rate is $160
  : (hours1 + hours2) * rate = 2400 := by
  sorry


end NUMINAMATH_CALUDE_mechanics_total_charge_l527_52722


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l527_52713

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 → W > 0 →
  (∃ d : ℤ, Z - Y = Y - X ∧ Z - Y = d) →
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →
  Z = (9 : ℚ) / 5 * Y →
  (∀ a b c d : ℕ, a > 0 → b > 0 → c > 0 → d > 0 →
    (∃ d' : ℤ, c - b = b - a ∧ c - b = d') →
    (∃ r' : ℚ, c = r' * b ∧ d = r' * c) →
    c = (9 : ℚ) / 5 * b →
    X + Y + Z + W ≤ a + b + c + d) →
  X + Y + Z + W = 156 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l527_52713


namespace NUMINAMATH_CALUDE_alyssas_spending_l527_52750

/-- Calculates the total spending given an amount paid and a refund. -/
def totalSpending (amountPaid refund : ℚ) : ℚ :=
  amountPaid - refund

/-- Proves that Alyssa's total spending is $2.23 given the conditions. -/
theorem alyssas_spending :
  let grapesPayment : ℚ := 12.08
  let cherriesRefund : ℚ := 9.85
  totalSpending grapesPayment cherriesRefund = 2.23 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_spending_l527_52750


namespace NUMINAMATH_CALUDE_exists_point_P_satisfying_condition_l527_52731

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with edge length 10 -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D
  D' : Point3D

/-- Represents the plane intersecting the cube -/
structure IntersectingPlane where
  P : Point3D
  Q : Point3D
  R : Point3D
  S : Point3D
  T : Point3D

/-- Function to calculate distance between two points -/
def distance (p1 p2 : Point3D) : ℝ :=
  sorry

/-- Theorem stating the existence of point P satisfying the condition -/
theorem exists_point_P_satisfying_condition 
  (cube : Cube) 
  (plane : IntersectingPlane) 
  (h1 : distance cube.A plane.R / distance plane.R cube.B = 7 / 3)
  (h2 : distance cube.C plane.S / distance plane.S cube.B = 7 / 3)
  (h3 : plane.P.x = cube.D.x ∧ plane.P.y = cube.D.y)
  (h4 : plane.Q.x = cube.A.x ∧ plane.Q.y = cube.A.y)
  (h5 : plane.R.z = cube.A.z ∧ plane.R.y = cube.A.y)
  (h6 : plane.S.z = cube.B.z ∧ plane.S.x = cube.B.x)
  (h7 : plane.T.x = cube.C.x ∧ plane.T.y = cube.C.y) :
  ∃ (P : Point3D), 
    P.x = cube.D.x ∧ P.y = cube.D.y ∧ 
    cube.D.z ≤ P.z ∧ P.z ≤ cube.D'.z ∧
    2 * distance plane.Q plane.R = distance P plane.Q + distance plane.R plane.S :=
sorry

end NUMINAMATH_CALUDE_exists_point_P_satisfying_condition_l527_52731


namespace NUMINAMATH_CALUDE_no_real_roots_for_polynomial_l527_52723

theorem no_real_roots_for_polynomial (a : ℝ) : 
  ¬∃ x : ℝ, x^4 + a^2*x^3 - 2*x^2 + a*x + 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_for_polynomial_l527_52723


namespace NUMINAMATH_CALUDE_polyhedron_edges_existence_l527_52767

/-- The number of edges in the initial polyhedra we can start with -/
def initial_edges : List Nat := [8, 9, 10]

/-- The number of edges added when slicing off a triangular angle -/
def edges_per_slice : Nat := 3

/-- Proposition: For any natural number n ≥ 8, there exists a polyhedron with exactly n edges -/
theorem polyhedron_edges_existence (n : Nat) (h : n ≥ 8) :
  ∃ (k : Nat) (m : Nat), k ∈ initial_edges ∧ n = k + m * edges_per_slice :=
sorry

end NUMINAMATH_CALUDE_polyhedron_edges_existence_l527_52767


namespace NUMINAMATH_CALUDE_expected_sixes_two_dice_l527_52769

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling a 6 on a single die -/
def prob_six : ℚ := 1 / num_sides

/-- The expected number of 6's when rolling two dice -/
def expected_sixes : ℚ := 1 / 4

/-- Theorem stating that the expected number of 6's when rolling two eight-sided dice is 1/4 -/
theorem expected_sixes_two_dice : 
  expected_sixes = 2 * prob_six := by sorry

end NUMINAMATH_CALUDE_expected_sixes_two_dice_l527_52769


namespace NUMINAMATH_CALUDE_regularity_lemma_l527_52782

/-- A graph represented as a set of vertices and a set of edges -/
structure Graph (V : Type) where
  vertices : Set V
  edges : Set (V × V)

/-- The maximum degree of a graph -/
def max_degree (G : Graph V) : ℕ := sorry

/-- A regularity graph with parameters ε, ℓ, and d -/
structure RegularityGraph (V : Type) extends Graph V where
  ε : ℝ
  ℓ : ℕ
  d : ℝ

/-- The s-closure of a regularity graph -/
def s_closure (R : RegularityGraph V) (s : ℕ) : Graph V := sorry

/-- Subgraph relation -/
def is_subgraph (H G : Graph V) : Prop := sorry

theorem regularity_lemma {V : Type} (d : ℝ) (Δ : ℕ) 
  (hd : d ∈ Set.Icc 0 1) (hΔ : Δ ≥ 1) :
  ∃ ε₀ > 0, ∀ (G H : Graph V) (s : ℕ) (R : RegularityGraph V),
    max_degree H ≤ Δ →
    R.ε ≤ ε₀ →
    R.ℓ ≥ 2 * s / d^Δ →
    R.d = d →
    is_subgraph H (s_closure R s) →
    is_subgraph H G :=
sorry

end NUMINAMATH_CALUDE_regularity_lemma_l527_52782


namespace NUMINAMATH_CALUDE_third_face_area_l527_52752

-- Define the properties of the cuboidal box
def cuboidal_box (l w h : ℝ) : Prop :=
  l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 72 ∧
  w * h = 60 ∧
  l * w * h = 720

-- Theorem statement
theorem third_face_area (l w h : ℝ) :
  cuboidal_box l w h → l * h = 120 := by
  sorry

end NUMINAMATH_CALUDE_third_face_area_l527_52752


namespace NUMINAMATH_CALUDE_quadrilateral_area_l527_52776

/-- The area of a quadrilateral with given sides and one angle -/
theorem quadrilateral_area (a b c d : Real) (α : Real) : 
  a = 52 →
  b = 56 →
  c = 33 →
  d = 39 →
  α = 112 + 37 / 60 + 12 / 3600 →
  ∃ (area : Real), abs (area - 1774) < 1 ∧ 
  area = (1/2) * a * d * Real.sin α + 
          Real.sqrt ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - b) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - c) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l527_52776


namespace NUMINAMATH_CALUDE_remainder_of_product_with_modular_inverse_l527_52721

theorem remainder_of_product_with_modular_inverse (n a b : ℤ) : 
  n > 0 → (a * b) % n = 1 % n → (a * b) % n = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_product_with_modular_inverse_l527_52721


namespace NUMINAMATH_CALUDE_inequality_proof_l527_52751

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l527_52751


namespace NUMINAMATH_CALUDE_remainder_of_sequence_sum_l527_52756

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem remainder_of_sequence_sum :
  ∃ n : ℕ, 
    arithmetic_sequence 1 6 n = 403 ∧ 
    sequence_sum 1 6 n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sequence_sum_l527_52756


namespace NUMINAMATH_CALUDE_inequality_solutions_l527_52747

theorem inequality_solutions :
  (∀ x : ℝ, x^2 - 5*x + 5 > 0 ↔ (x > (5 + Real.sqrt 5) / 2 ∨ x < (5 - Real.sqrt 5) / 2)) ∧
  (∀ x : ℝ, -2*x^2 + x - 3 < 0) := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l527_52747


namespace NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l527_52733

theorem prime_divides_sum_of_squares (m n : ℕ) :
  Prime (m + n + 1) →
  (m + n + 1) ∣ (2 * (m^2 + n^2) - 1) →
  m = n := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l527_52733


namespace NUMINAMATH_CALUDE_hidden_dots_sum_l527_52748

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def DieSum : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumDice : ℕ := 4

/-- The visible numbers on the stack -/
def VisibleNumbers : Finset ℕ := {1, 2, 3, 5, 6}

/-- The sum of visible numbers -/
def VisibleSum : ℕ := Finset.sum VisibleNumbers id

theorem hidden_dots_sum :
  NumDice * DieSum - VisibleSum = 67 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_sum_l527_52748


namespace NUMINAMATH_CALUDE_cookie_sheet_perimeter_is_24_l527_52785

/-- The perimeter of a rectangular cookie sheet -/
def cookie_sheet_perimeter (width length : ℝ) : ℝ :=
  2 * width + 2 * length

/-- Theorem: The perimeter of a rectangular cookie sheet with width 10 inches and length 2 inches is 24 inches -/
theorem cookie_sheet_perimeter_is_24 :
  cookie_sheet_perimeter 10 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_perimeter_is_24_l527_52785


namespace NUMINAMATH_CALUDE_exists_number_with_nine_nines_squared_l527_52770

theorem exists_number_with_nine_nines_squared : ∃ n : ℕ, 
  ∃ k : ℕ, n^2 = 999999999 * 10^k + m ∧ m < 10^k :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_nine_nines_squared_l527_52770


namespace NUMINAMATH_CALUDE_coffee_price_increase_percentage_l527_52775

def first_quarter_price : ℝ := 40
def fourth_quarter_price : ℝ := 60

theorem coffee_price_increase_percentage : 
  (fourth_quarter_price - first_quarter_price) / first_quarter_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_coffee_price_increase_percentage_l527_52775


namespace NUMINAMATH_CALUDE_expression_value_l527_52760

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 2) : 
  2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l527_52760


namespace NUMINAMATH_CALUDE_jose_remaining_caps_l527_52708

-- Define the initial number of bottle caps Jose has
def initial_caps : ℝ := 143.6

-- Define the number of bottle caps given to Rebecca
def given_to_rebecca : ℝ := 89.2

-- Define the number of bottle caps given to Michael
def given_to_michael : ℝ := 16.7

-- Theorem to prove the number of bottle caps Jose has left
theorem jose_remaining_caps :
  initial_caps - (given_to_rebecca + given_to_michael) = 37.7 := by
  sorry

end NUMINAMATH_CALUDE_jose_remaining_caps_l527_52708


namespace NUMINAMATH_CALUDE_tape_overlap_length_l527_52727

theorem tape_overlap_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (total_overlapped_length : ℝ) 
  (h1 : num_pieces = 4) 
  (h2 : piece_length = 250) 
  (h3 : total_overlapped_length = 925) :
  (num_pieces * piece_length - total_overlapped_length) / (num_pieces - 1) = 25 := by
sorry

end NUMINAMATH_CALUDE_tape_overlap_length_l527_52727


namespace NUMINAMATH_CALUDE_greg_read_more_than_brad_l527_52745

/-- Calculates the difference in pages read between Greg and Brad --/
def pages_difference : ℕ :=
  let greg_week1 := 7 * 18
  let greg_week2_3 := 14 * 22
  let greg_total := greg_week1 + greg_week2_3
  let brad_days1_5 := 5 * 26
  let brad_days6_17 := 12 * 20
  let brad_total := brad_days1_5 + brad_days6_17
  greg_total - brad_total

/-- The total number of pages both Greg and Brad need to read --/
def total_pages : ℕ := 800

/-- Theorem stating the difference in pages read between Greg and Brad --/
theorem greg_read_more_than_brad : pages_difference = 64 ∧ greg_total + brad_total = total_pages :=
  sorry

end NUMINAMATH_CALUDE_greg_read_more_than_brad_l527_52745


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l527_52735

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ
  vertex : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of the polygon formed by the intersection of a plane and a cube -/
def intersectionArea (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem stating the area of the intersection polygon -/
theorem intersection_area_theorem (c : Cube) (p q r : Point3D) : 
  c.edge_length = 30 →
  p.x = 10 ∧ p.y = 0 ∧ p.z = 0 →
  q.x = 30 ∧ q.y = 0 ∧ q.z = 10 →
  r.x = 30 ∧ r.y = 20 ∧ r.z = 30 →
  ∃ (plane : Plane), intersectionArea c plane = 450 := by
  sorry

#check intersection_area_theorem

end NUMINAMATH_CALUDE_intersection_area_theorem_l527_52735


namespace NUMINAMATH_CALUDE_angle_four_times_complement_l527_52765

theorem angle_four_times_complement (x : ℝ) : 
  (x = 4 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_four_times_complement_l527_52765


namespace NUMINAMATH_CALUDE_average_cost_calculation_l527_52746

/-- Calculates the average cost of products sold given the quantities and prices of different product types -/
theorem average_cost_calculation
  (iphone_quantity : ℕ) (iphone_price : ℕ)
  (ipad_quantity : ℕ) (ipad_price : ℕ)
  (appletv_quantity : ℕ) (appletv_price : ℕ)
  (h1 : iphone_quantity = 100)
  (h2 : iphone_price = 1000)
  (h3 : ipad_quantity = 20)
  (h4 : ipad_price = 900)
  (h5 : appletv_quantity = 80)
  (h6 : appletv_price = 200) :
  (iphone_quantity * iphone_price + ipad_quantity * ipad_price + appletv_quantity * appletv_price) /
  (iphone_quantity + ipad_quantity + appletv_quantity) = 670 :=
by sorry

end NUMINAMATH_CALUDE_average_cost_calculation_l527_52746


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l527_52707

/-- Given a real number a and a function f with the specified properties,
    prove that the tangent line to f at the origin has the equation 3x + y = 0 -/
theorem tangent_line_at_origin (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a-3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a-3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (∃ k : ℝ, ∀ x y, y = f x → (y - f 0) = k * (x - 0) → 3*x + y = 0) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l527_52707


namespace NUMINAMATH_CALUDE_inequalities_solution_sets_l527_52736

def inequality1 (x : ℝ) : Prop := x^2 + 3*x + 2 ≤ 0

def inequality2 (x : ℝ) : Prop := -3*x^2 + 2*x + 2 < 0

def solution_set1 : Set ℝ := {x | -2 ≤ x ∧ x ≤ -1}

def solution_set2 : Set ℝ := {x | x < (1 - Real.sqrt 7) / 3 ∨ x > (1 + Real.sqrt 7) / 3}

theorem inequalities_solution_sets :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) := by sorry

end NUMINAMATH_CALUDE_inequalities_solution_sets_l527_52736


namespace NUMINAMATH_CALUDE_school_population_after_new_students_l527_52718

theorem school_population_after_new_students (initial_avg_age initial_num_students new_students new_avg_age avg_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_avg_age = 32 →
  avg_decrease = 4 →
  (initial_avg_age * initial_num_students + new_avg_age * new_students) / (initial_num_students + new_students) = initial_avg_age - avg_decrease →
  initial_num_students + new_students = 480 := by
sorry

end NUMINAMATH_CALUDE_school_population_after_new_students_l527_52718


namespace NUMINAMATH_CALUDE_square_region_area_l527_52706

/-- A region consisting of equal squares inscribed in a rectangle -/
structure SquareRegion where
  num_squares : ℕ
  rect_width : ℝ
  rect_height : ℝ

/-- Calculate the area of a SquareRegion -/
def area (r : SquareRegion) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem square_region_area (r : SquareRegion) 
  (h1 : r.num_squares = 13)
  (h2 : r.rect_width = 28)
  (h3 : r.rect_height = 26) : 
  area r = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_region_area_l527_52706


namespace NUMINAMATH_CALUDE_euler_totient_equality_l527_52798

-- Define the Euler's totient function
def phi (n : ℕ) : ℕ := sorry

-- Define the property of being an odd number
def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Theorem statement
theorem euler_totient_equality (n : ℕ) (p : ℕ) (h_p : Prime p) :
  phi n = phi (n * p) ↔ p = 2 ∧ is_odd n :=
sorry

end NUMINAMATH_CALUDE_euler_totient_equality_l527_52798


namespace NUMINAMATH_CALUDE_crabapple_sequences_count_l527_52715

/-- The number of students in each class -/
def students_per_class : ℕ := 8

/-- The number of meetings per week for each class -/
def meetings_per_week : ℕ := 3

/-- The number of classes -/
def number_of_classes : ℕ := 2

/-- The total number of sequences of crabapple recipients for both classes in a week -/
def total_sequences : ℕ := (students_per_class ^ meetings_per_week) ^ number_of_classes

/-- Theorem stating that the total number of sequences is 262,144 -/
theorem crabapple_sequences_count : total_sequences = 262144 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_count_l527_52715


namespace NUMINAMATH_CALUDE_translation_left_2_units_l527_52774

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

/-- The problem statement -/
theorem translation_left_2_units :
  let P : Point2D := { x := -2, y := -1 }
  let A' : Point2D := translateLeft P 2
  A' = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_left_2_units_l527_52774


namespace NUMINAMATH_CALUDE_erdos_szekeres_l527_52714

theorem erdos_szekeres (n : ℕ) (seq : Fin (n^2 + 1) → ℝ) :
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), Monotone (seq ∘ subseq)) ∨
  (∃ (subseq : Fin (n + 1) → Fin (n^2 + 1)), StrictAnti (seq ∘ subseq)) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_l527_52714


namespace NUMINAMATH_CALUDE_election_votes_l527_52711

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (52 : ℚ) / 100 * total_votes - (48 : ℚ) / 100 * total_votes = 288) : 
  ((52 : ℚ) / 100 * total_votes : ℚ).floor = 3744 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l527_52711


namespace NUMINAMATH_CALUDE_max_value_trig_product_l527_52757

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (2*x) + Real.sin y + Real.sin (3*z)) * 
  (Real.cos (2*x) + Real.cos y + Real.cos (3*z)) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_product_l527_52757


namespace NUMINAMATH_CALUDE_geometric_sequence_relation_l527_52761

/-- Given a geometric sequence, prove the relation between its terms -/
theorem geometric_sequence_relation (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) → -- S_n formula
  (S 4 = 5 * S 2) →                                             -- Given condition
  (∃ q : ℝ, ∀ n, a n = (a 1) * q^(n-1)) →                       -- Geometric sequence definition
  (a 5)^2 / ((a 3) * (a 8)) = 1/2 ∨ 
  (a 5)^2 / ((a 3) * (a 8)) = -1/2 ∨ 
  (a 5)^2 / ((a 3) * (a 8)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_relation_l527_52761


namespace NUMINAMATH_CALUDE_cylindrical_tin_height_l527_52794

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylindrical_tin_height (diameter : ℝ) (volume : ℝ) (h_diameter : diameter = 8) (h_volume : volume = 80) :
  (volume / (π * (diameter / 2)^2)) = 80 / (π * 4^2) :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_tin_height_l527_52794


namespace NUMINAMATH_CALUDE_sequence_sum_l527_52797

theorem sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n : ℕ, S n = n^3) → 
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 1 = S 1 →
  a 5 + a 6 = 152 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l527_52797
