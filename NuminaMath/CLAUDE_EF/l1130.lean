import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_90_l1130_113094

/-- Represents a kilometer marker with two digits -/
structure KilometerMarker where
  tens : Nat
  ones : Nat
  is_valid : tens < 10 ∧ ones < 10

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  is_valid : hours < 24 ∧ minutes < 60

/-- Calculates the time difference in hours between two times -/
noncomputable def timeDiff (t1 t2 : Time) : ℝ :=
  (t2.hours - t1.hours : ℝ) + (t2.minutes - t1.minutes : ℝ) / 60

/-- Calculates the distance between two kilometer markers -/
def markerDiff (m1 m2 : KilometerMarker) : Int :=
  (m2.tens * 10 + m2.ones) - (m1.tens * 10 + m1.ones)

theorem car_speed_is_90 
  (x y : Nat)
  (hx : x < 10)
  (hy : y < 10)
  (hxy : y = 8 * x)
  (m1 : KilometerMarker)
  (m2 : KilometerMarker)
  (m3 : KilometerMarker)
  (t1 : Time)
  (t2 : Time)
  (t3 : Time)
  (h1 : m1.tens = x ∧ m1.ones = y)
  (h2 : m2.tens = y ∧ m2.ones = x)
  (h3 : m3.tens = x ∧ m3.ones = y)
  (ht1 : t1.hours = 12 ∧ t1.minutes = 0)
  (ht2 : t2.hours = 12 ∧ t2.minutes = 42)
  (ht3 : t3.hours = 13 ∧ t3.minutes = 0)
  : (markerDiff m1 m3 : ℝ) / (timeDiff t1 t3) = 90 := by
  sorry

#check car_speed_is_90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_is_90_l1130_113094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_equation_l1130_113097

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a hyperbola with the given properties, its eccentricity satisfies the equation -/
theorem hyperbola_eccentricity_equation (h : Hyperbola) 
  (M N : Point) 
  (h_M_asymptote : M.y = h.b / h.a * M.x) 
  (h_N_hyperbola : N.x^2 / h.a^2 - N.y^2 / h.b^2 = 1)
  (h_first_quadrant : M.x > 0 ∧ M.y > 0 ∧ N.x > 0 ∧ N.y > 0)
  (h_parallel : (M.y - (-h.a * eccentricity h)) / (M.x - (-h.a)) = N.y / N.x) :
  let e := eccentricity h
  (e^2 + 2*e - 2/e) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_equation_l1130_113097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_equals_one_l1130_113043

/-- The function f(x) that takes an extreme value at x = 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) - x^2 - x

/-- Theorem stating that if f(x) takes an extreme value at x = 0, then a = 1 -/
theorem extreme_value_implies_a_equals_one (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x| ∧ |x| < ε → f a 0 ≥ f a x) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_equals_one_l1130_113043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1130_113044

def digits : List Nat := [2, 0, 4, 1, 5, 8]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧
  (Nat.digits 10 n).toFinset = digits.toFinset ∧
  (Nat.digits 10 n).length = digits.length

def largest_number : Nat := 854210
def smallest_number : Nat := 102458

theorem sum_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number + smallest_number = 956668 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_largest_and_smallest_l1130_113044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_l1130_113003

/-- A weight with a specific mass -/
structure Weight where
  mass : ℕ

/-- A set of weights satisfying the given conditions -/
structure WeightSet where
  weights : Finset Weight
  different_masses : Finset.card (Finset.image Weight.mass weights) = 5
  equal_sums : ∀ w1 w2, w1 ∈ weights → w2 ∈ weights →
    ∃ w3 w4, w3 ∈ weights ∧ w4 ∈ weights ∧ w1.mass + w2.mass = w3.mass + w4.mass

/-- The theorem stating the minimum number of weights -/
theorem min_weights (ws : WeightSet) : Finset.card ws.weights ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weights_l1130_113003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1130_113024

-- Define the quadratic function f(x) = ax^2 + bx
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Define the conditions
def condition1 (a b : ℝ) : Prop := ∀ x, f a b x = f a b (-x - 2)
def condition2 (a b : ℝ) : Prop := ∃ x, f a b x = x ∧ ∀ y, y ≠ x → f a b y ≠ y

-- Define the range condition
def range_condition (x t : ℝ) : Prop := Real.pi^(f (1/2) 1 x) > (1/Real.pi)^(2 - t*x)

theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  condition1 a b → condition2 a b →
  (∃ x, f a b x = (1/2) * x^2 + x) ∧
  (∀ x, (∀ t, |t| ≤ 2 → range_condition x t) ↔ 
    (x < -3 - Real.sqrt 5 ∨ x > -3 + Real.sqrt 5)) :=
by
  sorry

#check quadratic_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1130_113024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l1130_113093

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- State the theorem
theorem inverse_f_undefined_at_one :
  ∀ x : ℝ, x ≠ 6 → (∃ y : ℝ, f y = x) → x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_undefined_at_one_l1130_113093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_quartic_roots_square_l1130_113005

/-- A quartic polynomial with integer coefficients -/
structure QuarticPolynomial where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- Helper function to determine if four complex numbers form a square -/
def is_square_in_complex_plane (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ (center : ℂ) (side : ℝ),
    (Complex.abs (z1 - center) = side) ∧
    (Complex.abs (z2 - center) = side) ∧
    (Complex.abs (z3 - center) = side) ∧
    (Complex.abs (z4 - center) = side) ∧
    (Complex.abs ((z1 - center) - (z2 - center)) = Complex.abs ((z2 - center) - (z3 - center))) ∧
    (Complex.abs ((z2 - center) - (z3 - center)) = Complex.abs ((z3 - center) - (z4 - center))) ∧
    (Complex.abs ((z3 - center) - (z4 - center)) = Complex.abs ((z4 - center) - (z1 - center)))

/-- The roots of a quartic polynomial form a square in the complex plane -/
def roots_form_square (poly : QuarticPolynomial) : Prop :=
  ∃ (z1 z2 z3 z4 : ℂ),
    (z1^4 + poly.p * z1^3 + poly.q * z1^2 + poly.r * z1 + poly.s = 0) ∧
    (z2^4 + poly.p * z2^3 + poly.q * z2^2 + poly.r * z2 + poly.s = 0) ∧
    (z3^4 + poly.p * z3^3 + poly.q * z3^2 + poly.r * z3 + poly.s = 0) ∧
    (z4^4 + poly.p * z4^3 + poly.q * z4^2 + poly.r * z4 + poly.s = 0) ∧
    (is_square_in_complex_plane z1 z2 z3 z4)

/-- The area of a square formed by four complex numbers -/
noncomputable def square_area (z1 z2 z3 z4 : ℂ) : ℝ :=
  Complex.abs ((z1 - z3) * (z2 - z4))

/-- The statement to be proved -/
theorem min_area_of_quartic_roots_square :
  ∀ (poly : QuarticPolynomial),
    roots_form_square poly →
    ∃ (z1 z2 z3 z4 : ℂ),
      (z1^4 + poly.p * z1^3 + poly.q * z1^2 + poly.r * z1 + poly.s = 0) ∧
      (z2^4 + poly.p * z2^3 + poly.q * z2^2 + poly.r * z2 + poly.s = 0) ∧
      (z3^4 + poly.p * z3^3 + poly.q * z3^2 + poly.r * z3 + poly.s = 0) ∧
      (z4^4 + poly.p * z4^3 + poly.q * z4^2 + poly.r * z4 + poly.s = 0) ∧
      (is_square_in_complex_plane z1 z2 z3 z4) ∧
      (square_area z1 z2 z3 z4 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_quartic_roots_square_l1130_113005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_divisibility_l1130_113082

theorem sin_sum_divisibility (n : ℕ) : 
  ∃ k : ℤ, (2 * Real.sin (π / 7 : ℝ))^(2*n) + 
            (2 * Real.sin ((2*π) / 7 : ℝ))^(2*n) + 
            (2 * Real.sin ((3*π) / 7 : ℝ))^(2*n) = 
            k * (7 : ℝ)^(Int.floor (n / 3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_divisibility_l1130_113082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_and_B_l1130_113081

-- Define the variables
variable (p q r s A B : ℝ)

-- Define the conditions
def average_pqrs (p q r s : ℝ) : Prop := (p + q + r + s) / 4 = 5
def average_pqrsA (p q r s A : ℝ) : Prop := (p + q + r + s + A) / 5 = 8
def perpendicular_lines (A B : ℝ) : Prop := (3 * (-1/B)) * (3/2) = -1

-- State the theorem
theorem find_A_and_B 
  (h1 : average_pqrs p q r s)
  (h2 : average_pqrsA p q r s A)
  (h3 : perpendicular_lines A B) :
  A = 20 ∧ B = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_A_and_B_l1130_113081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_max_triangle_area_l1130_113021

noncomputable section

-- Define the curve C₁
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

-- Define the relationship between O, M, and P
def OM_OP_relation (O M P : ℝ × ℝ) : Prop :=
  Real.sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2) *
  Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 16

-- Define the curve C₂
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4 ∧ x ≠ 0

-- Define point A
def A : ℝ × ℝ := (1, Real.sqrt 3)

-- State the theorems to be proved
theorem trajectory_equation (O M P : ℝ × ℝ) (h₁ : C₁ (Real.sqrt (M.1^2 + M.2^2)) (Real.arctan (M.2 / M.1)))
  (h₂ : OM_OP_relation O M P) :
  C₂ P.1 P.2 := by sorry

theorem max_triangle_area (B : ℝ × ℝ) (h : C₂ B.1 B.2) :
  (∀ B' : ℝ × ℝ, C₂ B'.1 B'.2 →
    (1/2 * Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2) * Real.sin (Real.arctan (B.2 / B.1) - Real.arctan (A.2 / A.1)) ≥
     1/2 * Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B'.1^2 + B'.2^2) * Real.sin (Real.arctan (B'.2 / B'.1) - Real.arctan (A.2 / A.1)))) ∧
  (1/2 * Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2) * Real.sin (Real.arctan (B.2 / B.1) - Real.arctan (A.2 / A.1)) = 2 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_max_triangle_area_l1130_113021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l1130_113025

/-- The probability of Terry and Mary picking the same color combination of candies -/
theorem same_color_combination_probability (red_candies blue_candies : ℕ) 
  (h1 : red_candies = 12) (h2 : blue_candies = 8) : ℚ := by
  -- Define the total number of candies
  let total_candies : ℕ := red_candies + blue_candies
  
  -- Define the probability
  let prob : ℚ := 77 / 4845
  
  -- Proof goes here
  sorry

#check same_color_combination_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_combination_probability_l1130_113025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_point_l1130_113037

theorem cos_double_angle_point (α : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_point_l1130_113037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1130_113074

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- A point is on the z-axis if its x and y coordinates are 0 -/
def onZAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0

theorem point_P_coordinates : 
  let A : Point3D := ⟨1, 2, 3⟩
  let B : Point3D := ⟨2, 2, 4⟩
  ∀ P : Point3D, onZAxis P → distance P A = distance P B → P = ⟨0, 0, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_coordinates_l1130_113074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1130_113046

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3 ∧ t.a = 6

-- Part I
theorem part_one (t : Triangle) (h : isValidTriangle t) (hc : t.c = 14) :
  Real.sin t.A = 3 * Real.sqrt 3 / 14 := by
  sorry

-- Part II
theorem part_two (t : Triangle) (h : isValidTriangle t) (harea : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3) :
  t.c = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1130_113046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pheromone_effect_l1130_113039

/-- Represents the population of a pest species -/
structure PestPopulation where
  maleCount : ℕ
  femaleCount : ℕ

/-- Represents the use of artificial sex pheromones -/
def useArtificialPheromones (pop : PestPopulation) : PestPopulation :=
  { maleCount := pop.maleCount - 1, femaleCount := pop.femaleCount }

/-- Represents the normal sex ratio of the population -/
def normalSexRatio (pop : PestPopulation) : Bool :=
  pop.maleCount = pop.femaleCount

/-- Represents the population density -/
def populationDensity (pop : PestPopulation) : ℕ :=
  pop.maleCount + pop.femaleCount

/-- Theorem stating that using artificial pheromones leads to decreased population density -/
theorem pheromone_effect (pop : PestPopulation) :
  normalSexRatio pop →
  ¬normalSexRatio (useArtificialPheromones pop) →
  populationDensity (useArtificialPheromones pop) < populationDensity pop :=
by
  intro h1 h2
  simp [populationDensity, useArtificialPheromones]
  sorry

#check pheromone_effect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pheromone_effect_l1130_113039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_inventory_l1130_113073

theorem bookstore_inventory (total_books : ℕ) 
  (hist_fic_percent science_fic_percent bio_percent mystery_percent : ℚ)
  (hist_fic_new_percent science_fic_new_percent bio_new_percent mystery_new_percent : ℚ)
  (discount_percent : ℚ)
  (h_total : total_books = 2000)
  (h_hist_fic : hist_fic_percent = 40 / 100)
  (h_science_fic : science_fic_percent = 25 / 100)
  (h_bio : bio_percent = 15 / 100)
  (h_mystery : mystery_percent = 20 / 100)
  (h_hist_fic_new : hist_fic_new_percent = 45 / 100)
  (h_science_fic_new : science_fic_new_percent = 30 / 100)
  (h_bio_new : bio_new_percent = 50 / 100)
  (h_mystery_new : mystery_new_percent = 35 / 100)
  (h_discount : discount_percent = 10 / 100)
  (h_sum_percent : hist_fic_percent + science_fic_percent + bio_percent + mystery_percent = 1) :
  let hist_fic_books := (hist_fic_percent * total_books).floor
  let science_fic_books := (science_fic_percent * total_books).floor
  let bio_books := (bio_percent * total_books).floor
  let mystery_books := (mystery_percent * total_books).floor
  let hist_fic_new := (hist_fic_new_percent * hist_fic_books).floor
  let science_fic_new := (science_fic_new_percent * science_fic_books).floor
  let bio_new := (bio_new_percent * bio_books).floor
  let mystery_new := (mystery_new_percent * mystery_books).floor
  let total_new := hist_fic_new + science_fic_new + bio_new + mystery_new
  (hist_fic_new : ℚ) / (total_new : ℚ) = 9 / 20 ∧ 
  (discount_percent * hist_fic_new).floor = 36 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_inventory_l1130_113073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_k_closing_price_l1130_113014

/-- Calculates the closing price of a stock given its opening price and percentage increase -/
noncomputable def closing_price (opening_price : ℝ) (percent_increase : ℝ) : ℝ :=
  opening_price * (1 + percent_increase / 100)

/-- Theorem stating that given an opening price of $8 and a 12.5% increase, the closing price is $9 -/
theorem stock_k_closing_price :
  closing_price 8 12.5 = 9 := by
  -- Unfold the definition of closing_price
  unfold closing_price
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_k_closing_price_l1130_113014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l1130_113067

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

def max_divisors (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sup (fun i => (divisors (i + a)).card)

theorem max_divisors_1_to_20 :
  max_divisors 1 20 = 6 ∧
  (divisors 12).card = 6 ∧
  (divisors 18).card = 6 ∧
  (divisors 20).card = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (divisors n).card ≤ 6 :=
by
  sorry

#eval max_divisors 1 20
#eval (divisors 12).card
#eval (divisors 18).card
#eval (divisors 20).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l1130_113067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_minus_alpha_abs_l1130_113066

/-- A polynomial with real coefficients satisfying specific conditions -/
noncomputable def p : ℝ → ℝ → ℝ → ℝ := sorry

/-- The function f in the homogeneity condition -/
noncomputable def f : ℝ → ℝ → ℝ := sorry

/-- Complex numbers α, β, γ that are roots of p -/
noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := sorry
noncomputable def γ : ℂ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem gamma_minus_alpha_abs (h1 : ∀ (x y z t : ℝ), p (t * x) (t * y) (t * z) = t^2 * f (y - x) (z - x))
                               (h2 : p 1 0 0 = 4)
                               (h3 : p 0 1 0 = 5)
                               (h4 : p 0 0 1 = 6)
                               (h5 : p α.re β.re γ.re = 0)
                               (h6 : Complex.abs (β - α) = 10) :
  Complex.abs (γ - α) = (5 * Real.sqrt 30) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gamma_minus_alpha_abs_l1130_113066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_iterations_l1130_113020

-- Define the function f(x)
noncomputable def f (x : ℝ) := 2^x + 3*x - 7

-- Define the precision
def precision : ℝ := 0.05

-- Define the initial interval
def a : ℝ := 1.25
def b : ℝ := 1.5

-- State the theorem
theorem bisection_method_iterations :
  f a < 0 → f b > 0 → ∃ n : ℕ, 
    ((b - a) / 2^n < precision) ∧ 
    (∀ m : ℕ, m < n → (b - a) / 2^m ≥ precision) ∧
    n = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_iterations_l1130_113020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1130_113058

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- Definition of the foci of an ellipse -/
noncomputable def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := by
  let c := a * eccentricity a b
  exact ((- c, 0), (c, 0))

/-- Definition of the maximum area of triangle F₁AB -/
noncomputable def max_triangle_area (a b : ℝ) : ℝ := by
  let c := a * eccentricity a b
  exact 3 * c^2

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = 1/2) (h4 : max_triangle_area a b = 6) :
  Ellipse 8 6 = Ellipse a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1130_113058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_zeros_l1130_113027

noncomputable def f (w : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (w * x - Real.pi / 6) * Real.cos (w * x) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem f_period_and_g_zeros (w : ℝ) (h : w > 0) :
  (∀ x, f w (x + Real.pi) = f w x) ∧ 
  (∀ x, f w x = f 1 x) ∧
  g (-Real.pi / 6) = 0 ∧ g (5 * Real.pi / 6) = 0 ∧
  (∀ x, -Real.pi ≤ x ∧ x ≤ Real.pi ∧ g x = 0 → x = -Real.pi / 6 ∨ x = 5 * Real.pi / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_zeros_l1130_113027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_third_term_is_45_l1130_113083

/-- A sequence where each term is 2 more than the previous term -/
def gameSequence : ℕ → ℕ
| 0 => 1  -- First term is 1
| n + 1 => gameSequence n + 2  -- Each subsequent term is 2 more than the previous

/-- Theorem stating that the 23rd term of the sequence is 45 -/
theorem twenty_third_term_is_45 : gameSequence 22 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_third_term_is_45_l1130_113083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_coincidence_l1130_113069

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
def Point : Type := ℝ × ℝ

-- Define a quadrilateral
def Quadrilateral : Type := Point → Point → Point → Point → Prop

-- Define the concept of a quadrilateral being inscribed in a circle
def inscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Define the concept of a chord of a circle
def chord (c : Circle) (p1 p2 : Point) : Prop := sorry

-- Define the concept of intersection between a line and a chord
def intersect (p1 p2 : Point) (ch : Point × Point) : Point := sorry

-- Main theorem
theorem quadrilateral_intersection_coincidence 
  (c : Circle) 
  (q1 q2 : Quadrilateral) 
  (ab : Point × Point) 
  (p q r s p' q' r' s' : Point) :
  inscribed q1 c →
  inscribed q2 c →
  chord c ab.1 ab.2 →
  (∃ (k l m n k' l' m' n' : Point), 
    q1 k l m n ∧ 
    q2 k' l' m' n' ∧
    p = intersect k l ab ∧
    q = intersect l m ab ∧
    r = intersect m n ab ∧
    s = intersect n k ab ∧
    p' = intersect k' l' ab ∧
    q' = intersect l' m' ab ∧
    r' = intersect m' n' ab ∧
    s' = intersect n' k' ab) →
  (p = p' ∧ q = q' ∧ r = r') →
  s = s' := by
  sorry

#check quadrilateral_intersection_coincidence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_intersection_coincidence_l1130_113069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_is_one_l1130_113063

-- Define the lines and circle
def line_l (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ 4 * x + a * y - 5 = 0
def line_l' : ℝ → ℝ → Prop := λ x y ↦ x - 2 * y = 0
def circle_C : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 = 2

-- Define points
def point_M : ℝ × ℝ := (-1, -1)
def point_N : ℝ × ℝ := (2, 0)

-- Define the theorem
theorem slope_of_PQ_is_one 
  (h1 : ∃ a : ℝ, (∀ x y, line_l a x y ↔ 4 * x + 2 * y - 5 = 0) ∧ 
    (∀ x y, line_l a x y → line_l' x y → x = y))
  (h2 : ∃ c : ℝ × ℝ, (c.1 - 2)^2 + (c.2 - 1)^2 = 2^2 + 1^2 ∧ 
    (∀ x y, circle_C x y ↔ (x - c.1)^2 + (y - c.2)^2 = (x - 2)^2 + (y - 1)^2))
  (h3 : circle_C point_M.1 point_M.2)
  (h4 : ∃ P Q : ℝ × ℝ, circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    (P.2 - point_M.2) * (Q.1 - point_M.1) = -(P.1 - point_M.1) * (Q.2 - point_M.2)) :
  ∃ P Q : ℝ × ℝ, circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    (Q.2 - P.2) / (Q.1 - P.1) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_PQ_is_one_l1130_113063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1130_113075

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem equation_solution (x : ℝ) : 
  (floor (2 * x) + floor (3 * x) + floor (7 * x) = 2008) ↔ 
  (∃ θ : ℝ, x = 167 + θ ∧ 3/7 ≤ θ ∧ θ < 1/2) := by
  sorry

#check equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1130_113075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_rook_placement_l1130_113052

/-- Represents a 10x10 chessboard with rooks --/
structure Chessboard where
  rooks : Finset (Fin 10 × Fin 10)
  white_count : ℕ
  black_count : ℕ

/-- Predicate to check if a square is attacked by any rook --/
def is_attacked (board : Chessboard) (square : Fin 10 × Fin 10) : Prop :=
  ∃ (rook : Fin 10 × Fin 10), rook ∈ board.rooks ∧ 
    (rook.1 = square.1 ∨ rook.2 = square.2)

/-- The main theorem --/
theorem additional_rook_placement (board : Chessboard) 
  (h1 : board.white_count = board.black_count)
  (h2 : ∀ (r1 r2 : Fin 10 × Fin 10), r1 ∈ board.rooks → r2 ∈ board.rooks → r1 ≠ r2 → r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2)
  : ∃ (square : Fin 10 × Fin 10), ¬is_attacked board square := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_rook_placement_l1130_113052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_max_min_difference_l1130_113042

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then (x - 1)^2 else (x + 1)^2

theorem even_function_max_min_difference (n m : ℝ) :
  (∀ x, f (-x) = f x) →
  (∀ x ∈ Set.Icc (-2) (-1/2), n ≤ f x ∧ f x ≤ m) →
  m - n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_max_min_difference_l1130_113042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1130_113061

/-- The area of a sector of a circle with diameter 10 meters and central angle 90 degrees -/
noncomputable def sector_area (π : ℝ) : ℝ := by
  -- Define the diameter and central angle
  let diameter : ℝ := 10
  let central_angle : ℝ := 90

  -- Define the radius (half the diameter)
  let radius : ℝ := diameter / 2

  -- Define the sector area formula
  let area := (central_angle / 360) * π * radius^2

  -- Calculate the result
  exact area

/-- Theorem: The area of the sector is equal to (25/4) * π -/
theorem sector_area_theorem : sector_area Real.pi = (25 / 4) * Real.pi := by
  -- Unfold the definition of sector_area
  unfold sector_area
  
  -- Simplify the expression
  simp [Real.pi]
  
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval with noncomputable functions, so we'll just state the theorem
#check sector_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_theorem_l1130_113061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_values_l1130_113050

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sqrt 3 * a * (Real.cos x)^2 + (Real.sqrt 3 / 2) * a + b

theorem function_extrema_values (a b : ℝ) (h_a : a > 0) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≥ -2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f a b x ≤ Real.sqrt 3) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f a b x₁ = -2) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f a b x₂ = Real.sqrt 3) →
  a = 2 ∧ b = -2 + Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_values_l1130_113050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sea_world_savings_l1130_113023

/-- Calculates the additional amount Sally needs to save for her trip to Sea World -/
theorem sea_world_savings (
  current_savings parking_cost entrance_fee meal_pass_cost
  distance_to_sea_world car_efficiency gas_price : ℕ
) (h1 : current_savings = 28)
  (h2 : parking_cost = 10)
  (h3 : entrance_fee = 55)
  (h4 : meal_pass_cost = 25)
  (h5 : distance_to_sea_world = 165)
  (h6 : car_efficiency = 30)
  (h7 : gas_price = 3) :
  (2 * distance_to_sea_world / car_efficiency * gas_price +
   parking_cost + entrance_fee + meal_pass_cost) - current_savings = 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sea_world_savings_l1130_113023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_area_enclosed_l1130_113000

noncomputable section

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

-- Define the tangent line slope
noncomputable def k : ℝ := (deriv f) 0

-- Define the area function
noncomputable def area_enclosed (m : ℝ) : ℝ :=
  ∫ x in (0:ℝ)..(2:ℝ), m * x - x^2

theorem tangent_area_enclosed :
  f 0 = 1 → area_enclosed k = 4/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_area_enclosed_l1130_113000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l1130_113026

theorem cube_root_problem (x : ℝ) (h : (2 * x - 1).sqrt = 7 ∨ (2 * x - 1).sqrt = -7) :
  (2 * x - 23) ^ (1/3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_problem_l1130_113026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1130_113070

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties : 
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧ 
  (∀ x, f (-Real.pi / 6 + x) = f (-Real.pi / 6 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1130_113070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1130_113062

/-- A journey with a change in speed -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  initialSpeed : ℝ
  initialTimeFraction : ℝ
  initialDistanceFraction : ℝ

/-- The speed required for the remaining part of the journey -/
noncomputable def requiredSpeed (j : Journey) : ℝ :=
  (j.totalDistance * (1 - j.initialDistanceFraction)) / (j.totalTime * (1 - j.initialTimeFraction))

/-- Theorem stating the required speed for the given journey conditions -/
theorem journey_speed_theorem (j : Journey) 
    (h1 : j.initialSpeed = 30)
    (h2 : j.initialTimeFraction = 1/3)
    (h3 : j.initialDistanceFraction = 2/3) :
    requiredSpeed j = 15 := by
  sorry

#check journey_speed_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1130_113062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_rate_calculation_l1130_113096

-- Define the reaction rate function
noncomputable def reaction_rate (c₁ c₂ t₁ t₂ : ℝ) : ℝ :=
  (c₁ - c₂) / (t₂ - t₁)

-- Define the theorem
theorem reaction_rate_calculation :
  let c₁ : ℝ := 3.5
  let c₂ : ℝ := 1.5
  let t₁ : ℝ := 0
  let t₂ : ℝ := 15
  abs (reaction_rate c₁ c₂ t₁ t₂ - 0.133) < 0.001 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_rate_calculation_l1130_113096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_equals_f_or_neg_f_abs_f_correct_l1130_113092

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2*(x - 2)
  else 0  -- undefined outside the given domain

-- State the theorem
theorem abs_f_equals_f_or_neg_f (x : ℝ) :
  (f x ≥ 0 → |f x| = f x) ∧ (f x < 0 → |f x| = -f x) := by
  sorry

-- Define the absolute value of f
noncomputable def abs_f (x : ℝ) : ℝ := |f x|

-- State the theorem that abs_f is the correct representation of |f(x)|
theorem abs_f_correct (x : ℝ) : abs_f x = max (f x) (-f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_f_equals_f_or_neg_f_abs_f_correct_l1130_113092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_second_quadrant_iff_a_gt_one_third_l1130_113088

/-- The complex number Z as a function of real number a -/
noncomputable def Z (a : ℝ) : ℂ := (a - 1 + 2*a*Complex.I) / (1 - Complex.I)

/-- Predicate to check if a complex number is in the second quadrant -/
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- Theorem stating the range of a for Z to be in the second quadrant -/
theorem Z_in_second_quadrant_iff_a_gt_one_third :
  ∀ a : ℝ, is_in_second_quadrant (Z a) ↔ a > 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_in_second_quadrant_iff_a_gt_one_third_l1130_113088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1130_113038

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - 1

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + 1) - 4

-- Theorem statement
theorem a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 > 1) 
  (h4 : ∀ x, x < 0 → g a x ≤ 0) : 2 < a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1130_113038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_greater_in_second_quadrant_l1130_113009

theorem cos_greater_in_second_quadrant (α β : Real) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : π/2 < β ∧ β < π) 
  (h3 : Real.sin α > Real.sin β) : 
  Real.cos α > Real.cos β := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_greater_in_second_quadrant_l1130_113009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rosie_pies_l1130_113065

/-- Represents the number of pies that can be made given the available ingredients -/
def max_pies (apples sugar : ℚ) : ℕ :=
  Int.toNat (min (⌊apples / (9/2)⌋) (⌊sugar / 2⌋))

/-- Theorem stating that Rosie can make 6 pies with 27 apples and 20 cups of sugar -/
theorem rosie_pies : max_pies 27 20 = 6 := by
  sorry

#eval max_pies 27 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rosie_pies_l1130_113065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_max_area_achievable_l1130_113091

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The circumradius of the triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := Real.sqrt 3

/-- The equation given in the problem -/
def equation (t : Triangle) : Prop :=
  2 * Real.sqrt 3 * (Real.sin t.A ^ 2 - Real.sin t.C ^ 2) = (t.a - t.b) * Real.sin t.B

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1 / 2) * t.a * t.b * Real.sin t.C

/-- The theorem to be proved -/
theorem max_area_of_triangle (t : Triangle) 
  (h1 : circumradius t = Real.sqrt 3) 
  (h2 : equation t) : 
  area t ≤ (9 * Real.sqrt 3) / 4 := by
  sorry

/-- The maximum area is achievable -/
theorem max_area_achievable : 
  ∃ t : Triangle, circumradius t = Real.sqrt 3 ∧ equation t ∧ area t = (9 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_triangle_max_area_achievable_l1130_113091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_values_surface_area_ratio_and_base_ratio_l1130_113012

/-- Represents a truncated cone with an inscribed hemisphere -/
structure TruncatedConeWithHemisphere where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  m : ℝ  -- Height of the truncated cone (equal to the radius of the hemisphere)
  h_positive : 0 < r ∧ r < R ∧ 0 < m
  h_sphere_touches : m^2 = 2 * R * r - r^2
  h_volume_ratio : (2 * Real.pi / 3) * m^3 = (6 / 7) * (Real.pi * m / 3) * (R^2 + R * r + r^2)

/-- The ratio of the lateral surface area of the truncated cone to the surface area of the hemisphere -/
noncomputable def surfaceAreaRatio (tc : TruncatedConeWithHemisphere) : ℝ :=
  let l := Real.sqrt (tc.R^2 + tc.m^2)
  (Real.pi * (tc.R + tc.r) * l) / (2 * Real.pi * tc.m^2)

/-- Theorem stating the possible values of the surface area ratio -/
theorem surface_area_ratio_values (tc : TruncatedConeWithHemisphere) :
  surfaceAreaRatio tc = 1 ∨ surfaceAreaRatio tc = Real.sqrt 7 / 6 := by
  sorry

/-- Theorem relating the surface area ratio to the ratio of base radii -/
theorem surface_area_ratio_and_base_ratio (tc : TruncatedConeWithHemisphere) :
  (tc.R / tc.r = 2 → surfaceAreaRatio tc = 1) ∧
  (tc.R / tc.r = 5 / 3 → surfaceAreaRatio tc = Real.sqrt 7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_values_surface_area_ratio_and_base_ratio_l1130_113012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_per_hour_l1130_113013

/-- Calculates a reporter's earnings per hour based on given conditions. -/
theorem reporter_earnings_per_hour
  (word_rate : ℝ)
  (article_rate : ℝ)
  (num_articles : ℕ)
  (total_hours : ℕ)
  (words_per_minute : ℕ)
  (h1 : word_rate = 0.1)
  (h2 : article_rate = 60)
  (h3 : num_articles = 3)
  (h4 : total_hours = 4)
  (h5 : words_per_minute = 10) :
  let total_minutes : ℕ := total_hours * 60
  let total_words : ℕ := total_minutes * words_per_minute
  let word_earnings : ℝ := (total_words : ℝ) * word_rate
  let article_earnings : ℝ := (num_articles : ℝ) * article_rate
  let total_earnings : ℝ := word_earnings + article_earnings
  let earnings_per_hour : ℝ := total_earnings / (total_hours : ℝ)
  earnings_per_hour = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reporter_earnings_per_hour_l1130_113013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_irrational_l1130_113031

-- Define the numbers
def a : ℚ := 22 / 7
noncomputable def b : ℝ := Real.sqrt 7
def c : ℚ := 3.1415926
def d : ℚ := 3  -- ∛27 = 3

-- State the theorem
theorem sqrt_seven_irrational :
  ¬ (∃ (q : ℚ), (q : ℝ) ^ 2 = 7) ∧ 
  (∃ (q : ℚ), (q : ℝ) = a) ∧
  (∃ (q : ℚ), (q : ℝ) = c) ∧
  (∃ (q : ℚ), (q : ℝ) = d) := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_irrational_l1130_113031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_calculation_l1130_113032

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.width * w.height

theorem wall_height_calculation
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (brickCount : ℕ)
  (h : ℝ → ℝ → ℝ → ℝ → ℕ → ℝ) :
  brick.length = 50 →
  brick.width = 11.25 →
  brick.height = 6 →
  wall.length = 800 →
  wall.width = 22.5 →
  brickCount = 3200 →
  h brick.length brick.width brick.height wall.width brickCount = 600 →
  wall.height = 600 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_height_calculation_l1130_113032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_properties_l1130_113053

/-- Represents a triangular pyramid with vertex V and base ABC -/
structure TriangularPyramid where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  dihedral_angle : ℝ

/-- Calculate the lateral surface area of the triangular pyramid -/
noncomputable def lateral_surface_area (p : TriangularPyramid) : ℝ :=
  (p.AB * p.AB / 2) + (p.AC * p.AC / 2) + (p.BC * p.BC / 2)

/-- Calculate the height of the triangular pyramid -/
noncomputable def pyramid_height (p : TriangularPyramid) : ℝ :=
  Real.sqrt (p.AB^2 - (p.BC / 2)^2)

/-- Theorem stating the lateral surface area and height of the specific pyramid -/
theorem specific_pyramid_properties :
  let p : TriangularPyramid := {
    AB := 10,
    AC := 10,
    BC := 12,
    dihedral_angle := 45
  }
  lateral_surface_area p = 172 ∧ pyramid_height p = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_properties_l1130_113053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocky_knockout_percentage_l1130_113010

theorem rocky_knockout_percentage 
  (total_fights : ℕ) 
  (first_round_knockout_percentage : ℚ) 
  (first_round_knockouts : ℕ) 
  (h1 : total_fights = 190)
  (h2 : first_round_knockout_percentage = 1/5)
  (h3 : first_round_knockouts = 19)
  : (first_round_knockouts : ℚ) / (first_round_knockout_percentage * (total_fights : ℚ)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocky_knockout_percentage_l1130_113010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_four_fifteenths_l1130_113086

/-- Represents the fraction of area shaded at each level of subdivision -/
def shadedFractionAtLevel (n : ℕ) : ℚ :=
  (1 / 4) * (1 / 16) ^ n

/-- The total fraction of the square that is shaded after infinite subdivisions -/
noncomputable def totalShadedFraction : ℚ :=
  ∑' n, shadedFractionAtLevel n

theorem shaded_fraction_is_four_fifteenths :
  totalShadedFraction = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_fraction_is_four_fifteenths_l1130_113086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_two_equals_fourteen_l1130_113072

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

-- State the theorem
theorem f_of_g_of_two_equals_fourteen : f (g 2) = 14 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_two_equals_fourteen_l1130_113072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_theorem_l1130_113048

/-- Number of common tangents for two circles -/
noncomputable def num_common_tangents (r R d : ℝ) : ℕ :=
  if d < R - r then 0
  else if d = R - r then 1
  else if R - r < d ∧ d < R + r then 2
  else if d = R + r then 3
  else 4

/-- Theorem stating the number of common tangents for two circles -/
theorem common_tangents_theorem (r R d : ℝ) (h : r < R) :
  num_common_tangents r R d =
    if d < R - r then 0
    else if d = R - r then 1
    else if R - r < d ∧ d < R + r then 2
    else if d = R + r then 3
    else 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangents_theorem_l1130_113048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_heron_l1130_113098

/-- Heron's formula for the area of a triangle -/
noncomputable def heronFormula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of a triangle with sides 2.5, 4.7, and 5.3 is approximately 6.64 -/
theorem triangle_area_heron : 
  let a : ℝ := 2.5
  let b : ℝ := 4.7
  let c : ℝ := 5.3
  ∃ ε > 0, |heronFormula a b c - 6.64| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_heron_l1130_113098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_axis_point_on_circle_l1130_113078

/-- Given a circle C with center at the origin and radius 16, if the point (-16, 0) lies on C,
    then the other point on C that lies on the x-axis is (16, 0). -/
theorem other_x_axis_point_on_circle (C : Set (ℝ × ℝ)) :
  (∀ (p : ℝ × ℝ), p ∈ C ↔ p.1^2 + p.2^2 = 16^2) →  -- Circle equation
  ((-16, 0) : ℝ × ℝ) ∈ C →                         -- (-16, 0) lies on C
  ((16, 0) : ℝ × ℝ) ∈ C ∧                          -- (16, 0) lies on C
  ∀ (x : ℝ), ((x, 0) : ℝ × ℝ) ∈ C → (x = -16 ∨ x = 16) := -- No other x-axis points on C
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_x_axis_point_on_circle_l1130_113078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_arrival_speed_l1130_113099

/-- The speed at which Arthur arrives exactly on time -/
noncomputable def n : ℚ := 72

/-- The distance from Arthur to David's house in km -/
noncomputable def d : ℚ := 140

/-- The time in hours for Arthur to drive to David's place at speed n -/
noncomputable def t : ℚ := 35 / 18

theorem arthur_arrival_speed : 
  (60 * (t + 1/12) = d) ∧ 
  (90 * (t - 1/12) = d) → 
  n * t = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_arrival_speed_l1130_113099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_theorem_l1130_113076

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle area ratio -/
noncomputable def triangle_area_ratio (N D C F M : Point) : ℝ :=
  let S_NDC := abs ((N.x - D.x) * (C.y - D.y) - (C.x - D.x) * (N.y - D.y)) / 2
  let S_FDM := abs ((F.x - D.x) * (M.y - D.y) - (M.x - D.x) * (F.y - D.y)) / 2
  S_NDC / S_FDM

/-- Main theorem -/
theorem parabola_line_intersection_theorem (E : Parabola) (l : Line) (A B C D F M N : Point) :
  E.focus = (1, 0) →
  E.equation = (λ x y ↦ y^2 = 4*x) →
  l.equation F.x F.y →
  l.equation A.x A.y →
  l.equation B.x B.y →
  l.equation C.x C.y →
  C.x = 0 →
  A.y^2 = 4*A.x →
  B.y^2 = 4*B.x →
  A.y/A.x + B.y/B.x = 4 →
  (∃ k, l.equation = (λ x y ↦ x + y = k)) →
  (∃ min_ratio, ∀ D M N, triangle_area_ratio N D C F M ≥ min_ratio ∧ 
    (∃ D' M' N', triangle_area_ratio N' D' C F M' = min_ratio)) →
  l.equation = (λ x y ↦ x + y = 1) ∧ 
  (∃ D' M' N', triangle_area_ratio N' D' C F M' = 2 ∧ 
    ∀ D M N, triangle_area_ratio N D C F M ≥ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_theorem_l1130_113076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_percentage_l1130_113051

theorem gilda_marbles_percentage (x : ℝ) (x_pos : x > 0) : 
  (x + 0.3 * x - 0.25 * (x + 0.3 * x) - 
   0.15 * (x + 0.3 * x - 0.25 * (x + 0.3 * x)) - 
   0.3 * (x + 0.3 * x - 0.25 * (x + 0.3 * x) - 
          0.15 * (x + 0.3 * x - 0.25 * (x + 0.3 * x)))) / 
  (x + 0.3 * x) * 100 = 44.625 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_percentage_l1130_113051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_is_20_l1130_113055

-- Define the rhombus and its properties
def Rhombus (AC BD : ℝ) : Prop :=
  AC > 0 ∧ BD > 0 ∧ AC^2 - 14*AC + 48 = 0 ∧ BD^2 - 14*BD + 48 = 0

-- Define the perimeter of the rhombus
noncomputable def RhombusPerimeter (AC BD : ℝ) : ℝ :=
  2 * Real.sqrt ((AC/2)^2 + (BD/2)^2)

-- Theorem statement
theorem rhombus_perimeter_is_20 (AC BD : ℝ) :
  Rhombus AC BD → RhombusPerimeter AC BD = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_is_20_l1130_113055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l1130_113016

/-- The curve in polar coordinates --/
noncomputable def r (φ : ℝ) : ℝ := 2 * Real.sin (4 * φ)

/-- The theorem stating that the area bounded by the curve r = 2 sin 4φ is 2π --/
theorem area_of_curve : 
  (∫ φ in (0)..(π/4), (1/2) * (r φ)^2) * 8 = 2 * π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_curve_l1130_113016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_romano_cheese_calculation_l1130_113089

/-- The amount of romano cheese needed to create a special blend -/
noncomputable def romano_cheese_needed (special_blend_cost mozzarella_cost romano_cost : ℝ) 
  (mozzarella_amount : ℝ) : ℝ :=
  let total_blend_cost := special_blend_cost * mozzarella_amount
  let mozzarella_total_cost := mozzarella_cost * mozzarella_amount
  let romano_total_cost := total_blend_cost - mozzarella_total_cost
  romano_total_cost / romano_cost

theorem romano_cheese_calculation :
  let special_blend_cost : ℝ := 696.05
  let mozzarella_cost : ℝ := 504.35
  let romano_cost : ℝ := 887.75
  let mozzarella_amount : ℝ := 19
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
    |romano_cheese_needed special_blend_cost mozzarella_cost romano_cost mozzarella_amount - 4.103| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_romano_cheese_calculation_l1130_113089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amp_1_2010_l1130_113030

-- Define the custom operation &
def amp : ℕ+ → ℕ+ → ℕ+
  | m, n => sorry  -- We'll leave the actual implementation as 'sorry' for now

-- Define the properties of the operation
axiom amp_base : amp 1 1 = 2
axiom amp_step (m n k : ℕ+) : amp m n = k → amp m (n + 1) = k + 3

-- State the theorem
theorem amp_1_2010 : amp 1 2010 = 6029 := by
  sorry

-- You can add more theorems or lemmas here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amp_1_2010_l1130_113030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_w_y_is_20_3_l1130_113095

-- Define the ratios
def ratio_w_x : ℚ := 5 / 2
def ratio_y_z : ℚ := 3 / 2
def ratio_z_x : ℚ := 1 / 4

-- Theorem statement
theorem ratio_w_y_is_20_3 : 
  ∀ (x y z w : ℚ),
  w / x = ratio_w_x →
  y / z = ratio_y_z →
  z / x = ratio_z_x →
  w / y = 20 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_w_y_is_20_3_l1130_113095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_proof_l1130_113001

theorem complex_modulus_proof : Complex.abs (3/4 - 3*Complex.I) = Real.sqrt 153 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_proof_l1130_113001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_probability_l1130_113077

-- Define the regular tetrahedron
structure RegularTetrahedron where
  height : ℝ
  baseEdgeLength : ℝ

-- Define a point inside the tetrahedron
structure PointInTetrahedron where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the probability function
noncomputable def probability_volume_less_than_half (t : RegularTetrahedron) : ℝ :=
  7/8

-- State the theorem
theorem tetrahedron_volume_probability 
  (t : RegularTetrahedron) 
  (h1 : t.height = 3) 
  (h2 : t.baseEdgeLength = 4) :
  probability_volume_less_than_half t = 7/8 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_probability_l1130_113077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_eight_dice_l1130_113060

-- Define the number of dice
def num_dice : ℕ := 8

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the target number we're looking for
def target_num : ℕ := 2

-- Define the number of dice we want to show the target number
def target_count : ℕ := 4

-- Define the probability of rolling the target number on a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of not rolling the target number on a single die
def single_prob_complement : ℚ := 1 - single_prob

-- Theorem statement
theorem probability_four_twos_in_eight_dice :
  (Nat.choose num_dice target_count : ℚ) * single_prob ^ target_count * single_prob_complement ^ (num_dice - target_count) =
  (70 : ℚ) * (5 : ℚ)^4 / (6 : ℚ)^8 := by
  sorry

#eval (Nat.choose num_dice target_count : ℚ) * single_prob ^ target_count * single_prob_complement ^ (num_dice - target_count)
#eval (70 : ℚ) * (5 : ℚ)^4 / (6 : ℚ)^8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_eight_dice_l1130_113060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_not_minus_two_two_l1130_113029

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem vertex_not_minus_two_two
  (a b c : ℝ)
  (h1 : quadratic_function a b c 1 = 0)
  (h2 : ∀ x, quadratic_function a b c x = quadratic_function a b c (4 - x)) :
  ¬(a * (-2)^2 + b * (-2) + c = 2 ∧ -b / (2 * a) = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_not_minus_two_two_l1130_113029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_rotation_in_dihedral_angle_l1130_113015

/-- Number of rotations a thin rigid ring makes before stopping in a dihedral angle -/
noncomputable def number_of_rotations (R ω μ g : ℝ) : ℝ :=
  (ω^2 * R * (1 + μ^2)) / (4 * Real.pi * g * μ * (1 + μ))

/-- Theorem stating the number of rotations a thin rigid ring makes before stopping in a dihedral angle -/
theorem ring_rotation_in_dihedral_angle 
  (R ω μ g : ℝ) 
  (h_R : R > 0) 
  (h_ω : ω > 0) 
  (h_μ : μ > 0) 
  (h_g : g > 0) :
  ∃ (n : ℝ), n = number_of_rotations R ω μ g ∧ n > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_rotation_in_dihedral_angle_l1130_113015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_product_l1130_113056

theorem absolute_value_equation_solution_product : 
  (∃ s : Finset ℝ, (∀ x : ℝ, x ∈ s ↔ |x - 5| - 4 = -1) ∧ 
  (s.prod id) = 16) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_product_l1130_113056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1130_113004

noncomputable def f (x : ℝ) := 2*x + 2/x

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x > 0 ∧ f x = y) ↔ y ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1130_113004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F1PF2_is_right_angle_l1130_113064

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem angle_F1PF2_is_right_angle (P : ℝ × ℝ) 
  (h1 : point_on_hyperbola P) 
  (h2 : distance P F1 * distance P F2 = 32) : 
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    Real.cos θ = (distance P F1)^2 + (distance P F2)^2 - (distance F1 F2)^2 / (2 * distance P F1 * distance P F2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_F1PF2_is_right_angle_l1130_113064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1130_113040

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- Check if a point is on the ellipse -/
def Ellipse.containsPoint (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : Ellipse) :
  e.eccentricity = Real.sqrt 5 / 5 →
  e.containsPoint (-5) 4 →
  e.a^2 = 45 ∧ e.b^2 = 36 := by
  sorry

#eval "Ellipse equation theorem defined"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l1130_113040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_unit_circle_l1130_113090

/-- Given two distinct real numbers a and b satisfying the specified equations,
    prove that the line connecting points A(a², a) and B(b², b) intersects
    the unit circle centered at the origin. -/
theorem line_intersects_unit_circle (θ a b : ℝ) (ha : a^2 * Real.sin θ + a * Real.cos θ - π/4 = 0)
    (hb : b^2 * Real.sin θ + b * Real.cos θ - π/4 = 0) (hab : a ≠ b) :
    ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (y - a = (b - a) / (b^2 - a^2) * (x - a^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_unit_circle_l1130_113090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1130_113018

noncomputable def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

def real_axis_length (a : ℝ) : ℝ := 2 * a

noncomputable def chord_length (x₁ x₂ y₁ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

def line (x m : ℝ) : ℝ := x + m

theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (eccentricity a (Real.sqrt 3 * a) = Real.sqrt 3) →
  (real_axis_length a = 2) →
  (∀ x y, hyperbola x y a b ↔ x^2 - y^2 / 2 = 1) ∧
  (∀ m, (∃ x₁ x₂ y₁ y₂, 
    hyperbola x₁ y₁ a b ∧ 
    hyperbola x₂ y₂ a b ∧ 
    y₁ = line x₁ m ∧ 
    y₂ = line x₂ m ∧
    chord_length x₁ x₂ y₁ y₂ = 4 * Real.sqrt 2) → 
  m = 1 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1130_113018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_M_l1130_113019

-- Define the set P
def P : Finset ℕ := {0, 1}

-- Define the set M
def M : Finset (Finset ℕ) := Finset.powerset P

-- Theorem statement
theorem number_of_subsets_of_M : Finset.card (Finset.powerset M) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_M_l1130_113019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_simplest_l1130_113017

/-- A quadratic radical is considered simpler if:
    1. It doesn't have a denominator under the square root
    2. The radicand cannot be factored to contain a perfect square
    3. The radicand is not itself a perfect square -/
noncomputable def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℚ, x ≠ (y : ℝ)^2 ∧ 
  ∀ z : ℕ, z > 1 → ¬ (∃ (a b : ℕ), x = (a * z : ℝ) ∧ b^2 = z)

/-- The given options for quadratic radicals -/
noncomputable def options : List ℝ := [Real.sqrt (1/3), Real.sqrt 7, Real.sqrt 9, Real.sqrt 20]

/-- Theorem stating that √7 is the simplest quadratic radical among the given options -/
theorem sqrt_seven_simplest : 
  ∀ x ∈ options, x ≠ Real.sqrt 7 → ¬(is_simplest_quadratic_radical x) :=
by
  sorry

#check sqrt_seven_simplest

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_seven_simplest_l1130_113017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1130_113036

theorem congruent_integers_count : 
  (Finset.filter (fun x => x > 0 ∧ x < 1200 ∧ x % 7 = 3) (Finset.range 1200)).card = 171 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_integers_count_l1130_113036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l1130_113087

noncomputable def f (x a : ℝ) : ℝ := 1 / (2^x + 2) + a

theorem odd_function_constant (a : ℝ) : 
  (∀ x, f x a = -f (-x) a) → a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_l1130_113087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_inverse_of_f_l1130_113068

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 - 3 * x

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := (3 - x) / 3

-- Theorem stating that g is the inverse of f
theorem g_is_inverse_of_f : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_inverse_of_f_l1130_113068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elena_has_four_l1130_113045

/-- Represents a player in the card game -/
inductive Player
| Elena
| Liam
| Hannah
| Felix
| Sara

/-- Represents a card in the deck -/
def Card := Fin 10

/-- A function that assigns a score to each player -/
def score : Player → Nat
| Player.Elena => 9
| Player.Liam => 8
| Player.Hannah => 13
| Player.Felix => 18
| Player.Sara => 17

/-- A function that assigns two cards to each player -/
def playerCards : Player → (Card × Card) := sorry

/-- The sum of the two cards assigned to a player equals their score -/
axiom score_sum (p : Player) : 
  (playerCards p).1.val + (playerCards p).2.val + 2 = score p

/-- All cards are unique (no card is assigned to more than one player) -/
axiom cards_unique : ∀ p1 p2 : Player, p1 ≠ p2 → 
  (playerCards p1).1 ≠ (playerCards p2).1 ∧ 
  (playerCards p1).1 ≠ (playerCards p2).2 ∧
  (playerCards p1).2 ≠ (playerCards p2).1 ∧
  (playerCards p1).2 ≠ (playerCards p2).2

/-- Elena was given card 4 -/
theorem elena_has_four : 
  (playerCards Player.Elena).1 = ⟨3, sorry⟩ ∨ (playerCards Player.Elena).2 = ⟨3, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elena_has_four_l1130_113045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2013_l1130_113007

def my_sequence : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | k + 3 => (my_sequence (k + 2) + my_sequence (k + 1) + 1) / my_sequence k

theorem my_sequence_2013 : my_sequence 2013 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2013_l1130_113007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l1130_113080

theorem election_votes (winning_percentage : ℝ) (majority : ℕ) : ℕ :=
  let total_votes := 840
  have h1 : winning_percentage = 75 / 100 := by sorry
  have h2 : majority = 420 := by sorry
  have h3 : total_votes = 840 := by sorry
  total_votes

#check election_votes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l1130_113080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_intersection_sum_l1130_113028

/-- Represents a 2000 × 2000 grid where each cell is either 1 or -1 -/
def Grid := Fin 2000 → Fin 2000 → Int

/-- Predicate to check if a grid contains only 1 or -1 -/
def validGrid (g : Grid) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

/-- The sum of all numbers in the grid -/
def gridSum (g : Grid) : Int :=
  Finset.sum (Finset.univ : Finset (Fin 2000)) (λ i => 
    Finset.sum (Finset.univ : Finset (Fin 2000)) (λ j => g i j))

/-- The sum of numbers at the intersections of given rows and columns -/
def intersectionSum (g : Grid) (rows cols : Finset (Fin 2000)) : Int :=
  Finset.sum rows (λ i => Finset.sum cols (λ j => g i j))

/-- Main theorem -/
theorem grid_intersection_sum (g : Grid) 
  (h_valid : validGrid g) 
  (h_sum : gridSum g ≥ 0) :
  ∃ (rows cols : Finset (Fin 2000)), 
    rows.card = 1000 ∧ 
    cols.card = 1000 ∧ 
    intersectionSum g rows cols ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_intersection_sum_l1130_113028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l1130_113054

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 + Real.sqrt 27 = 10 * Real.sqrt 2 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l1130_113054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l1130_113079

/-- Given that the terminal side of angle α passes through point (4,-3),
    prove that 2sin α + cos α = -2/5 -/
theorem angle_terminal_side_theorem (α : Real) :
  let P : Real × Real := (4, -3)
  let r : Real := Real.sqrt (P.1^2 + P.2^2)
  (∃ t : Real, t > 0 ∧ t * Real.cos α = P.1 ∧ t * Real.sin α = P.2) →
  2 * Real.sin α + Real.cos α = -2/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_theorem_l1130_113079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1130_113084

theorem simplify_expression (k : ℝ) (h : k ≠ 0) :
  ((-1 / (3 * k)) ^ (-3 : ℤ)) * ((2 * k) ^ (-2 : ℤ)) + k^2 = (4 * k^2 - 27 * k) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1130_113084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_to_fill_cube_l1130_113035

-- Define the base areas of the cubes
def small_base_area : ℝ := 4
def large_base_area : ℝ := 36

-- Define a function to calculate the volume of a cube given its base area
noncomputable def cube_volume (base_area : ℝ) : ℝ :=
  (Real.sqrt base_area) ^ 3

-- Define the volumes of the small and large cubes
noncomputable def small_volume : ℝ := cube_volume small_base_area
noncomputable def large_volume : ℝ := cube_volume large_base_area

-- State the theorem
theorem min_trips_to_fill_cube : 
  ⌈large_volume / small_volume⌉ = 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trips_to_fill_cube_l1130_113035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beths_remaining_crayons_is_2074_l1130_113011

/-- The number of crayons Beth has left after giving away 2/5 of her initial 3456 crayons -/
def beths_remaining_crayons : ℕ :=
  let initial_crayons : ℕ := 3456
  let fraction_given : ℚ := 2 / 5
  let crayons_given : ℕ := (fraction_given * initial_crayons).floor.toNat
  initial_crayons - crayons_given

/-- Proof that Beth has 2074 crayons left -/
theorem beths_remaining_crayons_is_2074 : beths_remaining_crayons = 2074 := by
  -- Unfold the definition of beths_remaining_crayons
  unfold beths_remaining_crayons
  -- Perform the calculation
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beths_remaining_crayons_is_2074_l1130_113011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1130_113057

/-- The volume of a cylindrical tank -/
noncomputable def cylinderVolume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical tank with diameter 20 feet and depth 6 feet is 600π cubic feet -/
theorem water_tank_volume :
  cylinderVolume 20 6 = 600 * Real.pi := by
  -- Unfold the definition of cylinderVolume
  unfold cylinderVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1130_113057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1130_113006

noncomputable def f (x : ℝ) := Real.sqrt (x + 1) + 1 / x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x ∈ Set.Ici (-1) ∧ x ≠ 0} :=
by sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1130_113006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_up_distance_l1130_113033

/-- The distance Flash needs to cover to catch up with Ace -/
noncomputable def catch_up_distance (x y : ℝ) : ℝ := 2 * x * y / (2 * x - 1)

/-- Theorem stating the distance Flash needs to cover to catch up with Ace -/
theorem flash_catch_up_distance (x y : ℝ) (hx : x > 0.5) :
  let ace_speed := (1 : ℝ)
  let flash_speed := 2 * x
  let head_start := y
  catch_up_distance x y = flash_speed * (head_start / (flash_speed - ace_speed)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catch_up_distance_l1130_113033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_200_terms_l1130_113047

open BigOperators

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a i

theorem sum_of_first_200_terms 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → a n + a (n - 1) = (-1)^n * 3) : 
  sequence_sum a 200 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_200_terms_l1130_113047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_theta_and_m_range_l1130_113002

noncomputable section

/-- Given function f --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x - (m - 1) / x - Real.log x

/-- Given function g --/
noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := 1 / (x * Real.cos θ) + Real.log x

/-- Function h --/
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := m * x - m / x - 2 * Real.log x

theorem tangent_line_and_theta_and_m_range :
  (∀ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (∀ x : ℝ, x ≥ 1 → Monotone (g θ)) → θ = 0) ∧
  (∀ x : ℝ, x > 0 → (deriv (f 3)) x = 4 → f 3 1 = 1 → 
    (fun x => 4 * x - 3) = fun x => f 3 x + 3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 0 → Monotone (h m)) → 
    m ∈ Set.Iic 0 ∪ Set.Ici 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_theta_and_m_range_l1130_113002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1130_113085

noncomputable def average_of_multiples (x : ℝ) : ℝ := (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7

def median_of_multiples (n : ℕ) : ℝ := 2 * n

theorem find_x (n : ℕ) (x : ℝ) (h1 : n = 10) 
  (h2 : (average_of_multiples x)^2 - (median_of_multiples n)^2 = 0) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1130_113085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_sum_l1130_113049

/-- Given two monic, non-constant polynomials with integer coefficients p and q
    such that x^8 - 50x^4 + 1 = p(x) * q(x), prove that p(1) + q(1) = 4 -/
theorem polynomial_factorization_sum (p q : Polynomial ℤ) :
  Polynomial.Monic p →
  Polynomial.Monic q →
  Polynomial.degree p > 0 →
  Polynomial.degree q > 0 →
  (∀ (x : ℤ), (X : Polynomial ℤ)^8 - 50*(X : Polynomial ℤ)^4 + 1 = p * q) →
  (Polynomial.eval 1 p + Polynomial.eval 1 q = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_sum_l1130_113049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_truck_cost_minimization_l1130_113041

-- Define the problem parameters
noncomputable def distance : ℝ := 130
noncomputable def gas_price : ℝ := 8
noncomputable def wage : ℝ := 80

-- Define the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  distance * (gas_price * (2 + x^2 / 360) / x + wage / x)

-- State the theorem
theorem freight_truck_cost_minimization
  (x : ℝ)
  (h_x_range : 50 ≤ x ∧ x ≤ 100) :
  -- 1. Total cost expression
  total_cost x = 130 * (96 / x + x / 45) ∧
  -- 2. Minimum cost occurs at x = 12√30
  (∀ y, 50 ≤ y ∧ y ≤ 100 → total_cost (12 * Real.sqrt 30) ≤ total_cost y) ∧
  -- 3. Minimum cost value
  total_cost (12 * Real.sqrt 30) = (208 / 3) * Real.sqrt 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_truck_cost_minimization_l1130_113041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l1130_113034

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The specific equation we're examining -/
noncomputable def f (x : ℝ) : ℝ := (1/3) * x - 2

/-- Theorem stating that our specific equation is linear -/
theorem f_is_linear : is_linear_equation f := by
  use (1/3), -2
  constructor
  · exact one_div_ne_zero three_ne_zero
  · intro x
    rfl

#check f_is_linear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_linear_l1130_113034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1130_113022

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 4√2. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_l1130_113022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1130_113059

/-- The price of an item after a percentage increase followed by a percentage discount -/
noncomputable def final_price (initial_price : ℝ) (increase_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  initial_price * (1 + increase_percent / 100) * (1 - discount_percent / 100)

/-- Theorem stating that an item initially priced at $50, after a 20% increase and 15% discount, costs $51 -/
theorem price_change_theorem :
  final_price 50 20 15 = 51 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l1130_113059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_vector_l1130_113071

/-- Given vectors a and b, if the projection of b onto a is equal to a, then the second component of b is 1. -/
theorem projection_equals_vector (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.1 = 2) :
  ((a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) • a = a → b.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equals_vector_l1130_113071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_7_l1130_113008

/-- The set of positive factors of 60 -/
def factors_of_60 : Finset Nat := Finset.filter (· ∣ 60) (Finset.range 61)

/-- The set of positive factors of 60 that are less than 7 -/
def factors_less_than_7 : Finset Nat := Finset.filter (· < 7) factors_of_60

/-- The probability of a randomly drawn positive factor of 60 being less than 7 -/
theorem probability_factor_less_than_7 : 
  (factors_less_than_7.card : ℚ) / (factors_of_60.card : ℚ) = 1 / 2 := by
  sorry

#eval factors_of_60
#eval factors_less_than_7
#eval factors_less_than_7.card
#eval factors_of_60.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_7_l1130_113008
