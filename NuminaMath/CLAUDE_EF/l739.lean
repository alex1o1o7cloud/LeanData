import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l739_73983

-- Define the constants
noncomputable def a : ℝ := (0.3 : ℝ) ^ 2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 2 ^ (0.3 : ℝ)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l739_73983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_sum_and_power_sum_integer_l739_73958

theorem odd_integer_sum_and_power_sum_integer (n : ℕ) :
  (n % 2 = 1) ↔ 
  (∃ (a b : ℚ), 
    0 < a ∧ 0 < b ∧ 
    ¬(∃ (m : ℤ), a = m) ∧ 
    ¬(∃ (m : ℤ), b = m) ∧
    (∃ (k : ℤ), a + b = k) ∧ 
    (∃ (k : ℤ), a^n + b^n = k)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_sum_and_power_sum_integer_l739_73958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accessible_functions_l739_73993

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 2/x
noncomputable def g (x : ℝ) : ℝ := Real.log x + 2

-- Define the distance function between f and g
noncomputable def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem accessible_functions : ∀ x > 0, distance x < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accessible_functions_l739_73993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l739_73924

-- Define the annual depletion rate
noncomputable def annual_depletion_rate : ℝ := 0.10

-- Define the value after 2 years
noncomputable def value_after_2_years : ℝ := 648

-- Define the number of years
def years : ℕ := 2

-- Define the present value function
noncomputable def present_value (future_value : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  future_value / ((1 - rate) ^ time)

-- Theorem statement
theorem machine_present_value :
  present_value value_after_2_years annual_depletion_rate years = 800 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l739_73924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tosses_is_three_l739_73991

/-- A fair coin is a coin with probability 0.5 for both heads and tails -/
def fair_coin (p : ℝ → Prop) : Prop := ∀ x, p x ↔ x = 1/2

/-- The event of getting both heads and tails -/
def both_outcomes (n : ℕ) : Prop := n ≥ 2

/-- The probability of getting the same outcome n-1 times and then a different outcome on the nth toss -/
noncomputable def prob_same_then_diff (n : ℕ) : ℝ := (1/2)^n

/-- The expected number of tosses -/
noncomputable def expected_tosses : ℝ := ∑' n, n * (prob_same_then_diff n) * 2

/-- Theorem stating that the expected number of tosses is 3 -/
theorem expected_tosses_is_three : 
  expected_tosses = 3 :=
sorry

#check expected_tosses_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tosses_is_three_l739_73991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l739_73979

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let x : ℝ := -π/12
  let a : ℝ × ℝ := (Real.cos (π/2 - x), Real.sin (π/2 + x))
  let b : ℝ × ℝ := (Real.sin (π/2 + x), Real.sin x)
  angle_between_vectors a b = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l739_73979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_covering_exists_l739_73986

/-- A domino is a 1x2 or 2x1 rectangle -/
inductive Domino
| horizontal : Domino
| vertical : Domino

/-- A 6x6 board -/
def Board := Fin 6 → Fin 6 → Bool

/-- The total number of dominoes -/
def totalDominoes : Nat := 18

/-- A placement of k dominoes on the board -/
def Placement (k : Nat) := Fin k → (Fin 6 × Fin 6) × Domino

/-- The smallest k for which a unique covering exists -/
def smallestK : Nat := 5

/-- Predicate to check if a board covering is valid -/
def valid_covering (b : Board) (p : Placement k) (p' : Placement (totalDominoes - k)) : Prop := sorry

theorem unique_covering_exists (k : Nat) :
  k = smallestK →
  ∃ (p : Placement k),
    (∀ (p' : Placement (totalDominoes - k)),
      ∃! (b : Board), valid_covering b p p') ∧
    (∀ (k' : Nat), k' < k →
      ¬∃ (p : Placement k'),
        ∀ (p' : Placement (totalDominoes - k')),
          ∃! (b : Board), valid_covering b p p') := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_covering_exists_l739_73986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_and_a_range_l739_73959

noncomputable def f (x : ℝ) : ℝ := ((x + 1) / x) ^ 2

noncomputable def f_inv (x : ℝ) : ℝ := 1 / (Real.sqrt x - 1)

theorem f_inv_and_a_range :
  (∀ x > 0, f x = ((x + 1) / x) ^ 2) ∧
  (∀ x > 1, f_inv x = 1 / (Real.sqrt x - 1)) ∧
  (∀ x ≥ 2, f (f_inv x) = x) ∧
  (∀ x ≥ 2, f_inv (f x) = x) ∧
  {a : ℝ | ∀ x ≥ 2, (x - 1) * (f_inv x) > a * (a - Real.sqrt x)} = Set.Ioo (-1) (1 + Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inv_and_a_range_l739_73959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l739_73936

/-- The area of a hexagon formed by removing three equilateral triangles from a larger equilateral triangle -/
theorem hexagon_area (side_large side1 side2 side3 : ℝ) : 
  side_large = 11 ∧ side1 = 1 ∧ side2 = 2 ∧ side3 = 6 →
  (Real.sqrt 3 / 4 * side_large^2) - (Real.sqrt 3 / 4 * side1^2) - (Real.sqrt 3 / 4 * side2^2) - (Real.sqrt 3 / 4 * side3^2) = 20 * Real.sqrt 3 :=
by
  intro h
  sorry

#check hexagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l739_73936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_3_monotonicity_l739_73969

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2

-- Part 1: Tangent line
theorem tangent_line_at_3 : 
  let f_2 := f 2
  let tangent_line (x y : ℝ) := 3 * x - y - 9
  tangent_line 3 (f_2 3) = 0 ∧ 
  ∀ x, tangent_line x (f_2 x) = 0 ↔ x = 3 := by sorry

-- Part 2: Monotonicity
theorem monotonicity (a : ℝ) :
  (a = 0 → ∀ x y, x < y → f a x < f a y) ∧
  (a < 0 → 
    (∀ x y, x < y ∧ y < a → f a x > f a y) ∧
    (∀ x y, a < x ∧ x < y ∧ y < 0 → f a x > f a y) ∧
    (∀ x y, 0 < x ∧ x < y → f a x < f a y)) ∧
  (a > 0 → 
    (∀ x y, x < y ∧ y < 0 → f a x < f a y) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < a → f a x > f a y) ∧
    (∀ x y, a < x ∧ x < y → f a x < f a y)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_3_monotonicity_l739_73969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_zero_one_interval_l739_73909

def f (x : ℝ) : ℝ := x^3 + 2*x - 1

theorem root_in_zero_one_interval (k : ℤ) :
  (∃ x : ℝ, x ∈ Set.Ioo (k : ℝ) ((k + 1) : ℝ) ∧ f x = 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_zero_one_interval_l739_73909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l739_73902

theorem smallest_positive_angle (α : Real) : 
  (∃ (x y : Real), x = Real.sin (2 * Real.pi / 3) ∧ y = Real.cos (2 * Real.pi / 3) ∧ 
   x = Real.sin α ∧ y = Real.cos α ∧ α > 0) →
  α = 11 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l739_73902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l739_73937

noncomputable section

-- Define the functions and interval
def f₁ (a x : ℝ) : ℝ := Real.log (x - 3*a) / Real.log a
def f₂ (a x : ℝ) : ℝ := Real.log (1 / (x - a)) / Real.log a
def I (a : ℝ) : Set ℝ := Set.Icc (a + 2) (a + 3)

-- Define the closeness property
def are_close (f g : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x ∈ s, |f x - g x| ≤ 1

-- Main theorem
theorem functions_properties (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  ((∀ x ∈ I a, (x - 3*a > 0 ∧ x - a > 0)) ↔ (0 < a ∧ a < 1)) ∧
  (are_close (f₁ a) (f₂ a) (I a) ↔ (0 < a ∧ a ≤ (9 - Real.sqrt 57) / 12)) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l739_73937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l739_73914

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum : 
  let a : ℝ := 2
  let r : ℝ := 2
  let n : ℕ := 10
  geometric_sum a r n = 2046 := by
  -- Unfold the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp [pow_add, pow_mul]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l739_73914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l739_73944

def sequence_elem (n : ℕ) : ℕ := 2^n - 3

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def subsequence (f : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ g : ℕ → ℕ, Monotone g ∧ ∀ n, f n = s (g n)

theorem infinite_coprime_subsequence :
  ∃ f : ℕ → ℕ, subsequence f sequence_elem ∧
    (∀ i j : ℕ, i ≠ j → is_relatively_prime (f i) (f j)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_subsequence_l739_73944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_guess_probability_proof_of_probability_l739_73976

-- Define the set of possible digits
def Digits : Finset Nat := Finset.range 10

-- Define the probability of guessing correctly
def probability_correct_guess : ℚ := 1 / 10

-- Theorem statement
theorem correct_guess_probability :
  Digits.card = 10 →
  probability_correct_guess = 1 / Digits.card :=
by
  intro h
  simp [probability_correct_guess, h]

-- The actual proof
theorem proof_of_probability :
  probability_correct_guess = 1 / 10 :=
by
  have h : Digits.card = 10 := by simp [Digits]
  exact correct_guess_probability h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_guess_probability_proof_of_probability_l739_73976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l739_73905

theorem equation_solutions (x b c : ℝ) : 
  (x^4 + 5*x^2 = 126 → x = 3 ∨ x = -3) ∧
  (x^4 + 24 = 10*x^2 → x = 2 ∨ x = -2 ∨ x = Real.sqrt 6 ∨ x = -Real.sqrt 6) ∧
  (x^4 = 2*x^2 + 8 → x = 2 ∨ x = -2) ∧
  (x^6 = 3*x^3 + 40 → x = 2 ∨ x = -(5^(1/3))) ∧
  (x ≠ 0 → x^7 = b*x^5 + c*x^3 → x^4 = b*x^2 + c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l739_73905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_60_feet_l739_73985

/-- Represents the characteristics of a tiller and a plot of land. -/
structure TillerAndPlot where
  plot_width : ℝ
  tiller_swath_width : ℝ
  tilling_speed : ℝ
  tilling_time : ℝ

/-- Calculates the length of a plot given tiller and plot characteristics. -/
noncomputable def calculate_plot_length (tp : TillerAndPlot) : ℝ :=
  (tp.tilling_time * 60 * tp.tiller_swath_width) / (2 * tp.plot_width)

/-- Theorem stating that the calculated plot length is 60 feet given the specific conditions. -/
theorem plot_length_is_60_feet (tp : TillerAndPlot)
  (h1 : tp.plot_width = 110)
  (h2 : tp.tiller_swath_width = 2)
  (h3 : tp.tilling_speed = 2)
  (h4 : tp.tilling_time = 220) :
  calculate_plot_length tp = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_60_feet_l739_73985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_pair_palindrome_l739_73929

inductive Word
  | a : Word
  | b : Word
  | concat : Word → Word → Word

inductive GoodPair : Word → Word → Prop
  | base : GoodPair Word.a Word.b
  | left : GoodPair u v → GoodPair (Word.concat u v) v
  | right : GoodPair u v → GoodPair u (Word.concat u v)

def isPalindrome : Word → Prop
  | Word.a => True
  | Word.b => True
  | Word.concat w1 w2 => isPalindrome w1 ∧ isPalindrome w2 ∧ w1 = w2

def concat : Word → Word → Word
  | w1, w2 => Word.concat w1 w2

theorem good_pair_palindrome (α β : Word) (h : GoodPair α β) :
  ∃ γ, isPalindrome γ ∧ concat α β = concat (Word.concat Word.a γ) Word.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_pair_palindrome_l739_73929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l739_73931

noncomputable def f (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 - 6*y + 9) + 
  Real.sqrt (x^2 + y^2 + 2*Real.sqrt 3*x + 3) + 
  Real.sqrt (x^2 + y^2 - 2*Real.sqrt 3*x + 3)

theorem f_minimum_value : ∀ x y : ℝ, f x y ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l739_73931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_five_nonzero_terms_l739_73901

-- Define the polynomials
def p (x : ℝ) : ℝ := 2*x - 3
def q (x : ℝ) : ℝ := 3*x^3 + 2*x^2 + x - 5
def r (x : ℝ) : ℝ := x^4 - x^3 + 2*x^2 - x + 1

-- Define the resulting polynomial
def result (x : ℝ) : ℝ := p x * q x - 4 * r x

-- Theorem statement
theorem expansion_has_five_nonzero_terms :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  ∀ x, result x = a*x^4 + b*x^3 + c*x^2 + d*x + e :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_five_nonzero_terms_l739_73901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_angle_equality_l739_73922

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a circle
def is_on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define a function to check if a point is on a line
def is_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the angle measure function
noncomputable def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Define a function to check if points are ordered on a line
def are_ordered_on_line (A B C D : Point) (l : Line) : Prop := sorry

-- Define the theorem
theorem intersecting_circles_angle_equality 
  (Γ₁ Γ₂ : Circle) 
  (P Q : Point) 
  (d : Line) 
  (A B C D : Point) :
  (is_on_circle P Γ₁ ∧ is_on_circle P Γ₂) →
  (is_on_circle Q Γ₁ ∧ is_on_circle Q Γ₂) →
  (is_on_circle A Γ₁ ∧ is_on_line A d) →
  (is_on_circle C Γ₁ ∧ is_on_line C d) →
  (is_on_circle B Γ₂ ∧ is_on_line B d) →
  (is_on_circle D Γ₂ ∧ is_on_line D d) →
  (are_ordered_on_line A B C D d) →
  angle_measure A P B = angle_measure C Q D :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_circles_angle_equality_l739_73922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_communication_l739_73920

/-- A type representing the three languages: English, French, and Russian -/
inductive ScientistLanguage
| English
| French
| Russian

/-- A function representing the language used between two scientists -/
def communication (n : ℕ) : (Fin n → Fin n → ScientistLanguage) → Prop :=
  λ f => ∀ i j : Fin n, i ≠ j → f i j = f j i

/-- The main theorem statement -/
theorem scientists_communication (f : Fin 17 → Fin 17 → ScientistLanguage) 
  (h : communication 17 f) : 
  ∃ (i j k : Fin 17), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    f i j = f j k ∧ f j k = f i k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientists_communication_l739_73920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_percentage_l739_73921

/-- Calculates the yield percentage of a stock -/
noncomputable def yield_percentage (dividend_rate : ℝ) (par_value : ℝ) (market_price : ℝ) : ℝ :=
  (dividend_rate * par_value / market_price) * 100

/-- Theorem: The yield percentage of a 21% stock quoted at $210 with a $100 par value is 10% -/
theorem stock_yield_percentage :
  let dividend_rate : ℝ := 0.21
  let par_value : ℝ := 100
  let market_price : ℝ := 210
  yield_percentage dividend_rate par_value market_price = 10 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_percentage_l739_73921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystalline_polyhedron_max_volume_l739_73926

/-- A crystalline polyhedron is a polyhedron inscribed in a sphere where n-1 of its vertices
    form tetrahedra of equal volume when any 4 are chosen. -/
structure CrystallinePolyhedron where
  n : ℕ  -- number of vertices
  R : ℝ  -- radius of the circumscribing sphere
  n_ge_4 : n ≥ 4  -- ensure we have at least 4 vertices

/-- The maximum volume of a crystalline polyhedron -/
noncomputable def max_volume (p : CrystallinePolyhedron) : ℝ :=
  (32 * (p.n - 1) * p.R^3 / 81) * Real.sin (2 * Real.pi / (p.n - 1 : ℝ))

/-- Theorem: The maximum volume of a crystalline polyhedron is given by the max_volume function -/
theorem crystalline_polyhedron_max_volume (p : CrystallinePolyhedron) :
  ∀ V : ℝ, V ≤ max_volume p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystalline_polyhedron_max_volume_l739_73926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_l739_73948

/-- The distance function of a particle moving in a straight line -/
noncomputable def S (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The instantaneous velocity of the particle at time t -/
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := deriv S t

theorem velocity_at_two :
  instantaneous_velocity 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_two_l739_73948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_l739_73942

-- Define the necessary structures and functions
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry
def RightTriangle (t : Triangle) : Prop := sorry
def Area (t : Triangle) : ℝ := sorry
def Hypotenuse (t : Triangle) : ℝ := sorry

theorem similar_triangles_hypotenuse (t1 t2 : Triangle) 
  (h_similar : SimilarTriangles t1 t2)
  (h_right1 : RightTriangle t1)
  (h_right2 : RightTriangle t2)
  (h_area1 : Area t1 = 8)
  (h_area2 : Area t2 = 200)
  (h_hyp1 : Hypotenuse t1 = 6) :
  Hypotenuse t2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_l739_73942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mass_correct_l739_73900

/-- Represents the mass of a tank filled with sand -/
structure SandTank where
  /-- Mass when the tank is three-fourths full -/
  mass_three_fourths : ℝ
  /-- Mass when the tank is one-third full -/
  mass_one_third : ℝ

/-- Calculates the total mass of the tank when completely filled with sand -/
noncomputable def total_mass (tank : SandTank) : ℝ :=
  (8 * tank.mass_three_fourths - 3 * tank.mass_one_third) / 5

/-- Theorem stating that the total mass calculation is correct -/
theorem total_mass_correct (tank : SandTank) :
  total_mass tank = (8 * tank.mass_three_fourths - 3 * tank.mass_one_third) / 5 := by
  -- Unfold the definition of total_mass
  unfold total_mass
  -- The equality holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_mass_correct_l739_73900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l739_73966

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P on the circle
def P_on_circle (a b : ℝ) : Prop := circle_eq a b

-- Define point D as the projection of P on the x-axis
def D (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the relationship between vectors DM and DP
def vector_relation (x y a b : ℝ) : Prop :=
  (x - a, y) = (0, 2 * b)

-- Theorem statement
theorem trajectory_of_M (a b x y : ℝ) :
  P_on_circle a b →
  vector_relation x y a b →
  4 * x^2 + y^2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l739_73966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l739_73915

theorem inequality_system_solutions (a : ℚ) : 
  (∃ (x₁ x₂ x₃ : ℤ) (h : x₁ < x₂ ∧ x₂ < x₃), 
    (∀ x : ℤ, (x - 3 < 6 * (x - 2) - 1 ∧ 5 + 2 * a - x > (5 - 2 * x) / 3) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)))
  → -5/6 < a ∧ a ≤ -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solutions_l739_73915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l739_73999

/-- The eccentricity of a hyperbola with asymptotes passing through a specific point -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let E := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x + 4*y = 0}
  let center_E := (1, -2)
  (∃ (x y : ℝ), (x, y) ∈ C ∧ y = (b/a) * x ∧ (x, y) = center_E) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l739_73999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l739_73939

theorem sin_cos_product (θ : ℝ) (h : (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2) :
  Real.sin θ * Real.cos θ = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l739_73939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l739_73968

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (k : ℝ), k * 3 = 4 ∧ k * a = b) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l739_73968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_proof_l739_73927

/-- Proves that a book has 500 pages given the reading progress over three nights and the remaining pages. -/
theorem book_pages_proof (first_night second_night third_night : Real) (pages_left : Nat) 
  (h1 : first_night = 0.2)
  (h2 : second_night = 0.2)
  (h3 : third_night = 0.3)
  (h4 : pages_left = 150)
  : 500 = pages_left / (1 - (first_night + second_night + third_night)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_proof_l739_73927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bound_smallest_bound_l739_73951

/-- The area of Figure n -/
noncomputable def A (n : ℕ) : ℝ :=
  2916 * (1 - (8/9) ^ (n + 1))

/-- The theorem stating that the area of any Figure n is less than 2916 -/
theorem area_bound (n : ℕ) : A n < 2916 := by
  sorry

/-- The theorem stating that 2916 is the smallest integer M such that A n < M for all n -/
theorem smallest_bound : ∀ M : ℕ, (∀ n : ℕ, A n < M) → 2916 ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bound_smallest_bound_l739_73951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l739_73990

theorem expression_evaluation (b : ℝ) (h : b = 4) : 
  (3 * b^(-(1/2 : ℝ)) + b^(-(1/2 : ℝ)) / 3) / b = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l739_73990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_flip_probability_l739_73949

/-- The probability of getting heads on the first flip and tails on the remaining 4 flips,
    when flipping a fair coin 5 times. -/
theorem fair_coin_flip_probability : ∃ (prob_heads : ℝ) (num_flips : ℕ),
  prob_heads = 1 / 2 ∧ num_flips = 5 ∧ prob_heads * (1 - prob_heads)^(num_flips - 1) = 1 / 32 := by
  -- Define the probability of heads for a single flip
  let prob_heads : ℝ := 1 / 2
  -- Define the number of flips
  let num_flips : ℕ := 5
  -- Provide the existence of prob_heads and num_flips
  use prob_heads, num_flips
  -- Prove the conditions and the final equality
  constructor
  · rfl  -- prob_heads = 1 / 2
  constructor
  · rfl  -- num_flips = 5
  -- The main probability calculation
  sorry  -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_flip_probability_l739_73949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l739_73906

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x - 3) / (x + 1)

-- Theorem for the monotonicity and max/min values of f
theorem f_properties :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 9 → f x ≤ 3/2) ∧
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 9 → 1/3 ≤ f x) ∧
  (f 9 = 3/2) ∧
  (f 2 = 1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l739_73906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_even_l739_73930

theorem a_is_even (a n : ℕ) (ha : a > 2) (hn : n > 1) (h_perfect_square : ∃ k : ℕ, a^n - 2^n = k^2) : 
  Even a := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_even_l739_73930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l739_73963

/-- The function f(x) defined as sin x + √3 cos 2x -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos (2 * x)

/-- The minimum positive period of f(x) is π -/
theorem min_period_of_f : ∃ T : ℝ, T > 0 ∧ T = π ∧ 
  (∀ x : ℝ, f (x + T) = f x) ∧
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_of_f_l739_73963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_karthik_weight_average_is_56_5_l739_73954

/-- Karthik's weight according to different opinions -/
def karthik_weight (w : ℝ) : Prop :=
  w > 55 ∧ w < 62 ∧ w > 50 ∧ w < 60 ∧ w ≤ 58

/-- The average of Karthik's probable weights -/
noncomputable def karthik_weight_average : ℝ := (55 + 58) / 2

/-- Theorem stating that the average of Karthik's probable weights is 56.5 kg -/
theorem karthik_weight_average_is_56_5 :
  karthik_weight_average = 56.5 := by
  unfold karthik_weight_average
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_karthik_weight_average_is_56_5_l739_73954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_num_extreme_points_range_of_a_for_nonnegative_f_l739_73923

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3*x + 2)

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > 0

-- Theorem for the tangent line equation when a = 0
theorem tangent_line_at_x_1 (x y : ℝ) (h : domain x) :
  f 0 x = Real.log x ∧ (x - y - 1 = 0 → y = f 0 x) := by
  sorry

-- Theorem for the number of extreme points
theorem num_extreme_points (a : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → ∀ x, domain x → (deriv (f a)) x > 0) ∧
  (a > 8/9 → ∃ x₁ x₂, domain x₁ ∧ domain x₂ ∧ x₁ < x₂ ∧ (deriv (f a)) x₁ = 0 ∧ (deriv (f a)) x₂ = 0) ∧
  (a < 0 → ∃ x₀, domain x₀ ∧ (deriv (f a)) x₀ = 0 ∧ ∀ x, domain x → x ≠ x₀ → (deriv (f a)) x ≠ 0) := by
  sorry

-- Theorem for the range of a when f(x) ≥ 0 for x ∈ [1, +∞)
theorem range_of_a_for_nonnegative_f :
  ∀ a, (∀ x, x ≥ 1 → f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_num_extreme_points_range_of_a_for_nonnegative_f_l739_73923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l739_73919

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1/x) - |x - 1/x|

-- Theorem for monotonicity when a = 1/2
theorem monotonicity_of_f :
  let a : ℝ := 1/2
  ∀ x y : ℝ, 
    ((0 < x ∧ x < y ∧ y ≤ 1) ∨ (x < y ∧ y ≤ -1)) → f a x < f a y ∧
    ((1 ≤ x ∧ x < y) ∨ (-1 ≤ x ∧ x < y ∧ y < 0)) → f a x > f a y :=
by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  (∀ x : ℝ, x > 0 → f a x ≥ (1/2) * x) ↔ a ≥ 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_a_l739_73919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l739_73987

noncomputable def equation (x : ℝ) : ℝ := 
  Real.sin x * (2 * Real.sin x ^ 2 - 1) * (Real.sin x - 2)

theorem equation_solutions_count :
  ∃ (S : Finset ℝ), S.card = 7 ∧
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ equation x = 0) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ equation x = 0 → x ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_count_l739_73987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_weight_calculation_l739_73934

/-- Represents the class of students with their weights -/
structure StudentClass where
  numStudents : ℕ
  initialAvgWeight : ℚ
  misreadWeights : List (ℚ × ℚ)

/-- Calculates the correct average weight of the class after adjusting for misread weights -/
def correctAverageWeight (c : StudentClass) : ℚ :=
  let initialTotalWeight := c.numStudents * c.initialAvgWeight
  let weightDifference := c.misreadWeights.foldl (fun acc (wrong, correct) => acc + correct - wrong) 0
  let adjustedTotalWeight := initialTotalWeight + weightDifference
  adjustedTotalWeight / c.numStudents

/-- The theorem stating the correct average weight for the given class -/
theorem correct_average_weight_calculation :
  let c : StudentClass := {
    numStudents := 30,
    initialAvgWeight := 584/10,
    misreadWeights := [(56, 62), (65, 59), (50, 54)]
  }
  correctAverageWeight c = 1756/30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_weight_calculation_l739_73934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_start_year_correct_max_avg_profit_year_correct_l739_73978

/-- Initial investment -/
noncomputable def initial_investment : ℝ := 720000

/-- First year expenses -/
noncomputable def first_year_expenses : ℝ := 120000

/-- Annual expense increase -/
noncomputable def annual_expense_increase : ℝ := 40000

/-- Annual sales -/
noncomputable def annual_sales : ℝ := 500000

/-- Net profit function -/
noncomputable def f (n : ℝ) : ℝ := -2 * n^2 + 40 * n - 72

/-- Average profit function -/
noncomputable def avg_profit (n : ℝ) : ℝ := -(2 * n + 72 / n) + 40

/-- The year when net profit begins -/
def profit_start_year : ℕ := sorry

/-- The year with the highest average net profit -/
def max_avg_profit_year : ℕ := sorry

theorem profit_start_year_correct :
  ∃ n : ℕ, f (n : ℝ) > 0 ∧ ∀ m : ℕ, m < n → f (m : ℝ) ≤ 0 := by sorry

theorem max_avg_profit_year_correct :
  ∀ n : ℕ, n ≥ 1 → avg_profit (max_avg_profit_year : ℝ) ≥ avg_profit (n : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_start_year_correct_max_avg_profit_year_correct_l739_73978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l739_73981

/-- Given a mixture of wine and water, prove the initial water percentage --/
theorem initial_water_percentage
  (total_initial : ℝ)
  (water_added : ℝ)
  (final_water_percentage : ℝ)
  (initial_water_percentage : ℝ)
  (h1 : total_initial = 150)
  (h2 : water_added = 30)
  (h3 : final_water_percentage = 25)
  (h4 : (final_water_percentage / 100) * (total_initial + water_added) =
        (initial_water_percentage / 100) * total_initial + water_added) :
  initial_water_percentage = 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_percentage_l739_73981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l739_73962

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 6*x + 5)

-- State the theorem
theorem f_decreasing_on_interval : 
  ∀ x y : ℝ, x ∈ Set.Ici 3 → y ∈ Set.Ici 3 → x ≤ y → f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l739_73962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_l739_73997

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 24 / (7 + 4 * x)

-- State the theorem
theorem inverse_g_cubed : (g⁻¹ 3)⁻¹^3 = 64 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_cubed_l739_73997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_values_l739_73917

theorem count_valid_a_values : 
  let valid_a (a : Int) := 
    (∀ x : Int, x > 0 → (3 * x > 4 * x - 6 ∧ 2 * x - a > -9) → x = 3) ∧
    (3 * 3 > 4 * 3 - 6 ∧ 2 * 3 - a > -9)
  (∃! (n : Nat), n > 0 ∧ (∃ (s : Finset Int), s.card = n ∧ ∀ a, a ∈ s ↔ valid_a a)) ∧
  (∃ (s : Finset Int), s.card = 2 ∧ ∀ a, a ∈ s ↔ valid_a a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_a_values_l739_73917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l739_73950

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of x
def domain : Set ℝ := Set.Icc (-3) 3

-- State that f is an even function
axiom f_even : ∀ x, f x = f (-x)

-- State the theorem
theorem inequality_solution :
  ∀ x ∈ domain, x^3 * f x < 0 ↔ x ∈ Set.Ioo (-3) (-1) ∪ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l739_73950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l739_73992

theorem solve_exponential_equation : 
  ∃ x : ℚ, (3 : ℝ) ^ (4 * (x : ℝ)) = (81 : ℝ) ^ (1/4 : ℝ) ∧ x = 1/4 := by
  -- Introduce the witness
  use (1/4 : ℚ)
  
  -- Split the goal into two parts
  constructor
  
  -- Prove the equation
  · norm_num
    -- Additional steps would go here, but we'll use sorry for now
    sorry
  
  -- Prove x = 1/4
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l739_73992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_moles_l739_73972

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction :=
  (nh4cl : Moles)  -- Ammonium chloride
  (naoh : Moles)   -- Sodium hydroxide
  (nh4oh : Moles)  -- Ammonium hydroxide
  (nacl : Moles)   -- Sodium chloride

/-- The stoichiometric coefficients of the reaction are all 1 -/
def isBalanced (r : Reaction) : Prop :=
  r.nh4cl = r.naoh ∧ r.nh4cl = r.nh4oh ∧ r.nh4cl = r.nacl

/-- Theorem: Given 1 mole of Sodium hydroxide and 1 mole of Ammonium hydroxide produced,
    the number of moles of Ammonium chloride combined is 1 -/
theorem ammonium_chloride_moles 
  (r : Reaction) 
  (h1 : isBalanced r) 
  (h2 : r.naoh = (1 : ℝ)) 
  (h3 : r.nh4oh = (1 : ℝ)) : 
  r.nh4cl = (1 : ℝ) := by 
  sorry

#check ammonium_chloride_moles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_chloride_moles_l739_73972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pairs_divisible_by_25_l739_73971

theorem equal_pairs_divisible_by_25 (n : ℕ) (h1 : n = 300) :
  let cards := Finset.range n
  let player1_cards := cards.filter (λ x => x % 2 = 0)
  let player2_cards := cards.filter (λ x => x % 2 = 1)
  let pairs1 := {p : ℕ × ℕ | p.1 ∈ player1_cards ∧ p.2 ∈ player1_cards ∧ (p.1 - p.2) % 25 = 0}
  let pairs2 := {p : ℕ × ℕ | p.1 ∈ player2_cards ∧ p.2 ∈ player2_cards ∧ (p.1 - p.2) % 25 = 0}
  Finset.card (Finset.filter (λ p => p ∈ pairs1) (player1_cards.product player1_cards)) =
  Finset.card (Finset.filter (λ p => p ∈ pairs2) (player2_cards.product player2_cards)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_pairs_divisible_by_25_l739_73971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_statements_l739_73957

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Define the planes and lines
variable (α β γ : Plane)
variable (m n : Line)

-- State the theorem
theorem geometry_statements :
  (∀ α β m n, parallel_line_plane m α → intersect α β n → parallel_line_line m n) ∧
  (∀ α β m n, perpendicular_line_plane m α → parallel_line_line m n → contains β n → planePerpendicular α β) ∧
  ¬(∀ α β γ, planePerpendicular α β → planePerpendicular γ β → planeParallel α γ) ∧
  (∀ α β γ m, intersect α β m → perpendicular_line_plane m γ → planePerpendicular α γ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_statements_l739_73957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_trigonometric_identity_l739_73973

/-- Given a line with equation 3x - y + 1 = 0 and inclination angle α, 
    prove that 1/2 * sin(2α) + cos²(α) = 2/5 -/
theorem line_inclination_trigonometric_identity 
  (x y : ℝ) (α : ℝ) 
  (h1 : 3 * x - y + 1 = 0) 
  (h2 : Real.tan α = 3) : 
  1/2 * Real.sin (2 * α) + Real.cos α ^ 2 = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_trigonometric_identity_l739_73973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_income_proof_l739_73977

/-- Represents the annual price increase factor -/
def annual_price_increase : ℝ := 1.05

/-- Represents the number of years -/
def years : ℕ := 3

/-- Represents the initial income increase factor for the first year -/
def initial_income_increase : ℝ := 1.10

/-- Calculates the required annual demand decrease to maintain or increase income -/
def required_demand_decrease : ℝ := 0.0166156

theorem maintain_income_proof (P : ℝ) (h : P > 0) :
  ∃ (d : ℝ), 
    (annual_price_increase ^ years) * (1 - years * d) ≥ initial_income_increase ∧
    abs (d - required_demand_decrease) < 0.000001 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintain_income_proof_l739_73977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_theorem_l739_73975

/-- Represents a player in the cheese division game -/
inductive Player
| A  -- First player
| B  -- Second player

/-- Represents the state of the cheese division game -/
structure GameState where
  pieces : List ℝ  -- List of cheese piece sizes
  turn : Player    -- Current player's turn

/-- The optimal strategy function for a player -/
noncomputable def optimalStrategy (player : Player) (state : GameState) : ℝ := sorry

/-- The game play function that simulates the entire game -/
noncomputable def playGame (initialState : GameState) : (ℝ × ℝ) := sorry

/-- Theorem stating the maximum guaranteed amounts for each player -/
theorem cheese_division_theorem :
  ∀ (initialState : GameState),
    initialState.pieces.sum = 50 ∧ 
    initialState.pieces.length = 2 ∧
    initialState.turn = Player.A →
    let (scoreA, scoreB) := playGame initialState
    scoreA ≥ 30 ∧ scoreB ≥ 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheese_division_theorem_l739_73975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l739_73964

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C --/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  cosine_law : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)

/-- The main theorem to be proved --/
theorem triangle_properties (t : AcuteTriangle) :
  (((t.b^2 - t.a^2 - t.c^2) / (t.a * t.c) = (Real.cos (t.A + t.C)) / ((Real.sin t.A) * (Real.cos t.A))) → t.A = π/4) ∧
  ((t.a = 2) → (∀ s : ℝ, s = (1/2) * t.b * t.c * (Real.sin t.A) → s ≤ Real.sqrt 2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l739_73964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_integer_l739_73941

theorem A_n_is_integer (a b n : ℕ) (h1 : a > b) (h2 : b > 0) (θ : ℝ) 
  (h3 : 0 < θ) (h4 : θ < Real.pi / 2) 
  (h5 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) :
  ∃ k : ℤ, (a^2 + b^2)^n * Real.sin (n * θ) = k := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_n_is_integer_l739_73941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_distribution_l739_73970

theorem sweets_distribution (total_children : ℕ) (absent_children : ℕ) (extra_sweets : ℕ) : 
  total_children = 190 → 
  absent_children = 70 → 
  extra_sweets = 14 → 
  (total_children - absent_children) * (total_children / (total_children - absent_children) + extra_sweets) = 38 * (total_children - absent_children) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sweets_distribution_l739_73970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_interval_l739_73928

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

-- State the theorem
theorem f_range_in_interval :
  ∃ (a b : ℝ), a = -3/2 ∧ b = 3 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_interval_l739_73928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l739_73946

open Real

/-- The function we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := (cos (x + π/4))^2 - (sin (x + π/4))^2

/-- The period we want to prove -/
noncomputable def period : ℝ := π

/-- Theorem stating that the smallest positive period of f is π -/
theorem smallest_positive_period_of_f : 
  (∀ x, f (x + period) = f x) ∧ 
  (∀ p, 0 < p → p < period → ∃ x, f (x + p) ≠ f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l739_73946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_score_l739_73960

/-- The average score of all students in a class given the average scores of boys and girls -/
theorem class_average_score (m n : ℝ) : 
  (20 * m + 23 * n) / (20 + 23) = 
  let num_boys : ℕ := 20
  let num_girls : ℕ := 23
  let boys_avg : ℝ := m
  let girls_avg : ℝ := n
  let total_students : ℕ := num_boys + num_girls
  let total_score : ℝ := num_boys * boys_avg + num_girls * girls_avg
  total_score / total_students :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_score_l739_73960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l739_73935

/-- The average speed for a round trip on a slope -/
noncomputable def averageSpeed (m n : ℝ) : ℝ :=
  (2 * m * n) / (m + n)

/-- Theorem: The average speed for a round trip on a slope with uphill speed m and downhill speed n is (2mn)/(m+n) -/
theorem round_trip_average_speed (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  averageSpeed m n = (2 * m * n) / (m + n) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_average_speed_l739_73935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_implies_tan_theta_l739_73974

-- Define a complex number z
noncomputable def z (θ : ℝ) : ℂ := (4/5 - Real.sin θ) + (Real.cos θ - 3/5) * Complex.I

-- State the theorem
theorem purely_imaginary_z_implies_tan_theta (θ : ℝ) :
  (z θ).re = 0 → Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_implies_tan_theta_l739_73974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_involution_iff_a_eq_one_l739_73903

/-- The function f(x) = (x^2 + ax) / (x^2 + x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x) / (x^2 + x + 1)

/-- Theorem stating that f(f(x)) = x for all x where f is defined if and only if a = 1 -/
theorem f_involution_iff_a_eq_one (a : ℝ) :
  (∀ x, f a (f a x) = x) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_involution_iff_a_eq_one_l739_73903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_area_ratio_l739_73953

theorem square_in_rectangle_area_ratio :
  ∀ (s : ℝ), s > 0 →
  (let square_side := s
   let rect_width := 3 * s
   let rect_length := (3/2) * rect_width
   let square_area := square_side ^ 2
   let rect_area := rect_length * rect_width
   square_area / rect_area) = 2/27 := by
  intro s hs
  simp [hs]
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_rectangle_area_ratio_l739_73953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vanessa_score_l739_73989

/-- Vanessa's basketball game score calculation -/
theorem vanessa_score (total_points : ℕ) (other_players : ℕ) (avg_points : ℚ) : 
  total_points = 75 → 
  other_players = 8 → 
  avg_points = 4.5 → 
  total_points - (↑other_players * avg_points).floor = 39 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vanessa_score_l739_73989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_approximation_l739_73984

/-- The function f(x) = sin x + sin(πx) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

/-- Theorem stating the existence of a sequence of periods for the given function -/
theorem periodic_approximation (d : ℝ) (hd : d > 0) :
  ∃ p : ℕ → ℝ, (∀ n x, |f (x + p n) - f x| < d) ∧ Filter.Tendsto p Filter.atTop Filter.atTop := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_approximation_l739_73984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_sum_l739_73945

/-- A configuration of 15 congruent disks on a unit circle. -/
structure DiskConfiguration where
  /-- The radius of each small disk -/
  small_radius : ℝ
  /-- The disks cover the unit circle -/
  covers_circle : small_radius > 0
  /-- The disks don't overlap -/
  no_overlap : small_radius ≤ Real.sin (π / 15)
  /-- Each disk is tangent to its neighbors -/
  tangent_neighbors : small_radius = 1 - Real.cos (π / 15)

/-- The theorem statement -/
theorem disk_area_sum (config : DiskConfiguration) :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (p : ℕ), Nat.Prime p → c % (p^2) ≠ 0) ∧
    15 * Real.pi * config.small_radius^2 = Real.pi * (a - b * Real.sqrt c) ∧
    a + b + c = 123 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_area_sum_l739_73945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_inscribed_circle_l739_73916

theorem shaded_area_square_inscribed_circle (square_side : ℝ) (h : square_side = 2) :
  let circle_radius : ℝ := square_side / Real.sqrt 2
  let circle_area : ℝ := Real.pi * circle_radius^2
  let square_area : ℝ := square_side^2
  let semicircle_area : ℝ := Real.pi * square_side^2 / 4
  let total_semicircle_area : ℝ := 4 * semicircle_area
  let outside_square_inside_circle : ℝ := circle_area - square_area
  let shaded_area : ℝ := total_semicircle_area - outside_square_inside_circle
  shaded_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_inscribed_circle_l739_73916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l739_73996

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  center : ℝ × ℝ
  fociOnAxes : Bool
  realAxis : ℝ
  asymptote : ℝ → ℝ

/-- The standard equation of a hyperbola -/
inductive StandardEquation
  | XMajor : StandardEquation  -- (x²/16) - (y²/64) = 1
  | YMajor : StandardEquation  -- (y²/16) - (x²/4) = 1

/-- Determines if a given equation is a valid standard equation for the hyperbola -/
def isValidStandardEquation (Γ : Hyperbola) (eq : StandardEquation) : Prop :=
  Γ.center = (0, 0) ∧
  Γ.fociOnAxes = true ∧
  Γ.realAxis = 4 ∧
  (∀ x, Γ.asymptote x = 2 * x) ∧
  (eq = StandardEquation.XMajor ∨ eq = StandardEquation.YMajor)

theorem hyperbola_standard_equation (Γ : Hyperbola) :
  ∃ eq : StandardEquation, isValidStandardEquation Γ eq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l739_73996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_one_l739_73907

noncomputable def a : ℝ := 0.9595
noncomputable def b : ℝ := 1.0555
noncomputable def c : ℝ := 0.9609
noncomputable def d : ℝ := 1.0400
noncomputable def e : ℝ := 0.9555

theorem closest_to_one :
  ∀ x ∈ ({a, b, c, d, e} : Set ℝ), |1 - c| ≤ |1 - x| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_one_l739_73907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l739_73994

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6) (1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l739_73994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_interval_value_l739_73932

/-- The probability distribution of X for n = 1, 2, 3, 4 -/
noncomputable def P (a : ℝ) (n : ℕ) : ℝ := a / (n * (n + 1))

/-- The sum of probabilities equals 1 -/
axiom sum_prob (a : ℝ) : (P a 1) + (P a 2) + (P a 3) + (P a 4) = 1

/-- The probability of 1/2 < X < 5/2 -/
noncomputable def prob_interval (a : ℝ) : ℝ := (P a 1) + (P a 2)

theorem prob_interval_value :
  ∃ a : ℝ, (∀ n : ℕ, n ≥ 1 ∧ n ≤ 4 → P a n ≥ 0) ∧ prob_interval a = 5/6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_interval_value_l739_73932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_l739_73911

/-- Given a population of 960 people and a systematic sample of 32 people,
    with the first number drawn being 9, prove that the number of people
    selected from the interval [450, 750] is 10. -/
theorem systematic_sample_count (population : ℕ) (sample_size : ℕ) (first_draw : ℕ) 
    (lower_bound upper_bound : ℕ) : 
    population = 960 → 
    sample_size = 32 → 
    first_draw = 9 → 
    lower_bound = 450 → 
    upper_bound = 750 → 
    (Finset.filter (λ n : ℕ ↦ 
      lower_bound ≤ (first_draw + (population / sample_size) * (n - 1)) ∧ 
      (first_draw + (population / sample_size) * (n - 1)) ≤ upper_bound) 
      (Finset.range (sample_size + 1))).card = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_l739_73911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_meeting_criterion_l739_73938

/-- A two-digit number -/
def TwoDigitNumber : Type := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- The tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- The result of subtracting the sum of digits from the number -/
def subtract_sum_of_digits (n : TwoDigitNumber) : ℕ := n.val - sum_of_digits n

/-- The criterion: subtracting the sum of digits results in a multiple of 3 -/
def meets_criterion (n : TwoDigitNumber) : Prop := 
  ∃ k : ℕ, subtract_sum_of_digits n = 3 * k

instance : Fintype TwoDigitNumber := sorry

instance : DecidablePred meets_criterion := sorry

theorem count_numbers_meeting_criterion : 
  (Finset.filter meets_criterion (Finset.univ : Finset TwoDigitNumber)).card = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_meeting_criterion_l739_73938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l739_73904

open Real NormedSpace

/-- The angle between two vectors given their magnitudes and the magnitude of their sum -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = sqrt 2) 
  (h2 : ‖b‖ = 1) 
  (h3 : ‖a + b‖ = 1) : 
  arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l739_73904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_mn_l739_73965

-- Define the points on a line
structure Point where
  x : ℝ

-- Define the order of points
def InOrder (a b c d : Point) : Prop :=
  a.x < b.x ∧ b.x < c.x ∧ c.x < d.x

-- Define the midpoint of two points
noncomputable def Midpoint (a b : Point) : Point :=
  ⟨(a.x + b.x) / 2⟩

-- Define the distance between two points
noncomputable def Distance (a b : Point) : ℝ :=
  |a.x - b.x|

-- Theorem statement
theorem segment_length_mn
  (a b c d : Point)
  (m n : Point)
  (h_order : InOrder a b c d)
  (h_m_midpoint : m = Midpoint a c)
  (h_n_midpoint : n = Midpoint b d)
  (h_ad_length : Distance a d = 68)
  (h_bc_length : Distance b c = 26) :
  Distance m n = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_mn_l739_73965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_points_l739_73956

/-- Represents the points accumulated by a player after a number of rotations -/
structure PlayerPoints where
  player : Nat
  points : Nat

/-- Represents the state of the game after a number of rotations -/
structure GameState where
  rotations : Nat
  playerPoints : List PlayerPoints

/-- Defines the circular arrangement of sectors on the table -/
def tableSectors : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

/-- The main theorem to prove -/
theorem player_one_points 
  (game : GameState) 
  (h1 : game.rotations = 13)
  (h2 : ∃ p ∈ game.playerPoints, p.player = 5 ∧ p.points = 72)
  (h3 : ∃ p ∈ game.playerPoints, p.player = 9 ∧ p.points = 84) :
  ∃ p ∈ game.playerPoints, p.player = 1 ∧ p.points = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_points_l739_73956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l739_73955

noncomputable def F (A B x : ℝ) : ℝ :=
  |Real.cos x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 2 + A * x + B|

theorem minimize_max_F :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧
  (∀ A B : ℝ, ∃ (M' : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 * Real.pi / 2 → F A B x ≤ M') ∧
    M ≤ M') ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 * Real.pi / 2 ∧ F 0 0 x = M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_F_l739_73955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l739_73961

/-- Represents a 3x3 matrix of integers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℤ

/-- The initial magic square -/
def initialSquare : Matrix3x3 := λ i j =>
  match i, j with
  | 0, 0 => 2 | 0, 1 => 7 | 0, 2 => 6
  | 1, 0 => 9 | 1, 1 => 5 | 1, 2 => 1
  | 2, 0 => 4 | 2, 1 => 3 | 2, 2 => 8

/-- Calculate the sum of a row -/
def rowSum (m : Matrix3x3) (row : Fin 3) : ℤ :=
  (m row 0) + (m row 1) + (m row 2)

/-- Calculate the sum of a column -/
def colSum (m : Matrix3x3) (col : Fin 3) : ℤ :=
  (m 0 col) + (m 1 col) + (m 2 col)

/-- Check if all sums are different -/
def allSumsDifferent (m : Matrix3x3) : Prop :=
  ∀ i j : Fin 3, i ≠ j →
    (rowSum m i ≠ rowSum m j) ∧
    (colSum m i ≠ colSum m j) ∧
    (rowSum m i ≠ colSum m j)

/-- Count the number of entries that differ between two matrices -/
def diffCount (m1 m2 : Matrix3x3) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 3)) (λ j =>
      if m1 i j ≠ m2 i j then 1 else 0))

/-- The main theorem -/
theorem min_changes_for_different_sums :
  ¬∃ m : Matrix3x3, (diffCount initialSquare m < 4) ∧ (allSumsDifferent m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_changes_for_different_sums_l739_73961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l739_73913

theorem trigonometric_identities :
  (∀ α : ℝ, Real.tan (π / 4 + α) = 1 / 2 → (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -2) ∧
  (Real.sin (π / 12) * Real.sin (5 * π / 12) = 1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l739_73913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winner_votes_l739_73982

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 7 / 10 →
  vote_difference = 280 →
  (winner_percentage * total_votes).floor - ((1 - winner_percentage) * total_votes).floor = vote_difference →
  (winner_percentage * total_votes).floor = 490 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_winner_votes_l739_73982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l739_73952

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the variables and conditions
variable (m n : ℝ)
variable (h1 : m > n)
variable (h2 : n > 1)

-- Define a and b
noncomputable def a (m n : ℝ) : ℝ := (lg (m * n))^(1/2) - (lg m)^(1/2)
noncomputable def b (m n : ℝ) : ℝ := (lg m)^(1/2) - (lg (m/n))^(1/2)

-- State the theorem
theorem a_less_than_b (m n : ℝ) (h1 : m > n) (h2 : n > 1) : a m n < b m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l739_73952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l739_73980

/-- The maximum distance from a point on the ellipse (x/4)² + (y/3)² = 1 to the line x + y = 2 -/
theorem max_distance_ellipse_to_line : 
  let ellipse := {p : ℝ × ℝ | (p.1/4)^2 + (p.2/3)^2 = 1}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 2}
  ∃ (d : ℝ), d = (7 * Real.sqrt 2) / 2 ∧ 
    (∀ p ∈ ellipse, ∀ q ∈ line, dist p q ≤ d) ∧
    (∃ p ∈ ellipse, ∃ q ∈ line, dist p q = d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l739_73980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C1_C2_l739_73912

/-- The circle C1 -/
def C1 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 2

/-- The ellipse C2 -/
def C2 (x y : ℝ) : Prop := x^2 / 10 + y^2 = 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The maximum distance between points on C1 and C2 -/
theorem max_distance_C1_C2 : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ (x1' y1' x2' y2' : ℝ), 
      C1 x1' y1' → C2 x2' y2' →
      distance x1 y1 x2 y2 ≥ distance x1' y1' x2' y2') ∧
    distance x1 y1 x2 y2 = 6 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C1_C2_l739_73912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l739_73943

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

theorem trapezium_other_side_length :
  ∀ (x : ℝ),
  trapeziumArea 20 x 15 = 270 →
  x = 16 := by
  intro x h
  unfold trapeziumArea at h
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l739_73943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_speed_calculation_l739_73947

/-- Represents a round trip journey on a mountain -/
structure MountainTrip where
  distanceOneWay : ℝ
  speedUp : ℝ
  speedDown : ℝ
  averageSpeed : ℝ

/-- Calculates the total time for a mountain trip -/
noncomputable def totalTime (trip : MountainTrip) : ℝ :=
  trip.distanceOneWay / trip.speedUp + trip.distanceOneWay / trip.speedDown

/-- Theorem stating that for a given average speed and downhill speed, 
    the uphill speed must be a specific value -/
theorem uphill_speed_calculation (trip : MountainTrip) 
  (h1 : trip.averageSpeed = 21)
  (h2 : trip.speedDown = 42)
  (h3 : trip.averageSpeed = (2 * trip.distanceOneWay) / (totalTime trip)) :
  trip.speedUp = 14 := by
  sorry

#check uphill_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_speed_calculation_l739_73947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l739_73998

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 7 = 0

-- Define the parabola
def parabola_eq (x y p : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the condition that the parabola's directrix is tangent to the circle
def directrix_tangent (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ (x - 3)^2 + y^2 = 16 ∧ (3 - p/2)^2 = 16

theorem parabola_circle_tangent :
  ∀ p : ℝ, (∃ x y : ℝ, parabola_eq x y p) → directrix_tangent p → p = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l739_73998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_integer_l739_73910

/-- A strictly increasing linear function from ℝ to ℝ -/
structure StrictlyIncreasingLinearFunction where
  f : ℝ → ℝ
  strictly_increasing : ∀ x y, x < y → f x < f y
  linear : ∃ a b, a > 0 ∧ ∀ x, f x = a * x + b

/-- Two strictly increasing linear functions with integer-valued correspondence -/
structure IntegerCorrespondingFunctions where
  f : StrictlyIncreasingLinearFunction
  g : StrictlyIncreasingLinearFunction
  integer_correspondence : ∀ x, Int.floor (f.f x) = Int.floor (g.f x)

/-- The main theorem -/
theorem difference_is_integer (funcs : IntegerCorrespondingFunctions) :
  ∀ x, ∃ n : ℤ, funcs.f.f x - funcs.g.f x = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_is_integer_l739_73910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l739_73908

/-- Calculates the average speed for a trip with two segments -/
noncomputable def average_speed (d1 d2 v1 v2 : ℝ) : ℝ :=
  (d1 + d2) / (d1 / v1 + d2 / v2)

/-- The average speed of a cyclist riding 7 km at 10 km/hr and 10 km at 7 km/hr is approximately 7.98 km/hr -/
theorem cyclist_average_speed :
  let d1 : ℝ := 7
  let d2 : ℝ := 10
  let v1 : ℝ := 10
  let v2 : ℝ := 7
  abs (average_speed d1 d2 v1 v2 - 7.98) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l739_73908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l739_73925

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem modulus_of_complex_fraction :
  Complex.abs ((2 : ℂ) - i) / ((1 : ℂ) + i) = Real.sqrt 10 / 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_fraction_l739_73925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_10_l739_73967

/-- The area of a triangle in the complex plane given three vertices --/
noncomputable def triangleArea (z1 z2 z3 : ℂ) : ℝ :=
  (1/2) * abs (z1.re * z2.im + z2.re * z3.im + z3.re * z1.im -
               z1.im * z2.re - z2.im * z3.re - z3.im * z1.re)

/-- The statement that 10 is the smallest positive integer satisfying the condition --/
theorem smallest_n_is_10 :
  (∀ n : ℕ, 0 < n → n < 10 →
    triangleArea (n + I) ((n + I)^2) ((n + I)^3) ≤ 4030) ∧
  4030 < triangleArea (10 + I) ((10 + I)^2) ((10 + I)^3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_is_10_l739_73967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_pentagon_regularity_l739_73933

/-- Represents a triangle --/
structure Triangle where
  -- Define triangle properties here
  mk :: -- Constructor

/-- Represents a pentagon --/
structure Pentagon where
  -- Define pentagon properties here
  mk :: -- Constructor

/-- Represents a five-pointed star with five congruent triangles and a pentagon --/
structure StarPentagon where
  /-- The five triangles in the star --/
  triangles : Fin 5 → Triangle
  /-- The pentagon in the center of the star --/
  pentagon : Pentagon
  /-- All triangles are congruent --/
  triangles_congruent : ∀ i j : Fin 5, i ≠ j → triangles i = triangles j
  /-- The triangles and pentagon form a valid five-pointed star --/
  valid_configuration : Bool

/-- Checks if all sides of a pentagon are equal --/
def AllSidesEqual (p : Pentagon) : Prop :=
  sorry

/-- Checks if all angles of a pentagon are equal --/
def AllAnglesEqual (p : Pentagon) : Prop :=
  sorry

/-- A regular pentagon has all sides equal and all angles equal --/
def IsRegularPentagon (p : Pentagon) : Prop :=
  AllSidesEqual p ∧ AllAnglesEqual p

/-- The main theorem: If a StarPentagon is formed, its pentagon is regular --/
theorem star_pentagon_regularity (s : StarPentagon) : IsRegularPentagon s.pentagon :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_pentagon_regularity_l739_73933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_dot_product_implies_theta_l739_73940

theorem even_dot_product_implies_theta (θ : ℝ) : 
  (∀ x : ℝ, (1 : ℝ) * Real.sin (x + θ) + Real.sqrt 3 * Real.cos (x + θ) = 
             (1 : ℝ) * Real.sin (-x + θ) + Real.sqrt 3 * Real.cos (-x + θ)) → 
  θ = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_dot_product_implies_theta_l739_73940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_tiling_l739_73918

/-- An L-shaped tile consisting of three cells -/
structure LTile where
  cells : Finset (ℕ × ℕ)
  is_valid : cells.card = 3 ∧ ∃ (a b : ℕ × ℕ), a ≠ b ∧ a ∈ cells ∧ b ∈ cells ∧
    ((a.1 = b.1 ∧ a.2 + 1 = b.2) ∨ (a.1 + 1 = b.1 ∧ a.2 = b.2))

/-- A grid of size 2^n x 2^n with one cell removed -/
structure Grid (n : ℕ) where
  cells : Finset (ℕ × ℕ)
  size_valid : cells.card = 2^n * 2^n - 1
  bounds : ∀ (x y : ℕ), (x, y) ∈ cells → x < 2^n ∧ y < 2^n

/-- A tiling of a grid using L-shaped tiles -/
def is_valid_tiling (n : ℕ) (g : Grid n) (tiling : List LTile) : Prop :=
  (∀ t ∈ tiling, ∀ c ∈ t.cells, c ∈ g.cells) ∧
  (∀ c ∈ g.cells, ∃ t ∈ tiling, c ∈ t.cells) ∧
  (∀ t1 t2, t1 ∈ tiling → t2 ∈ tiling → t1 ≠ t2 → t1.cells ∩ t2.cells = ∅)

/-- The main theorem: any 2^n x 2^n grid with one cell removed can be tiled with L-shaped pieces -/
theorem grid_tiling (n : ℕ) (g : Grid n) : ∃ tiling : List LTile, is_valid_tiling n g tiling := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_tiling_l739_73918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_sum_m_n_l739_73995

/-- The number of adults --/
def n : ℕ := 15

/-- The probability of a valid pairing --/
def valid_pairing_probability : ℚ := 1 / n + 2 / n.factorial

/-- Theorem stating the probability of valid pairings --/
theorem shoe_pairing_probability :
  (∀ k < 4, ¬ ∃ (pairs : Finset (Fin n × Fin n)),
    pairs.card = k ∧
    (∀ (i j : Fin n), (i, j) ∈ pairs → i ≠ j) ∧
    (∃ (adults : Finset (Fin n)), adults.card = k ∧
      (∀ (i : Fin n), i ∈ adults ↔ ∃ j, (i, j) ∈ pairs ∨ (j, i) ∈ pairs))) =
  (valid_pairing_probability = valid_pairing_probability) := by
  sorry

/-- The sum of m and n in the fraction m/n --/
def m_plus_n : ℕ := 17

/-- Theorem stating that m + n = 17 --/
theorem sum_m_n : m_plus_n = 17 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_pairing_probability_sum_m_n_l739_73995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denom_preserving_function_form_l739_73988

/-- A function that preserves denominators of rational numbers -/
def DenomPreserving (f : ℝ → ℝ) : Prop :=
  ∀ q : ℚ, ∃ a b : ℤ, (b > 0) ∧ (q.num.gcd q.den = 1) ∧ (q = a / b) ∧ (f q = (a : ℝ) / b)

/-- The main theorem -/
theorem denom_preserving_function_form
  (f : ℝ → ℝ) (hf : ContDiff ℝ 1 f) (hd : DenomPreserving f) :
  ∃ k n : ℤ, ∀ x : ℝ, f x = k * x + n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denom_preserving_function_form_l739_73988
