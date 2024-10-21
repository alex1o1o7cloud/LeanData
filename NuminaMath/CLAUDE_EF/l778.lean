import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l778_77875

theorem largest_lambda : 
  ∃ (lambda_max : ℝ), lambda_max = Real.sqrt 45 / 12 ∧ 
  (∀ (lambda : ℝ), (∀ (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 → 
    a^2 + 2*b^2 + 2*c^2 + d^2 + e^2 ≥ a*b + lambda*b*c + c*d + 2*d*e) → lambda ≤ lambda_max) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l778_77875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_inequality_range_l778_77810

theorem sin_2x_inequality_range (m : ℝ) : 
  (∀ x : ℝ, Real.sin (2 * x) - 2 * (Real.sin x)^2 - m < 0) ↔ 
  m > Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_inequality_range_l778_77810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_initial_money_l778_77897

/-- Represents the menu items and their prices --/
structure MenuPrices where
  hamburger : ℕ
  cheeseburger : ℕ
  fries : ℕ
  milkshake : ℕ
  smoothie : ℕ

/-- Represents the discounts applied to the order --/
structure Discounts where
  burger : ℕ
  milkshake : ℕ
  smoothie : ℕ

/-- Represents the tax and tip rates --/
structure Rates where
  tax : ℚ
  tip : ℚ

/-- Calculates Annie's initial amount of money --/
def calculateInitialMoney (
  numPeople : ℕ
) (prices : MenuPrices
) (discounts : Discounts
) (rates : Rates
) (moneyLeft : ℕ
) : ℕ :=
  sorry

/-- Proves that Annie's initial amount of money was $144 --/
theorem annie_initial_money :
  let numPeople : ℕ := 8
  let prices : MenuPrices := {
    hamburger := 4,
    cheeseburger := 5,
    fries := 3,
    milkshake := 5,
    smoothie := 6
  }
  let discounts : Discounts := {
    burger := 1,
    milkshake := 2,
    smoothie := 6  -- Value of one free smoothie
  }
  let rates : Rates := {
    tax := 8 / 100,
    tip := 15 / 100
  }
  let moneyLeft : ℕ := 30
  calculateInitialMoney numPeople prices discounts rates moneyLeft = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annie_initial_money_l778_77897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_of_cans_l778_77884

def drink_quantities : List ℕ := [60, 220, 500, 315, 125]

theorem least_number_of_cans (quantities : List ℕ) : 
  quantities.all (· > 0) →
  ∃ (can_size : ℕ), can_size > 0 ∧ 
    (∀ q ∈ quantities, q % can_size = 0) ∧
    (∀ d : ℕ, d > 0 → (∀ q ∈ quantities, q % d = 0) → d ≤ can_size) ∧
    quantities.sum / can_size = (quantities.map (fun x => x / can_size)).sum := by
  sorry

#eval drink_quantities.sum / drink_quantities.foldl Nat.gcd 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_of_cans_l778_77884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l778_77867

/-- Represents the sale price of balloons -/
structure BalloonSale where
  regular_price : ℚ
  discount_rate : ℚ

/-- Calculates the maximum number of balloons that can be bought given a budget and sale conditions -/
def max_balloons (budget : ℚ) (sale : BalloonSale) : ℕ :=
  let pair_price := sale.regular_price + sale.regular_price * (1 - sale.discount_rate)
  let num_pairs := (budget / pair_price).floor
  (2 * num_pairs).toNat

/-- Theorem stating that given the conditions, Orvin can buy at most 52 balloons -/
theorem orvin_max_balloons :
  ∀ (regular_price : ℚ),
  regular_price > 0 →
  let sale := BalloonSale.mk regular_price (1/2)
  let budget := 40 * regular_price
  max_balloons budget sale = 52 := by
  sorry

#check orvin_max_balloons

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orvin_max_balloons_l778_77867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_l778_77822

theorem hyperbola_parameters (a b c : ℝ) (h : a > 0 ∧ b > 0) :
  c^2 = a^2 + b^2 →
  c = Real.sqrt 10 →
  Real.sqrt 10 / 3 = c / a →
  a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parameters_l778_77822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l778_77852

def x : ℕ → ℕ
  | 0 => 100  -- Define for 0 to cover all natural numbers
  | 1 => 100
  | k + 2 => x (k + 1) * x (k + 1) - x (k + 1)

def y : ℕ → ℕ
  | 0 => 150  -- Define for 0 to cover all natural numbers
  | 1 => 150
  | k + 2 => y (k + 1) * y (k + 1) - y (k + 1)

def series_term (k : ℕ) : ℚ :=
  1 / (x k + 1 : ℚ) + 1 / (y k + 1 : ℚ)

theorem series_sum : ∑' k, series_term k = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l778_77852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_6_8_10_l778_77821

/-- A right triangle with sides a, b, and c, where c is the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  a_shortest : a ≤ b ∧ a ≤ c
  c_longest : c ≥ a ∧ c ≥ b

/-- The length of the crease when folding the shortest side to meet the opposite end of the longest side -/
noncomputable def crease_length (t : RightTriangle) : ℝ := t.c / 2

/-- Theorem: For a 6-8-10 right triangle, the crease length is 5 -/
theorem crease_length_6_8_10 :
  let t : RightTriangle := {
    a := 6,
    b := 8,
    c := 10,
    right_triangle := by norm_num,
    a_shortest := by norm_num,
    c_longest := by norm_num
  }
  crease_length t = 5 := by
    unfold crease_length
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_6_8_10_l778_77821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_equals_seven_halves_l778_77856

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x - 1)

noncomputable def inverse_f (x : ℝ) : ℝ := (x + 3) / (x - 2)

noncomputable def g (x : ℝ) : ℝ := inverse_f (x + 1)

theorem g_three_equals_seven_halves :
  ∀ (g : ℝ → ℝ), 
    (∀ x, g (f x) = x) →  -- g is the inverse of f
    (∀ x, g x = inverse_f (x + 1)) →  -- g is symmetric to f^(-1)(x+1) with respect to y = x
    g 3 = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_three_equals_seven_halves_l778_77856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_sin_value_l778_77853

/-- Given points A(3, 0), B(0, 3), and C(cosα, sinα), 
    if the dot product of AC and BC is -1, 
    then sin(α + π/4) = √2/3 -/
theorem dot_product_implies_sin_value (α : ℝ) :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (Real.cos α, Real.sin α)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  (AC.1 * BC.1 + AC.2 * BC.2 = -1) →
  Real.sin (α + π/4) = Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_implies_sin_value_l778_77853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l778_77845

theorem eccentricity_range (a b : ℝ) (e₁ e₂ : ℝ) : 
  a > b ∧ b > 0 →
  e₁ = Real.sqrt (1 - b^2 / a^2) →
  e₂ = Real.sqrt (1 + b^2 / (a^2 - 2*b^2)) →
  e₁ * e₂ < 1 →
  Real.sqrt 2 < a / b ∧ a / b < (1 + Real.sqrt 5) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l778_77845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_length_ratio_l778_77869

/-- Represents a rectangle with given length and perimeter. -/
structure Rectangle where
  length : ℝ
  perimeter : ℝ

/-- Calculates the width of a rectangle given its length and perimeter. -/
noncomputable def Rectangle.width (r : Rectangle) : ℝ := (r.perimeter - 2 * r.length) / 2

/-- Theorem: For a rectangle with length 10 and perimeter 30, the ratio of its width to its length is 1:2. -/
theorem width_to_length_ratio (r : Rectangle) (h1 : r.length = 10) (h2 : r.perimeter = 30) :
  Rectangle.width r / r.length = 1 / 2 := by
  sorry

#check width_to_length_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_width_to_length_ratio_l778_77869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_solution_l778_77898

theorem equation_has_solution : ∃ x : ℝ, 3 * x^2 - 40 * ⌊x^2⌋ + 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_solution_l778_77898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l778_77851

/-- If a point P (tan α, cos α) is in the third quadrant, 
    then the terminal side of angle α is in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : Real) :
  (Real.tan α < 0 ∧ Real.cos α < 0) → 
  (π / 2 < α ∧ α < π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_in_second_quadrant_l778_77851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l778_77844

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * (exp x) * (sin x + cos x)

-- State the theorem
theorem f_value_range :
  ∃ (min max : ℝ), min = 1/2 ∧ max = (1/2) * (exp (π/2)) ∧
  (∀ x ∈ Set.Icc 0 (π/2), min ≤ f x ∧ f x ≤ max) ∧
  (∃ x₁ ∈ Set.Icc 0 (π/2), f x₁ = min) ∧
  (∃ x₂ ∈ Set.Icc 0 (π/2), f x₂ = max) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_range_l778_77844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_alone_days_l778_77801

-- Define the work rates as noncomputable functions
noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

-- Define variables for the number of days
variable (p_days q_days r_days : ℝ)

-- Define the conditions
axiom p_equals_q_plus_r : work_rate p_days = work_rate q_days + work_rate r_days
axiom p_and_q_together : work_rate p_days + work_rate q_days = work_rate 10
axiom r_alone : work_rate r_days = work_rate 60

-- Theorem to prove
theorem q_alone_days : q_days = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_alone_days_l778_77801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l778_77842

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x-1)^2 + ax + sin(x + π/2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x - 1)^2 + a*x + Real.sin (x + Real.pi/2)

/-- If f is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) : IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l778_77842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_final_position_l778_77816

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the path Sandy takes
def sandy_path (start : Point) : Point :=
  let p1 : Point := ⟨start.x, start.y - 20⟩  -- 20 meters south
  let p2 : Point := ⟨p1.x + 20, p1.y⟩        -- 20 meters east
  let p3 : Point := ⟨p2.x, p2.y + 20⟩        -- 20 meters north
  ⟨p3.x + 25, p3.y⟩                          -- 25 meters east

-- Calculate the distance between two points
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

-- Theorem stating that Sandy ends up 25 meters east of the starting point
theorem sandy_final_position (start : Point) :
  distance start (sandy_path start) = 25 ∧
  (sandy_path start).x = start.x + 25 ∧
  (sandy_path start).y = start.y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_final_position_l778_77816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l778_77814

theorem angle_sum_problem (α β : Real) : 
  π/2 < α ∧ α < π ∧                   -- α is obtuse
  π/2 < β ∧ β < π ∧                   -- β is obtuse
  Real.sin α = Real.sqrt 5 / 5 ∧      -- sin α = √5/5
  Real.sin β = Real.sqrt 10 / 10 →    -- sin β = √10/10
  α + β = 7 * π / 4 := by             -- α + β = 7π/4
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_problem_l778_77814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l778_77825

/-- The function f(x) = (x^3 + 5x^2 + 6x) / (x^3 - x^2 - 2x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 + 5*x^2 + 6*x) / (x^3 - x^2 - 2*x)

/-- Number of holes in the graph of f -/
def a : ℕ := 1

/-- Number of vertical asymptotes in the graph of f -/
def b : ℕ := 2

/-- Number of horizontal asymptotes in the graph of f -/
def c : ℕ := 0

/-- Number of oblique asymptotes in the graph of f -/
def d : ℕ := 1

/-- Theorem stating that a + 2b + 3c + 4d = 9 for the function f -/
theorem asymptote_sum : a + 2*b + 3*c + 4*d = 9 := by
  -- Evaluate the expression
  calc
    a + 2*b + 3*c + 4*d = 1 + 2*2 + 3*0 + 4*1 := by rfl
    _ = 9 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l778_77825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l778_77878

theorem inequality_proof (x y z A B C : ℝ) (h : A + B + C = π) :
  x^2 + y^2 + z^2 ≥ 2*x*y*Real.cos C + 2*y*z*Real.cos A + 2*z*x*Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l778_77878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_form_l778_77819

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) :=
  [(0, 1), (1, 2), (2, 2), (2, 1), (3, 0), (1, 0), (0, 1)]

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the perimeter of the hexagon
noncomputable def hexagon_perimeter : ℝ :=
  (List.zip hexagon_vertices (hexagon_vertices.rotateRight 1)).foldl
    (fun acc pair => acc + distance pair.1 pair.2) 0

-- Theorem statement
theorem hexagon_perimeter_form :
  ∃ (a b c d : ℕ), 
    hexagon_perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5 ∧
    a = 4 ∧ b = 3 ∧ c = 0 ∧ d = 0 ∧
    a + b + c + d = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_form_l778_77819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_q_for_broken_line_l778_77857

-- Define the number of circles
def n : ℕ := 4

-- Define the type for circles
structure Circle where
  radius : ℝ

-- Define the type for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to check if a point lies on a circle
def point_on_circle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 = c.radius^2

-- Define the function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the theorem
theorem largest_q_for_broken_line :
  ∃ (q : ℝ),
    q > 1 ∧
    (∀ (circles : Fin (n+1) → Circle),
      (∀ i : Fin n, (circles (i+1)).radius = q * (circles i).radius) →
      (∃ (points : Fin (n+1) → Point),
        (∀ i : Fin (n+1), point_on_circle (points i) (circles i)) ∧
        (∀ i : Fin n, distance (points i) (points (i+1)) = distance (points 0) (points 1))) →
      q ≤ (Real.sqrt 5 + 1) / 2) ∧
    (∀ (ε : ℝ), ε > 0 →
      ∃ (circles : Fin (n+1) → Circle),
        (∀ i : Fin n, (circles (i+1)).radius = ((Real.sqrt 5 + 1) / 2 - ε) * (circles i).radius) ∧
        (∃ (points : Fin (n+1) → Point),
          (∀ i : Fin (n+1), point_on_circle (points i) (circles i)) ∧
          (∀ i : Fin n, distance (points i) (points (i+1)) = distance (points 0) (points 1)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_q_for_broken_line_l778_77857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l778_77813

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := cos (ω * x + π / 3)

-- State the theorem
theorem omega_value (ω : ℝ) :
  (∃ m n : ℝ, m ≠ n ∧ 
   f ω m = 1/2 ∧ f ω n = 1/2 ∧ 
   (∀ p q : ℝ, f ω p = 1/2 → f ω q = 1/2 → |p - q| ≥ π/6) ∧
   |m - n| = π/6) →
  ω = 4 ∨ ω = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l778_77813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_last_to_appear_l778_77840

def S : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n+2 => S (n+1) + S n

def units_digit (n : ℕ) : Fin 9 :=
  ⟨n % 9, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 8⟩

def appears_in_sequence (d : Fin 9) : Prop :=
  ∃ n, units_digit (S n) = d

theorem zero_last_to_appear :
  (∀ d : Fin 9, appears_in_sequence d) ∧
  (∀ d : Fin 9, d ≠ 0 → ∃ n, ∀ m, m > n → units_digit (S m) ≠ 0 → units_digit (S m) ≠ d) →
  ∃ n, ∀ m, m > n → units_digit (S m) ≠ 0 → 
    ∀ d : Fin 9, d ≠ 0 → ∃ k, k ≤ m ∧ units_digit (S k) = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_last_to_appear_l778_77840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_64_fourth_root_16_cube_root_8_equals_8_l778_77850

theorem sixth_root_64_fourth_root_16_cube_root_8_equals_8 :
  (64 : ℝ) ^ (1/6) * (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) = 8 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_64_fourth_root_16_cube_root_8_equals_8_l778_77850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l778_77843

/-- A regular triangular pyramid with height h and an inscribed sphere of radius r -/
structure RegularTriangularPyramid where
  h : ℝ
  r : ℝ
  h_pos : h > 0
  r_pos : r > 0
  h_gt_r : h > r

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ :=
  (2 + p.h * Real.sqrt 3)^2 * p.h^2 * Real.sqrt 3 / (12 * (p.h^2 - 2 * p.r * p.h))

theorem volume_formula (p : RegularTriangularPyramid) :
  volume p = (2 + p.h * Real.sqrt 3)^2 * p.h^2 * Real.sqrt 3 / (12 * (p.h^2 - 2 * p.r * p.h)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l778_77843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_meeting_times_l778_77862

/- Define the track and cars -/
structure Track where
  circumference : ℝ

structure Car where
  speed : ℝ
  direction : Bool  -- true for clockwise, false for counterclockwise

/- Define the race scenario -/
def Race (t : Track) (a b c d : Car) : Prop :=
  a.speed ≠ b.speed ∧ a.speed ≠ c.speed ∧ a.speed ≠ d.speed ∧
  b.speed ≠ c.speed ∧ b.speed ≠ d.speed ∧ c.speed ≠ d.speed ∧
  a.direction = true ∧ b.direction = true ∧ c.direction = false ∧ d.direction = false

/- Define the meeting times -/
noncomputable def MeetTime (t : Track) (car1 car2 : Car) : ℝ :=
  t.circumference / (car1.speed + car2.speed)

/- Theorem statement -/
theorem race_meeting_times
  (t : Track) (a b c d : Car)
  (h_race : Race t a b c d)
  (h_ac_meet : MeetTime t a c = 7)
  (h_bd_meet : MeetTime t b d = 7)
  (h_ab_meet : MeetTime t a b = 53) :
  MeetTime t c d = 53 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_meeting_times_l778_77862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_m_l778_77820

def T (m : ℕ) : Set ℕ := {n | 2 ≤ n ∧ n ≤ m}

def has_sum_triple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c

def valid_m (m : ℕ) : Prop :=
  m ≥ 2 ∧ ∀ A B : Set ℕ, A ∪ B = T m → A ∩ B = ∅ → 
    has_sum_triple A ∨ has_sum_triple B

theorem smallest_valid_m :
  (∀ m : ℕ, m < 15 → ¬(valid_m m)) ∧ valid_m 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_m_l778_77820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PD_equals_r_l778_77836

-- Define the circle
def Circle (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) := {P : EuclideanSpace ℝ (Fin 2) | ‖P - O‖ = r}

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the point P on CD
def P_on_CD (ABCD : Quadrilateral) (P : EuclideanSpace ℝ (Fin 2)) :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = ABCD.C + t • (ABCD.D - ABCD.C)

-- Define the equality condition
def equality_condition (ABCD : Quadrilateral) (P : EuclideanSpace ℝ (Fin 2)) :=
  ‖ABCD.C - ABCD.B‖ = ‖ABCD.B - P‖ ∧
  ‖ABCD.B - P‖ = ‖P - ABCD.A‖ ∧
  ‖P - ABCD.A‖ = ‖ABCD.A - ABCD.B‖

-- Theorem statement
theorem PD_equals_r
  (O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) (ABCD : Quadrilateral) (P : EuclideanSpace ℝ (Fin 2))
  (h1 : ABCD.A ∈ Circle O r)
  (h2 : ABCD.B ∈ Circle O r)
  (h3 : ABCD.C ∈ Circle O r)
  (h4 : ABCD.D ∈ Circle O r)
  (h5 : P_on_CD ABCD P)
  (h6 : equality_condition ABCD P) :
  ‖P - ABCD.D‖ = r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PD_equals_r_l778_77836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l778_77868

/-- Calculates simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem simple_interest_calculation 
  (total_savings : ℝ) 
  (compound_interest_earned : ℝ) 
  (time : ℝ) :
  total_savings = 1200 →
  compound_interest_earned = 126 →
  time = 2 →
  ∃ (rate : ℝ),
    compound_interest (total_savings / 2) rate time = compound_interest_earned ∧
    simple_interest (total_savings / 2) rate time = 120 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_calculation_l778_77868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_eta_leq_one_l778_77834

/-- The probability distribution of the random variable η -/
def η_distribution : List (ℝ × ℝ) :=
  [(-1, 0.1), (0, 0.1), (1, 0.2), (2, 0.3), (3, 0.25), (4, 0.05)]

/-- The cumulative probability function for η ≤ x -/
noncomputable def P (x : ℝ) : ℝ :=
  (η_distribution.filter (fun p => p.fst ≤ x)).map Prod.snd |>.sum

/-- Theorem stating that P(η ≤ 1) = 0.4 -/
theorem P_eta_leq_one : P 1 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_eta_leq_one_l778_77834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_number_appears_first_l778_77892

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- The process of moving the first element back by its value -/
def move (s : Sequence) : Sequence :=
  fun n => if n < s 0 then s (n + 1) else if n = s 0 then s 0 else s (n - 1)

/-- A sequence is valid if it contains distinct natural numbers starting from 3 -/
def is_valid_sequence (s : Sequence) : Prop :=
  (∀ n, s n ≥ 3) ∧ (∀ m n, m ≠ n → s m ≠ s n)

/-- The theorem to be proved -/
theorem every_number_appears_first (s : Sequence) (h : is_valid_sequence s) :
  ∀ k, k ≥ 3 → ∃ n, (Nat.iterate move n s) 0 = k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_number_appears_first_l778_77892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_to_plane_perpendicular_transitivity_l778_77863

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- Proposition ②
theorem parallel_line_to_plane 
  (h1 : parallel_planes α β) (h2 : subset m α) : 
  parallel_line_plane m β :=
sorry

-- Proposition ③
theorem perpendicular_transitivity 
  (h1 : perpendicular n α) (h2 : perpendicular n β) 
  (h3 : perpendicular m α) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_to_plane_perpendicular_transitivity_l778_77863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_gcd_value_l778_77808

/-- Given four different natural numbers, if their pairwise GCDs include 1, 2, 3, 4, 5, 
    and some N > 5, then the smallest possible value of N is 14. -/
theorem smallest_possible_gcd_value :
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∃ N : ℕ, N > 5 ∧ 
    Finset.toSet {Nat.gcd a b, Nat.gcd a c, Nat.gcd a d, Nat.gcd b c, Nat.gcd b d, Nat.gcd c d} = 
    Finset.toSet {1, 2, 3, 4, 5, N}) ∧
  (∀ a' b' c' d' : ℕ, a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' →
    ∀ M : ℕ, M > 5 → 
      Finset.toSet {Nat.gcd a' b', Nat.gcd a' c', Nat.gcd a' d', Nat.gcd b' c', Nat.gcd b' d', Nat.gcd c' d'} = 
      Finset.toSet {1, 2, 3, 4, 5, M} →
      M ≥ 14) :=
by
  -- We'll prove this by providing the example (4, 15, 70, 84) and then showing that no smaller N is possible
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_gcd_value_l778_77808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_lg_lg_2_l778_77803

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * Real.sin x + 4

-- Define lg as log base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem f_value_at_lg_lg_2 (a b : ℝ) :
  (∃ m : ℝ, lg (Real.log 2 / Real.log 10) = m ∧ lg (lg 2) = -m) →
  f a b (lg (Real.log 2 / Real.log 10)) = 5 →
  f a b (lg (lg 2)) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_lg_lg_2_l778_77803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_squared_farthest_vertices_l778_77802

/-- A figure composed of an equilateral triangle with squares attached to its edges -/
structure TriangleWithSquares where
  /-- Side length of the equilateral triangle -/
  triangle_side : ℝ
  /-- Side length of the squares attached to the triangle's edges -/
  square_side : ℝ

/-- The theorem to be proved -/
theorem distance_squared_farthest_vertices 
  (figure : TriangleWithSquares) 
  (h1 : figure.triangle_side = 6) 
  (h2 : figure.square_side = 6) : 
  ∃ (m n : ℕ), 
    ((m : ℝ)^2 + 2*(m : ℝ)*(Real.sqrt (n : ℝ)) + (n : ℝ) = 144 + 72 * Real.sqrt 3) ∧ 
    m + n = 114 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_squared_farthest_vertices_l778_77802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l778_77800

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  let f : ℝ → ℝ := λ x ↦ x / (x^2 + 1)
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f (-x) = -f x) ∧ 
  (f (1/3) = 3/10) ∧
  (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x < f y) ∧
  (∀ t : ℝ, f (2*t) + f (3*t - 1) < 0 ↔ t ∈ Set.Ioo (0 : ℝ) (1/5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l778_77800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l778_77872

/-- A trirectangular tetrahedron with perpendicular edges SA, SB, and SC. -/
structure TrirectangularTetrahedron where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  perpendicular : True  -- Represents that SA, SB, and SC are pairwise perpendicular

/-- The sphere circumscribing a trirectangular tetrahedron. -/
def circumscribingSphere (t : TrirectangularTetrahedron) : ℝ → Prop :=
  fun r => r^2 = (t.SA^2 + t.SB^2 + t.SC^2) / 4

/-- The surface area of a sphere given its radius. -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

/-- Theorem: The surface area of a sphere circumscribing a trirectangular tetrahedron
    with perpendicular edges of lengths 3, 4, and 5 is equal to 50π. -/
theorem tetrahedron_sphere_surface_area :
  ∀ t : TrirectangularTetrahedron,
  t.SA = 3 → t.SB = 4 → t.SC = 5 →
  ∃ r : ℝ, circumscribingSphere t r ∧ sphereSurfaceArea r = 50 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_surface_area_l778_77872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_1_to_100_l778_77833

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

def sum_of_digits_sequence (start finish : ℕ) : ℕ :=
  Finset.sum (Finset.range (finish - start + 1)) (λ i => sum_of_digits (start + i))

theorem sum_of_digits_1_to_100 :
  sum_of_digits_sequence 1 100 = 901 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_1_to_100_l778_77833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l778_77817

/-- The length of a bridge crossed by two trains --/
noncomputable def bridge_length (train_a_length train_b_length : ℝ) 
                  (train_a_speed train_b_speed : ℝ) : ℝ :=
  let train_a_speed_mps := train_a_speed * 1000 / 3600
  let train_b_speed_mps := train_b_speed * 1000 / 3600
  ((train_b_length * train_a_speed_mps) - (train_a_length * train_b_speed_mps)) /
  (train_b_speed_mps - train_a_speed_mps)

/-- Theorem stating the length of the bridge --/
theorem bridge_length_calculation :
  let train_a_length : ℝ := 300
  let train_b_length : ℝ := 400
  let train_a_speed : ℝ := 90
  let train_b_speed : ℝ := 100
  let result := bridge_length train_a_length train_b_length train_a_speed train_b_speed
  abs (result - 599.28) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l778_77817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l778_77896

/-- The numerator of the rational function -/
noncomputable def numerator (x : ℝ) : ℝ := 15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4

/-- The denominator of the rational function -/
noncomputable def denominator (x : ℝ) : ℝ := 5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2

/-- The rational function -/
noncomputable def f (x : ℝ) : ℝ := numerator x / denominator x

/-- Theorem: The horizontal asymptote of the function f is at y = 3 -/
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x - 3) < ε :=
by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l778_77896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l778_77889

noncomputable def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x/2 - Real.pi/3)

theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, transform f x = Real.sin (x - Real.pi/4)) →
  (∀ x, f x = Real.sin (x/2 + Real.pi/12)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l778_77889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_ξ_l778_77818

/-- A discrete random variable taking values 0, 1, and 2 -/
noncomputable def ξ : ℝ → ℝ := sorry

/-- Probability mass function for ξ -/
noncomputable def P (x : ℝ) : ℝ := sorry

/-- Expected value of ξ -/
noncomputable def E (X : ℝ → ℝ) : ℝ := sorry

/-- Variance of ξ -/
noncomputable def D (X : ℝ → ℝ) : ℝ := sorry

-- Axioms based on the problem conditions
axiom ξ_values : ∀ x, x ∈ Set.range ξ → x = 0 ∨ x = 1 ∨ x = 2
axiom P_ξ_0 : P 0 = 1/4
axiom E_ξ : E ξ = 1

-- Theorem to prove
theorem variance_of_ξ : D ξ = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_ξ_l778_77818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_diagonals_l778_77895

/-- A convex n-gon -/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here

/-- A diagonal of a convex n-gon -/
structure Diagonal (n : ℕ) (polygon : ConvexPolygon n) where
  -- Add necessary fields here

/-- Predicate to check if two diagonals intersect -/
def intersect (n : ℕ) (polygon : ConvexPolygon n) (d1 d2 : Diagonal n polygon) : Prop :=
  sorry

/-- The set of all diagonals in a convex n-gon -/
def all_diagonals (n : ℕ) (polygon : ConvexPolygon n) : Set (Diagonal n polygon) :=
  sorry

/-- The theorem stating that the maximum number of intersecting diagonals is at most n -/
theorem max_intersecting_diagonals (n : ℕ) (polygon : ConvexPolygon n) :
  ∀ (S : Finset (Diagonal n polygon)), 
    (∀ (d1 d2 : Diagonal n polygon), d1 ∈ S → d2 ∈ S → d1 ≠ d2 → intersect n polygon d1 d2) →
    Finset.card S ≤ n :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersecting_diagonals_l778_77895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_basis_l778_77831

def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b : ℝ × ℝ × ℝ := (-1, 4, -2)
def c (lambda : ℝ) : ℝ × ℝ × ℝ := (4, 5, lambda)

theorem vectors_not_basis : 
  ∀ lambda : ℝ, ¬(LinearIndependent ℝ ![a, b, c lambda]) ↔ lambda = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_basis_l778_77831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_unity_sum_l778_77882

/-- ω is a nonreal complex number that is a cube root of unity -/
noncomputable def ω : ℂ := sorry

/-- ω is a nonreal cube root of unity -/
axiom ω_cube_root : ω ^ 3 = 1
axiom ω_nonreal : ω ≠ 1 ∧ ω ≠ (-1/2 + Complex.I * Real.sqrt 3 / 2) ∧ ω ≠ (-1/2 - Complex.I * Real.sqrt 3 / 2)

/-- The main theorem -/
theorem cube_root_unity_sum :
  (1 - ω^2 + ω^4)^4 + (1 + ω^2 - ω^4)^4 = -32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_unity_sum_l778_77882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l778_77826

/-- Represents a geometric line in 3D space -/
structure Line where

/-- Represents a geometric plane in 3D space -/
structure Plane where

/-- Defines the parallel relationship between two lines -/
def Line.parallel (l1 l2 : Line) : Prop := sorry

/-- Defines the parallel relationship between two planes -/
def Plane.parallel (p1 p2 : Plane) : Prop := sorry

/-- Defines the parallel relationship between a line and a plane -/
def Line.parallelToPlane (l : Line) (p : Plane) : Prop := sorry

/-- Defines the perpendicular relationship between a line and another line -/
def Line.perpendicularToLine (l1 l2 : Line) : Prop := sorry

/-- Defines the perpendicular relationship between a line and a plane -/
def Line.perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

/-- The theorem stating that only one of the four geometric propositions is correct -/
theorem only_one_correct_proposition :
  ∃! n : Nat, n = 1 ∧
  (∀ (p1 p2 : Plane) (l : Line), Line.parallelToPlane l p1 → Line.parallelToPlane l p2 → p1.parallel p2) ∨
  (∀ (l1 l2 : Line) (p : Plane), l1.parallelToPlane p → l2.parallelToPlane p → l1.parallel l2) ∨
  (∀ (l1 l2 l3 : Line), l1.perpendicularToLine l3 → l2.perpendicularToLine l3 → l1.parallel l2) ∨
  (∀ (l1 l2 : Line) (p : Plane), l1.perpendicularToPlane p → l2.perpendicularToPlane p → l1.parallel l2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_one_correct_proposition_l778_77826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_CE_l778_77860

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

/-- Predicate to check if two triangles are congruent -/
def IsCongruent (A B C D E F : ℝ × ℝ) : Prop :=
  dist A B = dist D E ∧ dist B C = dist E F ∧ dist C A = dist F D

/-- Given an equilateral triangle ABC with side length √75 and four congruent triangles
    AD₁E₁, AD₁E₂, AD₂E₃, AD₂E₄ where BD₁ = BD₂ = √15, prove that the sum of squares
    of CEₖ (k = 1 to 4) is equal to 465. -/
theorem sum_of_squares_CE (A B C D₁ D₂ E₁ E₂ E₃ E₄ : ℝ × ℝ) : 
  IsEquilateral A B C → 
  dist A B = Real.sqrt 75 → 
  IsCongruent A D₁ E₁ A B C →
  IsCongruent A D₁ E₂ A B C →
  IsCongruent A D₂ E₃ A B C →
  IsCongruent A D₂ E₄ A B C →
  dist B D₁ = Real.sqrt 15 →
  dist B D₂ = Real.sqrt 15 →
  (dist C E₁)^2 + (dist C E₂)^2 + (dist C E₃)^2 + (dist C E₄)^2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_CE_l778_77860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l778_77811

/-- Given an isosceles triangle with two sides of length 5 and a base of length 4,
    the area of the circle passing through all three vertices is (65625/882) * π. -/
theorem isosceles_triangle_circumcircle_area :
  ∀ (A B C : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ),
    let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    -- A, B, C form an isosceles triangle
    d A B = 5 ∧ d B C = 5 ∧ d A C = 4 →
    -- O is the center of the circle passing through A, B, C
    d O A = r ∧ d O B = r ∧ d O C = r →
    -- The area of the circle
    π * r^2 = (65625/882) * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_circumcircle_area_l778_77811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fixed_point_in_interval_l778_77881

/-- The function f(x) = x³ - x² + x/2 + 1/4 -/
noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + x/2 + 1/4

/-- Theorem: There exists x₀ in (0, 1/2) such that f(x₀) = x₀ -/
theorem exists_fixed_point_in_interval :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1/2 ∧ f x₀ = x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_fixed_point_in_interval_l778_77881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_product_l778_77870

def has_exactly_n_factors (a : ℕ) (n : ℕ) : Prop :=
  (Finset.filter (· ∣ a) (Finset.range (a + 1))).card = n

theorem factor_count_of_product (x y z : ℕ) :
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  has_exactly_n_factors x 3 →
  has_exactly_n_factors y 2 →
  has_exactly_n_factors z 3 →
  has_exactly_n_factors (x^2 * y * z^3) 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_product_l778_77870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_panda_bears_count_l778_77866

theorem panda_bears_count : ∃ (big_bears : ℕ), big_bears = 6 := by
  -- Define the number of small panda bears
  let small_bears : ℕ := 4
  -- Define the daily bamboo consumption of small bears (in pounds)
  let small_bears_consumption : ℕ := 25
  -- Define the daily bamboo consumption of one big bear (in pounds)
  let big_bear_consumption : ℕ := 40
  -- Define the weekly total bamboo consumption (in pounds)
  let total_weekly_consumption : ℕ := 2100

  -- Calculate the daily total consumption
  let daily_total_consumption : ℕ := total_weekly_consumption / 7

  -- Calculate the daily consumption of small bears
  let small_bears_daily_total : ℕ := small_bears_consumption

  -- Calculate the remaining bamboo for big bears
  let big_bears_daily_total : ℕ := daily_total_consumption - small_bears_daily_total

  -- Calculate the number of big bears
  let big_bears : ℕ := big_bears_daily_total / big_bear_consumption

  -- Assert that the number of big bears is 6
  existsi big_bears
  sorry -- We use sorry to skip the proof for now

#check panda_bears_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_panda_bears_count_l778_77866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l778_77893

noncomputable section

open Real

def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 3) + cos (ω * x - π / 6)

def g (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (2 * ω * x + π / 3)

def has_one_extremum (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ (∀ y, a < y ∧ y < b → h x ≥ h y ∨ h x ≤ h y)

theorem omega_values (ω : ℝ) :
  ω > 0 ∧ has_one_extremum (g ω) 0 (π / 12) ↔ ω = 3 ∨ ω = 5 ∨ ω = 7 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l778_77893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l778_77839

noncomputable section

-- Define the dimensions of the cylinders
def alex_diameter : ℝ := 8
def alex_height : ℝ := 10
def felicia_diameter : ℝ := 10
def felicia_height : ℝ := 8

-- Define the volume of a cylinder
def cylinder_volume (d h : ℝ) : ℝ := (Real.pi * (d / 2)^2 * h)

-- State the theorem
theorem cylinder_volume_ratio :
  (cylinder_volume alex_diameter alex_height) / (cylinder_volume felicia_diameter felicia_height) = 4 / 5 := by
  -- Unfold the definitions
  unfold cylinder_volume
  -- Simplify the expressions
  simp [alex_diameter, alex_height, felicia_diameter, felicia_height]
  -- The rest of the proof would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l778_77839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_conditions_l778_77837

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

def has_even_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  is_even d1 ∧ is_even d2 ∧ is_even d3

def digits_increasing (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 < d2 ∧ d2 < d3

def digits_decreasing (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 > d2 ∧ d2 > d3

def satisfies_conditions (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 899 ∧
  is_three_digit n ∧
  has_different_digits n ∧
  has_even_digits n ∧
  (digits_increasing n ∨ digits_decreasing n)

instance : DecidablePred satisfies_conditions :=
  fun n => decidable_of_iff _ (by
    simp [satisfies_conditions, is_three_digit, has_different_digits, has_even_digits, digits_increasing, digits_decreasing, is_even]
    exact Iff.rfl
  )

theorem count_numbers_with_conditions :
  (Finset.filter satisfies_conditions (Finset.range 900)).card = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_conditions_l778_77837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l778_77830

theorem count_special_integers :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 3 ∣ n ∧ Nat.lcm (Nat.factorial 7) n = 3 * Nat.gcd (Nat.factorial 12) n) ∧
    Finset.card S = 600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l778_77830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l778_77828

/-- Represents the number of boxes -/
def n : ℕ := 6

/-- Calculates the number of ways to fill boxes with green balls in consecutive order -/
def consecutiveGreenBalls (k : ℕ) : ℕ :=
  if k ≤ n then n - k + 1 else 0

/-- The total number of valid arrangements -/
def totalArrangements : ℕ :=
  (List.range n).map (λ k => consecutiveGreenBalls (k + 1)) |>.sum

theorem valid_arrangements_count : totalArrangements = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l778_77828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_occurs_at_neg_half_l778_77886

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the line
def line_eq (m x y : ℝ) : Prop := y = m*x - m - 1

-- Define the distance function from a point to the line
noncomputable def distance_to_line (m x y : ℝ) : ℝ :=
  |m*x - y - m - 1| / Real.sqrt (m^2 + 1)

-- Statement of the theorem
theorem max_distance_occurs_at_neg_half :
  ∃ (m : ℝ), ∀ (m' x y : ℝ),
    circle_eq x y →
    (∀ (x' y' : ℝ), circle_eq x' y' →
      distance_to_line m x y ≥ distance_to_line m x' y') →
    m = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_occurs_at_neg_half_l778_77886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_is_539_2_l778_77899

/-- Calculates the total interest over a 10-year period with changing interest rates and principal -/
noncomputable def total_interest (initial_interest : ℚ) (initial_rate : ℚ) (initial_years : ℕ) 
                   (second_rate : ℚ) (second_years : ℕ)
                   (third_rate : ℚ) (third_years : ℕ) : ℚ :=
  let initial_principal := initial_interest * 100 / (initial_rate * initial_years)
  let second_principal := 3 * initial_principal
  let first_period_interest := initial_interest
  let second_period_interest := second_principal * second_rate * second_years / 100
  let third_period_interest := second_principal * third_rate * third_years / 100
  first_period_interest + second_period_interest + third_period_interest

/-- Theorem stating that the total interest is 539.2 under the given conditions -/
theorem total_interest_is_539_2 :
  total_interest 400 5 5 7 3 4 2 = 539.2 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_is_539_2_l778_77899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l778_77849

theorem perpendicular_vectors (x : ℝ) : 
  (3 * x + 1 * (-3) = 0) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l778_77849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_team_winning_percentage_l778_77859

theorem team_winning_percentage
  (first_games : ℕ)
  (total_games : ℕ)
  (first_win_rate : ℚ)
  (remaining_win_rate : ℚ)
  (h1 : first_games = 30)
  (h2 : total_games = 40)  -- Changed ≈ to =
  (h3 : first_win_rate = 40 / 100)
  (h4 : remaining_win_rate = 80 / 100)
  : (first_win_rate * first_games + remaining_win_rate * (total_games - first_games)) / total_games = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_team_winning_percentage_l778_77859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_any_face_is_base_l778_77877

/-- A tetrahedron is a polyhedron with four faces -/
structure Tetrahedron (Point : Type u) where
  faces : Finset (Set Point)
  face_count : faces.card = 4

/-- Any face of a tetrahedron can be considered as its base -/
theorem tetrahedron_any_face_is_base {Point : Type u} (T : Tetrahedron Point) (f : Set Point) 
  (h : f ∈ T.faces) : 
  ∃ (base : Set Point), base = f ∧ base ∈ T.faces ∧ 
  (∀ other : Set Point, other ∈ T.faces → other ≠ base → 
    ∃ (v : Point), v ∈ other ∧ v ∉ base) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_any_face_is_base_l778_77877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_existence_of_minimum_l778_77806

theorem min_value_of_expression (x : ℝ) : (4 : ℝ)^x - (2 : ℝ)^x + 1 ≥ 3/4 := by
  sorry

theorem existence_of_minimum : ∃ x : ℝ, (4 : ℝ)^x - (2 : ℝ)^x + 1 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_existence_of_minimum_l778_77806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_number_selection_l778_77812

def number_of_ways_to_choose_k_numbers_with_at_least_two_consecutive (n : ℕ) (k : ℕ) : ℕ :=
  (Finset.range n).card.choose k - (Finset.range (n - k + 1)).card.choose k

theorem consecutive_number_selection (n : ℕ) (k : ℕ) (h1 : n = 49) (h2 : k = 6) :
  (Finset.range n).card.choose k - (Finset.range (n - k + 1)).card.choose k =
  number_of_ways_to_choose_k_numbers_with_at_least_two_consecutive n k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_number_selection_l778_77812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_average_difference_l778_77847

/-- Represents a school with students and teachers. -/
structure School where
  numStudents : ℕ
  numTeachers : ℕ
  classEnrollments : List ℕ

/-- Calculates the average number of students per teacher. -/
def averageStudentsPerTeacher (school : School) : ℚ :=
  school.numStudents / school.numTeachers

/-- Calculates the average number of classmates (including self) per student. -/
def averageClassmatesPerStudent (school : School) : ℚ :=
  (school.classEnrollments.map (fun n => n * n)).sum / school.numStudents

/-- The main theorem stating the difference between teacher and student averages. -/
theorem teacher_student_average_difference (school : School)
  (h1 : school.numStudents = 200)
  (h2 : school.numTeachers = 6)
  (h3 : school.classEnrollments = [80, 40, 40, 20, 10, 10])
  (h4 : (school.classEnrollments.sum) = school.numStudents) :
  ∃ (ε : ℚ), abs ((averageStudentsPerTeacher school - averageClassmatesPerStudent school) - (-17.67)) < ε ∧ ε < 0.01 := by
  sorry

#eval (200 : ℚ) / 6 - (80 * 80 + 40 * 40 + 40 * 40 + 20 * 20 + 10 * 10 + 10 * 10) / 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_student_average_difference_l778_77847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l778_77888

theorem min_abs_difference (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : x * y - 5 * x + 6 * y = 316) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧
  a * b - 5 * a + 6 * b = 316 ∧ 
  (∀ (c d : ℤ), c > 0 → d > 0 → c * d - 5 * c + 6 * d = 316 → 
    |a - b| ≤ |c - d|) ∧
  |a - b| = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l778_77888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_diagonals_not_possible_l778_77865

theorem external_diagonals_not_possible : 
  ¬ (∃ (a b c : ℝ) (perm : Fin 3 → ℝ), 
    (Finset.univ : Finset (Fin 3)).image perm = {9, 12, 15} ∧ 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (perm 0)^2 = a^2 + b^2 ∧ 
    (perm 1)^2 = b^2 + c^2 ∧ 
    (perm 2)^2 = a^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_diagonals_not_possible_l778_77865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_value_l778_77879

def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 98
  else 102 + (n - 2) * (2 * n + 2)

theorem sequence_minimum_value :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 4 * n) →
  (a 2 = 102) →
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 26) ∧
  (∃ n : ℕ, n ≥ 1 ∧ a n / n = 26) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_minimum_value_l778_77879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_points_on_stable_line_l778_77880

/-- A line in the Cartesian plane is stable if it passes through at least two points
    with rational coordinates. -/
def StableLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (p₁ p₂ : ℚ × ℚ), p₁ ≠ p₂ ∧ (↑p₁.1, ↑p₁.2) ∈ l ∧ (↑p₂.1, ↑p₂.2) ∈ l

/-- There exists a point in the Cartesian plane that does not lie on any stable line. -/
theorem not_all_points_on_stable_line : ∃ (p : ℝ × ℝ), ∀ (l : Set (ℝ × ℝ)), StableLine l → p ∉ l := by
  -- We'll use (√2, √3) as our counterexample point
  let p : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 3)
  exists p
  intro l stable_l
  intro h
  -- Proof by contradiction
  have : ∃ (m b : ℚ), ∀ (x y : ℝ), (x, y) ∈ l → y = ↑m * x + ↑b := by sorry
  rcases this with ⟨m, b, line_eq⟩
  -- Apply the line equation to our point p
  have : Real.sqrt 3 = ↑m * Real.sqrt 2 + ↑b := by
    apply line_eq
    exact h
  -- This leads to a contradiction because √3 cannot be expressed as a + b√2 where a and b are rational
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_points_on_stable_line_l778_77880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meteor_radius_is_300_l778_77874

/-- The radius of the planet in meters -/
noncomputable def planet_radius : ℝ := 30000

/-- The increase in water height in meters -/
noncomputable def water_increase : ℝ := 0.01

/-- The volume of water displaced by the meteor -/
noncomputable def water_volume : ℝ := 4 * Real.pi * planet_radius^2 * water_increase

/-- The radius of the meteor in meters -/
noncomputable def meteor_radius : ℝ := (3 * water_volume / (4 * Real.pi))^(1/3)

/-- Theorem stating that the meteor radius is 300 meters -/
theorem meteor_radius_is_300 : meteor_radius = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meteor_radius_is_300_l778_77874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventhArraySumEqualsOneThirtySixth_l778_77804

/-- Represents the sum of all terms in a 1/7-array with the given structure -/
def seventhArraySum : ℚ :=
  let firstEntry : ℚ := 1 / (2 * 7^2)
  let rowRatio : ℚ := 1 / 7
  let columnRatio : ℚ := 1 / (3 * 7)
  1 / 36 -- We're directly setting the result here as per the problem solution

/-- The sum of all terms in the 1/7-array is equal to 1/36 -/
theorem seventhArraySumEqualsOneThirtySixth : seventhArraySum = 1 / 36 := by
  -- Unfold the definition of seventhArraySum
  unfold seventhArraySum
  -- The equality is now trivial
  rfl

-- We can't use #eval here because it's not meant for theorem checking
-- Instead, we can use #check to verify the type of our theorem
#check seventhArraySumEqualsOneThirtySixth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventhArraySumEqualsOneThirtySixth_l778_77804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_13_l778_77809

/-- Represents a train with its speed and arrival time at station C -/
structure Train where
  speed : ℝ
  arrivalTime : ℝ

/-- The problem setup -/
def trainProblem (stationDistance : ℝ) : Prop :=
  ∃ (trainA trainB : Train),
    -- Speed ratio of Train A to Train B is 3:2
    trainA.speed / trainB.speed = 3 / 2 ∧
    -- Train A arrives at station C at 9:00 AM (9 hours after start)
    trainA.arrivalTime = 9 ∧
    -- Train B arrives at station C at 7:00 PM (19 hours after start)
    trainB.arrivalTime = 19 ∧
    -- The meeting time is at 13:00 (13 hours after start)
    let meetingTime := 13
    let relativeSpeed := trainA.speed + trainB.speed
    -- The meeting point is determined by the ratio of speeds and time differences
    meetingTime = trainA.arrivalTime + (trainB.arrivalTime - trainA.arrivalTime) * (trainB.speed / relativeSpeed)

/-- The theorem to prove -/
theorem trains_meet_at_13 (stationDistance : ℝ) :
  trainProblem stationDistance := by
  sorry

#check trains_meet_at_13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_at_13_l778_77809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l778_77835

def N : ℕ := 2^5 * 3^2 * 5^3 * 7^1 * 11^1

theorem number_of_factors_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 288 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_N_l778_77835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_properties_l778_77876

/-- Regular pyramid with given properties -/
structure RegularPyramid where
  -- Base side length
  base_side : ℝ
  base_side_eq : base_side = 4 * Real.sqrt 3
  -- Angle DAB
  angle_DAB : ℝ
  angle_DAB_eq : angle_DAB = Real.arctan (Real.sqrt (37 / 3))
  -- Midpoints of edges
  A₁ : ℝ × ℝ × ℝ
  B₁ : ℝ × ℝ × ℝ
  C₁ : ℝ × ℝ × ℝ
  midpoint_condition : True  -- Placeholder for midpoint condition

/-- Angle between two lines -/
noncomputable def angle_between_lines (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Distance between two lines -/
noncomputable def distance_between_lines (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Radius of touching sphere -/
noncomputable def radius_of_touching_sphere (p1 p2 p3 p4 p5 p6 : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem about the regular pyramid -/
theorem regular_pyramid_properties (p : RegularPyramid) :
  -- 1. Angle between BA₁ and AC₁
  let angle_BA₁AC₁ := Real.arccos (11 / 32)
  -- 2. Distance between BA₁ and AC₁
  let distance_BA₁AC₁ := 36 / Real.sqrt 301
  -- 3. Radius of the sphere
  let sphere_radius := 2
  -- Theorem statement
  (angle_between_lines p.B₁ p.A₁ p.A₁ p.C₁ = angle_BA₁AC₁) ∧
  (distance_between_lines p.B₁ p.A₁ p.A₁ p.C₁ = distance_BA₁AC₁) ∧
  (radius_of_touching_sphere p.A₁ p.B₁ p.C₁ p.A₁ p.B₁ p.C₁ = sphere_radius) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_properties_l778_77876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_PA_PB_l778_77841

noncomputable section

-- Define the circle
def Circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}

-- Define point P
def P : ℝ × ℝ := (1, Real.sqrt 3)

-- State that P is on the circle
axiom P_on_circle : P ∈ Circle

-- Define A and B as variables
variable (A B : ℝ × ℝ)

-- State that A and B are on the circle
axiom A_on_circle : A ∈ Circle
axiom B_on_circle : B ∈ Circle

-- Define what it means for a line to be tangent to the circle
def IsTangentLine (C : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) : Prop :=
  P ∈ C ∧ Q ∈ C ∧ ∀ R ∈ C, R ≠ Q → (R.1 - P.1) * (Q.1 - P.1) + (R.2 - P.2) * (Q.2 - P.2) ≤ 0

-- State that PA and PB are tangent to the circle
axiom PA_tangent : IsTangentLine Circle P A
axiom PB_tangent : IsTangentLine Circle P B

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_PA_PB :
  dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = 3/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_PA_PB_l778_77841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l778_77871

theorem x_value_proof (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l778_77871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l778_77861

noncomputable def scores : List ℝ := [12, 9, 14, 12, 8]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (λ x => (x - mean xs)^2)).sum / xs.length

theorem variance_of_scores : variance scores = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l778_77861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_inequality_l778_77873

-- Define the points A, B, C, D in a 2D plane
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the property that ABCD is a convex quadrilateral
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the property that AB extended and CD extended intersect at a right angle
def perpendicular_extended (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Theorem statement
theorem quadrilateral_diagonal_inequality
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : perpendicular_extended A B C D) :
  dist A C * dist B D > dist A D * dist B C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_diagonal_inequality_l778_77873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_tan_pi_over_4x_l778_77894

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.pi / 4 * x)

theorem min_positive_period_of_tan_pi_over_4x :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_tan_pi_over_4x_l778_77894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l778_77838

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * a 1 + (n : ℝ) * ((n : ℝ) - 1) / 2 * (a n - a 1) / ((n : ℝ) - 1)

theorem max_sum_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_relation : 3 * a 8 = 5 * a 13) :
  ∃ n : ℕ, ∀ m : ℕ, sum_arithmetic_sequence a n ≥ sum_arithmetic_sequence a m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l778_77838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_XY_pass_through_fixed_point_l778_77885

noncomputable section

/-- Definition of a circle centered at the origin --/
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Definition of a point inside the circle --/
def InsideCircle (P : ℝ × ℝ) (r : ℝ) := P.1^2 + P.2^2 < r^2

/-- Definition of perpendicular vectors --/
def Perpendicular (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2 = 0

/-- Definition of the fixed point C --/
def FixedPoint (P : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (2 * P.1 * r^2 / (P.1^2 + P.2^2 + r^2), 2 * P.2 * r^2 / (P.1^2 + P.2^2 + r^2))

/-- Definition of a line through two points --/
def Line (A B : ℝ × ℝ) := {P : ℝ × ℝ | ∃ t : ℝ, P = (1 - t) • A + t • B}

/-- Definition of projection of a point onto a line --/
def IsProjection (X P : ℝ × ℝ) (L : Set (ℝ × ℝ)) :=
  X ∈ L ∧ ∀ Y ∈ L, (X.1 - P.1)^2 + (X.2 - P.2)^2 ≤ (Y.1 - P.1)^2 + (Y.2 - P.2)^2

/-- Definition of intersection point of tangents --/
def IsTangentIntersection (Y A B : ℝ × ℝ) (r : ℝ) :=
  (Y.1 - A.1)^2 + (Y.2 - A.2)^2 = r^2 ∧
  (Y.1 - B.1)^2 + (Y.2 - B.2)^2 = r^2

/-- Main theorem --/
theorem all_XY_pass_through_fixed_point
  (r : ℝ) (P : ℝ × ℝ) (h_P : InsideCircle P r)
  (A B : ℝ × ℝ) (h_A : A ∈ Circle r) (h_B : B ∈ Circle r)
  (h_perp : Perpendicular (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2)) :
  ∃ (X Y : ℝ × ℝ), X ∈ Line A B ∧ IsProjection X P (Line A B) ∧
                   IsTangentIntersection Y A B r ∧
                   FixedPoint P r ∈ Line X Y :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_XY_pass_through_fixed_point_l778_77885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_perimeter_ratio_l778_77827

/-- The side length of the regular hexagon -/
def side_length : ℝ := 10

/-- The perimeter of a regular hexagon -/
def hexagon_perimeter (s : ℝ) : ℝ := 6 * s

/-- The area of a regular hexagon -/
noncomputable def hexagon_area (s : ℝ) : ℝ := 6 * ((Real.sqrt 3 / 4) * s^2)

/-- The ratio of area to perimeter for a regular hexagon -/
noncomputable def area_perimeter_ratio (s : ℝ) : ℝ := hexagon_area s / hexagon_perimeter s

/-- Theorem stating the ratio of area to perimeter for a regular hexagon with side length 10 -/
theorem hexagon_area_perimeter_ratio :
  area_perimeter_ratio side_length = 5 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_perimeter_ratio_l778_77827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l778_77846

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, HasDerivAt f (f' x) x ∧ f' x < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l778_77846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l778_77829

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R,
    such that cos(P) = cos(Q) + cos(R), prove that its area is 52 + 6√55 when r = 4 and R = 13 -/
theorem triangle_area (r R : ℝ) (P Q R : ℝ) (h1 : r = 4) (h2 : R = 13)
  (h3 : Real.cos P = Real.cos Q + Real.cos R) :
  r * ((3 * Real.sqrt 55 + 26 * Real.sin Q + 26 * Real.sin R) / 2) = 52 + 6 * Real.sqrt 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l778_77829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_condition_l778_77823

-- Define the function
noncomputable def f (k x : ℝ) : ℝ := (k - 1) * x^(k^2 - k + 2) + k * x - 1

-- Theorem statement
theorem quadratic_condition (k : ℝ) :
  (∀ x, ∃ a b c : ℝ, f k x = a * x^2 + b * x + c) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_condition_l778_77823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_four_l778_77891

/-- The sum of the infinite series Σ(6n^2 - n + 1) / (n^5 - n^4 + n^3 - n^2 + n) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (6 * n^2 - n + 1) / (n^5 - n^4 + n^3 - n^2 + n)

/-- Theorem stating that the infinite series sum equals 4 -/
theorem infinite_series_sum_eq_four : infinite_series_sum = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_four_l778_77891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l778_77887

/-- A polynomial of degree 100 with specific terms -/
noncomputable def polynomial (C D : ℂ) (x : ℂ) : ℂ := x^100 + C*x^2 + D*x + 1

/-- The quadratic divisor -/
noncomputable def divisor (x : ℂ) : ℂ := x^2 + x + 1

/-- The complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

theorem polynomial_divisibility (C D : ℂ) :
  (∀ x, divisor x = 0 → polynomial C D x = 0) →
  C + D = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l778_77887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_m_for_union_equality_l778_77807

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m}

-- Part 1
theorem intersection_A_complement_B : 
  A ∩ (Set.univ \ B 3) = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

-- Part 2
theorem range_of_m_for_union_equality : 
  {m : ℝ | B m ∪ A = A} = Set.Iic (-1) ∪ Set.Icc 1 (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_range_of_m_for_union_equality_l778_77807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l778_77890

noncomputable section

-- Define the radius of the original circular sheet
def original_radius : ℝ := 9

-- Define the number of congruent sectors
def num_sectors : ℕ := 4

-- Define the radius of the cone's base
def cone_base_radius : ℝ := original_radius * Real.pi / (2 * num_sectors)

-- Define the slant height of the cone (equal to the original radius)
def slant_height : ℝ := original_radius

-- Theorem to prove: The height of the cone
theorem cone_height (original_radius : ℝ) (num_sectors : ℕ) 
  (cone_base_radius : ℝ) (slant_height : ℝ) 
  (h : cone_base_radius = original_radius * Real.pi / (2 * num_sectors)) 
  (h' : slant_height = original_radius) 
  (h'' : num_sectors = 4) :
  ∃ (height : ℝ), height^2 = slant_height^2 - cone_base_radius^2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_l778_77890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_reverse_l778_77824

/-- A move is defined as taking a small stack of cards and inserting them into any position without disturbing their order. -/
def Move (n : ℕ) := Unit

/-- The state of the stack of cards -/
def CardStack (n : ℕ) := Fin n → Fin n

/-- The initial state of the stack -/
def initialStack (n : ℕ) : CardStack n := fun i => i

/-- The reversed state of the stack -/
def reversedStack (n : ℕ) : CardStack n := fun i => ⟨n - 1 - i, sorry⟩

/-- A sequence of moves -/
def MoveSequence (n : ℕ) := List (Move n)

/-- Applying a sequence of moves to a stack -/
def applyMoves (n : ℕ) (stack : CardStack n) (moves : MoveSequence n) : CardStack n :=
  sorry

/-- A sequence of moves is valid if it reverses the stack -/
def isValidMoveSequence (n : ℕ) (moves : MoveSequence n) : Prop :=
  applyMoves n (initialStack n) moves = reversedStack n

/-- The main theorem: The minimum number of moves required to reverse the stack is ⌊n/2⌋ + 1 -/
theorem min_moves_to_reverse (n : ℕ) :
  (∃ (moves : MoveSequence n), isValidMoveSequence n moves ∧ moves.length = n / 2 + 1) ∧
  (∀ (moves : MoveSequence n), isValidMoveSequence n moves → moves.length ≥ n / 2 + 1) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_reverse_l778_77824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l778_77855

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem omega_value 
  (ω φ α β : ℝ) 
  (h1 : f ω φ α = -2) 
  (h2 : f ω φ β = 0) 
  (h3 : ∀ γ δ, f ω φ γ = -2 → f ω φ δ = 0 → |γ - δ| ≥ π) 
  (h4 : ∃ γ δ, f ω φ γ = -2 ∧ f ω φ δ = 0 ∧ |γ - δ| = π) 
  (h5 : ω > 0) : 
  ω = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l778_77855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_polynomial_divisibility_l778_77848

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Polynomial of degree 1006 matching Fibonacci sequence for k in {1008, ..., 2014} -/
noncomputable def f : ℕ → ℕ := sorry

/-- f matches Fibonacci sequence for k in {1008, ..., 2014} -/
axiom f_matches_fib : ∀ k : ℕ, 1008 ≤ k → k ≤ 2014 → f k = fib k

/-- Main theorem: 233 divides f(2015) + 1 -/
theorem fib_polynomial_divisibility : 233 ∣ (f 2015 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fib_polynomial_divisibility_l778_77848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_is_seven_and_one_third_l778_77805

/-- The time when Alice and Bob pass each other the second time -/
noncomputable def second_meeting_time (pool_length : ℝ) (alice_initial_speed : ℝ) (bob_speed : ℝ) (first_meeting_time : ℝ) (alice_speed_increase : ℝ) : ℝ :=
  let alice_new_speed := alice_initial_speed * (1 + alice_speed_increase)
  let first_meeting_distance := alice_initial_speed * first_meeting_time
  let remaining_distance := pool_length - first_meeting_distance
  first_meeting_time + remaining_distance / (alice_new_speed + bob_speed)

/-- Theorem stating that Alice and Bob pass each other the second time after 7 1/3 minutes -/
theorem second_meeting_time_is_seven_and_one_third :
  second_meeting_time 100 10 15 4 0.5 = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_time_is_seven_and_one_third_l778_77805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_geometric_sum_difference_l778_77858

/-- Geometric series sum for positive integers -/
def geometric_sum (a : ℕ) (n : ℕ) : ℚ :=
  (a ^ n - 1) / (a - 1)

/-- Main theorem -/
theorem divisibility_of_geometric_sum_difference (a s t : ℕ) 
  (ha : 1 < a)
  (hs : 0 < s)
  (ht : 0 < t)
  (h_diff : s ≠ t)
  (h_div : ∀ (p : ℕ) (hp : Nat.Prime p) (hps : p ∣ s - t), p ∣ a - 1) :
  ∃ (k : ℤ), (geometric_sum a s - geometric_sum a t : ℚ) = k * (s - t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_geometric_sum_difference_l778_77858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l778_77883

def a : ℝ × ℝ × ℝ := (2, -2, 3)
def b : ℝ × ℝ × ℝ := (1, 4, -1)

def collinear (x y z : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), y = (x.1 + t₁ * (z.1 - x.1), x.2.1 + t₁ * (z.2.1 - x.2.1), x.2.2 + t₁ * (z.2.2 - x.2.2)) ∧
                 z = (x.1 + t₂ * (y.1 - x.1), x.2.1 + t₂ * (y.2.1 - x.2.1), x.2.2 + t₂ * (y.2.2 - x.2.2))

def orthogonal (x y : ℝ × ℝ × ℝ) : Prop :=
  x.1 * y.1 + x.2.1 * y.2.1 + x.2.2 * y.2.2 = 0

theorem projection_theorem (p : ℝ × ℝ × ℝ) :
  collinear a b p ∧ orthogonal p (b.1 - a.1, b.2.1 - a.2.1, b.2.2 - a.2.2) →
  p = (3/2, 1, 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l778_77883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l778_77832

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l778_77832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_eq_pi_div_4tan2_54_l778_77815

/-- A regular pentagon with a circumscribed circle -/
structure RegularPentagonWithCircle where
  -- Side length of the pentagon
  s : ℝ
  -- Radius of the circle
  r : ℝ
  -- The circle is tangent to two sides of the pentagon
  tangent_to_sides : r > 0
  -- The circle's center is equidistant from two other sides
  equidistant_from_sides : r > 0

/-- The ratio of the circle's area to the square of the pentagon's side length -/
noncomputable def area_ratio (p : RegularPentagonWithCircle) : ℝ :=
  Real.pi * p.r^2 / p.s^2

/-- The main theorem: The ratio is equal to π / (4 * tan²(54°)) -/
theorem area_ratio_eq_pi_div_4tan2_54 (p : RegularPentagonWithCircle) :
  area_ratio p = Real.pi / (4 * Real.tan (54 * Real.pi / 180)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_eq_pi_div_4tan2_54_l778_77815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_average_speed_l778_77864

/-- Calculates the average speed of a crow accounting for wind resistance -/
theorem crow_average_speed (distance : ℝ) (time : ℝ) (trips : ℕ) (base_speed : ℝ) 
  (wind_reduction : ℝ) (wind_increase : ℝ) :
  distance = 400 →
  time = 1.5 →
  trips = 15 →
  base_speed = 25 →
  wind_reduction = 0.3 →
  wind_increase = 0.2 →
  ∃ (avg_speed : ℝ), abs (avg_speed - 8) < 0.01 ∧ 
    avg_speed = (2 * distance * trips) / (time * 3600) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crow_average_speed_l778_77864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_number_is_six_l778_77854

def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / xs.length

theorem ninth_number_is_six (xs : List ℕ) (h1 : xs.length = 9) 
  (h2 : mean xs = 57/10)
  (h3 : xs.take 8 = [1, 10, 4, 3, 3, 11, 3, 10]) :
  xs.get ⟨8, by 
    have : 8 < 9 := by norm_num
    exact Nat.lt_of_succ_le (Nat.le_of_eq h1.symm)⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_number_is_six_l778_77854
