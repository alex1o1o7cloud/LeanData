import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_PF2F1_eq_four_fifths_l1272_127225

/-- Represents a hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- Represents the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Represents the angle PF₂F₁ for a hyperbola -/
noncomputable def angle_PF2F1 (h : Hyperbola) : ℝ := sorry

/-- Theorem: For a hyperbola with eccentricity 5, cos(PF₂F₁) = 4/5 -/
theorem cos_angle_PF2F1_eq_four_fifths (h : Hyperbola) 
    (h_ecc : eccentricity h = 5) : 
    Real.cos (angle_PF2F1 h) = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_PF2F1_eq_four_fifths_l1272_127225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_white_ball_l1272_127226

/-- Represents a box containing a ball -/
structure Box :=
  (color : Bool)  -- true if white, false if not white

/-- Represents a question about two boxes -/
structure Question :=
  (box1 : Nat)
  (box2 : Nat)

/-- Function to answer a question -/
def answer (boxes : List Box) (q : Question) : Bool :=
  match boxes[q.box1]?, boxes[q.box2]? with
  | some b1, some b2 => b1.color || b2.color
  | _, _ => false

/-- Function to determine if a strategy is successful -/
def isSuccessfulStrategy (boxes : List Box) (questions : List Question) : Bool :=
  sorry  -- Implementation details omitted

/-- Theorem stating the minimum number of questions required -/
theorem min_questions_for_white_ball :
  ∀ (boxes : List Box),
    boxes.length = 2004 →
    (boxes.filter (λ b => b.color)).length % 2 = 0 →
    (∃ (questions : List Question),
      questions.length = 2003 ∧
      (∀ (boxArrangement : List Box),
        boxArrangement.length = 2004 →
        (boxArrangement.filter (λ b => b.color)).length % 2 = 0 →
        isSuccessfulStrategy boxArrangement questions)) ∧
    (∀ (questions : List Question),
      questions.length < 2003 →
      (∃ (boxArrangement : List Box),
        boxArrangement.length = 2004 ∧
        (boxArrangement.filter (λ b => b.color)).length % 2 = 0 ∧
        ¬ isSuccessfulStrategy boxArrangement questions)) :=
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_white_ball_l1272_127226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_nine_twos_squared_l1272_127294

def nine_twos : Nat := 222222222

theorem sum_of_digits_of_nine_twos_squared : 
  (Nat.digits 10 (nine_twos ^ 2)).sum = 324 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_nine_twos_squared_l1272_127294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l1272_127206

theorem distinct_remainders (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬(p ∣ a)) :
  ∀ x y : ℕ, 1 ≤ x → x < p → 1 ≤ y → y < p → x ≠ y →
    (x : ℤ) * (a : ℤ) % (p : ℤ) ≠ (y : ℤ) * (a : ℤ) % (p : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_l1272_127206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1272_127223

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is inside the unit square -/
def insideUnitSquare (p : Point) : Prop :=
  0 < p.x ∧ p.x < 1 ∧ 0 < p.y ∧ p.y < 1

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2

/-- Theorem: Given 9 points in a unit square, there exists a triangle with area ≤ 1/8 -/
theorem triangle_area_bound (points : Fin 9 → Point) 
  (h : ∀ i, insideUnitSquare (points i)) : 
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    triangleArea (points i) (points j) (points k) ≤ 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1272_127223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l1272_127298

theorem max_value_trig_sum :
  (∀ θ₁ θ₂ θ₃ θ₄ : ℝ, Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₁ ≤ 2) ∧
  (∃ θ₁ θ₂ θ₃ θ₄ : ℝ, Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₁ = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_sum_l1272_127298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_fair_coin_even_heads_probability_biased_coin_l1272_127239

/-- The probability of getting an even number of heads when tossing a coin n times -/
noncomputable def even_heads_probability (p : ℝ) (n : ℕ) : ℝ :=
  (1 + (1 - 2*p)^n) / 2

theorem even_heads_probability_fair_coin (n : ℕ) :
  even_heads_probability (1/2) n = 1/2 := by
  sorry

theorem even_heads_probability_biased_coin (p : ℝ) (n : ℕ) 
  (h1 : 0 < p) (h2 : p < 1) :
  even_heads_probability p n = (1 + (1 - 2*p)^n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_fair_coin_even_heads_probability_biased_coin_l1272_127239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_200_of_5_11_is_5_l1272_127275

/-- The 200th digit after the decimal point in the decimal representation of 5/11 -/
def digit_200_of_5_11 : ℕ :=
  5

theorem digit_200_of_5_11_is_5 : digit_200_of_5_11 = 5 := by
  rfl

#eval digit_200_of_5_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_200_of_5_11_is_5_l1272_127275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1272_127233

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | (n + 2) => 3 * sequence_a (n + 1) + 2 * (n + 1) - 1

theorem sequence_a_formula (n : ℕ) :
  sequence_a n = 2/3 * 3^n - n := by
  induction n with
  | zero => 
    simp [sequence_a]
    -- The proof for n = 0
    sorry
  | succ n ih =>
    simp [sequence_a]
    -- The inductive step
    sorry

#eval sequence_a 5  -- You can add this line to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1272_127233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_properties_l1272_127246

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * Real.cos (2 * x)

theorem symmetric_function_properties (a : ℝ) :
  (∀ x : ℝ, f a (π/3 + x) = f a (π/3 - x)) →
  (a = -Real.sqrt 3 / 3 ∧
   ∀ x : ℝ, f a (x - 5*π/12) = -f a (-x - 5*π/12)) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_properties_l1272_127246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_decomposition_l1272_127274

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, -2; -3, 5]

theorem B_power_six_decomposition :
  B^6 = 2999 • B + 2520 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_six_decomposition_l1272_127274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temp_rounded_l1272_127273

def sunday_temp : ℝ := 99.1
def monday_temp : ℝ := 98.2
def tuesday_temp : ℝ := 98.7
def wednesday_temp : ℝ := 99.3
def thursday_temp : ℝ := 99.8
def friday_temp : ℝ := 99.0
def saturday_temp : ℝ := 98.9

def week_temps : List ℝ := [sunday_temp, monday_temp, tuesday_temp, wednesday_temp, thursday_temp, friday_temp, saturday_temp]

noncomputable def average_temp : ℝ := (week_temps.sum) / 7

theorem average_temp_rounded : Int.floor (average_temp + 0.5) = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temp_rounded_l1272_127273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_130_meters_l1272_127202

-- Define the train's speed in km/hr
noncomputable def train_speed : ℝ := 45

-- Define the time to cross the bridge in seconds
noncomputable def crossing_time : ℝ := 30

-- Define the total length of the bridge and train in meters
noncomputable def total_length : ℝ := 245

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Theorem statement
theorem train_length_is_130_meters :
  let speed_m_s := train_speed * km_hr_to_m_s
  let distance_covered := speed_m_s * crossing_time
  let train_length := distance_covered - total_length
  train_length = 130 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_130_meters_l1272_127202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_area_ratio_l1272_127220

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def focal_line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define points A and B as intersections of the focal line and parabola
noncomputable def point_A (k : ℝ) : ℝ × ℝ := sorry

noncomputable def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define midpoint M
noncomputable def midpoint_M (k : ℝ) : ℝ × ℝ := sorry

-- Define origin O
def origin : ℝ × ℝ := (0, 0)

-- Define points P and Q
noncomputable def point_P (k : ℝ) : ℝ × ℝ := sorry

noncomputable def point_Q (k : ℝ) : ℝ × ℝ := sorry

-- Theorem for the trajectory of M
theorem trajectory_of_M :
  ∀ k : ℝ, let (x, y) := midpoint_M k; y^2 = 2*x - 2 := by
  sorry

-- Theorem for the ratio of areas
theorem area_ratio :
  ∀ k : ℝ, 
    let triangle_OPQ := abs (point_P k).2 - abs (point_Q k).2;
    let triangle_BOM := abs ((point_B k).2 - (midpoint_M k).2);
    triangle_OPQ / triangle_BOM = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_area_ratio_l1272_127220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_planes_l1272_127267

/-- Definition of a plane -/
def Plane : Type := ℝ → ℝ → ℝ → Prop

/-- Normal vector of a plane -/
noncomputable def NormalVector (p : Plane) : ℝ × ℝ × ℝ := sorry

/-- Dihedral angle between two planes -/
noncomputable def DihedralAngle (p q : Plane) : ℝ := sorry

/-- The dihedral angle between two planes with given normal vectors is either π/3 or 2π/3 -/
theorem dihedral_angle_of_planes (α β : Plane) :
  NormalVector α = (1, 0, -1) →
  NormalVector β = (0, -1, 1) →
  DihedralAngle α β = π/3 ∨ DihedralAngle α β = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_planes_l1272_127267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularCirclesExist_l1272_127245

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A circle with center and radius. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to check if a triangle is acute-angled. -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Function to check if two circles intersect perpendicularly. -/
def intersectPerpendicularly (c1 c2 : Circle) : Prop :=
  sorry

/-- Function to check if a point is on a circle. -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

/-- Function to get the altitude points of a triangle. -/
def altitudePoints (t : Triangle) : List (ℝ × ℝ) :=
  sorry

/-- Theorem: Circles can be constructed around the vertices of a triangle
    that intersect each other perpendicularly in pairs if and only if
    the triangle is acute-angled, and these circles pass through the
    altitude points of the triangle. -/
theorem perpendicularCirclesExist (t : Triangle) :
  (∃ cA cB cC : Circle,
    cA.center = t.A ∧ cB.center = t.B ∧ cC.center = t.C ∧
    intersectPerpendicularly cA cB ∧
    intersectPerpendicularly cB cC ∧
    intersectPerpendicularly cC cA ∧
    ∀ p ∈ altitudePoints t, isOnCircle p cA ∧ isOnCircle p cB ∧ isOnCircle p cC) ↔
  isAcuteAngled t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicularCirclesExist_l1272_127245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_plus_y_values_l1272_127255

theorem tan_x_plus_y_values (x y : ℝ) :
  (((Real.cos x - Real.sin x) / Real.sin y = (2 * Real.sqrt 2) / 5 * Real.tan ((x + y) / 2)) ∧
   ((Real.sin x + Real.cos x) / Real.cos y = -(5 / Real.sqrt 2) * (1 / Real.tan ((x + y) / 2)))) →
  (Real.tan (x + y) = -1 ∨ Real.tan (x + y) = 20/21 ∨ Real.tan (x + y) = -20/21) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x_plus_y_values_l1272_127255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_four_dividing_factorial_l1272_127270

theorem greatest_power_of_four_dividing_factorial :
  (∃ x : ℕ, 4^x ∣ Nat.factorial 21 ∧ ∀ y : ℕ, 4^y ∣ Nat.factorial 21 → y ≤ x) ∧
  (∀ x : ℕ, 4^x ∣ Nat.factorial 21 → x ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_power_of_four_dividing_factorial_l1272_127270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fib_representation_l1272_127283

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Representation of a number in Fibonacci numeral system -/
def fibRepresentation (m : ℕ) := { b : ℕ → Bool // ∀ k, k < m → b k → ¬b (k + 1) }

/-- Value of a Fibonacci representation -/
def fibValue (m : ℕ) (b : fibRepresentation m) : ℕ :=
  (Finset.range m).sum (λ k ↦ if b.val k then fib k else 0)

/-- Theorem: Unique representation in Fibonacci numeral system -/
theorem unique_fib_representation (m : ℕ) (n : ℕ) (h : n ≤ fib m) :
  ∃! (b : fibRepresentation m), fibValue m b = n := by
  sorry

#check unique_fib_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fib_representation_l1272_127283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_5pi_12_l1272_127287

theorem cos_2alpha_plus_5pi_12 (α : ℝ) (h1 : π < α) (h2 : α < 2*π) 
  (h3 : Real.sin (α + π/3) = -4/5) : Real.cos (2*α + 5*π/12) = 17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_5pi_12_l1272_127287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sample_more_accurate_l1272_127250

-- Define a population
def Population := Type

-- Define a sample from a population
def Sample (p : Population) := Type

-- Define a measure of estimation accuracy
def EstimationAccuracy : ℝ → ℝ := sorry

-- Define a function that relates sample size to accuracy
def AccuracyBySize (p : Population) : ℕ → ℝ := sorry

-- State the theorem
theorem larger_sample_more_accurate (p : Population) :
  ∀ n m : ℕ, n < m → EstimationAccuracy (AccuracyBySize p n) < EstimationAccuracy (AccuracyBySize p m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_sample_more_accurate_l1272_127250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_values_l1272_127254

theorem fraction_values (m n : ℕ) (h : Nat.Coprime m n) (hm : m > 0) (hn : n > 0) :
  (m^2 + 20*m*n + n^2) / (m^3 + n^3) = 1 ∨
  (m^2 + 20*m*n + n^2) / (m^3 + n^3) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_values_l1272_127254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_range_l1272_127261

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define the line y = kx + m
def line_y_eq_kx_plus_m (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem ellipse_equation_and_m_range 
  (a b : ℝ) 
  (h_a_gt_b : a > b) 
  (h_b_pos : b > 0) 
  (h_eccentricity : eccentricity a b = Real.sqrt 3 / 2)
  (xa ya xb yb : ℝ)
  (h_A : ellipse a b xa ya ∧ line_y_eq_x xa ya)
  (h_B : ellipse a b xb yb ∧ line_y_eq_x xb yb)
  (h_P : ellipse a b a 0)
  (h_PA_PB : distance (a - xa) (-ya) 0 0 + distance (a - xb) (-yb) 0 0 = 4)
  (k m : ℝ)
  (h_k_neq_0 : k ≠ 0)
  (h_m_neq_0 : m ≠ 0)
  (xm ym xn yn : ℝ)
  (h_M : ellipse a b xm ym ∧ line_y_eq_kx_plus_m k m xm ym)
  (h_N : ellipse a b xn yn ∧ line_y_eq_kx_plus_m k m xn yn)
  (h_M_neq_N : xm ≠ xn ∨ ym ≠ yn)
  (h_Q : distance xm ym 0 (-1/2) = distance xn yn 0 (-1/2)) :
  (ellipse 2 1 = ellipse a b) ∧ (1/6 < m ∧ m < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_range_l1272_127261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_base_max_base_total_possible_bases_l1272_127201

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

-- Define the function to count trailing zeroes in base B
noncomputable def trailingZeroes (n : ℕ) (B : ℕ) : ℕ :=
  sorry -- Implementation of trailing zeroes count

-- Define the condition
def condition (B : ℕ) : Prop :=
  trailingZeroes (factorial (2 + 2^96)) B = 2^93

-- Theorem statements
theorem min_base :
  ∃ (B : ℕ), condition B ∧ (∀ (B' : ℕ), condition B' → B ≤ B') ∧ B = 16 :=
sorry

theorem max_base :
  ∃ (B : ℕ), condition B ∧ (∀ (B' : ℕ), condition B' → B' ≤ B) ∧ B = 5040 :=
sorry

-- Define a decidable version of the condition
noncomputable def decidable_condition (B : ℕ) : Bool :=
  if trailingZeroes (factorial (2 + 2^96)) B = 2^93 then true else false

theorem total_possible_bases :
  (Finset.filter (λ B ↦ decidable_condition B) (Finset.range 5041)).card = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_base_max_base_total_possible_bases_l1272_127201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1272_127222

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : 2 * i^45 + 3 * i^123 = -i := by
  -- We'll use the properties of Complex.I
  have i_squared : i * i = -1 := Complex.I_mul_I
  
  -- Simplify i^45 and i^123 using the periodicity of i
  have i_45 : i^45 = i := by
    sorry -- Proof omitted
  
  have i_123 : i^123 = -i := by
    sorry -- Proof omitted
  
  -- Substitute and simplify
  calc
    2 * i^45 + 3 * i^123 = 2 * i + 3 * (-i) := by rw [i_45, i_123]
    _ = 2 * i - 3 * i := by ring
    _ = -i := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1272_127222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l1272_127241

/-- Represents the total profit of the business -/
def total_profit : ℝ → Prop := λ _ => True

/-- Mary's investment -/
def mary_investment : ℝ := 550

/-- Mike's investment -/
def mike_investment : ℝ := 450

/-- The difference between Mary's and Mike's profit shares -/
def profit_difference : ℝ := 1000

/-- Theorem stating the conditions and the result to be proved -/
theorem partnership_profit (P : ℝ) : 
  total_profit P →
  (P / 6 + (11 / 20) * (2 * P / 3)) - (P / 6 + (9 / 20) * (2 * P / 3)) = profit_difference →
  P = 15000 := by
  intro h1 h2
  sorry

#check partnership_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l1272_127241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_sine_and_tangent_l1272_127284

theorem inequality_of_sine_and_tangent (x y : ℝ) (h : 0 < y ∧ y < x ∧ x < Real.pi/2) :
  Real.sin x - Real.sin y < x - y ∧ x - y < Real.tan x - Real.tan y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_of_sine_and_tangent_l1272_127284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_boundary_integers_l1272_127237

theorem sqrt_boundary_integers : 
  (Finset.filter (fun x : ℕ => 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5) (Finset.range 25)).card = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_boundary_integers_l1272_127237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_walkers_l1272_127203

/-- Represents a person walking on a road --/
structure Walker where
  steps_per_minute : ℕ
  step_length : ℕ
  initial_distance : ℕ

/-- Calculates the position of a walker after a given number of steps --/
noncomputable def position (w : Walker) (steps : ℕ) : ℝ :=
  w.initial_distance - (steps * w.step_length : ℝ) / 100

/-- Theorem stating the minimum distance between two walkers --/
theorem min_distance_between_walkers (sasha dania : Walker)
  (h1 : sasha.steps_per_minute = 45)
  (h2 : sasha.step_length = 60)
  (h3 : sasha.initial_distance = 29000)
  (h4 : dania.steps_per_minute = 55)
  (h5 : dania.step_length = 65)
  (h6 : dania.initial_distance = 31000) :
  ∃ (sasha_steps dania_steps : ℕ),
    sasha_steps = 396 ∧
    dania_steps = 484 ∧
    (∀ (s d : ℕ),
      s % sasha.steps_per_minute = 0 →
      d % dania.steps_per_minute = 0 →
      |position sasha sasha_steps - position dania dania_steps| ≤
      |position sasha s - position dania d|) ∧
    |position sasha sasha_steps - position dania dania_steps| = 5700 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_walkers_l1272_127203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1272_127232

-- Define the constants
noncomputable def a : ℝ := (3 : ℝ) ^ (1/5 : ℝ)
noncomputable def b : ℝ := (0.3 : ℝ) ^ 2
noncomputable def c : ℝ := Real.log 2 / Real.log 0.3

-- State the theorem
theorem magnitude_relationship : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1272_127232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pr_length_in_right_triangle_l1272_127213

-- Define the right triangle PQR
structure RightTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the theorem
theorem pr_length_in_right_triangle (t : RightTriangle) 
  (cos_R : Real.cos (Real.arccos ((t.R.1 - t.Q.1) / Real.sqrt ((t.R.1 - t.Q.1)^2 + (t.R.2 - t.Q.2)^2))) = 5 * Real.sqrt 34 / 34)
  (hypotenuse : Real.sqrt ((t.P.1 - t.Q.1)^2 + (t.P.2 - t.Q.2)^2) = Real.sqrt 34) :
  Real.sqrt ((t.P.1 - t.R.1)^2 + (t.P.2 - t.R.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pr_length_in_right_triangle_l1272_127213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_condition_l1272_127230

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

/-- A triangle is obtuse if one of its angles is greater than π/2. -/
def Triangle.isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

/-- Main theorem: If c - b*cos(A) < 0, then the triangle is obtuse. -/
theorem triangle_obtuse_condition (t : Triangle) (h : t.c - t.b * Real.cos t.A < 0) : 
  t.isObtuse := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_obtuse_condition_l1272_127230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_angles_in_convex_polygon_l1272_127278

/-- Represents a convex polygon --/
structure ConvexPolygon where
  -- We'll leave the implementation details abstract for now
  mk :: -- Constructor

/-- Represents an angle --/
structure Angle where
  -- We'll leave the implementation details abstract for now
  mk ::

/-- Returns the set of right angles in a convex polygon --/
def rightAngles (p : ConvexPolygon) : Finset Angle :=
  sorry -- Implementation details omitted for now

/-- Theorem: The maximum number of right angles in a convex polygon is 4 --/
theorem max_right_angles_in_convex_polygon :
  ∀ (n : ℕ), n > 0 →
  (∃ (p : ConvexPolygon), (rightAngles p).card = n) →
  n ≤ 4 :=
by
  sorry -- Proof omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_angles_in_convex_polygon_l1272_127278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1272_127272

theorem expression_evaluation : |(-2 : ℝ)| + (Real.sqrt 2 - 1)^0 - (-5) - (1/3)^(-1 : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l1272_127272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1272_127204

noncomputable def line_equation (x y : ℝ) : Prop := y = 3 * x - 5

noncomputable def parameterization_A (t : ℝ) : ℝ × ℝ := (t, 3 * t - 5)
noncomputable def parameterization_B (t : ℝ) : ℝ × ℝ := (5/3 + 3 * t, t)
noncomputable def parameterization_C (t : ℝ) : ℝ × ℝ := (2 + 9 * t, 1 + 3 * t)
noncomputable def parameterization_D (t : ℝ) : ℝ × ℝ := (-5 + t/3, -20 + t)
noncomputable def parameterization_E (t : ℝ) : ℝ × ℝ := (-5/3 + t/9, -5 + t/3)

theorem valid_parameterizations :
  (∀ t, line_equation (parameterization_A t).1 (parameterization_A t).2) ∧
  (∀ t, line_equation (parameterization_C t).1 (parameterization_C t).2) ∧
  (¬ ∀ t, line_equation (parameterization_B t).1 (parameterization_B t).2) ∧
  (¬ ∀ t, line_equation (parameterization_D t).1 (parameterization_D t).2) ∧
  (¬ ∀ t, line_equation (parameterization_E t).1 (parameterization_E t).2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_parameterizations_l1272_127204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_friendship_l1272_127290

universe u

/-- Represents a group of people and their friendships -/
structure FriendshipGroup (α : Type u) where
  people : Finset α
  friendship : α → α → Bool
  symmetric : ∀ a b, friendship a b = friendship b a

/-- The number of friends a person has in the group -/
def friendCount {α : Type u} (G : FriendshipGroup α) (p : α) : ℕ :=
  (G.people.filter (fun q => G.friendship p q)).card

theorem pigeonhole_friendship {α : Type u} [Fintype α] (G : FriendshipGroup α) :
  ∃ p q : α, p ∈ G.people ∧ q ∈ G.people ∧ p ≠ q ∧ friendCount G p = friendCount G q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_friendship_l1272_127290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_monotonic_interval_l1272_127221

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 3/2

-- Define the theorem
theorem f_non_monotonic_interval (a : ℝ) :
  (∀ x, x > 0 → f x ≠ 0) →  -- f is defined on (0, +∞)
  (∃ x y, a - 1 < x ∧ x < y ∧ y < a + 1 ∧ (f x - f y) * (x - y) > 0) →  -- f is not monotonic in (a-1, a+1)
  1 ≤ a ∧ a < 3/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_monotonic_interval_l1272_127221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_converges_to_point_l1272_127244

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => λ x => 0  -- Add a case for 0 to avoid missing cases error
| 1 => λ x => Real.sqrt (1 - (x - 1)^2)
| (n+1) => λ x => f n (Real.sqrt ((n+1)^2 - (x - (n+1))^2))

-- Define the domain of a function
def domain (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | ∃ y : ℝ, f x = y}

-- Statement of the theorem
theorem f_domain_converges_to_point :
  ∃ N : ℕ, (∀ n > N, domain (f n) = ∅) ∧
           (domain (f N) = {5}) := by
  sorry

#check f_domain_converges_to_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_converges_to_point_l1272_127244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_condition_l1272_127205

theorem cubic_roots_condition (a b c t u v : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = t ∨ x = u ∨ x = v) →
  (∀ x : ℝ, x^3 + a^3*x^2 + b^3*x + c^3 = 0 ↔ x = t^3 ∨ x = u^3 ∨ x = v^3) ↔
  ∃ t : ℝ, a = t ∧ b = 0 ∧ c = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_condition_l1272_127205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l1272_127236

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x - 2*y + z = 1
def plane2 (x y z : ℝ) : Prop := 2*x + y - z = 4
def planeQ (x y z : ℝ) : Prop := -2*x + 3*y - z = 0

-- Define line M as the intersection of plane1 and plane2
def lineM (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the distance function from a point to a plane
noncomputable def distance_to_plane (x y z a b c d : ℝ) : ℝ :=
  abs (a*x + b*y + c*z + d) / Real.sqrt (a^2 + b^2 + c^2)

theorem plane_Q_satisfies_conditions :
  (∀ x y z, lineM x y z → planeQ x y z) ∧ 
  (∃ x y z, planeQ x y z ∧ ¬plane1 x y z) ∧
  (∃ x y z, planeQ x y z ∧ ¬plane2 x y z) ∧
  (distance_to_plane 1 2 3 (-2) 3 (-1) 0 = 3 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_Q_satisfies_conditions_l1272_127236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_six_l1272_127282

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x) / (Real.log 9)
  else 4^(-x) + 3/2

-- State the theorem
theorem f_sum_equals_six :
  f 27 + f (-((Real.log 3) / (Real.log 4))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_six_l1272_127282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_completion_l1272_127257

/-- A polynomial is a perfect square if it can be expressed as the square of another polynomial. -/
def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ q : Polynomial ℝ, p = q^2

/-- The set of monomials that can be added to 16x^2 + 1 to form a perfect square trinomial. -/
noncomputable def valid_monomials : Set (Polynomial ℝ) :=
  {64 * Polynomial.X^4, 8 * Polynomial.X, -8 * Polynomial.X, -1, -16 * Polynomial.X^2}

/-- The original polynomial 16x^2 + 1. -/
noncomputable def original_poly : Polynomial ℝ :=
  16 * Polynomial.X^2 + 1

theorem perfect_square_completion :
  ∀ m ∈ valid_monomials, is_perfect_square (original_poly + m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_completion_l1272_127257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_face_diagonal_distance_theorem_l1272_127271

/-- The distance between two non-intersecting face diagonals on adjacent faces of a unit cube -/
noncomputable def face_diagonal_distance : ℝ := Real.sqrt 3 / 3

/-- A unit cube is a cube with side length 1 -/
def is_unit_cube (c : Set (Fin 3 → ℝ)) : Prop :=
  ∀ x y, x ∈ c → y ∈ c → ∃ i, |x i - y i| = 1 ∧ ∀ j ≠ i, x j = y j

/-- Two lines are non-intersecting if they do not share any common point -/
def non_intersecting (l1 l2 : Set (Fin 3 → ℝ)) : Prop :=
  ∀ x y, x ∈ l1 → y ∈ l2 → x ≠ y

/-- A face diagonal is a line segment connecting two opposite vertices of a face -/
def is_face_diagonal (d : Set (Fin 3 → ℝ)) (c : Set (Fin 3 → ℝ)) : Prop :=
  ∃ x y, x ∈ c ∧ y ∈ c ∧ d = {z | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ z = (1 - t) • x + t • y} ∧
  (∃ i, |x i - y i| = 1 ∧ ∀ j ≠ i, x j = y j)

/-- Two face diagonals are on adjacent faces if they share exactly one vertex -/
def on_adjacent_faces (d1 d2 : Set (Fin 3 → ℝ)) (c : Set (Fin 3 → ℝ)) : Prop :=
  is_face_diagonal d1 c ∧ is_face_diagonal d2 c ∧
  ∃! v, v ∈ c ∧ v ∈ d1 ∧ v ∈ d2

theorem face_diagonal_distance_theorem (c : Set (Fin 3 → ℝ)) 
  (d1 d2 : Set (Fin 3 → ℝ)) : 
  is_unit_cube c → 
  non_intersecting d1 d2 → 
  on_adjacent_faces d1 d2 c → 
  ∃ x y, x ∈ d1 ∧ y ∈ d2 ∧ ‖x - y‖ = face_diagonal_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_face_diagonal_distance_theorem_l1272_127271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1272_127228

-- Define the circle C
def myCircle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
def myDist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem min_sum_distances :
  ∀ (x y : ℝ), myCircle x y →
    ∀ (ε : ℝ), ε > 0 →
      ∃ (x' y' : ℝ), myCircle x' y' ∧
        myDist (x', y') A + myDist (x', y') B ≥ 20 - ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1272_127228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_monotonically_increasing_l1272_127296

-- Define the functions
noncomputable def f (x : ℝ) := Real.exp x / x
noncomputable def g (x : ℝ) := x * (Real.log x - 1)

-- Define the property of being monotonically increasing on (1, +∞)
def MonotonicallyIncreasing (h : ℝ → ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → h x < h y

-- State the theorem
theorem f_and_g_monotonically_increasing :
  MonotonicallyIncreasing f ∧ MonotonicallyIncreasing g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_monotonically_increasing_l1272_127296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_polynomial_condition_l1272_127295

open Real

/-- A polynomial of degree 1 -/
def LinearPolynomial (c₁ c₂ : ℝ) (x : ℝ) : ℝ := c₁ * x + c₂

/-- The condition that must be satisfied for all a < b -/
def SatisfiesCondition (P : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a < b → (
    (⨆ x ∈ Set.Icc a b, P x) - (⨅ x ∈ Set.Icc a b, P x) = b - a
  )

/-- The main theorem -/
theorem linear_polynomial_condition (c₁ c₂ : ℝ) :
  SatisfiesCondition (LinearPolynomial c₁ c₂) ↔ (c₁ = 1 ∨ c₁ = -1) := by
  sorry

#check linear_polynomial_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_polynomial_condition_l1272_127295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_extension_theorem_l1272_127219

/-- Triangle with medians -/
structure TriangleWithMedians where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  E : ℝ × ℝ  -- midpoint of BC
  F : ℝ × ℝ  -- midpoint of AC

/-- Circumcircle of a triangle -/
def Circumcircle (t : TriangleWithMedians) : Set (ℝ × ℝ) :=
  sorry

/-- Point where a line intersects the circumcircle -/
def intersectCircumcircle (t : TriangleWithMedians) (p q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  sorry

/-- Check if a triangle is isosceles -/
def isIsosceles (t : TriangleWithMedians) : Prop :=
  sorry

/-- Side lengths of a triangle -/
def sideLengths (t : TriangleWithMedians) : ℝ × ℝ × ℝ :=
  sorry

theorem median_extension_theorem (t : TriangleWithMedians) :
  let A₁ := intersectCircumcircle t t.A t.E
  let B₁ := intersectCircumcircle t t.B t.F
  distance t.A A₁ = distance t.B B₁ →
  isIsosceles t ∨
  let (a, b, c) := sideLengths t
  c^4 = a^4 - a^2 * b^2 + b^4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_extension_theorem_l1272_127219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1272_127224

-- Define the ellipse C₁
noncomputable def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the line
def line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) : ℝ := |x + 1| / Real.sqrt 2

-- Define the common chord length
noncomputable def common_chord_length (a b p : ℝ) : ℝ := 2 * Real.sqrt 6

theorem ellipse_parabola_intersection
  (a b p : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : distance_point_to_line (p/2) 0 = Real.sqrt 2)
  (h5 : common_chord_length a b p = 2 * Real.sqrt 6) :
  (∃ (x y : ℝ), ellipse a b x y ∧ parabola p x y) ∧
  (p = 2) ∧
  (a^2 = 9) ∧
  (b^2 = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1272_127224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_six_l1272_127299

def S : Finset ℕ := {1, 2, 3, 4, 5}

def valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b > 6

def total_pairs : ℕ := Finset.card (S.powerset.filter (fun s => s.card = 2))

def valid_pairs : Finset (ℕ × ℕ) :=
  S.product S |>.filter (fun p => p.1 ≠ p.2 ∧ p.1 + p.2 > 6)

theorem prob_sum_greater_than_six :
  (valid_pairs.card : ℚ) / total_pairs = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_six_l1272_127299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_rectangle_dimensions_l1272_127231

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℚ
  width : ℚ

/-- Cuts a rectangle in half parallel to its longer side -/
def cut_rectangle (r : Rectangle) : Rectangle :=
  if r.length ≥ r.width then
    { length := r.length / 2, width := r.width }
  else
    { length := r.length, width := r.width / 2 }

theorem cut_rectangle_dimensions :
  let original := Rectangle.mk 12 6
  let cut := cut_rectangle original
  cut.length = 6 ∧ cut.width = 6 := by
  -- Unfold definitions
  simp [cut_rectangle]
  -- Split the conjunction
  apply And.intro
  -- Prove length = 6
  · rfl
  -- Prove width = 6
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cut_rectangle_dimensions_l1272_127231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_approx_l1272_127286

/-- Represents the number of carnations in the flower shop -/
def C : ℝ := 1  -- We set C to 1 for simplicity, as the ratio is what matters

/-- The number of violets in terms of carnations -/
noncomputable def violets : ℝ := (1/3) * C

/-- The number of tulips in terms of carnations -/
noncomputable def tulips : ℝ := (1/3) * violets

/-- The number of roses in terms of carnations -/
noncomputable def roses : ℝ := tulips

/-- The total number of flowers in the shop -/
noncomputable def total_flowers : ℝ := C + violets + tulips + roses

/-- The percentage of carnations in the flower shop -/
noncomputable def carnation_percentage : ℝ := (C / total_flowers) * 100

/-- Theorem stating that the percentage of carnations is approximately 64.29% -/
theorem carnation_percentage_approx :
  abs (carnation_percentage - 64.29) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carnation_percentage_approx_l1272_127286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_inside_circle_implies_m_greater_than_five_l1272_127214

/-- A circle with center (1, -2) and radius √m --/
def myCircle (m : ℝ) := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y + 2)^2 = m}

/-- The origin point (0, 0) --/
def origin : ℝ × ℝ := (0, 0)

/-- A point is inside a circle if its distance from the center is less than the radius --/
def is_inside (p : ℝ × ℝ) (m : ℝ) : Prop :=
  (p.1 - 1)^2 + (p.2 + 2)^2 < m

theorem origin_inside_circle_implies_m_greater_than_five (m : ℝ) :
  is_inside origin m → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_inside_circle_implies_m_greater_than_five_l1272_127214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_and_alpha_l1272_127285

-- Define α using the built-in GCD function
def α (a b p : ℕ) : ℕ := Nat.gcd (a + b) ((a^p + b^p) / (a + b))

-- State the theorem
theorem gcd_and_alpha (a b p : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : p ≥ 3) (h4 : Nat.Prime p) (h5 : Nat.gcd a b = 1) :
  (Nat.gcd a a = 1) ∧ (α a b p = 1 ∨ α a b p = p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_and_alpha_l1272_127285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_cost_properties_cost_effectiveness_l1272_127240

/-- Represents the cost calculation for travel agencies A and B -/
noncomputable def travel_cost (full_price : ℝ) (x : ℝ) : (ℝ × ℝ) :=
  let y_A := full_price + (full_price / 2) * x
  let y_B := full_price * (x + 1) * 0.6
  (y_A, y_B)

/-- Theorem stating the properties of travel costs for agencies A and B -/
theorem travel_cost_properties (x : ℝ) :
  let full_price : ℝ := 240
  let (y_A, y_B) := travel_cost full_price x
  (y_A = 120 * x + 240) ∧
  (y_B = 144 * x + 144) ∧
  (x < 4 → y_A > y_B) ∧
  (x = 4 → y_A = y_B) ∧
  (x > 4 → y_A < y_B) := by
  sorry

/-- Theorem stating which travel agency is more cost-effective based on the number of students -/
theorem cost_effectiveness (x : ℝ) :
  let full_price : ℝ := 240
  let (y_A, y_B) := travel_cost full_price x
  (x < 4 → y_B < y_A) ∧
  (x = 4 → y_A = y_B) ∧
  (x > 4 → y_A < y_B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_cost_properties_cost_effectiveness_l1272_127240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l1272_127277

/-- The function representing the curve y = x^3 - 12x -/
noncomputable def f (x : ℝ) : ℝ := x^3 - 12*x

/-- The slope of the line through (1,t) and (x, f x) -/
noncomputable def m (t x : ℝ) : ℝ := (f x - t) / (x - 1)

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 12

/-- The condition for the line to be tangent at x -/
def is_tangent (t x : ℝ) : Prop := m t x = f' x

/-- The statement of the problem -/
theorem tangent_range (t : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    is_tangent t x₁ ∧ is_tangent t x₂ ∧ is_tangent t x₃ ∧
    (∀ x : ℝ, is_tangent t x → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  t > -12 ∧ t < -11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_range_l1272_127277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1272_127200

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 0 else x - 1/x

theorem f_has_three_zeros :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = 0) ∧ S.card = 3 ∧ (∀ x : ℝ, f x = 0 → x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1272_127200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1272_127243

-- Define the types of statements
inductive Statement
| Input (s : String)
| Output (s : String)
| Assignment (s : String)

-- Define the rules for correct statements
def isCorrectInput (s : String) : Bool :=
  s.contains ',' && s.contains '='

def isCorrectOutput (s : String) : Bool :=
  s.contains '=' || (¬s.contains '=' && s.all (λ c => c.isDigit || c = '.' || c = '*'))

def isCorrectAssignment (s : String) : Bool :=
  let parts := s.split (λ c => c = '=')
  parts.length = 2 && parts[0]!.all Char.isAlpha

-- Define the list of statements
def statements : List Statement :=
  [Statement.Input "a; b; c",
   Statement.Input "x=3",
   Statement.Output "A=4",
   Statement.Output "20.3*2",
   Statement.Assignment "3=B",
   Statement.Assignment "x+y=0",
   Statement.Assignment "A=B=2",
   Statement.Assignment "T=T*T"]

-- Theorem to prove
theorem correct_statements :
  (statements.filter (λ s => 
    match s with
    | Statement.Input str => isCorrectInput str
    | Statement.Output str => isCorrectOutput str
    | Statement.Assignment str => isCorrectAssignment str
  )).map (λ s => 
    match s with
    | Statement.Input str => str
    | Statement.Output str => str
    | Statement.Assignment str => str
  ) = ["20.3*2", "T=T*T"] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1272_127243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_weight_calculation_l1272_127289

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The composition of the moon (and Mars) -/
def moon_composition : List (String × ℝ) := [("iron", 0.5), ("carbon", 0.2), ("other", 0.3)]

/-- The weight ratio of Mars to the moon -/
def mars_moon_ratio : ℝ := 2

/-- The weight of other elements on Mars in tons -/
def mars_other_elements : ℝ := 150

theorem moon_weight_calculation :
  (moon_composition.map (fun (_, p) => p)).sum = 1 ∧
  (moon_composition.find? (fun (e, _) => e = "other")).isSome ∧
  mars_moon_ratio * moon_weight = mars_other_elements / 
    ((moon_composition.find? (fun (e, _) => e = "other")).map Prod.snd).getD 0 →
  moon_weight = 250 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_weight_calculation_l1272_127289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_remaining_winnings_is_ten_l1272_127209

/-- Represents the lottery problem with given conditions --/
structure LotteryProblem where
  total_tickets : Nat
  ticket_cost : Nat
  winner_percentage : Rat
  five_dollar_winner_percentage : Rat
  grand_prize : Nat
  total_profit : Nat

/-- Calculates the average winning amount for the remaining tickets --/
def average_remaining_winnings (p : LotteryProblem) : Rat :=
  let total_spent := (p.total_tickets * p.ticket_cost : Nat)
  let total_winners := (p.winner_percentage * p.total_tickets : Rat).floor
  let five_dollar_winners := (p.five_dollar_winner_percentage * total_winners : Rat).floor
  let remaining_winners := total_winners - five_dollar_winners - 1
  let total_winnings := p.total_profit + total_spent
  let remaining_winnings := total_winnings - (5 * five_dollar_winners) - p.grand_prize
  (remaining_winnings : Rat) / remaining_winners

/-- The theorem stating that the average remaining winnings is $10 --/
theorem average_remaining_winnings_is_ten (p : LotteryProblem) 
  (h1 : p.total_tickets = 200)
  (h2 : p.ticket_cost = 2)
  (h3 : p.winner_percentage = 1/5)
  (h4 : p.five_dollar_winner_percentage = 4/5)
  (h5 : p.grand_prize = 5000)
  (h6 : p.total_profit = 4830) :
  average_remaining_winnings p = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_remaining_winnings_is_ten_l1272_127209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1272_127216

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0), prove its eccentricity is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → (x - c)^2 + y^2 = c^2) -- Focus condition
  (h5 : ∃ lambda mu : ℝ, c = (lambda + mu) * c ∧ b^2 / a = (lambda - mu) * b * c / a ∧ lambda * mu = 1/16) :
  c / a = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1272_127216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1272_127269

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + 2 * Real.pi / 3)

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = -Real.sqrt 3 ∧
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, f x ≥ min := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1272_127269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l1272_127215

def sequence_n (n : ℕ) : ℕ := 47 * (10^(2*n) - 1) / 99

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem only_first_prime :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → (is_prime (sequence_n n) ↔ n = 1) :=
by
  sorry

#eval sequence_n 1  -- To check if the function works as expected

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_l1272_127215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_widescreen_tv_horizontal_length_l1272_127248

/-- Calculates the horizontal length of a widescreen TV given its aspect ratio and diagonal length. -/
noncomputable def horizontalLength (aspectWidth aspectHeight diagonalLength : ℝ) : ℝ :=
  (aspectWidth * diagonalLength) / Real.sqrt (aspectWidth^2 + aspectHeight^2)

/-- Theorem: The horizontal length of a 16:9 aspect ratio TV with a 40-inch diagonal is 640/√337 inches. -/
theorem widescreen_tv_horizontal_length :
  horizontalLength 16 9 40 = 640 / Real.sqrt 337 := by
  -- Unfold the definition of horizontalLength
  unfold horizontalLength
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_widescreen_tv_horizontal_length_l1272_127248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1272_127211

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  0 < a ∧ 0 < b ∧ 0 < c →
  Real.sin A * Real.sin B * Real.sin C = 1 / 1000 →
  a * b * c = 1000 →
  let R := (a / Real.sin A + b / Real.sin B + c / Real.sin C) / 6
  (a * b * Real.sin C) / (4 * R) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1272_127211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_born_72_days_later_is_friday_l1272_127238

/-- Day of the week represented as an enumeration -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem born_72_days_later_is_friday 
  (john_birthday : DayOfWeek) 
  (alison_days_later : Nat) 
  (h1 : john_birthday = DayOfWeek.Wednesday) 
  (h2 : alison_days_later = 72) : 
  dayAfter john_birthday alison_days_later = DayOfWeek.Friday := by
  sorry

#check born_72_days_later_is_friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_born_72_days_later_is_friday_l1272_127238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_max_inscribed_rectangle_area_l1272_127249

-- Define the ellipse C
noncomputable def ellipse_C (φ : ℝ) : ℝ × ℝ := (5 * Real.cos φ, 3 * Real.sin φ)

-- Define the parallel line
def parallel_line (t : ℝ) : ℝ × ℝ := (4 - 2*t, 3 - t)

-- Define IsRectangle
def IsRectangle (A B C D : ℝ × ℝ) : Prop := sorry

-- Define AreaOfQuadrilateral
noncomputable def AreaOfQuadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Statement 1: Equation of line l
theorem line_l_equation : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (∃ (φ : ℝ), (x, y) = ellipse_C φ) → 
    (∃ (t : ℝ), (x, y) = parallel_line t) → 
    y = m * x + b ∧ m = 1/2 ∧ b = -2 := by sorry

-- Statement 2: Maximum area of inscribed rectangle
theorem max_inscribed_rectangle_area :
  ∃ (A B C D : ℝ × ℝ),
    (∀ (φ : ℝ), A ≠ ellipse_C φ ∧ B ≠ ellipse_C φ ∧ C ≠ ellipse_C φ ∧ D ≠ ellipse_C φ) →
    (∃ (φ₁ φ₂ φ₃ φ₄ : ℝ), A = ellipse_C φ₁ ∧ B = ellipse_C φ₂ ∧ C = ellipse_C φ₃ ∧ D = ellipse_C φ₄) →
    (IsRectangle A B C D) →
    (∀ (A' B' C' D' : ℝ × ℝ), 
      (∀ (φ : ℝ), A' ≠ ellipse_C φ ∧ B' ≠ ellipse_C φ ∧ C' ≠ ellipse_C φ ∧ D' ≠ ellipse_C φ) →
      (∃ (φ₁ φ₂ φ₃ φ₄ : ℝ), A' = ellipse_C φ₁ ∧ B' = ellipse_C φ₂ ∧ C' = ellipse_C φ₃ ∧ D' = ellipse_C φ₄) →
      (IsRectangle A' B' C' D') →
      AreaOfQuadrilateral A B C D ≥ AreaOfQuadrilateral A' B' C' D') →
    AreaOfQuadrilateral A B C D = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_equation_max_inscribed_rectangle_area_l1272_127249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1272_127247

/-- Calculates the total annual interest earned from two investments -/
noncomputable def total_interest (total_amount : ℝ) (first_part : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let second_part := total_amount - first_part
  let interest1 := first_part * (rate1 / 100)
  let interest2 := second_part * (rate2 / 100)
  interest1 + interest2

/-- Theorem stating that the total annual interest earned is 144 -/
theorem interest_calculation :
  total_interest 4000 2800 3 5 = 144 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval total_interest 4000 2800 3 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_calculation_l1272_127247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_is_four_l1272_127251

/-- Represents the movement of a lemming on a square -/
structure LemmingMovement where
  squareSide : ℝ
  diagonalMove : ℝ
  turnMove : ℝ

/-- Calculates the average distance from the lemming's final position to the sides of the square -/
noncomputable def averageDistanceToSides (m : LemmingMovement) : ℝ :=
  let finalX := m.diagonalMove / Real.sqrt 2 + m.turnMove
  let finalY := m.diagonalMove / Real.sqrt 2
  (finalX + finalY + (m.squareSide - finalX) + (m.squareSide - finalY)) / 4

/-- Theorem stating that for the given lemming movement, the average distance to sides is 4 -/
theorem lemming_average_distance_is_four :
  let m : LemmingMovement := ⟨8, 6.8, 2.5⟩
  averageDistanceToSides m = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemming_average_distance_is_four_l1272_127251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_theorem_l1272_127264

theorem zero_point_theorem (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_sign : f a * f b < 0) : 
  ∃ c ∈ Set.Ioo a b, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_theorem_l1272_127264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_exists_l1272_127268

def Answer : Type := String

def options : Set Answer := {"when", "whether", "why", "how"}

def is_correct (a : Answer) : Prop := a = "how"

theorem correct_answer_exists : ∃ (a : Answer), a ∈ options ∧ is_correct a := by
  use "how"
  constructor
  · simp [options]
  · rfl

#check correct_answer_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_exists_l1272_127268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l1272_127217

theorem matrix_transformation_proof :
  ∀ (a b c d : ℝ),
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]
  let X : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  M • X = !![2*a, 2*b; 3*c, 3*d] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_proof_l1272_127217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_of_angles_l1272_127291

theorem sin_difference_of_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = 4/5) (h4 : Real.cos β = 5/13) : Real.sin (β - α) = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_of_angles_l1272_127291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_area_pyramid_section_area_positive_pyramid_section_area_result_l1272_127288

theorem pyramid_section_area (s : ℝ) (h : s > 0) :
  let base_area := s
  let section_area := s / 4
  section_area = base_area / 4 := by
  -- Introduce the given variables
  let base_area := s
  let section_area := s / 4

  -- The proof
  calc
    section_area = s / 4 := by rfl
    _ = base_area / 4 := by rfl

theorem pyramid_section_area_positive (s : ℝ) (h : s > 0) :
  let section_area := s / 4
  section_area > 0 := by
  -- Introduce the given variable
  let section_area := s / 4

  -- The proof
  have h1 : s / 4 > 0 := by
    apply div_pos h
    norm_num
  exact h1

-- The main theorem combining both results
theorem pyramid_section_area_result (s : ℝ) (h : s > 0) :
  ∃ section_area : ℝ, section_area = s / 4 ∧ section_area > 0 := by
  use s / 4
  constructor
  · rfl
  · exact pyramid_section_area_positive s h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_area_pyramid_section_area_positive_pyramid_section_area_result_l1272_127288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1272_127208

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x - Real.pi / 6) + Real.sin (ω * x - Real.pi / 2)

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (x - Real.pi / 12)

theorem function_properties (ω : ℝ) 
  (h1 : 0 < ω) (h2 : ω < 3) (h3 : f ω (Real.pi / 6) = 0) :
  ω = 2 ∧ 
  ∃ x₀ ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4), 
    g x₀ = -3 / 2 ∧ ∀ x ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4), g x₀ ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1272_127208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1272_127253

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 1/2]

theorem inverse_of_A : 
  A⁻¹ = !![1, 0; 0, 2] := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_A_l1272_127253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_on_ellipse_l1272_127258

-- Define the ellipse P
def ellipse_P (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the right focus condition
def right_focus (a b : ℝ) : Prop :=
  a^2 - b^2 = 1

-- Define the point that the ellipse passes through
def passes_through (a b : ℝ) : Prop :=
  (2/3)^2 / a^2 + (2*Real.sqrt 6/3)^2 / b^2 = 1

-- Define the line that contains vertices B and D
def line_BD (x y : ℝ) : Prop :=
  7*x - 7*y + 1 = 0

-- Define the square ABCD
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  ellipse_P A.1 A.2 2 (Real.sqrt 3) ∧
  ellipse_P C.1 C.2 2 (Real.sqrt 3) ∧
  line_BD B.1 B.2 ∧
  line_BD D.1 D.2

-- The main theorem
theorem square_area_on_ellipse (A B C D : ℝ × ℝ) :
  square_ABCD A B C D →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (24/7)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_on_ellipse_l1272_127258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_curve_l1272_127276

/-- Represents the trajectory of a projectile. -/
structure Trajectory where
  u : ℝ  -- Initial velocity
  g : ℝ  -- Acceleration due to gravity
  φ : ℝ  -- Launch angle

/-- The x-coordinate of the projectile at time t. -/
noncomputable def x_coord (traj : Trajectory) (t : ℝ) : ℝ :=
  traj.u * t * Real.cos traj.φ

/-- The y-coordinate of the projectile at time t. -/
noncomputable def y_coord (traj : Trajectory) (t : ℝ) : ℝ :=
  traj.u * t * Real.sin traj.φ - (1/2) * traj.g * t^2

/-- The time at which the projectile reaches its highest point. -/
noncomputable def peak_time (traj : Trajectory) : ℝ :=
  traj.u * Real.sin traj.φ / traj.g

/-- The x-coordinate of the highest point of the trajectory. -/
noncomputable def peak_x (traj : Trajectory) : ℝ :=
  (traj.u^2 / (2 * traj.g)) * Real.sin (2 * traj.φ)

/-- The y-coordinate of the highest point of the trajectory. -/
noncomputable def peak_y (traj : Trajectory) : ℝ :=
  (traj.u^2 / (4 * traj.g)) * (1 - Real.cos (2 * traj.φ))

/-- The area of the closed curve formed by the highest points of all trajectories. -/
noncomputable def curve_area (traj : Trajectory) : ℝ :=
  Real.pi / 8 * (traj.u^4 / traj.g^2)

/-- Theorem stating that the area of the closed curve is π/8 · (u⁴/g²). -/
theorem area_of_closed_curve (traj : Trajectory) :
  curve_area traj = Real.pi / 8 * (traj.u^4 / traj.g^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_curve_l1272_127276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_on_assigned_day_l1272_127207

/-- Represents the percentage of students who took the exam on the assigned day -/
def x : ℝ := 70

/-- Total number of students in the class -/
def total_students : ℕ := 100

/-- Average score for students who took the exam on the assigned day -/
def assigned_day_avg : ℝ := 65

/-- Average score for students who took the exam on the make-up date -/
def makeup_day_avg : ℝ := 95

/-- Average score for the entire class -/
def class_avg : ℝ := 74

/-- Theorem stating that the percentage of students who took the exam on the assigned day is 70% -/
theorem percentage_on_assigned_day :
  x * assigned_day_avg + (total_students - x) * makeup_day_avg = total_students * class_avg :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_on_assigned_day_l1272_127207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_ten_l1272_127242

theorem arithmetic_sequence_square_root_ten (y : ℝ) :
  y > 0 ∧
  (2^2 + 4^2) / 2 = y^2 →
  y = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_ten_l1272_127242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_tan_inequality_l1272_127293

theorem min_m_for_tan_inequality : 
  ∃ (m : ℝ), (∀ x ∈ Set.Icc 0 (π/3), Real.tan x ≤ m) ∧ 
  (∀ m' : ℝ, (∀ x ∈ Set.Icc 0 (π/3), Real.tan x ≤ m') → m ≤ m') ∧ 
  m = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_tan_inequality_l1272_127293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1272_127265

def U : Type := ℝ

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 2}

theorem problem_solution :
  (∀ (a : ℝ), a = 3 →
    (A ∪ B a = {x : ℝ | 1 ≤ x ∧ x ≤ 5}) ∧
    ((B a) ∩ (Aᶜ) = {x : ℝ | 4 < x ∧ x ≤ 5})) ∧
  (∀ (a : ℝ), B a ⊆ A → 1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1272_127265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_equality_l1272_127281

-- Define the set P
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

-- Define the set Q
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- Define the complement of Q in ℝ
def complement_Q : Set ℝ := {x | x ∉ Q}

-- State the theorem
theorem set_union_equality : P ∪ complement_Q = Set.Iic (-2) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_union_equality_l1272_127281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_product_l1272_127297

theorem simplify_fraction_product (y : ℝ) (h : y ≠ 0) :
  (5 / (4 * y^(-4 : ℤ))) * ((4 * y^3) / 3) = (5 * y^7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_product_l1272_127297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_alpha_l1272_127280

theorem sin_two_pi_minus_alpha (α : Real) 
  (h1 : Real.cos (Real.pi + α) = -(1/2)) 
  (h2 : 3*Real.pi/2 < α) 
  (h3 : α < 2*Real.pi) : 
  Real.sin (2*Real.pi - α) = Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_alpha_l1272_127280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l1272_127235

theorem unique_solution_for_exponential_equation :
  ∀ k n m : ℕ,
    k > 0 → n > 0 → m ≥ 2 →
    (3 : ℤ)^k + (5 : ℤ)^k = (n : ℤ)^m →
    k = 1 ∧ n = 2 ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_for_exponential_equation_l1272_127235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1272_127252

/-- Represents a parabola of the form y = x^2 + mx - 6 --/
structure Parabola where
  m : ℝ

/-- Represents a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its base and height --/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

theorem parabola_triangle_area 
  (p : Parabola) 
  (A B C : Point) :
  A.x = -3 ∧ A.y = 0 ∧  -- Point A is (-3, 0)
  B.y = 0 ∧             -- Point B is on x-axis
  C.x = 0 ∧             -- Point C is on y-axis
  A.y = A.x^2 + p.m * A.x - 6 ∧  -- A satisfies parabola equation
  B.y = B.x^2 + p.m * B.x - 6 ∧  -- B satisfies parabola equation
  C.y = C.x^2 + p.m * C.x - 6    -- C satisfies parabola equation
  →
  triangleArea (B.x - A.x) C.y = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l1272_127252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1272_127227

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x^2 - 3 * x + 1) / Real.log (1/2)

-- Define the domain of the function
def domain : Set ℝ := {x | x < 1/2 ∨ x > 1}

-- State that log₁/₂ is decreasing
axiom log_half_decreasing : ∀ x y, x < y → f x > f y

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x ∈ domain, ∀ y ∈ domain, x > y → (f x < f y ↔ x > 1 ∧ y > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l1272_127227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l1272_127266

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)
variable (para_plane : Plane → Plane → Prop)
variable (para_line : Line → Line → Prop)

-- Define the intersection relation
variable (intersects : Line → Plane → Prop)

-- Define the given lines and planes
variable (l m : Line) (a b : Plane)

-- State the theorem
theorem only_fourth_proposition_correct 
  (h1 : perp_line_plane l a) 
  (h2 : intersects m b) : 
  (¬(para_plane a b → para_line l m)) ∧ 
  (¬(para_line l m → para_plane a b)) ∧ 
  (¬(perp_plane_plane a b → para_line l m)) ∧ 
  ((para_line l m → perp_plane_plane a b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l1272_127266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1272_127292

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 / 4^x) - (1 / 2^x) + 1

-- State the theorem
theorem f_min_max : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ≥ 3/4) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f x = 3/4) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f x ≤ 57) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f x = 57) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l1272_127292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l1272_127229

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  x * (deriv y x) - (y x + Real.sqrt ((x^2) + (y x)^2)) = 0

-- Define the general solution
def general_solution (x y C : ℝ) : Prop :=
  x^2 - 2*C*y = C^2

-- Theorem statement
theorem differential_equation_solution :
  ∀ (y : ℝ → ℝ) (x C : ℝ), differential_equation y x → 
  ∃ (y_val : ℝ), y x = y_val ∧ general_solution x y_val C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_equation_solution_l1272_127229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_theorem_l1272_127256

/-- Represents a line in a plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a plane containing lines --/
structure Plane where
  lines : List Line

/-- Checks if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Counts the number of intersection points between lines in a plane --/
def count_intersections (p : Plane) : ℕ → Prop
| 0 => ∀ l1 l2 l3, l1 ∈ p.lines → l2 ∈ p.lines → l3 ∈ p.lines → are_parallel l1 l2 ∧ are_parallel l1 l3
| 1 => ∃ l1 l2 l3, l1 ∈ p.lines ∧ l2 ∈ p.lines ∧ l3 ∈ p.lines ∧ 
       ¬(are_parallel l1 l2) ∧ ¬(are_parallel l1 l3) ∧ ¬(are_parallel l2 l3) ∧
       ∃ x y : ℝ, (y = l1.slope * x + l1.intercept) ∧ 
                  (y = l2.slope * x + l2.intercept) ∧ 
                  (y = l3.slope * x + l3.intercept)
| 2 => ∃ l1 l2 l3, l1 ∈ p.lines ∧ l2 ∈ p.lines ∧ l3 ∈ p.lines ∧ 
       are_parallel l2 l3 ∧ ¬(are_parallel l1 l2)
| 3 => ∃ l1 l2 l3, l1 ∈ p.lines ∧ l2 ∈ p.lines ∧ l3 ∈ p.lines ∧ 
       ¬(are_parallel l1 l2) ∧ ¬(are_parallel l1 l3) ∧ ¬(are_parallel l2 l3)
| _ => False

/-- Theorem stating that the number of intersections can only be 0, 1, 2, or 3 --/
theorem intersection_points_theorem (p : Plane) (h : p.lines.length = 3) :
  (∃ n : ℕ, count_intersections p n) → 
  (count_intersections p 0 ∨ count_intersections p 1 ∨ count_intersections p 2 ∨ count_intersections p 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_theorem_l1272_127256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_correct_l1272_127279

/-- Represents the six contestants in the speech competition. -/
inductive Contestant : Type
  | one | two | three | four | five | six

/-- Represents the four audience members who made guesses. -/
inductive AudienceMember : Type
  | A | B | C | D

/-- Represents a guess made by an audience member. -/
def Guess : Type := Contestant → Prop

/-- The actual winner of the competition. -/
axiom winner : Contestant

/-- A's guess: contestant 4 or 5 will win first place. -/
def guessA : Guess :=
  fun c => c = Contestant.four ∨ c = Contestant.five

/-- B's guess: contestant 3 cannot win first place. -/
def guessB : Guess :=
  fun c => c ≠ Contestant.three

/-- C's guess: contestant 1, 2, or 6 will win first place. -/
def guessC : Guess :=
  fun c => c = Contestant.one ∨ c = Contestant.two ∨ c = Contestant.six

/-- D's guess: contestants 4, 5, or 6 cannot win first place. -/
def guessD : Guess :=
  fun c => c ≠ Contestant.four ∧ c ≠ Contestant.five ∧ c ≠ Contestant.six

/-- Check if a guess is correct for the actual winner. -/
def isCorrectGuess (g : Guess) : Prop := g winner

/-- Theorem stating that D is the only person who guessed correctly. -/
theorem only_D_correct :
  isCorrectGuess guessD ∧
  ¬isCorrectGuess guessA ∧
  ¬isCorrectGuess guessB ∧
  ¬isCorrectGuess guessC :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_correct_l1272_127279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_b_in_special_triangle_l1272_127259

/-- 
Given a triangle ABC where:
- a, b, c are the lengths of the sides opposite to angles A, B, C respectively
- a, b, c form a geometric sequence
- 2c - 4a = 0

Prove that cos B = 3/4
-/
theorem cosine_b_in_special_triangle (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) 
  (hseq : b^2 = a*c) (hrel : 2*c - 4*a = 0) :
  (a^2 + c^2 - b^2) / (2*a*c) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_b_in_special_triangle_l1272_127259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ralphs_coupon_discount_l1272_127218

/-- Calculates the coupon discount percentage given the initial total, 
    discounted item price, discount percentage on the item, and final total. -/
noncomputable def coupon_discount_percentage (initial_total : ℝ) (item_price : ℝ) 
  (item_discount_percent : ℝ) (final_total : ℝ) : ℝ :=
  let discounted_item_price := item_price * (1 - item_discount_percent)
  let total_after_item_discount := initial_total - (item_price - discounted_item_price)
  let coupon_discount := total_after_item_discount - final_total
  (coupon_discount / total_after_item_discount) * 100

/-- Theorem stating that given the specific values in Ralph's purchase scenario,
    the coupon discount percentage is 10%. -/
theorem ralphs_coupon_discount :
  coupon_discount_percentage 54 20 0.2 45 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ralphs_coupon_discount_l1272_127218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_width_is_13_cm_l1272_127263

/-- Proves that the width of a rectangular box is 13 cm given specific conditions -/
theorem box_width_is_13_cm 
  (length : ℝ) 
  (height : ℝ) 
  (cube_volume : ℝ) 
  (min_cubes : ℕ) 
  (h1 : length = 10)
  (h2 : height = 5)
  (h3 : cube_volume = 5)
  (h4 : min_cubes = 130) :
  (↑min_cubes * cube_volume) / (length * height) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_width_is_13_cm_l1272_127263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_l1272_127262

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (m : ℝ) (x : ℝ) : ℝ := m

theorem monomial_coefficient :
  coefficient (-5 * Real.pi) (a^2 * b) = -5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_coefficient_l1272_127262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_q_for_positive_sum_l1272_127212

/-- An arithmetic-geometric sequence with common ratio q -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of an arithmetic-geometric sequence -/
noncomputable def SumArithmeticGeometric (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 1 else (a 1 * (1 - q^n)) / (1 - q)

/-- The theorem stating the range of q for positive sums -/
theorem range_of_q_for_positive_sum (a : ℕ → ℝ) (q : ℝ) :
  ArithmeticGeometricSequence a q →
  (∀ n : ℕ+, SumArithmeticGeometric a q n > 0) →
  q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_q_for_positive_sum_l1272_127212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1272_127234

theorem trigonometric_identities (α : ℝ) (h : Real.tan α = 2) :
  ((Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1/6) ∧
  (4 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1272_127234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1272_127210

theorem beta_value (α β : ℝ) 
  (h1 : Real.cos α = 1/7)
  (h2 : Real.cos (α + β) = -11/14)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : π/2 < α + β ∧ α + β < π) :
  β = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l1272_127210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_6_135_is_D_l1272_127260

/-- Represents the orientation of a digit --/
inductive DigitOrientation
| Standard
| Rotated135

/-- Represents the possible options for the rotated digit --/
inductive RotationOption
| A
| B
| Q
| C
| D
| E

/-- Represents the digit 6 --/
def Digit6 : ℕ := 6

/-- Defines a clockwise rotation of 135 degrees --/
def rotate135 (d : ℕ) : DigitOrientation := DigitOrientation.Rotated135

/-- Defines the correct option for the rotated digit 6 --/
def correctOption : RotationOption := RotationOption.D

/-- Theorem stating that rotating digit 6 by 135 degrees clockwise results in option D --/
theorem rotate_6_135_is_D : 
  rotate135 Digit6 = DigitOrientation.Rotated135 → correctOption = RotationOption.D :=
by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_6_135_is_D_l1272_127260
