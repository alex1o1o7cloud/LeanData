import Mathlib

namespace NUMINAMATH_CALUDE_f_composition_negative_four_l1321_132137

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + x - 2 else -Real.log x

-- State the theorem
theorem f_composition_negative_four (x : ℝ) : f (f (-4)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_four_l1321_132137


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l1321_132135

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → g x = g y ∧ x ≠ y) ↔ 
  d = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l1321_132135


namespace NUMINAMATH_CALUDE_arithmetic_sequence_64th_term_l1321_132116

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with specific properties, the 64th term is 129. -/
theorem arithmetic_sequence_64th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_3rd : a 3 = 7)
  (h_18th : a 18 = 37) :
  a 64 = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_64th_term_l1321_132116


namespace NUMINAMATH_CALUDE_binder_problem_l1321_132193

/-- Given that 18 binders can bind 900 books in 10 days, prove that 11 binders can bind 660 books in 12 days. -/
theorem binder_problem (binders_initial : ℕ) (books_initial : ℕ) (days_initial : ℕ)
  (binders_final : ℕ) (days_final : ℕ) 
  (h1 : binders_initial = 18) (h2 : books_initial = 900) (h3 : days_initial = 10)
  (h4 : binders_final = 11) (h5 : days_final = 12) :
  (books_initial * binders_final * days_final) / (binders_initial * days_initial) = 660 := by
sorry

end NUMINAMATH_CALUDE_binder_problem_l1321_132193


namespace NUMINAMATH_CALUDE_only_fourth_prop_true_l1321_132183

-- Define the propositions
def prop1 : Prop := ∀ a b m : ℝ, (a < b → a * m^2 < b * m^2)
def prop2 : Prop := ∀ p q : Prop, (p ∨ q → p ∧ q)
def prop3 : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)
def prop4 : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem statement
theorem only_fourth_prop_true : ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by
  sorry

end NUMINAMATH_CALUDE_only_fourth_prop_true_l1321_132183


namespace NUMINAMATH_CALUDE_monthly_parking_rate_l1321_132100

/-- Proves that the monthly parking rate is $24 given the specified conditions -/
theorem monthly_parking_rate (weekly_rate : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  yearly_savings = 232 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_rate : ℕ), monthly_rate = 24 ∧ weeks_per_year * weekly_rate - months_per_year * monthly_rate = yearly_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_parking_rate_l1321_132100


namespace NUMINAMATH_CALUDE_cylinder_radius_determination_l1321_132186

theorem cylinder_radius_determination (z : ℝ) : 
  let original_height : ℝ := 3
  let volume_increase (r : ℝ) : ℝ → ℝ := λ h => π * (r^2 * h - r^2 * original_height)
  ∀ r : ℝ, 
    (volume_increase r (original_height + 4) = z ∧ 
     volume_increase (r + 4) original_height = z) → 
    r = 8 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_determination_l1321_132186


namespace NUMINAMATH_CALUDE_chris_money_left_l1321_132106

/-- Calculates the money left over after purchases given the following conditions:
  * Video game cost: $60
  * Candy cost: $5
  * Babysitting pay rate: $8 per hour
  * Hours worked: 9
-/
def money_left_over (video_game_cost : ℕ) (candy_cost : ℕ) (pay_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  pay_rate * hours_worked - (video_game_cost + candy_cost)

theorem chris_money_left : money_left_over 60 5 8 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_left_l1321_132106


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1321_132103

theorem arithmetic_mean_problem (a b c : ℝ) :
  (a + b + c + 97) / 4 = 85 →
  (a + b + c) / 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1321_132103


namespace NUMINAMATH_CALUDE_unique_intersection_point_f_bijective_f_inv_is_inverse_l1321_132164

/-- The cubic function f(x) = x^3 + 6x^2 + 9x + 15 -/
def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 9*x + 15

/-- The theorem stating that the unique intersection point of f and its inverse is (-3, -3) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) := by
  sorry

/-- The function f is bijective -/
theorem f_bijective : Function.Bijective f := by
  sorry

/-- The inverse function of f exists -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

/-- The theorem stating that f_inv is indeed the inverse of f -/
theorem f_inv_is_inverse :
  Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_f_bijective_f_inv_is_inverse_l1321_132164


namespace NUMINAMATH_CALUDE_five_number_average_l1321_132161

theorem five_number_average (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a + b + c = 48 →
  a = 2 * b →
  (d + e) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_five_number_average_l1321_132161


namespace NUMINAMATH_CALUDE_sugar_calculation_l1321_132160

theorem sugar_calculation (standard_sugar : ℚ) (reduced_sugar : ℚ) : 
  standard_sugar = 10/3 → 
  reduced_sugar = (1/3) * standard_sugar →
  reduced_sugar = 10/9 :=
by sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1321_132160


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1321_132117

/-- A cone with three spheres inside it satisfying specific conditions -/
structure ConeWithSpheres where
  R : ℝ  -- Radius of the base of the cone
  r : ℝ  -- Radius of each sphere
  h : ℝ  -- Height of the cone
  -- The diameter of the base of the cone is equal to the slant height
  diam_eq_slant : R * 2 = Real.sqrt (R^2 + h^2)
  -- The spheres touch each other externally
  spheres_touch : True
  -- Two spheres touch the lateral surface and the base of the cone
  two_spheres_touch_base : True
  -- The third sphere touches the lateral surface at a point lying in the same plane with the centers of the spheres
  third_sphere_touch : True

/-- The ratio of the radius of the base of the cone to the radius of a sphere is (5/4 + √3) -/
theorem cone_sphere_ratio (c : ConeWithSpheres) : c.R / c.r = 5/4 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1321_132117


namespace NUMINAMATH_CALUDE_S_congruence_l1321_132123

def is_valid_N (N : ℕ) : Prop :=
  300 ≤ N ∧ N ≤ 600

def base_4_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 16, (N / 4) % 4, N % 4)

def base_7_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 49, (N / 7) % 7, N % 7)

def S (N : ℕ) : ℕ :=
  let (a₁, a₂, a₃) := base_4_repr N
  let (b₁, b₂, b₃) := base_7_repr N
  16 * a₁ + 4 * a₂ + a₃ + 49 * b₁ + 7 * b₂ + b₃

theorem S_congruence (N : ℕ) (h : is_valid_N N) :
  S N % 100 = (3 * N) % 100 ↔ (base_4_repr N).2.2 + (base_7_repr N).2.2 ≡ 3 * N [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_S_congruence_l1321_132123


namespace NUMINAMATH_CALUDE_apple_transport_trucks_l1321_132196

theorem apple_transport_trucks (total_apples : ℕ) (transported_apples : ℕ) (truck_capacity : ℕ) 
  (h1 : total_apples = 80)
  (h2 : transported_apples = 56)
  (h3 : truck_capacity = 4)
  : (total_apples - transported_apples) / truck_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_transport_trucks_l1321_132196


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l1321_132194

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n / 100 % 10) * 10 + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_five_digit n ∧ (n % (remove_middle_digit n) = 0)

theorem five_digit_divisibility :
  ∀ n : ℕ, satisfies_condition n ↔ ∃ N : ℕ, 10 ≤ N ∧ N ≤ 99 ∧ n = N * 1000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l1321_132194


namespace NUMINAMATH_CALUDE_sequences_satisfy_conditions_l1321_132178

-- Define the sequences A and B
def A (n : ℕ) : ℝ × ℝ := (n, n^3)
def B (n : ℕ) : ℝ × ℝ := (-n, -n^3)

-- Define a function to check if a point is on a line through two other points
def is_on_line (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

-- State the theorem
theorem sequences_satisfy_conditions :
  ∀ (i j k : ℕ), 1 ≤ i → i < j → j < k →
    (is_on_line (A i) (A j) (B k) ↔ k = i + j) ∧
    (is_on_line (B i) (B j) (A k) ↔ k = i + j) :=
by sorry

end NUMINAMATH_CALUDE_sequences_satisfy_conditions_l1321_132178


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1321_132128

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) : ℕ → ℚ := λ n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℚ) 
  (h_arith : ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h_a₁ : a 1 = 1/2) 
  (h_S₂ : arithmeticSum (a 1) (a 2 - a 1) 2 = a 3) :
  ∀ (n : ℕ), arithmeticSum (a 1) (a 2 - a 1) n = 1/4 * n^2 + 1/4 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1321_132128


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1321_132195

/-- A hyperbola with one asymptote defined by x±y=0 and passing through (-1,-2) -/
structure Hyperbola where
  /-- One asymptote of the hyperbola is defined by x±y=0 -/
  asymptote : ∀ (x y : ℝ), x = y ∨ x = -y
  /-- The hyperbola passes through the point (-1,-2) -/
  passes_through : ∃ (f : ℝ → ℝ → ℝ), f (-1) (-2) = 0

/-- The standard equation of the hyperbola is y²/3 - x²/3 = 1 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  ∃ (f : ℝ → ℝ → ℝ), (∀ x y, f x y = y^2/3 - x^2/3 - 1) ∧ (∀ x y, f x y = 0 ↔ h.passes_through.choose x y = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1321_132195


namespace NUMINAMATH_CALUDE_range_of_a_l1321_132127

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * x ≥ -x + Real.log (a * x)) ↔ (0 < a ∧ a ≤ Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1321_132127


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1321_132102

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1321_132102


namespace NUMINAMATH_CALUDE_angle_B_is_30_degrees_l1321_132157

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + t.b * Real.cos t.C = t.a * Real.sin t.A ∧
  (Real.sqrt 3 / 4) * (t.b^2 + t.a^2 - t.c^2) = (1/2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem angle_B_is_30_degrees (t : Triangle) 
  (h : satisfies_conditions t) : t.B = 30 * (Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_30_degrees_l1321_132157


namespace NUMINAMATH_CALUDE_exists_constant_function_l1321_132187

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem exists_constant_function (x : ℝ) : ∃ k : ℝ, 2 * f 3 - 10 = f k ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_constant_function_l1321_132187


namespace NUMINAMATH_CALUDE_reflect_F_twice_l1321_132125

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem stating that reflecting point F(1,3) over y-axis then x-axis results in F''(-1,-3) -/
theorem reflect_F_twice :
  let F : ℝ × ℝ := (1, 3)
  let F' := reflect_y F
  let F'' := reflect_x F'
  F'' = (-1, -3) := by sorry

end NUMINAMATH_CALUDE_reflect_F_twice_l1321_132125


namespace NUMINAMATH_CALUDE_total_coin_value_l1321_132121

-- Define the number of rolls for each coin type
def quarters_rolls : ℕ := 5
def dimes_rolls : ℕ := 4
def nickels_rolls : ℕ := 3
def pennies_rolls : ℕ := 2

-- Define the number of coins in each roll
def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def nickels_per_roll : ℕ := 40
def pennies_per_roll : ℕ := 50

-- Define the value of each coin in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Calculate the total value in cents
def total_value : ℕ :=
  quarters_rolls * quarters_per_roll * quarter_value +
  dimes_rolls * dimes_per_roll * dime_value +
  nickels_rolls * nickels_per_roll * nickel_value +
  pennies_rolls * pennies_per_roll * penny_value

-- Theorem to prove
theorem total_coin_value : total_value = 7700 := by
  sorry

end NUMINAMATH_CALUDE_total_coin_value_l1321_132121


namespace NUMINAMATH_CALUDE_series_evaluation_l1321_132158

open Real

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ)^2 / 3^k

theorem series_evaluation : series_sum = 7 := by sorry

end NUMINAMATH_CALUDE_series_evaluation_l1321_132158


namespace NUMINAMATH_CALUDE_product_defect_rate_l1321_132173

theorem product_defect_rate (stage1_defect_rate stage2_defect_rate : ℝ) 
  (h1 : stage1_defect_rate = 0.1)
  (h2 : stage2_defect_rate = 0.03) :
  1 - (1 - stage1_defect_rate) * (1 - stage2_defect_rate) = 0.127 := by
  sorry

end NUMINAMATH_CALUDE_product_defect_rate_l1321_132173


namespace NUMINAMATH_CALUDE_hyperbola_focus_directrix_distance_example_l1321_132146

/-- The distance between the right focus and left directrix of a hyperbola -/
def hyperbola_focus_directrix_distance (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  5

/-- Theorem: The distance between the right focus and left directrix of the hyperbola x²/4 - y²/12 = 1 is 5 -/
theorem hyperbola_focus_directrix_distance_example :
  hyperbola_focus_directrix_distance 2 (2 * Real.sqrt 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_directrix_distance_example_l1321_132146


namespace NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l1321_132115

theorem soccer_camp_afternoon_attendance (total_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : ∃ soccer_kids : ℕ, soccer_kids = total_kids / 2)
  (h3 : ∃ morning_kids : ℕ, morning_kids = soccer_kids / 4) :
  ∃ afternoon_kids : ℕ, afternoon_kids = 750 :=
by sorry

end NUMINAMATH_CALUDE_soccer_camp_afternoon_attendance_l1321_132115


namespace NUMINAMATH_CALUDE_tangent_lines_to_parabola_l1321_132153

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 9

-- Define the point B
def B : ℝ × ℝ := (-1, 2)

-- Define the two lines
def line1 (x : ℝ) : ℝ := -2*x
def line2 (x : ℝ) : ℝ := 6*x + 8

-- Theorem statement
theorem tangent_lines_to_parabola :
  (∃ x₀ : ℝ, line1 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line1 x < parabola x) ∧
    line1 (B.1) = B.2) ∧
  (∃ x₀ : ℝ, line2 x₀ = parabola x₀ ∧ 
    (∀ x : ℝ, x ≠ x₀ → line2 x < parabola x) ∧
    line2 (B.1) = B.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_parabola_l1321_132153


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1321_132109

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l1321_132109


namespace NUMINAMATH_CALUDE_correct_answer_l1321_132168

theorem correct_answer (x : ℝ) (h : x / 3 = 27) : x * 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l1321_132168


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1321_132199

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/4 + y²/3 = 1 -/
def Ellipse (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Checks if two points are symmetric about the x-axis -/
def SymmetricAboutXAxis (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

/-- Represents a line passing through two points -/
def Line (p1 p2 : Point) : Point → Prop :=
  λ p => (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)

/-- Checks if a point is on the x-axis -/
def OnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Checks if a line intersects the ellipse -/
def IntersectsEllipse (l : Point → Prop) : Prop :=
  ∃ p, l p ∧ Ellipse p

/-- Main theorem -/
theorem ellipse_intersection_theorem (D E A B : Point) 
    (hDE : SymmetricAboutXAxis D E) 
    (hD : Ellipse D) (hE : Ellipse E)
    (hA : OnXAxis A) (hB : OnXAxis B)
    (hDA : ¬IntersectsEllipse (Line D A))
    (hInt : IntersectsEllipse (Line D A) ∧ IntersectsEllipse (Line B E) ∧ 
            ∃ p, Line D A p ∧ Line B E p ∧ Ellipse p) :
    A.x * B.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1321_132199


namespace NUMINAMATH_CALUDE_distinct_roots_sum_bound_l1321_132145

theorem distinct_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 8 = 0 → 
  r₂^2 + p*r₂ + 8 = 0 → 
  |r₁ + r₂| > 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_bound_l1321_132145


namespace NUMINAMATH_CALUDE_find_x_l1321_132132

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) 
  (h : (2 * a) ^ (2 * b) = a ^ b * x ^ (2 * b)) : x = 2 * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1321_132132


namespace NUMINAMATH_CALUDE_partner_a_share_l1321_132140

/-- Calculates the share of profit for a partner in a partnership --/
def calculate_share (investment_a investment_b investment_c profit_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let total_profit := (profit_b * total_investment) / investment_b
  (investment_a * total_profit) / total_investment

theorem partner_a_share :
  let investment_a : ℚ := 7000
  let investment_b : ℚ := 11000
  let investment_c : ℚ := 18000
  let profit_b : ℚ := 2200
  calculate_share investment_a investment_b investment_c profit_b = 1400 := by
  sorry

#eval calculate_share 7000 11000 18000 2200

end NUMINAMATH_CALUDE_partner_a_share_l1321_132140


namespace NUMINAMATH_CALUDE_gcd_105_88_l1321_132119

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_88_l1321_132119


namespace NUMINAMATH_CALUDE_expand_cubic_sum_product_l1321_132150

theorem expand_cubic_sum_product (x : ℝ) : (x^3 + 3) * (x^3 + 4) = x^6 + 7*x^3 + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_cubic_sum_product_l1321_132150


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l1321_132198

/-- Calculates the maximum number of tiles that can fit on a rectangular floor --/
def max_tiles (floor_length floor_width tile_length tile_width : ℕ) : ℕ :=
  let orientation1 := (floor_length / tile_length) * (floor_width / tile_width)
  let orientation2 := (floor_length / tile_width) * (floor_width / tile_length)
  max orientation1 orientation2

/-- Theorem stating the maximum number of tiles that can be accommodated on the given floor --/
theorem max_tiles_on_floor :
  max_tiles 180 120 25 16 = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l1321_132198


namespace NUMINAMATH_CALUDE_inequality_proof_l1321_132107

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1321_132107


namespace NUMINAMATH_CALUDE_complex_fourth_power_integer_count_l1321_132118

theorem complex_fourth_power_integer_count : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + 2 * Complex.I) ^ 4 = m := by sorry

end NUMINAMATH_CALUDE_complex_fourth_power_integer_count_l1321_132118


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1321_132166

theorem rope_cutting_problem :
  let rope1 : ℕ := 44
  let rope2 : ℕ := 54
  let rope3 : ℕ := 74
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 2 := by
sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1321_132166


namespace NUMINAMATH_CALUDE_a_8_equals_8_l1321_132149

def sequence_property (a : ℕ+ → ℕ) : Prop :=
  ∀ (s t : ℕ+), a (s * t) = a s * a t

theorem a_8_equals_8 (a : ℕ+ → ℕ) (h1 : sequence_property a) (h2 : a 2 = 2) : 
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_8_l1321_132149


namespace NUMINAMATH_CALUDE_fourth_power_trinomial_coefficients_l1321_132134

/-- A trinomial that is an exact fourth power for all integers -/
def is_fourth_power (a b c : ℝ) : Prop :=
  ∀ x : ℤ, ∃ y : ℝ, a * x^2 + b * x + c = y^4

/-- If a trinomial is an exact fourth power for all integers, then its quadratic and linear coefficients are zero -/
theorem fourth_power_trinomial_coefficients (a b c : ℝ) :
  is_fourth_power a b c → a = 0 ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_power_trinomial_coefficients_l1321_132134


namespace NUMINAMATH_CALUDE_factor_expression_l1321_132184

theorem factor_expression (x : ℝ) : 3*x*(x-5) + 4*(x-5) + 6*x = (3*x + 4)*(x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1321_132184


namespace NUMINAMATH_CALUDE_herd_size_l1321_132114

theorem herd_size (bulls : ℕ) (h : bulls = 70) : 
  (2 / 3 : ℚ) * (1 / 3 : ℚ) * (total_herd : ℚ) = bulls → total_herd = 315 := by
  sorry

end NUMINAMATH_CALUDE_herd_size_l1321_132114


namespace NUMINAMATH_CALUDE_circle_properties_l1321_132192

theorem circle_properties (A : ℝ) (h : A = 64 * Real.pi) : ∃ (r C : ℝ), r = 8 ∧ C = 16 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1321_132192


namespace NUMINAMATH_CALUDE_sausage_distance_ratio_l1321_132136

/-- Represents the scenario of a dog and cat running towards sausages --/
structure SausageScenario where
  dog_speed : ℝ
  cat_speed : ℝ
  dog_eat_rate : ℝ
  cat_eat_rate : ℝ
  total_sausages : ℝ
  total_distance : ℝ

/-- The theorem to be proved --/
theorem sausage_distance_ratio 
  (scenario : SausageScenario)
  (h1 : scenario.cat_speed = 2 * scenario.dog_speed)
  (h2 : scenario.dog_eat_rate = scenario.cat_eat_rate / 2)
  (h3 : scenario.cat_eat_rate * 1 = scenario.total_sausages)
  (h4 : scenario.cat_speed * 1 = scenario.total_distance)
  (h5 : scenario.total_sausages > 0)
  (h6 : scenario.total_distance > 0) :
  ∃ (cat_distance dog_distance : ℝ),
    cat_distance + dog_distance = scenario.total_distance ∧
    cat_distance / dog_distance = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_sausage_distance_ratio_l1321_132136


namespace NUMINAMATH_CALUDE_square_sum_identity_l1321_132163

theorem square_sum_identity (a b : ℝ) : a^2 + b^2 = (a + b)^2 + (-2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l1321_132163


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1321_132171

theorem solution_set_inequality (x : ℝ) :
  (((1 - 2*x) / (3*x^2 - 4*x + 7)) ≥ 0) ↔ (x ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1321_132171


namespace NUMINAMATH_CALUDE_revenue_change_l1321_132120

/-- Proves that given a 75% price decrease and a specific ratio between percent increase in units sold
    and percent decrease in price, the new revenue is 50% of the original revenue -/
theorem revenue_change (P Q : ℝ) (P' Q' : ℝ) (h1 : P' = 0.25 * P) 
    (h2 : (Q' / Q - 1) / 0.75 = 1.3333333333333333) : P' * Q' = 0.5 * P * Q := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_l1321_132120


namespace NUMINAMATH_CALUDE_jordan_running_time_l1321_132142

/-- Given that Jordan ran 3 miles in 1/3 of the time it took Steve to run 4 miles,
    and Steve ran 4 miles in 32 minutes, prove that Jordan would take 224/9 minutes
    to run 7 miles. -/
theorem jordan_running_time (steve_time : ℝ) (jordan_distance : ℝ) :
  steve_time = 32 →
  jordan_distance = 7 →
  (3 / (1/3 * steve_time)) * jordan_distance = 224/9 := by
  sorry

end NUMINAMATH_CALUDE_jordan_running_time_l1321_132142


namespace NUMINAMATH_CALUDE_apple_cost_calculation_apple_cost_proof_l1321_132188

/-- Calculates the total cost of apples for a family after a price increase -/
theorem apple_cost_calculation (original_price : ℝ) (price_increase : ℝ) 
  (family_size : ℕ) (pounds_per_person : ℝ) : ℝ :=
  let new_price := original_price * (1 + price_increase)
  let total_pounds := (family_size : ℝ) * pounds_per_person
  new_price * total_pounds

/-- Proves that the total cost for the given scenario is $16 -/
theorem apple_cost_proof : 
  apple_cost_calculation 1.6 0.25 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_apple_cost_proof_l1321_132188


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1321_132181

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Statement to prove
theorem events_mutually_exclusive_but_not_complementary :
  ∃ (d : Distribution),
    (∀ (p : Person), ∃! (c : Card), d p = c) →
    (¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧
    (∃ (d' : Distribution), ¬(A_gets_clubs d') ∧ ¬(B_gets_clubs d')) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l1321_132181


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l1321_132175

theorem triangle_circles_area_sum (r s t : ℝ) : 
  r > 0 ∧ s > 0 ∧ t > 0 →
  r + s = 5 →
  r + t = 12 →
  s + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π :=
by sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l1321_132175


namespace NUMINAMATH_CALUDE_club_equation_solution_l1321_132190

-- Define the operation ♣
def club (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 5

-- Theorem statement
theorem club_equation_solution :
  ∃ B : ℝ, club 4 B = 101 ∧ B = 24 := by
  sorry

end NUMINAMATH_CALUDE_club_equation_solution_l1321_132190


namespace NUMINAMATH_CALUDE_num_distinct_paths_l1321_132152

/-- The number of rows in the grid -/
def rows : ℕ := 6

/-- The number of columns in the grid -/
def cols : ℕ := 5

/-- The number of dominoes used -/
def num_dominoes : ℕ := 5

/-- The number of moves to the right required to reach the bottom right corner -/
def moves_right : ℕ := cols - 1

/-- The number of moves down required to reach the bottom right corner -/
def moves_down : ℕ := rows - 1

/-- The total number of moves required to reach the bottom right corner -/
def total_moves : ℕ := moves_right + moves_down

/-- Theorem stating the number of distinct paths from top-left to bottom-right corner -/
theorem num_distinct_paths : (total_moves.choose moves_right) = 126 := by
  sorry

end NUMINAMATH_CALUDE_num_distinct_paths_l1321_132152


namespace NUMINAMATH_CALUDE_rancher_cows_count_l1321_132110

theorem rancher_cows_count (horses : ℕ) (cows : ℕ) : 
  cows = 5 * horses →
  cows + horses = 168 →
  cows = 140 := by
sorry

end NUMINAMATH_CALUDE_rancher_cows_count_l1321_132110


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1321_132170

/-- The vertex coordinates of the parabola y = x^2 - 4x + 3 are (2, -1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  ∃ x y : ℝ, (x = 2 ∧ y = -1) ∧
    (∀ t : ℝ, f t ≥ f x) ∧
    (y = f x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1321_132170


namespace NUMINAMATH_CALUDE_water_flow_proof_l1321_132159

theorem water_flow_proof (rate_second : ℝ) (total_flow : ℝ) : 
  rate_second = 36 →
  ∃ (rate_first rate_third : ℝ),
    rate_second = rate_first * 1.5 ∧
    rate_third = rate_second * 1.25 ∧
    total_flow = rate_first + rate_second + rate_third ∧
    total_flow = 105 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_proof_l1321_132159


namespace NUMINAMATH_CALUDE_diamond_club_evaluation_l1321_132139

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := (3 * a + b) / (a - b)

-- Define the club operation
def club (a b : ℚ) : ℚ := 2

-- Theorem statement
theorem diamond_club_evaluation :
  club (diamond 4 6) (diamond 7 5) = 2 := by sorry

end NUMINAMATH_CALUDE_diamond_club_evaluation_l1321_132139


namespace NUMINAMATH_CALUDE_ratio_equality_l1321_132143

theorem ratio_equality (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l1321_132143


namespace NUMINAMATH_CALUDE_gala_arrangement_count_l1321_132179

/-- The number of programs in the New Year's gala. -/
def total_programs : ℕ := 8

/-- The number of non-singing programs in the New Year's gala. -/
def non_singing_programs : ℕ := 3

/-- The number of singing programs in the New Year's gala. -/
def singing_programs : ℕ := total_programs - non_singing_programs

/-- A function that calculates the number of ways to arrange the programs
    such that non-singing programs are not adjacent and the first and last
    programs are singing programs. -/
def arrangement_count : ℕ :=
  Nat.choose (total_programs - 2) non_singing_programs *
  Nat.factorial non_singing_programs *
  Nat.factorial (singing_programs - 2)

/-- Theorem stating that the number of ways to arrange the programs
    under the given conditions is 720. -/
theorem gala_arrangement_count :
  arrangement_count = 720 :=
by sorry

end NUMINAMATH_CALUDE_gala_arrangement_count_l1321_132179


namespace NUMINAMATH_CALUDE_green_marbles_count_l1321_132112

theorem green_marbles_count (total : ℕ) (white : ℕ) 
  (h1 : white = 40)
  (h2 : (1 : ℚ) / 4 + (1 : ℚ) / 3 + (1 : ℚ) / 6 + (white : ℚ) / total = 1) :
  ⌊(1 : ℚ) / 6 * total⌋ = 27 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_count_l1321_132112


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1321_132156

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1321_132156


namespace NUMINAMATH_CALUDE_z_sixth_power_l1321_132144

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_z_sixth_power_l1321_132144


namespace NUMINAMATH_CALUDE_min_abs_sum_l1321_132104

theorem min_abs_sum (x₁ x₂ : ℝ) 
  (h : (2 + Real.sin x₁) * (2 + Real.sin (2 * x₂)) = 1) : 
  ∃ (k m : ℤ), |x₁ + x₂| ≥ π / 4 ∧ 
  |x₁ + x₂| = π / 4 ↔ x₁ = 3 * π / 2 + 2 * π * k ∧ x₂ = 3 * π / 4 + π * m := by
  sorry

end NUMINAMATH_CALUDE_min_abs_sum_l1321_132104


namespace NUMINAMATH_CALUDE_complex_magnitude_l1321_132122

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem complex_magnitude (h : z * i^2023 = 1 + i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1321_132122


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l1321_132101

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (θ.cos : ℝ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l1321_132101


namespace NUMINAMATH_CALUDE_problem_solution_l1321_132154

open Real

noncomputable def f (a k x : ℝ) : ℝ := a^x - (k-1)*a^(-x)

theorem problem_solution (a k : ℝ) (h_a : a > 0) (h_a_neq_1 : a ≠ 1)
  (h_odd : ∀ x, f a k (-x) = -f a k x) :
  (k = 2) ∧
  (f a k 1 < 0 →
    ∀ t, (∀ x, f a k (x^2 + t*x) + f a k (4 - x) < 0) ↔ -3 < t ∧ t < 5) ∧
  (f a k 1 = 3/2 →
    ∃ m, (∀ x ≥ 1, a^(2*x) + a^(-2*x) - m * f a k x ≥ 5/4) ∧
         (∃ x ≥ 1, a^(2*x) + a^(-2*x) - m * f a k x = 5/4) ∧
         m = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1321_132154


namespace NUMINAMATH_CALUDE_max_min_sum_l1321_132131

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval
def I : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem max_min_sum (M m : ℝ) 
  (hM : ∀ x ∈ I, f x ≤ M) 
  (hm : ∀ x ∈ I, m ≤ f x) 
  (hMexists : ∃ x ∈ I, f x = M) 
  (hmexists : ∃ x ∈ I, f x = m) : 
  M + m = -14 := by sorry

end NUMINAMATH_CALUDE_max_min_sum_l1321_132131


namespace NUMINAMATH_CALUDE_tan_ratio_equals_two_l1321_132105

theorem tan_ratio_equals_two (a β : ℝ) (h : 3 * Real.sin β = Real.sin (2 * a + β)) :
  Real.tan (a + β) / Real.tan a = 2 := by sorry

end NUMINAMATH_CALUDE_tan_ratio_equals_two_l1321_132105


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_l1321_132130

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere3D where
  center : Point3D
  radius : ℝ

/-- The theorem to be proved -/
theorem plane_sphere_intersection (a b c p q r : ℝ) 
  (plane : Plane3D) 
  (sphere : Sphere3D) : 
  (plane.a = a ∧ plane.b = b ∧ plane.c = c) →  -- Plane passes through (a,b,c)
  (∃ (α β γ : ℝ), 
    plane.a * α + plane.b * 0 + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects x-axis at (α,0,0)
    plane.a * 0 + plane.b * β + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects y-axis at (0,β,0)
    plane.a * 0 + plane.b * 0 + plane.c * γ + plane.d = 0) →  -- Plane intersects z-axis at (0,0,γ)
  (sphere.center = Point3D.mk (p+1) (q+1) (r+1)) →  -- Sphere center is shifted by (1,1,1)
  (∃ (α β γ : ℝ), 
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through origin
    sphere.radius^2 = ((p+1) - α)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through A
    sphere.radius^2 = (p+1)^2 + ((q+1) - β)^2 + (r+1)^2 ∧  -- Sphere passes through B
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + ((r+1) - γ)^2) →  -- Sphere passes through C
  a/p + b/q + c/r = 2 := by
  sorry


end NUMINAMATH_CALUDE_plane_sphere_intersection_l1321_132130


namespace NUMINAMATH_CALUDE_math_competition_proof_l1321_132180

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score : ℕ := 2 * sammy_score
  let cher_score : ℕ := 2 * gab_score
  let alex_score : ℕ := cher_score + (cher_score / 10)
  let combined_score : ℕ := sammy_score + gab_score + cher_score + alex_score
  combined_score - opponent_score = 143

theorem math_competition_proof :
  math_competition 20 85 := by sorry

end NUMINAMATH_CALUDE_math_competition_proof_l1321_132180


namespace NUMINAMATH_CALUDE_mean_of_cubic_solutions_l1321_132185

theorem mean_of_cubic_solutions (x : ℝ) :
  x^3 + 2*x^2 - 13*x - 10 = 0 →
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ y ∈ s, y^3 + 2*y^2 - 13*y - 10 = 0) ∧
  (s.sum id) / s.card = -1 :=
sorry

end NUMINAMATH_CALUDE_mean_of_cubic_solutions_l1321_132185


namespace NUMINAMATH_CALUDE_initial_workers_count_l1321_132191

/-- Represents the productivity of workers in digging holes -/
structure DiggingProductivity where
  initialWorkers : ℕ
  initialDepth : ℝ
  initialTime : ℝ
  newDepth : ℝ
  newTime : ℝ
  extraWorkers : ℕ

/-- Proves that the initial number of workers is 45 given the conditions -/
theorem initial_workers_count (p : DiggingProductivity) 
  (h1 : p.initialDepth = 30)
  (h2 : p.initialTime = 8)
  (h3 : p.newDepth = 45)
  (h4 : p.newTime = 6)
  (h5 : p.extraWorkers = 45)
  (h6 : p.initialWorkers > 0)
  (h7 : p.initialDepth > 0)
  (h8 : p.initialTime > 0)
  (h9 : p.newDepth > 0)
  (h10 : p.newTime > 0) :
  p.initialWorkers = 45 := by
  sorry


end NUMINAMATH_CALUDE_initial_workers_count_l1321_132191


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1321_132129

/-- Proves that the ratio of Sophie's jellybeans to Caleb's jellybeans is 1:2 -/
theorem jellybean_ratio (caleb_dozens : ℕ) (total : ℕ) : 
  caleb_dozens = 3 → total = 54 → 
  (total - caleb_dozens * 12) / (caleb_dozens * 12) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1321_132129


namespace NUMINAMATH_CALUDE_restaurant_group_size_l1321_132126

theorem restaurant_group_size :
  ∀ (adult_meal_cost : ℕ) (num_kids : ℕ) (total_cost : ℕ),
    adult_meal_cost = 3 →
    num_kids = 7 →
    total_cost = 15 →
    ∃ (num_adults : ℕ),
      num_adults * adult_meal_cost = total_cost ∧
      num_adults + num_kids = 12 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_size_l1321_132126


namespace NUMINAMATH_CALUDE_max_parts_three_planes_correct_l1321_132169

/-- The maximum number of parts into which three non-overlapping planes can divide space -/
def max_parts_three_planes : ℕ := 8

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this statement

/-- A configuration of three non-overlapping planes in 3D space -/
structure ThreePlaneConfiguration where
  plane1 : Plane3D
  plane2 : Plane3D
  plane3 : Plane3D
  non_overlapping : plane1 ≠ plane2 ∧ plane1 ≠ plane3 ∧ plane2 ≠ plane3

/-- The number of parts into which a configuration of three planes divides space -/
def num_parts (config : ThreePlaneConfiguration) : ℕ :=
  sorry -- The actual calculation would go here

theorem max_parts_three_planes_correct :
  ∀ (config : ThreePlaneConfiguration), num_parts config ≤ max_parts_three_planes ∧
  ∃ (config : ThreePlaneConfiguration), num_parts config = max_parts_three_planes :=
sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_correct_l1321_132169


namespace NUMINAMATH_CALUDE_tangent_angle_parabola_l1321_132197

/-- The angle of inclination of the tangent to y = x^2 at (1/2, 1/4) is 45° -/
theorem tangent_angle_parabola : 
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1/2
  let y₀ : ℝ := 1/4
  let m := (deriv f) x₀
  let θ := Real.arctan m
  θ = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_parabola_l1321_132197


namespace NUMINAMATH_CALUDE_solution_for_equation_l1321_132141

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem solution_for_equation (x : ℝ) (h1 : 0 < x) (h2 : x ≠ 1) :
  x^2 * log x 27 * log 9 x = x + 4 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_for_equation_l1321_132141


namespace NUMINAMATH_CALUDE_sector_area_l1321_132113

theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  area = (1 / 2) * 1 * θ →
  area = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l1321_132113


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1321_132108

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

theorem smallest_two_digit_prime_with_composite_reverse :
  ∃ (n : ℕ), is_two_digit n ∧ is_prime n ∧ ¬(is_prime (reverse_digits n)) ∧
  (∀ m : ℕ, is_two_digit m → is_prime m → m < n → is_prime (reverse_digits m)) ∧
  n = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1321_132108


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l1321_132176

/-- Greatest Common Factor of two natural numbers -/
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

/-- Least Common Multiple of two natural numbers -/
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

/-- Theorem: The GCF of the LCM of (9, 21) and the LCM of (8, 15) is 3 -/
theorem gcf_of_lcms : GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l1321_132176


namespace NUMINAMATH_CALUDE_curve_C_cartesian_equation_l1321_132124

/-- Given a curve C in polar coordinates, prove its Cartesian equation --/
theorem curve_C_cartesian_equation (ρ θ : ℝ) (h : ρ = ρ * Real.cos θ + 2) :
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ y^2 = 4*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_cartesian_equation_l1321_132124


namespace NUMINAMATH_CALUDE_dave_total_rides_l1321_132167

/-- The number of rides Dave went on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave went on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave went on over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides :
  total_rides = 7 := by sorry

end NUMINAMATH_CALUDE_dave_total_rides_l1321_132167


namespace NUMINAMATH_CALUDE_angle_sum_is_420_l1321_132138

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfig where
  A : Real
  B : Real
  C : Real
  D : Real
  E : Real
  F : Real

/-- The theorem stating that if angle E is 30 degrees, then the sum of all angles is 420 degrees -/
theorem angle_sum_is_420 (config : GeometricConfig) (h : config.E = 30) :
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end NUMINAMATH_CALUDE_angle_sum_is_420_l1321_132138


namespace NUMINAMATH_CALUDE_inequality_of_powers_l1321_132172

theorem inequality_of_powers (a b c d x : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d ≥ 0) 
  (h5 : a + d = b + c) (h6 : x > 0) : 
  x^a + x^d ≥ x^b + x^c := by
sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l1321_132172


namespace NUMINAMATH_CALUDE_patricia_hair_donation_l1321_132147

/-- Calculates the amount of hair to donate given the current length, additional growth, and desired final length -/
def hair_to_donate (current_length additional_growth final_length : ℕ) : ℕ :=
  (current_length + additional_growth) - final_length

/-- Proves that Patricia needs to donate 23 inches of hair -/
theorem patricia_hair_donation :
  let current_length : ℕ := 14
  let additional_growth : ℕ := 21
  let final_length : ℕ := 12
  hair_to_donate current_length additional_growth final_length = 23 := by
  sorry

end NUMINAMATH_CALUDE_patricia_hair_donation_l1321_132147


namespace NUMINAMATH_CALUDE_pictures_per_album_l1321_132155

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) (pictures_per_album : ℕ) : 
  total_pictures = 24 → 
  num_albums = 4 → 
  total_pictures = num_albums * pictures_per_album →
  pictures_per_album = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l1321_132155


namespace NUMINAMATH_CALUDE_sport_water_amount_l1321_132165

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

theorem sport_water_amount (corn_syrup_amount : ℚ) :
  corn_syrup_amount = 5 →
  sport.water / sport.corn_syrup * corn_syrup_amount = 75 := by
sorry

end NUMINAMATH_CALUDE_sport_water_amount_l1321_132165


namespace NUMINAMATH_CALUDE_solution_characterization_l1321_132162

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(4/3, 4/3, -5/3), (4/3, -5/3, 4/3), (-5/3, 4/3, 4/3),
   (-4/3, -4/3, 5/3), (-4/3, 5/3, -4/3), (5/3, -4/3, -4/3)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^2 - y*z = |y - z| + 1 ∧
  y^2 - z*x = |z - x| + 1 ∧
  z^2 - x*y = |x - y| + 1

theorem solution_characterization :
  {p : ℝ × ℝ × ℝ | satisfies_equations p.1 p.2.1 p.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1321_132162


namespace NUMINAMATH_CALUDE_power_calculation_l1321_132174

theorem power_calculation : (8^5 / 8^3) * 4^6 = 262144 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1321_132174


namespace NUMINAMATH_CALUDE_wall_width_theorem_l1321_132148

theorem wall_width_theorem (width height length : ℝ) (volume : ℝ) :
  height = 6 * width →
  length = 7 * height →
  volume = width * height * length →
  volume = 16128 →
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_wall_width_theorem_l1321_132148


namespace NUMINAMATH_CALUDE_function_relation_l1321_132111

/-- Given functions h and k, prove that C = 3D/4 -/
theorem function_relation (C D : ℝ) (h k : ℝ → ℝ) : 
  D ≠ 0 →
  (∀ x, h x = 2 * C * x - 3 * D^2) →
  (∀ x, k x = D * x) →
  h (k 2) = 0 →
  C = 3 * D / 4 := by
sorry

end NUMINAMATH_CALUDE_function_relation_l1321_132111


namespace NUMINAMATH_CALUDE_product_condition_l1321_132182

theorem product_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  (∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) :=
sorry

end NUMINAMATH_CALUDE_product_condition_l1321_132182


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1321_132177

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^5 - 2*X^3 + X - 1) * (X^3 - X + 1) = (X^2 + X + 1) * q + (2*X) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1321_132177


namespace NUMINAMATH_CALUDE_nested_bracket_equals_two_l1321_132151

-- Define the bracket operation
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- State the theorem
theorem nested_bracket_equals_two :
  bracket (bracket 45 15 60) (bracket 3 3 6) (bracket 20 10 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_equals_two_l1321_132151


namespace NUMINAMATH_CALUDE_sequence_relation_l1321_132189

-- Define the sequence u
def u (n : ℕ) : ℝ := 17^n * (n + 2)

-- State the theorem
theorem sequence_relation (a b : ℝ) :
  (∀ n : ℕ, u (n + 2) = a * u (n + 1) + b * u n) →
  a^2 - b = 144.5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_relation_l1321_132189


namespace NUMINAMATH_CALUDE_odd_function_property_l1321_132133

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  f 2016 = 2 → f (-2016) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1321_132133
