import Mathlib

namespace NUMINAMATH_CALUDE_equal_radii_of_intersecting_triangles_l3010_301006

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  vertices : Fin 3 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithInscribedCircle
  triangle2 : TriangleWithInscribedCircle
  smallTriangles : Fin 6 → TriangleWithInscribedCircle
  hexagon : Set (ℝ × ℝ)

/-- The theorem stating that the radii of the inscribed circles of the two original triangles are equal -/
theorem equal_radii_of_intersecting_triangles (config : IntersectingTriangles) 
  (h : ∀ i j : Fin 6, (config.smallTriangles i).radius = (config.smallTriangles j).radius) :
  config.triangle1.radius = config.triangle2.radius :=
sorry

end NUMINAMATH_CALUDE_equal_radii_of_intersecting_triangles_l3010_301006


namespace NUMINAMATH_CALUDE_worker_production_equations_l3010_301076

/-- Represents the daily production of workers in a company -/
structure WorkerProduction where
  novice : ℕ
  experienced : ℕ

/-- The conditions of the worker production problem -/
class WorkerProductionProblem (w : WorkerProduction) where
  experience_difference : w.experienced - w.novice = 30
  total_production : w.novice + 2 * w.experienced = 180

/-- The theorem stating the correct system of equations for the worker production problem -/
theorem worker_production_equations (w : WorkerProduction) [WorkerProductionProblem w] :
  (w.experienced - w.novice = 30) ∧ (w.novice + 2 * w.experienced = 180) := by
  sorry

end NUMINAMATH_CALUDE_worker_production_equations_l3010_301076


namespace NUMINAMATH_CALUDE_sample_in_range_l3010_301011

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (total / sampleSize) * n

/-- Theorem: The sample in the range [37, 54] is 42 -/
theorem sample_in_range (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 900 →
  sampleSize = 50 →
  start = 6 →
  ∃ n : ℕ, 
    37 ≤ systematicSample total sampleSize start n ∧ 
    systematicSample total sampleSize start n ≤ 54 ∧
    systematicSample total sampleSize start n = 42 :=
by
  sorry


end NUMINAMATH_CALUDE_sample_in_range_l3010_301011


namespace NUMINAMATH_CALUDE_problem_statement_l3010_301025

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6) + 1 / 2

theorem problem_statement :
  ∀ x A B C a b c : ℝ,
  -- Part 1 conditions
  (x ∈ Set.Icc 0 (Real.pi / 2)) →
  (f x = 11 / 10) →
  -- Part 1 conclusion
  (Real.cos x = (4 * Real.sqrt 3 - 3) / 10) ∧
  -- Part 2 conditions
  (0 < A ∧ A < Real.pi) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) →
  -- Part 2 conclusion
  (f B ∈ Set.Ioc 0 (1 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3010_301025


namespace NUMINAMATH_CALUDE_crystal_mass_ratio_l3010_301086

theorem crystal_mass_ratio : 
  ∀ (m1 m2 : ℝ), -- initial masses of crystals 1 and 2
  ∀ (r1 r2 : ℝ), -- yearly growth rates of crystals 1 and 2
  r1 > 0 ∧ r2 > 0 → -- growth rates are positive
  (3 * r1 * m1 = 7 * r2 * m2) → -- condition on 3-month and 7-month growth
  (r1 = 0.04) → -- 4% yearly growth for crystal 1
  (r2 = 0.05) → -- 5% yearly growth for crystal 2
  (m1 / m2 = 35 / 12) := by
sorry

end NUMINAMATH_CALUDE_crystal_mass_ratio_l3010_301086


namespace NUMINAMATH_CALUDE_budget_theorem_l3010_301005

/-- Represents a budget with three categories in a given ratio -/
structure Budget where
  ratio_1 : ℕ
  ratio_2 : ℕ
  ratio_3 : ℕ
  amount_2 : ℚ

/-- Calculates the total amount allocated in a budget -/
def total_amount (b : Budget) : ℚ :=
  (b.ratio_1 + b.ratio_2 + b.ratio_3) * (b.amount_2 / b.ratio_2)

/-- Theorem stating that for a budget with ratio 5:4:1 and $720 allocated to the second category,
    the total amount is $1800 -/
theorem budget_theorem (b : Budget) 
  (h1 : b.ratio_1 = 5)
  (h2 : b.ratio_2 = 4)
  (h3 : b.ratio_3 = 1)
  (h4 : b.amount_2 = 720) :
  total_amount b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_budget_theorem_l3010_301005


namespace NUMINAMATH_CALUDE_expected_gain_is_negative_three_halves_l3010_301016

/-- Represents the faces of the three-sided die -/
inductive DieFace
  | Heads
  | Tails
  | Edge

/-- The probability of rolling each face -/
def probability (face : DieFace) : ℚ :=
  match face with
  | DieFace.Heads => 1/4
  | DieFace.Tails => 1/4
  | DieFace.Edge => 1/2

/-- The gain (or loss) associated with each face -/
def gain (face : DieFace) : ℤ :=
  match face with
  | DieFace.Heads => 2
  | DieFace.Tails => 4
  | DieFace.Edge => -6

/-- The expected gain from rolling the die once -/
def expected_gain : ℚ :=
  (probability DieFace.Heads * gain DieFace.Heads) +
  (probability DieFace.Tails * gain DieFace.Tails) +
  (probability DieFace.Edge * gain DieFace.Edge)

theorem expected_gain_is_negative_three_halves :
  expected_gain = -3/2 := by sorry

end NUMINAMATH_CALUDE_expected_gain_is_negative_three_halves_l3010_301016


namespace NUMINAMATH_CALUDE_problem_statement_l3010_301021

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5 * a * b) :
  (∃ (x : ℝ), x ≥ a + b ∧ x ≥ 4/5) ∧
  (∀ (x : ℝ), x * a * b ≤ b^2 + 5*a → x ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3010_301021


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3010_301082

noncomputable def f (x : ℝ) := Real.exp (x - 1)

theorem intersection_point_x_coordinate 
  (A B C E : ℝ × ℝ) 
  (hA : A = (1, f 1)) 
  (hB : B = (Real.exp 3, f (Real.exp 3))) 
  (hC : C.2 = (2/3) * A.2 + (1/3) * B.2) 
  (hE : E.2 = f E.1 ∧ E.2 = C.2) :
  E.1 = Real.log ((2/3) + (1/3) * Real.exp (Real.exp 3 - 1)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l3010_301082


namespace NUMINAMATH_CALUDE_statement_a_not_proposition_l3010_301027

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  (s = "true" ∨ s = "false") ∧ ¬(s = "true" ∧ s = "false")

-- Define the statement
def statement_a : String := "It may rain tomorrow"

-- Theorem to prove
theorem statement_a_not_proposition : ¬(is_proposition statement_a) := by
  sorry

end NUMINAMATH_CALUDE_statement_a_not_proposition_l3010_301027


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l3010_301072

theorem radical_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l3010_301072


namespace NUMINAMATH_CALUDE_sum_1_to_99_plus_5_mod_7_l3010_301057

def sum_1_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_1_to_99_plus_5_mod_7 :
  (sum_1_to_n 99 + 5) % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_99_plus_5_mod_7_l3010_301057


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3010_301029

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/32 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3010_301029


namespace NUMINAMATH_CALUDE_watch_time_theorem_l3010_301081

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Represents a watch that loses time at a constant rate -/
structure Watch where
  lossRate : ℚ  -- Rate at which the watch loses time (in seconds per hour)

def Watch.actualTimeWhenShowing (w : Watch) (setTime : Time) (actualSetTime : Time) (showingTime : Time) : Time :=
  sorry  -- Implementation not required for the statement

theorem watch_time_theorem (w : Watch) :
  let noonTime : Time := ⟨12, 0, 0⟩
  let threeTime : Time := ⟨15, 0, 0⟩
  let watchAtThree : Time := ⟨14, 54, 30⟩
  let eightPM : Time := ⟨20, 0, 0⟩
  let actualEightPM : Time := ⟨20, 15, 8⟩
  w.actualTimeWhenShowing noonTime noonTime eightPM = actualEightPM :=
by sorry


end NUMINAMATH_CALUDE_watch_time_theorem_l3010_301081


namespace NUMINAMATH_CALUDE_vector_parallel_value_l3010_301090

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_parallel_value (x : ℝ) :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, x - 1)
  parallel a b → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_value_l3010_301090


namespace NUMINAMATH_CALUDE_cylinder_volume_approximation_l3010_301002

/-- The volume of a cylinder with diameter 14 cm and height 2 cm is approximately 307.88 cubic centimeters. -/
theorem cylinder_volume_approximation :
  let d : ℝ := 14  -- diameter in cm
  let h : ℝ := 2   -- height in cm
  let r : ℝ := d / 2  -- radius in cm
  let π : ℝ := Real.pi
  let V : ℝ := π * r^2 * h  -- volume formula
  ∃ ε > 0, abs (V - 307.88) < ε ∧ ε < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_approximation_l3010_301002


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3010_301093

theorem ratio_of_percentages (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.4 * R)
  (hR : R = 0.75 * P)
  (hP : P ≠ 0) :
  M / N = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3010_301093


namespace NUMINAMATH_CALUDE_front_parking_spaces_l3010_301044

theorem front_parking_spaces (back_spaces : ℕ) (total_parked : ℕ) (available_spaces : ℕ)
  (h1 : back_spaces = 38)
  (h2 : total_parked = 39)
  (h3 : available_spaces = 32)
  (h4 : back_spaces / 2 + available_spaces + total_parked = back_spaces + front_spaces) :
  front_spaces = 33 := by
  sorry

end NUMINAMATH_CALUDE_front_parking_spaces_l3010_301044


namespace NUMINAMATH_CALUDE_trapezoid_constructible_l3010_301079

/-- A trapezoid with side lengths a, b, c, and d, where a and b are the bases and c and d are the legs. -/
structure Trapezoid (a b c d : ℝ) : Prop where
  base1 : a > 0
  base2 : b > 0
  leg1 : c > 0
  leg2 : d > 0

/-- The condition for constructibility of a trapezoid. -/
def isConstructible (a b c d : ℝ) : Prop :=
  c > d ∧ c - d < a - b ∧ a - b < c + d

/-- Theorem stating the necessary and sufficient conditions for constructing a trapezoid. -/
theorem trapezoid_constructible {a b c d : ℝ} (t : Trapezoid a b c d) :
  isConstructible a b c d ↔ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a - b :=
sorry

end NUMINAMATH_CALUDE_trapezoid_constructible_l3010_301079


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3010_301000

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3010_301000


namespace NUMINAMATH_CALUDE_transportation_problem_l3010_301015

/-- Represents the capacity and cost of trucks -/
structure TruckInfo where
  a_capacity : ℝ
  b_capacity : ℝ
  a_cost : ℝ
  b_cost : ℝ

/-- Represents a transportation plan -/
structure TransportPlan where
  a_trucks : ℕ
  b_trucks : ℕ

/-- Theorem stating the properties of the transportation problem -/
theorem transportation_problem (info : TruckInfo) 
  (h1 : 3 * info.a_capacity + 2 * info.b_capacity = 90)
  (h2 : 5 * info.a_capacity + 4 * info.b_capacity = 160)
  (h3 : info.a_cost = 500)
  (h4 : info.b_cost = 400) :
  ∃ (plans : List TransportPlan),
    (info.a_capacity = 20 ∧ info.b_capacity = 15) ∧
    (plans.length = 3) ∧
    (∀ p ∈ plans, p.a_trucks * info.a_capacity + p.b_trucks * info.b_capacity = 190) ∧
    (∃ p ∈ plans, p.a_trucks = 8 ∧ p.b_trucks = 2 ∧
      ∀ p' ∈ plans, p'.a_trucks * info.a_cost + p'.b_trucks * info.b_cost ≥ 
                    p.a_trucks * info.a_cost + p.b_trucks * info.b_cost) := by
  sorry

end NUMINAMATH_CALUDE_transportation_problem_l3010_301015


namespace NUMINAMATH_CALUDE_greatest_perfect_square_under_500_l3010_301097

theorem greatest_perfect_square_under_500 : 
  ∀ n : ℕ, n^2 < 500 → n^2 ≤ 484 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_perfect_square_under_500_l3010_301097


namespace NUMINAMATH_CALUDE_rose_can_afford_supplies_l3010_301085

def budget : ℝ := 30

def paintbrush_cost : ℝ := 2.40
def paints_cost : ℝ := 9.20
def easel_cost : ℝ := 6.50
def canvas_cost : ℝ := 12.25
def drawing_pad_cost : ℝ := 4.75

def discount_rate : ℝ := 0.15

def total_cost_before_discount : ℝ := 
  paintbrush_cost + paints_cost + easel_cost + canvas_cost + drawing_pad_cost

def discount_amount : ℝ := discount_rate * total_cost_before_discount

def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

theorem rose_can_afford_supplies : 
  total_cost_after_discount ≤ budget ∧ 
  budget - total_cost_after_discount = 0.165 := by sorry

end NUMINAMATH_CALUDE_rose_can_afford_supplies_l3010_301085


namespace NUMINAMATH_CALUDE_expression_simplification_l3010_301040

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2 + 1) :
  (1 - 1 / (m + 1)) * ((m^2 - 1) / m) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3010_301040


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3010_301042

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3010_301042


namespace NUMINAMATH_CALUDE_problem_statement_l3010_301098

theorem problem_statement (a b : ℤ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^2008 - b^2008 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3010_301098


namespace NUMINAMATH_CALUDE_base_seven_subtraction_l3010_301031

/-- Represents a number in base 7 --/
def BaseSevenNumber := List Nat

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : BaseSevenNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Subtracts two base 7 numbers --/
def base_seven_subtract (a b : BaseSevenNumber) : BaseSevenNumber :=
  sorry

theorem base_seven_subtraction :
  let a : BaseSevenNumber := [4, 1, 2, 3]  -- 3214 in base 7
  let b : BaseSevenNumber := [4, 3, 2, 1]  -- 1234 in base 7
  let result : BaseSevenNumber := [0, 5, 6, 2]  -- 2650 in base 7
  base_seven_subtract a b = result := by sorry

end NUMINAMATH_CALUDE_base_seven_subtraction_l3010_301031


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l3010_301053

theorem tip_percentage_calculation (total_bill : ℝ) (num_people : ℕ) (individual_payment : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ individual_payment = 30.58 →
  (individual_payment * num_people - total_bill) / total_bill * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l3010_301053


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l3010_301074

theorem prime_square_sum_equation :
  ∀ (a b c k : ℕ),
    Prime a ∧ Prime b ∧ Prime c ∧ k > 0 ∧
    a^2 + b^2 + 16*c^2 = 9*k^2 + 1 →
    ((a = 3 ∧ b = 3 ∧ c = 2 ∧ k = 3) ∨
     (a = 3 ∧ b = 37 ∧ c = 3 ∧ k = 13) ∨
     (a = 37 ∧ b = 3 ∧ c = 3 ∧ k = 13) ∨
     (a = 3 ∧ b = 17 ∧ c = 3 ∧ k = 7) ∨
     (a = 17 ∧ b = 3 ∧ c = 3 ∧ k = 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l3010_301074


namespace NUMINAMATH_CALUDE_range_of_a_l3010_301075

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 ≥ a) → 
  a ∈ Set.Icc (-7) 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3010_301075


namespace NUMINAMATH_CALUDE_parabola_equation_l3010_301077

/-- Given a parabola C: y^2 = 2px and a circle x^2 + y^2 - 2x - 15 = 0,
    if the focus of the parabola coincides with the center of the circle,
    then the equation of the parabola C is y^2 = 4x. -/
theorem parabola_equation (p : ℝ) :
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 + y^2 - 2*x - 15 = 0 ∧
   (1, 0) = (x + p/2, 0)) →
  (∀ (x y : ℝ), y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3010_301077


namespace NUMINAMATH_CALUDE_polynomial_interpolation_l3010_301059

def p (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 2*x + 2

theorem polynomial_interpolation :
  (p (-2) = 2) ∧
  (p (-1) = 1) ∧
  (p 0 = 2) ∧
  (p 1 = -1) ∧
  (p 2 = 10) ∧
  (∀ q : ℝ → ℝ, (q (-2) = 2) ∧ (q (-1) = 1) ∧ (q 0 = 2) ∧ (q 1 = -1) ∧ (q 2 = 10) →
    (∃ a b c d e : ℝ, ∀ x, q x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (∀ x, q x = p x)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_interpolation_l3010_301059


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l3010_301070

theorem ratio_of_linear_system (x y c d : ℝ) (h1 : 3 * x + 2 * y = c) (h2 : 4 * y - 6 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l3010_301070


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l3010_301022

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors :
  ∃ (n : ℕ+), has_20_divisors n ∧ ∀ (m : ℕ+), has_20_divisors m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l3010_301022


namespace NUMINAMATH_CALUDE_power_sum_equality_l3010_301091

theorem power_sum_equality : (-2)^2007 + (-2)^2008 = 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l3010_301091


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3010_301036

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_shift_theorem :
  let initial_parabola : Parabola := { a := -1, h := 1, k := 2 }
  let shifted_parabola := shift_horizontal (shift_vertical initial_parabola 2) 1
  shifted_parabola = { a := -1, h := 0, k := 4 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3010_301036


namespace NUMINAMATH_CALUDE_grocer_sales_theorem_l3010_301060

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500, 6780, 7000, 7200]
def target_average : ℕ := 6600
def num_months : ℕ := 10

theorem grocer_sales_theorem : 
  let total_target := target_average * num_months
  let current_total := sales.sum
  let remaining_months := num_months - sales.length
  let remaining_sales := total_target - current_total
  remaining_sales / remaining_months = 9445 := by sorry

end NUMINAMATH_CALUDE_grocer_sales_theorem_l3010_301060


namespace NUMINAMATH_CALUDE_cost_of_horse_l3010_301039

/-- Given Albert's purchase and sale of horses and cows, prove the cost of a horse -/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧
  horse_profit_rate = 0.1 ∧
  cow_profit_rate = 0.2 ∧
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_horse_l3010_301039


namespace NUMINAMATH_CALUDE_visitor_and_revenue_properties_l3010_301008

/-- Represents the daily change in visitors (in 10,000 people) --/
def visitor_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

/-- The ticket price per person in yuan --/
def ticket_price : ℝ := 15

/-- Theorem stating the properties of visitor numbers and revenue --/
theorem visitor_and_revenue_properties (a : ℝ) : 
  let visitors_day3 := a + visitor_changes[0] + visitor_changes[1]
  let max_visitors := (List.map (λ i => a + (List.take i visitor_changes).sum) (List.range 7)).maximum?
  let total_visitors := a * 7 + visitor_changes.sum
  (visitors_day3 = a + 2.4) ∧ 
  (max_visitors = some (a + 2.8)) ∧
  (a = 2 → total_visitors * ticket_price * 10000 = 4.08 * 10^6) := by sorry

end NUMINAMATH_CALUDE_visitor_and_revenue_properties_l3010_301008


namespace NUMINAMATH_CALUDE_linear_function_composition_l3010_301032

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3010_301032


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l3010_301017

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) : 
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l3010_301017


namespace NUMINAMATH_CALUDE_john_spent_625_l3010_301063

/-- The amount John spent on his purchases with a coupon -/
def total_spent (vacuum_cost dishwasher_cost coupon_value : ℕ) : ℕ :=
  vacuum_cost + dishwasher_cost - coupon_value

/-- Theorem stating that John spent $625 on his purchases -/
theorem john_spent_625 :
  total_spent 250 450 75 = 625 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_625_l3010_301063


namespace NUMINAMATH_CALUDE_total_keys_needed_l3010_301013

theorem total_keys_needed 
  (num_complexes : ℕ) 
  (apartments_per_complex : ℕ) 
  (keys_per_apartment : ℕ) 
  (h1 : num_complexes = 2) 
  (h2 : apartments_per_complex = 12) 
  (h3 : keys_per_apartment = 3) : 
  num_complexes * apartments_per_complex * keys_per_apartment = 72 := by
sorry

end NUMINAMATH_CALUDE_total_keys_needed_l3010_301013


namespace NUMINAMATH_CALUDE_merchant_profit_l3010_301056

theorem merchant_profit (cost selling : ℝ) (h : 20 * cost = 16 * selling) :
  (selling - cost) / cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_l3010_301056


namespace NUMINAMATH_CALUDE_probability_exactly_three_less_than_seven_l3010_301003

def probability_less_than_7 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_exactly_three_less_than_seven :
  (choose number_of_dice target_count : ℚ) * probability_less_than_7^target_count * (1 - probability_less_than_7)^(number_of_dice - target_count) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_three_less_than_seven_l3010_301003


namespace NUMINAMATH_CALUDE_quadratic_coefficients_identify_coefficients_l3010_301047

theorem quadratic_coefficients (x : ℝ) : 
  5 * x^2 + 1/2 = 6 * x ↔ 5 * x^2 + (-6) * x + 1/2 = 0 :=
by sorry

theorem identify_coefficients :
  ∃ (a b c : ℝ), (∀ x, a * x^2 + b * x + c = 0 ↔ 5 * x^2 + (-6) * x + 1/2 = 0) ∧
  a = 5 ∧ b = -6 ∧ c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_identify_coefficients_l3010_301047


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3010_301054

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 6) = 7 → x = 43 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3010_301054


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_10201_l3010_301052

theorem largest_prime_factor_of_10201 : 
  (Nat.factors 10201).maximum? = some 37 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_10201_l3010_301052


namespace NUMINAMATH_CALUDE_penelope_savings_l3010_301046

theorem penelope_savings (daily_savings : ℕ) (total_savings : ℕ) (savings_period : ℕ) :
  daily_savings = 24 →
  total_savings = 8760 →
  savings_period * daily_savings = total_savings →
  savings_period = 365 := by
sorry

end NUMINAMATH_CALUDE_penelope_savings_l3010_301046


namespace NUMINAMATH_CALUDE_inequality_proof_l3010_301004

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_eq : x^2/(1+x^2) + y^2/(1+y^2) + z^2/(1+z^2) = 2) : 
  x/(1+x^2) + y/(1+y^2) + z/(1+z^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3010_301004


namespace NUMINAMATH_CALUDE_binary_101111011_equals_379_l3010_301094

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101111011₂ (least significant bit first) -/
def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111011_equals_379_l3010_301094


namespace NUMINAMATH_CALUDE_equal_means_sum_l3010_301055

theorem equal_means_sum (group1 group2 : Finset ℕ) : 
  (Finset.card group1 = 10) →
  (Finset.card group2 = 207) →
  (group1 ∪ group2 = Finset.range 217) →
  (group1 ∩ group2 = ∅) →
  (Finset.sum group1 id / Finset.card group1 = Finset.sum group2 id / Finset.card group2) →
  Finset.sum group1 id = 1090 := by
sorry

end NUMINAMATH_CALUDE_equal_means_sum_l3010_301055


namespace NUMINAMATH_CALUDE_bank_transaction_decrease_fraction_l3010_301024

/-- Represents a bank account transaction --/
structure BankTransaction where
  initialBalance : ℚ
  withdrawal : ℚ
  depositFraction : ℚ
  finalBalance : ℚ

/-- Calculates the fraction by which the account balance decreased after withdrawal --/
def decreaseFraction (t : BankTransaction) : ℚ :=
  t.withdrawal / t.initialBalance

/-- Theorem stating the conditions and the result to be proved --/
theorem bank_transaction_decrease_fraction 
  (t : BankTransaction)
  (h1 : t.withdrawal = 200)
  (h2 : t.depositFraction = 1/5)
  (h3 : t.finalBalance = 360)
  (h4 : t.finalBalance = t.initialBalance - t.withdrawal + t.depositFraction * (t.initialBalance - t.withdrawal)) :
  decreaseFraction t = 2/5 := by sorry


end NUMINAMATH_CALUDE_bank_transaction_decrease_fraction_l3010_301024


namespace NUMINAMATH_CALUDE_special_rectangle_area_l3010_301061

/-- Represents a rectangle with specific properties -/
structure SpecialRectangle where
  d : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq_3w : h = 3 * w  -- length is three times the width
  diagonal_eq : d^2 = w^2 + h^2  -- Pythagorean theorem

/-- The area of a SpecialRectangle is (3/10) * d^2 -/
theorem special_rectangle_area (r : SpecialRectangle) : r.w * r.h = (3/10) * r.d^2 := by
  sorry

#check special_rectangle_area

end NUMINAMATH_CALUDE_special_rectangle_area_l3010_301061


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3010_301007

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (4 / y - 7 * y^3) = 3 / y - 21 * y^3 / 4 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3010_301007


namespace NUMINAMATH_CALUDE_fifteenth_number_base5_l3010_301001

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The 15th number in base 5 counting system -/
def fifteenthNumberBase5 : List ℕ := toBase5 15

theorem fifteenth_number_base5 :
  fifteenthNumberBase5 = [3, 0] :=
sorry

end NUMINAMATH_CALUDE_fifteenth_number_base5_l3010_301001


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3010_301064

/-- The constant term in the expansion of (2/x + x)^4 is 24 -/
theorem constant_term_binomial_expansion :
  let n : ℕ := 4
  let a : ℚ := 2
  let b : ℚ := 1
  (Nat.choose n (n / 2)) * a^(n / 2) * b^(n / 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3010_301064


namespace NUMINAMATH_CALUDE_soda_cans_purchased_l3010_301095

/-- Given that S cans of soda can be purchased for Q quarters, and 1 dollar is worth 5 quarters due to a fee,
    the number of cans of soda that can be purchased for D dollars is (5 * D * S) / Q. -/
theorem soda_cans_purchased (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  (S / Q) * (5 * D) = (5 * D * S) / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_purchased_l3010_301095


namespace NUMINAMATH_CALUDE_P_intersect_Q_l3010_301037

/-- The set P of vectors -/
def P : Set (ℝ × ℝ) := {a | ∃ m : ℝ, a = (1, 0) + m • (0, 1)}

/-- The set Q of vectors -/
def Q : Set (ℝ × ℝ) := {b | ∃ n : ℝ, b = (1, 1) + n • (-1, 1)}

/-- The theorem stating that the intersection of P and Q is the singleton set containing (1,1) -/
theorem P_intersect_Q : P ∩ Q = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l3010_301037


namespace NUMINAMATH_CALUDE_speed_limit_exceeders_l3010_301028

/-- Represents the percentage of motorists who receive speeding tickets -/
def speeding_ticket_percentage : ℝ := 10

/-- Represents the percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 30

/-- Represents the total percentage of motorists exceeding the speed limit -/
def exceeding_speed_limit_percentage : ℝ := 14

theorem speed_limit_exceeders (total_motorists : ℝ) (total_motorists_pos : total_motorists > 0) :
  (speeding_ticket_percentage / 100) * total_motorists =
  ((100 - no_ticket_percentage) / 100) * (exceeding_speed_limit_percentage / 100) * total_motorists :=
by sorry

end NUMINAMATH_CALUDE_speed_limit_exceeders_l3010_301028


namespace NUMINAMATH_CALUDE_total_hats_bought_l3010_301080

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ)
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 548)
  (h4 : green_count = 38)
  (h5 : ∃ blue_count : ℕ, blue_cost * blue_count + green_cost * green_count = total_price) :
  ∃ total_count : ℕ, total_count = green_count + (total_price - green_cost * green_count) / blue_cost :=
by sorry

end NUMINAMATH_CALUDE_total_hats_bought_l3010_301080


namespace NUMINAMATH_CALUDE_quadrilateral_inequality_l3010_301068

-- Define the points
variable (A B C D E : EuclideanSpace ℝ (Fin 2))

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the intersection of diagonals
def diagonals_intersect (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define the distance function
def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  sorry

-- State the theorem
theorem quadrilateral_inequality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_intersect : diagonals_intersect A B C D E)
  (h_AB : distance A B = 1)
  (h_BC : distance B C = 1)
  (h_CD : distance C D = 1)
  (h_DE : distance D E = 1) :
  distance A D < 2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_inequality_l3010_301068


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l3010_301069

/-- The number of ways to place n distinguishable balls into k indistinguishable boxes -/
def ball_distribution (n k : ℕ) : ℕ := sorry

/-- The number of ways to place 5 distinguishable balls into 3 indistinguishable boxes is 36 -/
theorem five_balls_three_boxes : ball_distribution 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l3010_301069


namespace NUMINAMATH_CALUDE_min_sin_minus_cos_half_angle_l3010_301067

theorem min_sin_minus_cos_half_angle :
  let f : ℝ → ℝ := λ A ↦ Real.sin (A / 2) - Real.cos (A / 2)
  ∃ (min : ℝ) (A : ℝ), 
    (∀ x, f x ≥ min) ∧ 
    (f A = min) ∧ 
    (min = -Real.sqrt 2) ∧ 
    (A = 7 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_sin_minus_cos_half_angle_l3010_301067


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3010_301014

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 2 = (X^2 - 3*X + 2) * q + (15*X - 12) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3010_301014


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3010_301049

theorem diophantine_equation_solution :
  ∀ x y z : ℕ+,
    (x + y = z ∧ x^2 * y = z^2 + 1) →
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 13 ∧ z = 18)) :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3010_301049


namespace NUMINAMATH_CALUDE_power_of_eight_mod_five_l3010_301048

theorem power_of_eight_mod_five : 8^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_eight_mod_five_l3010_301048


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l3010_301045

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 2
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l3010_301045


namespace NUMINAMATH_CALUDE_two_cubic_feet_equals_3456_cubic_inches_l3010_301087

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Calculates the volume in cubic inches given the volume in cubic feet -/
def cubic_feet_to_cubic_inches (cf : ℚ) : ℚ :=
  cf * feet_to_inches^3

/-- Theorem stating that 2 cubic feet is equal to 3456 cubic inches -/
theorem two_cubic_feet_equals_3456_cubic_inches :
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_equals_3456_cubic_inches_l3010_301087


namespace NUMINAMATH_CALUDE_integral_gt_one_minus_one_over_n_l3010_301018

theorem integral_gt_one_minus_one_over_n (n : ℕ+) :
  ∫ x in (0:ℝ)..1, (1 / (1 + x ^ (n:ℝ))) > 1 - 1 / (n:ℝ) := by sorry

end NUMINAMATH_CALUDE_integral_gt_one_minus_one_over_n_l3010_301018


namespace NUMINAMATH_CALUDE_sum_of_triple_g_roots_l3010_301084

def g (x : ℝ) : ℝ := -x^2 + 6*x - 8

theorem sum_of_triple_g_roots (h : ∀ x, g x ≤ g 3) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (g (g x)) = 2) ∧ 
                  (∀ x, g (g (g x)) = 2 → x ∈ S) ∧
                  (S.sum id = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triple_g_roots_l3010_301084


namespace NUMINAMATH_CALUDE_theater_seats_l3010_301038

theorem theater_seats (adult_price child_price total_income num_children : ℚ)
  (h1 : adult_price = 3)
  (h2 : child_price = 3/2)
  (h3 : total_income = 510)
  (h4 : num_children = 60)
  (h5 : ∃ num_adults : ℚ, num_adults * adult_price + num_children * child_price = total_income) :
  ∃ total_seats : ℚ, total_seats = num_children + (total_income - num_children * child_price) / adult_price ∧ total_seats = 200 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l3010_301038


namespace NUMINAMATH_CALUDE_simplify_expression_l3010_301073

theorem simplify_expression (a b : ℝ) : (8*a - 7*b) - (4*a - 5*b) = 4*a - 2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3010_301073


namespace NUMINAMATH_CALUDE_zoo_animals_l3010_301062

/-- The number of animals in a zoo satisfies certain conditions. -/
theorem zoo_animals (parrots snakes monkeys elephants zebras : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 35 = monkeys →
  elephants - zebras = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l3010_301062


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3010_301050

/-- The sum of the lengths of quarter-circles approaches πC as n approaches infinity -/
theorem quarter_circles_sum_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * (C / (2 * π * n)) / 4) - π * C| < ε :=
by sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3010_301050


namespace NUMINAMATH_CALUDE_greatest_piece_length_l3010_301041

theorem greatest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_piece_length_l3010_301041


namespace NUMINAMATH_CALUDE_adjacent_different_country_probability_l3010_301034

/-- Represents a country with delegates -/
structure Country where
  delegates : Nat
  deriving Repr

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  total_seats : Nat
  countries : List Country
  deriving Repr

/-- Calculates the probability of each delegate sitting adjacent to at least one delegate from a different country -/
def probability_adjacent_different_country (arrangement : SeatingArrangement) : Rat :=
  sorry

/-- The specific seating arrangement from the problem -/
def problem_arrangement : SeatingArrangement :=
  { total_seats := 12
  , countries := List.replicate 4 { delegates := 2 }
  }

/-- Theorem stating the probability for the given seating arrangement -/
theorem adjacent_different_country_probability :
  probability_adjacent_different_country problem_arrangement = 4897683 / 9979200 :=
  sorry

end NUMINAMATH_CALUDE_adjacent_different_country_probability_l3010_301034


namespace NUMINAMATH_CALUDE_laptop_price_proof_l3010_301083

/-- The original sticker price of a laptop -/
def sticker_price : ℝ := 1004

/-- The discount rate at store A -/
def discount_A : ℝ := 0.20

/-- The rebate amount at store A -/
def rebate_A : ℝ := 120

/-- The discount rate at store B -/
def discount_B : ℝ := 0.30

/-- The tax rate applied at both stores -/
def tax_rate : ℝ := 0.07

/-- The price difference between stores A and B -/
def price_difference : ℝ := 21

theorem laptop_price_proof :
  let price_A := (sticker_price * (1 - discount_A) - rebate_A) * (1 + tax_rate)
  let price_B := sticker_price * (1 - discount_B) * (1 + tax_rate)
  price_B - price_A = price_difference :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l3010_301083


namespace NUMINAMATH_CALUDE_betty_age_l3010_301051

theorem betty_age (carol alice betty : ℝ) 
  (h1 : carol = 5 * alice)
  (h2 : carol = 2 * betty)
  (h3 : alice = carol - 12) :
  betty = 7.5 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l3010_301051


namespace NUMINAMATH_CALUDE_horner_rule_v4_l3010_301026

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 79
  v3 * x - 8

theorem horner_rule_v4 :
  horner_v4 (-4) = 220 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v4_l3010_301026


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3010_301030

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3010_301030


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3010_301078

theorem complex_fraction_simplification :
  let numerator := (10^4 + 500) * (25^4 + 500) * (40^4 + 500) * (55^4 + 500) * (70^4 + 500)
  let denominator := (5^4 + 500) * (20^4 + 500) * (35^4 + 500) * (50^4 + 500) * (65^4 + 500)
  ∀ x : ℕ, x^4 + 500 = (x^2 - 10*x + 50) * (x^2 + 10*x + 50) →
  (numerator / denominator : ℚ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3010_301078


namespace NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l3010_301043

theorem unique_number_with_specific_divisors : ∃! n : ℕ, 
  (9 ∣ n) ∧ (5 ∣ n) ∧ (Finset.card (Nat.divisors n) = 14) ∧ (n = 3645) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_divisors_l3010_301043


namespace NUMINAMATH_CALUDE_symmetry_yOz_correct_l3010_301099

/-- Given a point (x, y, z) in 3D space, this function returns its symmetrical point
    with respect to the yOz plane -/
def symmetry_yOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetry_yOz_correct :
  symmetry_yOz (1, 2, 1) = (-1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yOz_correct_l3010_301099


namespace NUMINAMATH_CALUDE_cards_distribution_l3010_301088

/-- Given a deck of 48 cards dealt as evenly as possible among 9 people,
    the number of people who receive fewer than 6 cards is 6. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 48) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 6 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l3010_301088


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l3010_301089

def C (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

theorem range_of_y_over_x : 
  ∀ x y : ℝ, C x y → ∃ t : ℝ, y / x = t ∧ -Real.sqrt 3 / 3 ≤ t ∧ t ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l3010_301089


namespace NUMINAMATH_CALUDE_triangle_area_l3010_301010

/-- Given a triangle with one side of length 14 units, the angle opposite to this side
    being 60 degrees, and the ratio of the other two sides being 8:5,
    prove that the area of the triangle is 40√3 square units. -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) :
  a = 14 →
  θ = 60 * π / 180 →
  b / c = 8 / 5 →
  (1 / 2) * b * c * Real.sin θ = 40 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3010_301010


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3010_301065

theorem min_value_of_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) ≥ 3 ∧
  ((a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3010_301065


namespace NUMINAMATH_CALUDE_variance_most_appropriate_for_stability_l3010_301096

-- Define a type for statistical measures
inductive StatMeasure
  | Mean
  | Variance
  | Mode
  | Median

-- Define a function to represent the appropriateness of a measure for stability evaluation
def stability_appropriateness (measure : StatMeasure) : Prop :=
  match measure with
  | StatMeasure.Variance => True
  | _ => False

-- Define the data set
def yield_data : List ℝ := sorry

-- Theorem stating that variance is the most appropriate measure for stability
theorem variance_most_appropriate_for_stability :
  ∀ (measure : StatMeasure),
    stability_appropriateness measure ↔ measure = StatMeasure.Variance :=
by sorry

end NUMINAMATH_CALUDE_variance_most_appropriate_for_stability_l3010_301096


namespace NUMINAMATH_CALUDE_floor_ceil_sum_equation_l3010_301012

theorem floor_ceil_sum_equation : ∃ (r s : ℝ), 
  (Int.floor r : ℝ) + r + (Int.ceil s : ℝ) = 10.7 ∧ r = 4.7 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_equation_l3010_301012


namespace NUMINAMATH_CALUDE_dusty_change_l3010_301009

def single_layer_cost : ℝ := 4
def single_layer_tax_rate : ℝ := 0.05
def double_layer_cost : ℝ := 7
def double_layer_tax_rate : ℝ := 0.10
def fruit_tart_cost : ℝ := 5
def fruit_tart_tax_rate : ℝ := 0.08

def single_layer_quantity : ℕ := 7
def double_layer_quantity : ℕ := 5
def fruit_tart_quantity : ℕ := 3

def payment_amount : ℝ := 200

theorem dusty_change :
  let single_layer_total := single_layer_quantity * (single_layer_cost * (1 + single_layer_tax_rate))
  let double_layer_total := double_layer_quantity * (double_layer_cost * (1 + double_layer_tax_rate))
  let fruit_tart_total := fruit_tart_quantity * (fruit_tart_cost * (1 + fruit_tart_tax_rate))
  let total_cost := single_layer_total + double_layer_total + fruit_tart_total
  payment_amount - total_cost = 115.90 := by
  sorry

end NUMINAMATH_CALUDE_dusty_change_l3010_301009


namespace NUMINAMATH_CALUDE_b_hire_charges_l3010_301020

/-- The hire charges for person b given the total cost and usage hours -/
def hire_charges_b (total_cost : ℚ) (hours_a hours_b hours_c : ℚ) : ℚ :=
  total_cost * (hours_b / (hours_a + hours_b + hours_c))

/-- Theorem stating that b's hire charges are 225 Rs given the problem conditions -/
theorem b_hire_charges :
  hire_charges_b 720 9 10 13 = 225 := by
  sorry

end NUMINAMATH_CALUDE_b_hire_charges_l3010_301020


namespace NUMINAMATH_CALUDE_largest_quantity_l3010_301071

theorem largest_quantity (a b c d : ℝ) : 
  (a + 2 = b - 1) ∧ (b - 1 = c + 3) ∧ (c + 3 = d - 4) →
  (d > a) ∧ (d > b) ∧ (d > c) := by
sorry

end NUMINAMATH_CALUDE_largest_quantity_l3010_301071


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3010_301035

theorem geese_percentage_among_non_swans 
  (total : ℝ) 
  (geese_percent : ℝ) 
  (swan_percent : ℝ) 
  (heron_percent : ℝ) 
  (duck_percent : ℝ) 
  (h1 : geese_percent = 30)
  (h2 : swan_percent = 25)
  (h3 : heron_percent = 10)
  (h4 : duck_percent = 35)
  (h5 : geese_percent + swan_percent + heron_percent + duck_percent = 100) :
  (geese_percent / (100 - swan_percent)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3010_301035


namespace NUMINAMATH_CALUDE_min_xy_value_l3010_301023

theorem min_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 5/x + 3/y = 1) :
  ∀ z : ℝ, x * y ≤ z → 60 ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l3010_301023


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3010_301092

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l3010_301092


namespace NUMINAMATH_CALUDE_f_zero_equals_one_l3010_301058

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function for y = f(x+1)
def isEvenShifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (-x + 1)

-- Theorem statement
theorem f_zero_equals_one
  (h_even : isEvenShifted f)
  (h_f_two : f 2 = 1) :
  f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_equals_one_l3010_301058


namespace NUMINAMATH_CALUDE_mustang_length_proof_l3010_301033

theorem mustang_length_proof (smallest_model : ℝ) (mid_size_model : ℝ) (full_size : ℝ)
  (h1 : smallest_model = 12)
  (h2 : smallest_model = mid_size_model / 2)
  (h3 : mid_size_model = full_size / 10) :
  full_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_mustang_length_proof_l3010_301033


namespace NUMINAMATH_CALUDE_solid_T_properties_l3010_301019

/-- A solid T formed from a cube --/
structure SolidT (c : ℝ) where
  -- Assume c > 0 for a valid cube
  c_pos : c > 0

/-- Properties of the solid T --/
def SolidT.properties (T : SolidT c) : Prop :=
  ∃ (surface_area volume longest_diagonal : ℝ),
    -- Surface area
    surface_area = (23 + 8 * Real.sqrt 2 + Real.sqrt 3) * c^2 / 8 ∧
    -- Volume
    volume = 11 * c^3 / 16 ∧
    -- Longest diagonal
    longest_diagonal = 3 * c / 2

/-- Theorem stating the properties of solid T --/
theorem solid_T_properties (c : ℝ) (T : SolidT c) : T.properties := by
  sorry

#check solid_T_properties

end NUMINAMATH_CALUDE_solid_T_properties_l3010_301019


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l3010_301066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (m n : Line) (α : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular n α) 
  (h3 : m ≠ n) : 
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l3010_301066
