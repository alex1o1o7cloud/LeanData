import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_student_height_l417_41709

def heights : List ℝ := [161.5, 154.3, 143.7, 160.1, 158.0, 153.5, 147.8]

theorem shortest_student_height : 
  List.minimum heights = some 143.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_student_height_l417_41709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l417_41747

/-- The number of women in the first group that satisfies the given conditions -/
def number_of_women : ℕ := 8

/-- Work rate of one man -/
noncomputable def man_rate : ℝ := 1

/-- Work rate of one woman -/
noncomputable def woman_rate : ℝ := 1/2

/-- Theorem stating the work rate equality conditions -/
theorem work_rate_equality :
  3 * man_rate + (number_of_women : ℝ) * woman_rate = 6 * man_rate + 2 * woman_rate ∧
  3 * man_rate + 2 * woman_rate = 4/7 * (3 * man_rate + (number_of_women : ℝ) * woman_rate) :=
by sorry

#check work_rate_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l417_41747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_f_geq_g_when_a_less_than_one_f_geq_g_when_a_greater_than_one_l417_41705

/-- Given a positive real number a not equal to 1, define functions f and g -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(3*x + 1)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/a)^(5*x - 2)

/-- Theorem for the range of x where f(x) < 1 when 0 < a < 1 -/
theorem f_less_than_one (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, f a x < 1 ↔ x > -1/3 := by
  sorry

/-- Theorem for the solution set of f(x) ≥ g(x) when 0 < a < 1 -/
theorem f_geq_g_when_a_less_than_one (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, f a x ≥ g a x ↔ x ≤ 1/8 := by
  sorry

/-- Theorem for the solution set of f(x) ≥ g(x) when a > 1 -/
theorem f_geq_g_when_a_greater_than_one (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, f a x ≥ g a x ↔ x ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_less_than_one_f_geq_g_when_a_less_than_one_f_geq_g_when_a_greater_than_one_l417_41705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_tax_deduction_l417_41764

/-- Calculates the investment tax deduction for a type B exemption on an Individual Investment Account -/
def investment_tax_deduction (annual_contribution : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  let final_amount := (annual_contribution * (1 + interest_rate)^years) + 
                      (annual_contribution * (1 + interest_rate)^(years - 1)) + 
                      (annual_contribution * (1 + interest_rate)^(years - 2))
  let total_contribution := annual_contribution * (years : ℝ)
  final_amount - total_contribution

/-- Theorem stating that the investment tax deduction for Vasya's case is 128,200 rubles -/
theorem vasya_tax_deduction :
  investment_tax_deduction 200000 0.1 3 = 128200 := by
  -- Proof goes here
  sorry

#eval investment_tax_deduction 200000 0.1 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_tax_deduction_l417_41764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_50pi_minus_200_l417_41725

noncomputable section

/-- The radius of each circle -/
def radius : ℝ := 5

/-- The number of shaded regions -/
def num_regions : ℕ := 8

/-- The area of a quarter circle with given radius -/
noncomputable def quarter_circle_area (r : ℝ) : ℝ := (Real.pi * r^2) / 4

/-- The area of a square with side length equal to the radius -/
def square_area (r : ℝ) : ℝ := r^2

/-- The area of a single shaded region -/
noncomputable def shaded_region_area (r : ℝ) : ℝ := quarter_circle_area r - square_area r

/-- The total area of all shaded regions -/
noncomputable def total_shaded_area (r : ℝ) (n : ℕ) : ℝ := n * shaded_region_area r

/-- Theorem stating that the total area of the shaded regions is 50π - 200 -/
theorem total_shaded_area_is_50pi_minus_200 : 
  total_shaded_area radius num_regions = 50 * Real.pi - 200 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shaded_area_is_50pi_minus_200_l417_41725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l417_41776

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^3 - 8 else ((-x)^3 - 8)

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) : 
  (∀ y : ℝ, f y = f (-y)) →  -- f is even
  (∀ y ≥ 0, f y = y^3 - 8) →  -- definition of f for non-negative x
  (f (x - 2) > 0 ↔ x < 0 ∨ x > 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l417_41776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_is_correct_l417_41714

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the larger base
  α : ℝ  -- Acute angle
  diagonal_perpendicular : Bool  -- Whether the diagonal is perpendicular to the lateral side

/-- The volume of the solid of revolution formed by rotating the trapezoid around its larger base -/
noncomputable def volume_of_revolution (t : IsoscelesTrapezoid) : ℝ :=
  (Real.pi * t.a^3 / 3) * (Real.sin (2 * t.α))^2 * Real.sin (t.α - Real.pi/6) * Real.sin (t.α + Real.pi/6)

/-- Theorem stating that the volume formula is correct for the given trapezoid -/
theorem volume_of_revolution_is_correct (t : IsoscelesTrapezoid) 
  (h : t.diagonal_perpendicular = true) : 
  volume_of_revolution t = (Real.pi * t.a^3 / 3) * (Real.sin (2 * t.α))^2 * Real.sin (t.α - Real.pi/6) * Real.sin (t.α + Real.pi/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_is_correct_l417_41714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_syllogism_form_of_reasoning_is_wrong_l417_41721

-- Define the sets
def RationalNumbers : Set ℚ := Set.univ
def ProperFractions : Set ℚ := {q : ℚ | 0 < q ∧ q < 1}
def Integers : Set ℤ := Set.univ

-- Define the premises and conclusion
def major_premise : Prop := ∃ (x : ℚ), x ∈ RationalNumbers ∧ x ∈ ProperFractions
def minor_premise : Prop := ∀ (x : ℤ), (x : ℚ) ∈ RationalNumbers
def conclusion : Prop := ∀ (x : ℤ), (x : ℚ) ∈ ProperFractions

-- Define the syllogism
def syllogism (major minor conclusion : Prop) : Prop :=
  (major ∧ minor) → conclusion

-- Theorem stating that the given syllogism is invalid
theorem invalid_syllogism : 
  ¬(syllogism major_premise minor_premise conclusion) := by
  sorry

-- Proof that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong :
  ¬∀ (A B C : Set ℚ),
    (∃ x, x ∈ A ∧ x ∈ B) →
    (∀ x, x ∈ C → x ∈ A) →
    (∀ x, x ∈ C → x ∈ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_syllogism_form_of_reasoning_is_wrong_l417_41721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_three_squares_l417_41726

/-- Represents the side length of a square in a sequence of squares -/
noncomputable def square_side (n : ℕ) : ℝ :=
  match n with
  | 0 => 3
  | k + 1 => 1.25 * square_side k

/-- Calculates the area of a square given its side length -/
noncomputable def square_area (side : ℝ) : ℝ := side * side

/-- Calculates the percent increase between two values -/
noncomputable def percent_increase (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

theorem area_increase_three_squares :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |percent_increase (square_area (square_side 0)) (square_area (square_side 2)) - 144.1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_increase_three_squares_l417_41726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothes_and_transport_expense_theorem_l417_41718

/-- Calculates the amount spent on clothes and transport per month given the annual savings --/
def clothes_and_transport_expense (annual_savings : ℚ) : ℚ :=
  let monthly_savings := annual_savings / 12
  let monthly_salary := monthly_savings / (1 - 0.6 - 0.5 * (1 - 0.6))
  0.5 * (1 - 0.6) * monthly_salary

/-- Theorem stating that given the annual savings of 46800, the monthly expense on clothes and transport is 3900 --/
theorem clothes_and_transport_expense_theorem :
  clothes_and_transport_expense 46800 = 3900 := by
  sorry

#eval clothes_and_transport_expense 46800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothes_and_transport_expense_theorem_l417_41718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41789

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x - Real.pi / 4))^2 - 1

-- State the theorem
theorem f_properties :
  (∀ x, f x = Real.sin (2 * x)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ h, h > 0 ∧ (∀ x, f (x + h) = f x) → h ≥ Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_problem_l417_41759

theorem beverage_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) (hcoprime : Nat.Coprime m n) :
  (m : ℚ) / n = 5 / 32 → m + n = 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_problem_l417_41759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_from_end_is_negative_thirteen_l417_41779

def mySequence : List ℤ := [-13, -7, -2, 2, 5, 7, 12, 19, 31]

def is_valid_sequence (s : List ℤ) : Prop :=
  s.length ≥ 7 ∧
  ∀ i, 4 ≤ i → i < s.length →
    s[i]! = s[i-1]! + s[i-2]! + s[i-3]!

theorem fourth_from_end_is_negative_thirteen (s : List ℤ) :
  s = mySequence →
  is_valid_sequence s →
  s[s.length - 7]! = -13 :=
by sorry

-- Example usage
#eval mySequence[0]!  -- Should output -13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_from_end_is_negative_thirteen_l417_41779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l417_41732

/-- The maximum area of an equilateral triangle inscribed in a rectangle --/
theorem max_equilateral_triangle_area_in_rectangle 
  (a b : ℝ) 
  (ha : a = 8) 
  (hb : b = 15) : 
  ∃ (s : ℝ), s > 0 ∧ 
    s ≤ min a b ∧ 
    s * (Real.sqrt 3 / 2) ≤ max a b ∧
    (Real.sqrt 3 / 4) * s^2 = 64 * Real.sqrt 3 ∧
    ∀ (t : ℝ), t > 0 → 
      t ≤ min a b → 
      t * (Real.sqrt 3 / 2) ≤ max a b → 
      (Real.sqrt 3 / 4) * t^2 ≤ 64 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l417_41732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_is_pi_third_l417_41793

/-- Represents a rectangular track with rounded ends -/
structure Track where
  width : ℝ
  time_difference : ℝ

/-- Calculates the jogging speed on the given track -/
noncomputable def jogging_speed (track : Track) : ℝ :=
  (2 * track.width * Real.pi) / track.time_difference

/-- Theorem stating that for the given track properties, the jogging speed is π/3 -/
theorem jogging_speed_is_pi_third (track : Track)
  (h_width : track.width = 10)
  (h_time : track.time_difference = 60) :
  jogging_speed track = Real.pi / 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval jogging_speed ⟨10, 60⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_is_pi_third_l417_41793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l417_41731

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculate the distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

theorem line_equation_proof (l : Line) : 
  (pointOnLine ⟨3, 4⟩ l) → 
  (distanceToLine ⟨-2, 2⟩ l = distanceToLine ⟨4, -2⟩ l) →
  ((l.a = 2 ∧ l.b = 3 ∧ l.c = -18) ∨ (l.a = 2 ∧ l.b = -1 ∧ l.c = -2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l417_41731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_food_cost_l417_41702

theorem max_food_cost (total_allowed : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_allowed = 75)
  (h2 : sales_tax_rate = 0.07)
  (h3 : tip_rate = 0.15) :
  ∃ (max_food_cost : ℝ), 
    (∀ (x : ℝ), x + sales_tax_rate * x + tip_rate * x ≤ total_allowed → x ≤ max_food_cost) ∧
    (max_food_cost + sales_tax_rate * max_food_cost + tip_rate * max_food_cost ≤ total_allowed) ∧
    (abs (max_food_cost - 61.48) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_food_cost_l417_41702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_circle_l417_41754

-- Define the set of points (x, y) that satisfy the polar equation
def polar_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ p.1 = 3 * Real.cos θ ∧ p.2 = 3 * Real.sin θ}

-- Define a circle with radius 3 centered at the origin
def unit_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

-- Theorem stating that the polar set is equal to the circle
theorem polar_set_is_circle : polar_set = unit_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_set_is_circle_l417_41754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_72_l417_41780

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))).sum id

theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_72_l417_41780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l417_41737

/-- The number of questions in the game show. -/
def num_questions : ℕ := 4

/-- The number of options for each question. -/
def num_options : ℕ := 4

/-- The minimum number of correct answers required to win. -/
def min_correct : ℕ := 3

/-- The probability of guessing a single question correctly. -/
def prob_correct : ℚ := 1 / num_options

/-- The probability of guessing a single question incorrectly. -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the game show. -/
def prob_winning : ℚ := 13 / 256

theorem game_show_probability : 
  (Nat.choose num_questions min_correct * prob_correct ^ min_correct * prob_incorrect ^ (num_questions - min_correct)) +
  (prob_correct ^ num_questions) = prob_winning := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_show_probability_l417_41737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trisection_trisectable_angle_l417_41733

def CompassAndStraightedge : Type := Unit

def ConstructsAngle (construction : CompassAndStraightedge) (θ : ℝ) : Prop := True

def AngleTrisectable (θ : ℝ) : Prop :=
  ∃ (construction : CompassAndStraightedge), ConstructsAngle construction (θ / 3)

theorem angle_trisection (n : ℕ) (h : n > 0) (h_not_div : ¬(3 ∣ n)) :
  ∃ (u v : ℤ), 1 = u * n - 3 * v :=
by sorry

theorem trisectable_angle (n : ℕ) (h : n > 0) (h_not_div : ¬(3 ∣ n)) :
  AngleTrisectable (180 / n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trisection_trisectable_angle_l417_41733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_covers_345_triangle_l417_41761

/-- The smallest side length of a square that can completely cover a 3-4-5 right triangle -/
noncomputable def min_square_side : ℝ := 16 * Real.sqrt 17 / 17

/-- Theorem stating that min_square_side is the smallest side length of a square
    that can completely cover a 3-4-5 right triangle -/
theorem min_square_covers_345_triangle :
  ∀ s : ℝ, s ≥ min_square_side ↔ 
    ∃ (x y : ℝ), x^2 + y^2 = 25 ∧ 0 ≤ x ∧ 0 ≤ y ∧ max x y ≤ s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_covers_345_triangle_l417_41761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellis_family_water_bottles_l417_41751

/-- Calculates the total number of water bottles needed for a family road trip -/
def water_bottles_needed (num_people : ℕ) (total_hours : ℕ) (bottles_per_hour : ℚ) : ℕ :=
  Int.toNat ((num_people * total_hours : ℚ) * bottles_per_hour).ceil

/-- Proves that Ellis' family needs 32 water bottles for their road trip -/
theorem ellis_family_water_bottles :
  water_bottles_needed 4 16 (1/2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellis_family_water_bottles_l417_41751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dali_prints_probability_l417_41758

theorem dali_prints_probability (n m : ℕ) (hn : n = 12) (hm : m = 4) :
  (Nat.factorial (n - m + 1) * Nat.factorial m) / Nat.factorial n = 1 / 55 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dali_prints_probability_l417_41758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OA_OB_l417_41711

open Real

theorem min_distance_OA_OB (θ : ℝ) : 
  sin (θ / 2) = -4/5 →
  cos (θ / 2) = 3/5 →
  (∃ (m : ℝ), m ≥ 24/25 ∧
    ∀ (B : ℝ × ℝ), (∃ (r : ℝ), r > 0 ∧ B = (-7*r, -24*r)) → 
      ‖(-1, 0) - B‖ ≥ m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_OA_OB_l417_41711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_shifted_sine_l417_41796

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 3)

theorem symmetric_axis_of_shifted_sine 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi) = f ω x) 
  (k : ℤ) :
  ∃ (g : ℝ → ℝ), 
    (∀ x, g x = f ω (x + Real.pi/4)) ∧ 
    (∀ x, g (k * Real.pi/2 + Real.pi/6 + x) = g (k * Real.pi/2 + Real.pi/6 - x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_shifted_sine_l417_41796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_cardinality_l417_41791

def symmetric_difference (x y : Finset ℤ) : Finset ℤ := (x \ y) ∪ (y \ x)

theorem symmetric_difference_cardinality 
  (x y : Finset ℤ) 
  (hx : x.card = 12) 
  (hy : y.card = 15) 
  (hxy : (x ∩ y).card = 9) :
  (symmetric_difference x y).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_difference_cardinality_l417_41791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_sixth_l417_41742

theorem sin_double_angle_plus_pi_sixth (α : ℝ) :
  Real.sin (π / 6 - α) = Real.sqrt 2 / 3 →
  Real.sin (2 * α + π / 6) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_plus_pi_sixth_l417_41742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l417_41744

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2*x - x^2) / Real.log (1/2)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = { y : ℝ | y ≥ -2 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l417_41744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_needs_amount_l417_41735

noncomputable def pen_cost : ℚ := 30
noncomputable def kate_has : ℚ := pen_cost / 3

theorem kate_needs_amount : pen_cost - kate_has = 20 := by
  -- Expand the definitions
  unfold pen_cost kate_has
  -- Perform the arithmetic
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kate_needs_amount_l417_41735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_l417_41715

/-- The birth rate in people per two seconds -/
noncomputable def birth_rate : ℚ := 4

/-- The death rate in people per two seconds -/
noncomputable def death_rate : ℚ := 3

/-- The number of seconds in a day -/
noncomputable def seconds_per_day : ℚ := 24 * 60 * 60

/-- The net population increase in one day -/
noncomputable def net_increase : ℚ := (birth_rate - death_rate) / 2 * seconds_per_day

theorem population_increase : net_increase = 43200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_l417_41715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_common_tangent_formula_l417_41767

/-- Two circles with radii R and r that are externally tangent to each other -/
structure ExternallyTangentCircles (R r : ℝ) where
  R_pos : R > 0
  r_pos : r > 0

/-- The distance from the point of tangency to the common external tangent -/
noncomputable def distance_to_common_tangent (R r : ℝ) (c : ExternallyTangentCircles R r) : ℝ :=
  2 * R * r / (R + r)

/-- Theorem: The distance from the point of tangency of two externally tangent circles 
    with radii R and r to their common external tangent is 2Rr / (R + r) -/
theorem distance_to_common_tangent_formula (R r : ℝ) (c : ExternallyTangentCircles R r) :
  distance_to_common_tangent R r c = 2 * R * r / (R + r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_common_tangent_formula_l417_41767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l417_41734

/-- A function that returns the digits of a positive integer as a list -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- A function that checks if a list represents a valid five-digit number according to the problem conditions -/
def is_valid (d : List ℕ) : Prop :=
  d.length = 5 ∧ 
  d.head! = d.getLast! ∧ 
  d.head! = 5 ∧
  d.sum % 5 = 0

/-- A decidable version of is_valid -/
def is_valid_decidable (d : List ℕ) : Bool :=
  d.length = 5 && 
  d.head! = d.getLast! &&
  d.head! = 5 &&
  d.sum % 5 = 0

/-- The main theorem statement -/
theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_decidable (digits n)) (Finset.range 100000)).card = 200 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l417_41734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41716

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin x

theorem f_properties :
  (∀ x y, x ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → 
          y ∈ Set.Icc (-Real.pi/4) (Real.pi/4) → 
          x < y → f x < f y) ∧
  (∀ x : ℝ, f (3*Real.pi/4 + x) = f (3*Real.pi/4 - x)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_tickets_l417_41792

/-- The number of people with tickets -/
def num_people : ℕ := 5

/-- The number of tickets each person has -/
def tickets_per_person : ℕ := 8

/-- The width of each ticket in centimeters -/
noncomputable def ticket_width : ℝ := 30

/-- The length of each ticket in centimeters -/
noncomputable def ticket_length : ℝ := 30

/-- The conversion factor from square centimeters to square meters -/
noncomputable def cm2_to_m2 : ℝ := 1 / 10000

/-- Theorem stating that the total area of tickets is 3.6 square meters -/
theorem total_area_of_tickets :
  (↑num_people * ↑tickets_per_person * ticket_width * ticket_length * cm2_to_m2 : ℝ) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_of_tickets_l417_41792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_theorem_l417_41740

/-- Represents a square garden with a pond -/
structure GardenWithPond where
  perimeter : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ
  semicircle_diameter : ℝ

/-- Calculates the area of the garden not taken up by the pond -/
noncomputable def area_not_taken_by_pond (g : GardenWithPond) : ℝ :=
  let garden_side := g.perimeter / 4
  let garden_area := garden_side * garden_side
  let rectangle_area := g.rectangle_length * g.rectangle_width
  let semicircle_area := Real.pi * (g.semicircle_diameter / 2)^2 / 2
  garden_area - (rectangle_area + semicircle_area)

/-- Theorem stating the area of the garden not taken up by the pond -/
theorem garden_area_theorem (g : GardenWithPond) 
  (h_perimeter : g.perimeter = 48)
  (h_rect_length : g.rectangle_length = 3)
  (h_rect_width : g.rectangle_width = 2)
  (h_semicircle_diam : g.semicircle_diameter = 4) :
  area_not_taken_by_pond g = 138 - 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_theorem_l417_41740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41749

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

-- State the theorem
theorem f_properties :
  -- The range of f(x) is [0, 4]
  (∀ x, 0 ≤ f x ∧ f x ≤ 4) ∧
  -- The smallest positive period of f(x) is π
  (∀ x, f (x + Real.pi) = f x) ∧
  (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  -- The intervals of monotonic increase for f(x) are [- π/6 + kπ, π/3 + kπ], where k ∈ ℤ
  (∀ k : ℤ, ∀ x y, -Real.pi/6 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi/3 + k*Real.pi → f x < f y) :=
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l417_41749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_chip_calories_l417_41741

/-- The number of calories in one potato chip -/
def chip_calories : ℚ := 6

/-- The number of potato chips -/
def num_chips : ℕ := 10

/-- The number of cheezits -/
def num_cheezits : ℕ := 6

/-- The total number of calories consumed -/
def total_calories : ℕ := 108

theorem potato_chip_calories : (num_chips : ℚ) * chip_calories = 60 :=
by
  -- Define the number of calories in one cheezit
  let cheezit_calories : ℚ := chip_calories + (1/3) * chip_calories
  
  -- State the total calorie equation
  have total_equation : (num_chips : ℚ) * chip_calories + (num_cheezits : ℚ) * cheezit_calories = total_calories := by
    sorry
  
  -- Simplify the left side of the equation
  calc
    (num_chips : ℚ) * chip_calories = 10 * 6 := by rfl
    _ = 60 := by norm_num
  
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_chip_calories_l417_41741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_l417_41790

noncomputable section

-- Define the coordinate systems
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the line l
def line_l (t α : ℝ) : ℝ × ℝ := (1/2 + t * Real.cos α, t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ / (Real.sin θ)^2

-- State the theorem
theorem curve_C_and_intersection :
  -- Part 1: Rectangular equation of curve C
  (∀ x y : ℝ, (∃ θ : ℝ, polar_to_rect (curve_C_polar θ) θ = (x, y)) ↔ y^2 = 2*x) ∧
  -- Part 2: Minimum distance between intersection points
  (∃ min_dist : ℝ, 
    (∀ α : ℝ, 0 < α ∧ α < Real.pi → 
      ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
        line_l t₁ α ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1} ∧ 
        line_l t₂ α ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1} ∧
        ((line_l t₁ α).1 - (line_l t₂ α).1)^2 + ((line_l t₁ α).2 - (line_l t₂ α).2)^2 ≥ min_dist^2) ∧
    min_dist = 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_and_intersection_l417_41790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l417_41781

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x / (x - 3)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y, f x = y

theorem domain_of_f : 
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ 0 ∧ x ≠ 3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l417_41781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l417_41703

def line_equation (x y : ℝ) : Prop := 8 * x + 10 * y = 80

def triangle_vertices : Set (ℝ × ℝ) := {(0, 0), (10, 0), (0, 8)}

def is_on_line (p : ℝ × ℝ) : Prop :=
  line_equation p.1 p.2

theorem sum_of_altitudes :
  ∃ (a b c : ℝ),
    a + b + c = 18 + 40 / Real.sqrt 41 ∧
    (∀ p ∈ triangle_vertices, 
      (∃ q : ℝ × ℝ, is_on_line q ∧ 
        (a = dist p q ∨ b = dist p q ∨ c = dist p q))) :=
by
  sorry

#check sum_of_altitudes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_l417_41703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equation_l417_41785

/-- The length of the arc of a curve in polar coordinates -/
noncomputable def arcLength (ρ : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

/-- The polar equation of the curve -/
noncomputable def ρ (φ : ℝ) : ℝ := 5 * Real.exp (5 * φ / 12)

theorem arc_length_equation : 
  arcLength ρ (-π/2) (π/2) = 26 * Real.sinh (5 * π / 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equation_l417_41785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_earnings_l417_41786

/-- Calculates the total earnings for two months given an initial salary and a percentage raise. -/
noncomputable def totalEarnings (initialSalary : ℝ) (raisePercentage : ℝ) : ℝ :=
  let newSalary := initialSalary * (1 + raisePercentage / 100)
  initialSalary + newSalary

/-- Proves that given an initial salary of $1000 and a 10% raise, the total earnings for two months is $2100. -/
theorem sallys_earnings : totalEarnings 1000 10 = 2100 := by
  -- Unfold the definition of totalEarnings
  unfold totalEarnings
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sallys_earnings_l417_41786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_l417_41704

-- Define the points
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (6, 4)
def F : ℝ × ℝ := (3, -2)
def Q : ℝ × ℝ := (2, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem fermat_point_distance :
  distance D Q + distance E Q + distance F Q = 5 + Real.sqrt 5 + Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_l417_41704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l417_41728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - abs x else x^2 - 4*x + 3

-- State the theorem
theorem range_of_m (m : ℝ) : 
  f (f m) ≥ 0 → m ∈ Set.Icc (-2) (2 + Real.sqrt 2) ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l417_41728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_and_positive_quadratic_l417_41723

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x - 3)

-- State the theorem
theorem increasing_function_and_positive_quadratic :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, x^2 - x + 2 > 0) := by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_and_positive_quadratic_l417_41723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_le_quadratic_mean_l417_41753

open BigOperators

variable {n : ℕ}
variable (a : Fin n → ℝ)

theorem arithmetic_mean_le_quadratic_mean (h : ∀ i, 0 < a i) :
  (∑ i, a i) / n ≤ Real.sqrt ((∑ i, (a i)^2) / n) ∧
  ((∑ i, a i) / n = Real.sqrt ((∑ i, (a i)^2) / n) ↔ ∀ i j, a i = a j) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_le_quadratic_mean_l417_41753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l417_41748

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α < 0) : 
  α > -Real.pi / 2 ∧ α < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l417_41748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_fractions_l417_41798

theorem min_sum_of_fractions (W X Y Z : ℕ) : 
  W ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  X ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  Y ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  Z ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (∀ (A B C D : ℕ), 
    A ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
    B ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
    C ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
    D ∈ ({1,2,3,4,5,6,7,8,9} : Set ℕ) →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (W : ℚ) / X + (Y : ℚ) / Z ≤ (A : ℚ) / B + (C : ℚ) / D) →
  (W : ℚ) / X + (Y : ℚ) / Z = 25 / 72 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_fractions_l417_41798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tesseract_diagonals_l417_41768

/-- Represents a tesseract (four-dimensional hypercube) -/
structure Tesseract where
  vertices : Fin 16
  edges : Fin 32

/-- Predicate to check if a pair of vertices forms an edge -/
def IsEdge (t : Tesseract) (pair : Fin 16 × Fin 16) : Prop := sorry

/-- A diagonal in a tesseract -/
def Diagonal (t : Tesseract) := { pair : Fin 16 × Fin 16 // pair.1 ≠ pair.2 ∧ ¬ IsEdge t pair }

/-- The number of diagonals in a tesseract -/
def numDiagonals (t : Tesseract) : ℕ := sorry

/-- Theorem stating that a tesseract has 88 diagonals -/
theorem tesseract_diagonals (t : Tesseract) : numDiagonals t = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tesseract_diagonals_l417_41768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a6_l417_41795

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  positive_terms : ∀ n, a n > 0
  geometric_property : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (g : GeometricSequence) (n : ℕ) : ℝ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_a6 (g : GeometricSequence) 
  (sum3 : sum_n g 3 = 14)
  (a3 : g.a 3 = 8) :
  g.a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_a6_l417_41795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_profit_percentage_l417_41752

/-- Represents the profit percentage calculation for an article sale --/
noncomputable def profit_percentage (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating the conditions and the result to be proved --/
theorem article_profit_percentage :
  let cost_price : ℝ := 60
  let reduced_cost_price : ℝ := cost_price * 0.8
  let reduced_selling_price : ℝ := reduced_cost_price * 1.3
  let selling_price : ℝ := reduced_selling_price + 12.60
  profit_percentage cost_price selling_price = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_profit_percentage_l417_41752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l417_41739

theorem max_value_expression (x y : ℝ) : 
  (2*x + 3*y + 4) / Real.sqrt (2*x^2 + 3*y^2 + 5) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l417_41739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l417_41763

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define the intersecting line
def intersecting_line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 3

-- Define the condition for point M
def point_m_condition (A B M : ℝ × ℝ) : Prop :=
  M.1^2 + M.2^2 = 4 ∧ M.1 = A.1 + B.1 ∧ M.2 = A.2 + B.2

theorem circle_intersection_theorem :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
    intersecting_line k A.1 A.2 ∧ intersecting_line k B.1 B.2 ∧
    (∃ M : ℝ × ℝ, point_m_condition A B M)) ↔
  k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l417_41763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l417_41775

theorem integers_between_cubes : ∃ n : ℕ, n = 44 ∧ 
  n = (Int.toNat ⌊(12.2 : ℝ)^3⌋) - (Int.toNat ⌈(12.1 : ℝ)^3⌉) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l417_41775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_sin_theta_value_l417_41706

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

-- Theorem for the increasing interval
theorem f_increasing_interval :
  ∀ x : ℝ, 0 < x → x < π → (0 < x → x < 5 * π / 6 → Monotone f) :=
sorry

-- Theorem for the value of sin θ
theorem sin_theta_value (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) (h3 : f θ = -6/5) :
  Real.sin θ = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_sin_theta_value_l417_41706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l417_41745

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) / (x - 1)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x | x < 1 ∨ (1 < x ∧ x ≤ 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l417_41745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_sum_l417_41773

/-- Symmetry with respect to the Oxy plane -/
def symmetry_oxy : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ
  | (x, y, z) => (x, y, -z)

/-- Given two points (3, -1, m) and (3, n, -2) that are symmetric with respect to the Oxy plane,
    prove that m + n = 1 -/
theorem symmetric_points_sum (m n : ℝ) : 
  symmetry_oxy (3, -1, m) = (3, n, -2) →
  m + n = 1 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_sum_l417_41773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l417_41799

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * x

-- State the theorem
theorem inequality_solution_set (a : ℝ) :
  (f (a^2 - 8) + f (2*a) < 0) ↔ (a < -4 ∨ a > 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l417_41799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l417_41794

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  ∃ m : ℝ, (HasDerivAt (f a) m 1 ∧ m = a + 1) :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≤ 0) ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l417_41794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_l417_41787

-- Define the normal filling time
noncomputable def normal_fill_time : ℝ := 10

-- Define the filling time with leak
noncomputable def leak_fill_time : ℝ := 12

-- Define the rate at which the cistern fills without leak
noncomputable def fill_rate : ℝ := 1 / normal_fill_time

-- Define the effective fill rate with leak
noncomputable def effective_fill_rate : ℝ := 1 / leak_fill_time

-- Define the leak rate
noncomputable def leak_rate : ℝ := fill_rate - effective_fill_rate

-- Theorem to prove
theorem leak_empty_time (h1 : normal_fill_time = 10) (h2 : leak_fill_time = 12) :
  1 / leak_rate = 60 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_empty_time_l417_41787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_minimum_dot_product_l417_41736

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector between two points -/
def vector (p q : Point) : Point :=
  ⟨q.x - p.x, q.y - p.y⟩

/-- Dot product of two vectors -/
def dot_product (v w : Point) : ℝ :=
  v.x * w.x + v.y * w.y

/-- Length of a vector -/
noncomputable def vector_length (v : Point) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

-- Define addition for Point
instance : Add Point where
  add (p q : Point) := ⟨p.x + q.x, p.y + q.y⟩

theorem semicircle_minimum_dot_product
  (A B O C : Point)
  (P : ℝ → Point) -- P is a function of a real parameter to represent a moving point
  (h_diameter : vector_length (vector A B) = 4)
  (h_center : O = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩)
  (h_semicircle : vector_length (vector O C) = 2 ∧ C ≠ A ∧ C ≠ B)
  (h_P_on_OC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P t = ⟨O.x + t * (C.x - O.x), O.y + t * (C.y - O.y)⟩) :
  ∀ t : ℝ, dot_product (vector (P t) A + vector (P t) B) (vector (P t) C) ≥ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_minimum_dot_product_l417_41736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pair_sum_l417_41712

/-- A sequence where each sum is one less than the actual sum of the two numbers -/
def SpecialSequence (seq : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (a b s : ℕ), (a, b, s) ∈ seq → s = a + b - 1

/-- The second pair in the sequence is (8, 9) -/
def SecondPairIs8And9 (seq : List (ℕ × ℕ × ℕ)) : Prop :=
  ∃ (s : ℕ), seq.get? 1 = some (8, 9, s)

theorem second_pair_sum (seq : List (ℕ × ℕ × ℕ)) 
  (h1 : SpecialSequence seq) (h2 : SecondPairIs8And9 seq) : 
  ∃ (s : ℕ), seq.get? 1 = some (8, 9, 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_pair_sum_l417_41712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l417_41730

theorem unique_polynomial_with_root_property :
  ∃! (a b c d e : ℝ),
    ∀ (s : ℂ),
      let p : ℂ → ℂ := λ x => x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + 2023
      p s = 0 → p ((-1 - Complex.I * Real.sqrt 3) / 2 * s) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_with_root_property_l417_41730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l417_41783

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  Real.sin A + Real.sqrt 3 * Real.cos A = 2 →
  a = 2 →
  c = Real.sqrt 3 * b →
  A = π / 6 ∧ (1 / 2 : ℝ) * b * c * Real.sin A = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l417_41783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l417_41760

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - 1 / x

theorem tangent_parallel_to_x_axis (a : ℝ) :
  f_derivative a 1 = 0 → a = 1/2 := by
  intro h
  have : 2 * a - 1 = 0 := by
    unfold f_derivative at h
    simp at h
    exact h
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_l417_41760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_equals_four_l417_41707

-- Define the function g
def g (c : ℤ) : ℝ → ℝ := λ x ↦ 2 * x + c

-- State the theorem
theorem intersection_point_d_equals_four (c d : ℤ) :
  (∃ (g_inv : ℝ → ℝ), Function.RightInverse g_inv (g c) ∧ Function.LeftInverse g_inv (g c)) →
  g c 4 = d →
  (∃ (g_inv : ℝ → ℝ), Function.RightInverse g_inv (g c) ∧ Function.LeftInverse g_inv (g c) ∧ g_inv 4 = d) →
  d = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_d_equals_four_l417_41707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_unique_solution_l417_41757

/-- The functional equation for f: ℝ → ℝ -/
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = max (f x) y + min (f y) x

/-- The identity function on ℝ -/
def id_func : ℝ → ℝ := λ x ↦ x

/-- Theorem stating that the identity function is the unique solution to the functional equation -/
theorem identity_unique_solution :
  (functional_equation id_func) ∧
  (∀ g : ℝ → ℝ, functional_equation g → g = id_func) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_unique_solution_l417_41757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discover_points_bound_l417_41765

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A rectangular sheet of paper -/
structure Sheet where
  width : ℝ
  height : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  sheet : Sheet
  circle : Circle
  n : ℕ
  points : Finset Point

/-- Predicate to represent that a point is discovered at a specific attempt -/
def PointDiscovered (p : Point) (i : ℕ) : Prop :=
  sorry  -- We'll leave this undefined for now

/-- Theorem stating that all points can be discovered within (n+1)² attempts -/
theorem discover_points_bound (setup : ProblemSetup) :
  ∃ (attempts : ℕ), attempts ≤ (setup.n + 1)^2 ∧
  (∀ p ∈ setup.points, ∃ (i : ℕ), i ≤ attempts ∧ PointDiscovered p i) := by
  sorry  -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discover_points_bound_l417_41765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l417_41750

/-- A function that returns the number of valid six-digit even numbers -/
def countValidNumbers : ℕ :=
  let n : ℕ := 6  -- number of digits
  let evenDigits : ℕ := 3  -- number of even digits (2, 4, 6)
  let nonAdjacentPlacements : ℕ := Nat.choose 5 2  -- ways to place 1 and 3 non-adjacently
  let remainingDigitArrangements : ℕ := Nat.factorial 4  -- arrangements of remaining 4 digits
  (evenDigits * nonAdjacentPlacements * remainingDigitArrangements) / 2

/-- Theorem stating that the number of valid six-digit even numbers is 360 -/
theorem valid_numbers_count : countValidNumbers = 360 := by
  sorry

#eval countValidNumbers  -- This will evaluate the function and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_numbers_count_l417_41750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_alcohol_ratio_l417_41743

/-- Given two bottles with alcohol-water ratios a:1 and b:1, 
    the mixed solution has an alcohol-water ratio of (a+b+2ab)/(a+b+2) -/
theorem mixed_alcohol_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b + 2*a*b) / (a + b + 2) = ((a / 1) * 1 + (b / 1) * 1) / (1 + 1) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_alcohol_ratio_l417_41743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l417_41722

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = (1/2 : ℝ) ∧ 
  (∀ x : ℝ, 3 * (16 : ℝ) ^ x + 2 * (81 : ℝ) ^ x = 5 * (36 : ℝ) ^ x ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l417_41722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_for_0_23_repeating_l417_41777

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem sum_of_numerator_and_denominator_for_0_23_repeating :
  let x : ℚ := RepeatingDecimal 2 3
  (x.num.natAbs + x.den) = 122 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_numerator_and_denominator_for_0_23_repeating_l417_41777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_decreasing_function_l417_41771

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + (b+3)*x

-- Define the function g(x)
def g (m : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + x - m*x

theorem extreme_value_and_decreasing_function :
  -- Part 1
  (∃ b : ℝ, (∀ x : ℝ, deriv (f b) 1 = 0) ∧ 
   b = -2 ∧ 
   (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f b x ≥ -4) ∧
   f b (-1) = -4) ∧
  -- Part 2
  (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → deriv (g m) x ≤ 0) ↔ m ≥ 21) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_decreasing_function_l417_41771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_theorem_l417_41755

def cupcake_problem (original_batch : ℕ) (sold_original : ℕ) (new_batch_percent : ℚ) 
                    (chocolate_percent : ℚ) (vanilla_percent : ℚ)
                    (chocolate_sold_percent : ℚ) (vanilla_sold_percent : ℚ) : ℕ :=
  let new_batch := Int.floor (new_batch_percent * original_batch)
  let chocolate := Int.floor (chocolate_percent * new_batch)
  let vanilla := new_batch - chocolate
  let chocolate_sold := Int.floor (chocolate_sold_percent * chocolate)
  let vanilla_sold := Int.floor (vanilla_sold_percent * vanilla)
  let remaining_original := original_batch - sold_original
  (remaining_original + (chocolate - chocolate_sold) + (vanilla - vanilla_sold)).toNat

theorem cupcake_theorem : 
  cupcake_problem 26 20 (3/4) (2/5) (3/5) (7/10) (1/2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcake_theorem_l417_41755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_interest_rate_l417_41774

/-- Calculates the compound interest given principal, rate, compounding frequency, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Proves that the given conditions result in a 4% annual interest rate -/
theorem savings_account_interest_rate : 
  ∀ (rate : ℝ), 
  compound_interest 5000 rate 4 0.5 = 5100.50 → 
  rate = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_interest_rate_l417_41774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_y_l417_41701

noncomputable def y (x : ℝ) : ℝ := Real.log (abs (Real.cos (Real.sqrt x))) + Real.sqrt x * Real.tan (Real.sqrt x)

theorem differential_y (x : ℝ) (h : x > 0) :
  deriv y x = 1 / (2 * (Real.cos (Real.sqrt x))^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_y_l417_41701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_proof_l417_41762

/-- The number of ways to arrange 8 distinct objects in a row, 
    where two specific objects must be at the ends and 
    another specific object cannot be adjacent to either end -/
def arrangement_count : ℕ := 960

/-- The number of objects to be arranged -/
def total_objects : ℕ := 8

/-- The number of objects that must be placed at the ends -/
def end_objects : ℕ := 2

/-- The number of positions where the tallest object can be placed -/
def tallest_positions : ℕ := 4

theorem arrangement_count_proof : 
  arrangement_count = 2 * tallest_positions * Nat.factorial (total_objects - end_objects - 1) := by
  sorry

#eval arrangement_count
#eval 2 * tallest_positions * Nat.factorial (total_objects - end_objects - 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_count_proof_l417_41762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_eq_one_l417_41782

theorem sin_minus_cos_eq_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = 1 ↔ x = Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_eq_one_l417_41782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_M_equation_l417_41713

/-- Given a line L with equation y = mx + b, this function returns a new line
    with twice the slope and half the y-intercept of L. -/
def transformLine (m b : ℚ) : ℚ × ℚ := (2 * m, b / 2)

theorem line_M_equation (x y : ℚ) :
  let original_line := λ (x : ℚ) => -5/4 * x - 3
  let (m', b') := transformLine (-5/4) (-3)
  let M := λ (x : ℚ) => m' * x + b'
  (y = M x ∧ M (-4) = original_line (-4)) → y = -5/2 * x - 8 := by
  sorry

#check line_M_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_M_equation_l417_41713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l417_41727

/-- Represents a 3x3 grid of digits --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if all digits from 1 to 9 are used exactly once in the grid --/
def all_digits_used (g : Grid) : Prop := sorry

/-- Checks if the middle row is fixed as "1de" --/
def middle_row_fixed (g : Grid) : Prop := g 1 0 = 0

/-- Checks if the sum of each row is equal --/
def row_sums_equal (g : Grid) : Prop := sorry

/-- Checks if the sum of each column is equal --/
def column_sums_equal (g : Grid) : Prop := sorry

/-- Checks if the product of the first row equals the product of the first column --/
def product_condition (g : Grid) : Prop := sorry

/-- A grid is valid if it satisfies all conditions --/
def is_valid_grid (g : Grid) : Prop :=
  all_digits_used g ∧ middle_row_fixed g ∧ row_sums_equal g ∧ 
  column_sums_equal g ∧ product_condition g

/-- Helper function to count valid grids --/
def count_valid_grids : Nat := sorry

/-- The main theorem stating that there are exactly 30 valid arrangements --/
theorem valid_arrangements_count : count_valid_grids = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l417_41727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_ingot_placements_l417_41756

def storage_complex : ℕ := 161

def placed_ingots : List ℕ := [1, 41, 61, 81, 161]

def is_valid_placement (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ storage_complex ∧ n ∉ placed_ingots

def distance_to_nearest (n : ℕ) : ℕ :=
  placed_ingots.foldl (fun acc x => min acc (Int.natAbs (x - n))) storage_complex

def is_optimal_placement (n : ℕ) : Prop :=
  is_valid_placement n ∧
  ∀ m, is_valid_placement m → distance_to_nearest n ≥ distance_to_nearest m

theorem sixth_ingot_placements :
  ∀ n, is_optimal_placement n ↔ n ∈ [21, 101, 141] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_ingot_placements_l417_41756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l417_41720

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x - 1

-- Theorem for the minimum value of f(x)
theorem f_minimum_value : 
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = -2/Real.exp 1 - 1 := by
  sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) : 
  (∀ (x : ℝ), x > 0 → f x ≤ 3 * x^2 + 2 * a * x) ↔ a ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_a_range_l417_41720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_two_miles_l417_41729

/-- Represents a wheel with a fixed center -/
structure Wheel where
  diameter : ℝ
  
/-- Calculates the number of revolutions for a given distance -/
noncomputable def revolutions (w : Wheel) (distance : ℝ) : ℝ :=
  distance / (w.diameter * Real.pi)

theorem wheel_revolutions_for_two_miles (w : Wheel) 
  (h1 : w.diameter = 10) 
  (h2 : (2 : ℝ) * 5280 = 10560) : 
  revolutions w 10560 = 1056 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_revolutions_for_two_miles_l417_41729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l417_41778

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then 2 else -1

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x < -5 ∨ x > 1

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x + 2*x*(f (x+1)) > 5 ↔ solution_set x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l417_41778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l417_41769

noncomputable section

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2 + 3

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ x in a..b, Real.sqrt (1 + (f' x)^2)

-- Theorem statement
theorem arc_length_of_curve :
  arcLength 0 2 = (Real.exp 2 - Real.exp (-2)) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l417_41769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l417_41746

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * (Real.cos (t.B / 2))^2 = Real.sqrt 3 * Real.sin t.B ∧
  t.a = 3 * t.c

-- Helper function for area
noncomputable def area (t : Triangle) : Real :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.B = π / 3 ∧
  Real.tan t.C = Real.sqrt 3 / 5 ∧
  (t.b = 1 → area t = 3 * Real.sqrt 3 / 28) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l417_41746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l417_41738

theorem number_relationship (a b c d : ℝ) 
  (ha : a = Real.rpow 0.2 3.5) 
  (hb : b = Real.rpow 0.2 4.1) 
  (hc : c = Real.exp 1.1) 
  (hd : d = Real.log 3 / Real.log 0.2) : 
  d < b ∧ b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_relationship_l417_41738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l417_41719

theorem imaginary_part_of_complex_number (b : ℝ) (z : ℂ) : 
  z = 1 + b * Complex.I → Complex.abs z = 2 → b = Real.sqrt 3 ∨ b = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_number_l417_41719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_range_of_g_l417_41700

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -1 then Int.ceil (2 / ((x + 2) * (x + 1)))
  else if x < -2 then Int.floor (2 / ((x + 2) * (x + 1)))
  else 0  -- arbitrary value for undefined region

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, (x > -1 ∨ x < -2) → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_not_in_range_of_g_l417_41700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cells_integer_perimeter_l417_41797

/-- Represents a cell in the rectangular grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the rectangular grid -/
structure Grid where
  cells : List Cell
  known_cells : List Cell
  unknown_cells : List Cell

/-- The perimeter of a cell -/
def perimeter (c : Cell) : ℤ := sorry

/-- Theorem: If 111 cells in an 11x11 grid have integer perimeters, 
    then all cells have integer perimeters -/
theorem all_cells_integer_perimeter (g : Grid) 
  (h1 : g.cells.length = 121)
  (h2 : g.known_cells.length = 111)
  (h3 : g.unknown_cells.length = 10)
  (h4 : ∀ c ∈ g.known_cells, ∃ n : ℤ, perimeter c = n) :
  ∀ c ∈ g.cells, ∃ n : ℤ, perimeter c = n := by
  sorry

#check all_cells_integer_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_cells_integer_perimeter_l417_41797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l417_41710

/-- Defines a valid triangle given three side lengths -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Calculates the circumradius of a triangle given its side lengths -/
noncomputable def circumradius (a b c : ℝ) : ℝ := sorry

/-- Calculates the inradius of a triangle given its side lengths -/
noncomputable def inradius (a b c : ℝ) : ℝ := sorry

/-- Given two triangles ABC and A'B'C', with sides (a, b, c) and (a', b', c'),
    circumradii R and R', and inradii r and r' respectively,
    prove that 36 r' ≤ a a' + b b' + c c' ≤ 9 R R'. -/
theorem triangle_inequality (a b c a' b' c' R R' r r' : ℝ) 
  (h_abc : Triangle a b c) 
  (h_a'b'c' : Triangle a' b' c')
  (h_R : R = circumradius a b c)
  (h_R' : R' = circumradius a' b' c')
  (h_r : r = inradius a b c)
  (h_r' : r' = inradius a' b' c') :
  36 * r' ≤ a * a' + b * b' + c * c' ∧ a * a' + b * b' + c * c' ≤ 9 * R * R' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l417_41710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l417_41770

noncomputable def f (x a : ℝ) : ℝ := 1 / (|x - 1| - a)^2

theorem function_inequality (a : ℝ) (k : ℝ) (h1 : a < 1) :
  (∀ x ∈ Set.Icc 0 2, f x a ≥ k * x^2) →
  ((a < 2/3 → k ≤ 1 / (4 * (1 - a)^2)) ∧
   (2/3 ≤ a → k ≤ 1 / a^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l417_41770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplicative_inverse_101_mod_401_l417_41717

theorem multiplicative_inverse_101_mod_401 :
  ∃ x : ℕ, x < 401 ∧ (101 * x) % 401 = 1 :=
by
  use 135
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplicative_inverse_101_mod_401_l417_41717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_ratio_l417_41788

/-- The volume of the cylinder in cubic centimeters -/
def V : ℝ := 500

/-- The surface area of a cylinder as a function of radius and height -/
noncomputable def surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h

/-- The volume constraint for the cylinder -/
def volume_constraint (r h : ℝ) : Prop := Real.pi * r^2 * h = V

/-- Theorem stating that the ratio h/r = 2 minimizes the surface area of the cylinder -/
theorem min_surface_area_ratio :
  ∀ r h : ℝ, r > 0 → h > 0 → volume_constraint r h →
  ∀ r' h' : ℝ, r' > 0 → h' > 0 → volume_constraint r' h' →
  h / r = 2 → surface_area r h ≤ surface_area r' h' := by
  sorry

#check min_surface_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_surface_area_ratio_l417_41788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arithmetic_progression_l417_41766

theorem right_triangle_arithmetic_progression (a b c : ℕ) : 
  a * a + b * b = c * c →  -- Pythagorean theorem
  b = a + 2 →              -- Arithmetic progression with difference 2
  c = b + 2 →              -- Arithmetic progression with difference 2
  a % 6 = 0 ∧ b % 8 = 0 ∧ c % 10 = 0 ∧   -- Sides are in ratio 6:8:10
  (a = 24 ∨ a = 60 ∨ a = 100 ∨ b = 24 ∨ b = 60 ∨ b = 100 ∨ c = 24 ∨ c = 60 ∨ c = 100) →
  (24 ∈ ({a, b, c} : Set ℕ) ∧ 60 ∈ ({a, b, c} : Set ℕ) ∧ 100 ∈ ({a, b, c} : Set ℕ)) ∧ 
  (85 ∉ ({a, b, c} : Set ℕ) ∧ 375 ∉ ({a, b, c} : Set ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_arithmetic_progression_l417_41766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l417_41772

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (1 + Complex.I)^4 / (1 - Complex.I)
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l417_41772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_14gon_marked_vertices_l417_41784

/-- A regular 14-gon -/
structure RegularPolygon14 :=
  (vertices : Fin 14 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 14, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1))

/-- A set of marked vertices in the 14-gon -/
def MarkedVertices (p : RegularPolygon14) := Finset (Fin 14)

/-- Check if four points form a rectangle -/
def IsRectangle (p : RegularPolygon14) (a b c d : Fin 14) : Prop := sorry

/-- Check if all quadrilaterals with parallel sides formed by marked vertices are rectangles -/
def AllParallelQuadrilateralsAreRectangles (p : RegularPolygon14) (m : MarkedVertices p) : Prop := sorry

/-- The main theorem -/
theorem regular_14gon_marked_vertices (p : RegularPolygon14) :
  (∃ (m : MarkedVertices p), m.card = 6 ∧ AllParallelQuadrilateralsAreRectangles p m) ∧
  (∀ (m : MarkedVertices p), m.card ≥ 7 → ¬AllParallelQuadrilateralsAreRectangles p m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_14gon_marked_vertices_l417_41784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_equals_zero_l417_41708

theorem cos_sum_equals_zero (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = Real.sqrt 2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A + B) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_equals_zero_l417_41708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_property_l417_41724

theorem power_of_two_property (k : ℕ) :
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * Nat.factorial (k*n) / Nat.factorial n ≤ Nat.factorial (k*n) / Nat.factorial n) ↔
  ∃ a : ℕ, k = 2^a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_property_l417_41724
