import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircular_sheet_to_cone_height_l1076_107649

/-- The height of a cone formed from a semicircular sheet of iron -/
noncomputable def cone_height (R : ℝ) : ℝ := (Real.sqrt 3 / 2) * R

/-- Theorem stating the height of the cone formed from a semicircular sheet -/
theorem semicircular_sheet_to_cone_height (R : ℝ) (h : ℝ) (h_pos : h > 0) :
  let r := R / 2  -- radius of the cone's base
  let l := R      -- slant height of the cone
  h^2 + r^2 = l^2 →  -- Pythagorean theorem
  h = cone_height R :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircular_sheet_to_cone_height_l1076_107649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_income_2011_is_13_l1076_107616

/-- Represents the company's financial data and growth rate --/
structure CompanyFinance where
  income_2010 : ℝ  -- Total operating income in 2010
  income_2012 : ℝ  -- Predicted total operating income in 2012
  accessories_percent : ℝ  -- Percentage of income from accessories in 2010
  accessories_income : ℝ  -- Income from accessories in 2010

/-- Calculates the annual growth rate --/
noncomputable def annual_growth_rate (cf : CompanyFinance) : ℝ :=
  (cf.income_2012 / cf.income_2010) ^ (1/2) - 1

/-- Calculates the estimated income for 2011 --/
noncomputable def estimated_income_2011 (cf : CompanyFinance) : ℝ :=
  cf.income_2010 * (1 + annual_growth_rate cf)

/-- Theorem stating that the estimated income for 2011 is 13 million Yuan --/
theorem estimated_income_2011_is_13 (cf : CompanyFinance) 
  (h1 : cf.income_2010 = 10)
  (h2 : cf.income_2012 = 16.9)
  (h3 : cf.accessories_percent = 0.4)
  (h4 : cf.accessories_income = 4) :
  estimated_income_2011 cf = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_income_2011_is_13_l1076_107616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1076_107644

structure Parallelogram :=
  (A B C D : ℂ)
  (P : ℂ)

def divides_internally (P A C : ℂ) (ratio : ℚ) : Prop :=
  P = (ratio * C + A) / (ratio + 1)

theorem locus_of_P (ABCD : Parallelogram) :
  ABCD.A = 0 →
  ABCD.B = 4 - 3*I →
  divides_internally ABCD.P ABCD.A ABCD.C (2/1) →
  Complex.abs (ABCD.D - ABCD.A) = 3 →
  ∃ (center : ℂ) (radius : ℝ), 
    center = 8/3 - 2*I ∧ 
    radius = 2 ∧
    Complex.abs (ABCD.P - center) = radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l1076_107644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_after_manipulations_l1076_107615

/-- Represents the charge of a conducting sphere -/
structure Charge where
  value : ℝ

/-- Represents the final charge of the second sphere after manipulations -/
noncomputable def final_charge (Q₁ Q₂ q₂ : Charge) : Charge :=
  { value := Q₂.value / 2 - q₂.value + Real.sqrt ((Q₂.value / 2) ^ 2 + Q₁.value * q₂.value) }

theorem charge_after_manipulations 
  (Q₁ Q₂ q₂ : Charge) 
  (h₁ : Q₁.value > 0)  -- First sphere initially has positive charge
  (h₂ : Q₂.value > 0)  -- Second sphere initially has positive charge
  (h₃ : q₂.value ≥ 0)  -- Final charge of the ball is non-negative
  : ∃ Q₂' : Charge, 
    Q₂'.value = (final_charge Q₁ Q₂ q₂).value ∨ 
    Q₂'.value = Q₂.value / 2 - q₂.value - Real.sqrt ((Q₂.value / 2) ^ 2 + Q₁.value * q₂.value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_charge_after_manipulations_l1076_107615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_ten_in_eighteen_factorial_l1076_107654

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem highest_power_of_ten_in_eighteen_factorial :
  (∃ k : ℕ, 10^k ∣ factorial 18 ∧ ¬(10^(k+1) ∣ factorial 18)) → 
  (∃ k : ℕ, k = 3 ∧ 10^k ∣ factorial 18 ∧ ¬(10^(k+1) ∣ factorial 18)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_power_of_ten_in_eighteen_factorial_l1076_107654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1076_107669

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (π/2) → a ≤ 1/Real.sin θ + 1/Real.cos θ) → 
  a ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1076_107669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_five_l1076_107680

/-- Represents the properties of a rectangular floor and its painting cost -/
structure RectangularFloor where
  length : ℝ
  breadth : ℝ
  total_cost : ℝ
  length_breadth_relation : length = 3 * breadth
  length_value : length = 19.595917942265423
  cost_value : total_cost = 640

/-- Calculates the painting rate per square meter for a given rectangular floor -/
noncomputable def painting_rate (floor : RectangularFloor) : ℝ :=
  floor.total_cost / (floor.length * floor.breadth)

/-- Theorem stating that the painting rate is 5 Rs. per square meter -/
theorem painting_rate_is_five (floor : RectangularFloor) :
  painting_rate floor = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_painting_rate_is_five_l1076_107680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_equals_one_l1076_107631

/-- Given a line and a circle that intersect, prove that the positive integer m must equal 1 -/
theorem intersection_implies_m_equals_one (m : ℕ+) : 
  (∃ x y : ℝ, 3 * x - 2 * y = 0 ∧ (x - m.val)^2 + y^2 = 1) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_equals_one_l1076_107631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_loss_is_25_percent_l1076_107619

/-- The height of one story in feet -/
noncomputable def story_height : ℝ := 10

/-- The number of stories Tom needs to lower the rope -/
def num_stories : ℕ := 6

/-- The length of one piece of rope in feet -/
noncomputable def rope_piece_length : ℝ := 20

/-- The number of rope pieces Tom needs to buy -/
def num_rope_pieces : ℕ := 4

/-- The total length of rope needed in feet -/
noncomputable def total_length_needed : ℝ := story_height * (num_stories : ℝ)

/-- The total length of rope bought before lashing in feet -/
noncomputable def total_length_bought : ℝ := rope_piece_length * (num_rope_pieces : ℝ)

/-- The percentage of rope length lost when lashing pieces together -/
noncomputable def rope_loss_percentage : ℝ := 
  (total_length_bought - total_length_needed) / total_length_bought * 100

theorem rope_loss_is_25_percent : rope_loss_percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_loss_is_25_percent_l1076_107619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l1076_107673

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  let b := l.y₁ - m * l.x₁
  (0, b)

/-- The specific line passing through (2, 3) and (6, 15) -/
def specific_line : Line :=
  { x₁ := 2, y₁ := 3, x₂ := 6, y₂ := 15 }

theorem line_intersects_y_axis :
  y_intercept specific_line = (0, -3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_y_axis_l1076_107673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1076_107685

-- Define the point P
noncomputable def P : ℝ × ℝ := (-Real.log 2, 2)

-- Define the curve function
noncomputable def curve (x : ℝ) : ℝ := Real.exp (-x)

-- Define the slope of the line 2x + y + 1 = 0
def line_slope : ℝ := -2

theorem point_satisfies_conditions :
  -- Condition 1: P is on the curve y = e^(-x)
  (P.2 = curve P.1) ∧
  -- Condition 2: The tangent line at P is parallel to 2x + y + 1 = 0
  (Real.exp (-P.1) = line_slope) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_satisfies_conditions_l1076_107685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l1076_107625

/-- The distance between two parallel lines in 2D space --/
noncomputable def distance_between_parallel_lines (a b c d : ℝ) : ℝ :=
  |c - d| / Real.sqrt (a^2 + b^2)

/-- The first line: x + y + 1 = 0 --/
def line1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}

/-- The second line: x + y - 1 = 0 --/
def line2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 1 = 0}

/-- Theorem: The distance between line1 and line2 is √2 --/
theorem distance_between_lines : 
  distance_between_parallel_lines 1 1 (-1) 1 = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l1076_107625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1076_107694

theorem trigonometric_identity (α β : Real) : 
  4.28 * (Real.sin (β/2 - π/2))^2 - (Real.cos (α - 3*π/2))^2 = Real.cos (α + β) * Real.cos (α - β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l1076_107694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_true_and_q_false_l1076_107676

theorem proposition_p_true_and_q_false :
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ 4) ∧ ¬(∃ x : ℝ, (2 : ℝ)^x = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_true_and_q_false_l1076_107676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_decreasing_g_implies_t_range_roots_of_equation_l1076_107614

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp x + a)
def g (lambda : ℝ) (a : ℝ) (x : ℝ) : ℝ := lambda * f a x + Real.sin x

-- Theorem 1
theorem odd_function_implies_a_zero (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → a = 0 := by sorry

-- Theorem 2
theorem decreasing_g_implies_t_range (lambda : ℝ) (t : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, HasDerivAt (g lambda 0) (lambda + Real.cos x) x) →
  (∀ x ∈ Set.Icc (-1) 1, g lambda 0 x ≤ t^2 + lambda*t + 1) →
  t ≤ -1 := by sorry

-- Theorem 3
theorem roots_of_equation (m : ℝ) :
  (∀ x > 0, (Real.log x / x = x^2 - 2 * Real.exp 1 * x + m) ↔
    (x = 0 ∨ 
     (m > Real.exp 2 + 1 / Real.exp 1) ∨
     (m = Real.exp 2 + 1 / Real.exp 1 ∧ x = Real.exp 1) ∨
     (m < Real.exp 2 + 1 / Real.exp 1 ∧ (x < Real.exp 1 ∨ x > Real.exp 1)))) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_zero_decreasing_g_implies_t_range_roots_of_equation_l1076_107614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_l1076_107662

/-- Represents the total worth of a stock in Rupees -/
def stock_worth (x : ℝ) : Prop :=
  (0.1 * x * 1.2 + 0.9 * x * 0.95 = x - 400)

/-- Theorem stating the total worth of the stock -/
theorem stock_value :
  ∃ x, stock_worth x ∧ x = 16000 := by
  use 16000
  constructor
  · -- Prove that 16000 satisfies the stock_worth condition
    simp [stock_worth]
    norm_num
  · -- Prove that x = 16000
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_l1076_107662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_needs_change_l1076_107687

/-- Represents the cost of a toy in quarters -/
def ToyCost := Fin 10 → Nat

/-- The machine with 10 toys -/
structure ToyMachine where
  toys : ToyCost
  /-- Each toy costs between 50 cents and $2.50 -/
  cost_range : ∀ i, 2 ≤ toys i ∧ toys i ≤ 10
  /-- Each toy is 25 cents more expensive than the next most expensive one -/
  cost_difference : ∀ i, i.val < 9 → toys i = toys i.succ + 1

/-- Sam's initial money in quarters -/
def initial_money : Nat := 10

/-- Cost of Sam's favorite toy in quarters -/
def favorite_toy_cost : Nat := 9

/-- Total number of possible toy dispensing orders -/
def total_orders : Nat := Nat.factorial 10

/-- Number of favorable orders where Sam can buy his favorite toy without change -/
def favorable_orders : Nat := 2 * Nat.factorial 8

/-- Probability that Sam needs to get change -/
def change_probability (m : ToyMachine) : Rat :=
  1 - (favorable_orders : Rat) / (total_orders : Rat)

/-- Main theorem: The probability that Sam needs to get change is 44/45 -/
theorem sam_needs_change (m : ToyMachine) : 
  change_probability m = 44 / 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sam_needs_change_l1076_107687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1076_107688

-- Define the function f(x) = ln((1-x)/(1+x))
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Theorem statement
theorem f_properties :
  -- f(x) is defined on the open interval (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → (1 - x) / (1 + x) > 0) ∧
  -- f(x) is an odd function
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  -- Addition property
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 →
    f x₁ + f x₂ = f ((x₁ + x₂) / (1 + x₁ * x₂))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1076_107688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1076_107646

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The foci of an ellipse -/
def Foci (E : Ellipse a b) : Set (ℝ × ℝ) := sorry

/-- A point on an ellipse -/
def PointOnEllipse (E : Ellipse a b) : Set (ℝ × ℝ) := sorry

/-- The angle between two points and a third point -/
noncomputable def Angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The distance between two points -/
noncomputable def Distance (A B : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse -/
noncomputable def Eccentricity (E : Ellipse a b) : ℝ := sorry

theorem ellipse_eccentricity (a b : ℝ) (E : Ellipse a b) 
  (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) :
  F₁ ∈ Foci E → F₂ ∈ Foci E → P ∈ PointOnEllipse E →
  Angle F₁ P F₂ = π / 3 →
  Distance P F₁ = 5 * Distance P F₂ →
  Eccentricity E = Real.sqrt 21 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1076_107646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_percentage_l1076_107659

/-- The side length of the main square -/
noncomputable def main_square_side : ℝ := 8

/-- The side length of the small shaded square in the corner -/
noncomputable def small_square_side : ℝ := 2

/-- The side length of the larger shaded square region -/
noncomputable def large_square_side : ℝ := 4

/-- The side length of the unshaded square hole in the larger shaded region -/
noncomputable def hole_side : ℝ := 3

/-- The total shaded area -/
noncomputable def shaded_area : ℝ := small_square_side ^ 2 + large_square_side ^ 2 - hole_side ^ 2

/-- The total area of the main square -/
noncomputable def total_area : ℝ := main_square_side ^ 2

/-- The percentage of the shaded area -/
noncomputable def shaded_percentage : ℝ := (shaded_area / total_area) * 100

theorem shaded_area_percentage :
  shaded_percentage = 17.1875 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_percentage_l1076_107659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1076_107658

noncomputable section

open Real

theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / (sin A) = b / (sin B) →
  a / (sin A) = c / (sin C) →
  b + c = 2 * a →
  2 * sin A = 3 * sin C →
  cos B = -1/4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1076_107658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_theorem_l1076_107605

/-- Represents the train's journey between two stations -/
structure TrainJourney where
  total_distance : ℝ
  flat_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ
  speed_ratio_uphill : ℝ
  speed_ratio_flat : ℝ
  speed_ratio_downhill : ℝ
  flat_speed_ab : ℝ

/-- Calculates the travel time for a given journey and speed -/
noncomputable def travel_time (j : TrainJourney) (flat_speed : ℝ) : ℝ :=
  j.flat_distance / flat_speed +
  j.uphill_distance / (flat_speed * j.speed_ratio_uphill / j.speed_ratio_flat) +
  j.downhill_distance / (flat_speed * j.speed_ratio_downhill / j.speed_ratio_flat)

/-- The main theorem stating the two parts of the problem -/
theorem train_journey_theorem (j : TrainJourney)
  (h1 : j.total_distance = 800)
  (h2 : j.flat_distance = 400)
  (h3 : j.uphill_distance = 300)
  (h4 : j.downhill_distance = 100)
  (h5 : j.speed_ratio_uphill = 3)
  (h6 : j.speed_ratio_flat = 4)
  (h7 : j.speed_ratio_downhill = 5)
  (h8 : j.flat_speed_ab = 80) :
  (∃ (t : ℝ), t = travel_time j j.flat_speed_ab - travel_time j (j.flat_speed_ab * j.speed_ratio_downhill / j.speed_ratio_uphill) ∧ t = 4/3) ∧
  (∃ (r : ℝ), r = j.flat_speed_ab / (j.flat_speed_ab * 29/33) ∧ 
    travel_time j j.flat_speed_ab = travel_time j (j.flat_speed_ab * 29/33)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_theorem_l1076_107605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pumped_in_30_minutes_l1076_107609

/-- The rate at which the pump operates in gallons per hour -/
noncomputable def pump_rate : ℚ := 500

/-- The time in minutes for which we want to calculate the water pumped -/
noncomputable def time_minutes : ℚ := 30

/-- Converts minutes to hours -/
noncomputable def minutes_to_hours (m : ℚ) : ℚ := m / 60

/-- Calculates the amount of water pumped given a rate and time -/
noncomputable def water_pumped (rate : ℚ) (time : ℚ) : ℚ := rate * time

theorem water_pumped_in_30_minutes :
  water_pumped pump_rate (minutes_to_hours time_minutes) = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pumped_in_30_minutes_l1076_107609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ellipse_theorem_l1076_107683

/-- The eccentricity of the "Golden Ellipse" -/
noncomputable def golden_ellipse_eccentricity : ℝ := (Real.sqrt 5 - 1) / 2

/-- Definition of an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_relation : a^2 = b^2 + c^2
  h_order : c < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The Golden Ellipse theorem -/
theorem golden_ellipse_theorem (e : Ellipse) :
  e.a^2 + e.c^2 = (e.a + e.c)^2 →
  eccentricity e = golden_ellipse_eccentricity := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ellipse_theorem_l1076_107683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mutually_acquainted_trio_l1076_107686

/-- Represents a school with students and their acquaintances -/
structure School (n : ℕ) where
  students : Finset (Fin n)
  acquaintances : Fin n → Finset (Fin n)
  acquaintance_count : ∀ s, (acquaintances s).card = n + 1

/-- The main theorem statement -/
theorem exists_mutually_acquainted_trio
  {n : ℕ} (school1 school2 school3 : School n) :
  ∃ (s1 : Fin n) (s2 : Fin n) (s3 : Fin n),
    s2 ∈ school2.acquaintances s1 ∧
    s3 ∈ school3.acquaintances s1 ∧
    s3 ∈ school2.acquaintances s2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_mutually_acquainted_trio_l1076_107686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1076_107647

/-- An arithmetic sequence with first term a₁ and common difference d -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * seq.a₁ + (n * (n - 1) / 2) * seq.d

/-- The line y = a₁x intersects the circle (x-2)² + y² = 4 at two points -/
def has_two_intersections (seq : ArithmeticSequence) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    y₁ = seq.a₁ * x₁ ∧
    y₂ = seq.a₁ * x₂ ∧
    (x₁ - 2)^2 + y₁^2 = 4 ∧
    (x₂ - 2)^2 + y₂^2 = 4 ∧
    x₁ ≠ x₂

/-- The intersection points are symmetric about the line x + y + d = 0 -/
def symmetric_about_line (seq : ArithmeticSequence) : Prop :=
  ∀ x₁ x₂ y₁ y₂ : ℝ,
    y₁ = seq.a₁ * x₁ →
    y₂ = seq.a₁ * x₂ →
    (x₁ - 2)^2 + y₁^2 = 4 →
    (x₂ - 2)^2 + y₂^2 = 4 →
    x₁ ≠ x₂ →
    x₁ + y₁ + seq.d = -(x₂ + y₂ + seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) (n : ℕ) :
  has_two_intersections seq →
  symmetric_about_line seq →
  sum_n seq n = -n^2 + 2*n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1076_107647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l1076_107666

/-- The repeating decimal 0.353535... as a real number -/
noncomputable def repeating_decimal : ℚ := 35 / 99

/-- The fraction 35/99 as a rational number -/
def fraction : ℚ := 35 / 99

/-- Theorem stating that the repeating decimal 0.353535... is equal to the fraction 35/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  -- Unfold the definitions
  unfold repeating_decimal fraction
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equals_fraction_l1076_107666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1076_107655

-- Define the ellipse (C)
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line (l)
def line (x₀ y₀ x y : ℝ) : Prop := x₀ * x / 2 + y₀ * y = 1

-- Define the point P on the ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse x₀ y₀ ∧ y₀ ≠ 0

-- Define the eccentricity of the ellipse
noncomputable def eccentricity : ℝ := Real.sqrt 2 / 2

-- Define the area of triangle OAB
noncomputable def area_OAB (x₀ y₀ : ℝ) : ℝ := 1 / (abs (x₀ * y₀))

-- Define the minimum area of triangle OAB
noncomputable def min_area_OAB : ℝ := Real.sqrt 2

-- Define the collinearity of points
def collinear (Q P F₂ : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q.1 - P.1 = t * (F₂.1 - P.1) ∧ Q.2 - P.2 = t * (F₂.2 - P.2)

-- Main theorem
theorem ellipse_properties (x₀ y₀ : ℝ) (h : point_on_ellipse x₀ y₀) :
  -- 1. The eccentricity of the ellipse is √2/2
  eccentricity = Real.sqrt 2 / 2 ∧
  -- 2. The minimum area of triangle OAB is √2
  (∀ x₀' y₀', point_on_ellipse x₀' y₀' → area_OAB x₀' y₀' ≥ min_area_OAB) ∧
  -- 3. Q, P, and F₂ are collinear
  ∃ (Q F₁ F₂ : ℝ × ℝ), collinear Q (x₀, y₀) F₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1076_107655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l1076_107612

-- Define the points
def A : ℝ × ℝ := (4, 7)
def D : ℝ × ℝ := (10, 7)

-- Define the reflection points
def B : ℝ × ℝ := (0, 7)
def C : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem laser_beam_distance :
  distance A B + distance B C + distance C D = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_beam_distance_l1076_107612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1076_107689

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- The asymptotes of a hyperbola -/
noncomputable def asymptotes (h : Hyperbola) : (ℝ → ℝ) × (ℝ → ℝ) :=
  (fun x => (h.b / h.a) * x, fun x => -(h.b / h.a) * x)

theorem hyperbola_eccentricity (h : Hyperbola) 
  (l₁ l₂ : ℝ → ℝ) 
  (hasymp : (l₁, l₂) = asymptotes h)
  (A B : ℝ × ℝ)
  (hA : A.2 > 0)
  (hB : B.2 > 0)
  (hOA : Real.sqrt (A.1^2 + A.2^2) = 3)
  (hOB : Real.sqrt (B.1^2 + B.2^2) = 5)
  (hperp : (B.2 - A.2) * (1 + l₁ A.1) = -(B.1 - A.1)) :
  eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1076_107689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1076_107648

/-- Two circles O₁ and O₂ tangent to x-axis with centers collinear with origin -/
structure TangentCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  tangent_to_x_axis : O₁.2 = O₁.1 ∧ O₂.2 = O₂.1
  collinear_with_origin : ∃ (k : ℝ), O₂.1 = k * O₁.1

/-- The product of x-coordinates of O₁ and O₂ is 6 -/
def product_of_x_coords (tc : TangentCircles) : Prop :=
  tc.O₁.1 * tc.O₂.1 = 6

/-- The line l: 2x - y - 8 = 0 -/
def line_l (x y : ℝ) : Prop :=
  2 * x - y - 8 = 0

/-- Point P is an intersection point of the two circles -/
def is_intersection_point (P : ℝ × ℝ) (tc : TangentCircles) : Prop :=
  (P.1 - tc.O₁.1)^2 + (P.2 - tc.O₁.2)^2 = tc.O₁.1^2 ∧
  (P.1 - tc.O₂.1)^2 + (P.2 - tc.O₂.2)^2 = tc.O₂.1^2

/-- The distance between two points -/
noncomputable def distance (P M : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)

/-- The minimum distance theorem -/
theorem min_distance_theorem (tc : TangentCircles) (P : ℝ × ℝ) 
    (h₁ : product_of_x_coords tc) (h₂ : is_intersection_point P tc) :
    (∀ M : ℝ × ℝ, line_l M.1 M.2 → distance P M ≥ 8 * Real.sqrt 5 / 5 - Real.sqrt 6) ∧
    (∃ M : ℝ × ℝ, line_l M.1 M.2 ∧ distance P M = 8 * Real.sqrt 5 / 5 - Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l1076_107648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1076_107663

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- The theorem stating that A(0, 0, -1) is equidistant from B(3, 1, 3) and C(1, 4, 2) -/
theorem equidistant_point : distance 0 0 (-1) 3 1 3 = distance 0 0 (-1) 1 4 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l1076_107663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_monotonic_g_l1076_107623

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

theorem min_phi_for_monotonic_g :
  ∀ φ : ℝ, φ > 0 →
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/6), StrictMono (g φ)) →
  φ ≥ Real.pi/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_monotonic_g_l1076_107623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1076_107682

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t / 2, 2 + (Real.sqrt 3 / 2) * t)

-- Define the curve C in polar form
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin θ

-- Define the Cartesian form of curve C
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 2 * Real.sqrt 3 * y

-- State the theorem
theorem intersection_length :
  ∃ A B : ℝ × ℝ,
    (∃ t : ℝ, line_l t = A) ∧
    (∃ t : ℝ, line_l t = B) ∧
    curve_C_cartesian A.1 A.2 ∧
    curve_C_cartesian B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt (2 * Real.sqrt 3 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1076_107682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_A_l1076_107626

/-- The distance between two points A and B in kilometers. -/
def distance_AB : ℝ := 100

/-- The time in hours it takes car A to travel from A to B. -/
def m : ℕ := sorry

/-- The time in hours it takes car B to travel from A to B. -/
def n : ℕ := sorry

/-- The time in hours it takes for the cars to meet. -/
def meeting_time : ℝ := 5

/-- The speed of car A in kilometers per hour. -/
noncomputable def speed_A : ℝ := distance_AB / m

/-- The speed of car B in kilometers per hour. -/
noncomputable def speed_B : ℝ := distance_AB / n

/-- The condition that the cars meet after 5 hours. -/
axiom cars_meet : speed_A * meeting_time + speed_B * meeting_time = distance_AB

/-- The condition that car A has covered half its journey when they meet. -/
axiom car_A_half_journey : 2 * (speed_A * meeting_time) = distance_AB

/-- The theorem stating the distance from C to A. -/
theorem distance_C_to_A : speed_A * meeting_time = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_A_l1076_107626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l1076_107610

/-- Given line with slope 5/3 and y-intercept 2 -/
noncomputable def given_line (x : ℝ) : ℝ := (5/3) * x + 2

/-- Distance between two parallel lines -/
noncomputable def distance_parallel_lines (m c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (m^2 + 1)

/-- Theorem: Equation of a parallel line at distance 3 -/
theorem parallel_line_equation (x : ℝ) :
  ∃ (c : ℝ), 
    (distance_parallel_lines (5/3) 2 c = 3) ∧
    ((c = 2 + Real.sqrt 34) ∨ (c = 2 - Real.sqrt 34)) ∧
    (given_line x + 3 * Real.sqrt ((5/3)^2 + 1) = (5/3) * x + c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_equation_l1076_107610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_accounts_not_always_one_account_l1076_107642

/-- Represents a bank account with an integer balance -/
structure BankAccount where
  balance : ℕ

/-- Represents three bank accounts -/
structure ThreeAccounts where
  account1 : BankAccount
  account2 : BankAccount
  account3 : BankAccount

/-- Transfers money from one account to another, doubling the receiving account's balance -/
def transfer (a b : BankAccount) : BankAccount × BankAccount :=
  (⟨a.balance - b.balance⟩, ⟨2 * b.balance⟩)

/-- Checks if it's possible to make one account's balance zero -/
def canMakeOneAccountZero (accounts : ThreeAccounts) : Prop :=
  ∃ (n : ℕ), ∃ (_transferSequence : Fin n → BankAccount × BankAccount),
    (accounts.account1.balance = 0 ∨ accounts.account2.balance = 0 ∨ accounts.account3.balance = 0)

/-- Checks if it's possible to transfer all money to one account -/
def canTransferAllToOne (accounts : ThreeAccounts) : Prop :=
  ∃ (n : ℕ), ∃ (_transferSequence : Fin n → BankAccount × BankAccount),
    (accounts.account1.balance = 0 ∧ accounts.account2.balance = 0) ∨
    (accounts.account1.balance = 0 ∧ accounts.account3.balance = 0) ∨
    (accounts.account2.balance = 0 ∧ accounts.account3.balance = 0)

theorem always_two_accounts (accounts : ThreeAccounts) :
  canMakeOneAccountZero accounts := by
  sorry

theorem not_always_one_account (accounts : ThreeAccounts) :
  (accounts.account1.balance + accounts.account2.balance + accounts.account3.balance) % 2 = 1 →
  ¬ canTransferAllToOne accounts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_accounts_not_always_one_account_l1076_107642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_bread_amount_bread_sum_correct_l1076_107636

/-- The number of loaves of wheat bread served at a restaurant -/
def wheat_bread : ℝ := 0.2

/-- The number of loaves of white bread served at a restaurant -/
def white_bread : ℝ := 0.4

/-- The total number of loaves of bread served at a restaurant -/
def total_bread : ℝ := 0.6

/-- Theorem stating that the number of loaves of wheat bread served is 0.2 -/
theorem wheat_bread_amount : wheat_bread = 0.2 := by
  -- The proof is trivial since we defined wheat_bread as 0.2
  rfl

/-- Theorem verifying that the sum of wheat and white bread equals the total bread -/
theorem bread_sum_correct : wheat_bread + white_bread = total_bread := by
  -- Unfold the definitions and perform the calculation
  unfold wheat_bread white_bread total_bread
  -- The proof is now just checking that 0.2 + 0.4 = 0.6
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheat_bread_amount_bread_sum_correct_l1076_107636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_mowing_lawn_l1076_107617

theorem mary_mowing_lawn (mary_rate : ℝ) (tom_rate : ℝ) (mary_time : ℝ) :
  mary_rate = 1/3 →
  tom_rate = 1/6 →
  mary_time = 1 →
  2/3 = 1 - mary_rate * mary_time :=
by
  intros h_mary h_tom h_time
  rw [h_mary, h_time]
  norm_num

#check mary_mowing_lawn

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_mowing_lawn_l1076_107617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l1076_107643

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 3*x + 2*a else x - a * Real.log x

theorem f_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (1 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

#check f_nonnegative_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_in_range_l1076_107643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_square_root_of_sixteen_l1076_107645

theorem arithmetic_square_root_of_sixteen : ∀ x : ℝ, x = Real.sqrt 16 ∧ x > 0 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_square_root_of_sixteen_l1076_107645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1076_107607

noncomputable section

def e : ℝ := Real.exp 1

def f (a b x : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1

def g (a b x : ℝ) : ℝ := Real.exp x - 2 * a * x - b

noncomputable def minValue (a b : ℝ) : ℝ :=
  if a ≤ 1/2 then 1 - b
  else if a < e/2 then 2 * a - 2 * a * Real.log (2 * a) - b
  else e - 2 * a - b

theorem min_value_g (a b : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g a b x ≥ minValue a b) ∧
  (∃ x ∈ Set.Icc 0 1, g a b x = minValue a b) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l1076_107607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_color_difference_l1076_107620

-- Define the chessboard plane
def ChessboardPlane := ℝ × ℝ

-- Define color as an inductive type
inductive Color
| Black
| Red
| Blue
| Green
| Yellow

-- Define the coloring function
noncomputable def color : ChessboardPlane → Color :=
  sorry

-- Define the line ℓ
noncomputable def line_ℓ : Set ChessboardPlane :=
  sorry

-- Define a line segment parallel to ℓ
def parallel_segment (I : Set ChessboardPlane) : Prop :=
  sorry

-- Define the length of red area on a segment
noncomputable def red_length (I : Set ChessboardPlane) : ℝ :=
  sorry

-- Define the length of blue area on a segment
noncomputable def blue_length (I : Set ChessboardPlane) : ℝ :=
  sorry

-- The main theorem
theorem bounded_color_difference :
  ∃ (C : ℝ), ∀ (I : Set ChessboardPlane),
    parallel_segment I →
    |red_length I - blue_length I| < C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_color_difference_l1076_107620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1076_107641

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  Real.cos (2 * C) - 3 * Real.cos (A + B) = 1 →
  -- Given c = √6
  c = Real.sqrt 6 →
  -- Prove C = π/3
  C = π / 3 ∧
  -- Prove maximum perimeter is 3√6
  ∃ (p : ℝ), p = a + b + c ∧ p ≤ 3 * Real.sqrt 6 ∧
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    a' + b' + c ≤ p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l1076_107641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_satisfying_inequality_l1076_107601

theorem greatest_x_satisfying_inequality :
  ∀ x : ℕ, x ≤ 4 ↔ (x : ℝ)^4 / (x : ℝ)^2 + 5 < 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_satisfying_inequality_l1076_107601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1076_107699

/-- Given a train and platform with specified lengths and crossing time, 
    calculate the time taken to cross a signal pole. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 600)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 39) : 
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 :=
by
  -- Convert the exact rational to a real number
  have : (600 : ℝ) / ((600 + 700) / 39) = 18 := by norm_num
  -- Replace the variables with their values
  rw [h1, h2, h3]
  -- The goal now matches our 'have' statement
  exact this

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1076_107699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_from_circle_to_line_l1076_107665

/-- The circle centered at (2, 0) with radius 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- The line x - y + 2 = 0 -/
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

/-- The shortest distance from a point on the circle to the line -/
noncomputable def shortest_distance : ℝ := 2 * Real.sqrt 2 - 1

theorem shortest_distance_from_circle_to_line :
  ∀ (P : ℝ × ℝ), circle_eq P.1 P.2 →
    ∃ (d : ℝ), d = shortest_distance ∧
      ∀ (Q : ℝ × ℝ), line_eq Q.1 Q.2 →
        d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_from_circle_to_line_l1076_107665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_prime_sum_sequence_l1076_107618

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d > 1 → d < n → n % d ≠ 0)

/-- Sums a finite subsequence of a sequence -/
def sumSubsequence (s : Sequence) (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (fun i => s (start + i)) |>.sum

/-- A sequence where no finite consecutive subsequence sums to a prime -/
def nonPrimeSequence (s : Sequence) : Prop :=
  ∀ start finish : ℕ, start ≤ finish → ¬isPrime (sumSubsequence s start finish)

/-- Every natural number appears in the sequence exactly once -/
def bijective (s : Sequence) : Prop :=
  Function.Bijective s

/-- There exists a sequence of natural numbers where no consecutive subsequence sums to a prime,
    and every natural number appears exactly once -/
theorem exists_non_prime_sum_sequence :
  ∃ s : Sequence, nonPrimeSequence s ∧ bijective s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_prime_sum_sequence_l1076_107618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_solution_l1076_107635

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem irrational_solution (x : ℝ) : frac x + frac (1 / x) = 1 → ¬ (∃ (q : ℚ), ↑q = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_solution_l1076_107635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_l1076_107690

-- Define the functions h and f
noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)

noncomputable def h (x : ℝ) : ℝ := 4 * (Function.invFun f x)

-- State the theorem
theorem h_equals_20 :
  ∃ x : ℝ, h x = 20 ∧ x = 30 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_equals_20_l1076_107690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_integers_count_l1076_107693

theorem eight_digit_integers_count : 
  9 * (10 ^ 7) = 90000000 := by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_integers_count_l1076_107693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_division_point_l1076_107611

/-- Given points O, A, B, and C on a plane, where C divides AB in the ratio of 2,
    prove that OC = (1/3)OA + (2/3)OB -/
theorem vector_division_point (O A B C : EuclideanSpace ℝ (Fin 2)) 
  (h : C - A = 2 • (B - C)) : 
  C - O = (1/3) • (A - O) + (2/3) • (B - O) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_division_point_l1076_107611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_property_l1076_107650

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse_M P.1 P.2

-- Define a line passing through (4,0) and another point
def line_through_4_0 (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 4)

-- Define the reflection of a point about the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Main theorem
theorem ellipse_focus_property (k : ℝ) (P Q : ℝ × ℝ) :
  k ≠ 0 →
  point_on_ellipse P →
  point_on_ellipse Q →
  line_through_4_0 k P.1 P.2 →
  line_through_4_0 k Q.1 Q.2 →
  let E := reflect_x_axis Q
  ∃ t : ℝ, t * P.1 + (1 - t) * E.1 = 1 ∧ t * P.2 + (1 - t) * E.2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_property_l1076_107650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compounding_difference_l1076_107639

/-- Calculates the amount owed after compound interest --/
noncomputable def amountOwed (principal : ℝ) (rate : ℝ) (compoundsPerYear : ℕ) (years : ℕ) : ℝ :=
  principal * (1 + rate / (compoundsPerYear : ℝ)) ^ ((compoundsPerYear : ℝ) * (years : ℝ))

/-- The difference in amount owed between monthly and semi-annual compounding --/
noncomputable def compoundingDifference (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  amountOwed principal rate 12 years - amountOwed principal rate 2 years

theorem loan_compounding_difference :
  let principal := (8000 : ℝ)
  let rate := (0.1 : ℝ)
  let years := (5 : ℕ)
  ∃ ε > 0, ε < 0.005 ∧ 
    |compoundingDifference principal rate years - 426.13| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_compounding_difference_l1076_107639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1076_107603

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  if a ≥ b then b else a

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := odot x (2 - x)

-- Statement of the theorem
theorem range_of_f :
  Set.range f = Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1076_107603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_billiard_distances_l1076_107695

/-- Represents an elliptical billiard table -/
structure EllipticalTable where
  a : ℝ  -- Half of the major axis length
  c : ℝ  -- Half of the focal distance
  h : 0 < c ∧ c < a  -- Conditions for a valid ellipse

/-- Represents a trajectory of a ball on the elliptical table -/
inductive Trajectory
  | LeftVertex  : Trajectory  -- Ball reflects off the left vertex
  | RightVertex : Trajectory  -- Ball reflects off the right vertex
  | OtherPoint  : Trajectory  -- Ball reflects off any other point

/-- Calculates the distance traveled by the ball for a given trajectory -/
def distanceTraveled (table : EllipticalTable) (traj : Trajectory) : ℝ :=
  match traj with
  | Trajectory.LeftVertex  => 2 * (table.a + table.c)
  | Trajectory.RightVertex => 2 * (table.a - table.c)
  | Trajectory.OtherPoint  => 4 * table.a

/-- Theorem stating the possible distances traveled by the ball -/
theorem elliptical_billiard_distances (table : EllipticalTable) :
  ∃ (traj : Trajectory), (distanceTraveled table traj) ∈ ({2*(table.a-table.c), 2*(table.a+table.c), 4*table.a} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_billiard_distances_l1076_107695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sequence_69th_term_l1076_107671

/-- Given a sequence where 3 numbers are inserted between each pair of consecutive terms,
    the 69th term of the new sequence corresponds to the 18th term of the original sequence. -/
theorem new_sequence_69th_term (a : ℕ → ℝ) : 
  let new_seq := λ (n : ℕ) => if n % 4 = 1 then a ((n - 1) / 4 + 1) else 0
  new_seq 69 = a 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_sequence_69th_term_l1076_107671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l1076_107657

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a segment connecting two points -/
structure Segment where
  p1 : Point
  p2 : Point

/-- The game setup -/
structure GameSetup where
  points : Finset Point
  segments : Finset Segment
  num_points : ℕ
  point_assignment : Point → ℕ
  segment_assignment : Segment → ℕ

/-- The game conditions -/
def valid_game_setup (g : GameSetup) : Prop :=
  g.num_points = 2005 ∧
  ∀ p1 p2 p3, p1 ∈ g.points → p2 ∈ g.points → p3 ∈ g.points → p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
    (p3.y - p1.y) * (p2.x - p1.x) ≠ (p2.y - p1.y) * (p3.x - p1.x) ∧
  ∀ p1 p2, p1 ∈ g.points → p2 ∈ g.points → p1 ≠ p2 → ∃ s ∈ g.segments, s.p1 = p1 ∧ s.p2 = p2

/-- The winning condition for the first player -/
def first_player_wins (g : GameSetup) : Prop :=
  ∃ s ∈ g.segments, ∃ p1 p2, p1 ∈ g.points ∧ p2 ∈ g.points ∧
    s.p1 = p1 ∧ s.p2 = p2 ∧
    g.point_assignment p1 = g.segment_assignment s ∧
    g.point_assignment p2 = g.segment_assignment s

/-- The main theorem -/
theorem first_player_always_wins (g : GameSetup) (h : valid_game_setup g) :
  ∃ segment_assignment : Segment → ℕ,
    ∀ point_assignment : Point → ℕ,
      first_player_wins {g with
        segment_assignment := segment_assignment,
        point_assignment := point_assignment
      } :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l1076_107657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_in_ellipse_l1076_107674

/-- The maximum area of a right triangle inscribed in an ellipse -/
noncomputable def max_area (a b : ℝ) : ℝ :=
  if a ≥ (Real.sqrt 2 + 1) * b
  then a * b / Real.exp 2
  else 4 * a^4 * b^2 / (a^2 + b^2)^2

theorem right_triangle_in_ellipse
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →
    ∃ (A B C : ℝ × ℝ),
      A = (0, b) ∧
      (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
      (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
      (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) ∧
  (b = 1 → (max_area a b = 27 / 8 → a = 3)) ∧
  (∀ S : ℝ, S ≤ max_area a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_in_ellipse_l1076_107674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l1076_107667

/-- The point symmetric to the origin with respect to the line x-2y+1=0 -/
noncomputable def symmetric_point : ℝ × ℝ := (-2/5, 4/5)

/-- The line with respect to which the point is symmetric -/
def line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating that the symmetric_point is indeed symmetric to the origin with respect to the given line -/
theorem symmetric_point_is_correct :
  let (x, y) := symmetric_point
  line (x/2) (y/2) ∧ 
  (y/x) * (1/2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_is_correct_l1076_107667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_in_cm2_l1076_107692

-- Define the length and width of the garden in meters
noncomputable def garden_length : ℚ := 5
noncomputable def garden_width : ℚ := 17/20

-- Define the conversion factor from square meters to square centimeters
def m2_to_cm2 : ℚ := 10000

-- Theorem statement
theorem garden_area_in_cm2 : 
  (garden_length * garden_width * m2_to_cm2 : ℚ) = 42500 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_in_cm2_l1076_107692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1076_107675

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -(f a x)) ∧
  (∀ x y : ℝ, x < y → f a x < f a y) :=
by
  constructor
  · -- Proof of odd function property
    intro x
    sorry
  · -- Proof of strictly increasing property
    intro x y hxy
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1076_107675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1076_107696

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := 
  if l.a ≠ 0 then -l.c / l.a else 0

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := 
  if l.b ≠ 0 then -l.c / l.b else 0

/-- Check if a point is on a line -/
def on_line (l : Line) (x y : ℝ) : Prop := l.a * x + l.b * y + l.c = 0

theorem line_equation (l : Line) : 
  (on_line l (-5) 2) ∧ 
  (x_intercept l = 2 * y_intercept l) → 
  ((l.a = 2 ∧ l.b = 5 ∧ l.c = 0) ∨ 
   (l.a = 1 ∧ l.b = 2 ∧ l.c = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1076_107696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_spherical_coordinates_l1076_107698

/-- The radius of the circle formed by points with spherical coordinates (3, θ, π/3) -/
noncomputable def circle_radius : ℝ := (3 * Real.sqrt 3) / 2

/-- Theorem stating that the radius of the circle formed by points with 
    spherical coordinates (3, θ, π/3) is equal to (3 * √3) / 2 -/
theorem circle_radius_spherical_coordinates :
  ∀ θ : ℝ, 
  let r := Real.sqrt ((3 * Real.sin (π/3) * Real.cos θ)^2 + (3 * Real.sin (π/3) * Real.sin θ)^2)
  r = circle_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_spherical_coordinates_l1076_107698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extinction_probability_correct_l1076_107628

/-- A branching process with the given probabilities -/
structure BranchingProcess where
  split_prob : ℚ
  die_prob : ℚ
  split_prob_nonneg : 0 ≤ split_prob
  die_prob_nonneg : 0 ≤ die_prob
  prob_sum_one : split_prob + die_prob = 1

/-- The probability of eventual extinction for the branching process -/
def extinction_probability (bp : BranchingProcess) : ℚ :=
  2/3

/-- Theorem stating that the extinction probability is correct -/
theorem extinction_probability_correct (bp : BranchingProcess) 
  (h_split : bp.split_prob = 6/10) (h_die : bp.die_prob = 4/10) : 
  extinction_probability bp = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extinction_probability_correct_l1076_107628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_alignment_time_l1076_107602

/-- Represents the state of the classroom with two political parties --/
structure ClassroomState where
  party1 : Nat
  party2 : Nat
  deriving Repr

/-- Represents a transition between classroom states --/
structure Transition where
  fromState : ClassroomState
  toState : ClassroomState
  probability : Rat
  deriving Repr

/-- Calculates the expected time for a transition --/
def expectedTime (t : Transition) : Rat :=
  1 / t.probability

/-- The initial state of the classroom --/
def initialState : ClassroomState :=
  { party1 := 6, party2 := 6 }

/-- The final state of the classroom --/
def finalState : ClassroomState :=
  { party1 := 12, party2 := 0 }

/-- The first transition from 6-6 to 3-9 --/
def transition1 : Transition :=
  { fromState := initialState, toState := { party1 := 3, party2 := 9 }, probability := 18/77 }

/-- The second transition from 3-9 to 0-12 --/
def transition2 : Transition :=
  { fromState := { party1 := 3, party2 := 9 }, toState := finalState, probability := 27/55 }

/-- Theorem stating the expected time for all students to align --/
theorem expected_alignment_time :
  expectedTime transition1 + expectedTime transition2 = 341/54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_alignment_time_l1076_107602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiber_length_related_to_soil_environment_l1076_107633

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![25, 35],
    ![15,  5]]

-- Define the total sample size
def n : ℕ := 80

-- Define the critical value for α = 0.01
noncomputable def critical_value : ℝ := 6.635

-- Define the Chi-Square test statistic function
noncomputable def chi_square (table : Matrix (Fin 2) (Fin 2) ℕ) : ℝ :=
  let a := (table 0 0 : ℝ)
  let b := (table 0 1 : ℝ)
  let c := (table 1 0 : ℝ)
  let d := (table 1 1 : ℝ)
  (n : ℝ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- Theorem statement
theorem fiber_length_related_to_soil_environment :
  chi_square contingency_table > critical_value := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiber_length_related_to_soil_environment_l1076_107633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decorative_window_properties_l1076_107681

/-- Represents a decorative window with a rectangle and two semicircles -/
structure DecorativeWindow where
  width : ℝ
  length : ℝ

/-- Properties of the decorative window -/
def window_properties (w : DecorativeWindow) : Prop :=
  w.length = 3 * w.width ∧ w.width = 40

/-- The ratio of the rectangle area to the semicircles area -/
noncomputable def area_ratio (w : DecorativeWindow) : ℝ :=
  (w.length * w.width) / (Real.pi * (w.width / 2)^2)

/-- The total area of the window -/
noncomputable def total_area (w : DecorativeWindow) : ℝ :=
  w.length * w.width + Real.pi * (w.width / 2)^2

theorem decorative_window_properties (w : DecorativeWindow) 
  (h : window_properties w) : 
  area_ratio w = 12 / Real.pi ∧ total_area w = 4800 + 400 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decorative_window_properties_l1076_107681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1076_107640

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = 40 →
  b = 20 * Real.sqrt 2 →
  A = 45 * π / 180 →
  Real.sin B = b * Real.sin A / a →
  0 < B →
  B < A →
  B = 30 * π / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l1076_107640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_f_l1076_107672

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((x+1)/(x-1)) + Real.log (x-1) + Real.log (a-x)

/-- The domain of f(x) -/
def f_domain (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

/-- Theorem stating that no such a exists that makes f(x) symmetric -/
theorem no_symmetric_f :
  ¬ ∃ (a : ℝ), a > 1 ∧
    ∃ (c : ℝ), ∀ (d : ℝ),
      (c - d) ∈ f_domain a → (c + d) ∈ f_domain a →
      f a (c - d) = f a (c + d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_symmetric_f_l1076_107672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l1076_107697

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l: x - y + 1 = 0
def line_l (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 + 1 = 0

-- Define the circle C
def circle_C : Circle :=
  { center := (-3, -2),
    radius := 5 }

-- Define the condition that the circle passes through two points
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the line l₁
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition that a line passes through a point
def line_passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define the chord length
noncomputable def chord_length (c : Circle) (l : Line) : ℝ :=
  2 * c.radius * Real.sqrt (1 - ((l.a * c.center.1 + l.b * c.center.2 + l.c) / (c.radius * Real.sqrt (l.a^2 + l.b^2)))^2)

theorem circle_and_line_theorem :
  line_l circle_C.center ∧
  passes_through circle_C (1, 1) ∧
  passes_through circle_C (2, -2) ∧
  (∃ l : Line, line_passes_through l (1, 1) ∧ chord_length circle_C l = 6) →
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ passes_through circle_C (x, y)) ∧
  (∃ l : Line, (l.a = 7 ∧ l.b = 24 ∧ l.c = -31) ∨ (l.a = 1 ∧ l.b = 0 ∧ l.c = -1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l1076_107697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_problem_l1076_107661

-- Define the types for different colored balls
structure Ball where
  weight : ℕ

def Green := Ball
def Blue := Ball
def Yellow := Ball
def White := Ball

-- Define the balance relationships
def green_blue_balance (g : Green) (b : Blue) : Prop := 4 * g.weight = 8 * b.weight
def yellow_blue_balance (y : Yellow) (b : Blue) : Prop := 3 * y.weight = 6 * b.weight
def blue_white_balance (b : Blue) (w : White) : Prop := 2 * b.weight = 3 * w.weight

-- Theorem statement
theorem balance_problem (g : Green) (b : Blue) (y : Yellow) (w : White) 
  (h1 : green_blue_balance g b)
  (h2 : yellow_blue_balance y b)
  (h3 : blue_white_balance b w) :
  3 * g.weight + 4 * y.weight + 3 * w.weight = 16 * b.weight :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_problem_l1076_107661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_decreasing_g_l1076_107677

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define that f is increasing on ℝ
def increasing_on_real (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the composition function g(x) = 2^(-f(x))
noncomputable def g (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2^(-f x)

-- Define that g is decreasing on ℝ
def decreasing_on_real (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x > g y

-- State the theorem
theorem increasing_f_implies_decreasing_g (f : ℝ → ℝ) :
  increasing_on_real f → decreasing_on_real (g f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_decreasing_g_l1076_107677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_zeros_l1076_107634

/-- The function f(x) = exp(x) - mx -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - m * x

theorem f_inequality_and_zeros (m : ℝ) :
  (∀ x > 0, (x - 2) * f m x + m * x^2 + 2 > 0) ↔ m ≥ 1/2 ∧
  ∀ x₁ x₂ : ℝ, f m x₁ = 0 → f m x₂ = 0 → x₁ + x₂ > 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_and_zeros_l1076_107634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_purchase_popularity_local_group_purchase_impracticality_l1076_107678

/-- Represents the benefits of group purchasing -/
structure GroupPurchaseBenefits where
  cost_savings : ℝ
  information_quality : ℝ

/-- Represents the challenges of local group purchasing -/
structure LocalGroupPurchaseChallenges where
  transaction_costs : ℝ
  coordination_effort : ℝ
  proximity_to_stores : ℝ

/-- Determines if group purchasing is beneficial -/
def is_group_purchase_beneficial (benefits : GroupPurchaseBenefits) (threshold : ℝ) : Prop :=
  benefits.cost_savings + benefits.information_quality > threshold

/-- Determines if local group purchasing is practical -/
def is_local_group_purchase_practical (challenges : LocalGroupPurchaseChallenges) (threshold : ℝ) : Prop :=
  challenges.transaction_costs + challenges.coordination_effort < threshold * challenges.proximity_to_stores

theorem group_purchase_popularity (benefits : GroupPurchaseBenefits) (threshold : ℝ) :
  is_group_purchase_beneficial benefits threshold → true :=
by
  intro h
  trivial

theorem local_group_purchase_impracticality (challenges : LocalGroupPurchaseChallenges) (threshold : ℝ) :
  ¬is_local_group_purchase_practical challenges threshold → true :=
by
  intro h
  trivial

#check group_purchase_popularity
#check local_group_purchase_impracticality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_purchase_popularity_local_group_purchase_impracticality_l1076_107678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_A_B_l1076_107679

def setA : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def setB : Set ℂ := {z : ℂ | z^4 - 8*z^3 + 18*z^2 - 27*z + 27 = 0}

noncomputable def distance (z w : ℂ) : ℝ := Complex.abs (z - w)

theorem greatest_distance_A_B :
  ∃ (a : ℂ) (b : ℂ), a ∈ setA ∧ b ∈ setB ∧
    (∀ (x : ℂ) (y : ℂ), x ∈ setA → y ∈ setB → distance x y ≤ distance a b) ∧
    distance a b = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_distance_A_B_l1076_107679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_l1076_107622

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  /-- Nandan's investment amount -/
  nandan_investment : ℝ
  /-- Duration of Nandan's investment -/
  nandan_time : ℝ
  /-- Krishan's investment amount -/
  krishan_investment : ℝ
  /-- Duration of Krishan's investment -/
  krishan_time : ℝ
  /-- Nandan's gain from the investment -/
  nandan_gain : ℝ
  /-- Total gain from both investments -/
  total_gain : ℝ

/-- 
The theorem statement based on the given problem:
Given the conditions of the business investment scenario,
proves that the ratio of Krishan's investment to Nandan's investment is 6:1
-/
theorem investment_ratio (b : BusinessInvestment) 
  (h1 : b.krishan_time = 2 * b.nandan_time)
  (h2 : b.nandan_gain = 6000)
  (h3 : b.total_gain = 78000)
  (h4 : b.nandan_gain = b.nandan_investment * b.nandan_time)
  (h5 : b.total_gain = b.nandan_gain + b.krishan_investment * b.krishan_time) :
  b.krishan_investment / b.nandan_investment = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_l1076_107622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_to_side_ratio_of_adjacent_squares_l1076_107630

/-- Given two squares sharing a common vertex, the ratio of the diagonal of one square to the side of the other square is √2 : 1. -/
theorem diagonal_to_side_ratio_of_adjacent_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt 2 * a) / b = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_to_side_ratio_of_adjacent_squares_l1076_107630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_x_l1076_107600

open Real MeasureTheory

theorem integral_exp_minus_x : ∫ x in Set.Icc 0 1, Real.exp (-x) = 1 - Real.exp (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_minus_x_l1076_107600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1076_107691

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ -3 ∧ x ≠ -2}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1076_107691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_squares_area_ratio_l1076_107608

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a pentagon in 2D space -/
structure Pentagon where
  vertices : List Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := s.sideLength * s.sideLength

/-- Calculates the area of a pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  sorry -- Actual calculation would go here

/-- Main theorem -/
theorem pentagon_to_squares_area_ratio :
  ∀ (squareA squareB squareC : Square) 
    (M N P Q C : Point) 
    (pentagon : Pentagon),
  squareA.sideLength = 1 →
  squareB.sideLength = 1 →
  squareC.sideLength = 1 →
  -- M is midpoint of side AB of squareA
  -- N is midpoint of side GH of squareB
  -- P is midpoint of side KL of squareC
  -- Q and C are vertices of squareA
  pentagon.vertices = [M, N, P, Q, C] →
  (pentagonArea pentagon) / (squareArea squareA + squareArea squareB + squareArea squareC) = 7/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_squares_area_ratio_l1076_107608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_distance_less_than_2016_l1076_107668

def b (n : ℕ) : ℚ :=
  match n % 4 with
  | 1 => 2
  | 2 => -3
  | 3 => -1/2
  | _ => 1/3

def c (n : ℕ) : ℚ :=
  match n % 4 with
  | 1 => 3
  | 2 => -2
  | 3 => -1/3
  | _ => 1/2

def sequence_distance (m : ℕ) : ℚ :=
  Finset.sum (Finset.range m) (fun i => abs (b (i + 1) - c (i + 1)))

theorem max_m_for_distance_less_than_2016 :
  ∃ m : ℕ, sequence_distance m < 2016 ∧ ∀ k > m, sequence_distance k ≥ 2016 :=
by
  sorry

#eval sequence_distance 3455
#eval sequence_distance 3456

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_distance_less_than_2016_l1076_107668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_intervals_l1076_107653

-- Define the real functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom fg_derivative_positive : ∀ x, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom g_zero_at_neg_two : g (-2) = 0

-- Define the solution set
def solution_set : Set ℝ := {x | f x * g x > 0}

-- State the theorem
theorem solution_set_eq_intervals : solution_set = Set.Ioo (-2) 0 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_intervals_l1076_107653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1076_107637

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/3

-- State the theorem
theorem integral_value : ∫ x in Set.Icc 0 1, f x = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l1076_107637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_problem_l1076_107652

theorem average_weight_problem (boys_avg : ℚ) (girls_avg : ℚ) :
  boys_avg = 155 →
  girls_avg = 125 →
  let boys_count : ℕ := 8
  let girls_count : ℕ := 7
  let total_count : ℕ := boys_count + girls_count
  let total_weight : ℚ := boys_avg * boys_count + girls_avg * girls_count
  total_weight / total_count = 141 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_weight_problem_l1076_107652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1076_107660

noncomputable def g (x : ℝ) : ℝ := 5 / (3 * x^4 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1076_107660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l1076_107664

/-- Represents a point on the straightened diagram --/
structure DiagramPoint where
  x : ℝ
  u : ℝ

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- Represents the line equation u = (x - a) / σ --/
noncomputable def line_equation (nd : NormalDistribution) (x : ℝ) : ℝ :=
  (x - nd.mean) / nd.std_dev

/-- The method of straightened diagrams confirms the normal distribution --/
def confirms_normal_distribution (points : List DiagramPoint) (nd : NormalDistribution) : Prop :=
  ∀ p ∈ points, abs (p.u - line_equation nd p.x) < 0.1 -- Using a small threshold instead of ≈

theorem normal_distribution_properties 
  (points : List DiagramPoint) 
  (nd : NormalDistribution) 
  (h : confirms_normal_distribution points nd) :
  ∃ x_L x_N : ℝ,
    (x_L = nd.mean) ∧ 
    (x_L - x_N = nd.std_dev) ∧
    (line_equation nd x_L = 0) ∧
    (line_equation nd x_N = -1) := by
  sorry

#check normal_distribution_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_properties_l1076_107664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1076_107606

/-- The function f(x) = x / (x + 1) -/
noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

/-- The sequence a_n defined recursively -/
noncomputable def a (n : ℕ) (a₀ : ℝ) : ℝ :=
  match n with
  | 0 => a₀
  | m + 1 => f (a m a₀)

/-- The theorem stating the closed form of the sequence -/
theorem a_formula (n : ℕ) (a₀ : ℝ) (h : a₀ > 0) :
  a n a₀ = a₀ / (1 + n * a₀) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1076_107606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_placement_exists_l1076_107656

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a placement of squares on a cube -/
structure Placement where
  cube : Cube
  squares : List Square

/-- Checks if two squares overlap -/
def squaresOverlap (s1 s2 : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Checks if two squares share an edge or part of an edge -/
def squaresShareEdge (s1 s2 : ℝ × ℝ × ℝ) : Prop :=
  sorry

/-- Checks if a placement is valid according to the given conditions -/
def isValidPlacement (p : Placement) : Prop :=
  p.cube.edgeLength = 2 ∧
  p.squares.length = 10 ∧
  ∀ s ∈ p.squares, s.sideLength = 1 ∧
  ∃ (arrangement : List (ℝ × ℝ × ℝ)),
    arrangement.length = 10 ∧
    ∀ (i j : Nat), i < j → i < arrangement.length → j < arrangement.length →
      ¬ squaresOverlap arrangement[i]! arrangement[j]! ∧
      ¬ squaresShareEdge arrangement[i]! arrangement[j]!

/-- Theorem stating that a valid placement exists -/
theorem valid_placement_exists : ∃ (p : Placement), isValidPlacement p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_placement_exists_l1076_107656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1076_107670

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt (4 - x) + 4 * Real.sqrt (x - 3)

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), M = 5 ∧ ∀ x, 3 ≤ x ∧ x ≤ 4 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1076_107670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sample_number_for_given_conditions_l1076_107604

/-- Represents a systematic sampling process -/
structure SystematicSampling where
  total_workers : ℕ
  sample_percentage : ℚ
  start_number : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSampling) : ℕ :=
  let sample_size := (s.total_workers * s.sample_percentage.num / s.sample_percentage.den).toNat
  let interval := s.total_workers / sample_size
  s.start_number + (sample_size - 1) * interval

/-- Theorem stating the largest number in the systematic sample -/
theorem largest_sample_number_for_given_conditions :
  let s : SystematicSampling :=
    { total_workers := 620
    , sample_percentage := 1/10
    , start_number := 7 }
  largest_sample_number s = 617 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sample_number_for_given_conditions_l1076_107604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gulliver_kefir_bottles_l1076_107684

/-- Represents the process of buying and drinking kefir -/
def kefirProcess (initialMoney : ℕ) (initialKefirCost : ℕ) (initialBottleCost : ℕ) : ℕ :=
  let firstPurchase := initialMoney / initialKefirCost
  let geometricSum := firstPurchase * 2
  geometricSum

/-- Theorem stating the total number of kefir bottles Gulliver drinks -/
theorem gulliver_kefir_bottles :
  kefirProcess 7000000 7 1 = 2000000 := by
  rfl

#eval kefirProcess 7000000 7 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gulliver_kefir_bottles_l1076_107684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_first_term_is_integer_last_integer_is_first_term_l1076_107621

def sequenceQ (n : ℕ) : ℚ :=
  (1024000 : ℚ) / (3 ^ n)

def isInt (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∀ n : ℕ, n > 0 → ¬ (isInt (sequenceQ n)) :=
by sorry

theorem first_term_is_integer :
  isInt (sequenceQ 0) :=
by sorry

theorem last_integer_is_first_term :
  ∀ n : ℕ, isInt (sequenceQ n) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_integer_in_sequence_first_term_is_integer_last_integer_is_first_term_l1076_107621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_problem_l1076_107638

/-- Calculates the time taken to swim against the current given the swimmer's speed,
    current speed, and time taken to swim with the current. -/
noncomputable def time_against_current (swimmer_speed : ℝ) (current_speed : ℝ) (time_with_current : ℝ) : ℝ :=
  let speed_with_current := swimmer_speed + current_speed
  let distance := speed_with_current * time_with_current
  let speed_against_current := swimmer_speed - current_speed
  distance / speed_against_current

/-- Theorem stating that given the specific conditions, the time taken to swim against
    the current is 10.5 hours. -/
theorem swimmer_problem (swimmer_speed : ℝ) (current_speed : ℝ) (time_with_current : ℝ)
    (h1 : swimmer_speed = 4)
    (h2 : current_speed = 2)
    (h3 : time_with_current = 3.5) :
    time_against_current swimmer_speed current_speed time_with_current = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimmer_problem_l1076_107638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_length_l1076_107629

theorem fly_path_length (r : ℝ) (angle : ℝ) (third_side : ℝ) : 
  r = 75 → 
  angle = 120 * π / 180 → 
  third_side = 95 → 
  (2 * r) + (2 * r^2 * (1 - Real.cos angle)).sqrt + third_side = 361 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_length_l1076_107629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BCD_l1076_107651

-- Define the points
variable (A' B' C' D' : ℝ × ℝ)

-- Define the conditions
def aligned (A' B' C' D' : ℝ × ℝ) : Prop := sorry

-- Define the areas and distances
def area_ABC' : ℝ := 36
def dist_AC' : ℝ := 12
def dist_CD' : ℝ := 30

-- Define the area of a triangle function
def area_triangle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem area_BCD' (h_aligned : aligned A' B' C' D')
                  (h_area_ABC' : area_ABC' = 36)
                  (h_dist_AC' : dist_AC' = 12)
                  (h_dist_CD' : dist_CD' = 30) :
  area_triangle B' C' D' = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BCD_l1076_107651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_tetrahedron_volume_ratio_l1076_107613

/-- The ratio of the volume of a sphere inscribed in a regular tetrahedron
    to the volume of the tetrahedron. -/
noncomputable def sphere_tetrahedron_volume_ratio : ℝ := Real.pi * Real.sqrt 3 / 27

/-- Theorem stating that the ratio of the volume of a sphere inscribed in a regular tetrahedron
    to the volume of the tetrahedron is π√3 / 27. -/
theorem sphere_inscribed_in_tetrahedron_volume_ratio :
  ∀ (s r : ℝ),
  s > 0 →  -- side length of tetrahedron is positive
  r > 0 →  -- radius of sphere is positive
  r = s * Real.sqrt 6 / 12 →  -- relation between radius and side length
  (4 / 3 * Real.pi * r^3) / (s^3 * Real.sqrt 2 / 12) = sphere_tetrahedron_volume_ratio :=
by
  sorry

#check sphere_tetrahedron_volume_ratio
#check sphere_inscribed_in_tetrahedron_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_inscribed_in_tetrahedron_volume_ratio_l1076_107613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1076_107624

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / Real.sqrt (x^2 - x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem stating that the domain of f is correct
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1076_107624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_half_beta_l1076_107632

theorem cos_alpha_plus_half_beta (α β : ℝ) 
  (h1 : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h2 : Real.cos (π/4 + α) = 1/3)
  (h3 : Real.cos (π/4 - β/2) = Real.sqrt 3 / 3) :
  Real.cos (α + β/2) = 5 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_half_beta_l1076_107632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foot_height_estimate_correct_l1076_107627

/-- Represents the linear regression problem for foot length and height --/
structure FootHeightRegression where
  n : ℕ -- number of students
  x_sum : ℚ -- sum of foot lengths
  y_sum : ℚ -- sum of heights
  b_hat : ℚ -- estimated slope

/-- Calculates the estimated height for a given foot length --/
def estimate_height (r : FootHeightRegression) (x : ℚ) : ℚ :=
  let x_mean := r.x_sum / r.n
  let y_mean := r.y_sum / r.n
  let a_hat := y_mean - r.b_hat * x_mean
  r.b_hat * x + a_hat

/-- Theorem stating that the estimated height for a foot length of 24 cm is 166 cm --/
theorem foot_height_estimate_correct (r : FootHeightRegression) : 
  r.n = 10 → r.x_sum = 225 → r.y_sum = 1600 → r.b_hat = 4 → estimate_height r 24 = 166 := by
  sorry

/-- Example calculation --/
def example_regression : FootHeightRegression := {
  n := 10,
  x_sum := 225,
  y_sum := 1600,
  b_hat := 4
}

#eval estimate_height example_regression 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foot_height_estimate_correct_l1076_107627
