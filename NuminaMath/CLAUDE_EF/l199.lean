import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_recipe_fills_l199_19973

/-- Represents the number of times a measuring cup must be filled -/
def fillCount (amount : ℚ) (cupSize : ℚ) : ℕ :=
  (Int.ceil (amount / cupSize)).toNat

/-- The problem statement -/
theorem cake_recipe_fills :
  let flourNeeded : ℚ := 15 / 4
  let flourCupSize : ℚ := 1 / 3
  let milkNeeded : ℚ := 3 / 2
  let milkCupSize : ℚ := 1 / 2
  (fillCount flourNeeded flourCupSize) + (fillCount milkNeeded milkCupSize) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_recipe_fills_l199_19973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_count_l199_19981

-- Define the clock and its properties
structure Clock where
  hour_hand : ℝ
  minute_hand : ℝ
  second_hand : ℝ

-- Define the constant speeds of the hands
noncomputable def hour_speed : ℝ := 1/720    -- 360° / (12 * 60) minutes
noncomputable def minute_speed : ℝ := 1/60   -- 360° / 60 minutes
noncomputable def second_speed : ℝ := 6      -- 360° / 60 seconds

-- Define the initial position at 3:00
def initial_clock : Clock :=
  { hour_hand := 90,   -- 90°
    minute_hand := 0,  -- 0°
    second_hand := 0   -- 0°
  }

-- Define a function to check if one hand divides the angle between the other two equally
def divides_equally (c : Clock) : Prop := sorry

-- Define a function to update the clock state after t seconds
noncomputable def update_clock (c : Clock) (t : ℝ) : Clock := sorry

-- Theorem statement
theorem equal_division_count :
  ∃ (count : ℕ), count = 4 ∧
  (∀ t ∈ Set.Icc 0 60, 
    divides_equally (update_clock initial_clock t)) ↔ 
    (∃ (times : Finset ℝ), times.card = count ∧ 
      ∀ τ ∈ times, τ ∈ Set.Icc 0 60 ∧ 
      divides_equally (update_clock initial_clock τ)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_count_l199_19981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l199_19940

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * Real.sin x - Real.sqrt 3)

-- Define the domain set
def domain : Set ℝ := {x | ∃ k : ℤ, Real.pi/3 + 2*k*Real.pi ≤ x ∧ x ≤ 2*Real.pi/3 + 2*k*Real.pi}

-- Theorem stating that the domain of f is equal to the defined domain set
theorem f_domain : {x : ℝ | f x ≥ 0} = domain := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l199_19940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_fourth_power_l199_19983

theorem min_value_sin_cos_fourth_power :
  ∀ x : ℝ, Real.sin x ^ 4 + (3/2) * Real.cos x ^ 4 ≥ 3/5 ∧
  ∃ y : ℝ, Real.sin y ^ 4 + (3/2) * Real.cos y ^ 4 = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_cos_fourth_power_l199_19983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l199_19910

/-- Calculates the harmonic mean of three positive real numbers -/
noncomputable def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1/a + 1/b + 1/c)

theorem triathlon_average_speed :
  let swim_speed : ℝ := 2
  let bike_speed : ℝ := 25
  let run_speed : ℝ := 12
  let average_speed := harmonic_mean swim_speed bike_speed run_speed
  ∀ ε > 0, |average_speed - 4.8| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l199_19910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_distribution_events_l199_19920

structure Person : Type :=
  (name : String)

structure Card : Type :=
  (color : String)

def distribute_cards (people : List Person) (cards : List Card) : Prop :=
  people.length = 4 ∧ cards.length = 4 ∧ ∀ p : Person, p ∈ people → ∃! c : Card, c ∈ cards

def event_A_red (distribution : List Person → List Card → Prop) : Prop :=
  ∃ (people : List Person) (cards : List Card),
    distribution people cards ∧
    ∃ (a : Person) (red : Card),
      a ∈ people ∧ a.name = "A" ∧ red ∈ cards ∧ red.color = "red"

def event_D_red (distribution : List Person → List Card → Prop) : Prop :=
  ∃ (people : List Person) (cards : List Card),
    distribution people cards ∧
    ∃ (d : Person) (red : Card),
      d ∈ people ∧ d.name = "D" ∧ red ∈ cards ∧ red.color = "red"

theorem card_distribution_events :
  let distribution := distribute_cards
  (¬(event_A_red distribution ∧ event_D_red distribution)) ∧
  (¬(event_A_red distribution ↔ ¬event_D_red distribution)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_distribution_events_l199_19920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_3y_l199_19922

/-- The width of the rectangular region -/
def width : ℝ := 2010

/-- The height of the rectangular region -/
def rectangle_height : ℝ := 2011

/-- The rectangular region from which points are randomly chosen -/
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ width ∧ 0 ≤ p.2 ∧ p.2 ≤ rectangle_height}

/-- The subset of the rectangle where x > 3y -/
def subset : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 > 3 * p.2}

/-- The probability that a randomly chosen point (x,y) from the rectangle satisfies x > 3y -/
theorem probability_x_gt_3y : 
  (MeasureTheory.volume subset) / (MeasureTheory.volume rectangle) = 670 / 2011 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_gt_3y_l199_19922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_july_production_capacity_l199_19955

-- Define the production function
noncomputable def production_capacity (a b : ℝ) (x : ℕ) : ℝ := a * (1/2)^(x - 3) + b

-- State the theorem
theorem july_production_capacity :
  ∀ a b : ℝ,
  (production_capacity a b 4 = 1) →
  (production_capacity a b 5 = 3/2) →
  (production_capacity a b 7 = 15/8) :=
by
  -- Introduce variables and hypotheses
  intro a b h1 h2
  -- Expand the definition of production_capacity
  simp [production_capacity] at *
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_july_production_capacity_l199_19955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_for_rectangles_l199_19960

noncomputable section

-- Define the rectangles
def rectangle1_width : ℝ := 3
def rectangle1_height : ℝ := 4
def rectangle2_width : ℝ := 4
def rectangle2_height : ℝ := 5

-- Define the diagonal lengths
noncomputable def diagonal1 : ℝ := Real.sqrt (rectangle1_width ^ 2 + rectangle1_height ^ 2)
noncomputable def diagonal2 : ℝ := Real.sqrt (rectangle2_width ^ 2 + rectangle2_height ^ 2)

-- Define the minimum square side length
noncomputable def min_square_side : ℝ := max diagonal1 diagonal2

-- Theorem statement
theorem min_square_area_for_rectangles :
  min_square_side ^ 2 = 41 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_area_for_rectangles_l199_19960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_dot_product_close_functions_example_l199_19978

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

noncomputable def angle_between (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((dot_product v1 v2) / (Real.sqrt (dot_product v1 v1) * Real.sqrt (dot_product v2 v2)))

def close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

theorem acute_angle_dot_product (v1 v2 : ℝ × ℝ) :
  angle_between v1 v2 < π / 2 ↔ dot_product v1 v2 > 0 := by sorry

theorem close_functions_example :
  close_functions (fun x => x^2 - 3*x + 4) (fun x => 2*x - 3) 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_dot_product_close_functions_example_l199_19978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_with_equal_intercepts_l199_19914

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The circle x^2 + (y-2)^2 = 2 -/
def is_on_circle (x y : ℝ) : Prop := x^2 + (y-2)^2 = 2

/-- Predicate to check if a line is tangent to the circle -/
def is_tangent_to_circle (l : Line) : Prop :=
  ∃ (x y : ℝ), is_on_circle x y ∧ y = l.slope * x + l.intercept ∧
    ∀ (x' y' : ℝ), y' = l.slope * x' + l.intercept → (x' = x ∧ y' = y) ∨ ¬(is_on_circle x' y')

/-- Predicate to check if a line has equal intercepts on both axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.intercept ≠ 0 → l.intercept = -l.intercept / l.slope

/-- The main theorem stating that there are exactly 4 lines satisfying the conditions -/
theorem four_tangent_lines_with_equal_intercepts :
  ∃! (lines : Finset Line), lines.card = 4 ∧
    ∀ l ∈ lines, is_tangent_to_circle l ∧ has_equal_intercepts l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangent_lines_with_equal_intercepts_l199_19914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_proof_l199_19925

/-- The official exchange rate in euros per dollar -/
noncomputable def official_rate : ℚ := 5

/-- The amount of euros Willie has -/
def willie_euros : ℚ := 70

/-- The amount of dollars Willie receives -/
def willie_dollars : ℚ := 10

/-- The airport exchange rate as a fraction of the official rate -/
def airport_rate_fraction : ℚ := 5 / 7

theorem exchange_rate_proof :
  willie_euros = willie_dollars * (1 / airport_rate_fraction) * official_rate →
  official_rate = 5 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_rate_proof_l199_19925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l199_19915

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_3 : (a 1) + (a 2) + (a 3) = 0
  sum_5 : (a 1) + (a 2) + (a 3) + (a 4) + (a 5) = -5

/-- The general term of the arithmetic sequence -/
def generalTerm (n : ℕ) : ℤ := 2 - n

/-- The sum of every third term starting from the first -/
def sumEveryThird (n : ℕ) : ℚ :=
  (n + 1 : ℚ) * (2 - 3 * n) / 2

/-- The main theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = generalTerm n) ∧
  (∀ n : ℕ, (Finset.sum (Finset.range (n + 1)) (fun i => seq.a (3 * i + 1))) = sumEveryThird n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l199_19915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_divisibility_l199_19994

theorem pigeonhole_divisibility (S : Finset ℤ) (h : S.card = 2023) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_divisibility_l199_19994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_tickets_correct_l199_19943

/-- Given a group of t people where the total cost of tickets is $190,
    children's tickets cost $5, and adult tickets cost $9,
    the number of children's tickets C is (9t - 190) / 4 -/
def childrens_tickets (t : ℕ) : ℕ :=
  -- Define the total cost
  let total_cost : ℕ := 190
  -- Define the cost of children's and adult's tickets
  let child_ticket_cost : ℕ := 5
  let adult_ticket_cost : ℕ := 9
  -- Define the number of children's tickets as a function of t
  let C : ℕ := (9 * t - 190) / 4
  -- Return the number of children's tickets
  C

theorem childrens_tickets_correct (t : ℕ) :
  let C := childrens_tickets t
  let A := t - C
  A * 9 + C * 5 = 190 := by
  sorry  -- The proof is omitted for now

#eval childrens_tickets 30  -- Should evaluate to 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_childrens_tickets_correct_l199_19943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l199_19987

-- Define the function f(x) = log_a(a - x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a - x) / Real.log a

-- State the theorem
theorem domain_of_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {x : ℝ | ∃ y, f a x = y} = {x : ℝ | x < a} := by
  sorry

-- Note: {x : ℝ | ∃ y, f a x = y} represents the domain of the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_log_function_l199_19987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_na_l199_19921

def a : ℕ → ℚ
  | 0 => 0  -- Added case for 0
  | 1 => -1
  | 2 => -1/2
  | (n+3) => (4 * a (n+2) - 2 * a (n+1) + 3) / 2

theorem min_value_na : 
  (∀ n : ℕ, n * a n ≥ -5/4) ∧ ∃ m : ℕ, m * a m = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_na_l199_19921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_two_iff_x_eq_neg_ten_l199_19997

/-- The function f(x) = (x - 6) / (x + 2) -/
noncomputable def f (x : ℝ) : ℝ := (x - 6) / (x + 2)

/-- Theorem stating that f(x) = 2 if and only if x = -10 -/
theorem f_eq_two_iff_x_eq_neg_ten : ∀ x : ℝ, f x = 2 ↔ x = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_two_iff_x_eq_neg_ten_l199_19997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lab_mixture_concentration_l199_19927

/-- Represents an acid solution with a volume and concentration -/
structure AcidSolution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the total volume of acid in a solution -/
noncomputable def acidVolume (solution : AcidSolution) : ℝ :=
  solution.volume * solution.concentration

/-- Calculates the final concentration of a mixture of acid solutions -/
noncomputable def finalConcentration (solutions : List AcidSolution) : ℝ :=
  let totalAcid := (solutions.map acidVolume).sum
  let totalVolume := (solutions.map (·.volume)).sum
  totalAcid / totalVolume

/-- The theorem stating that the final concentration of the mixed solutions is 48% -/
theorem lab_mixture_concentration :
  let solutions := [
    { volume := 1, concentration := 0.20 },
    { volume := 2, concentration := 0.40 },
    { volume := 3, concentration := 0.60 },
    { volume := 4, concentration := 0.50 }
  ]
  finalConcentration solutions = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lab_mixture_concentration_l199_19927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_complex_condition_l199_19928

/-- A complex number is an "ideal complex number" if its real part and imaginary part are additive inverses of each other. -/
def is_ideal_complex (z : ℂ) : Prop :=
  z.re = -z.im

/-- Given complex number z expressed as a fraction plus a multiple of i. -/
noncomputable def z (a b : ℝ) : ℂ :=
  a / (1 - 2*Complex.I) + b*Complex.I

theorem ideal_complex_condition (a b : ℝ) :
  is_ideal_complex (z a b) → 3*a + 5*b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_complex_condition_l199_19928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l199_19956

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2^(x-4) + 3
noncomputable def g (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem fixed_point_of_f :
  (g 0 = 1) → (f 4 = 4) :=
by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l199_19956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_A_is_half_l199_19968

/-- Represents the percentage of type A trees -/
noncomputable def percentage_A : ℝ := by sorry

/-- Total number of trees -/
def total_trees : ℕ := 10

/-- Number of oranges produced by tree A per month -/
def oranges_A : ℕ := 10

/-- Number of oranges produced by tree B per month -/
def oranges_B : ℕ := 15

/-- Percentage of good oranges from tree A -/
noncomputable def good_percentage_A : ℝ := 0.6

/-- Percentage of good oranges from tree B -/
noncomputable def good_percentage_B : ℝ := 1/3

/-- Total number of good oranges produced per month -/
def total_good_oranges : ℕ := 55

/-- Theorem stating that the percentage of type A trees is 50% -/
theorem percentage_A_is_half : percentage_A = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_A_is_half_l199_19968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l199_19990

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := (2^x - 2)^2 + (2^(-x) + 2)^2

/-- The transformed function -/
def g (t m : ℝ) : ℝ := t^2 - 4*t + m

/-- The transformation function -/
noncomputable def φ (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem transformation_theorem :
  ∃ m : ℝ, ∀ x : ℝ, f x = g (φ x) m :=
by
  use 10  -- We use 10 as the value for m
  intro x
  -- Expand the definitions and simplify
  simp [f, g, φ]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_theorem_l199_19990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dad_drive_time_approx_l199_19993

/-- The time it takes Jake's dad to drive to the water park -/
noncomputable def dad_drive_time (jake_bike_speed jake_bike_time dad_speed1 dad_speed2 : ℝ) : ℝ :=
  let distance := jake_bike_speed * jake_bike_time
  let time1 := (distance / 2) / dad_speed1
  let time2 := (distance / 2) / dad_speed2
  (time1 + time2) * 60

/-- Theorem stating the approximate time it takes Jake's dad to drive to the water park -/
theorem dad_drive_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |dad_drive_time 11 2 28 60 - 34.57| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dad_drive_time_approx_l199_19993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_theorem_l199_19950

def original_fraction : ℚ := 3/4

def new_numerator (n : ℚ) : ℚ := n * (1 + 15/100)

def new_denominator (d : ℚ) : ℚ := d * (1 - 8/100)

theorem fraction_change_theorem :
  let new_fraction := new_numerator (Rat.num original_fraction) / new_denominator (Rat.den original_fraction)
  new_fraction = 15/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_theorem_l199_19950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l199_19967

def M : Set ℝ := {x | 4 ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 16}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem intersection_M_N : M ∩ N = Set.Ico 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l199_19967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l199_19985

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add this case to handle Nat.zero
  | 1 => 1/2
  | n + 1 => sequence_a n - (sequence_a n)^2

theorem sequence_a_inequality : ∀ n : ℕ, n ≥ 1 → sequence_a n ≤ 2 * sequence_a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_inequality_l199_19985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_parallelogram_l199_19989

-- Define the points
variable (A B C D I J K L : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A

-- Define equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- Define outward and inward triangles
def is_outward_triangle (P Q R S : ℝ × ℝ) : Prop :=
  is_equilateral_triangle P Q R ∧ 
  (P.1 - Q.1) * (R.2 - Q.2) - (R.1 - Q.1) * (P.2 - Q.2) > 0

def is_inward_triangle (P Q R S : ℝ × ℝ) : Prop :=
  is_equilateral_triangle P Q R ∧ 
  (P.1 - Q.1) * (R.2 - Q.2) - (R.1 - Q.1) * (P.2 - Q.2) < 0

-- Define parallelogram
def is_parallelogram (P Q R S : ℝ × ℝ) : Prop :=
  P.1 + R.1 = Q.1 + S.1 ∧ P.2 + R.2 = Q.2 + S.2

-- State the theorem
theorem quadrilateral_to_parallelogram 
  (h1 : is_quadrilateral A B C D)
  (h2 : is_outward_triangle A B I D)
  (h3 : is_outward_triangle C D K B)
  (h4 : is_inward_triangle B C J A)
  (h5 : is_inward_triangle D A L C) :
  is_parallelogram I J K L :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_to_parallelogram_l199_19989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_3290_deg_in_fourth_quadrant_l199_19937

def angle_to_quadrant (angle : Int) : Nat :=
  match (angle % 360 + 360) % 360 with
  | n => if 0 < n && n < 90 then 1
         else if 90 ≤ n && n < 180 then 2
         else if 180 ≤ n && n < 270 then 3
         else 4

theorem negative_3290_deg_in_fourth_quadrant :
  angle_to_quadrant (-3290) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_3290_deg_in_fourth_quadrant_l199_19937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l199_19961

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property
def symmetric_about_neg_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f (-x - 2)

-- Define the monotonically decreasing property on [0, 3]
def monotone_decreasing_on_zero_three (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 3 → f (x + 2) > f (y + 2)

-- State the theorem
theorem f_order (h1 : symmetric_about_neg_two f) (h2 : monotone_decreasing_on_zero_three f) :
  f (-1) > f 2 ∧ f 2 > f 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_order_l199_19961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_6_l199_19939

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a₁ + (n - 1 : ℝ) * d)

/-- Theorem: The sum of the first 6 terms of an arithmetic sequence
    with first term 3 and common difference 2 is 48 -/
theorem arithmetic_sum_6 :
  arithmetic_sum 3 2 6 = 48 := by
  -- Unfold the definition of arithmetic_sum
  unfold arithmetic_sum
  -- Simplify the expression
  simp [Nat.cast_add, Nat.cast_one, Nat.cast_mul]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sum_6_l199_19939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l199_19964

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.log x

theorem root_difference_bound (a b : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 1) 
  (h_x : 0 < x₁ ∧ x₁ < x₂) 
  (h_root₁ : f a x₁ = b) 
  (h_root₂ : f a x₂ = b) : 
  x₂ - x₁ ≤ b * (1 / Real.log a - 1 / (1 - a)) + (a - 1) := by
  sorry

#check root_difference_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_bound_l199_19964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l199_19919

def sequenceA (n : ℕ) : ℚ := (1/2) * n^2 + (3/2) * n

theorem sequence_proof :
  (sequenceA 1 = 2) ∧
  (sequenceA 1 + sequenceA 2 = 7) ∧
  (sequenceA 1 + sequenceA 2 + sequenceA 3 = 16) ∧
  (∀ n : ℕ, sequenceA n = (1/2) * n^2 + (3/2) * n) ∧
  (sequenceA 1 + sequenceA 2 + sequenceA 3 + sequenceA 4 + sequenceA 5 = 50) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l199_19919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l199_19984

-- Define x and y as noncomputable
noncomputable def x : ℝ := Real.log 7 / Real.log 0.1
noncomputable def y : ℝ := (1/2) * Real.log 7

-- Define propositions p and q
def p : Prop := x + y < x * y
def q : Prop := x + y > 0

-- Theorem statement
theorem problem_statement : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l199_19984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_last_element_is_5049_l199_19948

/-- Represents the alternating sequence of squared numbers from 100^2 to 1^2 -/
def alternatingSquareSequence : List ℤ := sorry

/-- The sum of the alternating square sequence -/
def sequenceSum : ℤ := List.sum alternatingSquareSequence

/-- The second to last element of the alternating square sequence -/
def secondToLastElement : ℤ := sorry

theorem second_to_last_element_is_5049 
  (h1 : alternatingSquareSequence.length = 100)
  (h2 : alternatingSquareSequence.head? = some (100^2))
  (h3 : alternatingSquareSequence.getLast? = some 1)
  (h4 : sequenceSum = 5050) :
  secondToLastElement = 5049 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_last_element_is_5049_l199_19948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_line_l199_19972

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The set of points formed by the intersections of external tangent lines -/
def TangentPoints (circles : Finset Circle) : Set (ℝ × ℝ) := sorry

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

/-- Checks if a point lies on a line -/
def PointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem all_points_on_line 
  (circles : Finset Circle) 
  (h1 : circles.card = 2012)
  (h2 : ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → c1.radius ≠ c2.radius)
  (h3 : ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → 
    (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > (c1.radius + c2.radius)^2)
  (T : Set (ℝ × ℝ))
  (hT : T = TangentPoints circles)
  (hTcard : T.ncard = 2023066)
  (S : Set (ℝ × ℝ))
  (hS : S ⊆ T)
  (hScard : S.ncard = 2021056)
  (l : Line)
  (hSline : ∀ p ∈ S, PointOnLine p l) :
  ∀ p ∈ T, PointOnLine p l :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_points_on_line_l199_19972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_l199_19958

theorem cos_sum_max (x y : ℝ) (h : Real.cos x - Real.cos y = 1/4) :
  ∃ (max : ℝ), max = 31/32 ∧ Real.cos (x + y) ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_l199_19958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_approx_l199_19908

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The ratio of volumes of two cones -/
noncomputable def volume_ratio (r1 h1 r2 h2 : ℝ) : ℝ :=
  (cone_volume r1 h1) / (cone_volume r2 h2)

/-- The problem statement -/
theorem cone_volume_ratio_approx :
  let r_C : ℝ := 22.2
  let h_C : ℝ := 42.45
  let r_D : ℝ := 56.6
  let h_D : ℝ := 29.6
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
    |volume_ratio r_C h_C r_D h_D - (221/1000)| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_approx_l199_19908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l199_19949

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g x = -(g (-x)) := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l199_19949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_butter_jar_price_l199_19930

-- Define the properties of a jar
structure Jar where
  diameter : ℝ
  height : ℝ
  price : ℝ

-- Define the volume of a cylinder
noncomputable def cylinderVolume (d h : ℝ) : ℝ := Real.pi * (d / 2) ^ 2 * h

-- Define the theorem
theorem peanut_butter_jar_price 
  (jar1 jar2 : Jar)
  (h1 : jar1.diameter = 3)
  (h2 : jar1.height = 4)
  (h3 : jar1.price = 0.60)
  (h4 : jar2.diameter = 6)
  (h5 : jar2.height = 6)
  (h6 : ∀ (j1 j2 : Jar), j1.price / cylinderVolume j1.diameter j1.height = 
                         j2.price / cylinderVolume j2.diameter j2.height) :
  jar2.price = 3.60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peanut_butter_jar_price_l199_19930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l199_19945

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := (3*x - 1) / Real.sqrt (x^2 - 25)

-- Define the domain of h
def domain_h : Set ℝ := {x | x < -5 ∨ x > 5}

-- Theorem stating that domain_h is the correct domain for h
theorem h_domain : 
  ∀ x : ℝ, h x ∈ Set.univ ↔ x ∈ domain_h := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma sqrt_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ Real.sqrt x := by
  sorry

lemma domain_condition (x : ℝ) : x ∈ domain_h ↔ x^2 > 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_domain_l199_19945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_eq_four_has_four_solutions_l199_19932

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -3 then x^2 - 3
  else if x ≥ -6 then x + 4
  else -2*x

-- State the theorem
theorem g_composition_eq_four_has_four_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g (g x) = 4) ∧ S.card = 4 ∧
    (∀ y : ℝ, g (g y) = 4 → y ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_eq_four_has_four_solutions_l199_19932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_l199_19988

theorem cosine_sum_equation (x : ℝ) : 
  Real.cos x + Real.cos (2*x) + Real.cos (3*x) + Real.cos (4*x) = 0 ↔ 
  (∃ k : ℤ, x = π/2 + 2*k*π ∨ 
            x = 3*π/2 + 2*k*π ∨ 
            x = π/5 + 2*k*π/5 ∨ 
            x = 3*π/5 + 2*k*π/5 ∨ 
            x = π + 2*k*π/5 ∨ 
            x = 7*π/5 + 2*k*π/5 ∨ 
            x = 9*π/5 + 2*k*π/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equation_l199_19988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_shapes_l199_19995

/-- Represents a tetrahedron with perpendicular edges -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- The side length of the largest cube within the tetrahedron -/
noncomputable def largest_cube (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)

/-- The dimensions of the largest rectangular parallelepiped within the tetrahedron -/
noncomputable def largest_parallelepiped (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  (t.a / 3, t.b / 3, t.c / 3)

theorem tetrahedron_largest_shapes (t : Tetrahedron) :
  (largest_cube t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)) ∧
  (largest_parallelepiped t = (t.a / 3, t.b / 3, t.c / 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_largest_shapes_l199_19995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l199_19926

-- Define a full rotation
def full_rotation : ℕ := 360

-- Define the angle in question
def angle : ℕ := 1000

-- Define a function to determine the quadrant
def quadrant (θ : ℕ) : ℕ :=
  let remainder := θ % full_rotation
  if remainder > 0 && remainder ≤ 90 then 1
  else if remainder > 90 && remainder ≤ 180 then 2
  else if remainder > 180 && remainder ≤ 270 then 3
  else 4

theorem angle_in_fourth_quadrant : quadrant angle = 4 := by
  sorry

#eval quadrant angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l199_19926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_proof_l199_19938

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Theorem statement
theorem solution_set_proof (h : ∀ x : ℝ, f' x > f x) :
  {x : ℝ | Real.exp 2 * f (2 * x - 1) - Real.exp (3 * x) * f (1 - x) > 0} = {x : ℝ | x > 2/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_proof_l199_19938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_probability_l199_19946

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  passengers : Fin n → Passenger

/-- Represents a passenger -/
inductive Passenger
  | scientist
  | regular (id : ℕ)

/-- The seating rule for passengers -/
def seatingRule (n : ℕ) (b : Bus n) (p : Passenger) : Fin n → Option (Fin n) :=
  sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerInAssignedSeatProbability (n : ℕ) : ℚ :=
  1 / 2

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 -/
theorem last_passenger_probability (n : ℕ) :
  lastPassengerInAssignedSeatProbability n = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_probability_l199_19946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l199_19952

/-- The function g(x) = (ax+b)/(cx+d) where a, b, c, d are nonzero real numbers -/
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- Theorem stating that under given conditions, 15 is not in the range of g -/
theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 5 = 5)
  (h2 : g a b c d 25 = 25)
  (h3 : ∀ x, x ≠ -d/c → g a b c d (g a b c d x) = x) :
  ∀ y, y ≠ 15 → ∃ x, g a b c d x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l199_19952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l199_19986

-- Define the point on the terminal side of angle α
noncomputable def point : ℝ × ℝ := (1, -Real.sqrt 3)

-- Define the angle α
noncomputable def α : ℝ := Real.arctan (point.2 / point.1)

-- Theorem statement
theorem sin_alpha_value :
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l199_19986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fraction_value_l199_19980

-- Define α as a real number representing an internal angle of a triangle
variable (α : Real)

-- Define the condition that α is an internal angle of a triangle
variable (h_triangle : 0 < α ∧ α < Real.pi)

-- Define the given condition
variable (h_sum : Real.sin α + Real.cos α = 1/5)

-- Theorem 1: tan α = -4/3
theorem tan_value : Real.tan α = -4/3 := by
  sorry

-- Theorem 2: 1 / (2cos²α + sinα - sin²α) = 25/22
theorem fraction_value : 1 / (2 * (Real.cos α)^2 + Real.sin α - (Real.sin α)^2) = 25/22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_fraction_value_l199_19980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_le_p_l199_19933

-- Define an acute triangle with ordered side lengths
def AcuteTriangle (a b c : ℝ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ 
  0 < a^2 + b^2 - c^2 ∧ 0 < a^2 + c^2 - b^2 ∧ 0 < b^2 + c^2 - a^2

-- Define the semi-perimeter p
noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2

-- Define the sum q involving cosines
noncomputable def q (a b c A B C : ℝ) : ℝ := a * Real.cos A + b * Real.cos B + c * Real.cos C

-- Theorem statement
theorem q_le_p (a b c A B C : ℝ) 
  (h_acute : AcuteTriangle a b c)
  (h_angles : A + B + C = Real.pi)
  (h_cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (h_cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (h_cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) :
  q a b c A B C ≤ p a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_le_p_l199_19933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemons_for_50_gallons_l199_19905

/-- The number of lemons needed for a given number of gallons of lemonade -/
noncomputable def lemons_needed (gallons : ℝ) : ℝ :=
  if gallons ≤ 40 then
    (30 / 40) * gallons
  else
    30 + (gallons - 40) * (1 + 30 / 40)

/-- Theorem stating the number of lemons needed for 50 gallons of lemonade -/
theorem lemons_for_50_gallons :
  lemons_needed 50 = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemons_for_50_gallons_l199_19905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l199_19942

theorem problem_statement (a b : ℕ) : 
  a > b ∧ b > 0 ∧ 
  Nat.Coprime a b ∧ 
  (a^3 - b^3) / (a - b)^3 = 131/5 → 
  a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l199_19942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_inverse_power_l199_19991

theorem x_plus_inverse_power (φ : ℝ) (x : ℂ) (h1 : 0 < φ) (h2 : φ < π) (h3 : x + 1/x = 2 * Real.sin φ) :
  ∀ n : ℕ, x^n + 1/(x^n) = 2 * Real.sin (n * φ) :=
by
  intro n
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_inverse_power_l199_19991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_solution_l199_19911

theorem no_fractional_solution (x y : ℚ) : 
  (∃ m n : ℤ, (13 : ℚ) * x + (4 : ℚ) * y = ↑m ∧ (10 : ℚ) * x + (3 : ℚ) * y = ↑n) → 
  (∃ a b : ℤ, x = ↑a ∧ y = ↑b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_fractional_solution_l199_19911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l199_19924

/-- Represents the time it takes to fill a tank with multiple pipes -/
noncomputable def fill_time (input_pipe1 : ℝ) (input_pipe2 : ℝ) (output_pipe : ℝ) : ℝ :=
  1 / (1 / input_pipe1 + 1 / input_pipe2 - 1 / output_pipe)

/-- Theorem stating that with given pipe filling/emptying times, the tank will be filled in 9 minutes -/
theorem tank_fill_time :
  fill_time 15 15 45 = 9 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval fill_time 15 15 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l199_19924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_count_l199_19975

theorem cubic_equation_solution_count : 
  ∃! (x : ℤ), x ≥ 0 ∧ x^3 = -6*x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_solution_count_l199_19975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l199_19901

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

-- State the theorem
theorem max_value_of_function (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 4 ∧ DifferentiableAt ℝ (f a) x ∧ deriv (f a) x = 0) →
  (∃ x : ℝ, x = 3 ∧ DifferentiableAt ℝ (f a) x ∧ deriv (f a) x = 0) →
  (∀ x : ℝ, x ∈ Set.Icc 0 4 → f a x ≤ 8) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 4 ∧ f a x = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_function_l199_19901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l199_19935

/-- The function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ := (18*x^5 + 6*x^3 + 3*x^2 + 5*x + 4) / (6*x^5 + 4*x^3 + 5*x^2 + 2*x + 1)

/-- The statement that the horizontal asymptote of f is 3 -/
theorem horizontal_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 3| < ε := by
  sorry

#check horizontal_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l199_19935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l199_19954

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_foci_distance 
  (x y fx1 fy1 fx2 fy2 : ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (h_focus1 : distance_to_focus x y fx1 fy1 = 4) :
  distance_to_focus x y fx2 fy2 = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l199_19954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l199_19912

theorem unique_positive_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₃ > 0 → x₄ > 0 → x₅ > 0 →
  x₁ + x₂ = x₃^2 →
  x₂ + x₃ = x₄^2 →
  x₃ + x₄ = x₅^2 →
  x₄ + x₅ = x₁^2 →
  x₅ + x₁ = x₂^2 →
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := by
  sorry

#check unique_positive_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l199_19912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l199_19951

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - 2^x) / (a + 2^(x+1))

-- State the theorem
theorem odd_function_properties (a : ℝ) :
  -- f is an odd function
  (∀ x, f a (-x) = -(f a x)) →
  -- a = 2
  (a = 2) ∧
  -- f is strictly decreasing on ℝ
  (∀ x y, x < y → f 2 x > f 2 y) ∧
  -- The range of f is (-∞, -1/2) ∪ (-1/2, +∞)
  (∀ y, y ≠ -1/2 → ∃ x, f 2 x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l199_19951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l199_19962

/-- Calculates the total profit of a partnership given the investments and one partner's profit share -/
def calculate_total_profit (invest_a invest_b invest_c a_profit : ℕ) : ℕ :=
  let gcd := invest_a.gcd (invest_b.gcd invest_c)
  let ratio_sum := invest_a / gcd +
                   invest_b / gcd +
                   invest_c / gcd
  let a_ratio := invest_a / gcd
  (ratio_sum * a_profit) / a_ratio

/-- The total profit of the partnership is 12600 given the specified investments and A's profit share -/
theorem partnership_profit :
  calculate_total_profit 6300 4200 10500 3780 = 12600 := by
  sorry

#eval calculate_total_profit 6300 4200 10500 3780

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_l199_19962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_mono_increasing_g_symmetric_about_three_halves_l199_19907

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Define the property of being monotonically increasing on an interval
def MonoIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Statement 1
theorem not_mono_increasing_g :
  ¬ MonoIncreasing g (-Real.pi/3) 0 := by sorry

-- Define the property of function symmetry about a line
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Statement 2
theorem symmetric_about_three_halves (f : ℝ → ℝ) 
  (h : ∀ x, f (-x) = f (3 + x)) :
  SymmetricAbout f (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_mono_increasing_g_symmetric_about_three_halves_l199_19907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_ideal_function_l199_19959

noncomputable def ideal_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 else x^2

theorem f_is_ideal_function : ideal_function f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_ideal_function_l199_19959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_l199_19918

theorem magnitude_of_complex :
  Complex.abs ((5 * Complex.I) / (1 + 2 * Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_l199_19918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_l199_19979

-- Define the functions f and g
def f (x : ℝ) : ℝ := 1 - 2 * x^2
def g (x : ℝ) : ℝ := x^2 - 2 * x

-- Define the piecewise function F
noncomputable def F (x : ℝ) : ℝ := 
  if f x ≥ g x then g x else f x

-- Theorem stating that the maximum value of F is 7/9
theorem max_value_F : ∃ (M : ℝ), M = 7/9 ∧ ∀ (x : ℝ), F x ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_F_l199_19979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_to_line_l199_19903

/-- A curve y = e^x + a is tangent to the line y = x if and only if a = -1 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, (Real.exp x + a = x) ∧ (Real.exp x = 1)) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_to_line_l199_19903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_sine_sum_maximum_l199_19965

-- Define two triangles
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the problem statement
theorem triangle_sine_sum_inequality (abc pqr : Triangle) 
  (h1 : abc.A = pqr.A)
  (h2 : |abc.B - abc.C| < |pqr.B - pqr.C|)
  (h3 : abc.A + abc.B + abc.C = Real.pi)
  (h4 : pqr.A + pqr.B + pqr.C = Real.pi) :
  Real.sin abc.A + Real.sin abc.B + Real.sin abc.C > 
  Real.sin pqr.A + Real.sin pqr.B + Real.sin pqr.C := by
  sorry

-- Define the maximization problem
noncomputable def sine_sum (t : Triangle) : ℝ :=
  Real.sin t.A + Real.sin t.B + Real.sin t.C

theorem sine_sum_maximum (t : Triangle) 
  (h : t.A + t.B + t.C = Real.pi) :
  sine_sum t ≤ sine_sum { A := Real.pi / 3, B := Real.pi / 3, C := Real.pi / 3 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_sine_sum_maximum_l199_19965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_last_count_l199_19947

/-- Represents the number of members in each team -/
def team_sizes : List Nat := [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

/-- Calculates the last number assigned to each team -/
def last_numbers (sizes : List Nat) : List Nat :=
  List.scanl (·+·) 0 sizes

/-- Counts how many teams have an odd last number -/
def count_odd_last (numbers : List Nat) : Nat :=
  numbers.filter (fun n => n % 2 = 1) |>.length

/-- The main theorem to be proved -/
theorem odd_last_count :
  count_odd_last (last_numbers team_sizes) = 10 := by
  sorry

#eval count_odd_last (last_numbers team_sizes)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_last_count_l199_19947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l199_19902

theorem triangle_side_calculation (a b : ℝ) (A B C : ℝ) :
  0 < a →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a = 8 →
  B = Real.pi / 3 →
  C = 5 * Real.pi / 12 →
  b = a * (Real.sin B / Real.sin A) →
  b = 4 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l199_19902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l199_19974

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x ≤ y → f y ≤ f x

theorem f_negative_range (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_decreasing : decreasing_on f (Set.Ici 0))
  (h_zero : f (-2) = 0) :
  ∀ x, f x < 0 ↔ x ∈ Set.Ioo (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_range_l199_19974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angles_relation_l199_19953

/-- Represents a right circular cone -/
structure RightCircularCone where
  vertexAngle : Real
  generatorAngle : Real
  projectedGeneratorAngle : Real

/-- Indicates that two cones share a common base plane, axis, and apex -/
def RightCircularCone.sharedBaseAxisApex (cone1 cone2 : RightCircularCone) : Prop :=
  sorry

theorem cone_angles_relation (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π) :
  let φ := Real.arcsin (Real.sin (α / 2) / Real.sin (β / 2))
  let x := Real.arccos ((1 + Real.cos β + Real.sqrt ((1 - Real.cos β) * (Real.cos α - Real.cos β))) / 2)
  ∃ (cone1 cone2 : RightCircularCone),
    RightCircularCone.sharedBaseAxisApex cone1 cone2 ∧
    cone1.vertexAngle = 2 * φ ∧
    cone2.vertexAngle = φ ∧
    cone1.generatorAngle = α ∧
    cone1.projectedGeneratorAngle = β ∧
    cone2.projectedGeneratorAngle = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angles_relation_l199_19953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_l199_19923

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
  t.c * Real.cos t.B = t.a + (1/2) * t.b ∧
  (Real.sqrt 3 / 12) * t.c = (1/2) * t.a * t.b * Real.sin t.C

-- Theorem statement
theorem min_side_c (t : Triangle) (h : satisfies_conditions t) : 
  t.c ≥ 1 ∧ ∃ (t' : Triangle), satisfies_conditions t' ∧ t'.c = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_c_l199_19923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_difference_l199_19931

theorem arctan_tan_difference : 
  Real.arctan (Real.tan (5 * Real.pi / 12) - 3 * Real.tan (Real.pi / 12)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctan_tan_difference_l199_19931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_divisibility_l199_19916

/-- Represents a three-digit number ABC -/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C : Nat
  h1 : A ≠ 0
  h2 : B ≠ C

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- Reverses a ThreeDigitNumber to CBA -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

/-- The theorem to be proved -/
theorem square_difference_divisibility (n : ThreeDigitNumber) :
  ∃ k : Int, (n.toNat ^ 2 : Int) - (n.reverse ^ 2 : Int) = k * (n.A - n.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_divisibility_l199_19916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l199_19970

theorem condition_relationship : 
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → (2 : ℝ)^x > 2) ∧ 
  (∃ x : ℝ, (2 : ℝ)^x > 2 ∧ (x - 2) * (x - 4) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_relationship_l199_19970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l199_19992

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2) / x

noncomputable def tangent_line (f : ℝ → ℝ) (x₀ : ℝ) : ℝ → ℝ :=
  λ x ↦ (f x₀) + (deriv f x₀) * (x - x₀)

theorem tangent_line_at_one :
  ∀ x y, tangent_line f 1 x = y ↔ x + y - 3 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l199_19992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sets_l199_19944

/-- The function f(x) = -x/(1+|x|) -/
noncomputable def f (x : ℝ) : ℝ := -x / (1 + |x|)

/-- The set M = [a,b] -/
def M (a b : ℝ) : Set ℝ := Set.Icc a b

/-- The set N = {y | y = f(x), x ∈ M} -/
def N (a b : ℝ) : Set ℝ := {y | ∃ x ∈ M a b, f x = y}

/-- There are no real number pairs (a,b) with a < b such that M = N -/
theorem no_equal_sets : ¬∃ a b : ℝ, a < b ∧ M a b = N a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_sets_l199_19944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l199_19900

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = 9 * Real.sqrt 7 / 14)
  (h2 : 0 < α)
  (h3 : α < π/3) :
  Real.sin α = 2 * Real.sqrt 7 / 7 ∧ 
  Real.cos (2*α - π/4) = (4 * Real.sqrt 6 - Real.sqrt 2) / 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l199_19900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_100gon_division_congruent_triangles_l199_19982

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : Prop

/-- A division of a polygon into parallelograms and triangles -/
structure PolygonDivision (n : ℕ) where
  polygon : RegularPolygon n
  parallelograms : List (List (Fin n))
  triangles : List (List (Fin n))
  is_valid_division : Prop

/-- Two triangles are congruent -/
def are_congruent {n : ℕ} (t1 t2 : List (Fin n)) : Prop := sorry

theorem regular_100gon_division_congruent_triangles :
  ∀ (d : PolygonDivision 100),
    d.triangles.length = 2 →
    are_congruent (d.triangles.get! 0) (d.triangles.get! 1) :=
by
  intro d h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_100gon_division_congruent_triangles_l199_19982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l199_19977

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^3 - a*x) / Real.log a

-- State the theorem
theorem monotone_increasing_f_implies_a_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : StrictMonoOn (f a) (Set.Ioo (-1/2) 0)) :
  3/4 ≤ a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_range_l199_19977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_positive_sequence_l199_19934

-- Define a space with a distance function
variable {α : Type*} [MetricSpace α]

-- Define the countable set X
variable (X : ℕ → α)

-- Define the theorem
theorem existence_of_positive_sequence :
  ∃ (a : ℕ → ℝ), (∀ k, a k > 0) ∧
  ∀ Z : α, Z ∉ Set.range X →
    {k : ℕ | ∀ i ≤ k, dist Z (X i) ≥ a k}.Infinite :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_positive_sequence_l199_19934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_equality_implies_ratio_l199_19969

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def PointOnSegment (A B M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the incircle radius of a triangle
noncomputable def inradius (A B C : ℝ × ℝ) : ℝ :=
  sorry

-- Main theorem
theorem incircle_radius_equality_implies_ratio (ABC : Triangle) (M : ℝ × ℝ) :
  distance ABC.A ABC.B = 12 →
  distance ABC.B ABC.C = 13 →
  distance ABC.A ABC.C = 15 →
  PointOnSegment ABC.A ABC.C M →
  inradius ABC.A ABC.B M = inradius ABC.B ABC.C M →
  (distance ABC.A M) / (distance M ABC.C) = 22 / 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_equality_implies_ratio_l199_19969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l199_19957

/-- The number of sides in an octagon -/
def n : ℕ := 8

/-- The sum of interior angles of a polygon with n sides -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular polygon has all interior angles equal -/
noncomputable def interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle :
  interior_angle n = 135 := by
  -- Expand the definitions
  unfold interior_angle
  unfold sum_interior_angles
  unfold n
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l199_19957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l199_19909

/-- Represents a table tennis player --/
inductive Player
| Experienced
| New

/-- Represents a team arrangement --/
def Arrangement := Fin 3 → Player

/-- The total number of players --/
def total_players : Nat := 5

/-- The number of experienced players --/
def experienced_players : Nat := 2

/-- The number of new players --/
def new_players : Nat := 3

/-- Checks if an arrangement has at least one experienced player --/
def has_experienced (a : Arrangement) : Prop :=
  ∃ i, a i = Player.Experienced

/-- Checks if an arrangement has at least one new player in position 1 or 2 --/
def has_new_in_first_two (a : Arrangement) : Prop :=
  a 0 = Player.New ∨ a 1 = Player.New

/-- Provide instances for Fintype and DecidablePred --/
instance : Fintype Arrangement := by sorry

instance : DecidablePred (λ a : Arrangement => has_experienced a ∧ has_new_in_first_two a) := by sorry

/-- The main theorem --/
theorem valid_arrangements_count :
  (Finset.filter (λ a : Arrangement => has_experienced a ∧ has_new_in_first_two a)
    (Finset.univ : Finset Arrangement)).card = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l199_19909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_count_l199_19966

theorem student_selection_count : 
  let total_students : ℕ := 7
  let boys : ℕ := 4
  let girls : ℕ := 3
  let selection_size : ℕ := 4
  let valid_selection (b g : ℕ) := b + g = selection_size ∧ b > 0 ∧ g > 0
  let count_selections (b g : ℕ) := Nat.choose boys b * Nat.choose girls g
  (Finset.sum (Finset.range (selection_size + 1)) (fun b => 
    Finset.sum (Finset.range (selection_size + 1)) (fun g => 
      if valid_selection b g then count_selections b g else 0))) = 34
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_selection_count_l199_19966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l199_19971

/-- The area of a rectangle inscribed in a triangle -/
theorem inscribed_rectangle_area (b h x : ℝ) (hb : b > 0) (hh : h > 0) (hx : x > 0) (hxh : x < h) :
  b * x^2 / h = (b * x / h) * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l199_19971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l199_19941

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (x y k m : ℝ) : Prop := y = k*x - m

-- Define point A
def point_A : ℝ × ℝ := (0, 1)

-- Define the angle BAC
def angle_BAC : ℝ := 60

-- Define the areas ratio
def areas_ratio (S₁ S₂ : ℝ) : Prop := S₁ = 2 * S₂

-- Theorem statement
theorem line_circle_intersection 
  (k m : ℝ) 
  (h_intersect : ∃ (B C : ℝ × ℝ), my_circle B.1 B.2 ∧ my_circle C.1 C.2 ∧ my_line B.1 B.2 k m ∧ my_line C.1 C.2 k m)
  (h_angle : ∃ (B C : ℝ × ℝ), angle_BAC = 60)
  (h_areas : ∃ (S₁ S₂ : ℝ), areas_ratio S₁ S₂) :
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l199_19941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_set_T_l199_19904

def average (s : Finset ℕ) : ℚ := (s.sum (fun x => (x : ℚ))) / s.card

theorem average_of_set_T (T : Finset ℕ) (hT : T.Nonempty) : average T = 55 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_set_T_l199_19904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l199_19998

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the line mx + (2m-1)y + 1 = 0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -m / (2*m - 1)

/-- The slope of the line 3x + my + 3 = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -3 / m

/-- m = -1 is a sufficient but not necessary condition for perpendicularity -/
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → perpendicular (slope1 m) (slope2 m)) ∧
  ¬(perpendicular (slope1 m) (slope2 m) → m = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l199_19998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_implies_f_form_max_value_in_interval_a_range_for_inequality_l199_19999

noncomputable section

variable (a b : ℝ)
variable (f : ℝ → ℝ)

def f_def (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x

theorem unique_zero_point_implies_f_form (h1 : a ≠ 0) (h2 : f_def a b 2 = 0) 
  (h3 : ∃! x, f_def a b x - x = 0) : 
  f_def a b = fun x ↦ -(1/2) * x^2 + x := by sorry

theorem max_value_in_interval (h1 : a ≠ 0) (h2 : f_def a b 2 = 0) :
  (∀ x ∈ Set.Icc (-1) 2, f_def a b x ≤ (if a > 0 then 3*a else -a)) ∧ 
  (∃ x ∈ Set.Icc (-1) 2, f_def a b x = (if a > 0 then 3*a else -a)) := by sorry

theorem a_range_for_inequality (h1 : a ≠ 0) (h2 : f_def a b 2 = 0) :
  (∀ x ≥ 2, f_def a b x ≥ 2 - a) ↔ a ≥ 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_point_implies_f_form_max_value_in_interval_a_range_for_inequality_l199_19999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_equals_265_l199_19996

-- Define the velocity function
def v (t : ℝ) : ℝ := 3 * t^2 + 10 * t + 3

-- Define the distance function as the integral of velocity
noncomputable def distance (a b : ℝ) : ℝ := ∫ t in a..b, v t

-- Theorem statement
theorem distance_traveled_equals_265 : distance 0 5 = 265 := by
  -- Expand the definition of distance
  unfold distance
  -- Evaluate the integral
  simp [v]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_equals_265_l199_19996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l199_19976

theorem cos_double_angle_special_case (θ : ℝ) : 
  Real.sin (π / 2 + θ) = 3 / 5 → Real.cos (2 * θ) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l199_19976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l199_19963

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 0)

theorem angle_between_vectors :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l199_19963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dressing_mixture_theorem_l199_19917

/-- Represents a salad dressing with vinegar and oil percentages -/
structure Dressing where
  vinegar_percent : ℝ
  oil_percent : ℝ
  sum_to_100 : vinegar_percent + oil_percent = 100

/-- Represents a mixture of two dressings -/
def mix_dressings (p q : Dressing) (p_ratio : ℝ) : Dressing :=
  { vinegar_percent := p.vinegar_percent * p_ratio + q.vinegar_percent * (1 - p_ratio),
    oil_percent := p.oil_percent * p_ratio + q.oil_percent * (1 - p_ratio),
    sum_to_100 := by sorry }

theorem dressing_mixture_theorem (p q : Dressing) 
    (hp : p.vinegar_percent = 30 ∧ p.oil_percent = 70)
    (hq : q.vinegar_percent = 10 ∧ q.oil_percent = 90)
    (h_new : ∃ (x : ℝ), (mix_dressings p q x).vinegar_percent = 12) :
    ∃ (x : ℝ), mix_dressings p q x = mix_dressings p q 0.1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dressing_mixture_theorem_l199_19917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_theorem_l199_19906

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate -/
noncomputable def water_flow_per_minute (depth : ℝ) (width : ℝ) (flow_rate_kmph : ℝ) : ℝ :=
  let cross_sectional_area := depth * width
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  cross_sectional_area * flow_rate_mpm

/-- Theorem stating that for a river with depth 2 m, width 45 m, and flow rate 3 kmph, 
    the volume of water flowing into the sea per minute is 4500 cubic meters -/
theorem river_flow_theorem : 
  water_flow_per_minute 2 45 3 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_theorem_l199_19906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_and_range_of_m_xor_l199_19929

/-- Line equation mx - y + 1 = 0 -/
def line_equation (m x y : ℝ) : Prop := m * x - y + 1 = 0

/-- Circle equation (x - 2)^2 + y^2 = 4 -/
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- Hyperbola equation x^2 / (m - 1) + y^2 / (2 - m) = 1 -/
def hyperbola_equation (m x y : ℝ) : Prop := x^2 / (m - 1) + y^2 / (2 - m) = 1

/-- Condition p: The line intersects the circle -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, line_equation m x y ∧ circle_equation x y

/-- Condition q: The equation represents a hyperbola -/
def q (m : ℝ) : Prop := ∃ x y : ℝ, hyperbola_equation m x y

theorem range_of_m_and :
  {m : ℝ | p m ∧ q m} = Set.Iic (3/4) :=
sorry

theorem range_of_m_xor :
  {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)} = Set.Ioo (3/4) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_and_range_of_m_xor_l199_19929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l199_19936

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, -1, 1)
  let b : ℝ × ℝ × ℝ := (m, -1, 1)
  parallel a b → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l199_19936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_decreasing_function_l199_19913

/-- The function f(x) defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x * Real.exp (t * x) - Real.exp x + 1

/-- Theorem 1: Condition for no real roots -/
theorem no_real_roots (t : ℝ) :
  (∀ x : ℝ, f t x ≠ 1) ↔ t < 1 - 1 / Real.exp 1 := by
  sorry

/-- Theorem 2: Condition for decreasing function -/
theorem decreasing_function (t : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > x → f t y < f t x)) ↔ t ≤ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_decreasing_function_l199_19913
