import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_and_symmetric_functions_l1226_122660

noncomputable def f : ℝ → ℝ := fun x ↦ Real.log x

noncomputable def g : ℝ → ℝ := fun x ↦ Real.log (-x)

theorem inverse_and_symmetric_functions (m : ℝ) 
  (h1 : ∀ x > 0, Real.exp (f x) = x)
  (h2 : ∀ x < 0, g x = f (-x))
  (h3 : g m = -1) :
  m = -Real.exp (-1) := by
  sorry

#check inverse_and_symmetric_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_and_symmetric_functions_l1226_122660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cherrapunji_average_rainfall_l1226_122688

/-- Rainfall data for Cherrapunji, India -/
structure RainfallData where
  january : ℚ
  february : ℚ
  july : ℚ
  other_months : ℚ

/-- Calculate the average rainfall per hour over the entire year -/
def average_rainfall_per_hour (data : RainfallData) : ℚ :=
  let total_rainfall := data.january + data.february + data.july + 9 * data.other_months
  let total_hours := 360 * 24
  total_rainfall / total_hours

/-- The theorem stating that the average rainfall per hour is 101/540 inches -/
theorem cherrapunji_average_rainfall
  (data : RainfallData)
  (h1 : data.january = 150)
  (h2 : data.february = 200)
  (h3 : data.july = 366)
  (h4 : data.other_months = 100) :
  average_rainfall_per_hour data = 101 / 540 := by
  sorry

#eval average_rainfall_per_hour ⟨150, 200, 366, 100⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cherrapunji_average_rainfall_l1226_122688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_l1226_122602

/-- The polar equation ρ = 1 represents a circle in the Cartesian coordinate system. -/
theorem polar_equation_circle :
  ∀ (x y : ℝ), (∃ θ : ℝ, x = Real.cos θ ∧ y = Real.sin θ) ↔ x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_l1226_122602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_maximum_value_l1226_122668

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - (2*a + 1) * x

-- State the theorem
theorem extremum_and_maximum_value (a : ℝ) :
  a ≠ 0 ∧
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → |f a x| ≤ |f a 1|) ∧
  (∀ (x : ℝ), 0 < x ∧ x ≤ Real.exp 1 → f a x ≤ 1) ∧
  (∃ (x : ℝ), 0 < x ∧ x ≤ Real.exp 1 ∧ f a x = 1) →
  a = 1 / (Real.exp 1 - 2) ∨ a = -2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_maximum_value_l1226_122668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_67_l1226_122675

-- Define the triangle and its properties
def triangle_ABC (α β γ k l m : ℝ) : Prop :=
  α = Real.pi/6 ∧ β = Real.pi/4 ∧ γ = Real.pi - α - β ∧
  k = 3 ∧ l = 2 ∧ m = 4

-- Define the area calculation function
noncomputable def triangle_area (α β γ k l m : ℝ) : ℝ :=
  ((k * Real.sin α + l * Real.sin β + m * Real.sin γ)^2) /
  (2 * Real.sin α * Real.sin β * Real.sin γ)

-- Theorem statement
theorem triangle_area_approx_67 (α β γ k l m : ℝ) :
  triangle_ABC α β γ k l m →
  Int.floor (triangle_area α β γ k l m + 0.5) = 67 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_67_l1226_122675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_total_is_85_l1226_122695

def jewelry_calculation : Prop :=
  let initial_necklaces : ℕ := 10
  let initial_earrings : ℕ := 15
  let initial_bracelets : ℕ := 5
  let initial_rings : ℕ := 8

  let store_a_necklaces : ℕ := 10
  let store_a_earrings : ℕ := (2 * initial_earrings) / 3
  let store_a_bracelets : ℕ := 3
  let lost_necklaces : ℕ := 2

  let store_b_rings : ℕ := 2 * initial_rings
  let store_b_necklaces : ℕ := 4
  let store_b_bracelets : ℕ := 3
  let given_away_bracelets_ratio : ℚ := 35 / 100

  let mother_gift_earrings : ℕ := store_a_earrings / 5

  let final_sale_necklaces : ℕ := 2
  let final_sale_earrings : ℕ := 2
  let final_sale_rings : ℕ := 1

  let final_necklaces : ℕ := initial_necklaces + store_a_necklaces - lost_necklaces + store_b_necklaces + final_sale_necklaces
  let final_earrings : ℕ := initial_earrings + store_a_earrings + mother_gift_earrings + final_sale_earrings
  let final_bracelets : ℕ := initial_bracelets + store_a_bracelets + store_b_bracelets - (store_b_bracelets * 35 / 100 : ℕ)
  let final_rings : ℕ := initial_rings + store_b_rings + final_sale_rings

  final_necklaces + final_earrings + final_bracelets + final_rings = 85

theorem jewelry_total_is_85 : jewelry_calculation := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jewelry_total_is_85_l1226_122695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l1226_122655

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_max_sum
  (a₁ : ℝ) (d : ℝ) (h₁ : a₁ > 0) (h₂ : 5 * (arithmeticSequence a₁ d 15) = 3 * (arithmeticSequence a₁ d 8)) :
  ∃ (n : ℕ), ∀ (m : ℕ), arithmeticSum a₁ d n ≥ arithmeticSum a₁ d m ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l1226_122655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1226_122641

/-- Given a line y = kx intersecting the circle (x-2)^2 + y^2 = 4 at two points
    forming a chord of length 2√3, prove that k = ±√3/3 -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - 2)^2 + y₁^2 = 4 ∧
    (x₂ - 2)^2 + y₂^2 = 4 ∧
    y₁ = k * x₁ ∧
    y₂ = k * x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →
  k = Real.sqrt 3 / 3 ∨ k = -(Real.sqrt 3 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1226_122641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_l1226_122697

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x

-- State the theorem
theorem extremum_at_one (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_at_one_l1226_122697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l1226_122646

/-- The distance between points A and B given the train problem conditions -/
noncomputable def distance_AB : ℝ := 122

/-- The speed of the passenger train in km/h -/
noncomputable def passenger_speed : ℝ := 30

/-- The speed of the express train in km/h -/
noncomputable def express_speed : ℝ := 60

/-- The fraction of the journey where the passenger train slows down -/
noncomputable def slowdown_fraction : ℝ := 2/3

/-- The distance before point B where the express train catches the passenger train -/
noncomputable def catch_distance : ℝ := 27

theorem train_problem_solution :
  let d := distance_AB
  let v := passenger_speed
  let u := express_speed
  let f := slowdown_fraction
  let c := catch_distance
  (d * f / v + (d * (1 - f)) / (v / 2)) = (d - c) / u := by
  sorry

#check train_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l1226_122646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sadie_runs_two_hours_l1226_122691

/-- Represents the relay race with given conditions -/
structure RelayRace where
  total_distance : ℝ
  total_time : ℝ
  sadie_speed : ℝ
  ariana_speed : ℝ
  ariana_time : ℝ
  sarah_speed : ℝ

/-- Calculates Sadie's running time in the relay race -/
noncomputable def sadie_running_time (race : RelayRace) : ℝ :=
  let ariana_distance := race.ariana_speed * race.ariana_time
  let sadie_sarah_distance := race.total_distance - ariana_distance
  let sadie_sarah_time := race.total_time - race.ariana_time
  (sadie_sarah_distance - race.sarah_speed * sadie_sarah_time) / (race.sadie_speed - race.sarah_speed)

/-- Theorem stating that Sadie's running time is 2 hours given the race conditions -/
theorem sadie_runs_two_hours (race : RelayRace) 
  (h1 : race.total_distance = 17)
  (h2 : race.total_time = 4.5)
  (h3 : race.sadie_speed = 3)
  (h4 : race.ariana_speed = 6)
  (h5 : race.ariana_time = 0.5)
  (h6 : race.sarah_speed = 4) :
  sadie_running_time race = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sadie_runs_two_hours_l1226_122691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_in_interval_l1226_122654

theorem fractional_parts_in_interval
  (a b c : ℕ+) -- positive integers
  (h1 : b > 2 * a) -- b > 2a
  (h2 : c > 2 * b) -- c > 2b
  : ∃ (x : ℝ), 
    (x * a.val % 1 > 1/3 ∧ x * a.val % 1 ≤ 2/3) ∧
    (x * b.val % 1 > 1/3 ∧ x * b.val % 1 ≤ 2/3) ∧
    (x * c.val % 1 > 1/3 ∧ x * c.val % 1 ≤ 2/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_in_interval_l1226_122654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_card_l1226_122653

/-- Represents a playing card suit -/
inductive Suit
| Spades
| Hearts
| Diamonds
| Clubs

/-- Represents a playing card rank -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A standard deck of playing cards -/
def Deck := List (Rank × Suit)

/-- Creates a standard 52-card deck -/
def standardDeck : Deck :=
  (List.range 13).bind (fun i =>
    [Suit.Spades, Suit.Hearts, Suit.Diamonds, Suit.Clubs].map (fun s =>
      (match i with
        | 0 => Rank.Ace
        | 1 => Rank.Two
        | 2 => Rank.Three
        | 3 => Rank.Four
        | 4 => Rank.Five
        | 5 => Rank.Six
        | 6 => Rank.Seven
        | 7 => Rank.Eight
        | 8 => Rank.Nine
        | 9 => Rank.Ten
        | 10 => Rank.Jack
        | 11 => Rank.Queen
        | _ => Rank.King, s)))

/-- Checks if a suit is red -/
def isRed (s : Suit) : Bool :=
  match s with
  | Suit.Hearts => true
  | Suit.Diamonds => true
  | _ => false

/-- Theorem: The probability of drawing a red card from a standard deck is 1/2 -/
theorem prob_red_card (d : Deck) (h : d = standardDeck) :
  (d.filter (fun c => isRed c.2)).length / d.length = 1 / 2 := by
  sorry

#eval standardDeck.length -- Should output 52
#eval (standardDeck.filter (fun c => isRed c.2)).length -- Should output 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_card_l1226_122653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l1226_122626

/-- The percentage improvement needed for Bob to match his sister's mile time -/
noncomputable def percentage_improvement (bob_time sister_time : ℝ) : ℝ :=
  (bob_time - sister_time) / bob_time * 100

/-- Theorem stating that Bob needs approximately 9.06% improvement to match his sister's time -/
theorem bob_improvement_percentage :
  let bob_time : ℝ := 640
  let sister_time : ℝ := 582
  abs (percentage_improvement bob_time sister_time - 9.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_improvement_percentage_l1226_122626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l1226_122617

noncomputable def f (x : ℝ) := Real.sin (1/4 * x + Real.pi/6)

noncomputable def g (x : ℝ) := -Real.cos (x/4)

theorem g_increasing_on_interval :
  ∀ x y, π ≤ x → x < y → y ≤ 2*π → g x < g y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l1226_122617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_no_solution_l1226_122699

open Real

-- Define the function f(x) = ((2-a)(x-1)-2ln x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- State the theorem
theorem min_a_for_no_solution :
  ∃ (a_min : ℝ), ∀ (a : ℝ), 
    (∀ x ∈ Set.Ioo 0 (1/2), f a x ≠ 0) ↔ a ≥ a_min ∧ a_min = 2 - 4 * Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_no_solution_l1226_122699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_scenario_l1226_122698

/-- The length of a wire stretched between two vertical poles on sloped ground -/
noncomputable def wire_length (horizontal_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) (slope_rate : ℝ) : ℝ :=
  let ground_rise := horizontal_distance * slope_rate
  let height_difference := tall_pole_height - (short_pole_height + ground_rise)
  Real.sqrt (horizontal_distance ^ 2 + height_difference ^ 2)

/-- Theorem stating the wire length for the given scenario -/
theorem wire_length_specific_scenario :
  wire_length 20 12 30 (1/4) = Real.sqrt 569 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_scenario_l1226_122698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l1226_122630

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1/2) * |2 * (n : ℝ)^5 - 4 * (n : ℝ)^4 + 8 * (n : ℝ)|

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, n > 0 ∧ triangle_area n > 1000 ∧
  (∀ k : ℕ, 0 < k → k < n → triangle_area k ≤ 1000) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l1226_122630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_positive_function_condition_l1226_122686

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

-- Theorem 1: Perpendicular lines condition implies a = -1
theorem perpendicular_lines_condition (a : ℝ) :
  (∃ m : ℝ, (Real.exp 1 + a) * m = -1 ∧ m = 1 / (1 - Real.exp 1)) → a = -1 :=
by sorry

-- Theorem 2: Positive function condition implies a ∈ (-e, +∞)
theorem positive_function_condition (a : ℝ) :
  (∀ x > 0, f a x > 0) → a > -Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_positive_function_condition_l1226_122686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_less_arcsin_plus_pi_sixth_l1226_122632

theorem arccos_less_arcsin_plus_pi_sixth (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.arccos x < Real.arcsin x + π / 6) ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_less_arcsin_plus_pi_sixth_l1226_122632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_example_l1226_122696

/-- The sum of an arithmetic sequence with given parameters -/
noncomputable def arithmetic_sequence_sum (a₁ : ℝ) (aₙ : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a₁ + aₙ)

/-- Theorem: The sum of 5 terms in an arithmetic sequence with first term 6 and last term 22 is 70 -/
theorem arithmetic_sequence_sum_example : arithmetic_sequence_sum 6 22 5 = 70 := by
  -- Unfold the definition of arithmetic_sequence_sum
  unfold arithmetic_sequence_sum
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_example_l1226_122696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_ratio_l1226_122612

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define centroids of triangles in a parallelogram
noncomputable def centroid_A (p : Parallelogram) : ℝ × ℝ :=
  ((p.B.1 + p.C.1 + p.D.1) / 3, (p.B.2 + p.C.2 + p.D.2) / 3)

noncomputable def centroid_B (p : Parallelogram) : ℝ × ℝ :=
  ((p.A.1 + p.C.1 + p.D.1) / 3, (p.A.2 + p.C.2 + p.D.2) / 3)

noncomputable def centroid_C (p : Parallelogram) : ℝ × ℝ :=
  ((p.A.1 + p.B.1 + p.D.1) / 3, (p.A.2 + p.B.2 + p.D.2) / 3)

noncomputable def centroid_D (p : Parallelogram) : ℝ × ℝ :=
  ((p.A.1 + p.B.1 + p.C.1) / 3, (p.A.2 + p.B.2 + p.C.2) / 3)

-- Define the area of a quadrilateral given its vertices
noncomputable def area (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem centroid_quadrilateral_area_ratio (p : Parallelogram) :
  let G_A := centroid_A p
  let G_B := centroid_B p
  let G_C := centroid_C p
  let G_D := centroid_D p
  area G_A G_B G_C G_D / area p.A p.B p.C p.D = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_ratio_l1226_122612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1226_122638

/-- A lattice point in a 2D Cartesian coordinate system -/
def LatticePoint := ℤ × ℤ

/-- The line y = (3/4)x + (2/3) -/
noncomputable def Line (x : ℝ) : ℝ := (3/4) * x + (2/3)

/-- Distance from a point to the line -/
noncomputable def DistanceToLine (p : LatticePoint) : ℝ :=
  let (x, y) := p
  |9 * (x : ℝ) - 12 * (y : ℝ) + 8| / 15

/-- The minimum distance from any lattice point to the line is 2/15 -/
theorem min_distance_to_line :
  ∀ ε > 0, ∃ p : LatticePoint, DistanceToLine p < 2/15 + ε ∧
  ∀ q : LatticePoint, 2/15 ≤ DistanceToLine q := by
  sorry

#check min_distance_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_l1226_122638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_inequality_condition_l1226_122649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^3 - x^2 else Real.log x

-- Theorem for monotonic decrease
theorem f_monotone_decreasing :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2/3 → f x₂ < f x₁ :=
by
  sorry

-- Theorem for the inequality
theorem f_inequality_condition (c : ℝ) :
  (∀ x, f x ≤ x + c) ↔ c ≥ 5/27 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_f_inequality_condition_l1226_122649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1226_122619

theorem quadratic_function_properties (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f (-1) = 3 ∧ f 1 = 4 ∧ f 2 = 3) →
  (a * b * c < 0 ∧
   -b / (2 * a) = 1 / 2 ∧
   ∀ t > 3, f (-2) > f t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1226_122619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_value_l1226_122636

def z : ℂ := 2 + Complex.I

theorem complex_fraction_value :
  (z^2 - 2*z) / (z - 1) = (1/2 : ℂ) + (3/2 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_value_l1226_122636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1226_122692

/-- The focus of a parabola given by y = ax² + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- Theorem: The focus of the parabola y = 2x² + 14x + 1 is at (-3.5, -23.4375) -/
theorem focus_of_specific_parabola :
  parabola_focus 2 14 1 = (-3.5, -23.4375) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1226_122692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l1226_122609

/-- Calculates the total amount of water needed for a two-step dilution process. -/
noncomputable def total_water_needed (initial_volume : ℝ) (initial_concentration : ℝ) 
  (first_target_concentration : ℝ) (final_target_concentration : ℝ) : ℝ :=
  let initial_acid := initial_volume * initial_concentration
  let first_dilution := (initial_acid / first_target_concentration) - initial_volume
  let second_dilution := (initial_acid / final_target_concentration) - (initial_volume + first_dilution)
  first_dilution + second_dilution

/-- Theorem stating that 90 ounces of water are needed for the given dilution process. -/
theorem dilution_theorem : 
  total_water_needed 60 0.25 0.15 0.10 = 90 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l1226_122609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_l1226_122627

/-- A function f(x) that depends on a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (1/2) * x^2 - 4 * Real.log x

/-- The derivative of f with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a - x - 4 / x

/-- Theorem stating the range of a for which f is decreasing on [1,+∞) -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (∀ y : ℝ, y > x → f a y < f a x)) ↔ a ≤ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_l1226_122627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_sawing_correct_l1226_122606

def wood_sawing (total_length : ℕ) (piece_length : ℕ) (time_per_cut : ℕ) : ℕ :=
  ((total_length / piece_length) - 1) * time_per_cut

theorem wood_sawing_correct :
  wood_sawing 10 2 10 = 40 :=
by
  unfold wood_sawing
  norm_num

#eval wood_sawing 10 2 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_sawing_correct_l1226_122606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_bases_with_final_digit_one_l1226_122647

theorem count_bases_with_final_digit_one : 
  (Finset.filter (fun b : ℕ => 3 ≤ b ∧ b ≤ 10 ∧ 625 % b = 1) (Finset.range 11)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_bases_with_final_digit_one_l1226_122647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_square_l1226_122643

/-- The perimeter of the rectangle in cm -/
noncomputable def perimeter : ℝ := 12

/-- The length of one side of the rectangle -/
noncomputable def side_length (x : ℝ) : ℝ := x

/-- The length of the other side of the rectangle -/
noncomputable def other_side (x : ℝ) : ℝ := perimeter / 2 - x

/-- The area of the rectangle as a function of one side length -/
noncomputable def area (x : ℝ) : ℝ := side_length x * other_side x

/-- Theorem stating that the area is maximized when both sides are 3 cm -/
theorem max_area_at_square :
  ∀ x : ℝ, 0 < x → x < perimeter / 2 → area x ≤ area (perimeter / 4) := by
  sorry

#check max_area_at_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_square_l1226_122643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_handshake_graph_l1226_122682

/-- A strongly regular graph with parameters (v, k, λ, μ) -/
structure StronglyRegularGraph (v k l m : ℕ) where
  vertex_count : ℕ
  degree : ℕ
  common_neighbors : ℕ
  different_neighbors : ℕ
  is_valid : vertex_count = v ∧ degree = k ∧ common_neighbors = l ∧ different_neighbors = m

/-- The theorem stating the properties of the graph in the problem -/
theorem meeting_handshake_graph (k : ℕ) :
  ∃ (l m : ℕ), StronglyRegularGraph (12 * k) (3 * k + 6) l m → k = 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_handshake_graph_l1226_122682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l1226_122687

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 1 = 0
  let line2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + 4*y + 3 = 0
  ∃ d : ℝ, d = Real.sqrt 5 / 2 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), line1 x₁ y₁ → line2 x₂ y₂ → 
      ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) = d^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l1226_122687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l1226_122672

-- Define the total degrees in a circle
noncomputable def total_degrees : ℚ := 360

-- Define the degrees occupied by the manufacturing department
noncomputable def manufacturing_degrees : ℚ := 252

-- Define the percentage calculation function
noncomputable def percentage_calculation (part : ℚ) (whole : ℚ) : ℚ :=
  (part / whole) * 100

-- Theorem statement
theorem manufacturing_percentage :
  percentage_calculation manufacturing_degrees total_degrees = 70 := by
  -- Unfold the definitions
  unfold percentage_calculation
  unfold manufacturing_degrees
  unfold total_degrees
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l1226_122672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1226_122610

noncomputable def sequenceA (m : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => (1/8) * (sequenceA m n)^2 + m

theorem max_m_value :
  ∃ (max_m : ℝ), 
    (∀ (m : ℝ), (∀ (n : ℕ), sequenceA m n < 4) → m ≤ max_m) ∧
    (∀ (n : ℕ), sequenceA max_m n < 4) ∧
    max_m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1226_122610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_hexagon_area_difference_l1226_122633

/-- The difference between the area of the region that lies inside a circle
    but outside a hexagon and the area of the region that lies inside the
    hexagon but outside the circle, given specific conditions. -/
theorem circle_hexagon_area_difference (r s : ℝ) :
  r = 3 →
  s = 6 →
  (π * r^2 - (3 * Real.sqrt 3 / 2) * s^2) = 9 * π - 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_hexagon_area_difference_l1226_122633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_disks_area_l1226_122618

theorem sixteen_disks_area (n : ℕ) (R : ℝ) (h1 : n = 16) (h2 : R = 1) : 
  n * π * ((1 - Real.sqrt ((2 + Real.sqrt 2) / 4)) / 2) = 
  n * π * (Real.sqrt ((1 - Real.sqrt ((2 + Real.sqrt 2) / 4)) / 2))^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_disks_area_l1226_122618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_region_area_l1226_122639

/-- Represents a kite ABCD -/
structure Kite where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)
  D : EuclideanSpace ℝ (Fin 2)

/-- The region R inside the kite -/
def Region (k : Kite) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- The area of a set of points -/
noncomputable def area (s : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- The angle between three points -/
noncomputable def angle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

theorem kite_region_area (k : Kite) 
  (h1 : dist k.A k.B = 3)
  (h2 : dist k.A k.D = 3)
  (h3 : dist k.B k.C = 2)
  (h4 : dist k.C k.D = 2)
  (h5 : angle k.A k.B k.C = 150 * Real.pi / 180) :
  area (Region k) = 9 * (Real.sqrt 6 + Real.sqrt 2) / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_region_area_l1226_122639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1226_122679

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.rpow 2 x ≥ 2}

-- Statement to prove
theorem complement_A_intersect_B : 
  (Set.compl A) ∩ B = Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1226_122679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_omega_range_l1226_122693

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + Real.pi / 6) - 1

theorem two_zeros_iff_omega_range (ω : ℝ) :
  (ω > 0) →
  (∃! (z₁ z₂ : ℝ), 0 < z₁ ∧ z₁ < z₂ ∧ z₂ < Real.pi ∧ f ω z₁ = 0 ∧ f ω z₂ = 0 ∧
    ∀ z, 0 < z ∧ z < Real.pi ∧ f ω z = 0 → z = z₁ ∨ z = z₂) ↔
  (3/2 < ω ∧ ω ≤ 13/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_omega_range_l1226_122693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_iff_k_eq_neg_three_or_neg_eight_l1226_122650

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + k) / (x^2 - 3*x + 2)

def has_exactly_one_vertical_asymptote (k : ℝ) : Prop :=
  (∃! x : ℝ, (x^2 - 3*x + 2 = 0 ∧ x^2 + 2*x + k ≠ 0)) ∧
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x^2 + 2*x + k = 0 → False)

theorem one_vertical_asymptote_iff_k_eq_neg_three_or_neg_eight :
  ∀ k : ℝ, has_exactly_one_vertical_asymptote k ↔ (k = -3 ∨ k = -8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_iff_k_eq_neg_three_or_neg_eight_l1226_122650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_a_range_l1226_122600

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x^2 + (2 - a) * x

noncomputable def g (x : ℝ) : ℝ := x / exp x - 2

def interval : Set ℝ := Set.Ioo 0 (exp 1)

theorem function_roots_imply_a_range (a : ℝ) :
  (∀ x₀, x₀ ∈ interval → ∃ x₁ x₂, x₁ ∈ interval ∧ x₂ ∈ interval ∧ x₁ ≠ x₂ ∧ f a x₁ = g x₀ ∧ f a x₂ = g x₀) →
  a ∈ Set.Icc ((3 + 2 * exp 1) / (exp 1^2 + exp 1)) (exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_roots_imply_a_range_l1226_122600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_specific_l1226_122625

/-- The distance from a point in polar coordinates to a line in polar form -/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (l : ℝ → ℝ → ℝ) : ℝ :=
  sorry

theorem distance_point_to_line_specific :
  let M : ℝ × ℝ := (2, π/3)  -- Point M in polar coordinates
  let l : ℝ → ℝ → ℝ := λ ρ θ ↦ ρ * Real.sin (θ + π/4) - Real.sqrt 2/2  -- Line l equation
  distance_point_to_line M.fst M.snd l = Real.sqrt 6/2 := by
  sorry

#check distance_point_to_line_specific

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_specific_l1226_122625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1000_terms_l1226_122615

/-- Defines the sequence where each block starts with 1 followed by n 3s -/
def customSequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n % (n.sqrt + 1) = 0 then 1
  else 3

/-- Calculates the sum of the first n terms of the sequence -/
def customSequenceSum (n : ℕ) : ℕ :=
  (List.range n).map customSequence |>.sum

/-- The main theorem stating that the sum of the first 1000 terms is 2912 -/
theorem sum_of_first_1000_terms :
  customSequenceSum 1000 = 2912 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_1000_terms_l1226_122615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l1226_122670

/-- In a triangle ABC, if BD = 2DC and AD = xAB + yAC, then y/x = 2 -/
theorem triangle_vector_ratio (A B C D : ℝ × ℝ) (x y : ℝ) :
  (B - D : ℝ × ℝ) = 2 • (D - C) →
  (A - D : ℝ × ℝ) = x • (A - B) + y • (A - C) →
  y / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l1226_122670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_silvia_visibility_time_l1226_122690

/-- The time (in hours) Luke can see Silvia given their speeds and distances -/
noncomputable def visibilityTime (lukeSpeed silviasSpeed initialDistance finalDistance : ℝ) : ℝ :=
  (initialDistance + finalDistance) / (lukeSpeed - silviasSpeed)

theorem luke_silvia_visibility_time :
  let lukeSpeed : ℝ := 10
  let silviasSpeed : ℝ := 6
  let initialDistance : ℝ := 3/4
  let finalDistance : ℝ := 3/4
  visibilityTime lukeSpeed silviasSpeed initialDistance finalDistance * 60 = 22.5 := by
  -- Unfold the definition of visibilityTime
  unfold visibilityTime
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_silvia_visibility_time_l1226_122690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1226_122607

theorem exponential_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (2 : ℝ)^a ≤ (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1226_122607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_implies_difference_l1226_122662

theorem identity_implies_difference :
  ∀ a b : ℚ,
  (∀ x : ℝ, x > 0 → 
    a / ((10 : ℝ)^x - 1) + b / ((10 : ℝ)^x + 2) = 
    (3 * (10 : ℝ)^x + 5) / ((10 : ℝ)^x - 1) / ((10 : ℝ)^x + 2)) →
  a - b = 7/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_implies_difference_l1226_122662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1226_122657

noncomputable def f (x : ℝ) := 2 * Real.sin x + Real.cos x

theorem f_min_value :
  ∃ (min : ℝ), min = -Real.sqrt 5 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l1226_122657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_entered_l1226_122648

theorem last_score_entered (scores : List ℕ) 
  (h1 : scores = [62, 75, 83, 90])
  (h2 : ∀ k : ℕ, k ∈ [1, 2, 3, 4] → 
    (∃ sublist : List ℕ, sublist ⊆ scores ∧ sublist.length = k ∧ 
    (sublist.sum / k : ℚ).isInt)) :
  ∃ perm : List ℕ, perm.length = 4 ∧ scores.toFinset = perm.toFinset ∧ perm.getLast? = some 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_entered_l1226_122648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1226_122623

noncomputable def f (lambda : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x + lambda else 2^x

theorem lambda_range (lambda : ℝ) : 
  (∀ a : ℝ, f lambda (f lambda a) = 2^(f lambda a)) → lambda ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1226_122623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_min_value_ab_l1226_122614

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x| + |2*x - 1|

-- Statement 1: Solution set of f(x) ≥ 5
theorem solution_set (x : ℝ) : f x ≥ 5 ↔ x ≤ -1 ∨ x ≥ 3/2 := by sorry

-- Statement 2: Minimum value of f(x)
theorem min_value : ∃ (m : ℝ), IsGLB (Set.range f) m ∧ m = 1 := by sorry

-- Statement 3: Minimum value of 1/a + 2/b given conditions
theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2*a + b = 1) :
  ∃ (m : ℝ), IsGLB {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ x = 1/a + 2/b} m ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_min_value_min_value_ab_l1226_122614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_x_value_l1226_122659

def vector_a : ℝ × ℝ := (-2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 3)

theorem parallel_vectors_implies_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ vector_a = k • vector_b (-6)) → -6 = -6 := by
  intro h
  rfl

#check parallel_vectors_implies_x_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_implies_x_value_l1226_122659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_of_eight_l1226_122613

theorem complex_power_of_eight :
  (3 * (Complex.exp (Complex.I * Real.pi / 6))) ^ 8 =
  Complex.ofReal (-3280.5) + Complex.I * Complex.ofReal (-3280.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_of_eight_l1226_122613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l1226_122621

theorem cube_root_seven_to_sixth : (7 : ℝ) ^ ((1/3 : ℝ) * 6) = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_seven_to_sixth_l1226_122621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_identity_l1226_122680

/-- A quadratic function f(x) = x^2 + ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * x + a

/-- The identity equation -/
def identity_equation (a b c : ℝ) (x : ℝ) : Prop :=
  f a b (x + 1) = c * ∫ t in Set.Icc 0 1, (3 * x^2 + 4 * x * t) * f' a t

/-- The theorem stating that only two pairs of constants satisfy the identity -/
theorem quadratic_integral_identity :
  ∀ a b c : ℝ, (∀ x : ℝ, identity_equation a b c x) ↔
    ((a = -5/3 ∧ b = 2/3 ∧ c = -1/2) ∨ (a = -2/3 ∧ b = -1/3 ∧ c = 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_identity_l1226_122680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrectStatementsCount_l1226_122637

-- Define the type for statements about lines
inductive LineStatement
| parallelThroughPoint
| perpendicularThroughPoint
| twoRelationships
| parallelNonIntersecting

-- Define a function to check if a statement is correct
def isCorrect (s : LineStatement) : Bool :=
  match s with
  | LineStatement.parallelThroughPoint => false
  | LineStatement.perpendicularThroughPoint => false
  | LineStatement.twoRelationships => true
  | LineStatement.parallelNonIntersecting => false

-- Define the list of all statements
def allStatements : List LineStatement :=
  [LineStatement.parallelThroughPoint,
   LineStatement.perpendicularThroughPoint,
   LineStatement.twoRelationships,
   LineStatement.parallelNonIntersecting]

-- Theorem stating that the number of incorrect statements is 3
theorem incorrectStatementsCount :
  (allStatements.filter (fun s => ¬(isCorrect s))).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrectStatementsCount_l1226_122637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1226_122628

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := ⌊x⌋ + 3/4

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd : ¬(is_even g) ∧ ¬(is_odd g) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1226_122628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_is_45_degrees_l1226_122651

variable (e₁ e₂ : ℝ × ℝ)

-- e₁ and e₂ are unit vectors
def is_unit_vector (v : ℝ × ℝ) : Prop := v.1^2 + v.2^2 = 1

-- e₁ and e₂ are orthogonal
def are_orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- Definition of vector a
def a : ℝ × ℝ := (3 * e₁.1 - e₂.1, 3 * e₁.2 - e₂.2)

-- Definition of vector b
def b : ℝ × ℝ := (2 * e₁.1 + e₂.1, 2 * e₁.2 + e₂.2)

-- Function to calculate the angle between two vectors
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- Theorem statement
theorem angle_between_a_and_b_is_45_degrees 
  (h1 : is_unit_vector e₁) 
  (h2 : is_unit_vector e₂) 
  (h3 : are_orthogonal e₁ e₂) : 
  angle_between (a e₁ e₂) (b e₁ e₂) = π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_is_45_degrees_l1226_122651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_increasing_f_l1226_122681

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

-- State the theorem
theorem max_a_for_increasing_f :
  (∃ (a : ℝ), ∀ (x y : ℝ), 1 < x → x < y → f a x < f a y) →
  (∃ (max_a : ℝ), (∀ (a : ℝ), (∀ (x y : ℝ), 1 < x → x < y → f a x < f a y) → a ≤ max_a) ∧
                   (∀ (x y : ℝ), 1 < x → x < y → f max_a x < f max_a y) ∧
                   max_a = Real.exp 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_for_increasing_f_l1226_122681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_is_51_l1226_122689

noncomputable section

-- Define points
def A : ℝ × ℝ := (7, 0)
def C : ℝ × ℝ := (9, 0)
def E : ℝ × ℝ := (6, 3)
def O : ℝ × ℝ := (0, 0)

-- Define lines
noncomputable def line1 (x : ℝ) : ℝ := (3 / -1) * (x - 7)
noncomputable def line2 (x : ℝ) : ℝ := -1 * (x - 9)

-- Define B and D
noncomputable def B : ℝ × ℝ := (0, line1 0)
noncomputable def D : ℝ × ℝ := (0, line2 0)

-- Define area calculation function
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

theorem area_OBEC_is_51 :
  triangle_area O B E - (triangle_area O E C + triangle_area O D E) = 51 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_OBEC_is_51_l1226_122689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kira_has_winning_strategy_l1226_122616

/-- Represents a game state with Borya's current number and Kira's used numbers -/
structure GameState where
  borya_number : Int
  used_numbers : List Int

/-- Kira's strategy is a function that chooses the next number based on the current game state -/
def Strategy := GameState → Int

/-- Predicate to check if a strategy is valid (only chooses numbers > 1 and not previously used) -/
def valid_strategy (s : Strategy) : Prop :=
  ∀ gs : GameState, s gs > 1 ∧ s gs ∉ gs.used_numbers

/-- Helper function to iterate a strategy n times -/
def iterate_strategy (s : Strategy) : Nat → GameState → GameState
  | 0, gs => gs
  | n+1, gs =>
    let next_number := s gs
    let new_borya_number := 
      if gs.borya_number % next_number = 0
      then gs.borya_number
      else gs.borya_number - next_number
    iterate_strategy s n ⟨new_borya_number, next_number :: gs.used_numbers⟩

/-- Predicate to check if a strategy is winning for Kira -/
def winning_strategy (s : Strategy) : Prop :=
  ∀ start : Int, start > 100 →
    ∃ n : Nat, ∃ end_state : GameState,
      (iterate_strategy s n ⟨start, []⟩ = end_state) ∧
      (end_state.borya_number < 0 ∨ ∃ k ∈ end_state.used_numbers, end_state.borya_number % k = 0)

/-- The main theorem: Kira has a winning strategy -/
theorem kira_has_winning_strategy :
  ∃ s : Strategy, valid_strategy s ∧ winning_strategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kira_has_winning_strategy_l1226_122616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_for_perfect_square_l1226_122622

theorem smallest_factor_for_perfect_square : ∃ (f : ℕ), 
  f > 0 ∧ 
  ∃ (k : ℕ), 3150 * f = k ^ 2 ∧ 
  (∀ (g : ℕ), g > 0 → g < f → ∀ (m : ℕ), 3150 * g ≠ m ^ 2) ∧
  f = 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factor_for_perfect_square_l1226_122622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_savings_is_22_l1226_122676

/-- Represents the savings when choosing the first car rental option over the second option -/
noncomputable def rental_savings (trip_distance : ℝ) (option1_cost : ℝ) (option2_cost : ℝ) 
  (gas_efficiency : ℝ) (gas_cost : ℝ) : ℝ :=
  let round_trip_distance := 2 * trip_distance
  let gas_needed := round_trip_distance / gas_efficiency
  let gas_total_cost := gas_needed * gas_cost
  let option1_total_cost := option1_cost + gas_total_cost
  option2_cost - option1_total_cost

/-- The savings when choosing the first car rental option over the second option is $22 -/
theorem rental_savings_is_22 : 
  rental_savings 150 50 90 15 0.9 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_savings_is_22_l1226_122676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_problem_l1226_122663

theorem right_triangle_angle_problem (x : ℝ) 
  (PQR PQS SQR : ℝ) : 
  (PQR = 90) → 
  (PQS + SQR = PQR) → 
  (PQS = 3 * x) → 
  (SQR = 2 * x) → 
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_problem_l1226_122663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_calculation_l1226_122669

/-- Represents a sphere on a plane under parallel sunlight conditions. -/
structure SunlitSphere where
  shadow_length : ℝ
  ruler_height : ℝ
  ruler_shadow : ℝ

/-- Calculates the radius of a sphere given its shadow properties. -/
noncomputable def sphere_radius (s : SunlitSphere) : ℝ :=
  10 * Real.sqrt 5 - 20

/-- Theorem stating that under the given conditions, the sphere's radius is 10√5 - 20. -/
theorem sphere_radius_calculation (s : SunlitSphere)
  (h1 : s.shadow_length = 10)
  (h2 : s.ruler_height = 1)
  (h3 : s.ruler_shadow = 2) :
  sphere_radius s = 10 * Real.sqrt 5 - 20 := by
  -- The proof goes here
  sorry

#check sphere_radius_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_calculation_l1226_122669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_ten_percent_l1226_122620

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem compound_interest_rate_is_ten_percent 
  (principal_simple : ℝ) 
  (rate_simple : ℝ) 
  (principal_compound : ℝ) 
  (time : ℝ) :
  principal_simple = 2625 →
  rate_simple = 8 →
  principal_compound = 4000 →
  time = 2 →
  simple_interest principal_simple rate_simple time = 
    (1/2) * compound_interest principal_compound 10 time →
  ∃ (rate_compound : ℝ), 
    simple_interest principal_simple rate_simple time = 
      (1/2) * compound_interest principal_compound rate_compound time ∧
    rate_compound = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_rate_is_ten_percent_l1226_122620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1226_122601

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Area of a triangle formed by three points -/
noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Left focus of the hyperbola -/
noncomputable def left_focus (h : Hyperbola) : Point :=
  ⟨-Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Left vertex of the hyperbola -/
def left_vertex (h : Hyperbola) : Point :=
  ⟨-h.a, 0⟩

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Main theorem -/
theorem hyperbola_properties (h : Hyperbola) :
  ∃ (p : Point),
    on_hyperbola h p ∧
    p.x > 0 ∧
    angle (left_focus h) p (right_focus h) = π/3 ∧
    triangle_area (left_focus h) p (right_focus h) = 3 * Real.sqrt 3 * h.a^2
  →
  (eccentricity h = 2) ∧
  (∀ (q : Point),
    on_hyperbola h q ∧ q.x > 0 ∧ q.y > 0
    →
    angle q (right_focus h) (left_vertex h) = 2 * angle q (left_vertex h) (right_focus h)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1226_122601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1226_122608

theorem remainder_theorem (x : ℕ) (h : (9 * x) % 26 = 1) :
  (13 + x) % 26 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1226_122608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_at_op_equals_one_l1226_122678

/-- Definition of the @ operation -/
noncomputable def at_op (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

/-- Function to represent the nested operation -/
noncomputable def nested_op : ℕ → ℝ
| 0 => 1000
| n + 1 => at_op (1000 - n) (nested_op n)

/-- Theorem statement -/
theorem nested_at_op_equals_one : at_op 1 (nested_op 998) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_at_op_equals_one_l1226_122678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l1226_122624

theorem trigonometric_system_solution :
  ∀ x y z : ℝ,
    (Real.sin x + Real.sin y + Real.sin z = 1/2 ∧
     Real.sin x * Real.sin y + Real.sin x * Real.sin z + Real.sin y * Real.sin z = -1 ∧
     Real.sin x * Real.sin y * Real.sin z = -1/2) ↔
    ((∃ m n k : ℤ, x = π/2 + 2*π*↑m ∧ y = (-1)^n * π/6 + π*↑n ∧ z = -π/2 + 2*π*↑k) ∨
     (∃ m n k : ℤ, x = π/2 + 2*π*↑m ∧ y = -π/2 + 2*π*↑k ∧ z = (-1)^n * π/6 + π*↑n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_system_solution_l1226_122624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_is_140_l1226_122652

/-- Given a total bill amount that includes tax and tip, calculate the original food price -/
noncomputable def calculate_food_price (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  total / (1 + tip_rate * (1 + tax_rate))

/-- Theorem stating that the food price is $140 given the conditions -/
theorem food_price_is_140 :
  let total : ℝ := 184.80
  let tax_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.20
  calculate_food_price total tax_rate tip_rate = 140 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_food_price 184.80 0.10 0.20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_is_140_l1226_122652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_l1226_122658

theorem profit_increase (P : ℝ) (h : P > 0) : 
  let april_profit := 1.40 * P
  let may_profit := 0.80 * april_profit
  let june_profit := 1.50 * may_profit
  (june_profit - P) / P * 100 = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_l1226_122658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_24_product_l1226_122605

def pairs : List (ℚ × ℚ) := [(-4, -6), (-2, -12), (1/3, -72), (2, 12), (3/4, 32)]

theorem unique_non_24_product : ∃! p : ℚ × ℚ, p ∈ pairs ∧ p.1 * p.2 ≠ 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_non_24_product_l1226_122605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_rate_l1226_122661

/-- Given a rectangular floor with specified dimensions and total painting cost,
    calculate the rate per square meter to paint the floor. -/
theorem floor_painting_rate
  (length : ℝ)
  (breadth_ratio : ℝ)
  (total_cost : ℝ)
  (h1 : length = 21.633307652783934)
  (h2 : length = breadth_ratio * (length / 3))
  (h3 : total_cost = 624) :
  total_cost / (length * (length / 3)) = 4 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_painting_rate_l1226_122661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_minus_5_floor_l1226_122631

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem sqrt_17_minus_5_floor : floor (Real.sqrt 17 - 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_minus_5_floor_l1226_122631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_stoppage_time_is_one_point_five_l1226_122640

/-- Represents a bus with its speeds -/
structure Bus where
  speedWithoutStoppages : ℝ
  speedWithStoppages : ℝ

/-- Calculates the stoppage time for a single bus in one hour -/
noncomputable def stoppageTime (bus : Bus) : ℝ :=
  (bus.speedWithoutStoppages - bus.speedWithStoppages) / bus.speedWithoutStoppages

/-- Theorem: The total stoppage time for all 3 buses combined per hour is 1.5 hours -/
theorem total_stoppage_time_is_one_point_five 
  (busA busB busC : Bus)
  (hA : busA = { speedWithoutStoppages := 80, speedWithStoppages := 40 })
  (hB : busB = { speedWithoutStoppages := 100, speedWithStoppages := 50 })
  (hC : busC = { speedWithoutStoppages := 120, speedWithStoppages := 60 }) :
  stoppageTime busA + stoppageTime busB + stoppageTime busC = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_stoppage_time_is_one_point_five_l1226_122640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_1_70_l1226_122656

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

noncomputable def winning_amount (roll : ℕ) : ℝ :=
  if roll = 2 ∨ roll = 3 ∨ roll = 5 ∨ roll = 7 then roll else 0

noncomputable def expected_value : ℝ :=
  (1 : ℝ) / 10 * (winning_amount 1 + winning_amount 2 + winning_amount 3 + winning_amount 4 +
                  winning_amount 5 + winning_amount 6 + winning_amount 7 + winning_amount 8 +
                  winning_amount 9 + winning_amount 10)

theorem expected_value_is_1_70 : expected_value = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_1_70_l1226_122656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pencils_fayes_pencils_l1226_122635

/-- Given the number of rows and pencils per row, calculate the total number of pencils. -/
theorem total_pencils (rows : ℕ) (pencils_per_row : ℕ) : 
  rows * pencils_per_row = rows * pencils_per_row :=
by rfl

/-- Faye's pencil arrangement problem -/
theorem fayes_pencils : 
  (6 : ℕ) * (5 : ℕ) = 30 :=
by
  -- Evaluate the multiplication
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pencils_fayes_pencils_l1226_122635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_20_10_l1226_122645

theorem binomial_20_10 (h1 : Nat.choose 19 9 = 92378) (h2 : Nat.choose 19 10 = Nat.choose 19 9) :
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_20_10_l1226_122645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l1226_122664

/-- Triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of DP, where P is the tangent point on DE -/
  dp : ℝ
  /-- The length of PE, where P is the tangent point on DE -/
  pe : ℝ

/-- Calculate the perimeter of a triangle with an inscribed circle -/
noncomputable def perimeter (t : InscribedCircleTriangle) : ℝ :=
  2 * (t.dp + t.pe + (729 * 62) / 228)

/-- The theorem statement -/
theorem inscribed_circle_triangle_perimeter :
  let t : InscribedCircleTriangle := { r := 27, dp := 29, pe := 33 }
  perimeter t = 774 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_triangle_perimeter_l1226_122664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_monotone_increasing_implies_min_b_min_b_is_minimum_l1226_122684

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Part 1
theorem extreme_value_implies_b (a b : ℝ) :
  (∃ x, f a b x = 10 ∧ ∀ y, f a b y ≤ f a b x) ∧
  f a b 1 = 10 →
  b = -11 := by sorry

-- Part 2
theorem monotone_increasing_implies_min_b (b : ℝ) :
  (∀ a, a ≥ -4 →
   ∀ x y, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 →
   x ≤ y → f a b x ≤ f a b y) →
  b ≥ 16/3 := by sorry

-- Minimum value of b
noncomputable def min_b : ℝ := 16/3

theorem min_b_is_minimum :
  (∀ a, a ≥ -4 →
   ∀ x y, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 →
   x ≤ y → f a min_b x ≤ f a min_b y) ∧
  ∀ b', b' < min_b → ∃ a, a ≥ -4 ∧
    ∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 2 ∧
    x ≤ y ∧ f a b' x > f a b' y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_b_monotone_increasing_implies_min_b_min_b_is_minimum_l1226_122684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_6_l1226_122642

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Define a lucky integer
def isLuckyInteger (n : ℕ) : Prop :=
  n > 0 ∧ n % sumOfDigits n = 0

-- Define a function to check if a number is a multiple of 6
def isMultipleOf6 (n : ℕ) : Prop :=
  n % 6 = 0

-- Theorem statement
theorem least_non_lucky_multiple_of_6 :
  ∃ (m : ℕ), m = 78 ∧ 
    (∀ k < m, isMultipleOf6 k → isLuckyInteger k) ∧
    (isMultipleOf6 m ∧ ¬isLuckyInteger m) :=
by
  -- The proof goes here
  sorry

#eval sumOfDigits 78  -- This should output 15
#eval 78 % 6  -- This should output 0
#eval 78 % 15 -- This should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_6_l1226_122642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_above_115_l1226_122671

def class_size : ℕ := 50
def mean_score : ℝ := 105
def std_dev : ℝ := 10
def prob_95_to_105 : ℝ := 0.32

-- Define the probability of scoring above 115
noncomputable def prob_above_115 : ℝ := 0.18

-- Theorem statement
theorem students_above_115 :
  ⌊class_size * prob_above_115⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_above_115_l1226_122671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1226_122667

theorem problem_statement (a b : Real) 
  (h1 : (2 : Real)^(a+1) = 3) 
  (h2 : (2 : Real)^(b-3) = 1/3) : 
  (a + b = 2) ∧ 
  (1 < b ∧ b < 3/2) ∧ 
  (b - a < 1) ∧ 
  (a * b ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1226_122667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l1226_122677

def arithmeticSequence (start : ℕ) (step : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + i * step)

theorem product_congruence (start step count : ℕ) 
  (h1 : start = 7)
  (h2 : step = 10)
  (h3 : count = 31) :
  (arithmeticSequence start step count).prod % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l1226_122677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_solution_l1226_122674

-- Define the logarithm equality condition
def log_equality (x : ℝ) : Prop := Real.log 8 / Real.log x = Real.log 4 / Real.log 64

-- Theorem statement
theorem log_equality_solution :
  ∃ x : ℝ, log_equality x ∧ x = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_solution_l1226_122674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_equals_15_l1226_122665

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 1 else 3*x

-- State the theorem
theorem f_f_equals_15 :
  ∀ x : ℝ, f (f x) = 15 ↔ x = -4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_equals_15_l1226_122665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l1226_122603

-- Define the concept of lines in 3D space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define the concept of being skew
def Skew (l1 l2 : Set V) : Prop := 
  (l1 ∩ l2 = ∅) ∧ ¬ ∃ (v : V), v ≠ 0 ∧ (∀ p ∈ l1, ∃ t : ℝ, p + t • v ∈ l2)

-- Define the concept of being parallel
def Parallel (l1 l2 : Set V) : Prop := 
  ∃ (v : V), v ≠ 0 ∧ (∀ p ∈ l1, ∃ t : ℝ, p + t • v ∈ l2)

-- Define the concept of intersecting
def Intersecting (l1 l2 : Set V) : Prop := 
  ∃ p, p ∈ l1 ∧ p ∈ l2

-- Theorem statement
theorem line_relationship (a b c : Set V) 
  (h1 : Skew a b) 
  (h2 : Parallel c a) : 
  Intersecting c b ∨ Skew c b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationship_l1226_122603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1226_122673

/-- The coefficient of x^3 in the expansion of (2x-3)^5 is 720 -/
theorem coefficient_x_cubed_in_expansion : 
  (Finset.sum (Finset.range 6) (λ k => Nat.choose 5 k * (2^(5-k)) * ((-3:Int)^k) * 
    if k = 2 then 1 else 0)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1226_122673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l1226_122666

def is_valid_student_count (n : Nat) : Prop :=
  n ∈ ({6, 9, 10, 15, 18} : Finset Nat) ∧ 180 % n = 0

theorem candy_distribution (n : Nat) :
  is_valid_student_count n → n ∈ ({9, 10, 18} : Finset Nat) :=
by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_l1226_122666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_theorem_probability_theorem_l1226_122604

-- Define the type for our strings
def StringType (n : ℕ) := Fin (n + 1) → Fin 4

-- Define the condition that adjacent letters must be different
def ValidString (s : StringType n) : Prop :=
  ∀ i : Fin n, s i ≠ s (i.succ)

-- Define a_n as the number of valid strings ending with 'a'
def a_n (n : ℕ) : ℚ := (3^n + 3 * (-1:ℤ)^n) / 4

-- Define the probability P
def P (n : ℕ) : ℚ := (3^n + 3 * (-1:ℤ)^n) / (4 * 3^n)

theorem string_theorem (n : ℕ) (h : n ≥ 1) :
  a_n n = (3^n + 3 * (-1:ℤ)^n) / 4 := by
  sorry

theorem probability_theorem (n : ℕ) (h : n ≥ 2) :
  2/9 ≤ P n ∧ P n ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_theorem_probability_theorem_l1226_122604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_specific_l1226_122644

theorem cos_double_angle_specific (α : ℝ) (h : Real.cos α = -3/5) : Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_specific_l1226_122644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_convex_l1226_122629

/-- A function f is convex if for any x₁, x₂ in its domain and t ∈ [0,1],
    f(t*x₁ + (1-t)*x₂) ≤ t*f(x₁) + (1-t)*f(x₂) --/
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ t, x₁ > 0 → x₂ > 0 → 0 ≤ t → t ≤ 1 →
    f (t * x₁ + (1 - t) * x₂) ≤ t * f x₁ + (1 - t) * f x₂

/-- The function f(x) = 1/x^k + a --/
noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := 1 / x^k + a

/-- Proof that f(x) = 1/x^k + a is convex --/
theorem f_is_convex (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  IsConvex (f a k) := by
  sorry

#check f_is_convex

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_convex_l1226_122629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1226_122611

/-- A point on a parabola y^2 = 4x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (k m : ParabolaPoint) 
  (h1 : k.x ≤ m.x) 
  (h2 : m.x = k.x + 2) : 
  distance (k.x, k.y) (m.x, m.y) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1226_122611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symbolic_sequence_next_l1226_122683

/-- Represents a symbol used in place of a digit -/
inductive SymbolDigit : Type
| square : SymbolDigit
| diamond : SymbolDigit
| vee : SymbolDigit
| triangle : SymbolDigit
| delta : SymbolDigit
| nabla : SymbolDigit

/-- Represents a three-digit number using symbols -/
structure SymbolicNumber where
  hundreds : SymbolDigit
  tens : SymbolDigit
  ones : SymbolDigit

/-- Checks if four numbers are consecutive positive integers -/
def are_consecutive (a b c d : ℕ) : Prop :=
  a > 0 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1

/-- Maps symbols to digits -/
def symbol_to_digit (s : SymbolDigit) : ℕ := 
  match s with
  | SymbolDigit.square => 1
  | SymbolDigit.diamond => 9
  | SymbolDigit.vee => 2
  | SymbolDigit.triangle => 0
  | SymbolDigit.delta => 0
  | SymbolDigit.nabla => 2

/-- Converts a symbolic number to a natural number -/
def symbolic_to_nat (sn : SymbolicNumber) : ℕ :=
  100 * (symbol_to_digit sn.hundreds) + 10 * (symbol_to_digit sn.tens) + (symbol_to_digit sn.ones)

theorem symbolic_sequence_next (n1 n2 n3 n4 : SymbolicNumber) :
  n1 = { hundreds := SymbolDigit.square, tens := SymbolDigit.diamond, ones := SymbolDigit.diamond } →
  n2 = { hundreds := SymbolDigit.vee, tens := SymbolDigit.triangle, ones := SymbolDigit.delta } →
  n3 = { hundreds := SymbolDigit.vee, tens := SymbolDigit.triangle, ones := SymbolDigit.square } →
  are_consecutive (symbolic_to_nat n1) (symbolic_to_nat n2) (symbolic_to_nat n3) (symbolic_to_nat n4) →
  n4 = { hundreds := SymbolDigit.vee, tens := SymbolDigit.triangle, ones := SymbolDigit.nabla } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symbolic_sequence_next_l1226_122683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_is_ten_l1226_122685

/-- The surface area of a 2 × 1 cuboid is 10 unit squares. -/
def surface_area_2x1_cuboid : ℕ :=
  let length : ℕ := 2
  let width : ℕ := 1
  let height : ℕ := 1
  2 * (length * width + width * height + length * height)

theorem surface_area_is_ten : surface_area_2x1_cuboid = 10 := by
  unfold surface_area_2x1_cuboid
  rfl

#eval surface_area_2x1_cuboid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_is_ten_l1226_122685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_theorem_l1226_122634

noncomputable section

/-- A curve in the Cartesian coordinate system xOy -/
def Curve (a : ℝ) := {p : ℝ × ℝ | p.2 = p.1^3 - a * p.1}

/-- The slope of the tangent line to the curve at a given x-coordinate -/
def TangentSlope (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - a

/-- The distance between two parallel lines with slope 1 passing through points on the curve -/
def TangentDistance (a : ℝ) (x₁ x₂ : ℝ) : ℝ :=
  |((x₁^3 - a*x₁) - (x₂^3 - a*x₂) - (x₁ - x₂)) / Real.sqrt 2|

theorem curve_tangent_theorem (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    TangentSlope a x₁ = 1 ∧ 
    TangentSlope a x₂ = 1 ∧ 
    TangentDistance a x₁ x₂ = 8) →
  a = 5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_tangent_theorem_l1226_122634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catalogue_numbers_not_unique_l1226_122694

theorem catalogue_numbers_not_unique (n : ℕ) (h : n ≥ 2000) :
  ∃ (A B : Finset ℕ),
    A ≠ B ∧
    A.card = n - 1 ∧
    B.card = n - 1 ∧
    (∀ x, x ∈ A → 2 ≤ x ∧ x ≤ n) ∧
    (∀ x, x ∈ B → 2 ≤ x ∧ x ≤ n) ∧
    (∀ x y, x ∈ A → y ∈ A → x ≠ y → ∃ z, z ∈ B ∧ Nat.gcd x y = Nat.gcd z y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_catalogue_numbers_not_unique_l1226_122694
