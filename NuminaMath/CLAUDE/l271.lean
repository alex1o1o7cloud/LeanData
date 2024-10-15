import Mathlib

namespace NUMINAMATH_CALUDE_outfit_combinations_l271_27179

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 6
def number_of_belts : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_belts = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l271_27179


namespace NUMINAMATH_CALUDE_function_properties_l271_27194

def f (a x : ℝ) : ℝ := |2*x + a| + |x - 1|

theorem function_properties :
  (∀ x : ℝ, f 3 x < 6 ↔ -8/3 < x ∧ x < 4/3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x + f a (-x) ≥ 5) ↔ a ≤ -3/2 ∨ a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l271_27194


namespace NUMINAMATH_CALUDE_total_weekly_egg_supply_l271_27169

/-- Represents the daily egg supply to a store -/
structure DailySupply where
  oddDays : ℕ
  evenDays : ℕ

/-- Calculates the total eggs supplied to a store in a week -/
def weeklySupply (supply : DailySupply) : ℕ :=
  4 * supply.oddDays + 3 * supply.evenDays

/-- Converts dozens to individual eggs -/
def dozensToEggs (dozens : ℕ) : ℕ :=
  dozens * 12

theorem total_weekly_egg_supply :
  let store1 := DailySupply.mk (dozensToEggs 5) (dozensToEggs 5)
  let store2 := DailySupply.mk 30 30
  let store3 := DailySupply.mk (dozensToEggs 25) (dozensToEggs 15)
  weeklySupply store1 + weeklySupply store2 + weeklySupply store3 = 2370 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_egg_supply_l271_27169


namespace NUMINAMATH_CALUDE_return_trip_duration_l271_27185

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  outbound_time : ℝ  -- time for outbound trip (against wind)
  still_air_time : ℝ  -- time for return trip in still air

/-- The theorem stating the return trip duration --/
theorem return_trip_duration (fs : FlightScenario) : 
  fs.outbound_time = 120 →  -- Condition 1
  fs.d = fs.outbound_time * (fs.p - fs.w) →  -- Derived from Condition 1
  fs.still_air_time = fs.d / fs.p →  -- Definition of still air time
  fs.d / (fs.p + fs.w) = fs.still_air_time - 15 →  -- Condition 3
  (fs.d / (fs.p + fs.w) = 15 ∨ fs.d / (fs.p + fs.w) = 85) :=
by sorry

#check return_trip_duration

end NUMINAMATH_CALUDE_return_trip_duration_l271_27185


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l271_27177

/-- Given a geometric sequence where the fifth term is 45 and the sixth term is 60,
    prove that the first term is 1215/256. -/
theorem geometric_sequence_first_term
  (a : ℚ)  -- First term of the sequence
  (r : ℚ)  -- Common ratio of the sequence
  (h1 : a * r^4 = 45)  -- Fifth term is 45
  (h2 : a * r^5 = 60)  -- Sixth term is 60
  : a = 1215 / 256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l271_27177


namespace NUMINAMATH_CALUDE_different_chord_length_l271_27136

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the chord length for a line y = ax + b on the ellipse
noncomputable def chordLength (m a b : ℝ) : ℝ :=
  let A := 4 + m * a^2
  let B := 2 * m * a
  let C := m * (b^2 - 1)
  Real.sqrt ((B^2 - 4*A*C) / A^2)

-- Theorem statement
theorem different_chord_length (k m : ℝ) (hm : m > 0) :
  chordLength m k 1 ≠ chordLength m (-k) 2 :=
sorry

end NUMINAMATH_CALUDE_different_chord_length_l271_27136


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l271_27153

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  m ∈ Set.Ioo (-Real.sqrt 2 / 2) 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l271_27153


namespace NUMINAMATH_CALUDE_unique_positive_solution_l271_27125

theorem unique_positive_solution : ∃! (y : ℝ), y > 0 ∧ (y / 100) * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l271_27125


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l271_27172

/-- The number of jelly beans remaining in the container after distribution --/
def remaining_jelly_beans (initial : ℕ) (people : ℕ) (first_group : ℕ) (last_group : ℕ) (last_group_takes : ℕ) : ℕ :=
  initial - (first_group * (2 * last_group_takes) + last_group * last_group_takes)

/-- Theorem stating the number of remaining jelly beans --/
theorem jelly_bean_problem :
  remaining_jelly_beans 8000 10 6 4 400 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l271_27172


namespace NUMINAMATH_CALUDE_cookies_for_guests_l271_27161

/-- Given the total number of cookies and cookies per guest, calculate the number of guests. -/
def number_of_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem stating that the number of guests is 2 when there are 38 total cookies and 19 cookies per guest. -/
theorem cookies_for_guests : number_of_guests 38 19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_guests_l271_27161


namespace NUMINAMATH_CALUDE_odd_function_property_l271_27152

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : OddFunction f) (h_fa : f a = 11) : f (-a) = -11 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l271_27152


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l271_27126

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem to be proved --/
theorem parallel_lines_condition (a : ℝ) :
  (parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩ → a = 1) ∧
  ¬(a = 1 → parallel ⟨a, 1, -1⟩ ⟨1, a, 2⟩) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l271_27126


namespace NUMINAMATH_CALUDE_painted_stripe_area_l271_27100

/-- The area of a painted stripe on a cylindrical tank -/
theorem painted_stripe_area (d h w1 w2 r1 r2 : ℝ) (hd : d = 40) (hh : h = 100) 
  (hw1 : w1 = 5) (hw2 : w2 = 7) (hr1 : r1 = 3) (hr2 : r2 = 3) : 
  w1 * (π * d * r1) + w2 * (π * d * r2) = 1440 * π := by
  sorry

end NUMINAMATH_CALUDE_painted_stripe_area_l271_27100


namespace NUMINAMATH_CALUDE_local_minimum_value_inequality_condition_l271_27157

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Theorem for part (1)
theorem local_minimum_value (a : ℝ) :
  (∃ k, ∀ x, x ≠ 1 → (f a x - f a 1) / (x - 1) = k) →
  (∃ x₀, ∀ x, x ≠ x₀ → f a x > f a x₀) →
  f a x₀ = -Real.log 2 - 5/4 := by sorry

-- Theorem for part (2)
theorem inequality_condition (x₁ x₂ m : ℝ) :
  1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 →
    f 1 x₁ - f 1 x₂ > m * (x₂ - x₁) / (x₁ * x₂)) ↔
  m ≤ -6 := by sorry

end NUMINAMATH_CALUDE_local_minimum_value_inequality_condition_l271_27157


namespace NUMINAMATH_CALUDE_complex_division_l271_27113

theorem complex_division (i : ℂ) : i^2 = -1 → (2 : ℂ) / (1 + i) = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l271_27113


namespace NUMINAMATH_CALUDE_extended_triangles_similarity_l271_27170

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  x : ℝ
  y : ℝ

/-- Represents a triangle in the complex plane -/
structure Triangle where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint

/-- Extends a side of a triangle by a factor k -/
def extendSide (A B : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (B.x - A.x),
    y := A.y + k * (B.y - A.y) }

/-- Extends an altitude of a triangle by a factor k -/
def extendAltitude (A B C : ComplexPoint) (k : ℝ) : ComplexPoint :=
  { x := A.x + k * (C.y - B.y),
    y := A.y - k * (C.x - B.x) }

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
    (T1.B.x - T1.A.x)^2 + (T1.B.y - T1.A.y)^2 = r * ((T2.B.x - T2.A.x)^2 + (T2.B.y - T2.A.y)^2) ∧
    (T1.C.x - T1.B.x)^2 + (T1.C.y - T1.B.y)^2 = r * ((T2.C.x - T2.B.x)^2 + (T2.C.y - T2.B.y)^2) ∧
    (T1.A.x - T1.C.x)^2 + (T1.A.y - T1.C.y)^2 = r * ((T2.A.x - T2.C.x)^2 + (T2.A.y - T2.C.y)^2)

theorem extended_triangles_similarity (ABC : Triangle) :
  ∃ (k : ℝ), k > 1 ∧
    let P := extendSide ABC.A ABC.B k
    let Q := extendSide ABC.B ABC.C k
    let R := extendSide ABC.C ABC.A k
    let A' := extendAltitude ABC.A ABC.B ABC.C k
    let B' := extendAltitude ABC.B ABC.C ABC.A k
    let C' := extendAltitude ABC.C ABC.A ABC.B k
    areSimilar
      { A := P, B := Q, C := R }
      { A := A', B := B', C := C' } :=
by sorry

end NUMINAMATH_CALUDE_extended_triangles_similarity_l271_27170


namespace NUMINAMATH_CALUDE_video_call_cost_proof_l271_27188

/-- Calculates the cost of a video call given the charge rate and duration. -/
def video_call_cost (charge_rate : ℕ) (charge_interval : ℕ) (duration : ℕ) : ℕ :=
  (duration / charge_interval) * charge_rate

/-- Proves that a 2 minute and 40 second video call costs 480 won at a rate of 30 won per 10 seconds. -/
theorem video_call_cost_proof :
  let charge_rate : ℕ := 30
  let charge_interval : ℕ := 10
  let duration : ℕ := 2 * 60 + 40
  video_call_cost charge_rate charge_interval duration = 480 := by
  sorry

#eval video_call_cost 30 10 (2 * 60 + 40)

end NUMINAMATH_CALUDE_video_call_cost_proof_l271_27188


namespace NUMINAMATH_CALUDE_parallel_transitivity_l271_27159

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  Parallel a c → Parallel b c → Parallel a b := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l271_27159


namespace NUMINAMATH_CALUDE_excircle_tangency_triangle_ratio_l271_27115

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the excircle
structure Excircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the tangency triangle
structure TangencyTriangle :=
  (area : ℝ)

-- Define the theorem
theorem excircle_tangency_triangle_ratio
  (ABC : Triangle)
  (ωA ωB ωC : Excircle)
  (TA TB TC : TangencyTriangle)
  (h1 : TA.area = 4)
  (h2 : TB.area = 5)
  (h3 : TC.area = 6) :
  ∃ (k : ℝ), k > 0 ∧ ABC.a = 15 * k ∧ ABC.b = 12 * k ∧ ABC.c = 10 * k :=
sorry

end NUMINAMATH_CALUDE_excircle_tangency_triangle_ratio_l271_27115


namespace NUMINAMATH_CALUDE_expression_simplification_l271_27198

theorem expression_simplification (a : ℝ) (ha : a = 2018) :
  (a^2 - 3*a) / (a^2 + a) / ((a - 3) / (a^2 - 1)) * ((a + 1) / (a - 1)) = a := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l271_27198


namespace NUMINAMATH_CALUDE_right_triangle_properties_l271_27196

/-- A triangle with side lengths 13, 84, and 85 is a right triangle with area 546, semiperimeter 91, and inradius 6 -/
theorem right_triangle_properties : ∃ (a b c : ℝ), 
  a = 13 ∧ b = 84 ∧ c = 85 ∧
  a^2 + b^2 = c^2 ∧
  (1/2 * a * b : ℝ) = 546 ∧
  ((a + b + c) / 2 : ℝ) = 91 ∧
  (546 / 91 : ℝ) = 6 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_properties_l271_27196


namespace NUMINAMATH_CALUDE_eleven_one_base_three_is_perfect_square_l271_27123

/-- Represents a number in a given base --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Checks if a number is a perfect square --/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- The main theorem --/
theorem eleven_one_base_three_is_perfect_square :
  isPerfectSquare (toDecimal [1, 1, 1, 1, 1] 3) := by
  sorry

end NUMINAMATH_CALUDE_eleven_one_base_three_is_perfect_square_l271_27123


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l271_27119

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ P : ℝ × ℝ, P.1 = -c ∧ (P.2 = b^2 / a ∨ P.2 = -b^2 / a)) →
  (Real.arctan ((2 * c) / (b^2 / a)) = π / 3) →
  c / a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l271_27119


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l271_27183

theorem like_terms_exponent_difference (m n : ℤ) : 
  (∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ 4 * x^(2*m+2) * y^(n-1) = -3 * x^(3*m+1) * y^(3*n-5)) → 
  m - n = -1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l271_27183


namespace NUMINAMATH_CALUDE_expression_value_l271_27182

theorem expression_value : (-1/2)^2023 * 2^2024 = -2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l271_27182


namespace NUMINAMATH_CALUDE_mothers_age_l271_27168

/-- Proves that the mother's age this year is 39 years old -/
theorem mothers_age (sons_age : ℕ) (mothers_age : ℕ) : mothers_age = 39 :=
  by
  -- Define the son's current age
  have h1 : sons_age = 12 := by sorry
  
  -- Define the relationship between mother's and son's ages three years ago
  have h2 : mothers_age - 3 = 4 * (sons_age - 3) := by sorry
  
  -- Prove that the mother's age is 39
  sorry


end NUMINAMATH_CALUDE_mothers_age_l271_27168


namespace NUMINAMATH_CALUDE_blouse_cost_l271_27145

/-- Given information about Jane's purchase of skirts and blouses, prove the cost of each blouse. -/
theorem blouse_cost (num_skirts : ℕ) (skirt_price : ℕ) (num_blouses : ℕ) (total_paid : ℕ) (change : ℕ) :
  num_skirts = 2 →
  skirt_price = 13 →
  num_blouses = 3 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - num_skirts * skirt_price) / num_blouses = 6 := by
  sorry

#eval (100 - 56 - 2 * 13) / 3

end NUMINAMATH_CALUDE_blouse_cost_l271_27145


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l271_27193

theorem inequality_and_equality_condition (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) : 
  (1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) ≥ 2^((n : ℕ) + 1) ∧ 
  ((1 + a / b)^(n : ℕ) + (1 + b / a)^(n : ℕ) = 2^((n : ℕ) + 1) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l271_27193


namespace NUMINAMATH_CALUDE_problem_solution_l271_27116

def A : Set ℝ := {x : ℝ | x^2 + 3*x - 28 < 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 2 < x ∧ x < m + 1}

theorem problem_solution (m : ℝ) :
  (3 ∈ B m → 2 < m ∧ m < 5) ∧
  (B m ⊂ A → -5 ≤ m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l271_27116


namespace NUMINAMATH_CALUDE_chairs_to_remove_l271_27192

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_participants : ℕ)
  (h1 : initial_chairs = 196)
  (h2 : chairs_per_row = 14)
  (h3 : expected_participants = 120)
  (h4 : chairs_per_row > 0) :
  let remaining_chairs := ((expected_participants + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - remaining_chairs = 70 := by
sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l271_27192


namespace NUMINAMATH_CALUDE_kid_tickets_sold_l271_27187

/-- Proves the number of kid tickets sold given ticket prices, total tickets, and profit -/
theorem kid_tickets_sold 
  (adult_price : ℕ) 
  (kid_price : ℕ) 
  (total_tickets : ℕ) 
  (total_profit : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : total_tickets = 175)
  (h4 : total_profit = 750) :
  ∃ (adult_tickets kid_tickets : ℕ), 
    adult_tickets + kid_tickets = total_tickets ∧
    adult_price * adult_tickets + kid_price * kid_tickets = total_profit ∧
    kid_tickets = 75 :=
by sorry

end NUMINAMATH_CALUDE_kid_tickets_sold_l271_27187


namespace NUMINAMATH_CALUDE_sum_of_fractions_l271_27184

theorem sum_of_fractions : (1 : ℚ) / 2 + (1 : ℚ) / 4 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l271_27184


namespace NUMINAMATH_CALUDE_parabola_vertex_on_line_l271_27162

/-- The value of d for which the vertex of the parabola y = x^2 - 10x + d lies on the line y = 2x --/
theorem parabola_vertex_on_line (d : ℝ) : 
  (∃ x y : ℝ, y = x^2 - 10*x + d ∧ 
              y = 2*x ∧ 
              ∀ t : ℝ, (t^2 - 10*t + d) ≥ (x^2 - 10*x + d)) → 
  d = 35 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_line_l271_27162


namespace NUMINAMATH_CALUDE_paperclip_theorem_l271_27121

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the day of the week after n days from Monday -/
def dayAfter (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Monday
  | 1 => DayOfWeek.Tuesday
  | 2 => DayOfWeek.Wednesday
  | 3 => DayOfWeek.Thursday
  | 4 => DayOfWeek.Friday
  | 5 => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday

/-- Number of paperclips after n doublings -/
def paperclips (n : ℕ) : ℕ := 5 * 2^n

theorem paperclip_theorem :
  (∃ n : ℕ, paperclips n > 200 ∧ paperclips (n-1) ≤ 200) ∧
  (∀ n : ℕ, paperclips n > 200 → n ≥ 6) ∧
  dayAfter 12 = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_paperclip_theorem_l271_27121


namespace NUMINAMATH_CALUDE_smallest_valid_m_l271_27130

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 3 / 2 ∧ Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1}

def is_valid_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = Complex.I

theorem smallest_valid_m : 
  (is_valid_m 6) ∧ (∀ m : ℕ, m < 6 → ¬(is_valid_m m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l271_27130


namespace NUMINAMATH_CALUDE_symmetric_line_l271_27150

/-- Given a line l and another line, find the equation of the line symmetric to the given line with respect to l -/
theorem symmetric_line (a b c d e f : ℝ) :
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let given_line : ℝ → ℝ := λ x => x - 2
  let symmetric_line : ℝ → ℝ := λ x => -7 * x - 22
  (∀ x, given_line x = x - (l x)) →
  (∀ x, symmetric_line x = (l x) - (given_line x - (l x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l271_27150


namespace NUMINAMATH_CALUDE_expression_equality_l271_27190

theorem expression_equality : 
  Real.sqrt 12 + |Real.sqrt 3 - 2| + 3 - (Real.pi - 3.14)^0 = Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l271_27190


namespace NUMINAMATH_CALUDE_maria_final_amount_l271_27108

def salary : ℝ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def utility_rate : ℝ := 0.25

def remaining_after_deductions : ℝ := salary * (1 - tax_rate - insurance_rate)
def utility_bill : ℝ := remaining_after_deductions * utility_rate
def final_amount : ℝ := remaining_after_deductions - utility_bill

theorem maria_final_amount : final_amount = 1125 := by
  sorry

end NUMINAMATH_CALUDE_maria_final_amount_l271_27108


namespace NUMINAMATH_CALUDE_fraction_of_students_with_B_l271_27164

theorem fraction_of_students_with_B (fraction_A : Real) (fraction_A_or_B : Real) 
  (h1 : fraction_A = 0.7)
  (h2 : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_students_with_B_l271_27164


namespace NUMINAMATH_CALUDE_gcd_7392_15015_l271_27140

theorem gcd_7392_15015 : Nat.gcd 7392 15015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7392_15015_l271_27140


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l271_27141

/-- Given an ellipse with equation x²/a² + y²/b² = 1 (a > b > 0), 
    where its right focus is at (1,0) and b²/a = 2, 
    prove that the length of its major axis is 2√2 + 2. -/
theorem ellipse_major_axis_length 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : b^2 / a = 2) : 
  2 * a = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l271_27141


namespace NUMINAMATH_CALUDE_abc_sum_l271_27107

theorem abc_sum (a b c : ℕ) : 
  (10 ≤ a ∧ a < 100) → 
  (10 ≤ b ∧ b < 100) → 
  (10 ≤ c ∧ c < 100) → 
  a < b → 
  b < c → 
  a * b * c = 3960 → 
  Even (a + b + c) → 
  a + b + c = 50 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l271_27107


namespace NUMINAMATH_CALUDE_infinite_greater_than_index_l271_27156

theorem infinite_greater_than_index :
  ∀ (a : ℕ → ℕ), (∀ n, a n ≠ 1) →
  ¬ (∃ N, ∀ n > N, a n ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_infinite_greater_than_index_l271_27156


namespace NUMINAMATH_CALUDE_solve_chicken_problem_l271_27174

def chicken_problem (chicken_cost total_spent potato_cost : ℕ) : Prop :=
  chicken_cost > 0 ∧
  total_spent > potato_cost ∧
  (total_spent - potato_cost) % chicken_cost = 0 ∧
  (total_spent - potato_cost) / chicken_cost = 3

theorem solve_chicken_problem :
  chicken_problem 3 15 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_chicken_problem_l271_27174


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l271_27118

theorem sugar_profit_percentage (total_sugar : ℝ) (sugar_at_12_percent : ℝ) (overall_profit_percent : ℝ) :
  total_sugar = 1600 →
  sugar_at_12_percent = 1200 →
  overall_profit_percent = 11 →
  let remaining_sugar := total_sugar - sugar_at_12_percent
  let profit_12_percent := sugar_at_12_percent * 12 / 100
  let total_profit := total_sugar * overall_profit_percent / 100
  let remaining_profit := total_profit - profit_12_percent
  remaining_profit / remaining_sugar * 100 = 8 :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l271_27118


namespace NUMINAMATH_CALUDE_purely_imaginary_product_l271_27158

theorem purely_imaginary_product (x : ℝ) : 
  (Complex.I : ℂ).im * ((x + 2 * Complex.I) * ((x + 3) + 2 * Complex.I) * ((x + 5) + 2 * Complex.I)).re = 0 ↔ 
  x = -5 ∨ x = -4 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_product_l271_27158


namespace NUMINAMATH_CALUDE_cereal_box_cups_l271_27178

/-- Calculates the total number of cups in a cereal box -/
def total_cups (servings : ℕ) (cups_per_serving : ℕ) : ℕ :=
  servings * cups_per_serving

/-- Theorem: A cereal box with 9 servings and 2 cups per serving contains 18 cups of cereal -/
theorem cereal_box_cups : total_cups 9 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_cups_l271_27178


namespace NUMINAMATH_CALUDE_power_of_product_l271_27180

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l271_27180


namespace NUMINAMATH_CALUDE_soda_price_proof_l271_27120

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.8

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 34.56

theorem soda_price_proof : 
  (discounted_price * 72 = total_price) → regular_price = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_proof_l271_27120


namespace NUMINAMATH_CALUDE_hexagon_side_length_l271_27111

/-- A regular hexagon with six segments drawn inside -/
structure SegmentedHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The lengths of the six segments -/
  segment_lengths : Fin 6 → ℝ
  /-- The segments are drawn sequentially with right angles between them -/
  segments_right_angled : Bool
  /-- The segments have lengths from 1 to 6 -/
  segment_lengths_valid : ∀ i, segment_lengths i = (i : ℝ) + 1

/-- The theorem stating that the side length of the hexagon is 15/2 -/
theorem hexagon_side_length (h : SegmentedHexagon) : h.side_length = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_side_length_l271_27111


namespace NUMINAMATH_CALUDE_factorial_equality_l271_27167

theorem factorial_equality : 6 * 8 * 3 * 280 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l271_27167


namespace NUMINAMATH_CALUDE_max_subway_employees_l271_27195

theorem max_subway_employees (total_employees : ℕ) 
  (h_total : total_employees = 48) 
  (part_time full_time : ℕ) 
  (h_sum : part_time + full_time = total_employees)
  (subway_part_time subway_full_time : ℕ)
  (h_part_time : subway_part_time * 3 ≤ part_time)
  (h_full_time : subway_full_time * 4 ≤ full_time) :
  subway_part_time + subway_full_time ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_subway_employees_l271_27195


namespace NUMINAMATH_CALUDE_divisibility_cycle_l271_27166

theorem divisibility_cycle (a b c : ℕ+) : 
  (∃ k₁ : ℕ, (2^(a:ℕ) - 1) = k₁ * (b:ℕ)) ∧
  (∃ k₂ : ℕ, (2^(b:ℕ) - 1) = k₂ * (c:ℕ)) ∧
  (∃ k₃ : ℕ, (2^(c:ℕ) - 1) = k₃ * (a:ℕ)) →
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_cycle_l271_27166


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l271_27171

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l271_27171


namespace NUMINAMATH_CALUDE_vertical_stripe_percentage_is_ten_percent_l271_27109

/-- Represents the distribution of shirt types in a college cafeteria. -/
structure ShirtDistribution where
  total : Nat
  checkered : Nat
  polkaDotted : Nat
  plain : Nat
  horizontalMultiplier : Nat

/-- Calculates the percentage of people wearing vertical stripes. -/
def verticalStripePercentage (d : ShirtDistribution) : Rat :=
  let stripes := d.total - (d.checkered + d.polkaDotted + d.plain)
  let horizontal := d.checkered * d.horizontalMultiplier
  let vertical := stripes - horizontal
  (vertical : Rat) / d.total * 100

/-- Theorem stating that the percentage of people wearing vertical stripes is 10%. -/
theorem vertical_stripe_percentage_is_ten_percent : 
  let d : ShirtDistribution := {
    total := 100,
    checkered := 12,
    polkaDotted := 15,
    plain := 3,
    horizontalMultiplier := 5
  }
  verticalStripePercentage d = 10 := by sorry

end NUMINAMATH_CALUDE_vertical_stripe_percentage_is_ten_percent_l271_27109


namespace NUMINAMATH_CALUDE_mars_visibility_time_l271_27137

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 ≥ totalMinutes1 then
    totalMinutes2 - totalMinutes1
  else
    (24 * 60) - (totalMinutes1 - totalMinutes2)

/-- Subtracts a given number of minutes from a time -/
def subtractMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes
  let newTotalMinutes := (totalMinutes + 24 * 60 - m) % (24 * 60)
  { hours := newTotalMinutes / 60,
    minutes := newTotalMinutes % 60,
    h_valid := by sorry }

theorem mars_visibility_time 
  (jupiter_after_mars : ℕ) 
  (uranus_after_jupiter : ℕ)
  (uranus_appearance : Time)
  (h1 : jupiter_after_mars = 2 * 60 + 41)
  (h2 : uranus_after_jupiter = 3 * 60 + 16)
  (h3 : uranus_appearance = { hours := 6, minutes := 7, h_valid := by sorry }) :
  let jupiter_time := subtractMinutes uranus_appearance uranus_after_jupiter
  let mars_time := subtractMinutes jupiter_time jupiter_after_mars
  mars_time = { hours := 0, minutes := 10, h_valid := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_mars_visibility_time_l271_27137


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_implies_a_range_l271_27191

theorem cubic_equation_one_root_implies_a_range (a : ℝ) : 
  (∃! x : ℝ, x^3 + (1-a)*x^2 - 2*a*x + a^2 = 0) → a < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_implies_a_range_l271_27191


namespace NUMINAMATH_CALUDE_intersection_equals_B_l271_27144

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x = 3}

-- Define the set of possible values for a
def possible_a : Set ℝ := {0, -1, 3}

-- State the theorem
theorem intersection_equals_B (a : ℝ) :
  (A ∩ B a = B a) ↔ a ∈ possible_a :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l271_27144


namespace NUMINAMATH_CALUDE_nancys_weight_l271_27129

/-- 
Given that Nancy's total daily water intake (including water from food) is 62 pounds,
and she drinks 75% of her body weight in water plus 2 pounds from food,
prove that her weight is 80 pounds.
-/
theorem nancys_weight (W : ℝ) : 0.75 * W + 2 = 62 → W = 80 := by
  sorry

end NUMINAMATH_CALUDE_nancys_weight_l271_27129


namespace NUMINAMATH_CALUDE_jason_pears_l271_27146

theorem jason_pears (mike_pears jason_pears total_pears : ℕ) 
  (h1 : mike_pears = 8)
  (h2 : total_pears = 15)
  (h3 : total_pears = mike_pears + jason_pears) :
  jason_pears = 7 := by
  sorry

end NUMINAMATH_CALUDE_jason_pears_l271_27146


namespace NUMINAMATH_CALUDE_probability_three_red_one_blue_l271_27143

theorem probability_three_red_one_blue (total_red : Nat) (total_blue : Nat) 
  (draw_count : Nat) (red_count : Nat) (blue_count : Nat) :
  total_red = 10 →
  total_blue = 5 →
  draw_count = 4 →
  red_count = 3 →
  blue_count = 1 →
  (Nat.choose total_red red_count * Nat.choose total_blue blue_count : ℚ) / 
  (Nat.choose (total_red + total_blue) draw_count) = 40 / 91 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_red_one_blue_l271_27143


namespace NUMINAMATH_CALUDE_problem_solution_l271_27114

theorem problem_solution :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ m : ℕ, 180 = 9 * m) ∧
  (∃ k : ℕ, 209 = 19 * k) ∧
  (∃ l : ℕ, 57 = 19 * l) ∧
  (∃ p : ℕ, 90 = 30 * p) ∧
  (∃ q : ℕ, 34 = 17 * q) ∧
  (∃ r : ℕ, 51 = 17 * r) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l271_27114


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l271_27186

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2 * x - y + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l271_27186


namespace NUMINAMATH_CALUDE_lab_coat_uniform_ratio_l271_27173

theorem lab_coat_uniform_ratio :
  ∀ (num_uniforms num_lab_coats num_total : ℕ),
    num_uniforms = 12 →
    num_lab_coats = 6 * num_uniforms →
    num_total = num_lab_coats + num_uniforms →
    num_total % 14 = 0 →
    num_lab_coats / num_uniforms = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_lab_coat_uniform_ratio_l271_27173


namespace NUMINAMATH_CALUDE_problem_statement_l271_27133

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1 / a^2 + 1 / b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l271_27133


namespace NUMINAMATH_CALUDE_arthur_total_distance_l271_27149

/-- Calculates the total distance walked by Arthur in miles -/
def arthur_walk (block_length : ℚ) (east west north south : ℕ) : ℚ :=
  ((east + west + north + south) : ℚ) * block_length

/-- Theorem: Arthur's total walk distance is 4.5 miles -/
theorem arthur_total_distance :
  arthur_walk (1/4) 8 0 15 5 = 4.5 := by sorry

end NUMINAMATH_CALUDE_arthur_total_distance_l271_27149


namespace NUMINAMATH_CALUDE_credit_card_balance_calculation_l271_27155

/-- Calculates the final balance on a credit card after two interest applications and an additional charge. -/
def finalBalance (initialBalance : ℝ) (interestRate : ℝ) (additionalCharge : ℝ) : ℝ :=
  let balanceAfterFirstInterest := initialBalance * (1 + interestRate)
  let balanceAfterCharge := balanceAfterFirstInterest + additionalCharge
  balanceAfterCharge * (1 + interestRate)

/-- Theorem stating that given the specific conditions, the final balance is $96.00 -/
theorem credit_card_balance_calculation :
  finalBalance 50 0.2 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_calculation_l271_27155


namespace NUMINAMATH_CALUDE_smallest_multiple_with_factors_l271_27127

theorem smallest_multiple_with_factors : ∃ (n : ℕ+), 
  (∀ (m : ℕ+), (1452 * m : ℕ) % 2^4 = 0 ∧ 
                (1452 * m : ℕ) % 3^3 = 0 ∧ 
                (1452 * m : ℕ) % 13^3 = 0 → n ≤ m) ∧
  (1452 * n : ℕ) % 2^4 = 0 ∧ 
  (1452 * n : ℕ) % 3^3 = 0 ∧ 
  (1452 * n : ℕ) % 13^3 = 0 ∧
  n = 79092 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_factors_l271_27127


namespace NUMINAMATH_CALUDE_cheese_purchase_l271_27139

theorem cheese_purchase (initial_amount : ℕ) (cheese_cost beef_cost : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_cheese_purchase_l271_27139


namespace NUMINAMATH_CALUDE_lcm_gcd_product_15_75_l271_27138

theorem lcm_gcd_product_15_75 : Nat.lcm 15 75 * Nat.gcd 15 75 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_15_75_l271_27138


namespace NUMINAMATH_CALUDE_mrs_thomson_savings_l271_27154

theorem mrs_thomson_savings (incentive : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (saved_amount : ℝ)
  (h1 : incentive = 240)
  (h2 : food_fraction = 1/3)
  (h3 : clothes_fraction = 1/5)
  (h4 : saved_amount = 84) :
  let remaining := incentive - (food_fraction * incentive) - (clothes_fraction * incentive)
  saved_amount / remaining = 3/4 := by
sorry

end NUMINAMATH_CALUDE_mrs_thomson_savings_l271_27154


namespace NUMINAMATH_CALUDE_stamp_block_bounds_l271_27106

/-- 
b(n) is the smallest number of blocks of three adjacent stamps that can be torn out 
from an n × n sheet to make it impossible to tear out any more such blocks.
-/
noncomputable def b (n : ℕ) : ℕ := sorry

/-- 
There exist real constants c and d such that for all positive integers n, 
the function b(n) satisfies the inequality (1/7)n^2 - cn ≤ b(n) ≤ (1/5)n^2 + dn
-/
theorem stamp_block_bounds : 
  ∃ (c d : ℝ), ∀ (n : ℕ), n > 0 → 
    ((1 : ℝ) / 7) * (n : ℝ)^2 - c * (n : ℝ) ≤ (b n : ℝ) ∧ 
    (b n : ℝ) ≤ ((1 : ℝ) / 5) * (n : ℝ)^2 + d * (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_stamp_block_bounds_l271_27106


namespace NUMINAMATH_CALUDE_smallest_perfect_squares_l271_27122

theorem smallest_perfect_squares (a b : ℕ+) 
  (h1 : ∃ x : ℕ, (15 * a + 16 * b : ℕ) = x^2)
  (h2 : ∃ y : ℕ, (16 * a - 15 * b : ℕ) = y^2) :
  ∃ (x y : ℕ), x^2 = 231361 ∧ y^2 = 231361 ∧ 
    (∀ (x' y' : ℕ), (15 * a + 16 * b : ℕ) = x'^2 → (16 * a - 15 * b : ℕ) = y'^2 → 
      x'^2 ≥ 231361 ∧ y'^2 ≥ 231361) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_squares_l271_27122


namespace NUMINAMATH_CALUDE_trig_inequality_l271_27117

theorem trig_inequality : 
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l271_27117


namespace NUMINAMATH_CALUDE_right_triangle_area_l271_27101

/-- The area of a right triangle with hypotenuse 10√2 and one 45° angle is 50 square inches. -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse length
  α = 45 * π / 180 →      -- one angle in radians
  A = 50 →                -- area
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧     -- Pythagorean theorem
    Real.cos α = a / h ∧  -- cosine of the angle
    A = (1/2) * a * b     -- area formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l271_27101


namespace NUMINAMATH_CALUDE_product_inequality_l271_27181

theorem product_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l271_27181


namespace NUMINAMATH_CALUDE_tetrahedron_face_sum_squares_l271_27112

/-- A tetrahedron with circumradius 1 and face triangles with sides a, b, and c -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the face triangle sides of a tetrahedron with circumradius 1 is equal to 8 -/
theorem tetrahedron_face_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_sum_squares_l271_27112


namespace NUMINAMATH_CALUDE_chess_tournament_games_l271_27134

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) * 2

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, the total number of games is 480 -/
theorem chess_tournament_games :
  tournament_games 16 = 480 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l271_27134


namespace NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l271_27151

/-- Given two vectors a and b in R³, if they are orthogonal and have equal magnitudes,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (4, p, -2))
  (h_b : b = (3, 2, q))
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0)
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2)
  : p = -29/12 ∧ q = 43/12 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l271_27151


namespace NUMINAMATH_CALUDE_simplify_expression_l271_27142

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (25 * x^3) * (8 * x^4) * (1 / (4 * x^2)^3) = 25/8 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l271_27142


namespace NUMINAMATH_CALUDE_positive_A_value_l271_27175

theorem positive_A_value : ∃ A : ℕ+, A^2 - 1 = 3577 * 3579 ∧ A = 3578 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l271_27175


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l271_27110

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 800) 
  (h2 : blue_percent = 45/100) 
  (h3 : red_percent = 23/100) 
  (h4 : green_percent = 15/100) : 
  ℕ := by
  sorry

#check students_wearing_other_colors

end NUMINAMATH_CALUDE_students_wearing_other_colors_l271_27110


namespace NUMINAMATH_CALUDE_total_lunch_spending_l271_27148

def lunch_problem (your_spending friend_spending total_spending : ℕ) : Prop :=
  friend_spending = 11 ∧
  friend_spending = your_spending + 3 ∧
  total_spending = your_spending + friend_spending

theorem total_lunch_spending : ∃ (your_spending friend_spending total_spending : ℕ),
  lunch_problem your_spending friend_spending total_spending ∧ total_spending = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_lunch_spending_l271_27148


namespace NUMINAMATH_CALUDE_pentagon_area_ratio_l271_27199

-- Define the pentagon
structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def is_convex (p : Pentagon) : Prop := sorry

-- Define parallel lines
def parallel (a b c d : ℝ × ℝ) : Prop := sorry

-- Define angle measurement
def angle (a b c : ℝ × ℝ) : ℝ := sorry

-- Define distance between points
def distance (a b : ℝ × ℝ) : ℝ := sorry

-- Define area of a triangle
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem pentagon_area_ratio (p : Pentagon) :
  is_convex p →
  parallel p.F p.G p.I p.J →
  parallel p.G p.H p.F p.I →
  parallel p.G p.I p.H p.J →
  angle p.F p.G p.H = 120 * π / 180 →
  distance p.F p.G = 4 →
  distance p.G p.H = 6 →
  distance p.H p.J = 18 →
  (triangle_area p.F p.G p.H) / (triangle_area p.G p.H p.I) = 16 / 171 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_ratio_l271_27199


namespace NUMINAMATH_CALUDE_value_std_dev_from_mean_l271_27131

/-- Proves that for a normal distribution with mean 16.5 and standard deviation 1.5,
    the value 13.5 is 2 standard deviations less than the mean. -/
theorem value_std_dev_from_mean :
  let μ : ℝ := 16.5  -- mean
  let σ : ℝ := 1.5   -- standard deviation
  let x : ℝ := 13.5  -- value in question
  (x - μ) / σ = -2
  := by sorry

end NUMINAMATH_CALUDE_value_std_dev_from_mean_l271_27131


namespace NUMINAMATH_CALUDE_square_field_area_l271_27160

/-- The area of a square field with a diagonal of 30 meters is 450 square meters. -/
theorem square_field_area (diagonal : ℝ) (h : diagonal = 30) : 
  (diagonal ^ 2) / 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l271_27160


namespace NUMINAMATH_CALUDE_ant_return_probability_l271_27189

/-- A modified lattice with an extra horizontal connection --/
structure ModifiedLattice :=
  (extra_connection : Bool)

/-- An ant on the modified lattice --/
structure Ant :=
  (position : ℤ × ℤ)
  (moves : ℕ)

/-- The probability of the ant returning to its starting point --/
def return_probability (l : ModifiedLattice) (a : Ant) : ℚ :=
  sorry

/-- Theorem stating the probability of returning to the starting point after 6 moves --/
theorem ant_return_probability (l : ModifiedLattice) (a : Ant) : 
  l.extra_connection = true →
  a.moves = 6 →
  return_probability l a = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l271_27189


namespace NUMINAMATH_CALUDE_special_line_equation_l271_27163

/-- A line passing through point (3,-1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (3,-1) -/
  passes_through_point : slope * 3 + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_condition : y_intercept ≠ 0 → -y_intercept / slope = 2 * y_intercept

theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2 * y - 1 = 0) ∨
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 3 * y = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l271_27163


namespace NUMINAMATH_CALUDE_cos_sin_sum_l271_27104

theorem cos_sin_sum (α : Real) (h : Real.cos (π/6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5*π/6 + α) + (Real.sin (α - π/6))^2 = (2 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l271_27104


namespace NUMINAMATH_CALUDE_high_scam_probability_l271_27128

/-- Represents an email message -/
structure Email :=
  (claims_prize : Bool)
  (asks_for_phone : Bool)
  (requests_payment : Bool)
  (payment_amount : ℕ)

/-- Represents the probability of an email being a scam -/
def scam_probability (e : Email) : ℝ := sorry

/-- Theorem: Given an email with specific characteristics, the probability of it being a scam is high -/
theorem high_scam_probability (e : Email) 
  (h1 : e.claims_prize = true)
  (h2 : e.asks_for_phone = true)
  (h3 : e.requests_payment = true)
  (h4 : e.payment_amount = 150) :
  scam_probability e > 0.9 := by sorry

end NUMINAMATH_CALUDE_high_scam_probability_l271_27128


namespace NUMINAMATH_CALUDE_inequality_of_four_positives_l271_27147

theorem inequality_of_four_positives (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((abc + abd + acd + bcd) / 4) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_of_four_positives_l271_27147


namespace NUMINAMATH_CALUDE_card_cost_calculation_l271_27124

theorem card_cost_calculation (christmas_cards : ℕ) (birthday_cards : ℕ) (total_spent : ℕ) : 
  christmas_cards = 20 →
  birthday_cards = 15 →
  total_spent = 70 →
  (total_spent : ℚ) / (christmas_cards + birthday_cards : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_card_cost_calculation_l271_27124


namespace NUMINAMATH_CALUDE_prob_red_then_king_diamonds_standard_deck_l271_27197

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (red_cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Probability of drawing a red card first and then the King of Diamonds second -/
def prob_red_then_king_diamonds (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.red_cards = 26 ∧ d.ranks = 13 ∧ d.suits = 4 then
    1 / 102
  else
    0

/-- Theorem stating the probability of drawing a red card first and then the King of Diamonds second -/
theorem prob_red_then_king_diamonds_standard_deck :
  ∃ (d : Deck), prob_red_then_king_diamonds d = 1 / 102 :=
sorry

end NUMINAMATH_CALUDE_prob_red_then_king_diamonds_standard_deck_l271_27197


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l271_27135

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 12 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 58956 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l271_27135


namespace NUMINAMATH_CALUDE_conditional_prob_specific_given_different_l271_27102

/-- The number of attractions available for tourists to choose from. -/
def num_attractions : ℕ := 5

/-- The probability that two tourists choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that one tourist chooses a specific attraction and the other chooses any of the remaining attractions. -/
def prob_one_specific_others_different : ℚ := 8 / 25

/-- Theorem stating the conditional probability of both tourists choosing a specific attraction given they choose different attractions. -/
theorem conditional_prob_specific_given_different :
  prob_one_specific_others_different / prob_different_attractions = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_specific_given_different_l271_27102


namespace NUMINAMATH_CALUDE_modifiedLucas_100th_term_divisible_by_5_l271_27105

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 4
  | (n + 2) => (modifiedLucas n + modifiedLucas (n + 1)) % 5

theorem modifiedLucas_100th_term_divisible_by_5 : modifiedLucas 99 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modifiedLucas_100th_term_divisible_by_5_l271_27105


namespace NUMINAMATH_CALUDE_redox_agents_identification_l271_27165

/-- Represents a chemical species with its oxidation state -/
structure Species where
  element : String
  oxidation_state : Int

/-- Represents a half-reaction in a redox reaction -/
structure HalfReaction where
  reactant : Species
  product : Species
  electrons : Int
  is_reduction : Bool

/-- Represents a full redox reaction -/
structure RedoxReaction where
  oxidation : HalfReaction
  reduction : HalfReaction

def is_oxidizing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.reduction.reactant

def is_reducing_agent (s : Species) (r : RedoxReaction) : Prop :=
  s = r.oxidation.reactant

theorem redox_agents_identification
  (s0 : Species)
  (h20 : Species)
  (h2plus : Species)
  (s2minus : Species)
  (reduction : HalfReaction)
  (oxidation : HalfReaction)
  (full_reaction : RedoxReaction)
  (h_s0 : s0 = ⟨"S", 0⟩)
  (h_h20 : h20 = ⟨"H2", 0⟩)
  (h_h2plus : h2plus = ⟨"H2", 1⟩)
  (h_s2minus : s2minus = ⟨"S", -2⟩)
  (h_reduction : reduction = ⟨s0, s2minus, 2, true⟩)
  (h_oxidation : oxidation = ⟨h20, h2plus, -2, false⟩)
  (h_full_reaction : full_reaction = ⟨oxidation, reduction⟩)
  : is_oxidizing_agent s0 full_reaction ∧ is_reducing_agent h20 full_reaction := by
  sorry


end NUMINAMATH_CALUDE_redox_agents_identification_l271_27165


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l271_27132

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is on the line if it satisfies the line equation -/
def point_on_line (x y : ℝ) : Prop := line_equation x y

/-- The point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

/-- The point (x, y) is equidistant from coordinate axes -/
def equidistant_from_axes (x y : ℝ) : Prop := x = y

/-- The theorem stating that (12/7, 12/7) is the unique point satisfying all conditions -/
theorem unique_equidistant_point :
  ∃! (x y : ℝ), point_on_line x y ∧ in_first_quadrant x y ∧ equidistant_from_axes x y ∧ x = 12/7 ∧ y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l271_27132


namespace NUMINAMATH_CALUDE_subset_condition_implies_upper_bound_l271_27103

theorem subset_condition_implies_upper_bound (a : ℝ) :
  let A := {x : ℝ | x > 3}
  let B := {x : ℝ | x > a}
  A ⊆ B → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_subset_condition_implies_upper_bound_l271_27103


namespace NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l271_27176

theorem quadratic_roots_when_positive_discriminant
  (a b c : ℝ) (ha : a ≠ 0) (h_disc : b^2 - 4*a*c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + b*x₁ + c = 0 ∧ a*x₂^2 + b*x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_when_positive_discriminant_l271_27176
