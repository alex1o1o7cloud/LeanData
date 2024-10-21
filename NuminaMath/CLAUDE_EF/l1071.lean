import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cutting_lines_l1071_107159

-- Define a type for polygons in the plane
structure Polygon where
  vertices : Set (ℝ × ℝ)

-- Define what it means for polygons to be properly placed
def properly_placed (polygons : Set Polygon) : Prop :=
  ∀ p q, p ∈ polygons → q ∈ polygons → p ≠ q → ∃ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ t : ℝ, (x, y) = (t, 0) ∨ (x, y) = (0, t)) ∧
    (∃ (a b : ℝ × ℝ), a ∈ p.vertices ∧ b ∈ p.vertices ∧ a ∈ l ∧ b ∈ l) ∧
    (∃ (c d : ℝ × ℝ), c ∈ q.vertices ∧ d ∈ q.vertices ∧ c ∈ l ∧ d ∈ l)

-- Define what it means for a line to cut a polygon
def cuts (l : Set (ℝ × ℝ)) (p : Polygon) : Prop :=
  ∃ (a b : ℝ × ℝ), a ∈ p.vertices ∧ b ∈ p.vertices ∧ a ∈ l ∧ b ∈ l

-- The main theorem
theorem min_cutting_lines (polygons : Set Polygon) (h : properly_placed polygons) :
  (∃ (l₁ l₂ : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l₁ ↔ ∃ t : ℝ, (x, y) = (t, 0) ∨ (x, y) = (0, t)) ∧
    (∀ (x y : ℝ), (x, y) ∈ l₂ ↔ ∃ t : ℝ, (x, y) = (t, 0) ∨ (x, y) = (0, t)) ∧
    (∀ p, p ∈ polygons → cuts l₁ p ∨ cuts l₂ p)) ∧
  (∀ l : Set (ℝ × ℝ), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ t : ℝ, (x, y) = (t, 0) ∨ (x, y) = (0, t)) →
    ¬(∀ p, p ∈ polygons → cuts l p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cutting_lines_l1071_107159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vote_difference_l1071_107170

/-- Calculate total votes for a proposal given the difference in votes and percentage against --/
def totalVotes (diffVotes : ℤ) (percentAgainst : ℚ) : ℤ :=
  Int.floor ((diffVotes : ℚ) / (1 - 2 * percentAgainst))

/-- Main theorem statement --/
theorem vote_difference (proposal1_diff : ℤ) (proposal1_against : ℚ)
                        (proposal2_diff : ℤ) (proposal2_against : ℚ)
                        (proposal3_diff : ℤ) (proposal3_against : ℚ) :
  proposal1_diff = 58 →
  proposal1_against = 2/5 →
  proposal2_diff = 72 →
  proposal2_against = 9/20 →
  proposal3_diff = 30 →
  proposal3_against = 11/20 →
  let votes1 := totalVotes proposal1_diff proposal1_against
  let votes2 := totalVotes proposal2_diff proposal2_against
  let votes3 := totalVotes proposal3_diff proposal3_against
  max votes1 (max votes2 votes3) - min votes1 (min votes2 votes3) = 430 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vote_difference_l1071_107170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1071_107196

theorem inequality_problem (x y z : ℝ) 
  (h_pos : x > 1 ∧ y > 1 ∧ z > 1) 
  (h_sum : x + y + z = 3 * Real.sqrt 3) : 
  (x^2 / (x + 2*y + 3*z) + y^2 / (y + 2*z + 3*x) + z^2 / (z + 2*x + 3*y) ≥ Real.sqrt 3 / 2) ∧
  (1 / (Real.log x / Real.log 3 + Real.log y / Real.log 3) + 
   1 / (Real.log y / Real.log 3 + Real.log z / Real.log 3) + 
   1 / (Real.log z / Real.log 3 + Real.log x / Real.log 3) ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l1071_107196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row8_sum_l1071_107164

/-- Pascal's Triangle as a function from row and column to natural number -/
def pascal : ℕ → ℕ → ℕ
| 0, _ => 1
| n+1, 0 => 1
| n+1, k+1 => pascal n k + pascal n (k+1)

/-- Sum of numbers in a row of Pascal's Triangle -/
def rowSum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map (pascal n) |>.sum

/-- Theorem: The sum of numbers in Row 8 of Pascal's Triangle is 2^8 -/
theorem pascal_row8_sum : rowSum 8 = 2^8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_row8_sum_l1071_107164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_when_intersecting_l1071_107178

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

/-- A circle with diameter equal to the distance between foci of the ellipse -/
def focal_circle (e : Ellipse a b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = (2 * a * eccentricity e)^2}

/-- The set of points on the ellipse -/
def ellipse_points (e : Ellipse a b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- The theorem stating the range of eccentricity when the focal circle intersects the ellipse -/
theorem eccentricity_range_when_intersecting {a b : ℝ} (e : Ellipse a b) :
  (focal_circle e ∩ ellipse_points e).Nonempty →
  (Real.sqrt 2 / 2 ≤ eccentricity e) ∧ (eccentricity e < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_when_intersecting_l1071_107178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_condition_l1071_107188

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes_condition 
  (α β : Plane) (m : Line) 
  (h_distinct : α ≠ β) 
  (h_subset : subset m α) :
  (∀ γ δ : Plane, ∀ l : Line, 
    subset l γ → perp_line_plane l δ → perp_planes γ δ) →
  (perp_line_plane m β → perp_planes α β) ∧
  ¬(perp_planes α β → perp_line_plane m β) :=
by
  intro h
  constructor
  · intro h_perp_line
    exact h α β m h_subset h_perp_line
  · intro h_necessary
    sorry -- The proof of this part is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_condition_l1071_107188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1071_107168

/-- The vector a in R³ --/
def a : Fin 3 → ℝ := ![0, 1, 1]

/-- The vector b in R³ --/
def b : Fin 3 → ℝ := ![1, 0, 1]

/-- The angle between vectors a and b is π/3 (60 degrees) --/
theorem angle_between_vectors :
  Real.arccos (((a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2)) / 
    (Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2) * 
     Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2))) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1071_107168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_nonzero_l1071_107137

theorem polynomial_coefficient_nonzero 
  (Q : ℂ → ℂ) 
  (a b c d e f : ℝ) :
  (∀ x : ℂ, Q x = x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    ∀ x : ℂ, Q x = x * (x - Complex.I) * (x + Complex.I) * (x - p) * (x - q) * (x - r) * (x - s)) →
  Q 0 = 0 →
  Q Complex.I = 0 →
  e ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_nonzero_l1071_107137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_range_l1071_107157

/-- The circle equation -/
def circle_eq (x y r : ℝ) : Prop := (x - 3)^2 + (y + 5)^2 = r^2

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 4*x - 3*y - 2 = 0

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |4*x - 3*y - 2| / Real.sqrt (4^2 + 3^2)

/-- Theorem stating the range of r given the conditions -/
theorem radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃! (p q : ℝ × ℝ), circle_eq p.1 p.2 r ∧ circle_eq q.1 q.2 r ∧ 
    p ≠ q ∧ 
    distance_to_line p.1 p.2 = 1 ∧ 
    distance_to_line q.1 q.2 = 1) →
  4 < r ∧ r < 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_range_l1071_107157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_coordinate_l1071_107162

/-- A rectangle in the x-y plane with vertices at (0,0), (0,4), (x,4), and (x,0) -/
structure Rectangle where
  x : ℝ
  h_positive : x > 0

/-- The probability that a randomly chosen point in the rectangle satisfies x + y < 4 -/
noncomputable def probability (r : Rectangle) : ℝ := 
  (1/2 * 4 * 4) / (4 * r.x)

/-- Theorem stating that if the probability is 0.4, then x = 5 -/
theorem rectangle_x_coordinate (r : Rectangle) (h : probability r = 0.4) : r.x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_coordinate_l1071_107162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1071_107171

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 + 3*x + k) / (x^2 - 3*x - 28)

theorem one_vertical_asymptote (k : ℝ) : 
  (∃! x : ℝ, ∀ y : ℝ, g k x ≠ y) ↔ (k = -70 ∨ k = -4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l1071_107171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_sunny_days_probability_l1071_107173

theorem exactly_two_sunny_days_probability :
  let n : ℕ := 5  -- total number of days
  let k : ℕ := 2  -- number of sunny days we want
  let p : ℚ := 2/5  -- probability of a sunny day (40% = 2/5)
  Finset.sum (Finset.range (n + 1)) (λ i ↦ if i = k then Nat.choose n i * p^i * (1 - p)^(n - i) else 0) = 216/625 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_sunny_days_probability_l1071_107173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_probabilities_equal_l1071_107104

variable (N n : ℕ)
variable (p₁ p₂ p₃ : ℝ)

/-- Simple random sampling probability -/
def simple_random_prob : ℝ := p₁

/-- Systematic sampling probability -/
def systematic_prob : ℝ := p₂

/-- Stratified sampling probability -/
def stratified_prob : ℝ := p₃

/-- Theorem: The probability of an individual being selected is equal for all three sampling methods -/
theorem sampling_probabilities_equal
  (h₁ : 0 < N)
  (h₂ : n ≤ N)
  (h₃ : 0 ≤ p₁ ∧ p₁ ≤ 1)
  (h₄ : 0 ≤ p₂ ∧ p₂ ≤ 1)
  (h₅ : 0 ≤ p₃ ∧ p₃ ≤ 1) :
  simple_random_prob = systematic_prob ∧
  systematic_prob = stratified_prob :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_probabilities_equal_l1071_107104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_and_g_max_a_l1071_107152

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 2 * x - 3

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.log x + a * (x - 1) / x

theorem f_unique_zero_and_g_max_a :
  (∃! x : ℝ, x ≥ 1 ∧ f x = 0) ∧
  (∀ a : ℝ, (∀ x ≥ 1, Monotone (g a)) → a ≤ 6) ∧
  (∃ a : ℝ, a > 5 ∧ ∀ x ≥ 1, Monotone (g a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_and_g_max_a_l1071_107152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_five_l1071_107166

def lcm_three (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

theorem third_number_is_five (n : ℕ) (h : n > 0) :
  lcm_three 24 36 n = 360 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_five_l1071_107166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_l1071_107151

def U : Set ℝ := {1, 3, 5, 7}

def M (a : ℝ) : Set ℝ := {1, |a - 5|}

def M_complement : Set ℝ := {5, 7}

theorem a_values : ∃ (a : ℝ), M a ∪ M_complement = U ∧ (a = 2 ∨ a = 8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_l1071_107151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cosine_condition_l1071_107132

-- Define the interval from which x is chosen
def interval : Set ℝ := Set.Icc (-3/2) (3/2)

-- Define the condition for cos(πx/3) being between 1/2 and 1
def cosCondition (x : ℝ) : Prop := 1/2 < Real.cos (Real.pi * x / 3) ∧ Real.cos (Real.pi * x / 3) ≤ 1

-- Define the set of x values satisfying the cosine condition
def satisfyingSet : Set ℝ := {x ∈ interval | cosCondition x}

-- State the theorem
theorem probability_cosine_condition :
  (MeasureTheory.volume satisfyingSet) / (MeasureTheory.volume interval) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_cosine_condition_l1071_107132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_variance_l1071_107191

noncomputable def scores : List ℝ := [110, 114, 121, 119, 126]

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem scores_variance :
  variance scores = 30.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scores_variance_l1071_107191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1071_107135

-- Define the base-2 logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- Define the function f(x) = lg(2^x - 1)
noncomputable def f (x : ℝ) := lg (2^x - 1)

-- Theorem stating the domain of f
theorem domain_of_f :
  ∀ x : ℝ, x ∈ Set.Ioi 0 ↔ ∃ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1071_107135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_neg_4_l1071_107112

def b : ℕ → ℤ
  | 0 => 1  -- Define b₀ to be 1 (same as b₁)
  | 1 => 1
  | 2 => 4
  | n + 3 => b (n + 2) - b (n + 1)

theorem b_2023_eq_neg_4 : b 2023 = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_eq_neg_4_l1071_107112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinguishable_colorings_l1071_107144

/-- Represents the colors that can be used to paint the cube faces -/
inductive Color
| Red
| White
| Blue
| Green

/-- Represents a coloring of a cube -/
def Coloring := Fin 6 → Color

/-- Two colorings are considered equivalent if they can be rotated to look identical -/
def equivalent_colorings : Coloring → Coloring → Prop := sorry

/-- The set of all possible colorings of the cube -/
def all_colorings : Set Coloring := sorry

/-- The setoid structure for colorings based on equivalence -/
def coloring_setoid : Setoid Coloring :=
{ r := equivalent_colorings,
  iseqv := sorry }

/-- The set of distinguishable colorings -/
def distinguishable_colorings : Type :=
  Quotient coloring_setoid

/-- Assume that the distinguishable colorings form a finite type -/
instance : Fintype distinguishable_colorings := sorry

theorem number_of_distinguishable_colorings :
  Fintype.card distinguishable_colorings = 76 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_distinguishable_colorings_l1071_107144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1071_107161

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- Theorem statement
theorem zero_point_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1071_107161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1071_107194

/-- Represents the speed for each half of a journey leg -/
structure LegSpeed where
  first_half : ℚ
  second_half : ℚ

/-- Calculates the time taken for a journey leg given its speed -/
def leg_time (speed : LegSpeed) (distance : ℚ) : ℚ :=
  distance / (2 * speed.first_half) + distance / (2 * speed.second_half)

/-- The journey consists of three legs with given speeds -/
def journey_speeds : List LegSpeed := [
  { first_half := 20, second_half := 25 },  -- To office
  { first_half := 30, second_half := 35 },  -- To friend's house
  { first_half := 40, second_half := 45 }   -- Back home
]

/-- Theorem: The average speed of the entire journey is 4320 / 351 miles per hour -/
theorem journey_average_speed (distance : ℚ) (h : distance > 0) :
  let total_distance := 3 * distance
  let total_time := (journey_speeds.map (leg_time · distance)).sum
  total_distance / total_time = 4320 / 351 := by
  sorry

#eval (4320 : ℚ) / 351

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l1071_107194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allay_max_debt_allowed_l1071_107118

/-- Represents the internet service provider's policy and Allay's usage --/
structure InternetService where
  daily_cost : ℚ
  initial_payment : ℚ
  initial_balance : ℚ
  days_connected : ℕ

/-- Calculates the maximum debt allowed before service discontinuation --/
def max_debt_allowed (service : InternetService) : ℚ :=
  service.daily_cost * service.days_connected - service.initial_payment - service.initial_balance

/-- Theorem stating the maximum debt allowed for Allay's specific case --/
theorem allay_max_debt_allowed :
  let service : InternetService := {
    daily_cost := 1/2,
    initial_payment := 7,
    initial_balance := 0,
    days_connected := 25
  }
  max_debt_allowed service = 11/2 := by
  sorry

#eval max_debt_allowed {
  daily_cost := 1/2,
  initial_payment := 7,
  initial_balance := 0,
  days_connected := 25
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_allay_max_debt_allowed_l1071_107118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l1071_107147

def C : Set Nat := {67, 71, 73, 76, 79}

def smallest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).head!

theorem smallest_prime_factor_in_C :
  ∀ x ∈ C, smallest_prime_factor 76 ≤ smallest_prime_factor x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l1071_107147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1071_107165

theorem trigonometric_equation_reduction :
  (∃ (x : ℝ), (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2) →
  (∃ (x : ℝ), Real.cos (8*x) * Real.cos (4*x) * Real.cos (2*x) = 0) ∧
  (8 + 4 + 2 = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1071_107165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supporting_function_propositions_l1071_107142

-- Define the concept of a supporting function
def is_supporting_function (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x ≥ g x

-- Define the specific functions mentioned in the propositions
noncomputable def f1 : ℝ → ℝ
| x => if x > 0 then Real.log x else 1

noncomputable def f2 (x : ℝ) : ℝ := x + Real.sin x

noncomputable def f3 (x : ℝ) : ℝ := Real.exp x

-- State the propositions
def prop1 : Prop := is_supporting_function f1 (λ _ => -2)

def prop2 : Prop := is_supporting_function f2 (λ x => x - 1)

def prop3 : Prop := ∀ a : ℝ, (is_supporting_function f3 (λ x => a * x)) → (0 ≤ a ∧ a ≤ Real.exp 1)

def prop4 : Prop := ∀ f : ℝ → ℝ, (Set.range f = Set.univ) → (¬∃ g : ℝ → ℝ, is_supporting_function f g)

-- The main theorem
theorem supporting_function_propositions :
  (prop1 = false) ∧ (prop2 = true) ∧ (prop3 = true) ∧ (prop4 = false) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_supporting_function_propositions_l1071_107142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1071_107198

/-- The length of a train in meters -/
noncomputable def train_length (speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) * time_seconds

/-- Theorem stating the length of the train -/
theorem train_length_calculation :
  let speed_kmph : ℝ := 54
  let time_seconds : ℝ := 3.9996800255979523
  abs (train_length speed_kmph time_seconds - 59.9952) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1071_107198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1071_107122

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((1 + x) / (1 + a * x))

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 / (1 + 2^x)

theorem problem_solution (a : ℝ) (h1 : a ≠ 1) (h2 : is_odd (f a)) :
  a = -1 ∧ g a (1/2) + g a (-1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1071_107122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edges_partition_l1071_107117

-- Define a tetrahedron as a structure with 6 edge lengths
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ
  -- Ensure all edge lengths are positive
  ab_pos : AB > 0
  ac_pos : AC > 0
  ad_pos : AD > 0
  bc_pos : BC > 0
  bd_pos : BD > 0
  cd_pos : CD > 0
  -- Ensure triangle inequality holds for each face
  face_abc : AB < AC + BC ∧ AC < AB + BC ∧ BC < AB + AC
  face_abd : AB < AD + BD ∧ AD < AB + BD ∧ BD < AB + AD
  face_acd : AC < AD + CD ∧ AD < AC + CD ∧ CD < AC + AD
  face_bcd : BC < BD + CD ∧ BD < BC + CD ∧ CD < BC + BD

-- Define a predicate for a valid triangle
def IsValidTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem tetrahedron_edges_partition (t : Tetrahedron) :
  ∃ (e1 e2 e3 e4 e5 e6 : ℝ),
    Multiset.toFinset {e1, e2, e3, e4, e5, e6} = Multiset.toFinset {t.AB, t.AC, t.AD, t.BC, t.BD, t.CD} ∧
    IsValidTriangle e1 e2 e3 ∧
    IsValidTriangle e4 e5 e6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_edges_partition_l1071_107117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1071_107110

def next_sequence (seq : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := seq
  (a * b, b * c, c * d, d * a)

def generate_sequences (a b c d : ℝ) : ℕ → ℝ × ℝ × ℝ × ℝ
  | 0 => (a, b, c, d)
  | n + 1 => next_sequence (generate_sequences a b c d n)

theorem sequence_convergence (a b c d : ℝ) :
  (∀ m n : ℕ, m ≠ n → generate_sequences a b c d m ≠ generate_sequences a b c d n) ∨
  (∃ N : ℕ, ∀ n ≥ N, generate_sequences a b c d n = generate_sequences a b c d N) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l1071_107110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_20_degrees_l1071_107108

/-- The area of a figure formed by rotating a semicircle about one of its endpoints -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (α / (2 * Real.pi)) * (2 * Real.pi * R^2)

/-- Theorem: The area of a figure formed by rotating a semicircle of radius R
    about one of its endpoints by an angle of 20° is equal to (2 * π * R^2) / 9 -/
theorem rotated_semicircle_area_20_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (20 * Real.pi / 180) = (2 * Real.pi * R^2) / 9 := by
  sorry

#check rotated_semicircle_area_20_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_20_degrees_l1071_107108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_p_squared_minus_k_div_15_l1071_107146

-- Define p as the largest prime with 2023 digits
noncomputable def p : ℕ := sorry

-- Define the property that p is the largest prime with 2023 digits
axiom p_is_largest_prime_2023_digits : 
  Nat.Prime p ∧ (∀ q, Nat.Prime q → (Nat.digits 10 q).length = 2023 → q ≤ p)

-- Define the function to check if a number is divisible by 15
def divisible_by_15 (n : ℤ) : Prop := n % 15 = 0

-- Theorem statement
theorem smallest_k_for_p_squared_minus_k_div_15 :
  ∃ k : ℕ+, (∀ m : ℕ+, divisible_by_15 (p^2 - m) → k ≤ m) ∧ 
    divisible_by_15 (p^2 - k) ∧ k = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_p_squared_minus_k_div_15_l1071_107146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positivity_implies_m_range_l1071_107185

/-- Given two functions f and g, prove that if at least one of them is always positive, 
    then the parameter m is in the open interval (0,8) -/
theorem function_positivity_implies_m_range 
  (f g : ℝ → ℝ) 
  (m : ℝ)
  (hf : f = λ x ↦ 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (hg : g = λ x ↦ m * x) 
  (h_pos : ∀ x, f x > 0 ∨ g x > 0) : 
  m ∈ Set.Ioo 0 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positivity_implies_m_range_l1071_107185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_y_greater_2x_l1071_107125

open MeasureTheory Measure ProbabilityTheory Set

-- Define the sample space
def Ω : Set (ℝ × ℝ) := Icc 0 1 ×ˢ Icc 0 1

-- Define the event
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ Ω ∧ p.2 > 2 * p.1}

-- Define the probability measure
noncomputable def P : Measure (ℝ × ℝ) := 
  volume.restrict Ω

-- The theorem to prove
theorem prob_y_greater_2x : P E = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_y_greater_2x_l1071_107125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_min_max_values_l1071_107160

theorem function_min_max_values 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_discriminant : b^2 - 4*a*c < 0) :
  let f : ℝ × ℝ → ℝ := fun p ↦ a * p.1^2 + c * p.2^2
  ∀ x y, a * x^2 - b * x * y + c * y^2 = d →
    (2 * d * Real.sqrt (a * c)) / (b + 2 * Real.sqrt (a * c)) ≤ f (x, y) ∧
    f (x, y) ≤ (2 * d * Real.sqrt (a * c)) / (b - 2 * Real.sqrt (a * c)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_min_max_values_l1071_107160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_equivalent_l1071_107150

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 2)

-- Define the transformations
noncomputable def transform_A (x : ℝ) : ℝ := -Real.cos (3 * x)
noncomputable def transform_B (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 2)
noncomputable def transform_D (x : ℝ) : ℝ := -Real.cos (3 * x)

-- Theorem stating that the transformations result in the original function
theorem transformations_equivalent :
  (∀ x, f x = transform_A x) ∧
  (∀ x, f x = transform_B x) ∧
  (∀ x, f x = transform_D x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_equivalent_l1071_107150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_equals_repunit_ones_l1071_107158

theorem period_equals_repunit_ones (p : ℕ) (hp : Nat.Prime p) (hp_neq_3 : p ≠ 3) :
  (∃ d : ℕ, d > 0 ∧ (∀ k : ℕ, 0 < k ∧ k < d → ¬(p ∣ (10^k - 1))) ∧ (p ∣ (10^d - 1))) ↔
  (∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, 0 < m ∧ m < k → ¬(p ∣ ((10^m - 1) / 9))) ∧ (p ∣ ((10^k - 1) / 9))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_equals_repunit_ones_l1071_107158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_sixes_twelve_dice_l1071_107102

/-- The probability of rolling exactly 4 sixes with 12 six-sided dice -/
theorem probability_four_sixes_twelve_dice :
  (Finset.card (Finset.filter (fun s => s.card = 4) (Finset.powerset (Finset.range 12))) *
   (1 / 6 : ℚ)^4 * (5 / 6 : ℚ)^8 : ℚ) =
  (495 * 390625 : ℕ) / 2176782336 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_sixes_twelve_dice_l1071_107102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1071_107182

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The focal distance of an ellipse -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Predicate to check if a hyperbola has a given asymptote slope -/
def Hyperbola.hasAsymptoteSlope (h : Hyperbola) (slope : ℝ) : Prop :=
  h.b / h.a = slope

/-- Predicate to check if a hyperbola shares a focus with an ellipse -/
noncomputable def sharesFocus (h : Hyperbola) (e : Ellipse) : Prop :=
  Real.sqrt (h.a^2 + h.b^2) = e.focalDistance

/-- Main theorem statement -/
theorem hyperbola_equation (h : Hyperbola) (e : Ellipse) :
    e.a^2 = 12 ∧ e.b^2 = 3 ∧
    h.hasAsymptoteSlope (Real.sqrt 5 / 2) ∧
    sharesFocus h e →
    h.a^2 = 4 ∧ h.b^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1071_107182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1071_107148

theorem complex_equality (a : ℝ) : 
  (Complex.re ((1 + 3*Complex.I) * (2*a + Complex.I)) = Complex.im ((1 + 3*Complex.I) * (2*a + Complex.I))) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equality_l1071_107148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_L_condition_geometric_L_condition_range_geometric_sum_L_condition_l1071_107187

/-- Definition of L(k) condition for a sequence -/
def L_condition (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ m n : ℕ, m ≠ n → |a m - a n| ≤ k * |Int.ofNat m - Int.ofNat n|

/-- Arithmetic sequence with common difference d -/
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

/-- Geometric sequence with first term a and common ratio r -/
def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- Sum of first n terms of geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem arithmetic_L_condition :
  L_condition (arithmetic_seq 0 2) 2 := by sorry

theorem geometric_L_condition_range (q : ℝ) :
  (q > 0 ∧ L_condition (geometric_seq 1 q) (1/2)) → 1/2 ≤ q ∧ q ≤ 1 := by sorry

theorem geometric_sum_L_condition (q : ℝ) (hq : 1/2 ≤ q ∧ q < 1) :
  ∃ k₀ : ℝ, k₀ > 0 ∧ L_condition (geometric_sum 1 q) k₀ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_L_condition_geometric_L_condition_range_geometric_sum_L_condition_l1071_107187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l1071_107175

theorem triangle_angle_cosine (α β γ : Real) (h1 : α < 40 * Real.pi / 180)
  (h2 : β < 80 * Real.pi / 180) (h3 : Real.sin γ = 5/8) :
  Real.cos γ = -Real.sqrt 39 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_l1071_107175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1071_107184

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 3*a else a^x

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  (a ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ) ∧ a ≠ 1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1071_107184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_even_l1071_107181

-- Define m
noncomputable def m : ℝ := (-1 + Real.sqrt 17) / 2

-- Define the polynomial P(x)
def P (n : ℕ) (a : ℕ → ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range (n + 1)) (λ i ↦ (a i : ℝ) * x ^ i)

-- Define the theorem
theorem sum_of_coefficients_even
  (n : ℕ)
  (a : ℕ → ℕ)
  (h_positive : ∀ i, i ≤ n → a i > 0)
  (h_eval : P n a m = 2018) :
  2 ∣ Finset.sum (Finset.range (n + 1)) a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_even_l1071_107181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_digits_correct_l1071_107121

/-- The probability of a randomly chosen three-digit integer between 100 and 999 having all different digits -/
def prob_different_digits : ℚ :=
  99 / 100

/-- The set of all three-digit integers -/
def three_digit_integers : Finset ℕ :=
  Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999) (Finset.range 1000)

/-- The set of three-digit integers with all different digits -/
def different_digits : Finset ℕ :=
  Finset.filter (fun n =>
    let digits := [n / 100, (n / 10) % 10, n % 10]
    List.Nodup digits
  ) three_digit_integers

theorem prob_different_digits_correct :
  (Finset.card different_digits : ℚ) / (Finset.card three_digit_integers) = prob_different_digits :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_different_digits_correct_l1071_107121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_is_systematic_sampling_l1071_107128

/-- Represents a sampling method --/
inductive SamplingMethod
  | StratifiedSampling
  | DrawingLots
  | RandomNumberTable
  | SystematicSampling

/-- Represents a class of students --/
structure StudentClass where
  students : Finset Nat
  student_count : students.card = 50

/-- Represents a grade with multiple classes --/
structure Grade where
  classes : Finset StudentClass
  class_count : classes.card = 16

/-- Defines the selection process for the exchange --/
def selectForExchange (g : Grade) : Finset Nat :=
  g.classes.biUnion (λ c => c.students.filter (λ s => s = 14))

/-- Theorem stating that the selection process is systematic sampling --/
theorem exchange_is_systematic_sampling (g : Grade) : 
  SamplingMethod.SystematicSampling = 
    (if (∀ c ∈ g.classes, ∃ s ∈ c.students, s = 14 ∧ s ∈ selectForExchange g) 
     then SamplingMethod.SystematicSampling 
     else SamplingMethod.DrawingLots) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_is_systematic_sampling_l1071_107128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1071_107130

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (x^3 + Real.sqrt (1 + x^6))

-- Theorem statement
theorem g_is_odd : ∀ x, g (-x) = -g x := by
  -- The proof is omitted and replaced with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1071_107130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_c_l1071_107133

noncomputable def vector_a : ℝ × ℝ := (1, Real.sqrt 3)

theorem magnitude_of_c (c : ℝ × ℝ) :
  (vector_a.1 * c.1 + vector_a.2 * c.2 = 2) →
  (vector_a.1 * c.1 + vector_a.2 * c.2 = Real.sqrt (vector_a.1^2 + vector_a.2^2) * Real.sqrt (c.1^2 + c.2^2) * Real.cos (π/3)) →
  Real.sqrt (c.1^2 + c.2^2) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_c_l1071_107133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_n_l1071_107193

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2012 * 7 -/
def digitSum : ℕ := 13

/-- The number we're interested in -/
def n : ℕ := 2^2010 * 5^2012 * 7

/-- Function to calculate the sum of digits -/
def sumOfDigits (m : ℕ) : ℕ := sorry

theorem digit_sum_of_n : sumOfDigits n = digitSum := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_n_l1071_107193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1071_107127

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x^3 + 2*x^2 - 3*x)

def holes (f : ℝ → ℝ) : ℕ := sorry
def vertical_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def horizontal_asymptotes (f : ℝ → ℝ) : ℕ := sorry
def oblique_asymptotes (f : ℝ → ℝ) : ℕ := sorry

theorem asymptote_sum (f : ℝ → ℝ) : 
  holes f + 2 * vertical_asymptotes f + 3 * horizontal_asymptotes f + 4 * oblique_asymptotes f = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1071_107127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zero_l1071_107156

/-- Given real numbers a, b, and c satisfying the cube root condition,
    the polynomial P(x, y, z) = 27xyz - (z - x - y)³ evaluates to zero. -/
theorem polynomial_zero (a b c : ℝ) (h : (a ^ (1/3 : ℝ)) + (b ^ (1/3 : ℝ)) = (c ^ (1/3 : ℝ))) :
  27 * a * b * c - (c - a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_zero_l1071_107156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l1071_107154

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Triangle ABC with Fermat point P and additional point D -/
structure TriangleWithFermatPoint where
  A : Point
  B : Point
  C : Point
  P : Point
  D : Point

/-- The specific triangle from the problem -/
def specificTriangle : TriangleWithFermatPoint where
  A := ⟨0, 0⟩
  B := ⟨10, 0⟩
  C := ⟨7, 5⟩
  P := ⟨5, 3⟩
  D := ⟨2, 3⟩

theorem fermat_point_distance_sum :
  ∃ (m n k : ℤ),
    distance specificTriangle.P specificTriangle.A +
    distance specificTriangle.P specificTriangle.B +
    distance specificTriangle.P specificTriangle.C +
    distance specificTriangle.P specificTriangle.D =
    ↑m + ↑n * Real.sqrt 34 + ↑k * Real.sqrt 2 ∧
    m + n = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_distance_sum_l1071_107154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicle_profit_l1071_107106

/-- Represents the total expenditure in millions of yuan for n years of car use -/
noncomputable def totalExpenditure (n : ℕ) : ℝ := 0.25 * n^2 + 0.25 * n + 0.09

/-- Represents the total income in millions of yuan for n years of car use -/
noncomputable def totalIncome (n : ℕ) : ℝ := 5.25 * n

/-- Represents the profit in millions of yuan for n years of car use -/
noncomputable def profit (n : ℕ) : ℝ := totalIncome n - totalExpenditure n

/-- Represents the average annual profit in millions of yuan for n years of car use -/
noncomputable def avgAnnualProfit (n : ℕ) : ℝ := if n = 0 then 0 else profit n / n

theorem new_energy_vehicle_profit :
  (∀ k : ℕ, k < 3 → profit k ≤ 0) ∧ 
  profit 3 > 0 ∧
  (∀ n : ℕ, n ∈ Finset.range 9 → n ≠ 0 → avgAnnualProfit n ≤ avgAnnualProfit 6) :=
by
  sorry

#check new_energy_vehicle_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_energy_vehicle_profit_l1071_107106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_sides_perp_third_l1071_107115

-- Define a point in 3D space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a vector in 3D space
structure Vec where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle ABC
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a line
structure Line where
  p : Point
  v : Vec

-- Define perpendicularity between a line and a side of the triangle
def perpendicular (l : Line) (p q : Point) : Prop :=
  sorry -- The actual implementation would go here

-- Theorem statement
theorem line_perp_two_sides_perp_third (t : Triangle) (l : Line) :
  perpendicular l t.A t.B →
  perpendicular l t.B t.C →
  perpendicular l t.A t.C :=
by
  sorry -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_two_sides_perp_third_l1071_107115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1071_107100

/-- For a real number m, define the complex number z as (1+i)m^2 - m(5+3i) + 6 -/
def z (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - m * (5 + 3 * Complex.I) + 6

/-- Theorem: The complex number z(m) is a real number if and only if m is 0 or 3 -/
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 0 ∨ m = 3 := by sorry

/-- Theorem: The complex number z(m) is an imaginary number if and only if m is not 0 and not 3 -/
theorem z_is_imaginary (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 0 ∧ m ≠ 3 := by sorry

/-- Theorem: The complex number z(m) is a pure imaginary number if and only if m is 2 -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_is_pure_imaginary_l1071_107100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1071_107192

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 3 then 2*a*x + 4
  else if 2 < x ∧ x < 3 then (a*x + 2) / (x - 2)
  else 0  -- undefined for x ≤ 2

theorem function_decreasing_implies_a_range (a : ℝ) :
  (∀ x y, 2 < x ∧ x < y → f a y < f a x) →
  -1 < a ∧ a ≤ -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_range_l1071_107192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_subset_probability_l1071_107113

theorem odd_sum_subset_probability (n : ℕ) (h : n = 2017) :
  let S := Finset.range n
  let non_empty_subsets := Finset.powerset S \ {∅}
  let odd_sum_subsets := non_empty_subsets.filter (λ A => (A.sum id) % 2 = 1)
  (odd_sum_subsets.card : ℚ) / (non_empty_subsets.card : ℚ) = 2^2016 / (2^2017 - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_subset_probability_l1071_107113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1071_107155

theorem sin_graph_shift (x : ℝ) : 
  Real.sin (2 * (x + π / 8) - π / 4) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1071_107155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_minimized_l1071_107153

/-- The side length of the square -/
def squareSide : ℝ := 1

/-- The radius of the circle O -/
def radius (x : ℝ) : ℝ := x

/-- The area S of the region not passed through by circle O -/
noncomputable def S (x : ℝ) : ℝ :=
  if x < 1/4 then
    (20 - Real.pi) * (x - 4 / (20 - Real.pi))^2 + (4 - Real.pi) / (20 - Real.pi)
  else
    (4 - Real.pi) * x^2

/-- The theorem stating that S is minimized when x = 4 / (20 - π) -/
theorem S_minimized :
  ∃ (x : ℝ), x > 0 ∧ x ≤ 1/2 ∧
  (∀ (y : ℝ), y > 0 → y ≤ 1/2 → S x ≤ S y) ∧
  x = 4 / (20 - Real.pi) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_minimized_l1071_107153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1071_107195

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factorization (n : ℕ) : List ℕ := sorry

def reverse_digits (n : ℕ) : ℕ := sorry

def count_three_prime_products (sum : ℕ) : ℕ := sorry

theorem main_theorem (X : ℕ) : count_three_prime_products 118 = X := by
  let n := 7402
  let reversed := reverse_digits 2047
  let factors := prime_factorization n
  have h1 : reversed = n := by sorry
  have h2 : factors.length = 3 := by sorry
  have h3 : ∀ p ∈ factors, is_prime p := by sorry
  have h4 : factors.sum = 118 := by sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l1071_107195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1071_107172

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.tan α = Real.sqrt 3) 
  (h2 : π < α) 
  (h3 : α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -((Real.sqrt 3 - 1) / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1071_107172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_function_l1071_107183

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem periodic_sine_function 
  (A ω φ α : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : 0 < φ ∧ φ < Real.pi / 2) 
  (h4 : ∀ x, f A ω φ (x + Real.pi) = f A ω φ x) 
  (h5 : f A ω φ α = 1) :
  f A ω φ (α + 3 * Real.pi / 2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sine_function_l1071_107183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentrations_l1071_107103

noncomputable section

structure Solution where
  volume : ℝ
  substanceA : ℝ
  substanceB : ℝ
  substanceC : ℝ

def initial_solution : Solution :=
  { volume := 5
  , substanceA := 5 * 0.30
  , substanceB := 5 * 0.40
  , substanceC := 5 * 0.30 }

def add_water (s : Solution) (water : ℝ) : Solution :=
  { volume := s.volume + water
  , substanceA := s.substanceA
  , substanceB := s.substanceB
  , substanceC := s.substanceC }

def evaporate (s : Solution) (amount : ℝ) : Solution :=
  { volume := s.volume - amount
  , substanceA := s.substanceA
  , substanceB := s.substanceB
  , substanceC := s.substanceC }

def concentration (substance : ℝ) (total : ℝ) : ℝ :=
  substance / total * 100

theorem final_concentrations :
  let s1 := add_water initial_solution 2
  let s2 := evaporate s1 1
  concentration s2.substanceA s2.volume = 25 ∧
  concentration s2.substanceB s2.volume = 100/3 ∧
  concentration s2.substanceC s2.volume = 25 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentrations_l1071_107103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_range_l1071_107186

theorem max_value_and_inequality_range 
  (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 6) :
  (∃ (max : ℝ), max = 6 ∧ 
    (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 6 → x + 2*y + z ≤ max)) ∧
  (∃ (S : Set ℝ), S = Set.Iic (-7/3) ∧ 
    (∀ a : ℝ, (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 6 → 
      |a + 1| - 2*a ≥ x + 2*y + z) ↔ a ∈ S)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_range_l1071_107186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approximately_28_percent_l1071_107174

/-- Calculates the markup rate given selling price, profit rate, and expense rate -/
noncomputable def calculate_markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) : ℝ :=
  let cost := selling_price * (1 - profit_rate - expense_rate)
  let markup := (selling_price - cost) / cost
  markup * 100

theorem markup_rate_approximately_28_percent :
  let selling_price : ℝ := 10
  let profit_rate : ℝ := 0.12
  let expense_rate : ℝ := 0.10
  let markup_rate := calculate_markup_rate selling_price profit_rate expense_rate
  ∃ ε > 0, abs (markup_rate - 28) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_markup_rate_approximately_28_percent_l1071_107174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1071_107179

noncomputable def f (a x : ℝ) : ℝ := |2*x + a| + |x - 1/a|

theorem f_properties (a : ℝ) (ha : a < 0) :
  (f a 0 > 5/2 ↔ (a < -2 ∨ -1/2 < a)) ∧
  (∀ x : ℝ, f a x ≥ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1071_107179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximately_4484_l1071_107149

noncomputable section

-- Define the initial loan amount
def initial_loan : ℝ := 10000

-- Define the annual interest rate
def annual_rate : ℝ := 0.08

-- Define the loan term in years
def loan_term : ℕ := 10

-- Define the midterm payment year
def midterm_year : ℕ := 5

-- Define the midterm payment for Plan 2
def midterm_payment : ℝ := 2000

-- Function to calculate compound interest (generalized for different compounding frequencies)
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) (frequency : ℕ) : ℝ :=
  principal * (1 + rate / (frequency : ℝ)) ^ ((frequency : ℝ) * (time : ℝ))

-- Calculate the balance after 5 years for Plan 1
def plan1_midterm_balance : ℝ :=
  compound_interest initial_loan annual_rate midterm_year 2

-- Calculate the final payment for Plan 1
def plan1_final_payment : ℝ :=
  compound_interest (plan1_midterm_balance / 2) annual_rate midterm_year 2

-- Calculate the total payment for Plan 1
def plan1_total_payment : ℝ :=
  plan1_midterm_balance / 2 + plan1_final_payment

-- Calculate the remaining balance after 5 years for Plan 2
def plan2_midterm_balance : ℝ :=
  compound_interest (initial_loan - midterm_payment) annual_rate midterm_year 1

-- Calculate the total payment for Plan 2
def plan2_total_payment : ℝ :=
  midterm_payment + plan2_midterm_balance

-- State the theorem
theorem payment_difference_approximately_4484 :
  ∃ ε > 0, abs (plan1_total_payment - plan2_total_payment - 4484) < ε :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_payment_difference_approximately_4484_l1071_107149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt6_div_7_l1071_107140

def point_P : Fin 3 → ℝ := ![2, 0, 1]
def point_A : Fin 3 → ℝ := ![1, 3, 2]
def point_B : Fin 3 → ℝ := ![2, -1, 0]

def line_direction : Fin 3 → ℝ := λ i => point_B i - point_A i

-- Function to calculate the distance from a point to a line
noncomputable def distance_point_to_line (P A B : Fin 3 → ℝ) : ℝ :=
  let AP := λ i => P i - A i
  let AB := λ i => B i - A i
  let cross_product := ![
    AP 1 * AB 2 - AP 2 * AB 1,
    AP 2 * AB 0 - AP 0 * AB 2,
    AP 0 * AB 1 - AP 1 * AB 0
  ]
  let magnitude_cross_product := Real.sqrt (cross_product 0^2 + cross_product 1^2 + cross_product 2^2)
  let magnitude_AB := Real.sqrt (AB 0^2 + AB 1^2 + AB 2^2)
  magnitude_cross_product / magnitude_AB

theorem distance_is_sqrt6_div_7 : 
  distance_point_to_line point_P point_A point_B = Real.sqrt 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt6_div_7_l1071_107140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increase_after_time_reduction_l1071_107180

theorem efficiency_increase_after_time_reduction :
  ∀ (original_time : ℝ) (original_efficiency : ℝ),
    original_time > 0 → original_efficiency > 0 →
    let new_time := original_time * (1 - 0.2)
    let new_efficiency := original_efficiency * (original_time / new_time)
    new_efficiency = 1.25 * original_efficiency :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_increase_after_time_reduction_l1071_107180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_l1071_107101

noncomputable def f (x : ℝ) : ℝ := (9 * x - 5) / (x + 3)

theorem fixed_points_of_f :
  ∃! (s : Set (ℝ × ℝ)), s = {(1, 1), (5, 5)} ∧
  ∀ (x : ℝ), x ≠ -3 → (x, x) ∈ s ↔ f x = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_l1071_107101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_price_l1071_107197

/-- Given a series of transactions involving a cycle, electric scooter, skateboard, roller skates, and sports equipment,
    this theorem proves that the price of the sports equipment before import tax is approximately 1941.18 Rs. --/
theorem sports_equipment_price
  (cycle_price : ℝ)
  (cycle_loss_percent : ℝ)
  (scooter_gain_percent : ℝ)
  (skateboard_price : ℝ)
  (skateboard_loss_percent : ℝ)
  (roller_skates_price : ℝ)
  (roller_skates_gain_percent : ℝ)
  (import_tax_percent : ℝ)
  (earnings_coverage_percent : ℝ)
  (h1 : cycle_price = 1600)
  (h2 : cycle_loss_percent = 12)
  (h3 : scooter_gain_percent = 15)
  (h4 : skateboard_price = 1000)
  (h5 : skateboard_loss_percent = 10)
  (h6 : roller_skates_price = 600)
  (h7 : roller_skates_gain_percent = 25)
  (h8 : import_tax_percent = 5)
  (h9 : earnings_coverage_percent = 85) :
  ∃ (sports_equipment_price : ℝ), 
    (sports_equipment_price ≥ 1941.17 ∧ sports_equipment_price ≤ 1941.19) ∧
    (cycle_price * (1 - cycle_loss_percent / 100) * (1 + scooter_gain_percent / 100) +
     skateboard_price * (1 - skateboard_loss_percent / 100) +
     roller_skates_price * (1 + roller_skates_gain_percent / 100)) =
    (sports_equipment_price * earnings_coverage_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_equipment_price_l1071_107197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_open_interval_l1071_107105

-- Define the function f as noncomputable due to the use of real exponentiation
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (2 : ℝ)^x else x^2 + 1

-- State the theorem
theorem solution_set_eq_open_interval :
  {x : ℝ | f x < 2} = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_open_interval_l1071_107105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rates_of_change_order_l1071_107107

noncomputable def f (x : ℝ) : ℝ := 1 / x

noncomputable def avg_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

noncomputable def k₁ : ℝ := avg_rate_of_change f 1 2
noncomputable def k₂ : ℝ := avg_rate_of_change f 2 3
noncomputable def k₃ : ℝ := avg_rate_of_change f 3 4

theorem rates_of_change_order : k₁ < k₂ ∧ k₂ < k₃ := by
  sorry

#check rates_of_change_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rates_of_change_order_l1071_107107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1071_107190

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x-1) + x - 5

-- State the theorem
theorem zero_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  f 2 < 0 →
  f 3 > 0 →
  ∃! x₀ : ℝ, 2 < x₀ ∧ x₀ < 3 ∧ f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1071_107190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_l1071_107124

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem statement
theorem g_equals_g_inv_at_three :
  ∃! x : ℝ, g x = g_inv x := by
  -- We'll use 3 as our witness
  use 3
  constructor
  · -- Show that g(3) = g⁻¹(3)
    simp [g, g_inv]
    norm_num
  · -- Show uniqueness
    intro y h
    -- Expand the equality in the hypothesis
    have : 5 * y - 12 = (y + 12) / 5 := h
    -- Solve the equation
    linarith
  
-- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_three_l1071_107124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_day_four_distance_is_ten_l1071_107120

/-- Represents the race scenario between Jesse and Mia --/
structure RaceScenario where
  totalDistance : ℝ
  raceDuration : ℕ
  jesseFirstThreeDayAverage : ℝ
  miaFirstFourDayAverage : ℝ
  lastThreeDayAverageOfAverages : ℝ

/-- Calculates Jesse's day four distance --/
noncomputable def jesseDayFourDistance (scenario : RaceScenario) : ℝ :=
  scenario.totalDistance - (3 * scenario.jesseFirstThreeDayAverage) -
  (scenario.raceDuration - 4) * (2 * scenario.lastThreeDayAverageOfAverages -
  (scenario.totalDistance - 4 * scenario.miaFirstFourDayAverage) / 3)

/-- Theorem stating Jesse's day four distance is 10 miles --/
theorem jesse_day_four_distance_is_ten (scenario : RaceScenario)
    (h1 : scenario.totalDistance = 30)
    (h2 : scenario.raceDuration = 7)
    (h3 : scenario.jesseFirstThreeDayAverage = 2/3)
    (h4 : scenario.miaFirstFourDayAverage = 3)
    (h5 : scenario.lastThreeDayAverageOfAverages = 6) :
    jesseDayFourDistance scenario = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_day_four_distance_is_ten_l1071_107120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l1071_107177

/-- A line in polar coordinates passing through (2, π/4) and parallel to the polar axis -/
noncomputable def polar_line (r θ : ℝ) : Prop :=
  r * Real.sin θ = Real.sqrt 2

/-- The given point in polar coordinates -/
noncomputable def given_point : ℝ × ℝ := (2, Real.pi / 4)

theorem polar_line_equation :
  polar_line given_point.1 given_point.2 ∧
  ∀ (r θ : ℝ), polar_line r θ → 
    (∃ (k : ℝ), r * Real.cos θ = k) -- Parallel to polar axis condition
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l1071_107177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_number_l1071_107167

/-- A six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n < 1000000 }

/-- Function to move the first three digits of a six-digit number to the end -/
def moveFirstThreeToEnd (n : SixDigitNumber) : SixDigitNumber :=
  ⟨(n.val % 1000) * 1000 + (n.val / 1000), by {
    sorry  -- Proof that the result is a valid SixDigitNumber
  }⟩

/-- The property that a number increases by 6 times when its first three digits are moved to the end -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  (6 * n.val : ℕ) = (moveFirstThreeToEnd n).val

theorem unique_six_digit_number :
  ∃! (n : SixDigitNumber), satisfiesCondition n ∧ n.val = 142857 :=
by
  -- Existence
  use ⟨142857, by {
    constructor
    · simp
    · simp
  }⟩
  constructor
  · -- Proof that 142857 satisfies the condition
    sorry
  · -- Uniqueness
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_six_digit_number_l1071_107167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1071_107199

theorem geometric_sequence_sum (n : ℕ) :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 2
  let sum := a * (1 - r^n) / (1 - r)
  sum = 63 / 128 → n = 6 := by
  intro a r sum h
  -- The proof steps would go here
  sorry

#check geometric_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1071_107199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1071_107136

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 - 1/x + 2
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3*x + m
noncomputable def h (x : ℝ) : ℝ := x^2

theorem find_m (m : ℝ) : f 3 - g m 3 + h 3 = 5 → m = 122/3 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_l1071_107136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wizards_count_l1071_107126

def island_population : ℕ := 10
def reported_sum : ℕ := 36

def is_valid_report (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

theorem min_wizards_count (reports : Fin island_population → ℕ)
  (h_valid : ∀ i, is_valid_report (reports i))
  (h_sum : (Finset.sum (Finset.univ : Finset (Fin island_population)) reports) = reported_sum) :
  ∃ (wizards : Finset (Fin island_population)),
    wizards.card ≥ 5 ∧
    ∀ (actual : Fin island_population → ℕ),
      (∀ i, is_valid_report (actual i)) →
      (Finset.sum (Finset.range 10) id ≥ Finset.sum (Finset.univ : Finset (Fin island_population)) actual) →
      ∃ (diff : Fin island_population → ℤ),
        (∀ i, diff i = (actual i : ℤ) - (reports i : ℤ)) ∧
        (∀ i ∉ wizards, diff i = 0) ∧
        (Finset.sum (Finset.univ : Finset (Fin island_population)) diff).natAbs = 
          Finset.sum (Finset.range 10) id - reported_sum :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wizards_count_l1071_107126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_partition_l1071_107143

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point3D
  sideLength : ℝ

/-- Represents a convex polyhedron -/
structure ConvexPolyhedron where
  vertices : List Point3D

/-- Function to check if a sphere is contained within a cube -/
def sphereInCube (s : Sphere) (c : Cube) : Prop := sorry

/-- Function to check if two spheres are non-overlapping -/
def spheresNonOverlapping (s1 s2 : Sphere) : Prop := sorry

/-- Theorem stating that a cube containing non-overlapping spheres can be partitioned into convex polyhedra, each containing one sphere -/
theorem cube_sphere_partition (c : Cube) (spheres : List Sphere) :
  (∀ s, s ∈ spheres → sphereInCube s c) →
  (∀ s1 s2, s1 ∈ spheres → s2 ∈ spheres → s1 ≠ s2 → spheresNonOverlapping s1 s2) →
  ∃ (polyhedra : List ConvexPolyhedron),
    (∀ p, p ∈ polyhedra → ∃! s, s ∈ spheres ∧ sorry) ∧
    (∀ x : Point3D, sorry) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sphere_partition_l1071_107143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1071_107129

noncomputable def f (x : ℝ) := Real.exp (abs x) - 1 / (x^2 + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 ≤ x → x < y → f x < f y) ∧
  (∀ x y : ℝ, x < y → y ≤ 0 → f y < f x) ∧
  (∀ x : ℝ, f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1071_107129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_path_length_l1071_107114

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Calculates the diagonal length of a rectangle using the Pythagorean theorem -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ :=
  Real.sqrt (r.a^2 + r.b^2)

/-- Calculates the arc length for a 90-degree rotation -/
noncomputable def arcLength90 (radius : ℝ) : ℝ :=
  (1/4) * 2 * Real.pi * radius

/-- The total path length of point A in the described rotations -/
noncomputable def totalPathLength (r : Rectangle) : ℝ :=
  arcLength90 r.diagonal + arcLength90 r.b + arcLength90 r.a

theorem rectangle_rotation_path_length :
  ∀ r : Rectangle, r.a = 3 ∧ r.b = 4 → totalPathLength r = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_path_length_l1071_107114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_halves_triangle_sides_values_l1071_107131

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * t.c = 5 ∧ 
  Real.cos (t.A / 2) = 3 * Real.sqrt 10 / 10

-- Define the area function
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem for part 1
theorem triangle_area_is_three_halves (t : Triangle) 
  (h : triangle_conditions t) : 
  triangle_area t = 3 / 2 := by
  sorry

-- Additional condition for part 2
def additional_condition (t : Triangle) : Prop :=
  Real.sin t.B = 5 * Real.sin t.C

-- Theorem for part 2
theorem triangle_sides_values (t : Triangle) 
  (h1 : triangle_conditions t) 
  (h2 : additional_condition t) : 
  t.a = 3 * Real.sqrt 2 ∧ t.b = 5 ∧ t.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_halves_triangle_sides_values_l1071_107131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a2014_gt_f_a2015_l1071_107139

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_neg (x : ℝ) : x < 0 → f x > 1
axiom f_mult (x y : ℝ) : f x * f y = f (x + y)
axiom a_rec (n : ℕ) : f (a (n + 1)) = 1 / f (1 / (1 + a n))
axiom a1_ne_f0 : a 1 ≠ f 0

-- The theorem to be proved
theorem f_a2014_gt_f_a2015 : f (a 2014) > f (a 2015) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_a2014_gt_f_a2015_l1071_107139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_solution_l1071_107145

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f_n n (f x)

theorem f_100_solution :
  ∃! x : ℝ, f_n 100 x = 1 ∧ x = -1/99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_100_solution_l1071_107145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_a_production_rate_l1071_107169

/-- Represents the production rate of a machine in sprockets per hour -/
@[reducible] def ProductionRate := ℝ

/-- Represents the time taken by a machine to produce a given number of sprockets -/
@[reducible] def ProductionTime := ℝ

structure MachineData where
  rate : ProductionRate
  time : ProductionTime

theorem machine_a_production_rate 
  (total_sprockets : ℕ)
  (machine_p machine_q machine_a : MachineData)
  (h_total : total_sprockets = 660)
  (h_time_diff : machine_p.time = machine_q.time + 10)
  (h_rate_ratio : machine_q.rate = 1.1 * machine_a.rate)
  (h_p_production : machine_p.rate * machine_p.time = total_sprockets)
  (h_q_production : machine_q.rate * machine_q.time = total_sprockets) :
  machine_a.rate = 6 := by
  sorry

#check machine_a_production_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_a_production_rate_l1071_107169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangles_from_medians_and_side_l1071_107123

/-- Given the medians ma and mb, and side c of a triangle, 
    proves the existence of two possible triangles with their properties. -/
theorem two_triangles_from_medians_and_side 
  (ma mb c : ℝ) 
  (h_ma : ma = 10.769) 
  (h_mb : mb = 24.706) 
  (h_c : c = 28) : 
  ∃ (a b α β γ : ℝ) (a1 b1 α1 β1 γ1 : ℝ),
  -- First triangle
  (abs (a - 24.82) < 0.01 ∧ abs (b - 10.818) < 0.01 ∧ 
   abs (α - 61.925833) < 0.000001 ∧ abs (β - 22.619444) < 0.000001 ∧ abs (γ - 95.454722) < 0.000001) ∧ 
  -- Second triangle
  (abs (a1 - 39) < 0.01 ∧ abs (b1 - 17) < 0.01 ∧ 
   abs (α1 - 118.074167) < 0.000001 ∧ abs (β1 - 22.619444) < 0.000001 ∧ abs (γ1 - 39.306389) < 0.000001) ∧
  -- Relationships between sides and medians
  (abs (2 * ma^2 - (b^2 + c^2 - a^2 / 2)) < 0.01) ∧
  (abs (2 * mb^2 - (a^2 + c^2 - b^2 / 2)) < 0.01) ∧
  -- Angle sum property
  (abs (α + β + γ - 180) < 0.000001) ∧
  (abs (α1 + β1 + γ1 - 180) < 0.000001) ∧
  -- Sine rule for medians
  (abs (ma / Real.sin (α * π / 180) - c / 2) < 0.01) ∧
  (abs (mb / Real.sin (β * π / 180) - c / 2) < 0.01) ∧
  (abs (ma / Real.sin (γ1 * π / 180) - c / 2) < 0.01) ∧
  (abs (mb / Real.sin (α1 * π / 180) - c / 2) < 0.01) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangles_from_medians_and_side_l1071_107123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1071_107189

noncomputable def f (x : ℝ) := -1/3 * Real.sin (4 * x)

theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Icc (-Real.pi/8) (Real.pi/8), 
    ∀ y ∈ Set.Icc (-Real.pi/8) (Real.pi/8),
      x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l1071_107189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_count_l1071_107134

/-- Represents the five possible colors for flag strips -/
inductive Color
  | Red
  | White
  | Blue
  | Green
  | Yellow

/-- Represents a three-strip flag -/
structure ThreeStripFlag where
  top : Color
  middle : Color
  bottom : Color

/-- Checks if two colors are different -/
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

/-- Represents a valid flag configuration -/
def valid_flag (f : ThreeStripFlag) : Prop :=
  different_colors f.top f.middle ∧ different_colors f.middle f.bottom

/-- The number of valid flag configurations -/
def num_valid_flags : ℕ := sorry

theorem flag_count : num_valid_flags = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_count_l1071_107134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_abs_l1071_107116

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + (y - a)^2 = 2*(a - 2)^2

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- Define tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_equation x y a ∧ y = abs_func x ∧
  ∀ x' y' : ℝ, circle_equation x' y' a → y' ≥ abs_func x'

-- The main theorem
theorem tangent_circle_abs (a : ℝ) (h1 : a > 0) (h2 : a ≠ 2) :
  is_tangent a → a = 4/3 ∨ a = 4 := by
  sorry

#check tangent_circle_abs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_abs_l1071_107116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_tax_rate_is_eight_percent_l1071_107176

/-- Represents the shopping breakdown and tax rates for Jill's purchase --/
structure ShoppingTax where
  clothing_percent : ℚ
  food_percent : ℚ
  other_percent : ℚ
  clothing_tax_rate : ℚ
  food_tax_rate : ℚ
  total_tax_rate : ℚ

/-- Calculates the tax rate on other items given the shopping breakdown and known tax rates --/
def calculate_other_tax_rate (s : ShoppingTax) : ℚ :=
  ((s.total_tax_rate - s.clothing_tax_rate * s.clothing_percent) / s.other_percent)

/-- Theorem stating that the tax rate on other items is 8% given the problem conditions --/
theorem other_tax_rate_is_eight_percent :
  let s : ShoppingTax := {
    clothing_percent := 6/10,
    food_percent := 1/10,
    other_percent := 3/10,
    clothing_tax_rate := 4/100,
    food_tax_rate := 0,
    total_tax_rate := 48/1000
  }
  calculate_other_tax_rate s = 8/100 := by sorry

#eval calculate_other_tax_rate {
  clothing_percent := 6/10,
  food_percent := 1/10,
  other_percent := 3/10,
  clothing_tax_rate := 4/100,
  food_tax_rate := 0,
  total_tax_rate := 48/1000
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_tax_rate_is_eight_percent_l1071_107176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_increasing_sequence_l1071_107141

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x

-- Define the sequence a_n
def a (n : ℕ) : ℝ := n + 1

-- Define the sequence c_n
noncomputable def c (m : ℝ) (n : ℕ) : ℝ := f m (a n) * Real.log (f m (a n))

theorem arithmetic_and_increasing_sequence 
  (m : ℝ) 
  (h_m_pos : m > 0) 
  (h_m_neq_one : m ≠ 1) 
  (h_geometric : ∀ n : ℕ, f m (a n) = m^2 * m^(n-1)) :
  (∀ n : ℕ, a (n + 1) - a n = 1) ∧ 
  ((0 < m ∧ m < Real.exp (-1)) ∨ m > 1 ↔ ∀ n : ℕ+, c m n < c m (n + 1)) := by
  sorry

#check arithmetic_and_increasing_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_and_increasing_sequence_l1071_107141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1071_107111

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_value (a : ℝ) :
  (power_function a 3 = Real.sqrt 3 / 3) →
  (power_function a 9 = 1 / 3) :=
by
  -- Introduce the hypothesis
  intro h
  -- Unfold the definition of power_function
  unfold power_function at *
  -- Use the hypothesis to determine the value of a
  have ha : a = -1/2 := by
    -- This step would require more detailed proof, which we'll skip for now
    sorry
  -- Substitute the value of a and simplify
  rw [ha]
  -- The final equality would require more steps to prove
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l1071_107111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zero_points_implies_a_eq_neg_e_l1071_107119

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x
  else if x = 0 then 0
  else Real.exp (-x) - a * x

-- State the theorem
theorem three_zero_points_implies_a_eq_neg_e (a : ℝ) :
  (∃ x y z, x < y ∧ y < z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  a = -Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zero_points_implies_a_eq_neg_e_l1071_107119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kathleen_paint_time_l1071_107138

/-- Proves that Kathleen can paint a room alone in 3 hours given the conditions -/
theorem kathleen_paint_time (anthony_time : ℝ) (joint_time : ℝ) (kathleen_time : ℝ) : 
  anthony_time = 4 →
  joint_time = 3.428571428571429 →
  (1 / kathleen_time + 1 / anthony_time) * joint_time = 2 →
  kathleen_time = 3 := by
  intros h1 h2 h3
  sorry

#check kathleen_paint_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kathleen_paint_time_l1071_107138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_chord_length_two_l1071_107163

-- Define the line l: 2x - y + m = 0
def line (m : ℝ) (x y : ℝ) : Prop := 2 * x - y + m = 0

-- Define the circle C: x² + y² = 5
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem for part I
theorem no_common_points (m : ℝ) :
  (∀ x y : ℝ, ¬(line m x y ∧ myCircle x y)) ↔ (m > 5 ∨ m < -5) :=
sorry

-- Theorem for part II
theorem chord_length_two (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line m x₁ y₁ ∧ line m x₂ y₂ ∧ 
    myCircle x₁ y₁ ∧ myCircle x₂ y₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) ↔ 
  (m = 2 * Real.sqrt 5 ∨ m = -2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_chord_length_two_l1071_107163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_one_a_range_when_intersection_empty_l1071_107109

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

-- Part 1
theorem union_when_a_is_one :
  A 1 ∪ B = Set.Iic 3 ∪ Set.Ici 4 := by sorry

-- Part 2
theorem a_range_when_intersection_empty (a : ℝ) (h : a > 0) :
  A a ∩ B = ∅ → 0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_one_a_range_when_intersection_empty_l1071_107109
