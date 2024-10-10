import Mathlib

namespace domain_equals_range_l941_94199

-- Define the function f(x) = |x-2| - 2
def f (x : ℝ) : ℝ := |x - 2| - 2

-- Define the domain set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the range set N
def N : Set ℝ := f '' M

-- Theorem stating that M equals N
theorem domain_equals_range : M = N := by sorry

end domain_equals_range_l941_94199


namespace centroid_of_S_l941_94182

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a set
def centroid (T : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Theorem statement
theorem centroid_of_S :
  centroid S = (0, 13/5) := by sorry

end centroid_of_S_l941_94182


namespace quadratic_root_implies_m_value_l941_94115

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, (2^2 + m*2 + 2 = 0) → m = -3 :=
by
  sorry

end quadratic_root_implies_m_value_l941_94115


namespace candy_distribution_bijective_l941_94146

/-- The candy distribution function -/
def f (n : ℕ) (x : ℕ) : ℕ := (x * (x + 1) / 2) % n

/-- Proposition: The candy distribution function is bijective iff n is a power of 2 -/
theorem candy_distribution_bijective (n : ℕ) (h : n > 0) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end candy_distribution_bijective_l941_94146


namespace evaluate_expression_l941_94153

theorem evaluate_expression : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end evaluate_expression_l941_94153


namespace x_value_l941_94145

theorem x_value (x : ℝ) (h : (1 / 4 : ℝ) - (1 / 5 : ℝ) = 5 / x) : x = 100 := by
  sorry

end x_value_l941_94145


namespace smallest_integer_y_smallest_integer_y_is_six_l941_94123

theorem smallest_integer_y (y : ℤ) : (10 - 5*y < -15) ↔ (y ≥ 6) := by
  sorry

theorem smallest_integer_y_is_six : ∃ (y : ℤ), (10 - 5*y < -15) ∧ (∀ (z : ℤ), (10 - 5*z < -15) → z ≥ y) ∧ y = 6 := by
  sorry

end smallest_integer_y_smallest_integer_y_is_six_l941_94123


namespace dee_has_least_money_l941_94122

-- Define the people
inductive Person : Type
  | Ada : Person
  | Ben : Person
  | Cal : Person
  | Dee : Person
  | Eve : Person

-- Define a function to represent the amount of money each person has
variable (money : Person → ℕ)

-- Define the conditions
axiom different_amounts : ∀ (p q : Person), p ≠ q → money p ≠ money q
axiom cal_more_than_ada_ben : money Person.Cal > money Person.Ada ∧ money Person.Cal > money Person.Ben
axiom ada_eve_more_than_dee : money Person.Ada > money Person.Dee ∧ money Person.Eve > money Person.Dee
axiom ben_between_ada_dee : money Person.Ben > money Person.Dee ∧ money Person.Ben < money Person.Ada

-- Theorem to prove
theorem dee_has_least_money :
  ∀ (p : Person), p ≠ Person.Dee → money Person.Dee < money p :=
sorry

end dee_has_least_money_l941_94122


namespace a_perpendicular_to_a_minus_b_l941_94169

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem a_perpendicular_to_a_minus_b : a • (a - b) = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l941_94169


namespace fourth_guard_distance_l941_94108

theorem fourth_guard_distance (l w : ℝ) (h1 : l = 300) (h2 : w = 200) : 
  let P := 2 * (l + w)
  let three_guards_distance := 850
  let fourth_guard_distance := P - three_guards_distance
  fourth_guard_distance = 150 := by
sorry

end fourth_guard_distance_l941_94108


namespace parallel_line_plane_perpendicular_transitivity_l941_94186

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Axioms for the properties of lines and planes
axiom different_lines : ∀ (m n : Line), m ≠ n
axiom different_planes : ∀ (α β γ : Plane), α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Theorem 1
theorem parallel_line_plane (m n : Line) (α : Plane) :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) := by sorry

-- Theorem 2
theorem perpendicular_transitivity (m : Line) (α β γ : Plane) :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ := by sorry

end parallel_line_plane_perpendicular_transitivity_l941_94186


namespace negation_of_existence_l941_94118

theorem negation_of_existence (Z : Type) [Ring Z] : 
  (¬ ∃ x : Z, x^2 = 2*x) ↔ (∀ x : Z, x^2 ≠ 2*x) := by
  sorry

end negation_of_existence_l941_94118


namespace pythagorean_triplets_l941_94138

theorem pythagorean_triplets :
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 ↔ 
    ∃ (d p q : ℤ), a = 2*d*p*q ∧ b = d*(q^2 - p^2) ∧ c = d*(p^2 + q^2) :=
by sorry

end pythagorean_triplets_l941_94138


namespace rationalize_denominator_l941_94155

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧ 
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
  sorry

end rationalize_denominator_l941_94155


namespace total_heads_l941_94147

theorem total_heads (hens cows : ℕ) : 
  hens = 28 →
  2 * hens + 4 * cows = 144 →
  hens + cows = 50 := by
sorry

end total_heads_l941_94147


namespace cos_pi_third_minus_alpha_l941_94133

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 1 / 3) :
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end cos_pi_third_minus_alpha_l941_94133


namespace division_problem_l941_94126

theorem division_problem (n : ℕ) : 
  n / 22 = 12 ∧ n % 22 = 1 → n = 265 := by
  sorry

end division_problem_l941_94126


namespace may_profit_max_profit_l941_94188

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28
  else if 6 < x ∧ x ≤ 12 then 200 - 14 * x
  else 0

-- Theorem for May's profit
theorem may_profit : profit 5 = 88 := by sorry

-- Theorem for maximum profit
theorem max_profit :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 12 → profit x ≤ profit 7 ∧ profit 7 = 102 := by sorry

end may_profit_max_profit_l941_94188


namespace students_at_higher_fee_l941_94119

/-- Represents the inverse proportionality between number of students and tuition fee -/
def inverse_proportional (s f : ℝ) : Prop := ∃ k : ℝ, s * f = k

/-- Theorem: Given inverse proportionality and initial conditions, prove the number of students at $2500 -/
theorem students_at_higher_fee 
  (s₁ s₂ f₁ f₂ : ℝ) 
  (h_inverse : inverse_proportional s₁ f₁ ∧ inverse_proportional s₂ f₂)
  (h_initial : s₁ = 40 ∧ f₁ = 2000)
  (h_new_fee : f₂ = 2500) :
  s₂ = 32 := by
  sorry

end students_at_higher_fee_l941_94119


namespace sum_of_rectangle_areas_l941_94157

def rectangle_lengths : List ℕ := [1, 9, 25, 49, 81, 121]
def common_width : ℕ := 3

theorem sum_of_rectangle_areas :
  (rectangle_lengths.map (λ l => l * common_width)).sum = 858 := by
  sorry

end sum_of_rectangle_areas_l941_94157


namespace isosceles_triangle_perimeter_l941_94180

/-- An isosceles triangle with two side lengths of 6 and 8 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 8 ∧ (c = a ∨ c = b) →  -- Triangle is isosceles with sides 6 and 8
  a + b + c = 22 :=                  -- Perimeter is 22
by
  sorry


end isosceles_triangle_perimeter_l941_94180


namespace hyperbola_asymptote_slope_sine_l941_94109

/-- For a hyperbola with eccentricity √10 and transverse axis along the y-axis,
    the sine of the slope angle of its asymptote is √10/10. -/
theorem hyperbola_asymptote_slope_sine (e : ℝ) (h : e = Real.sqrt 10) :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ = Real.sqrt 10 / 10 := by
  sorry

end hyperbola_asymptote_slope_sine_l941_94109


namespace porch_length_calculation_l941_94177

/-- Given the dimensions of a house and porch, and the total area needing shingles,
    calculate the length of the porch. -/
theorem porch_length_calculation
  (house_length : ℝ)
  (house_width : ℝ)
  (porch_width : ℝ)
  (total_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_width = 4.5)
  (h4 : total_area = 232) :
  (total_area - house_length * house_width) / porch_width = 6 := by
  sorry

end porch_length_calculation_l941_94177


namespace average_speed_calculation_l941_94125

/-- Given a distance of 8640 meters and a time of 36 minutes, 
    the average speed is 4 meters per second. -/
theorem average_speed_calculation (distance : ℝ) (time_minutes : ℝ) :
  distance = 8640 ∧ time_minutes = 36 →
  (distance / (time_minutes * 60)) = 4 := by
  sorry

end average_speed_calculation_l941_94125


namespace st_length_l941_94187

/-- Triangle PQR with given side lengths and points S, T on its sides --/
structure TrianglePQR where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Point S on side PQ --/
  s : ℝ
  /-- Point T on side PR --/
  t : ℝ
  /-- PQ = 13 --/
  pq_eq : pq = 13
  /-- PR = 14 --/
  pr_eq : pr = 14
  /-- QR = 15 --/
  qr_eq : qr = 15
  /-- S is between P and Q --/
  s_between : 0 ≤ s ∧ s ≤ pq
  /-- T is between P and R --/
  t_between : 0 ≤ t ∧ t ≤ pr
  /-- ST is parallel to QR --/
  st_parallel_qr : (s / pq) = (t / pr)
  /-- ST contains the incenter of triangle PQR --/
  st_contains_incenter : ∃ (k : ℝ), 0 < k ∧ k < 1 ∧
    k * s / (1 - k) * (pq - s) = pr / (pr + qr) ∧
    k * t / (1 - k) * (pr - t) = pq / (pq + qr)

/-- The main theorem --/
theorem st_length (tri : TrianglePQR) : (tri.s * tri.pr + tri.t * tri.pq) / (tri.pq + tri.pr) = 135 / 14 := by
  sorry

end st_length_l941_94187


namespace south_movement_representation_l941_94136

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

/-- The problem statement -/
theorem south_movement_representation :
  let north20 : Movement := ⟨20, Direction.North⟩
  let south120 : Movement := ⟨120, Direction.South⟩
  north20.toSigned = 20 →
  south120.toSigned = -120 := by
  sorry

end south_movement_representation_l941_94136


namespace class_gender_composition_l941_94141

theorem class_gender_composition (num_boys num_girls : ℕ) :
  num_boys = 2 * num_girls →
  num_boys = num_girls + 7 →
  num_girls - 1 = 6 := by
sorry

end class_gender_composition_l941_94141


namespace min_value_theorem_l941_94170

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 9/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 2 ∧ 1/x₀ + 9/y₀ = 8 :=
sorry

end min_value_theorem_l941_94170


namespace roy_sports_time_l941_94181

/-- Calculates the total time spent on sports activities for a specific week --/
def total_sports_time (
  basketball_time : ℝ)
  (swimming_time : ℝ)
  (track_time : ℝ)
  (school_days : ℕ)
  (missed_days : ℕ)
  (weekend_soccer : ℝ)
  (weekend_basketball : ℝ)
  (canceled_swimming : ℕ) : ℝ :=
  let school_sports := (basketball_time + swimming_time + track_time) * (school_days - missed_days : ℝ) - 
                       swimming_time * canceled_swimming
  let weekend_sports := weekend_soccer + weekend_basketball
  school_sports + weekend_sports

/-- Theorem stating that Roy's total sports time for the specific week is 13.5 hours --/
theorem roy_sports_time : 
  total_sports_time 1 1.5 1 5 2 1.5 3 1 = 13.5 := by
  sorry

end roy_sports_time_l941_94181


namespace light_bulbs_theorem_l941_94152

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3/5

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1/3

/-- The total number of unbroken light bulbs in both the foyer and kitchen -/
def total_unbroken : ℕ := 34

theorem light_bulbs_theorem : 
  kitchen_bulbs * (1 - kitchen_broken_fraction) + 
  (foyer_broken / foyer_broken_fraction) * (1 - foyer_broken_fraction) = total_unbroken := by
sorry

end light_bulbs_theorem_l941_94152


namespace complex_fraction_simplification_l941_94165

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - (1/13) * Complex.I :=
by sorry

end complex_fraction_simplification_l941_94165


namespace mixture_capacity_l941_94102

/-- Represents a vessel containing a mixture of alcohol and water -/
structure Vessel where
  capacity : ℝ
  alcohol_percentage : ℝ

/-- Represents the final mixture -/
structure FinalMixture where
  total_volume : ℝ
  vessel_capacity : ℝ

def mixture_problem (vessel1 vessel2 : Vessel) (final : FinalMixture) : Prop :=
  vessel1.capacity = 2 ∧
  vessel1.alcohol_percentage = 0.35 ∧
  vessel2.capacity = 6 ∧
  vessel2.alcohol_percentage = 0.50 ∧
  final.total_volume = 8 ∧
  final.vessel_capacity = 10 ∧
  vessel1.capacity + vessel2.capacity = final.total_volume

theorem mixture_capacity (vessel1 vessel2 : Vessel) (final : FinalMixture) 
  (h : mixture_problem vessel1 vessel2 final) : 
  final.vessel_capacity = 10 := by
  sorry

#check mixture_capacity

end mixture_capacity_l941_94102


namespace right_triangle_min_perimeter_l941_94192

theorem right_triangle_min_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 1 →
  a^2 + b^2 = c^2 →
  a + b + c ≤ 5 :=
sorry

end right_triangle_min_perimeter_l941_94192


namespace inequality_equivalence_l941_94103

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x > Real.sqrt (9 * x^2)) ↔ ((y > 3 * x ∧ x ≥ 0) ∨ (y > 0 ∧ x < 0)) := by
  sorry

end inequality_equivalence_l941_94103


namespace sum_first_15_odd_integers_l941_94149

/-- The sum of the first n odd positive integers -/
def sumFirstNOddIntegers (n : ℕ) : ℕ :=
  n * n

theorem sum_first_15_odd_integers : sumFirstNOddIntegers 15 = 225 := by
  sorry

end sum_first_15_odd_integers_l941_94149


namespace circle_properties_l941_94179

/-- Given a circle C with equation x^2 + 8x - 2y = 1 - y^2, 
    prove that its center is (-4, 1), its radius is 3√2, 
    and the sum of its center coordinates and radius is -3 + 3√2 -/
theorem circle_properties : 
  ∃ (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + 8*x - 2*y = 1 - y^2) ∧
    center = (-4, 1) ∧
    radius = 3 * Real.sqrt 2 ∧
    center.1 + center.2 + radius = -3 + 3 * Real.sqrt 2 :=
by sorry

end circle_properties_l941_94179


namespace honor_guard_subsets_l941_94173

theorem honor_guard_subsets (n : ℕ) (h : n = 60) :
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by sorry

end honor_guard_subsets_l941_94173


namespace ordering_of_special_values_l941_94175

theorem ordering_of_special_values :
  let a := Real.exp (1/2)
  let b := Real.log (1/2)
  let c := Real.sin (1/2)
  a > c ∧ c > b := by sorry

end ordering_of_special_values_l941_94175


namespace problem_statement_l941_94154

theorem problem_statement (p q r : ℝ) 
  (h1 : p * r / (p + q) + q * p / (q + r) + r * q / (r + p) = -8)
  (h2 : q * r / (p + q) + r * p / (q + r) + p * q / (r + p) = 9) :
  q / (p + q) + r / (q + r) + p / (r + p) = 10 := by
  sorry

end problem_statement_l941_94154


namespace cyclic_iff_arithmetic_progression_l941_94129

/-- A quadrilateral with sides a, b, d, c (in that order) -/
structure Quadrilateral :=
  (a b d c : ℝ)

/-- The property of sides forming an arithmetic progression -/
def is_arithmetic_progression (q : Quadrilateral) : Prop :=
  ∃ k : ℝ, q.b = q.a + k ∧ q.d = q.a + 2*k ∧ q.c = q.a + 3*k

/-- The property of a quadrilateral being cyclic (inscribable in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop :=
  q.a + q.c = q.b + q.d

/-- Theorem: A quadrilateral is cyclic if and only if its sides form an arithmetic progression -/
theorem cyclic_iff_arithmetic_progression (q : Quadrilateral) :
  is_cyclic q ↔ is_arithmetic_progression q :=
sorry

end cyclic_iff_arithmetic_progression_l941_94129


namespace correct_expansion_of_expression_l941_94135

theorem correct_expansion_of_expression (a : ℝ) : 
  5 + a - 2 * (3 * a - 5) = 5 + a - 6 * a + 10 := by
  sorry

end correct_expansion_of_expression_l941_94135


namespace line_slope_intercept_sum_l941_94130

/-- Given two points C(3, 7) and D(8, 10), prove that the sum of the slope and y-intercept
    of the line passing through these points is 29/5 -/
theorem line_slope_intercept_sum (C D : ℝ × ℝ) : 
  C = (3, 7) → D = (8, 10) → 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 29/5 := by sorry

end line_slope_intercept_sum_l941_94130


namespace f_2018_equals_neg_8_l941_94139

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2018_equals_neg_8 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 3) = -1 / f x)
  (h3 : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 2018 = -8 := by
  sorry

end f_2018_equals_neg_8_l941_94139


namespace max_distance_to_line_l941_94124

noncomputable section

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 3 + y^2 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | x - y - 4 = 0}

-- Define point M
def M : ℝ × ℝ := (-2, 2)

-- Define the midpoint P of MN
def P (N : ℝ × ℝ) : ℝ × ℝ := ((N.1 + M.1) / 2, (N.2 + M.2) / 2)

-- Define the distance function from a point to a line
def dist_to_line (P : ℝ × ℝ) : ℝ :=
  |P.1 - P.2 - 4| / Real.sqrt 2

-- Theorem statement
theorem max_distance_to_line :
  ∃ (max_dist : ℝ), max_dist = 7 * Real.sqrt 2 / 2 ∧
  ∀ (N : ℝ × ℝ), N ∈ C → dist_to_line (P N) ≤ max_dist :=
sorry

end max_distance_to_line_l941_94124


namespace k_squared_minus_3k_minus_4_l941_94174

theorem k_squared_minus_3k_minus_4 (a b c d k : ℝ) :
  (2 * a / (b + c + d) = k) ∧
  (2 * b / (a + c + d) = k) ∧
  (2 * c / (a + b + d) = k) ∧
  (2 * d / (a + b + c) = k) →
  (k^2 - 3*k - 4 = -50/9) ∨ (k^2 - 3*k - 4 = 6) :=
by sorry

end k_squared_minus_3k_minus_4_l941_94174


namespace arithmetic_sequence_sum_l941_94151

/-- Given an arithmetic sequence where:
    - S_n is the sum of the first n terms
    - S_{2n} is the sum of the first 2n terms
    - S_{3n} is the sum of the first 3n terms
    This theorem proves that if S_n = 45 and S_{2n} = 60, then S_{3n} = 65. -/
theorem arithmetic_sequence_sum (n : ℕ) (S_n S_2n S_3n : ℝ) 
  (h1 : S_n = 45)
  (h2 : S_2n = 60) :
  S_3n = 65 := by
  sorry

end arithmetic_sequence_sum_l941_94151


namespace no_extreme_points_implies_m_range_l941_94111

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + m*x + 1

-- Define what it means for f to have no extreme points
def has_no_extreme_points (m : ℝ) : Prop :=
  ∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → f m y ≠ f m x ∨ (f m y < f m x ↔ y < x)

-- State the theorem
theorem no_extreme_points_implies_m_range (m : ℝ) :
  has_no_extreme_points m → m ≥ 1/3 :=
by sorry

end no_extreme_points_implies_m_range_l941_94111


namespace castle_tour_limit_l941_94116

structure Castle where
  side_length : ℝ
  num_halls : ℕ
  hall_side_length : ℝ
  has_doors : Bool

def max_visitable_halls (c : Castle) : ℕ :=
  sorry

theorem castle_tour_limit (c : Castle) 
  (h1 : c.side_length = 100)
  (h2 : c.num_halls = 100)
  (h3 : c.hall_side_length = 10)
  (h4 : c.has_doors = true) :
  max_visitable_halls c ≤ 91 :=
sorry

end castle_tour_limit_l941_94116


namespace gcd_2728_1575_l941_94195

theorem gcd_2728_1575 : Nat.gcd 2728 1575 = 1 := by
  sorry

end gcd_2728_1575_l941_94195


namespace half_product_uniqueness_l941_94114

theorem half_product_uniqueness (x : ℕ) :
  (∃ n : ℕ, x = n * (n + 1) / 2) →
  (∀ n k : ℕ, x = n * (n + 1) / 2 ∧ x = k * (k + 1) / 2 → n = k) :=
by sorry

end half_product_uniqueness_l941_94114


namespace problem_statement_l941_94105

theorem problem_statement (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (a * Real.sin (π/5) + b * Real.cos (π/5)) / (a * Real.cos (π/5) - b * Real.sin (π/5)) = Real.tan (8*π/15)) : 
  b / a = Real.sqrt 3 := by
sorry

end problem_statement_l941_94105


namespace y1_less_than_y2_l941_94191

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def onLine (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

theorem y1_less_than_y2 
  (f : LinearFunction)
  (p1 p2 : Point)
  (h1 : f.m = 8)
  (h2 : f.b = -1)
  (h3 : p1.x = 3)
  (h4 : p2.x = 4)
  (h5 : onLine p1 f)
  (h6 : onLine p2 f) :
  p1.y < p2.y := by
  sorry

end y1_less_than_y2_l941_94191


namespace evaluate_expression_l941_94176

theorem evaluate_expression (x : ℝ) (h : x = 2) : x^3 + x^2 + x + Real.exp x = 14 + Real.exp 2 := by
  sorry

end evaluate_expression_l941_94176


namespace total_books_l941_94134

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end total_books_l941_94134


namespace independent_of_b_implies_k_equals_two_l941_94117

/-- If the algebraic expression ab(5ka-3b)-(ka-b)(3ab-4a²) is independent of b, then k = 2 -/
theorem independent_of_b_implies_k_equals_two (a b k : ℝ) :
  (∀ b, ∃ C, a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2) = C) →
  k = 2 := by
  sorry

end independent_of_b_implies_k_equals_two_l941_94117


namespace inequality_holds_iff_p_geq_five_l941_94101

theorem inequality_holds_iff_p_geq_five (p : ℝ) :
  (∀ x : ℝ, x > 0 → Real.log (x + p) - (1/2 : ℝ) ≥ Real.log (Real.sqrt (2*x))) ↔ p ≥ 5 := by
  sorry

end inequality_holds_iff_p_geq_five_l941_94101


namespace abs_neg_six_equals_six_l941_94113

theorem abs_neg_six_equals_six : abs (-6 : ℤ) = 6 := by
  sorry

end abs_neg_six_equals_six_l941_94113


namespace imaginary_part_of_z_l941_94121

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (Complex.I + 1)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_l941_94121


namespace triangle_area_sum_l941_94132

-- Define points on a line
variable (A B C D E : ℝ)

-- Define lengths
variable (AB BC CD : ℝ)

-- Define areas
variable (S_MAC S_NBC S_MCD S_NCE : ℝ)

-- State the theorem
theorem triangle_area_sum :
  A < B ∧ B < C ∧ C < D ∧ D < E →  -- Points are on the same line in order
  AB = 4 →
  BC = 3 →
  CD = 2 →
  S_MAC + S_NBC = 51 →
  S_MCD + S_NCE = 32 →
  S_MCD + S_NBC = 18 := by
  sorry

end triangle_area_sum_l941_94132


namespace root_difference_squared_l941_94131

theorem root_difference_squared (a : ℝ) (r s : ℝ) : 
  r^2 - (a+1)*r + a = 0 → 
  s^2 - (a+1)*s + a = 0 → 
  (r-s)^2 = a^2 - 2*a + 1 := by
sorry

end root_difference_squared_l941_94131


namespace third_triangular_square_l941_94148

/-- A number that is both triangular and square --/
def TriangularSquare (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * (a + 1) / 2 ∧ n = b * b

/-- The first two triangular square numbers --/
def FirstTwoTriangularSquares : Prop :=
  TriangularSquare 1 ∧ TriangularSquare 36

/-- Checks if a number is the third triangular square number --/
def IsThirdTriangularSquare (n : ℕ) : Prop :=
  TriangularSquare n ∧
  FirstTwoTriangularSquares ∧
  ∀ m : ℕ, m < n → TriangularSquare m → (m = 1 ∨ m = 36)

/-- 1225 is the third triangular square number --/
theorem third_triangular_square :
  IsThirdTriangularSquare 1225 :=
sorry

end third_triangular_square_l941_94148


namespace water_content_in_fresh_grapes_fresh_grapes_water_percentage_l941_94178

theorem water_content_in_fresh_grapes 
  (dried_water_content : Real) 
  (fresh_weight : Real) 
  (dried_weight : Real) : Real :=
  let solid_content := dried_weight * (1 - dried_water_content)
  let water_content := fresh_weight - solid_content
  let water_percentage := (water_content / fresh_weight) * 100
  90

theorem fresh_grapes_water_percentage :
  let dried_water_content := 0.20
  let fresh_weight := 10
  let dried_weight := 1.25
  water_content_in_fresh_grapes dried_water_content fresh_weight dried_weight = 90 := by
  sorry

end water_content_in_fresh_grapes_fresh_grapes_water_percentage_l941_94178


namespace rhombus_count_in_triangle_l941_94198

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  positive : sideLength > 0

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  smallTriangles : ℕ

/-- The number of rhombuses in a large equilateral triangle -/
def countRhombuses (largeTriangle : EquilateralTriangle) (smallTriangleSideLength : ℝ) (rhombusSize : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem rhombus_count_in_triangle :
  let largeTriangle : EquilateralTriangle := ⟨10, by norm_num⟩
  let smallTriangleSideLength : ℝ := 1
  let rhombusSize : ℕ := 8
  countRhombuses largeTriangle smallTriangleSideLength rhombusSize = 84 := by
  sorry

end rhombus_count_in_triangle_l941_94198


namespace right_triangle_sets_l941_94128

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only {2, 2, 3} cannot form a right-angled triangle -/
theorem right_triangle_sets :
  is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 2 3 := by
  sorry


end right_triangle_sets_l941_94128


namespace total_blue_balloons_l941_94167

theorem total_blue_balloons (joan sally jessica : ℕ) 
  (h1 : joan = 9) 
  (h2 : sally = 5) 
  (h3 : jessica = 2) : 
  joan + sally + jessica = 16 := by
sorry

end total_blue_balloons_l941_94167


namespace cory_candy_purchase_l941_94144

/-- The amount of money Cory has initially -/
def cory_money : ℝ := 20

/-- The cost of one pack of candies -/
def candy_pack_cost : ℝ := 49

/-- The number of candy packs Cory wants to buy -/
def num_packs : ℕ := 2

/-- The additional amount of money Cory needs -/
def additional_money_needed : ℝ := num_packs * candy_pack_cost - cory_money

theorem cory_candy_purchase :
  additional_money_needed = 78 := by
  sorry

end cory_candy_purchase_l941_94144


namespace pure_imaginary_solutions_l941_94197

theorem pure_imaginary_solutions : 
  let f (x : ℂ) := x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 27*x^2 - 18*x - 8
  let y := Real.sqrt ((Real.sqrt 52 - 5) / 3)
  f (Complex.I * y) = 0 ∧ f (-Complex.I * y) = 0 := by
  sorry

end pure_imaginary_solutions_l941_94197


namespace same_gender_probability_l941_94140

/-- The probability of selecting 2 students of the same gender from a group of 5 students with 3 boys and 2 girls -/
theorem same_gender_probability (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 5)
  (h2 : boys = 3)
  (h3 : girls = 2)
  (h4 : total_students = boys + girls) :
  (Nat.choose boys 2 + Nat.choose girls 2) / Nat.choose total_students 2 = 2 / 5 := by
sorry

end same_gender_probability_l941_94140


namespace triangle_with_given_altitudes_is_obtuse_l941_94143

/-- A triangle with given altitudes --/
structure Triangle where
  alt1 : ℝ
  alt2 : ℝ
  alt3 : ℝ

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  ∃ θ : ℝ, θ > Real.pi / 2 ∧ θ < Real.pi ∧
    (Real.cos θ = -(5 : ℝ) / 16)

/-- Theorem: A triangle with altitudes 1/2, 1, and 2/5 is obtuse --/
theorem triangle_with_given_altitudes_is_obtuse :
  let t : Triangle := { alt1 := 1/2, alt2 := 1, alt3 := 2/5 }
  isObtuse t := by
  sorry


end triangle_with_given_altitudes_is_obtuse_l941_94143


namespace regular_soda_bottles_count_l941_94190

/-- The number of regular soda bottles in a grocery store -/
def regular_soda_bottles : ℕ := 30

/-- The total number of bottles in the store -/
def total_bottles : ℕ := 38

/-- The number of diet soda bottles in the store -/
def diet_soda_bottles : ℕ := 8

/-- Theorem stating that the number of regular soda bottles is correct -/
theorem regular_soda_bottles_count : 
  regular_soda_bottles = total_bottles - diet_soda_bottles :=
by sorry

end regular_soda_bottles_count_l941_94190


namespace juice_sales_theorem_l941_94159

/-- Represents the capacity of a can in liters -/
structure CanCapacity where
  large : ℝ
  medium : ℝ
  liter : ℝ

/-- Represents the daily sales data -/
structure DailySales where
  large : ℕ
  medium : ℕ
  liter : ℕ

/-- Calculates the total volume of juice sold in a day -/
def dailyVolume (c : CanCapacity) (s : DailySales) : ℝ :=
  c.large * s.large + c.medium * s.medium + c.liter * s.liter

theorem juice_sales_theorem (c : CanCapacity) 
  (s1 s2 s3 : DailySales) : 
  c.liter = 1 →
  s1 = ⟨1, 4, 0⟩ →
  s2 = ⟨2, 0, 6⟩ →
  s3 = ⟨1, 3, 3⟩ →
  dailyVolume c s1 = dailyVolume c s2 →
  dailyVolume c s2 = dailyVolume c s3 →
  (dailyVolume c s1 + dailyVolume c s2 + dailyVolume c s3) = 54 := by
  sorry

#check juice_sales_theorem

end juice_sales_theorem_l941_94159


namespace car_distribution_l941_94160

theorem car_distribution (total_cars_per_column : ℕ) 
                         (total_zhiguli : ℕ) 
                         (zhiguli_first : ℕ) 
                         (zhiguli_second : ℕ) :
  total_cars_per_column = 28 →
  total_zhiguli = 11 →
  zhiguli_first + zhiguli_second = total_zhiguli →
  (total_cars_per_column - zhiguli_first) = 2 * (total_cars_per_column - zhiguli_second) →
  (total_cars_per_column - zhiguli_first = 21 ∧ total_cars_per_column - zhiguli_second = 24) :=
by
  sorry

#check car_distribution

end car_distribution_l941_94160


namespace loan_amount_proof_l941_94183

/-- The interest rate at which A lends to B (as a decimal) -/
def rate_A_to_B : ℚ := 15 / 100

/-- The interest rate at which B lends to C (as a decimal) -/
def rate_B_to_C : ℚ := 185 / 1000

/-- The number of years for which the loan is given -/
def years : ℕ := 3

/-- The gain of B in the given period -/
def gain_B : ℕ := 294

/-- The amount lent by A to B -/
def amount_lent : ℕ := 2800

theorem loan_amount_proof :
  ∃ (P : ℕ), 
    (P : ℚ) * rate_B_to_C * years - (P : ℚ) * rate_A_to_B * years = gain_B ∧
    P = amount_lent :=
by sorry

end loan_amount_proof_l941_94183


namespace exists_divisible_by_five_l941_94150

def T : Set ℤ := {s | ∃ a : ℤ, s = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2}

theorem exists_divisible_by_five : ∃ s ∈ T, 5 ∣ s := by sorry

end exists_divisible_by_five_l941_94150


namespace quadratic_inequality_solution_set_l941_94112

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by
  sorry

end quadratic_inequality_solution_set_l941_94112


namespace counterexample_exists_l941_94158

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l941_94158


namespace b3f_hex_to_decimal_l941_94184

/-- Converts a single hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hexToDecimal c) 0

theorem b3f_hex_to_decimal :
  hexStringToDecimal "B3F" = 2879 := by
  sorry

end b3f_hex_to_decimal_l941_94184


namespace wall_thickness_l941_94120

/-- Calculates the thickness of a wall given its dimensions and the number of bricks used. -/
theorem wall_thickness
  (wall_length : Real)
  (wall_height : Real)
  (brick_length : Real)
  (brick_width : Real)
  (brick_height : Real)
  (num_bricks : Nat)
  (h_wall_length : wall_length = 9)
  (h_wall_height : wall_height = 6)
  (h_brick_length : brick_length = 0.25)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 7200) :
  ∃ (wall_thickness : Real),
    wall_thickness = 0.225 ∧
    wall_length * wall_height * wall_thickness =
      num_bricks * brick_length * brick_width * brick_height :=
by sorry

end wall_thickness_l941_94120


namespace muffin_cost_is_correct_l941_94156

/-- The cost of a muffin given the total cost and the cost of juice -/
def muffin_cost (total_cost juice_cost : ℚ) : ℚ :=
  (total_cost - juice_cost) / 3

theorem muffin_cost_is_correct (total_cost juice_cost : ℚ) 
  (h1 : total_cost = 370/100) 
  (h2 : juice_cost = 145/100) : 
  muffin_cost total_cost juice_cost = 75/100 := by
  sorry

#eval muffin_cost (370/100) (145/100)

end muffin_cost_is_correct_l941_94156


namespace two_digit_number_difference_l941_94166

/-- Given a two-digit number, prove that if the difference between the original number
    and the number with interchanged digits is 54, then the difference between its two digits is 6. -/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 54 → x - y = 6 := by
  sorry

end two_digit_number_difference_l941_94166


namespace cooking_time_per_potato_l941_94189

theorem cooking_time_per_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_rest : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : time_for_rest = 45) : 
  (time_for_rest : ℚ) / ((total_potatoes - cooked_potatoes) : ℚ) = 5 := by
  sorry

end cooking_time_per_potato_l941_94189


namespace circle_radius_with_tangent_parabola_l941_94162

theorem circle_radius_with_tangent_parabola :
  ∀ r : ℝ,
  (∃ x : ℝ, x^2 + r = x) →  -- Parabola y = x^2 + r is tangent to line y = x
  (∀ x : ℝ, x^2 + r ≥ x) →  -- Parabola lies above or on the line
  r = (1 : ℝ) / 4 :=
by sorry

end circle_radius_with_tangent_parabola_l941_94162


namespace number_theory_problem_l941_94161

theorem number_theory_problem :
  (∃ n : ℤ, 35 = 5 * n) ∧
  (∃ n : ℤ, 252 = 21 * n) ∧ ¬(∃ m : ℤ, 48 = 21 * m) ∧
  (∃ k : ℤ, 180 = 9 * k) := by
  sorry

end number_theory_problem_l941_94161


namespace supermarket_distribution_l941_94168

/-- Proves that given a total of 420 supermarkets divided between two countries,
    with one country having 56 more supermarkets than the other,
    the country with more supermarkets has 238 supermarkets. -/
theorem supermarket_distribution (total : ℕ) (difference : ℕ) (more : ℕ) (less : ℕ) :
  total = 420 →
  difference = 56 →
  more = less + difference →
  total = more + less →
  more = 238 := by
  sorry

end supermarket_distribution_l941_94168


namespace integral_roots_problem_l941_94100

theorem integral_roots_problem (x y z : ℕ) : 
  z^x = y^(2*x) ∧ 
  2^z = 2*(8^x) ∧ 
  x + y + z = 20 →
  x = 5 ∧ y = 4 ∧ z = 16 := by
sorry

end integral_roots_problem_l941_94100


namespace inequality_solution_set_l941_94185

theorem inequality_solution_set (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end inequality_solution_set_l941_94185


namespace function_positive_l941_94104

theorem function_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x + 1) * f x + x * (deriv^[2] f x) > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end function_positive_l941_94104


namespace cubic_root_product_l941_94196

theorem cubic_root_product (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (2+a)*(2+b)*(2+c) = 120 := by
  sorry

end cubic_root_product_l941_94196


namespace sqrt_product_equals_240_l941_94171

theorem sqrt_product_equals_240 : Real.sqrt 128 * Real.sqrt 50 * (27 ^ (1/3 : ℝ)) = 240 := by
  sorry

end sqrt_product_equals_240_l941_94171


namespace david_average_marks_l941_94127

def david_marks : List Nat := [96, 95, 82, 97, 95]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 93 := by sorry

end david_average_marks_l941_94127


namespace b_income_percentage_over_c_l941_94107

/-- Given the monthly incomes of A, B, and C, prove that B's monthly income is 12% more than C's. -/
theorem b_income_percentage_over_c (a_annual : ℕ) (c_monthly : ℕ) (h1 : a_annual = 571200) (h2 : c_monthly = 17000) :
  let a_monthly : ℕ := a_annual / 12
  let b_monthly : ℕ := (2 * a_monthly) / 5
  (b_monthly : ℚ) / c_monthly - 1 = 12 / 100 := by sorry

end b_income_percentage_over_c_l941_94107


namespace product_one_to_six_l941_94142

theorem product_one_to_six : (List.range 6).foldl (· * ·) 1 = 720 := by
  sorry

end product_one_to_six_l941_94142


namespace carbonic_acid_formation_l941_94194

-- Define the molecules and their quantities
structure Molecule where
  name : String
  moles : ℕ

-- Define the reaction
def reaction (reactant1 reactant2 product : Molecule) : Prop :=
  reactant1.name = "CO2" ∧ 
  reactant2.name = "H2O" ∧ 
  product.name = "H2CO3" ∧
  reactant1.moles = reactant2.moles ∧
  product.moles = min reactant1.moles reactant2.moles

-- Theorem statement
theorem carbonic_acid_formation 
  (co2 : Molecule) 
  (h2o : Molecule) 
  (h2co3 : Molecule) :
  co2.name = "CO2" →
  h2o.name = "H2O" →
  h2co3.name = "H2CO3" →
  co2.moles = 3 →
  h2o.moles = 3 →
  reaction co2 h2o h2co3 →
  h2co3.moles = 3 :=
by sorry

end carbonic_acid_formation_l941_94194


namespace arithmetic_sequence_theorem_l941_94164

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions of the problem
def sequence_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  a 1 = 1 ∧
  a 3 = Real.sqrt (a 1 * a 9)

-- State the theorem
theorem arithmetic_sequence_theorem (a : ℕ → ℝ) :
  sequence_conditions a → ∀ n : ℕ, a n = n := by sorry

end arithmetic_sequence_theorem_l941_94164


namespace diagonal_shorter_than_midpoint_distance_l941_94172

-- Define the quadrilateral ABCD
variables {A B C D : EuclideanSpace ℝ (Fin 2)}

-- Define the property that a circle through three points is tangent to a line segment
def is_tangent_circle (P Q R S T : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    dist center P = radius ∧ dist center Q = radius ∧ dist center R = radius ∧
    dist center S = radius ∧ dist S T = dist center S + dist center T

-- State the theorem
theorem diagonal_shorter_than_midpoint_distance
  (h1 : is_tangent_circle A B C C D)
  (h2 : is_tangent_circle A C D A B) :
  dist A C < (dist A D + dist B C) / 2 := by
  sorry

end diagonal_shorter_than_midpoint_distance_l941_94172


namespace square_sum_problem_l941_94106

theorem square_sum_problem (square triangle : ℝ) 
  (h1 : 2 * square + 2 * triangle = 16)
  (h2 : 2 * square + 3 * triangle = 19) :
  4 * square = 20 := by
  sorry

end square_sum_problem_l941_94106


namespace largest_of_three_consecutive_multiples_l941_94163

theorem largest_of_three_consecutive_multiples (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * n + 3 ∧ c = 3 * n + 6) →  -- Consecutive multiples of 3
  a + b + c = 117 →                                      -- Sum is 117
  c = 42 ∧ c ≥ a ∧ c ≥ b                                 -- c is the largest and equals 42
  := by sorry

end largest_of_three_consecutive_multiples_l941_94163


namespace experiment_comparison_l941_94110

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  red : Nat
  black : Nat

/-- Represents the result of an experiment -/
structure ExperimentResult where
  expectation : ℚ
  variance : ℚ

/-- Calculates the result of drawing with replacement -/
def drawWithReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

/-- Calculates the result of drawing without replacement -/
def drawWithoutReplacement (bag : BagContents) (draws : Nat) : ExperimentResult :=
  sorry

theorem experiment_comparison (bag : BagContents) (draws : Nat) :
  let withReplacement := drawWithReplacement bag draws
  let withoutReplacement := drawWithoutReplacement bag draws
  (bag.total = 5 ∧ bag.red = 2 ∧ bag.black = 3 ∧ draws = 2) →
  (withReplacement.expectation = withoutReplacement.expectation ∧
   withReplacement.variance > withoutReplacement.variance) :=
by sorry

end experiment_comparison_l941_94110


namespace isabellas_paintable_area_l941_94193

/-- Calculates the total paintable area for a set of identical rooms -/
def totalPaintableArea (
  numRooms : ℕ
  ) (length width height : ℝ
  ) (unpaintableAreaPerRoom : ℝ
  ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let paintableAreaPerRoom := wallArea - unpaintableAreaPerRoom
  numRooms * paintableAreaPerRoom

/-- Proves that the total paintable area for Isabella's bedrooms is 1592 square feet -/
theorem isabellas_paintable_area :
  totalPaintableArea 4 15 11 9 70 = 1592 := by
  sorry

end isabellas_paintable_area_l941_94193


namespace range_of_g_range_of_g_complete_l941_94137

def f (x : ℝ) : ℝ := 5 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ y ∈ Set.range g, -157 ≤ y ∧ y ≤ 1093 :=
sorry

theorem range_of_g_complete :
  ∀ y, -157 ≤ y ∧ y ≤ 1093 → ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = y :=
sorry

end range_of_g_range_of_g_complete_l941_94137
