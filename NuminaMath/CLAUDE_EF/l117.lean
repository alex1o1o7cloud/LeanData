import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l117_11764

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_d_range (a₁ d : ℝ) :
  (S a₁ d 2 * S a₁ d 4) / 2 + (S a₁ d 3)^2 / 9 + 2 = 0 →
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_d_range_l117_11764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_textbooks_probability_l117_11715

/-- Represents the number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the total number of ways to pack 15 books into 3 boxes of sizes 4, 5, and 6 -/
def totalArrangements : ℕ := binomial 15 4 * binomial 11 5 * binomial 6 6

/-- Calculates the number of ways to pack books with all 4 math books in the same box -/
def favorableArrangements : ℕ :=
  binomial 11 5 * binomial 6 6 +  -- 4 math books in the 4-book box
  binomial 11 1 * binomial 10 4 * binomial 6 6 +  -- 4 math books in the 5-book box
  binomial 11 2 * binomial 9 4 * binomial 5 5  -- 4 math books in the 6-book box

/-- Represents the probability as a fraction -/
noncomputable def probability : ℚ := favorableArrangements / totalArrangements

theorem math_textbooks_probability :
  ∃ (m n : ℕ), n ≠ 0 ∧ Nat.Coprime m n ∧ probability = m / n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_textbooks_probability_l117_11715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_stones_l117_11720

/-- Definition of "many" stones -/
def many_stones (piles : List ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.length = 50 ∧ 
    ∀ i, i < 50 → subset.get! i ≥ i + 1

/-- Initial configuration of stones -/
def initial_piles : List ℕ := List.replicate 100 100

/-- Removing stones from piles -/
noncomputable def remove_stones (piles : List ℕ) (n : ℕ) : List ℕ :=
  sorry  -- Implementation not required for the statement

/-- The main theorem -/
theorem largest_removable_stones :
  (∀ n : ℕ, n ≤ 5099 → ∀ removal : List ℕ, 
    removal.sum = n → many_stones (remove_stones initial_piles n)) ∧
  (∃ removal : List ℕ, removal.sum = 5100 ∧ 
    ¬many_stones (remove_stones initial_piles 5100)) :=
by
  sorry

#check largest_removable_stones

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_removable_stones_l117_11720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_six_digit_divisible_by_99_l117_11700

/-- Represents a six-digit number with distinct digits between 1 and 8 -/
structure SixDigitNumber where
  digits : Fin 6 → Fin 8
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

/-- Converts a SixDigitNumber to a natural number -/
def SixDigitNumber.toNat (n : SixDigitNumber) : ℕ :=
  (n.digits 0).val * 100000 + (n.digits 1).val * 10000 + (n.digits 2).val * 1000 +
  (n.digits 3).val * 100 + (n.digits 4).val * 10 + (n.digits 5).val

/-- Checks if a natural number is divisible by 99 -/
def isDivisibleBy99 (n : ℕ) : Prop := n % 99 = 0

/-- The main theorem -/
theorem max_six_digit_divisible_by_99 :
  ∀ n : SixDigitNumber, isDivisibleBy99 (n.toNat) →
  n.toNat ≤ 87653412 := by
  sorry

#check max_six_digit_divisible_by_99

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_six_digit_divisible_by_99_l117_11700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_between_52_and_53_l117_11746

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

theorem distance_AD_between_52_and_53 
  (A B C D : Point)
  (h1 : B.x = A.x + 15 ∧ B.y = A.y)  -- B is due east of A
  (h2 : C.x = B.x ∧ C.y = B.y + 15)  -- C is due north of B
  (h3 : distance A C = 15 * Real.sqrt 2)  -- Distance AC = 15√2
  (h4 : Real.arctan ((C.y - A.y) / (C.x - A.x)) = π / 4)  -- Angle BAC = 45°
  (h5 : D.x = C.x ∧ D.y = C.y + 35)  -- D is 35 meters due north of C
  : 52 < distance A D ∧ distance A D < 53 := by
  sorry

#check distance_AD_between_52_and_53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_between_52_and_53_l117_11746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_not_monotone_F_has_max_F_properties_l117_11730

/-- Represents the volume of a tetrahedron with one edge of length x and the remaining edges of length 1 -/
noncomputable def F : ℝ → ℝ := sorry

/-- The domain of F is (0, √3) -/
axiom F_domain : Set.Ioo 0 (Real.sqrt 3) = {x | F x ≠ 0}

/-- F is not monotonically increasing -/
theorem F_not_monotone : ¬Monotone F := by
  sorry

/-- F has a maximum value -/
theorem F_has_max : ∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.sqrt 3) ∧ ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.sqrt 3) → F y ≤ F x := by
  sorry

/-- Main theorem: F is not monotonically increasing and has a maximum value -/
theorem F_properties : ¬Monotone F ∧ (∃ (x : ℝ), x ∈ Set.Ioo 0 (Real.sqrt 3) ∧ ∀ (y : ℝ), y ∈ Set.Ioo 0 (Real.sqrt 3) → F y ≤ F x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_not_monotone_F_has_max_F_properties_l117_11730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_x_squared_y_cubed_l117_11797

theorem partial_derivatives_x_squared_y_cubed (x y : ℝ) :
  let z := fun (x y : ℝ) => x^2 * y^3
  (deriv (fun x => z x y) x = 2 * x * y^3) ∧
  (deriv (fun y => z x y) y = 3 * x^2 * y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_x_squared_y_cubed_l117_11797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_correct_l117_11783

/-- The intersection point of two lines, where one is y = -3x - 1 and the other is perpendicular to it passing through (4, 0) -/
noncomputable def intersection_point : ℝ × ℝ :=
  (1/10, -13/10)

/-- The slope of the first line -/
def m₁ : ℝ := -3

/-- The slope of the perpendicular line -/
noncomputable def m₂ : ℝ := -1 / m₁

/-- The point through which the perpendicular line passes -/
def point : ℝ × ℝ := (4, 0)

/-- The equation of the first line: y = m₁ * x + b₁ -/
def line₁ (x y : ℝ) : Prop :=
  y = m₁ * x - 1

/-- The equation of the perpendicular line: y - y₀ = m₂ * (x - x₀) -/
def line₂ (x y : ℝ) : Prop :=
  y - point.2 = m₂ * (x - point.1)

theorem intersection_point_is_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_is_correct_l117_11783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_age_theorem_l117_11752

/-- Walter's age at the end of 1994 -/
def walter_age_1994 : ℝ := sorry

/-- Walter's grandmother's age at the end of 1994 -/
def grandmother_age_1994 : ℝ := sorry

/-- The year 1994 -/
def year_1994 : ℕ := 1994

/-- The year 2000 -/
def year_2000 : ℕ := 2000

/-- Walter's age is 1/3 of his grandmother's age in 1994 -/
axiom age_relation : walter_age_1994 = grandmother_age_1994 / 3

/-- The sum of their birth years is 3858 -/
axiom birth_years_sum : (year_1994 - walter_age_1994) + (year_1994 - grandmother_age_1994) = 3858

/-- Walter's age at the end of 2000 -/
def walter_age_2000 : ℝ := walter_age_1994 + (year_2000 - year_1994)

theorem walter_age_theorem : walter_age_2000 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walter_age_theorem_l117_11752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_are_9_and_11_l117_11736

/-- Triangle with sides 13, 17, and 24 units -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 17
  hc : c = 24

/-- Dividing points on the longest side -/
noncomputable def dividing_points (t : Triangle) : ℝ × ℝ := (t.c / 3, 2 * t.c / 3)

/-- Distances from dividing points to the opposite vertex -/
noncomputable def distances_to_vertex (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The distances from the dividing points to the opposite vertex are 9 and 11 units -/
theorem distances_are_9_and_11 (t : Triangle) :
  distances_to_vertex t = (9, 11) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distances_are_9_and_11_l117_11736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_fuel_consumption_l117_11762

def distances : List Int := [2, -3, 4, -2, -8, 17, -2, -3, 12, 7, -5]
def fuel_rate : Float := 0.4

def net_distance (distances : List Int) : Int :=
  distances.sum

def total_fuel_consumption (distances : List Int) (fuel_rate : Float) : Float :=
  fuel_rate * (distances.map Int.natAbs).sum.toFloat

theorem distance_and_fuel_consumption :
  net_distance distances = 19 ∧
  total_fuel_consumption distances fuel_rate = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_fuel_consumption_l117_11762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_value_l117_11785

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 3 then x^2 - 2*x else 2*x + 1

-- Theorem statement
theorem f_composite_value : f (f 1) = 3 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(1)) = f(3)
  have h2 : f 3 = 3 := by
    simp [f]
    norm_num
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composite_value_l117_11785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_sum_l117_11771

theorem square_roots_sum (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (Real.sqrt x = a + 3 ∧ Real.sqrt x = 2*a + 3)) → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_sum_l117_11771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_positive_l117_11798

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the condition for the graph to be in the first and third quadrants
def in_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

-- Theorem statement
theorem inverse_proportion_k_positive
  (k : ℝ) (h1 : k ≠ 0) (h2 : in_first_and_third_quadrants (inverse_proportion k)) :
  k > 0 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_positive_l117_11798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_form_parabola_l117_11755

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle constructed on a segment -/
structure EquilateralTriangle where
  base_start : Point
  base_end : Point
  apex : Point

/-- The sequence of segment lengths -/
def segment_length (k : ℕ) : ℝ := 2 * k - 1

/-- The y-coordinate of the k-th segment's endpoint -/
def segment_endpoint (k : ℕ) : ℝ := k^2

/-- The apex of the k-th equilateral triangle -/
noncomputable def triangle_apex (k : ℕ) : Point :=
  { x := (segment_length k) * (Real.sqrt 3) / 2,
    y := segment_endpoint k - (segment_length k) / 2 }

/-- The equation of the parabola -/
def parabola_equation (p : Point) : Prop :=
  p.y = (1/3) * p.x^2 + 1/4

theorem equilateral_triangles_form_parabola :
  ∀ k : ℕ, parabola_equation (triangle_apex k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_form_parabola_l117_11755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_club_committee_probability_verify_probability_l117_11759

/-- The probability of choosing a 4-person committee with at least one boy and one girl
    from a group of 24 members (14 boys and 10 girls) is 11439/12650. -/
theorem grammar_club_committee_probability : 
  (12650 - (1001 + 210)) / 12650 = 11439 / 12650 := by
  sorry

/-- Helper function to calculate the probability -/
def calculate_probability (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_committees := Nat.choose total_members committee_size
  let all_boy_committees := Nat.choose boys committee_size
  let all_girl_committees := Nat.choose girls committee_size
  (total_committees - (all_boy_committees + all_girl_committees)) / total_committees

/-- Verify that the calculated probability matches the expected result -/
theorem verify_probability :
  calculate_probability 24 14 10 4 = 11439 / 12650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grammar_club_committee_probability_verify_probability_l117_11759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_climb_time_l117_11795

/-- A monkey climbing a tree -/
structure MonkeyClimb where
  treeHeight : ℕ
  hopDistance : ℕ
  slipDistance : ℕ

/-- Calculate the time taken for the monkey to reach the top of the tree -/
def timeToReachTop (climb : MonkeyClimb) : ℕ :=
  let netGainPerHour := climb.hopDistance - climb.slipDistance
  let timeToNearTop := (climb.treeHeight - climb.hopDistance) / netGainPerHour
  timeToNearTop + 1

/-- Theorem stating that the monkey will reach the top in 20 hours -/
theorem monkey_climb_time (climb : MonkeyClimb) 
  (h1 : climb.treeHeight = 21)
  (h2 : climb.hopDistance = 3)
  (h3 : climb.slipDistance = 2) :
  timeToReachTop climb = 20 := by
  sorry

#eval timeToReachTop { treeHeight := 21, hopDistance := 3, slipDistance := 2 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_climb_time_l117_11795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_when_f_equals_g_l117_11731

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 3) - 1
def g (x : ℝ) : ℝ := x^2 - 2*x - 1

-- State the theorem
theorem b_range_when_f_equals_g :
  ∀ a b : ℝ, f a = g b → b ∈ (Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_when_f_equals_g_l117_11731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_construction_l117_11738

-- Define a regular dodecagon
structure RegularDodecagon where
  vertices : Fin 12 → ℝ × ℝ
  is_regular : ∀ i j : Fin 12, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_equilateral : ∀ i j : Fin 3, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

-- Define the construction of equilateral triangles on the dodecagon
noncomputable def construct_triangles (d : RegularDodecagon) : Fin 12 → EquilateralTriangle :=
  sorry

-- Define the new vertices formed by the triangles
noncomputable def new_vertices (d : RegularDodecagon) : Fin 12 → ℝ × ℝ :=
  λ i ↦ (construct_triangles d i).vertices 2

-- Define the area between two sets of vertices
noncomputable def area_between (inner : Fin 12 → ℝ × ℝ) (outer : Fin 12 → ℝ × ℝ) : ℝ :=
  sorry

-- Define the area of a dodecagon
noncomputable def area (d : RegularDodecagon) : ℝ :=
  sorry

-- State the theorem
theorem dodecagon_construction (d : RegularDodecagon) :
  -- 1. The new vertices form a regular dodecagon
  (∃ d' : RegularDodecagon, d'.vertices = new_vertices d) ∧
  -- 2. The area between the two dodecagons equals the area of the inner dodecagon
  (area_between d.vertices (new_vertices d) = area d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecagon_construction_l117_11738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_correct_l117_11776

/-- A rectangle with a regular octagon inscribed in it. -/
structure OctagonInRectangle where
  /-- Length of the rectangle -/
  length : ℝ
  /-- Width of the rectangle -/
  width : ℝ
  /-- The perimeter of the rectangle is 200 cm -/
  perimeter_eq : length * 2 + width * 2 = 200
  /-- The ratio of length to width is 2:1 -/
  ratio_eq : length = 2 * width
  /-- The octagon vertices trisect the longer sides -/
  trisect_long : True
  /-- The octagon vertices bisect the shorter sides -/
  bisect_short : True

/-- The area of the inscribed regular octagon -/
noncomputable def octagon_area (r : OctagonInRectangle) : ℝ :=
  17500 / 9

/-- Theorem stating that the area of the inscribed regular octagon is 17500/9 cm² -/
theorem octagon_area_is_correct (r : OctagonInRectangle) : 
  octagon_area r = 17500 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_is_correct_l117_11776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_less_than_x0_l117_11727

/-- Given a function f and parameters a, x1, x2, and x0, prove that (x1 + x2) / 2 < x0 under specified conditions. -/
theorem mean_less_than_x0 (a x1 x2 x0 : ℝ) (f : ℝ → ℝ) : 
  a < -1/2 →
  1 < x1 →
  x1 < x2 →
  x0 ∈ Set.Ioo x1 x2 →
  (∀ x, f x = Real.log x - a * x^2 + (2 - a) * x) →
  (deriv f x0 = (f x2 - f x1) / (x2 - x1)) →
  (x1 + x2) / 2 < x0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_less_than_x0_l117_11727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l117_11790

/-- Calculates the final amount for an investment with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (times : ℕ) : ℝ :=
  principal * (1 + rate) ^ times

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_difference : 
  let principal := 50000
  let yearly_rate := 0.04
  let years := 2
  let quarterly_rate := yearly_rate / 4
  let quarters := years * 4
  
  let jose_investment := compound_interest principal yearly_rate years
  let patricia_investment := compound_interest principal quarterly_rate quarters
  
  round_to_nearest (patricia_investment - jose_investment) = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l117_11790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_correct_l117_11705

/-- Represents a strategy for guessing a number between 1 and n using yes/no questions. -/
def GuessingStrategy (n : ℕ) := ℕ → List (Fin n → Bool)

/-- The minimum number of questions needed to determine a number between 1 and n. -/
def min_questions (n : ℕ) : ℕ := n - 1

/-- Theorem stating that the minimum number of questions needed is n-1. -/
theorem min_questions_correct (n : ℕ) (h : n > 0) :
  ∀ (s : GuessingStrategy n),
  (∀ x : Fin n, ∃ k : ℕ, k ≤ min_questions n ∧
    (∀ y : Fin n, y ≠ x → ∃ i : Fin k, (s k).get ⟨i.val, sorry⟩ x ≠ (s k).get ⟨i.val, sorry⟩ y)) ∧
  ¬∃ m : ℕ, m < min_questions n ∧
    (∀ x : Fin n, ∃ k : ℕ, k ≤ m ∧
      (∀ y : Fin n, y ≠ x → ∃ i : Fin k, (s k).get ⟨i.val, sorry⟩ x ≠ (s k).get ⟨i.val, sorry⟩ y)) :=
by sorry

#check min_questions_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_correct_l117_11705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_plus_2_squared_cos_3x_l117_11760

theorem definite_integral_x_plus_2_squared_cos_3x :
  ∫ x in (-2)..0, (x + 2)^2 * Real.cos (3 * x) = (12 - 2 * Real.sin 6) / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_x_plus_2_squared_cos_3x_l117_11760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_theorem_l117_11799

/-- The radius of the inscribed sphere in a regular quadrangular pyramid -/
noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 2 - 1) / 2

/-- 
Theorem: For a regular quadrangular pyramid with base side length a and 
lateral face forming a 45° angle with the base, the radius of the inscribed 
sphere is a(√2 - 1)/2.
-/
theorem inscribed_sphere_radius_theorem (a : ℝ) (h : a > 0) :
  let base_side := a
  let lateral_angle := Real.pi / 4  -- 45° in radians
  inscribed_sphere_radius a = a * (Real.sqrt 2 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_theorem_l117_11799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l117_11701

-- Define the function as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 - 6 * m * x + m + 8)

-- State the theorem
theorem domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ (0 ≤ m ∧ m ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l117_11701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vershik_theorem_l117_11741

variable {n : ℕ}

/-- A random vector in ℝⁿ -/
structure RandomVector (n : ℕ) where
  components : Fin n → ℝ

/-- Covariance matrix of a random vector -/
def covarianceMatrix (X : RandomVector n) : Matrix (Fin n) (Fin n) ℝ := sorry

/-- Inner product of a vector and a random vector -/
def innerProduct (a : Fin n → ℝ) (X : RandomVector n) : ℝ := sorry

/-- Two random variables are uncorrelated -/
def isUncorrelated (X Y : ℝ) : Prop := sorry

/-- Two random variables are independent -/
def isIndependent (X Y : ℝ) : Prop := sorry

/-- A random vector is Gaussian -/
def isGaussian (X : RandomVector n) : Prop := sorry

/-- Vershik's theorem -/
theorem vershik_theorem (X : RandomVector n) 
  (h_nondeg : Matrix.det (covarianceMatrix X) ≠ 0)
  (h_indep : ∀ (a b : Fin n → ℝ), 
    isUncorrelated (innerProduct a X) (innerProduct b X) → 
    isIndependent (innerProduct a X) (innerProduct b X)) :
  isGaussian X := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vershik_theorem_l117_11741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_perpendicularity_l117_11770

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- The focal length of an ellipse -/
noncomputable def focal_length (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- The chord length through the focus perpendicular to the major axis -/
def chord_length (e : Ellipse) : ℝ := 2 * e.b

theorem ellipse_equation_and_perpendicularity 
  (e : Ellipse) 
  (h_ecc : eccentricity e = 1/2) 
  (h_chord : chord_length e = 3) :
  (∃ (a b : ℝ), a = 2 ∧ b^2 = 3 ∧ 
    ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ x^2/4 + y^2/3 = 1) ∧ 
  (∀ (A B : Point), 
    (A.x - 2)^2 + A.y^2 = (B.x - 2)^2 + B.y^2 ∧ 
    A.y^2 = 2*A.x ∧ B.y^2 = 2*B.x → 
    A.x * B.x + A.y * B.y = 2^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_perpendicularity_l117_11770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_symmetry_l117_11779

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.cos x + Real.sin x

theorem min_theta_for_symmetry :
  ∃ (θ : ℝ), θ > 0 ∧
  (∀ (x : ℝ), f (x - θ) = f (-(x - θ))) ∧
  (∀ (θ' : ℝ), θ' > 0 ∧ (∀ (x : ℝ), f (x - θ') = f (-(x - θ'))) → θ' ≥ θ) ∧
  θ = 5 * Real.pi / 6 := by
  sorry

#check min_theta_for_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_theta_for_symmetry_l117_11779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l117_11713

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l117_11713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_successor_l117_11789

/-- sum_of_digits n returns the sum of the decimal digits of n -/
def sum_of_digits (n : ℕ) : ℕ :=
sorry

/-- minimum_successor_digit_sum n returns the smallest possible 
    sum of digits of n+1 -/
def minimum_successor_digit_sum (n : ℕ) : ℕ :=
sorry

/-- Given a natural number n whose decimal digits sum to 2017, 
    the smallest possible sum of the digits of n+1 is 2. -/
theorem smallest_digit_sum_of_successor (n : ℕ) : 
  (sum_of_digits n = 2017) → (minimum_successor_digit_sum n = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_successor_l117_11789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_l117_11769

/-- Two lines intersect at a specific point -/
theorem lines_intersection :
  ∃ (s v : ℚ),
    (3 - 2*s, 4 + 3*s) = (1 - 3*v, 5 + 2*v) ∧
    (3 - 2*s, 4 + 3*s) = (25/13, 73/13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersection_l117_11769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_highest_points_area_l117_11763

noncomputable section

/-- Represents the area of the curve traced by highest points of projectile trajectories -/
def projectile_curve_area (u g : ℝ) : ℝ :=
  (Real.pi / 8) * (u^4 / g^2)

/-- The parametric equations for the projectile motion -/
def x (u t φ : ℝ) : ℝ := u * t * Real.cos φ
def y (u t φ g : ℝ) : ℝ := u * t * Real.sin φ - (1/2) * g * t^2

theorem projectile_highest_points_area 
  (u g : ℝ) 
  (hu : u > 0) 
  (hg : g > 0) :
  ∃ d : ℝ, d * (u^4 / g^2) = projectile_curve_area u g ∧
  ∀ φ : ℝ, 0 ≤ φ ∧ φ ≤ Real.pi/2 →
    ∃ t : ℝ, t > 0 ∧
      ∀ t' : ℝ, t' ≥ 0 → y u t' φ g ≤ y u t φ g :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_highest_points_area_l117_11763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l117_11794

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, if CF = 2, then DF = 2√5 -/
theorem ellipse_chord_theorem (e : Ellipse) (C D F : Point) :
  e.a = 6 ∧ e.b = 4 ∧  -- Ellipse parameters
  isOnEllipse e C ∧ isOnEllipse e D ∧  -- C and D are on the ellipse
  F.x = 2 * Real.sqrt 5 ∧ F.y = 0 ∧  -- F is a focus
  distance C F = 2 →  -- CF = 2
  distance D F = 2 * Real.sqrt 5 :=  -- DF = 2√5
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l117_11794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l117_11739

theorem division_problem (x y : ℕ) (h1 : x % y = 4) (h2 : (x : ℝ) / (y : ℝ) = 96.16) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l117_11739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_diameter_l117_11716

/-- The number of smaller circles arranged inside the larger circle -/
def n : ℕ := 8

/-- The radius of each smaller circle in units -/
def r : ℝ := 4

/-- The angle between the centers of adjacent smaller circles in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The radius of the larger circle -/
noncomputable def R : ℝ := r * (1 + 1 / Real.sin (θ / 2))

/-- The diameter of the larger circle -/
noncomputable def D : ℝ := 2 * R

theorem larger_circle_diameter :
  ‖D - 28.9442‖ < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_circle_diameter_l117_11716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_divides_segment_l117_11712

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- The center of mass of two weighted points divides the line segment between them proportionally to their weights. -/
theorem center_of_mass_divides_segment (A B O : V) (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a • (O - A) + b • (O - B) = 0 ↔ ∃ (t : ℝ), t > 0 ∧ t < 1 ∧ O = (1 - t) • A + t • B ∧ t / (1 - t) = a / b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_mass_divides_segment_l117_11712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l117_11718

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem symmetry_of_f :
  (∀ x : ℝ, f (11 * Real.pi / 12 - x) = f (11 * Real.pi / 12 + x)) ∧
  (∀ x : ℝ, f (2 * Real.pi / 3 + x) = -f (2 * Real.pi / 3 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l117_11718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_2023_divisors_l117_11781

def is_valid_number (n : ℕ) : Prop :=
  ∃ (m k : ℕ), n = m * (6 ^ k) ∧ ¬(6 ∣ m)

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (·∣n) (Finset.range (n + 1))).card

theorem least_number_with_2023_divisors :
  ∃ (n : ℕ),
    is_valid_number n ∧
    number_of_divisors n = 2023 ∧
    (∀ (m : ℕ), m < n → is_valid_number m → number_of_divisors m ≠ 2023) ∧
    (∃ (m k : ℕ), n = m * (6 ^ k) ∧ ¬(6 ∣ m) ∧ m + k = 60466182) :=
  sorry

#check least_number_with_2023_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_2023_divisors_l117_11781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_implies_negative_one_l117_11754

theorem sine_equality_implies_negative_one (α : ℝ) :
  Real.sin α * Real.sin (π / 3 - α) = 3 * Real.cos α * Real.sin (α + π / 6) →
  Real.sin (2 * α + π / 6) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equality_implies_negative_one_l117_11754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_trapezoid_traces_ellipse_l117_11728

/-- Definition of a point in 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a segment in 2D plane -/
structure Segment2D where
  start : Point2D
  finish : Point2D

/-- Definition of a quadrilateral in 2D plane -/
structure Quadrilateral2D where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Definition of an inscribed trapezoid -/
def is_inscribed_trapezoid (q : Quadrilateral2D) : Prop :=
  ∃ (circle : Set Point2D), 
    q.A ∈ circle ∧ q.B ∈ circle ∧ q.C ∈ circle ∧ q.D ∈ circle ∧
    (q.A.y - q.B.y) / (q.A.x - q.B.x) = (q.C.y - q.D.y) / (q.C.x - q.D.x)

/-- Definition of a point being on a segment -/
def point_on_segment (p : Point2D) (s : Segment2D) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    p.x = s.start.x + t * (s.finish.x - s.start.x) ∧
    p.y = s.start.y + t * (s.finish.y - s.start.y)

/-- Definition of an ellipse -/
def is_ellipse (path : Set Point2D) : Prop :=
  ∃ (f1 f2 : Point2D) (a : ℝ),
    ∀ p ∈ path, 
      Real.sqrt ((p.x - f1.x)^2 + (p.y - f1.y)^2) +
      Real.sqrt ((p.x - f2.x)^2 + (p.y - f2.y)^2) = 2 * a

/-- Theorem: A point on CD of an inscribed trapezoid ABCD traces an elliptical path -/
theorem point_on_trapezoid_traces_ellipse 
  (AB CD : Segment2D) 
  (P : Point2D) 
  (h_parallel : (AB.finish.y - AB.start.y) / (AB.finish.x - AB.start.x) = 
                (CD.finish.y - CD.start.y) / (CD.finish.x - CD.start.x))
  (h_inscribed : ∀ (A B : Point2D), 
    is_inscribed_trapezoid (Quadrilateral2D.mk A B CD.finish CD.start))
  (h_P_on_CD : point_on_segment P CD) :
  ∃ (path : Set Point2D), P ∈ path ∧ is_ellipse path :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_trapezoid_traces_ellipse_l117_11728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l117_11711

-- Define the radii of the circles
def r₁ : ℝ := 3
def r₂ : ℝ := 5

-- Define the triangle PQR
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

-- Define the property of being tangent to both circles
def isTangentToBothCircles (t : Triangle) : Prop := sorry

-- Define the property of having two congruent sides
def hasTwoCongruentSides (t : Triangle) : Prop := sorry

-- Define the area function for a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Main theorem
theorem triangle_area (t : Triangle) 
  (h₁ : isTangentToBothCircles t) 
  (h₂ : hasTwoCongruentSides t) : 
  Real.sqrt ((area t) * (area t)) = 60 * Real.sqrt 5.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l117_11711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l117_11706

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (Real.sin t.A) / t.a = (Real.sqrt 3 * Real.cos t.B) / t.b

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : Real :=
  1/2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem max_triangle_area (t : Triangle) (h : triangle_conditions t) :
  ∃ (max_area : Real), ∀ (s : Triangle), triangle_conditions s → triangle_area s ≤ max_area ∧
  max_area = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l117_11706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l117_11733

-- Define the necessary types and functions
def ConvexPentagon : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry
def Diagonal : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry
def AreaTriangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry
def AreaPentagon : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

theorem pentagon_area (A B C D E : ℝ × ℝ) : 
  ConvexPentagon A B C D E →
  (∀ (X Y Z : ℝ × ℝ), Diagonal A B C D E X Y → AreaTriangle X Y Z = 1) →
  AreaPentagon A B C D E = (5 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l117_11733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_unique_zero_l117_11725

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1+a) * x^2 + 4*a*x + 24*a

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(1+a)*x + 4*a

theorem monotonicity_and_unique_zero (a : ℝ) (h : a > 1) :
  (∀ x < 2, (f' a x > 0)) ∧
  (∀ x > 2*a, (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo 2 (2*a), (f' a x < 0)) ∧
  (∃! x, f a x = 0) → 1 < a ∧ a < 6 :=
by
  sorry

#check monotonicity_and_unique_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_unique_zero_l117_11725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_theorem_l117_11753

def triangle_OPQ (P : ℝ × ℝ) : Prop :=
  P.1 ≥ 0 ∧ P.2 ≥ 0 ∧  -- P is in the first quadrant
  (P.2 / P.1 = 1) ∧     -- ∠POQ = 45°
  (P.1 - 8) * P.1 + P.2 * 8 = 0  -- ∠PQO = 90°

noncomputable def rotate_point (p : ℝ × ℝ) (θ : ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos θ - p.2 * Real.sin θ, p.1 * Real.sin θ + p.2 * Real.cos θ)

theorem triangle_rotation_theorem (P : ℝ × ℝ) 
  (h : triangle_OPQ P) : 
  rotate_point P (2 * Real.pi / 3) = (-4 * Real.sqrt 2 - 4 * Real.sqrt 6, 4 * Real.sqrt 6 - 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_theorem_l117_11753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l117_11719

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -1 + Real.sqrt 3 * t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * Real.cos θ

-- Define point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B (existence assumed)
axiom exists_intersection_points : ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
  (∃ (t : ℝ), line_l t = A) ∧ 
  (∃ (t : ℝ), line_l t = B) ∧
  (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A) ∧
  (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B)

-- State the theorem
theorem sum_of_reciprocal_distances :
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
  (∃ (t : ℝ), line_l t = A) ∧ 
  (∃ (t : ℝ), line_l t = B) ∧
  (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A) ∧
  (∃ (θ : ℝ), (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B) →
  1 / dist point_P A + 1 / dist point_P B = (2 * Real.sqrt 3 + 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocal_distances_l117_11719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l117_11724

def S : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n ≤ 200) (Finset.range 201)

def is_units_digit_3 (n : ℕ) : Bool := n % 10 = 3

def probability_units_digit_3_and_even_sum : ℚ :=
  (Finset.filter (fun (pair : ℕ × ℕ) =>
    is_units_digit_3 (3^pair.1 + 7^pair.2) ∧ Even (pair.1 + pair.2))
    (Finset.product S S)).card /
  (Finset.filter (fun (pair : ℕ × ℕ) => Even (pair.1 + pair.2))
    (Finset.product S S)).card

theorem probability_theorem :
  probability_units_digit_3_and_even_sum = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l117_11724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l117_11734

def A : Set ℤ := {-3, -2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 ≤ 3}

theorem A_intersect_B : A ∩ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l117_11734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l117_11788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 5)^2 + 6 * Real.log x

theorem function_properties (a : ℝ) :
  (∀ x > 0, ∃ y, y = f a x) →
  (∃ b c : ℝ, ∀ x, (b * x + c = f a 1 + (deriv (f a) 1) * (x - 1)) ∧ 
               b * 0 + c = 6) →
  (a = 1/2) ∧ 
  (∀ x, 2 * x + 6 = f (1/2) 1 + (deriv (f (1/2)) 1) * (x - 1)) ∧
  (∃ x_max > 0, ∀ y > 0, f (1/2) x_max ≥ f (1/2) y) ∧
  (∃ x_min > 0, ∀ y > 0, f (1/2) x_min ≤ f (1/2) y) ∧
  (f (1/2) (Real.exp 1) = 9/2 + 6 * Real.log 2) ∧
  (f (1/2) 3 = 2 + 6 * Real.log 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l117_11788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l117_11767

/-- Proves the length of two trains given their speeds and passing time -/
theorem train_length_problem (speed1 speed2 : ℝ) (passing_time : ℝ) 
  (h1 : speed1 = 30) (h2 : speed2 = 30) (h3 : passing_time = 60) :
  let relative_speed := (speed1 + speed2) * (5 / 18)
  let total_length := relative_speed * passing_time
  let train_length := total_length / 2
  train_length = 500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l117_11767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l117_11756

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ - x₂ = π → f x₁ = f x₂) ∧
  (∀ x : ℝ, f (π/12 + x) = -f (π/12 - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l117_11756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_is_cos_l117_11722

open Real

-- Define the sequence of functions
noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

-- State the theorem
theorem f_2013_is_cos : f 2013 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_is_cos_l117_11722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetricPointsRange_l117_11792

/-- The function f(x) = ln(x) + x^3 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3

/-- The function g(x) = x^3 - ax, parameterized by a -/
def g (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x

/-- The set of a values for which f and g have symmetric points with respect to the origin -/
def symmetricPointsSet : Set ℝ := {a : ℝ | ∃ x y : ℝ, f x = g a y ∧ f (-y) = g a (-x)}

/-- The theorem stating the range of a values for which f and g have symmetric points -/
theorem symmetricPointsRange : symmetricPointsSet = Set.Iic (1 / Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetricPointsRange_l117_11792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_with_distances_l117_11765

-- Define a square in a plane
structure Square where
  vertices : Fin 4 → ℝ × ℝ
  is_square : ∀ (i j : Fin 4), i ≠ j → 
    (vertices i).1^2 + (vertices i).2^2 + 
    (vertices j).1^2 + (vertices j).2^2 = 
    ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The main theorem
theorem no_square_with_distances :
  ¬ ∃ (s : Square) (p : ℝ × ℝ), 
    ∃ (perm : Fin 4 → Fin 4), Function.Bijective perm ∧
    (distance p (s.vertices (perm 0)) = 1) ∧
    (distance p (s.vertices (perm 1)) = 1) ∧
    (distance p (s.vertices (perm 2)) = 2) ∧
    (distance p (s.vertices (perm 3)) = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_square_with_distances_l117_11765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_width_is_1210_l117_11793

/-- The width of a rectangular plot of farmland -/
noncomputable def plot_width (rent_per_acre : ℝ) (total_rent : ℝ) (length : ℝ) (sqft_per_acre : ℝ) : ℝ :=
  (total_rent / rent_per_acre * sqft_per_acre) / length

/-- Theorem: Given the specified conditions, the width of the plot is 1210 feet -/
theorem plot_width_is_1210 :
  plot_width 60 600 360 43560 = 1210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_width_is_1210_l117_11793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_third_term_l117_11773

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = 2) ∧ a 1 = 1

theorem sequence_third_term (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_third_term_l117_11773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_c_l117_11721

theorem value_range_of_c (c : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 
    max (|x + c/x|) (|x + c/x + 2|) ≥ 5) →
  c ≤ -18 ∨ c ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_c_l117_11721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_cubic_function_l117_11761

/-- Given a function f(x) = x^3 - kx that is not monotonic in the interval (-3, -1),
    prove that 3 < k < 27 -/
theorem non_monotonic_cubic_function (k : ℝ) :
  (∃ x y, -3 < x ∧ x < y ∧ y < -1 ∧
    ((x^3 - k*x > y^3 - k*y) ∨ (x^3 - k*x < y^3 - k*y))) →
  3 < k ∧ k < 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_cubic_function_l117_11761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_b_value_special_case_l117_11742

-- Define the ellipse
def is_on_ellipse (x y b : ℝ) : Prop :=
  x^2 / 100 + y^2 / b^2 = 1 ∧ 0 < b ∧ b < 10

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) (b : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = 100 - b^2 ∧
  F₁ = (-c, 0) ∧ F₂ = (c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) (b : ℝ) : Prop :=
  is_on_ellipse P.1 P.2 b

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem 1: Maximum value of |PF₁| * |PF₂|
theorem max_product_foci_distances (b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 b →
  are_foci F₁ F₂ b →
  point_on_ellipse P b →
  (∀ Q : ℝ × ℝ, point_on_ellipse Q b → distance Q F₁ * distance Q F₂ ≤ 100) ∧
  (∃ R : ℝ × ℝ, point_on_ellipse R b ∧ distance R F₁ * distance R F₂ = 100) := by sorry

-- Theorem 2: Value of b when angle F₁PF₂ = 60° and area of triangle F₁PF₂ = 64√3/3
theorem b_value_special_case (b : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 b →
  are_foci F₁ F₂ b →
  point_on_ellipse P b →
  Real.arccos ((distance P F₁)^2 + (distance P F₂)^2 - (distance F₁ F₂)^2) / (2 * distance P F₁ * distance P F₂) = π/3 →
  1/2 * distance P F₁ * distance P F₂ * Real.sin (π/3) = 64 * Real.sqrt 3 / 3 →
  b = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_foci_distances_b_value_special_case_l117_11742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l117_11710

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_equivalence (P : ℝ) (h : simple_interest P 5 8 = 840) :
  simple_interest P 8 5 = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_equivalence_l117_11710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_127_not_present_l117_11784

def f (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_step (acc : ℝ) (x : ℝ) (coeff : ℝ) : ℝ := acc * x + coeff

def horner_values (x : ℝ) : List ℝ :=
  [5, 4, 3, 2, 1, 1].foldl (λ acc coeff => 
    match acc with
    | [] => [coeff]
    | h::_ => acc ++ [horner_step h x coeff]
  ) []

theorem horner_rule_127_not_present : 127 ∉ horner_values 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_rule_127_not_present_l117_11784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_increasing_f_l117_11743

/-- The function f(x) = sin(x) + √3 * cos(x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

/-- The theorem stating the maximum value of m -/
theorem max_m_for_increasing_f :
  (∃ (m : ℝ), m > 0 ∧ 
    (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) ∧
    (∀ m' : ℝ, m' > m → ∃ x y : ℝ, -m' ≤ x ∧ x < y ∧ y ≤ m' ∧ f x ≥ f y)) →
  (∃ (m : ℝ), m = π / 6 ∧ m > 0 ∧ 
    (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) ∧
    (∀ m' : ℝ, m' > m → ∃ x y : ℝ, -m' ≤ x ∧ x < y ∧ y ≤ m' ∧ f x ≥ f y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_increasing_f_l117_11743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_salary_l117_11757

/-- Calculates the new salary after a percentage increase -/
noncomputable def new_salary (original : ℝ) (percentage_increase : ℝ) : ℝ :=
  original * (1 + percentage_increase / 100)

/-- Proves that John's new salary is approximately $70 after a 7.69% raise -/
theorem johns_new_salary :
  let original_salary := (65 : ℝ)
  let percentage_increase := (7.69 : ℝ)
  abs (new_salary original_salary percentage_increase - 70) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_salary_l117_11757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_or_diff_constant_l117_11723

/-- Two polynomials with real coefficients that take integer values at the same points -/
structure IntegerValuedPolynomials where
  F : ℝ → ℝ
  G : ℝ → ℝ
  is_polynomial_F : Polynomial ℝ
  is_polynomial_G : Polynomial ℝ
  F_eval : F = λ x => (is_polynomial_F).eval x
  G_eval : G = λ x => (is_polynomial_G).eval x
  integer_valued : ∀ x : ℝ, (∃ n : ℤ, F x = n) ↔ (∃ m : ℤ, G x = m)

/-- The main theorem: either the sum or the difference of F and G is constant -/
theorem sum_or_diff_constant (p : IntegerValuedPolynomials) :
  (∃ c : ℝ, ∀ x : ℝ, p.F x + p.G x = c) ∨
  (∃ c : ℝ, ∀ x : ℝ, p.F x - p.G x = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_or_diff_constant_l117_11723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l117_11745

theorem complex_inequality (a b c : ℂ) (m n : ℝ) 
  (h1 : Complex.abs (a + b) = m)
  (h2 : Complex.abs (a - b) = n)
  (h3 : m * n ≠ 0) :
  max (Complex.abs (a * c + b)) (Complex.abs (a + b * c)) ≥ 
    (m * n) / Real.sqrt (m^2 + n^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_inequality_l117_11745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l117_11750

/-- The time (in seconds) it takes for a train to pass a man running in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) (wind_resistance : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let man_speed_ms := man_speed * 1000 / 3600
  let effective_train_speed := train_speed_ms * (1 - wind_resistance)
  let relative_speed := effective_train_speed + man_speed_ms
  train_length / relative_speed

/-- Theorem stating that the time for a 300m train with initial speed 90 km/hr,
    affected by 5% wind resistance, to pass a man running at 15 km/hr
    in the opposite direction is approximately 10.75 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 300 90 15 0.05 - 10.75| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l117_11750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_equality_l117_11791

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the centroid of a triangle -/
noncomputable def triangleCentroid (A B C : Point) : Point :=
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3 }

/-- Calculates the centroid of a quadrilateral -/
noncomputable def quadrilateralCentroid (q : Quadrilateral) : Point :=
  { x := (q.A.x + q.B.x + q.C.x + q.D.x) / 4,
    y := (q.A.y + q.B.y + q.C.y + q.D.y) / 4 }

/-- Theorem: The centroid of ABCD is the same as the centroid of KLMN -/
theorem centroid_equality (ABCD : Quadrilateral) : 
  let K := triangleCentroid ABCD.A ABCD.B ABCD.C
  let L := triangleCentroid ABCD.B ABCD.C ABCD.D
  let M := triangleCentroid ABCD.C ABCD.D ABCD.A
  let N := triangleCentroid ABCD.D ABCD.A ABCD.B
  let KLMN : Quadrilateral := { A := K, B := L, C := M, D := N }
  quadrilateralCentroid ABCD = quadrilateralCentroid KLMN := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_equality_l117_11791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l117_11717

theorem integers_between_cubes : 
  (⌊(10.2 : ℝ)^3⌋ : ℤ) - (⌈(10.1 : ℝ)^3⌉ : ℤ) + 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_between_cubes_l117_11717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l117_11777

/-- The function f(x) = sin(ωx) + cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

/-- f is monotonically increasing on (-ω, ω) -/
def monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- f is symmetric with respect to x = ω -/
def symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

/-- Main theorem -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0)
    (h_monotone : monotone_increasing (f ω) (-ω) ω)
    (h_symmetric : symmetric_about (f ω) ω) :
    ω = Real.sqrt Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l117_11777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l117_11735

/-- Two circles are tangent if they touch at exactly one point -/
def are_tangent (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  (‖O₁ - O₂‖ = r₁ + r₂) ∨ (‖O₁ - O₂‖ = |r₁ - r₂|)

theorem tangent_circles_radius (O₁ O₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  are_tangent O₁ O₂ r₁ r₂ →
  r₁ = 3 →
  ‖O₁ - O₂‖ = 7 →
  r₂ = 4 ∨ r₂ = 10 := by
  sorry

#check tangent_circles_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circles_radius_l117_11735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l117_11774

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

-- State the theorem
theorem monotonic_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1/3 ≤ a ∧ a ≤ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_iff_a_in_range_l117_11774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l117_11748

/-- A function f satisfying the given conditions -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

/-- The theorem statement -/
theorem function_property (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|) :
  ∀ x : ℝ, f a b (-π/6 - x) + f a b x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l117_11748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_decrease_is_28_percent_l117_11744

/-- Represents a towel with its folding and bleaching properties -/
structure Towel where
  length : ℝ
  width : ℝ
  max_folds : ℕ
  length_folds : ℕ
  width_folds : ℕ
  length_loss_percent : ℝ
  width_loss_percent : ℝ

/-- Calculates the area decrease percentage after bleaching -/
noncomputable def area_decrease_percentage (t : Towel) : ℝ :=
  let new_length := t.length * (1 - t.length_loss_percent)
  let new_width := t.width * (1 - t.width_loss_percent)
  let original_area := t.length * t.width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area * 100

/-- Theorem stating that the maximum area decrease percentage is 28% -/
theorem max_area_decrease_is_28_percent (t : Towel) 
  (h1 : t.length > 0)
  (h2 : t.width > 0)
  (h3 : t.length_folds + t.width_folds ≤ t.max_folds)
  (h4 : t.length_loss_percent = 0.2)
  (h5 : t.width_loss_percent = 0.1) :
  area_decrease_percentage t = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_decrease_is_28_percent_l117_11744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_shannen_olaf_l117_11708

theorem age_difference_shannen_olaf :
  ∀ (beckett olaf shannen jack : ℕ),
    beckett = 12 →
    beckett = olaf - 3 →
    ∃ x, shannen = olaf - x →
    jack = 2 * shannen + 5 →
    beckett + olaf + shannen + jack = 71 →
    olaf - shannen = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_shannen_olaf_l117_11708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_left_focus_l117_11709

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem shortest_distance_to_left_focus :
  ∀ x y : ℝ, is_on_hyperbola x y → distance x y (-5) 0 ≥ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_left_focus_l117_11709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_value_l117_11726

theorem smallest_k_value (m n : ℤ) (h1 : 10 * n - 9 * m = 7) (h2 : m ≤ 2018) :
  let k := 20 - 18 * m / n
  k ≥ 2/9 ∧ ∃ (m' n' : ℤ), 10 * n' - 9 * m' = 7 ∧ m' ≤ 2018 ∧ 20 - 18 * m' / n' = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_value_l117_11726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l117_11749

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => (2 * a n) / (2 + a n)

theorem a_formula : ∀ n : ℕ, a n = 2 / (n + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l117_11749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_operation_characterization_l117_11786

-- Define the binary operation type
def BinaryOp : Type := ℝ → ℝ → ℝ

-- State the theorem
theorem binary_operation_characterization :
  ∃ (f g : BinaryOp),
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → f a (f b c) = (f a b) * c) ∧
    (∀ (a : ℝ), a ≥ 1 → f a a ≥ 1) ∧
    (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → g a (g b c) = (g a b) * c) ∧
    (∀ (a : ℝ), a ≥ 1 → g a a ≥ 1) ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → f a b = a * b) ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → g a b = a / b) ∧
    (∀ (h : BinaryOp),
      ((∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → h a (h b c) = (h a b) * c) ∧
       (∀ (a : ℝ), a ≥ 1 → h a a ≥ 1)) →
      ((∀ (a b : ℝ), a > 0 → b > 0 → h a b = a * b) ∨
       (∀ (a b : ℝ), a > 0 → b > 0 → h a b = a / b))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_operation_characterization_l117_11786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_problem_l117_11751

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem statement
theorem logarithm_problem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 4 = 2) :
  a = 2 ∧ Set.Ioo (-1 : ℝ) 0 = {x : ℝ | f a (x + 1) < 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_problem_l117_11751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_formula_l117_11740

/-- The area of a triangle given the lengths of its medians -/
noncomputable def triangleAreaFromMedians (s_a s_b s_c : ℝ) : ℝ :=
  (1/3) * Real.sqrt ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c))

/-- Theorem: The area of a triangle with medians of lengths s_a, s_b, and s_c
    is equal to (1/3) * sqrt((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)) -/
theorem triangle_area_from_medians_formula (s_a s_b s_c : ℝ) (h_pos : s_a > 0 ∧ s_b > 0 ∧ s_c > 0) :
  ∃ (A : ℝ), A = triangleAreaFromMedians s_a s_b s_c ∧ A > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_medians_formula_l117_11740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_functions_l117_11782

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x) + 1
noncomputable def f3 (x : ℝ) : ℝ := 2 * Real.sin (-x + Real.pi / 4)
noncomputable def f4 (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- Define what it means for two functions to be siblings
def are_siblings (f g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = g (a * x + b) ∨ f x = g (-a * x + b)

-- State the theorem
theorem sibling_functions :
  are_siblings f3 f4 ∧
  ¬ are_siblings f1 f2 ∧
  ¬ are_siblings f1 f3 ∧
  ¬ are_siblings f1 f4 ∧
  ¬ are_siblings f2 f3 ∧
  ¬ are_siblings f2 f4 :=
by
  sorry

#check sibling_functions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sibling_functions_l117_11782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l117_11737

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2 else x - 2

-- Define the solution set S
def S : Set ℝ := {x | f x < x^2}

-- Theorem statement
theorem solution_set_equality : S = Set.union (Set.Ioi 2) (Set.Iic 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_equality_l117_11737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_method_is_none_l117_11775

/-- Enumeration of possible methods for determining relationships between categorical variables -/
inductive Method
  | ThreeDBarChart
  | TwoDBarChart
  | ContourBarChart
  | None

/-- The most accurate method for determining relationships between categorical variables -/
def mostAccurateMethod : Method := Method.None

/-- Theorem stating that the most accurate method is None of the given options -/
theorem most_accurate_method_is_none : 
  mostAccurateMethod = Method.None := by
  -- The actual proof would require statistical evidence and argumentation
  -- which is beyond the scope of Lean's theorem proving capabilities
  sorry

#check most_accurate_method_is_none

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_method_is_none_l117_11775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l117_11796

def sequence_a : ℕ → ℤ
  | 0 => 2
  | n + 1 => 4 * sequence_a n - 3

theorem a_10_value : sequence_a 9 = 2^18 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l117_11796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_cube_root_equality_l117_11766

theorem floor_cube_root_equality (n : ℕ+) : 
  ⌊(7 * n + 2 : ℝ) ^ (1/3)⌋ = ⌊(7 * n + 2 : ℝ) ^ (1/3)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_cube_root_equality_l117_11766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_angle_measure_l117_11729

-- Define the angle measure type
def AngleMeasure := ℝ

-- Define parallel lines
def ParallelLines (l m : Set (ℝ × ℝ)) : Prop := sorry

-- Define the angle measure function
def MeasureAngle (p q r : ℝ × ℝ) : AngleMeasure := sorry

-- Theorem statement
theorem parallel_lines_angle_measure 
  (l m : Set (ℝ × ℝ)) 
  (A B C : ℝ × ℝ) : 
  ParallelLines l m → 
  MeasureAngle A B C = (135 : ℝ) → 
  MeasureAngle B A C = (150 : ℝ) → 
  MeasureAngle C A B = (75 : ℝ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_angle_measure_l117_11729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l117_11768

open Complex

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define Euler's formula
axiom euler_formula (x : ℝ) : exp (x * i) = cos x + i * sin x

-- Define π
noncomputable def π : ℝ := Real.pi

-- Define the complex number in question
noncomputable def z : ℂ := i / exp ((π / 4) * i)

-- Theorem statement
theorem z_in_first_quadrant : re z > 0 ∧ im z > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l117_11768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l117_11758

theorem cube_root_sum_equals_one :
  (7 + 2 * Real.sqrt 21) ^ (1/3) + (7 - 2 * Real.sqrt 21) ^ (1/3) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l117_11758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_four_sum_divisible_by_20_l117_11702

/-- Given a set of integers, returns true if there exist four distinct elements a, b, c, d
    such that 20 divides (a + b - c - d) -/
def has_four_sum_divisible_by_20 (S : Finset Int) : Prop :=
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (20 ∣ (a + b - c - d))

/-- The main theorem stating that 9 is the smallest positive integer satisfying the condition -/
theorem smallest_n_for_four_sum_divisible_by_20 :
  (∀ S : Finset Int, S.card ≥ 9 → has_four_sum_divisible_by_20 S) ∧
  (∃ S : Finset Int, S.card = 8 ∧ ¬has_four_sum_divisible_by_20 S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_four_sum_divisible_by_20_l117_11702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_zero_necessary_not_sufficient_l117_11732

-- We don't need to redefine Complex as it's already defined in Mathlib
-- def Complex := ℂ

-- We don't need to redefine I as it's already defined in Mathlib
-- def I : Complex := Complex.I

-- Define a function to check if a complex number is purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem ab_zero_necessary_not_sufficient :
  (∀ a b : ℝ, isPurelyImaginary (Complex.mk a (-b)) → a * b = 0) ∧
  (∃ a b : ℝ, a * b = 0 ∧ ¬isPurelyImaginary (Complex.mk a (-b))) := by
  sorry

-- You can add more theorems or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_zero_necessary_not_sufficient_l117_11732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l117_11787

/-- The distance from the center of the circle ρ = 4cos(θ) to the line ρsin(θ + π/4) = 2√2 is √2 -/
theorem distance_circle_center_to_line :
  let circle : ℝ → ℝ → Prop := λ ρ θ ↦ ρ = 4 * Real.cos θ
  let line : ℝ → ℝ → Prop := λ ρ θ ↦ ρ * Real.sin (θ + π/4) = 2 * Real.sqrt 2
  let circle_center : ℝ × ℝ := (2, 0)
  let distance := Real.sqrt 2
  ∀ ρ θ, circle ρ θ → line ρ θ →
    (let d := abs (circle_center.1 + circle_center.2 - 4) / Real.sqrt 2
     d = distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l117_11787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_1986_a_6891_l117_11780

def sequenceA : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * sequenceA (n + 1) + sequenceA n

theorem gcd_a_1986_a_6891 : Int.gcd (sequenceA 1986) (sequenceA 6891) = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_a_1986_a_6891_l117_11780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_pattern_forms_tetrahedron_l117_11704

/-- A flat pattern of squares -/
structure FlatPattern where
  squares : Finset (ℝ × ℝ)
  num_squares : Nat
  is_diamond_shape : Bool

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Finset (ℝ × ℝ × ℝ)
  num_faces : Nat
  face_shape : String

/-- Function to fold a flat pattern -/
noncomputable def fold (pattern : FlatPattern) : RegularTetrahedron :=
  sorry

/-- Theorem stating that folding a diamond-shaped pattern of four squares forms a regular tetrahedron -/
theorem diamond_pattern_forms_tetrahedron (pattern : FlatPattern) :
  pattern.num_squares = 4 ∧ pattern.is_diamond_shape = true →
  ∃ (tetra : RegularTetrahedron), fold pattern = tetra ∧ tetra.num_faces = 4 ∧ tetra.face_shape = "triangle" :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_pattern_forms_tetrahedron_l117_11704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_symmetry_l117_11772

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin x * Real.sin (x + 3 * φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.cos (2 * x - φ)

theorem odd_function_and_symmetry 
  (h1 : ∀ x, f x φ = -f (-x) φ)  -- f is an odd function
  (h2 : 0 < φ ∧ φ < Real.pi / 2) -- φ ∈ (0, π/2)
  : φ = Real.pi / 6 ∧ 
    ∀ x, g x φ = g (-5 * Real.pi / 6 - x) φ  -- g is symmetric about x = -5π/12
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_symmetry_l117_11772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l117_11707

/-- Given an ellipse defined by x²/a² + y² = 1 passing through (1, √6/3), 
    its eccentricity is √6/3 -/
theorem ellipse_eccentricity (a : ℝ) :
  (1 / a^2 + (Real.sqrt 6 / 3)^2 = 1) →
  (Real.sqrt (a^2 - 1)) / a = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l117_11707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_square_area_l117_11714

/-- Given a regular hexagon with side length 4 cm, prove that a square made from the same total length of yarn as the hexagon has an area of 36 cm². -/
theorem hexagon_to_square_area (hexagon_side : ℝ) (hexagon_sides : ℕ) (square_sides : ℕ) : 
  hexagon_side = 4 → 
  hexagon_sides = 6 → 
  square_sides = 4 → 
  (hexagon_side * (hexagon_sides : ℝ) / (square_sides : ℝ)) ^ 2 = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_square_area_l117_11714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l117_11747

-- Define the set of numbers
def S : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a function to check if a number is odd
def isOdd (n : Nat) : Prop := n % 2 = 1

-- Define the events
def event1 (a b : Nat) : Prop := (isOdd a ∧ ¬isOdd b) ∨ (¬isOdd a ∧ isOdd b)
def event2 (a b : Nat) : Prop := isOdd a ∨ isOdd b
def event3 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ ¬isOdd a ∧ ¬isOdd b
def event4 (a b : Nat) : Prop := (isOdd a ∨ isOdd b) ∧ (¬isOdd a ∨ ¬isOdd b)

-- Theorem to prove
theorem mutually_exclusive_events :
  ∀ (a b : Nat), a ∈ S → b ∈ S → event3 a b → ¬(event1 a b ∨ event2 a b ∨ event4 a b) :=
by
  intros a b ha hb h3
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l117_11747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_participants_l117_11778

/-- Represents the number of participants and their ages at a tree planting event. -/
structure TreePlantingEvent where
  boys : ℕ
  girls : ℕ
  boysTotalAge : ℕ
  girlsTotalAge : ℕ

/-- The conditions of the tree planting event. -/
def eventConditions (event : TreePlantingEvent) : Prop :=
  (3 : ℚ) * event.girls = (5 : ℚ) * event.boys ∧
  event.girls = event.boys + 400 ∧
  event.boysTotalAge = 72 * event.boys ∧
  event.girlsTotalAge = 96 * (event.boys + 400) ∧
  (event.boysTotalAge : ℚ) / event.boys = 12 ∧
  (event.girlsTotalAge : ℚ) / event.girls = 16

/-- The theorem stating the total number of participants at the event. -/
theorem total_participants (event : TreePlantingEvent) 
  (h : eventConditions event) : event.boys + event.girls = 1600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_participants_l117_11778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l117_11703

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ := c / b

/-- A line is defined by the equation ax + by = c, where a, b, and c are real numbers and b ≠ 0. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  b_nonzero : b ≠ 0

theorem y_intercept_of_line (l : Line) :
  y_intercept l.a l.b l.c = -2 ↔ l.a = 2 ∧ l.b = -3 ∧ l.c = 6 := by
  sorry

#check y_intercept_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l117_11703
